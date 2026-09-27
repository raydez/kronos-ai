"""Compute cost probe（基线文档 §16 / §49，ADR-019）。

正式全量 benchmark 之前必须先回答一个问题：**在这台机器上跑得完吗**（§16）。
本模块是那次「探针」的引擎：它不做数据加载、不做预测，也不生产收益指标；它的输入是
一个显式声明的 *pilot matrix* 与一个**必须**设置的预算上限，输出一份
:class:`ComputeBudgetReport`（含 §16 的 throughput-memory curve）。

三条硬约束被写成类型不变量，而不是留给纪律：

1. **禁止全因子笛卡尔积**：设备目标是单一值（§16「先锁定目标设备」），扫描维度只有
   ``sample_count × batch``，且总组数上限 :data:`MAX_PROBE_CELLS` = 10。超限直接拒绝
   构造，不提供「悄悄截断到 10 组」的降级（ADR-010）。
2. **不允许无上限运行**：:class:`ProbeBudget` 的 wall-clock 与内存上限都没有默认值。
   每次迭代前查 wall-clock，每次迭代后查峰值内存，循环结束后再复核一次总耗时
   （单次 ``measure`` 调用无法被抢占：末轮超限只能靠事后复核发现），触发即
   **中止该组合**。
3. **超限不算「跑完了」**：该组合状态显式为 ``INFEASIBLE_ON_THIS_DEVICE`` 并附原因；
   报告保留已完成的迭代次数与内存峰值，但**不**给出 p50 / throughput——从被截断的
   运行里算出来的 p50 是最容易骗过评审的数字。

指标数学用**最近秩法**（nearest-rank），定义与理由见，:func:`percentile`；口径版本是
:data:`COMPUTE_METRICS_VERSION`（§49 要求指标数学版本化）。

依赖方向：本模块只依赖 ``domain`` 的哈希与 pydantic，不 import torch / pandas /
provider。cost probe 必须能在廉价环境（含 CI）里对 baseline backend 跑通（§48「先
小规模」），这也是它属于 ``evaluation`` 层而不是推理层的原因。Kronos 的 VRAM 观测由
调用方以 ``vram_sampler`` 注入（RX-KAI-019），本模块不反向依赖推理实现。
"""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from kronos_ai.domain.hashing import sha256_hex
from kronos_ai.errors import ConfigurationError

COMPUTE_METRICS_VERSION = "compute-metrics-v1"

#: §16 的硬上限：「probe 总组数 <= 10」。
MAX_PROBE_CELLS = 10

#: 与 §6 / §17 runtime 的 device-class 同源（``forecast.backends.kronos.runtime``）。
#: 在此重述是因为本模块不得引入 torch；一致性由
#: ``tests/unit/test_evaluation_compute_metrics.py::TestDeviceClassContract`` 用
#: :func:`typing.get_args` 断言两边相等（与 CLI cutoff choices 同一套约定）。
ProbeDeviceClass = Literal["cpu", "mps", "cuda"]

#: §16 的超限标记；这是报告里唯一被允许表示「没跑成」的取值（不含「跳过」）。
ProbeCellStatus = Literal["MEASURED", "INFEASIBLE_ON_THIS_DEVICE"]


def percentile(samples: Sequence[float], q: float) -> float:
    """最近秩法分位数：``samples`` 升序排序后取第 ``ceil(q * n)`` 个元素（1-based）。

    选择最近秩而不是线性插值，是为了让 p95 一定是**真实观测到的一次延迟**，而不是
    两个邻居插出来的、没有任何一次运行经历过的数字。§49 的 p50 / p95 会直接写进
    compute budget 报告并被用来判断「这台机器跑不跑得完」，所以「能被指到某一次
    具体运行」比平滑更重要。这与 Prometheus / HAProxy 的 summary 口径一致。

    前置条件：``samples`` 非空且每个元素都是非负有限数，``0 < q <= 1``。
    """
    if not samples:
        raise ConfigurationError("percentile requires at least one sample")
    if not 0 < q <= 1:
        raise ConfigurationError(f"percentile quantile must be in (0, 1], got {q}")
    for value in samples:
        if not math.isfinite(value) or value < 0:
            raise ConfigurationError(
                f"percentile samples must be finite and non-negative, got {value!r}"
            )
    ordered = sorted(samples)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


class LatencySummary(BaseModel):
    """一次 probe 组合内每次迭代的延迟分布（毫秒）。

    一次「迭代」= 一次 :data:`MeasureCallable` 调用，它可以包含 ``batch_size`` 个
    forecast（§16 的 batch 维度）。因此这里记录的是**批量延迟**，而
    :attr:`ProbeCellResult.throughput_per_second` 是按完成的 forecast 数折算的吞吐。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    count: int = Field(ge=1)
    p50_ms: float = Field(ge=0)
    p95_ms: float = Field(ge=0)
    mean_ms: float = Field(ge=0)
    min_ms: float = Field(ge=0)
    max_ms: float = Field(ge=0)

    @classmethod
    def from_samples(cls, samples_ms: Sequence[float]) -> LatencySummary:
        """从原始迭代延迟构造；分位数口径见 :func:`percentile`。"""
        if not samples_ms:
            raise ConfigurationError("LatencySummary requires at least one sample")
        return cls(
            count=len(samples_ms),
            p50_ms=percentile(samples_ms, 0.5),
            p95_ms=percentile(samples_ms, 0.95),
            mean_ms=math.fsum(samples_ms) / len(samples_ms),
            min_ms=min(samples_ms),
            max_ms=max(samples_ms),
        )

    @model_validator(mode="after")
    def _ordered(self) -> LatencySummary:
        bounds = (
            ("p50", self.p50_ms),
            ("p95", self.p95_ms),
            ("mean", self.mean_ms),
        )
        for name, value in bounds:
            if not self.min_ms <= value <= self.max_ms:
                raise ValueError(
                    f"latency distribution is inconsistent: min={self.min_ms} "
                    f"{name}={value} max={self.max_ms}"
                )
        return self


@dataclass(frozen=True)
class ResourceSample:
    """一次资源采样。

    ``None`` 表示**该维度在本设备上不可观测**，不是 0：VRAM 在纯 CPU 环境上确实
    不存在，用 0 冒充会让「内存预算是否被守住」看起来永远成立（ADR-010）。
    """

    ram_bytes: int | None = None
    vram_bytes: int | None = None

    def __post_init__(self) -> None:
        for name in ("ram_bytes", "vram_bytes"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative or None, got {value}")


class ResourcePeak(BaseModel):
    """一次 probe 组合期间观测到的资源峰值（§16 的 RAM peak / VRAM peak）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ram_bytes: int | None = Field(default=None, ge=0)
    vram_bytes: int | None = Field(default=None, ge=0)

    def exceeded(self, limit_bytes: int) -> str | None:
        """返回超限描述；未超限返回 ``None``。

        同一个 ``limit_bytes`` 分别施加在**每个可观测维度**上（§16 的预算是一个标量
        上限）：RAM 与 VRAM 是两种异构存储，各自独立地与上限比较。

        两个维度都不可观测时这里返回 ``None``，因此还要靠
        :meth:`_PeakTracker.require_observable` 显式失败——否则「预算守住了」会是
        因为什么都没测到。
        """
        for name, value in (("ram", self.ram_bytes), ("vram", self.vram_bytes)):
            if value is not None and value > limit_bytes:
                return f"{name} peak {value} bytes > memory budget {limit_bytes} bytes"
        return None


class _PeakTracker:
    """可变峰值累加器（:class:`ResourcePeak` 本身保持不可变）。"""

    __slots__ = ("_ram", "_vram")

    def __init__(self) -> None:
        self._ram: int | None = None
        self._vram: int | None = None

    def observe(self, sample: ResourceSample) -> None:
        if sample.ram_bytes is not None:
            self._ram = sample.ram_bytes if self._ram is None else max(self._ram, sample.ram_bytes)
        if sample.vram_bytes is not None:
            self._vram = (
                sample.vram_bytes if self._vram is None else max(self._vram, sample.vram_bytes)
            )

    def require_observable(self, cell: ProbeCell) -> None:
        """内存预算必须能在**宿主 RAM** 上执行，否则拒绝运行该组合（ADR-010）。

        不要求 VRAM（纯 CPU 环境确实没有），但要求 RAM：任何进程都有宿主内存，若采样器
        连 RAM 都观测不到，内存上限就只对 VRAM 生效、对 RAM 实际无界，报告里的
        ``all_feasible`` 会变成一句空话（§16）。
        """
        if self._ram is None:
            raise ConfigurationError(
                f"resource sampler reported no ram observation for cell {cell.label}: "
                "a memory budget that cannot be enforced on host ram would let the cell "
                "run unbounded (vram-only sampling is not enough, §16)"
            )

    def peak(self) -> ResourcePeak:
        return ResourcePeak(ram_bytes=self._ram, vram_bytes=self._vram)


@runtime_checkable
class ResourceSampler(Protocol):
    """资源采样器契约：一次调用返回当前 RAM / VRAM 占用。"""

    def sample(self) -> ResourceSample: ...


def max_rss_bytes() -> int:
    """本进程 RSS 峰值（``getrusage(RUSAGE_SELF).ru_maxrss``）。

    POSIX 没有规定 ``ru_maxrss`` 的单位：macOS 是 bytes，Linux 是 kibibytes。
    这里按平台换算并记录在 ADR-019 里；不猜第三种平台的行为——未知平台按 Linux 口径
    处理（最保守：数值更大，更容易触发上限而不是更容易漏过）。
    """
    import resource  # POSIX-only；放在函数内，保持模块 import 廉价且平台可探测

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if usage <= 0:
        raise ConfigurationError(f"getrusage returned a non-positive ru_maxrss: {usage}")
    return usage if sys.platform == "darwin" else usage * 1024


class ProcessResourceSampler:
    """默认采样器：RAM 取本进程 RSS 峰值；VRAM 由可选的 ``vram_sampler`` 注入。

    不注入 ``vram_sampler`` 时 VRAM 为 ``None``（不可观测），绝不用 0 冒充——纯 CPU
    环境上这是正确结论，而不是缺失。

    **RAM 口径的已知偏保守性质**：``ru_maxrss`` 是**进程生命周期的高水位线**，单调
    不降，因此：(a) 每个组合读到的值是「到该组合结束时为止的进程峰值」，永远不会
    低估内存占用；(b) 一旦某个组合把进程峰值推过预算，其后每个组合在基线采样处就
    已超限、首轮即中止（级联 ``INFEASIBLE_ON_THIS_DEVICE``）——这是刻意的：进程确实
    到过那个水位，不该被当成「没超限」。需要**按组合归因**的内存曲线（§16 的
    throughput-memory curve）时，应由调用方注入每次调用返回**当前**占用的采样器
    （RX-KAI-019 接 torch 侧统计），:class:`_PeakTracker` 对当前值采样同样成立。
    """

    __slots__ = ("_vram_sampler",)

    def __init__(self, *, vram_sampler: Callable[[], int | None] | None = None) -> None:
        self._vram_sampler = vram_sampler

    def sample(self) -> ResourceSample:
        vram = None if self._vram_sampler is None else self._vram_sampler()
        return ResourceSample(ram_bytes=max_rss_bytes(), vram_bytes=vram)


class ProbeCell(BaseModel):
    """pilot matrix 的一组：一个 ``(sample_count, batch_size)`` 组合。

    ``batch_size`` 是 §16 的 batch 维度：``Kronos`` 原生实现把 batch 与 sample_count
    相乘后统一前向，实际内存随 ``batch × sample_count`` 增长。它表达「一次前向提交多少个
    origin」。``device_class`` 来自所属 matrix（§16 先锁定目标设备，不参与笛卡尔积）。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    device_class: ProbeDeviceClass
    sample_count: int = Field(ge=1)
    batch_size: int = Field(ge=1)

    @property
    def label(self) -> str:
        return f"{self.device_class}/samples={self.sample_count}/batch={self.batch_size}"

    @property
    def forward_width(self) -> int:
        """一次前向的宽度 = ``batch × sample_count``（§16 的内存相乘项）。"""
        return self.batch_size * self.sample_count


def _scan_axis(values: tuple[int, ...], *, name: str) -> tuple[int, ...]:
    if not values:
        raise ValueError(f"{name} must not be empty")
    if any(value < 1 for value in values):
        raise ValueError(f"{name} entries must be >= 1, got {values}")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates, got {values}")
    if tuple(sorted(values)) != values:
        raise ValueError(f"{name} must be ascending, got {values}")
    return values


class PilotMatrix(BaseModel):
    """Cost probe 的扫描计划（§16）。

    只有三个维度：**单一** ``device_class``、``sample_counts``、``batch_sizes``。
    §16 明确「先锁定目标设备（默认主力开发机一种）/ 其他设备仅跑单点验证」，所以设备
    不是扫描轴，而是 matrix 的一个标量；其他设备各自建一个单点 matrix 来验证。组数上限
    :data:`MAX_PROBE_CELLS` 由 model validator 守住，组合顺序规范化为
    ``sample_count`` 在外、``batch_size`` 在内，使报告可逐行 diff。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    device_class: ProbeDeviceClass
    sample_counts: tuple[int, ...]
    batch_sizes: tuple[int, ...]
    #: 全量 run 的规模，用于把测得的吞吐外推成 wall-clock 估计（§16 的目标之一）。
    symbols: int = Field(ge=1)
    forecast_origins: int = Field(ge=1)

    @field_validator("sample_counts", "batch_sizes")
    @classmethod
    def _valid_axis(cls, value: tuple[int, ...], info: Any) -> tuple[int, ...]:
        return _scan_axis(value, name=info.field_name)

    @model_validator(mode="after")
    def _bounded_scan(self) -> PilotMatrix:
        cells = len(self.sample_counts) * len(self.batch_sizes)
        if cells > MAX_PROBE_CELLS:
            raise ValueError(
                f"pilot matrix declares {cells} cells "
                f"({len(self.sample_counts)} sample_counts x "
                f"{len(self.batch_sizes)} batch_sizes) but §16 caps a probe at "
                f"{MAX_PROBE_CELLS}: trim the scan, do not run a full cartesian product"
            )
        return self

    @property
    def cells(self) -> tuple[ProbeCell, ...]:
        """规范化顺序的组列表：``sample_count`` 外层升序、``batch_size`` 内层升序。"""
        return tuple(
            ProbeCell(
                device_class=self.device_class,
                sample_count=sample_count,
                batch_size=batch_size,
            )
            for sample_count in self.sample_counts
            for batch_size in self.batch_sizes
        )

    @property
    def declared_forecasts(self) -> int:
        """按 matrix 声明，全量 run 需要多少次 forecast（§16 的外推基数）。"""
        return self.symbols * self.forecast_origins


class ProbeBudget(BaseModel):
    """Probe 的预算上限（§16「Probe 不允许无上限运行」）。

    两个字段都没有默认值：想跑 probe 就必须先说明愿意花多少 wall-clock 与多少内存。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    wall_clock_seconds_per_cell: float = Field(gt=0)
    memory_bytes: int = Field(gt=0)

    @field_validator("wall_clock_seconds_per_cell")
    @classmethod
    def _finite_wall_clock(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError(f"wall_clock_seconds_per_cell must be finite, got {value}")
        return value


class ProbeObservation(BaseModel):
    """一次迭代的产物侧观测（由调用方提供，probe 自己不认识 forecast 类型）。

    ``forecast_count`` 是这次迭代实际完成的 forecast 数（``batch_size`` 个 origin 即
    为 ``batch_size``）。``artifact_bytes`` / ``was_cached`` 是可选的：缺失表示这次迭代
    没有落盘 / 没有缓存层，报告里对应指标为 ``None``，而不是 0。``artifact_bytes`` 是
    这次迭代写出的总字节数，报告按这次迭代完成的 forecast 数摊成 §49 的
    「artifact bytes / forecast origin」。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    forecast_count: int = Field(default=1, ge=1)
    artifact_bytes: int | None = Field(default=None, ge=0)
    was_cached: bool | None = None


class ProbeCellResult(BaseModel):
    """一个组合的测量结果。

    ``MEASURED`` 与 ``INFEASIBLE_ON_THIS_DEVICE`` 是两个互斥的完备口径：前者一定带
    ``latency`` / ``throughput_per_second`` 且没有 ``abort_reason``；后者一定带
    ``abort_reason`` 且**不带**任何从被截断运行算出来的吞吐指标。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    cell: ProbeCell
    status: ProbeCellStatus
    completed_repeats: int = Field(ge=0)
    wall_clock_seconds: float = Field(ge=0)
    resources: ResourcePeak
    latency: LatencySummary | None = None
    throughput_per_second: float | None = Field(default=None, gt=0)
    artifact_bytes_per_forecast: float | None = Field(default=None, ge=0)
    cache_hit_ratio: float | None = Field(default=None, ge=0, le=1)
    abort_reason: str | None = None

    @model_validator(mode="after")
    def _status_is_complete(self) -> ProbeCellResult:
        if self.status == "MEASURED":
            if self.latency is None or self.throughput_per_second is None:
                raise ValueError(
                    f"MEASURED cell {self.cell.label} must report latency and throughput"
                )
            if self.abort_reason is not None:
                raise ValueError(f"MEASURED cell {self.cell.label} must not carry an abort reason")
        else:
            if self.abort_reason is None:
                raise ValueError(f"INFEASIBLE cell {self.cell.label} must carry an abort reason")
            if (
                self.latency is not None
                or self.throughput_per_second is not None
                or self.artifact_bytes_per_forecast is not None
                or self.cache_hit_ratio is not None
            ):
                raise ValueError(
                    f"INFEASIBLE cell {self.cell.label} must not report metrics computed "
                    "from a truncated run (§16)"
                )
        return self


MeasureCallable = Callable[[ProbeCell], ProbeObservation]


class ThroughputMemoryPoint(BaseModel):
    """§16 throughput-memory curve 上的一点：一个组合的实测吞吐与资源峰值。

    ``ram_bytes`` / ``vram_bytes`` 沿用 :class:`ResourcePeak` 的语义：``None`` 表示该
    维度不可观测，不是 0。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    cell: ProbeCell
    throughput_per_second: float = Field(gt=0)
    ram_bytes: int | None = Field(default=None, ge=0)
    vram_bytes: int | None = Field(default=None, ge=0)

    @property
    def forward_width(self) -> int:
        """曲线横轴：``batch × sample_count``（§16「内存随两者相乘增长」）。"""
        return self.cell.forward_width


def _run_cell(
    cell: ProbeCell,
    *,
    repeats: int,
    budget: ProbeBudget,
    measure: MeasureCallable,
    sampler: ResourceSampler,
    clock: Callable[[], float],
) -> ProbeCellResult:
    """执行一个组合，直到跑满 ``repeats`` 次或触发预算上限。"""
    start = clock()
    latencies_ms: list[float] = []
    forecast_count = 0
    artifact_bytes_total = 0
    artifact_forecast_count = 0
    cached_flags: list[bool] = []
    peak = _PeakTracker()
    peak.observe(sampler.sample())
    peak.require_observable(cell)
    abort_reason: str | None = None

    for _ in range(repeats):
        if clock() - start > budget.wall_clock_seconds_per_cell:
            abort_reason = (
                f"wall-clock budget {budget.wall_clock_seconds_per_cell}s per cell exceeded "
                f"after {len(latencies_ms)} of {repeats} repeats"
            )
            break
        before = clock()
        observation = measure(cell)
        latencies_ms.append((clock() - before) * 1000.0)
        forecast_count += observation.forecast_count
        if observation.artifact_bytes is not None:
            artifact_bytes_total += observation.artifact_bytes
            artifact_forecast_count += observation.forecast_count
        if observation.was_cached is not None:
            cached_flags.append(observation.was_cached)
        peak.observe(sampler.sample())
        exceeded = peak.peak().exceeded(budget.memory_bytes)
        if exceeded is not None:
            abort_reason = f"memory budget exceeded after {len(latencies_ms)} repeats: {exceeded}"
            break

    wall_clock_seconds = clock() - start
    resources = peak.peak()
    if abort_reason is None and wall_clock_seconds > budget.wall_clock_seconds_per_cell:
        # 单次 measure 调用无法被抢占：末轮把预算撑爆时只能事后发现（§16「任一组合
        # 超限」），不能因为「循环跑完了」就报成 MEASURED。
        abort_reason = (
            f"wall-clock budget {budget.wall_clock_seconds_per_cell}s per cell exceeded: "
            f"the cell took {wall_clock_seconds}s over {len(latencies_ms)} repeats"
        )
    if abort_reason is not None:
        return ProbeCellResult(
            cell=cell,
            status="INFEASIBLE_ON_THIS_DEVICE",
            completed_repeats=len(latencies_ms),
            wall_clock_seconds=wall_clock_seconds,
            resources=resources,
            abort_reason=abort_reason,
        )
    measured_seconds = math.fsum(latencies_ms) / 1000.0
    if measured_seconds <= 0:
        raise ConfigurationError(
            f"cell {cell.label} measured a non-positive elapsed time ({measured_seconds}s); "
            "the injected clock must advance monotonically"
        )
    return ProbeCellResult(
        cell=cell,
        status="MEASURED",
        completed_repeats=len(latencies_ms),
        wall_clock_seconds=wall_clock_seconds,
        resources=resources,
        latency=LatencySummary.from_samples(latencies_ms),
        throughput_per_second=forecast_count / measured_seconds,
        artifact_bytes_per_forecast=(
            artifact_bytes_total / artifact_forecast_count if artifact_forecast_count else None
        ),
        cache_hit_ratio=(
            sum(1 for flag in cached_flags if flag) / len(cached_flags) if cached_flags else None
        ),
    )


def run_cost_probe(
    *,
    matrix: PilotMatrix,
    budget: ProbeBudget,
    repeats_per_cell: int,
    measure: MeasureCallable,
    resource_sampler: ResourceSampler | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> ComputeBudgetReport:
    """按 ``matrix`` 逐组执行 probe，产出 §16 的 Compute Budget Report。

    ``measure`` 由调用方注入：执行**一次**迭代（可以是 ``batch_size`` 个 origin 的
    一次前向）并返回 :class:`ProbeObservation`。这使 probe 引擎不认识 backend、
    artifact 或 provider，可以在 CI 里用假 measure 验证预算与统计口径本身
    （RX-KAI-019 负责把 Kronos / baseline backend 接上）。

    循环是**串行**的：并发由调用方在 ``measure`` 内部实现（§16 的 worker 并行度）。
    这样「测到的吞吐」与「实际并行度」的对应关系由调用方显式声明，而不是由引擎暗中
    产生。

    ``clock`` 可注入（默认 :func:`time.perf_counter`），使预算逻辑可被确定性测试。
    """
    if repeats_per_cell < 1:
        raise ConfigurationError(f"repeats_per_cell must be >= 1, got {repeats_per_cell}")
    sampler: ResourceSampler = (
        resource_sampler if resource_sampler is not None else ProcessResourceSampler()
    )
    if not isinstance(sampler, ResourceSampler):
        raise ConfigurationError(
            "resource_sampler must implement ResourceSampler.sample(), got "
            f"{type(sampler).__name__}"
        )
    return ComputeBudgetReport(
        matrix=matrix,
        budget=budget,
        repeats_per_cell=repeats_per_cell,
        cells=tuple(
            _run_cell(
                cell,
                repeats=repeats_per_cell,
                budget=budget,
                measure=measure,
                sampler=sampler,
                clock=clock,
            )
            for cell in matrix.cells
        ),
    )


class ComputeBudgetReport(BaseModel):
    """§16 的 Benchmark Compute Budget Report。

    报告必须**完整覆盖** matrix 声明的每一组：``cells`` 的顺序与 :attr:`PilotMatrix.cells`
    逐一相同。手工删掉一个 INFEASIBLE 组合拼一份「全部可行」的报告会被 model validator
    拒绝——§16 存在的意义就是让「这台机器跑不动」这个结论无法被安静地抹掉。

    同理，``MEASURED`` 组合必须真的跑满 ``repeats_per_cell`` 次、且总耗时不得超出该组合的
    wall-clock 预算：手工把一个只跑了 1/10 次的结果标成 ``MEASURED`` 并附上由那 1 次算出的
    p50 会被拒绝（否则就绕过了 :class:`ProbeCellResult` 的「截断运行不给指标」约束）；
    手工把一个跑满但超预算的单元标成 ``MEASURED`` 同样会被拒（§16 的可行性标签不该靠
    构造报告的人自觉）。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    matrix: PilotMatrix
    budget: ProbeBudget
    repeats_per_cell: int = Field(ge=1)
    cells: tuple[ProbeCellResult, ...]
    version: str = COMPUTE_METRICS_VERSION

    @field_validator("version")
    @classmethod
    def _known_version(cls, value: str) -> str:
        if value != COMPUTE_METRICS_VERSION:
            raise ValueError(
                f"unknown compute metrics version {value!r}; expected {COMPUTE_METRICS_VERSION!r}"
            )
        return value

    @model_validator(mode="after")
    def _covers_declared_matrix(self) -> ComputeBudgetReport:
        declared = self.matrix.cells
        if len(declared) > MAX_PROBE_CELLS:
            # 正常构造路径已被 PilotMatrix 挡住；这一条防的是 model_construct 之类的
            # 绕过（§16 的组数上限是报告级不变量，不只是 matrix 级）。
            raise ValueError(
                f"pilot matrix declares {len(declared)} cells but §16 caps a probe at "
                f"{MAX_PROBE_CELLS}"
            )
        actual = tuple(cell.cell for cell in self.cells)
        if actual != declared:
            raise ValueError(
                f"report must cover the declared pilot matrix in order: expected "
                f"{[cell.label for cell in declared]}, got {[cell.label for cell in actual]}"
            )
        for cell in self.cells:
            if cell.status != "MEASURED":
                continue
            if cell.completed_repeats != self.repeats_per_cell:
                raise ValueError(
                    f"MEASURED cell {cell.cell.label} completed {cell.completed_repeats} of "
                    f"{self.repeats_per_cell} repeats: a truncated run must be reported as "
                    "INFEASIBLE_ON_THIS_DEVICE (§16)"
                )
            if cell.wall_clock_seconds > self.budget.wall_clock_seconds_per_cell:
                raise ValueError(
                    f"MEASURED cell {cell.cell.label} took {cell.wall_clock_seconds}s, over the "
                    f"{self.budget.wall_clock_seconds_per_cell}s per-cell wall-clock budget: a "
                    "cell that overshoots the budget must be reported as "
                    "INFEASIBLE_ON_THIS_DEVICE (§16)"
                )
        return self

    @property
    def throughput_memory_curve(self) -> tuple[ThroughputMemoryPoint, ...]:
        """§16 的 throughput-memory curve：实测吞吐随 ``batch × sample_count`` 的变化。

        只取 ``MEASURED`` 组合（被截断的运行没有可用吞吐），按 ``forward_width`` 升序；
        ``forward_width`` 相同的组合保持 matrix 的规范顺序（稳定排序），因此同一份输入
        永远给出同一份曲线。一个组合都没测到时返回空元组——不伪造曲线。
        """
        points: list[ThroughputMemoryPoint] = []
        for cell in self.measured_cells:
            throughput = cell.throughput_per_second
            assert throughput is not None, (
                "MEASURED cells always carry throughput (see ProbeCellResult._status_is_complete)"
            )
            points.append(
                ThroughputMemoryPoint(
                    cell=cell.cell,
                    throughput_per_second=throughput,
                    ram_bytes=cell.resources.ram_bytes,
                    vram_bytes=cell.resources.vram_bytes,
                )
            )
        return tuple(sorted(points, key=lambda point: point.forward_width))

    @property
    def measured_cells(self) -> tuple[ProbeCellResult, ...]:
        return tuple(cell for cell in self.cells if cell.status == "MEASURED")

    @property
    def infeasible_cells(self) -> tuple[ProbeCellResult, ...]:
        return tuple(cell for cell in self.cells if cell.status == "INFEASIBLE_ON_THIS_DEVICE")

    @property
    def all_feasible(self) -> bool:
        """只有当每一组都被测到才为真；任何 INFEASIBLE 组合都让它为假（§16）。"""
        return not self.infeasible_cells

    @property
    def total_wall_clock_seconds(self) -> float:
        return math.fsum(cell.wall_clock_seconds for cell in self.cells)

    @property
    def total_forecasts(self) -> int:
        """实际完成的 forecast 数（不含被中止组合中未跑完的部分）。"""
        return sum(cell.completed_repeats * cell.cell.batch_size for cell in self.measured_cells)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def report_hash(self) -> str:
        """报告内容的 sha256（同 ``dataset_hash`` 的做法：实时派生、不落盘）。"""
        return compute_report_hash(
            matrix=self.matrix,
            budget=self.budget,
            repeats_per_cell=self.repeats_per_cell,
            cells=self.cells,
        )


def _cell_payload(cell: ProbeCellResult) -> dict[str, Any]:
    return {
        "cell": cell.cell.model_dump(),
        "status": cell.status,
        "completed_repeats": cell.completed_repeats,
        "wall_clock_seconds": cell.wall_clock_seconds,
        "resources": cell.resources.model_dump(),
        "latency": None if cell.latency is None else cell.latency.model_dump(),
        "throughput_per_second": cell.throughput_per_second,
        "artifact_bytes_per_forecast": cell.artifact_bytes_per_forecast,
        "cache_hit_ratio": cell.cache_hit_ratio,
        "abort_reason": cell.abort_reason,
    }


def compute_report_hash(
    *,
    matrix: PilotMatrix,
    budget: ProbeBudget,
    repeats_per_cell: int,
    cells: tuple[ProbeCellResult, ...],
) -> str:
    """Compute budget report 的哈希载荷（§32 的可追溯性要求）。

    版本号进载荷，使「换了指标数学」与「换了机器」产生不同的 report_hash；两个都换了
    却撞同一个 hash 是不可能的。
    """
    payload: dict[str, Any] = {
        "kind": "compute_budget_report",
        "compute_metrics_version": COMPUTE_METRICS_VERSION,
        "matrix": matrix.model_dump(),
        "budget": budget.model_dump(),
        "repeats_per_cell": repeats_per_cell,
        "cells": [_cell_payload(cell) for cell in cells],
    }
    assert set(payload) == {
        "kind",
        "compute_metrics_version",
        "matrix",
        "budget",
        "repeats_per_cell",
        "cells",
    }, "compute_report_hash payload must cover every report dimension"
    return sha256_hex(payload)


def project_full_run_seconds(
    *,
    cell_result: ProbeCellResult,
    full_forecasts: int,
    parallelism: int = 1,
) -> float:
    """用某个**已测得**组合的吞吐外推全量 run 的 wall-clock（§16 的目标之一）。

    这是外推，不是测量：它假设吞吐不随规模变化、且全量 run 使用同一设备与同一
    ``sample_count`` / ``batch``。§16 因此要求 probe 报告只作为决策输入——「full
    universe / full period / sample_count / hardware requirement」在读完报告之后才确定。

    ``parallelism`` 是 worker 并行度（§16 的 batch / worker 维度）。线性假设同样写在
    这里：并行效率不会超过 1。
    """
    if cell_result.status != "MEASURED":
        raise ConfigurationError(
            f"cannot project from cell {cell_result.cell.label} with status "
            f"{cell_result.status}: run it inside the budget first (§16)"
        )
    throughput = cell_result.throughput_per_second
    if throughput is None or throughput <= 0:
        raise ConfigurationError(
            f"cell {cell_result.cell.label} has no usable throughput to project from"
        )
    if full_forecasts < 1:
        raise ConfigurationError(f"full_forecasts must be >= 1, got {full_forecasts}")
    if parallelism < 1:
        raise ConfigurationError(f"parallelism must be >= 1, got {parallelism}")
    return full_forecasts / (throughput * parallelism)
