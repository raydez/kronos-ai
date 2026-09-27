"""Forecast Benchmark v1 runner（基线文档 §41 / §48 / §49；ADR-023）。

§41 把 Phase 2 的产出定义为一次 walk-forward benchmark run：在**同一批** point-in-time
origin 上跑多个 forecast backend，用同一份样本/指标代码比较，并落盘成可追溯 artifact。
本模块就是那次 run 的编排层，它把三样东西接起来：

```text
WalkForwardDataset（§27，哪些 origin、哪一段、label 窗口）   ← RX-KAI-017
        +
MarketDataProvider（§18，origin 当时可见的 history）        ← RX-KAI-004
        +
LabelDataProvider（本模块，label 窗口的 ground truth）
        ↓
ForecastBackend.forecast（§17，Kronos / baseline 同接口）
        ↓
ForecastEvalRecord × backends  →  evaluate_forecast_records（§49）→ 报告
```

三条边界必须显式，否则 benchmark 会产出「看起来能比」的假比较：

1. **未来数据只能进 label，不能进 feature**。history 由 provider 按 ``knowledge_cutoff``
   截断（§29 的第一道守卫在 :class:`~kronos_ai.domain.market.MarketHistory` 的构造期）；
   label 窗口的 bar 只能经 :class:`LabelDataProvider` 取得，且只喂给 ``build_label``。
   runner 自身不做任何「为了凑数」的回填。
2. **分段与 embargo 不在 runner 里重新实现**：分段来自 dataset（§27/ADR-022），runner 只
   按 dataset 给定的段遍历，因此不可能出现「runner 自己发明的切分」。
3. **失败显式**：某个 origin 的 history 不足 ``lookback_bars``、backend 抛错、label 越界
   都会让整次 run 失败（ADR-010），而不是静默跳过该 origin 后给出一份更漂亮的指标。

``origin_limit`` 是**唯一**允许缩小样本量的旋钮（§48 的 pilot）：它被显式记录在结果里
（``truncated`` / ``considered_origins`` / ``evaluated_origins``），因此「这份报告是在
多少个 origin 上得出的」永远可以从 artifact 里读出来，而不必去猜配置。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kronos_ai.data.base import MarketDataProvider
from kronos_ai.domain.forecast import ForecastRequest, ForecastResult, SamplingConfig
from kronos_ai.domain.hashing import sha256_hex
from kronos_ai.domain.market import MarketBar
from kronos_ai.domain.time import SessionDate
from kronos_ai.errors import ConfigurationError, InsufficientHistoryError
from kronos_ai.evaluation.dataset import (
    ForecastLabel,
    ForecastOrigin,
    LabelPolicy,
    WalkForwardDataset,
    build_label,
    direction_label,
)
from kronos_ai.evaluation.forecast_metrics import (
    DEFAULT_COVERAGE_SPEC,
    FORECAST_EVAL_METRICS_VERSION,
    CoverageSpec,
    ForecastEvalMetrics,
    ForecastEvalRecord,
    coverage_spec_hash,
    evaluate_forecast_records,
)
from kronos_ai.evaluation.regimes import (
    DEFAULT_REGIME_SPEC,
    RegimeSpec,
    classify_regime,
    regime_spec_hash,
)
from kronos_ai.evaluation.walk_forward import SegmentName
from kronos_ai.forecast.base import ForecastBackend, require_aligned

BENCHMARK_VERSION = "forecast-benchmark-v1"


@runtime_checkable
class LabelDataProvider(Protocol):
    """label 窗口的 ground truth 来源（与 §18 的 PIT provider 分开）。

    这是一个**故意**分开的接口。forecast 输入必须是 ``available_at <= knowledge_cutoff``
    的 history；而 label 本质上需要 origin 之后的价格——把它塞进同一个 provider 会让
    「未来数据」与「当时可见数据」共享一条调用路径，泄漏与否只能靠纪律区分。分成两个对象
    之后，runner 里「哪一行代码碰了未来数据」是一眼可见的。

    ``data_coverage_end`` 是 provider 已发布的最后一个 market session（全局口径），
    :func:`~kronos_ai.evaluation.dataset.build_label` 用它区分「停牌」与「数据末端」。
    """

    def get_label_bars(
        self, symbol: str, market_date: date, sessions: Sequence[date]
    ) -> tuple[MarketBar, ...]:
        """返回 ``sessions`` 上的 bar（缺失的 session 直接不出现，即停牌）。

        实现不得为了凑齐 ``sessions`` 而合成 bar：缺失本身就是 label 的证据
        （``SUSPENDED``），补一根假 bar 会让停牌在指标里消失。
        """
        ...

    @property
    def data_coverage_end(self) -> date:
        """provider 已发布覆盖的最后一个 market session（含）。"""
        ...


class ForecastBenchmarkSpec(BaseModel):
    """一次 benchmark run 的显式规格（§41 / §48）。

    ``backends`` 是**名字**而不是实例：实例由调用方（CLI / 测试）装配后以 mapping 传入，
    未在 mapping 里出现的名字显式失败（§3.2），避免「配置里写了 kronos 但实际跑的是
    last_value」这种无法从 artifact 看出的错配。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    backends: tuple[str, ...]
    origin_limit: int | None = Field(default=None, ge=1)

    coverage: CoverageSpec = DEFAULT_COVERAGE_SPEC
    regime: RegimeSpec = DEFAULT_REGIME_SPEC
    version: str = BENCHMARK_VERSION

    @field_validator("backends")
    @classmethod
    def _backends_nonempty_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("backends must not be empty")
        cleaned = tuple(name.strip() for name in value)
        if any(not name for name in cleaned):
            raise ValueError("backend names must be non-empty")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError(f"backend names must be unique, got {list(cleaned)}")
        return cleaned

    @field_validator("version")
    @classmethod
    def _version_matches(cls, value: str) -> str:
        if value != BENCHMARK_VERSION:
            raise ValueError(
                f"unknown benchmark version {value!r}; this build emits {BENCHMARK_VERSION!r}"
            )
        return value

    def hashing_payload(self) -> dict[str, Any]:
        """进 config_hash 的 benchmark 维；完整覆盖所有字段（避免漏项导致 hash 失真）。"""
        payload: dict[str, Any] = {
            "version": self.version,
            "backends": list(self.backends),
            "origin_limit": self.origin_limit,
            "coverage": self.coverage.hashing_payload(),
            "regime": self.regime.hashing_payload(),
        }
        assert set(payload) == set(type(self).model_fields), (
            "ForecastBenchmarkSpec.hashing_payload must cover every model field"
        )
        return payload


class ForecastBenchmarkResult(BaseModel):
    """一次 benchmark run 的完整产物（内存态；落盘由 :func:`write_benchmark_artifacts`）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = BENCHMARK_VERSION
    dataset_hash: str
    label_policy_version: str
    lookback_bars: int = Field(ge=1)
    horizon_sessions: int = Field(ge=1)

    #: label 判定的数据覆盖末端（provider 已发布覆盖的最后一个 session）。
    #: ``build_label`` 用它区分「停牌」与「数据末端」，因此它直接决定哪些 origin 有 label；
    #: 进 hash 与 metadata，使「这份报告是在覆盖到哪一天的数据上算的」可被指认（§32）。
    data_coverage_end: SessionDate

    spec: ForecastBenchmarkSpec
    sampling: SamplingConfig

    records: tuple[ForecastEvalRecord, ...]
    metrics: tuple[ForecastEvalMetrics, ...]

    considered_origins: int = Field(ge=0)
    evaluated_origins: int = Field(ge=0)
    truncated: bool

    regime_spec_hash: str
    coverage_spec_hash: str

    @model_validator(mode="after")
    def _consistent(self) -> ForecastBenchmarkResult:
        if self.version != BENCHMARK_VERSION:
            raise ValueError(
                f"unknown benchmark version {self.version!r}; expected {BENCHMARK_VERSION!r}"
            )
        if self.evaluated_origins > self.considered_origins:
            raise ValueError(
                f"evaluated_origins {self.evaluated_origins} exceeds considered_origins "
                f"{self.considered_origins}"
            )
        if self.truncated != (self.evaluated_origins < self.considered_origins):
            raise ValueError(
                "truncated must be exactly the statement 'fewer origins were evaluated than "
                "considered'; the flag records the pilot subset, it is not free-form"
            )
        self._validate_metric_coverage()
        if self.records and self.data_coverage_end < max(
            record.market_date for record in self.records
        ):
            raise ValueError(
                f"data_coverage_end {self.data_coverage_end} precedes an evaluated origin; "
                "coverage must at least reach every origin it scored"
            )
        return self

    def _validate_metric_coverage(self) -> None:
        """每个 backend 必须有且只有一条跨段汇总（``segment=None``）的 ``overall`` 指标，
        且它的 ``sample_count`` 正好等于该 backend 的记录数（§49）。

        这条不变量是「报告标题数字 = 本次 run 的证据量」的机器可检查形式：缺一条会让
        ``metrics_for(..., segment=None)`` 静默返回 ``None``（报告显示 n/a），多一条或数量
        对不上则说明指标集合与记录集合不是同一份证据算出来的。
        """
        for backend in self.spec.backends:
            run_level = [
                entry
                for entry in self.metrics
                if entry.backend == backend
                and entry.segment is None
                and entry.group.axis == "overall"
            ]
            if len(run_level) != 1:
                raise ValueError(
                    f"backend {backend!r} must have exactly one run-level (segment=None) overall "
                    f"metric entry, found {len(run_level)}"
                )
            expected = sum(1 for record in self.records if record.backend == backend)
            if run_level[0].sample_count != expected:
                raise ValueError(
                    f"run-level metric for backend {backend!r} covers "
                    f"{run_level[0].sample_count} records but the result carries {expected}; "
                    "metrics and records must come from the same evidence"
                )

    @property
    def backends(self) -> tuple[str, ...]:
        return self.spec.backends

    @property
    def metrics_version(self) -> str:
        return FORECAST_EVAL_METRICS_VERSION

    def metrics_for(
        self, backend: str, *, segment: SegmentName | None = None, group: str = "all"
    ) -> ForecastEvalMetrics | None:
        """按 ``(backend, segment, group)`` 精确取指标；未找到返回 ``None``（不伪造 0）。

        ``segment=None`` 取的是**跨段汇总**那一条（整次 run 口径），不是「任意一段」：
        语义必须唯一，否则 CLI 摘要会把某一段的数字当成整体数字展示。
        """
        for entry in self.metrics:
            if entry.backend != backend:
                continue
            if entry.segment is not segment:
                continue
            if entry.group.label != group:
                continue
            return entry
        return None

    def hashing_payload(self) -> dict[str, Any]:
        """进 ``report_hash`` 的规范 payload。

        ``data_coverage_end`` **刻意缺席**：它是 run 当时的环境事实（数据发布到哪一天），
        不是结论的一部分——它唯一的后果已经逐条写在 ``records[].label_status`` 里，而记录
        本身进 hash。把它算进去，同一次 run 在第二天重跑就会得到不同 ``report_hash``，即使
        每一个数字都没变（与 ``git_commit`` 只进 metadata 同一个理由）。
        """
        payload: dict[str, Any] = {
            "kind": "forecast_benchmark_result",
            "version": self.version,
            "dataset_hash": self.dataset_hash,
            "label_policy_version": self.label_policy_version,
            "lookback_bars": self.lookback_bars,
            "horizon_sessions": self.horizon_sessions,
            "spec": self.spec.hashing_payload(),
            "sampling": self.sampling.hashing_payload(),
            "records": [record.hashing_payload() for record in self.records],
            "metrics": [entry.hashing_payload() for entry in self.metrics],
            "considered_origins": self.considered_origins,
            "evaluated_origins": self.evaluated_origins,
            "truncated": self.truncated,
            "regime_spec_hash": self.regime_spec_hash,
            "coverage_spec_hash": self.coverage_spec_hash,
        }
        hashed_fields = set(type(self).model_fields) - {"data_coverage_end"}
        assert set(payload) == hashed_fields | {"kind"}, (
            "ForecastBenchmarkResult.hashing_payload must cover every model field except "
            "data_coverage_end (see its docstring for why it is metadata-only)"
        )
        return payload

    @property
    def report_hash(self) -> str:
        """结果内容的 sha256（实时派生，与 dataset_hash 同一约定）。"""
        return sha256_hex(self.hashing_payload())


def _coverage_bounds(
    result: ForecastResult, coverage: CoverageSpec
) -> tuple[float | None, float | None]:
    """从 ``ForecastDistribution.quantiles`` 里取中心区间端点。

    端点缺失时返回 ``(None, None)``：coverage 对该 backend 无定义，而不是「覆盖率为 0」。
    分位是全等（确定性 baseline）时仍按实际取值返回，不特判——口径统一比「好看」重要。
    """
    lower: float | None = None
    upper: float | None = None
    for entry in result.distribution.quantiles:
        if entry.metric != "horizon_return":
            continue
        if entry.quantile == coverage.lower_quantile:
            lower = entry.value
        elif entry.quantile == coverage.upper_quantile:
            upper = entry.value
    if lower is None or upper is None:
        return None, None
    return lower, upper


def _build_request(
    origin: ForecastOrigin, sampling: SamplingConfig, horizon: int
) -> ForecastRequest:
    return ForecastRequest(
        symbol=origin.symbol,
        market_date=origin.market_date,
        knowledge_cutoff=origin.knowledge_cutoff,
        horizon=horizon,
        sampling=sampling,
    )


def run_forecast_benchmark(
    *,
    dataset: WalkForwardDataset,
    provider: MarketDataProvider,
    label_provider: LabelDataProvider,
    backends: Mapping[str, ForecastBackend],
    spec: ForecastBenchmarkSpec,
    sampling: SamplingConfig,
    clock: Callable[[], float] = time.perf_counter,
) -> ForecastBenchmarkResult:
    """在 ``dataset`` 的每个 origin 上跑 ``spec.backends``，产出 §49 指标（§41）。

    ``clock`` 可注入，使 latency 结果可被确定性测试（默认 ``time.perf_counter``）；
    它只用于记录，不参与任何指标计算。

    失败语义：backend 名缺失、history 不足、label 越界、backend 抛错都会让整次 run 失败
    （ADR-010）。runner 绝不「跳过这个 origin 继续」——那会让报告的分母不可解释。
    """
    missing = sorted(set(spec.backends) - set(backends))
    if missing:
        raise ConfigurationError(
            f"benchmark spec requests backends {missing} that were not provided; "
            f"provided: {sorted(backends)}"
        )
    policy: LabelPolicy = dataset.label_policy
    horizon = policy.horizon_sessions
    if dataset.lookback_bars < 1:
        raise ConfigurationError(f"dataset lookback_bars must be >= 1, got {dataset.lookback_bars}")
    regime_required = spec.regime.required_bars
    if dataset.lookback_bars < regime_required:
        raise ConfigurationError(
            f"regime spec needs at least {regime_required} bars of history "
            f"(trend_window={spec.regime.trend_window_sessions}, "
            f"volatility_window={spec.regime.volatility_window_sessions}) but the dataset only "
            f"provides lookback_bars={dataset.lookback_bars}; increase lookback_bars or shrink "
            "the regime windows (§49) — silently using a shorter window would make one regime "
            "label mean different things at different origins"
        )

    records: list[ForecastEvalRecord] = []
    considered = 0
    evaluated = 0
    coverage_end = label_provider.data_coverage_end

    for segment in dataset.segments:
        # 段内按 **(market_date, symbol)** 遍历：§48 的 pilot 是「先取最早的 session」，
        # 一个 session 上的全部 symbol 一起进样本。若按 symbol 优先遍历，``origin_limit``
        # 比 symbol 数还小时会只覆盖第一个 symbol，而 metadata 里只写了一个总数——那正是
        # 「universe 静默缩水」。顺序因此是契约的一部分，与 dataset 的段顺序叠加。
        for origin in sorted(segment.origins, key=lambda item: (item.market_date, item.symbol)):
            considered += 1
            if spec.origin_limit is not None and evaluated >= spec.origin_limit:
                continue
            if len(origin.label_sessions) != horizon:
                raise ConfigurationError(
                    f"origin {origin.symbol}@{origin.market_date} carries "
                    f"{len(origin.label_sessions)} label sessions but the label policy horizon "
                    f"is {horizon}"
                )
            request = _build_request(origin, sampling, horizon)
            history = provider.get_history(
                symbol=origin.symbol,
                market_date=origin.market_date,
                knowledge_cutoff=origin.knowledge_cutoff,
                lookback_bars=dataset.lookback_bars,
            )
            # 前置条件在 runner 里也验一次：backend 自己会验（§17 要求实现校验），但
            # 「runner 有没有把配对的输入交给 backend」不该由每个 backend 的实现纪律来保证。
            require_aligned(history, request)
            if len(history.bars) < dataset.lookback_bars:
                raise InsufficientHistoryError(
                    f"origin {origin.symbol}@{origin.market_date} got "
                    f"{len(history.bars)} history bars but the dataset declares "
                    f"lookback_bars={dataset.lookback_bars}; a shorter window would make one "
                    "regime/forecast label mean different things at different origins"
                )
            regime = classify_regime(history, spec=spec.regime)
            origin_close = history.bars[-1].close
            label = build_label(
                origin=origin,
                policy=policy,
                origin_close=origin_close,
                future_bars=label_provider.get_label_bars(
                    origin.symbol, origin.market_date, origin.label_sessions
                ),
                data_coverage_end=coverage_end,
            )
            for name in spec.backends:
                backend = backends[name]
                started = clock()
                result = backend.forecast(history, request)
                latency_ms = max(0.0, (clock() - started) * 1000.0)
                _require_result_matches_origin(name, origin, result)
                lower, upper = _coverage_bounds(result, spec.coverage)
                records.append(
                    _record_from(
                        segment=segment.name,
                        result=result,
                        label=label,
                        regime_trend=regime.trend,
                        regime_volatility=regime.volatility,
                        coverage=(lower, upper),
                        policy=policy,
                        latency_ms=latency_ms,
                    )
                )
            evaluated += 1

    return ForecastBenchmarkResult(
        dataset_hash=dataset.dataset_hash,
        label_policy_version=policy.version,
        lookback_bars=dataset.lookback_bars,
        horizon_sessions=horizon,
        data_coverage_end=coverage_end,
        spec=spec,
        sampling=sampling,
        records=tuple(records),
        metrics=evaluate_forecast_records(records, coverage=spec.coverage),
        considered_origins=considered,
        evaluated_origins=evaluated,
        truncated=evaluated < considered,
        regime_spec_hash=regime_spec_hash(spec.regime),
        coverage_spec_hash=coverage_spec_hash(spec.coverage),
    )


def _require_result_matches_origin(
    backend_name: str, origin: ForecastOrigin, result: ForecastResult
) -> None:
    """校验 backend 产出的 forecast 确实属于**这个** origin（§3.2 / ADR-023 §1）。

    runner 把 result 当作 origin 的预测、把 label 当作 origin 的真相：若 result 属于另一个
    ``(symbol, market_date)``，record 会把两条不同时点的证据拼成一条，而 artifact 里看不出
    任何异常（record 的 symbol/market_date 取自 result 自身）。这与 §17 的
    ``require_aligned``（history ↔ request）是同一类错配，只是发生在出口，必须同样显式失败。
    """
    if result.symbol != origin.symbol:
        raise ConfigurationError(
            f"backend {backend_name!r} returned a forecast for symbol {result.symbol!r} while "
            f"the origin is {origin.symbol!r}"
        )
    if result.market_date != origin.market_date:
        raise ConfigurationError(
            f"backend {backend_name!r} returned a forecast for market_date "
            f"{result.market_date} while the origin is {origin.market_date}"
        )
    if result.knowledge_cutoff != origin.knowledge_cutoff:
        raise ConfigurationError(
            f"backend {backend_name!r} returned a forecast with knowledge_cutoff "
            f"{result.knowledge_cutoff.isoformat()} while the origin is "
            f"{origin.knowledge_cutoff.isoformat()}"
        )
    if result.model.backend != backend_name:
        raise ConfigurationError(
            f"backend registered as {backend_name!r} produced a forecast whose "
            f"model.backend is {result.model.backend!r}; the record would be filed under a name "
            "the spec never asked for"
        )


def _record_from(
    *,
    segment: str,
    result: ForecastResult,
    label: ForecastLabel,
    regime_trend: str,
    regime_volatility: str,
    coverage: tuple[float | None, float | None],
    policy: LabelPolicy,
    latency_ms: float,
) -> ForecastEvalRecord:
    lower, upper = coverage
    return ForecastEvalRecord(
        segment=segment,  # type: ignore[arg-type]  # SegmentName 由 dataset 保证
        symbol=result.symbol,
        market_date=result.market_date,
        backend=result.model.backend,
        model_revision=result.model.revision,
        artifact_id=result.artifact_id,
        predicted_return=result.distribution.expected_return,
        median_return=result.distribution.median_return,
        predicted_direction=direction_label(result.distribution.expected_return, policy),
        coverage_lower=lower,
        coverage_upper=upper,
        label_status=label.status,
        realized_return=label.horizon_return,
        realized_direction=label.direction,
        trend_regime=regime_trend,  # type: ignore[arg-type]  # 由 classify_regime 给出
        volatility_regime=regime_volatility,  # type: ignore[arg-type]
        latency_ms=latency_ms,
    )


def evaluated_origins_by_symbol(result: ForecastBenchmarkResult) -> dict[str, int]:
    """每个 symbol 实际被评估了多少个 origin（由记录派生，不是从配置猜）。

    非空 origin 集合的分布是「pilot 有没有悄悄缩掉 universe」的唯一可见证据：只看
    ``evaluated_origins`` 一个总数，看不出 50 个 origin 是 50 个 symbol 各 1 次，
    还是 1 个 symbol 的 50 个 session。
    """
    counts: dict[str, int] = {}
    for record in result.records:
        counts[record.symbol] = counts.get(record.symbol, 0) + 1
    backends = max(1, len(result.backends))
    if any(count % backends for count in counts.values()):
        raise ConfigurationError(
            "records per symbol are not a multiple of the backend count; the record set is "
            f"inconsistent: {counts} vs {backends} backends"
        )
    return {symbol: count // backends for symbol, count in sorted(counts.items())}


def build_benchmark_run_metadata(
    *,
    result: ForecastBenchmarkResult,
    run_id: str,
    git_commit: str | None,
    config_hash: str,
    adjustment: str | None = None,
    backend_identities: Mapping[str, Mapping[str, str]] | None = None,
    environment: Mapping[str, str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """组装 §32 的 run metadata（可 trace git / data / model / config / seed）。

    ``adjustment`` / ``backend_identities`` / ``environment`` 都是**环境或数据身份**事实：
    metadata 是它们的落点（不进 ``report_hash``，理由同 ``git_commit``）。缺省时字段直接
    缺席，而不是写一个猜出来的值。
    """
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "kind": "benchmark_forecast",
        "git_commit": git_commit,
        "config_hash": config_hash,
        "dataset_hash": result.dataset_hash,
        "report_hash": result.report_hash,
        "benchmark_version": result.version,
        "forecast_eval_metrics_version": result.metrics_version,
        "label_policy_version": result.label_policy_version,
        "lookback_bars": result.lookback_bars,
        "horizon_sessions": result.horizon_sessions,
        "data_coverage_end": result.data_coverage_end.isoformat(),
        "backends": list(result.backends),
        "sampling": result.sampling.model_dump(mode="json"),
        "considered_origins": result.considered_origins,
        "evaluated_origins": result.evaluated_origins,
        "evaluated_origins_by_symbol": evaluated_origins_by_symbol(result),
        "truncated": result.truncated,
        "regime_spec_hash": result.regime_spec_hash,
        "coverage_spec_hash": result.coverage_spec_hash,
    }
    if adjustment is not None:
        metadata["adjustment"] = adjustment
    if backend_identities is not None:
        metadata["backend_identities"] = {
            name: dict(identity) for name, identity in sorted(backend_identities.items())
        }
    if environment is not None:
        metadata["environment"] = dict(environment)
    if extra is not None:
        metadata.update(extra)
    return metadata


class ConstantLabelProvider:
    """测试/离线用 label provider：显式提供每个 origin 的 label bar。

    它存在的意义是让 benchmark 的**编排**可被单测，而不是给生产路径提供「兜底数据」；
    因此它要求调用方显式给出每个 ``(symbol, market_date)`` 的 bar，缺失即显式失败
    （ADR-010：不合成、不插值）。
    """

    def __init__(
        self,
        *,
        bars: Mapping[tuple[str, date], Sequence[MarketBar]],
        data_coverage_end: date,
    ) -> None:
        self._bars = {key: tuple(value) for key, value in bars.items()}
        self._coverage_end = data_coverage_end

    @property
    def data_coverage_end(self) -> date:
        return self._coverage_end

    def get_label_bars(
        self, symbol: str, market_date: date, sessions: Sequence[date]
    ) -> tuple[MarketBar, ...]:
        key = (symbol, market_date)
        if key not in self._bars:
            raise ConfigurationError(
                f"no label bars registered for {symbol}@{market_date.isoformat()}; "
                "ConstantLabelProvider never synthesizes missing ground truth"
            )
        window = set(sessions)
        selected = tuple(bar for bar in self._bars[key] if bar.timestamp.date() in window)
        stray = {bar.timestamp.date() for bar in self._bars[key]} - window
        if stray:
            raise ConfigurationError(
                f"label bars for {symbol}@{market_date.isoformat()} contain sessions outside "
                f"the label window: {sorted(stray)}"
            )
        return selected


__all__ = [
    "BENCHMARK_VERSION",
    "ConstantLabelProvider",
    "ForecastBenchmarkResult",
    "ForecastBenchmarkSpec",
    "LabelDataProvider",
    "build_benchmark_run_metadata",
    "evaluated_origins_by_symbol",
    "run_forecast_benchmark",
]
