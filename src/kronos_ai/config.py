"""Experiment Config（基线文档 §32.1 / §34；ADR-023）。

§32.1 定义了三层配置：

```text
Pydantic Settings（环境变量 + 默认值）   → 进程级开关（artifacts 目录、device 等）
YAML Experiment Config（本模块）          → 一次可复现实验的完整描述
Environment Secret（仅环境变量）          → API_KEY / HF_TOKEN 之类
```

本模块只负责第二层，并落下三条硬规则：

1. **config 是可复现实验的单一描述**：YAML 解析成 :class:`ExperimentConfig` 后逐字段
   校验，``config_hash`` 覆盖全部字段（规范化 JSON 的 sha256）。注释与键顺序不影响
   hash，任何语义变化都会改变它——与 ``dataset_hash`` 同一约定。
2. **secret 只能来自环境变量**：YAML 里出现 ``api_key`` / ``token`` / ``secret`` /
   ``password`` 一类键即拒绝加载。这样「不小心把 token 写进 config 并归档进 run
   目录」在类型层不可能发生（§32.1）。
3. **embargo / horizon 的单一真源是 LabelPolicy**：config 里写了与版本化定义不一致的
   值时拒绝运行，而不是覆盖（§27、§32.1）。改 embargo 必须先新增 LabelPolicy 版本。

``sessions`` 不在 config 里逐日枚举，而是给出 ``start_session`` / ``end_session``：真实
时间轴必须来自 TradingCalendar（§6.6 的唯一真源），由调用方按区间切片并校验长度恰好
等于 ``lookback_bars + Σsegments + embargo × gaps``。长度不符即显式失败，不静默截断。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kronos_ai.data.adjustment import AdjustmentMode
from kronos_ai.domain.forecast import SamplingConfig
from kronos_ai.domain.hashing import sha256_hex
from kronos_ai.domain.time import KnowledgeCutoffPolicy, SessionDate
from kronos_ai.errors import ConfigurationError
from kronos_ai.evaluation.benchmark import ForecastBenchmarkSpec
from kronos_ai.evaluation.dataset import (
    LABEL_POLICY_VERSION,
    LabelPolicy,
    label_policy_from_experiment_config,
)
from kronos_ai.evaluation.gate import load_gate_criteria
from kronos_ai.evaluation.regimes import DEFAULT_REGIME_SPEC, RegimeSpec
from kronos_ai.evaluation.walk_forward import SegmentName, SegmentSpec, WalkForwardPlan

EXPERIMENT_CONFIG_VERSION = "experiment-config-v1"

CUTOFF_POLICIES: tuple[KnowledgeCutoffPolicy, ...] = (
    "market_close",
    "same_day_evening",
    "explicit",
)

#: 拒绝出现在 YAML 里的键名（大小写无关、下划线/连字符等价）。secret 只从环境变量读。
_SECRET_KEY_PATTERN = re.compile(r"(api[_-]?key|token|secret|password|credential)", re.IGNORECASE)

DeviceName = Literal["auto", "cpu", "mps", "cuda"]
DtypeName = Literal["auto", "float32"]


def _reject_secret_keys(payload: Any, *, path: str = "$") -> None:
    """递归拒绝 secret 键；错误消息给出 JSON 路径，便于定位。"""
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_text = str(key)
            if _SECRET_KEY_PATTERN.search(key_text):
                raise ConfigurationError(
                    f"experiment config contains a secret-like key at {path}.{key_text}: "
                    "secrets (API_KEY / HF_TOKEN / ...) must come from environment variables "
                    "only and must never be written into config or artifacts (§32.1)"
                )
            _reject_secret_keys(value, path=f"{path}.{key_text}")
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            _reject_secret_keys(item, path=f"{path}[{index}]")


class DataSection(BaseModel):
    """§6.1 / §32.1 ``data``：复权口径属于**数据身份**。

    raw / hfq / qfq 改变每一根 bar，因而改变每一个预测与每一个 label（ADR-007）。它必须
    随 run 归档并进 ``config_hash``：flag 只存在于命令行时，同一份 config.yaml 会对应两个
    不同结论，而 artifact 里没有一处能解释 ``report_hash`` 为什么变了（ADR-023 §6）。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: 直接用域里的 :data:`~kronos_ai.data.adjustment.AdjustmentMode`（§6.1 的单一真源）：
    #: 在这里再写一份字面量会让「新增一种口径」出现两个必须先同步的位置。
    adjustment: AdjustmentMode = "raw"

    def hashing_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"adjustment": self.adjustment}
        assert set(payload) == set(type(self).model_fields), (
            "DataSection.hashing_payload must cover every model field"
        )
        return payload


class RuntimeSection(BaseModel):
    """§32.1 ``runtime``：设备与精度只表达**意图**，resolved 值由 runtime 决定。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    device: DeviceName = "auto"
    dtype: DtypeName = "auto"

    def hashing_payload(self) -> dict[str, Any]:
        return {"device": self.device, "dtype": self.dtype}


class SamplingSection(BaseModel):
    """§32.1 ``forecast.sampling``；字段与 §8 SamplingConfig 一一对应。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = Field(default=42, ge=0)
    sample_count: int = Field(default=64, ge=1)
    temperature: float = Field(default=1.0, gt=0)
    top_k: int = Field(default=0, ge=0)
    top_p: float = Field(default=0.9, gt=0, le=1)

    def to_sampling_config(self) -> SamplingConfig:
        """物化为 §8 契约；seed 等字段的单点流转（不靠调用方拼装）。"""
        return SamplingConfig(
            seed=self.seed,
            sample_count=self.sample_count,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )

    def hashing_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "seed": self.seed,
            "sample_count": self.sample_count,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }
        assert set(payload) == set(type(self).model_fields), (
            "SamplingSection.hashing_payload must cover every model field"
        )
        return payload


class ForecastSection(BaseModel):
    """§32.1 ``forecast``：backend 名 + 模型名的**意图**（resolved 由 runtime 负责）。

    ``lookback_bars`` 在这里是单一真源：§8 明确它属于 backend/model 配置而不是 Request，
    dataset 与所有 backend 都用同一个值，避免「dataset 按 32 根切、backend 按 256 根读」。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: str = "kronos"
    model: str | None = None
    lookback_bars: int = Field(default=32, ge=1)
    sampling: SamplingSection = SamplingSection()

    @field_validator("backend")
    @classmethod
    def _backend_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("forecast.backend must be non-empty")
        return value.strip()

    def hashing_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "backend": self.backend,
            "model": self.model,
            "lookback_bars": self.lookback_bars,
            "sampling": self.sampling.hashing_payload(),
        }
        assert set(payload) == set(type(self).model_fields), (
            "ForecastSection.hashing_payload must cover every model field"
        )
        return payload


class UniverseSection(BaseModel):
    """§32.1 ``universe``；``snapshot`` 只允许 ``point-in-time``（§6.2）。

    **当前状态：声明性字段。** 生效的名单是 ``dataset.symbols``（本次 run 实际使用的
    symbol 列表），本节只记录「这份名单自称来自哪个 PIT universe」。把 ``source`` 解析成
    具体名单（``BaoStockUniverseLoader`` + 快照归档 ``universe.parquet``）是 §33 尚未落地
    的一环，已登记在 ADR-023 §8 的显式 defer 里——在那之前，不要把它读成「名单已被校验」。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = "csi300"
    snapshot: Literal["point-in-time"] = "point-in-time"

    @field_validator("source")
    @classmethod
    def _source_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("universe.source must be non-empty")
        return value.strip()

    def hashing_payload(self) -> dict[str, Any]:
        return {"source": self.source, "snapshot": self.snapshot}


class EvaluationSection(BaseModel):
    """§32.1 ``evaluation``；walk-forward 是唯一正式模式（§27 / DoD 23）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    walk_forward: Literal[True] = True
    label_policy_version: str = LABEL_POLICY_VERSION
    embargo_sessions: int | None = Field(default=None, ge=1)
    horizon_sessions: int | None = Field(default=None, ge=1)

    def hashing_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "walk_forward": self.walk_forward,
            "label_policy_version": self.label_policy_version,
            "embargo_sessions": self.embargo_sessions,
            "horizon_sessions": self.horizon_sessions,
        }
        assert set(payload) == set(type(self).model_fields), (
            "EvaluationSection.hashing_payload must cover every model field"
        )
        return payload


class SegmentSection(BaseModel):
    """§27 的一段：名字 + 以 market session 计的长度。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: SegmentName
    length_sessions: int = Field(ge=1)

    def hashing_payload(self) -> dict[str, Any]:
        return {"name": self.name, "length_sessions": self.length_sessions}


class DatasetSection(BaseModel):
    """§27 walk-forward dataset 的声明（时间轴由 TradingCalendar 提供）。

    ``symbols`` 显式列出：universe snapshot（§6.2）解析成具体 symbol 列表这一步属于
    provider/universe loader，config 里写「已经确定的名单」比写「某个指数名」更可复现
    ——指数成分会变，而本次 run 用的名单不该随时间漂移。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbols: tuple[str, ...]
    start_session: SessionDate
    end_session: SessionDate
    segments: tuple[SegmentSection, ...]

    @field_validator("symbols")
    @classmethod
    def _symbols_nonempty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("dataset.symbols must not be empty")
        cleaned = tuple(symbol.strip() for symbol in value)
        if any(not symbol for symbol in cleaned):
            raise ValueError("dataset.symbols entries must be non-empty")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError(f"dataset.symbols must be unique, got {list(cleaned)}")
        # 规范化排序：dataset 本身会把 symbols 排序（同一份数据不该因为书写顺序换
        # dataset_hash），config_hash 也必须与书写顺序无关——否则「键顺序无关、语义才有关」
        # 这条自述在 symbols 上就不成立。
        return tuple(sorted(cleaned))

    @field_validator("segments")
    @classmethod
    def _segments_nonempty(cls, value: tuple[SegmentSection, ...]) -> tuple[SegmentSection, ...]:
        if not value:
            raise ValueError("dataset.segments must not be empty")
        return value

    @model_validator(mode="after")
    def _window_ordered(self) -> DatasetSection:
        if self.start_session > self.end_session:
            raise ValueError(
                f"dataset.start_session {self.start_session} must be <= end_session "
                f"{self.end_session}"
            )
        return self

    def to_plan(self) -> WalkForwardPlan:
        """物化为 §27 的分段计划（embargo 不在这里：它是 LabelPolicy 的职责）。"""
        return WalkForwardPlan(
            segments=tuple(
                SegmentSpec(name=segment.name, length_sessions=segment.length_sessions)
                for segment in self.segments
            )
        )

    def hashing_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "symbols": list(self.symbols),
            "start_session": self.start_session.isoformat(),
            "end_session": self.end_session.isoformat(),
            "segments": [segment.hashing_payload() for segment in self.segments],
        }
        assert set(payload) == set(type(self).model_fields), (
            "DatasetSection.hashing_payload must cover every model field"
        )
        return payload


class GateSection(BaseModel):
    """§19 / §42 的 gate 预注册声明。

    只声明**判据文档在哪**：判据本身是独立文档（:class:`~kronos_ai.evaluation.gate.GateCriteria`），
    其 canonical hash 与原文随 run 归档。把路径而不是内容放进 config，是为了让
    ``config_hash`` 只依赖 config 自身（判据的内容身份由 ``gate_criteria_hash`` 承担，
    两者在 run metadata 里都在），同时「这次 run 用的是哪份判据」仍可从 config 读出。

    ``criteria_file`` 相对**本 config 所在目录**解析，因此 ``configs/`` 里的一组文件可以
    整体搬动而不失效。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    criteria_file: str

    @field_validator("criteria_file")
    @classmethod
    def _nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("gate.criteria_file must be non-empty")
        return value.strip()

    def hashing_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"criteria_file": self.criteria_file}
        assert set(payload) == set(type(self).model_fields), (
            "GateSection.hashing_payload must cover every model field"
        )
        return payload


class BenchmarkSection(BaseModel):
    """§48/§49 的 benchmark 声明：比较哪些 backend、是否 pilot 截断。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    backends: tuple[str, ...]
    origin_limit: int | None = Field(default=None, ge=1)

    @field_validator("backends")
    @classmethod
    def _backends_nonempty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("benchmark.backends must not be empty")
        cleaned = tuple(name.strip() for name in value)
        if any(not name for name in cleaned):
            raise ValueError("benchmark.backends entries must be non-empty")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError(f"benchmark.backends must be unique, got {list(cleaned)}")
        return cleaned

    def hashing_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "backends": list(self.backends),
            "origin_limit": self.origin_limit,
        }
        assert set(payload) == set(type(self).model_fields), (
            "BenchmarkSection.hashing_payload must cover every model field"
        )
        return payload


class ExperimentConfig(BaseModel):
    """§32.1 的一次可复现实验描述。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = EXPERIMENT_CONFIG_VERSION
    runtime: RuntimeSection = RuntimeSection()
    data: DataSection = DataSection()
    forecast: ForecastSection = ForecastSection()
    knowledge_cutoff_policy: KnowledgeCutoffPolicy = "same_day_evening"
    universe: UniverseSection = UniverseSection()
    evaluation: EvaluationSection = EvaluationSection()
    dataset: DatasetSection
    benchmark: BenchmarkSection
    regime: RegimeSpec = DEFAULT_REGIME_SPEC
    #: §42 的 gate 预注册声明；缺省表示这次 run 不判决（判决产物直接缺席，不留假结论）。
    gate: GateSection | None = None

    @field_validator("version")
    @classmethod
    def _version_matches(cls, value: str) -> str:
        if value != EXPERIMENT_CONFIG_VERSION:
            raise ValueError(
                f"unknown experiment config version {value!r}; this build emits "
                f"{EXPERIMENT_CONFIG_VERSION!r}"
            )
        return value

    @property
    def sampling(self) -> SamplingConfig:
        return self.forecast.sampling.to_sampling_config()

    def label_policy(self) -> LabelPolicy:
        """物化 §28 LabelPolicy，并拒绝 config 与版本化定义的不一致（§32.1）。"""
        canonical = label_policy_from_experiment_config(
            label_policy_version=self.evaluation.label_policy_version,
            embargo_sessions=self.evaluation.embargo_sessions,
        )
        horizon = self.evaluation.horizon_sessions
        if horizon is not None and horizon != canonical.horizon_sessions:
            raise ConfigurationError(
                f"experiment config horizon_sessions={horizon} differs from "
                f"{canonical.version}'s horizon_sessions={canonical.horizon_sessions}; "
                "horizon 的单一真源是 LabelPolicy（§28），必须升版本而不是在 config 里覆盖"
            )
        return canonical

    def benchmark_spec(self) -> ForecastBenchmarkSpec:
        return ForecastBenchmarkSpec(
            backends=self.benchmark.backends,
            origin_limit=self.benchmark.origin_limit,
            regime=self.regime,
        )

    def required_sessions(self) -> int:
        """时间轴必须拥有的 session 总数（lookback + 各段 + 段间 embargo）。"""
        policy = self.label_policy()
        return self.dataset.to_plan().required_sessions(
            embargo_sessions=policy.embargo_sessions,
            history_prefix=self.forecast.lookback_bars,
        )

    def hashing_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": self.version,
            "runtime": self.runtime.hashing_payload(),
            "data": self.data.hashing_payload(),
            "forecast": self.forecast.hashing_payload(),
            "knowledge_cutoff_policy": self.knowledge_cutoff_policy,
            "universe": self.universe.hashing_payload(),
            "evaluation": self.evaluation.hashing_payload(),
            "dataset": self.dataset.hashing_payload(),
            "benchmark": self.benchmark.hashing_payload(),
            "regime": self.regime.hashing_payload(),
            "gate": None if self.gate is None else self.gate.hashing_payload(),
        }
        assert set(payload) == set(type(self).model_fields), (
            "ExperimentConfig.hashing_payload must cover every model field"
        )
        return payload

    @property
    def config_hash(self) -> str:
        """config 的内容 hash（实时派生；config.yaml 本身随 run 归档）。"""
        return sha256_hex(self.hashing_payload())


def gate_criteria_path(config: ExperimentConfig, config_path: Path | str) -> Path | None:
    """``gate.criteria_file`` 解析成绝对路径（相对 config 所在目录）；未声明返回 ``None``。"""
    if config.gate is None:
        return None
    return (Path(config_path).resolve().parent / config.gate.criteria_file).resolve()


def load_experiment_config(path: Path | str) -> tuple[ExperimentConfig, str]:
    """读取 YAML experiment config，返回 ``(config, raw_text)``。

    ``raw_text`` 会随 run 原样归档（§32.1）；config_hash 由其**语义**派生，因此注释与
    键顺序的变化不会换 hash，而任何字段变化都会。

    声明了 ``gate`` 时，判据文档必须存在且合法——**在 run 开始之前**（§42 的预注册：
    缺失的判据不允许「先跑再补」）。
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigurationError(f"experiment config not found: {config_path}")
    raw_text = config_path.read_text(encoding="utf-8")
    try:
        payload = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"experiment config is not valid YAML: {exc}") from exc
    if payload is None:
        raise ConfigurationError(f"experiment config {config_path} is empty")
    if not isinstance(payload, dict):
        raise ConfigurationError(
            f"experiment config {config_path} must be a mapping, got {type(payload).__name__}"
        )
    _reject_secret_keys(payload)
    try:
        config = ExperimentConfig.model_validate(payload)
        # 触发 LabelPolicy 一致性校验：embargo / horizon 漂移在这里就被拒绝，
        # 而不是等到 run 跑完才发现 dataset_hash 无法解释。
        config.label_policy()
    except ConfigurationError:
        raise
    except Exception as exc:  # pydantic ValidationError 等
        raise ConfigurationError(f"invalid experiment config {config_path}: {exc}") from exc
    criteria_path = gate_criteria_path(config, config_path)
    if criteria_path is not None:
        # 预注册：判据必须在 run 前可读、自洽，且**指向本次 run 真的会跑的 backend 与段**。
        # 这两件事在这里校验一次；取对象由调用方（CLI）再调同一个 load_gate_criteria——
        # 两次读之间文件被替换时，归档与判决都用第二次读到的内容（即「校验的」可能是 A、
        # 「归档的」是 B，窗口只有毫秒级，见 ADR-024 §9 的 defer）。
        criteria, _raw = load_gate_criteria(criteria_path)
        unknown_backends = sorted(
            {criteria.candidate, *criteria.baselines} - set(config.benchmark.backends)
        )
        if unknown_backends:
            raise ConfigurationError(
                f"gate criteria {criteria_path} names backends {unknown_backends} that this "
                f"config's benchmark.backends {list(config.benchmark.backends)} do not include; "
                "judging a backend that will not run is not a comparison, and spending the run "
                "before noticing it is worse"
            )
        unknown_segments = sorted(
            set(criteria.evidence_segments) - {segment.name for segment in config.dataset.segments}
        )
        if unknown_segments:
            raise ConfigurationError(
                f"gate criteria {criteria_path} counts segments {unknown_segments} as evidence "
                f"but this config's dataset.segments are "
                f"{[segment.name for segment in config.dataset.segments]}; the judgement would "
                "silently rest on a different evidence slice than the pre-registered one"
            )
    return config, raw_text


__all__ = [
    "CUTOFF_POLICIES",
    "EXPERIMENT_CONFIG_VERSION",
    "BenchmarkSection",
    "DataSection",
    "DatasetSection",
    "EvaluationSection",
    "ExperimentConfig",
    "ForecastSection",
    "GateSection",
    "RuntimeSection",
    "SamplingSection",
    "SegmentSection",
    "UniverseSection",
    "gate_criteria_path",
    "load_experiment_config",
]
