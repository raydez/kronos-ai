"""Forecast 请求契约（基线文档 §8）。

- ForecastRequest 描述「对哪个 symbol、在哪个研究时点、预测未来多少个 market session」
- SamplingConfig 是采样参数唯一真源：seed 必须随 Request → Backend → Sampler →
  Result Metadata 全链路传递（§8/§9），不允许只存在于 Evaluation Run metadata
- lookback_bars 不属于 Request（属于 Forecast Backend / Model 配置，§8），
  但它决定输入窗口，必须进入 config_hash / input_data_hash，由 backend 侧承担

hashing_payload() 的构成属于契约：修改会改变 ForecastArtifactKey（§15）与
既有 artifact 的有效性，必须同步 golden 测试并升级 FORECAST_CONTRACT_VERSION。
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from kronos_ai.domain.symbols import validate_normalized_symbol
from kronos_ai.domain.time import ResearchTime, ensure_shanghai_aware

FORECAST_CONTRACT_VERSION = "forecast-contract-v2"

HASH_KIND_SAMPLING = "sampling_config"
HASH_KIND_REQUEST = "forecast_request"

# torch.Generator.manual_seed 接受 [-(2**63), 2**64-1]，内部把负 seed 重映射到
# 无符号区间（-1 与 2**64-1 等价）。本契约只接受 [0, 2**63-1]：拒绝负数避免
# 「同一采样流、两条不同 config」映射到不同 artifact key 的歧义，上界取 2**63-1
# 保证在 int64 范围内可跨语言/跨存储移植。
MAX_SEED = 2**63 - 1

_HEX256 = re.compile(r"[0-9a-f]{64}")


def _require_nonempty(value: str, field: str) -> str:
    """规范化 provenance 字符串：去除首尾空白，空串显式失败。

    不去空白会让 " cpu" 与 "cpu" 成为两条本应相同的 provenance 记录。
    """
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field} must be non-empty")
    return stripped


def _validate_sha256(value: str) -> str:
    if not _HEX256.fullmatch(value):
        raise ValueError("hash must be a 64-char lowercase sha256 hex digest")
    return value


Hash256 = Annotated[str, AfterValidator(_validate_sha256)]


def _reject_bool(value: object, field: str) -> None:
    """bool 是 int 子类，pydantic 宽松模式下 True 会静默变成 1。

    采样参数与 horizon 出现 bool 一律是调用方 bug，必须显式失败（§3.2）。
    """
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an int, not bool")


def _validate_finite(value: float, field: str) -> float:
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{field} must be finite")
    return value


class SamplingConfig(BaseModel):
    """采样参数（§8）；CPU / MPS / CUDA 同一 config 语义一致。

    seed 限定在 [0, 2**63-1]：torch.Generator.manual_seed 接受 int64，负 seed
    与同余取模后的正值不可区分，接受负数只会制造两个等价 config 产生不同
    artifact key 的歧义，故在契约层收窄。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = Field(ge=0, le=MAX_SEED)
    sample_count: int = Field(default=64, ge=1)
    temperature: float = Field(default=1.0, gt=0)
    top_k: int = Field(default=0, ge=0)
    top_p: float = Field(default=0.9, gt=0, le=1)

    @field_validator("seed", "sample_count", "top_k", mode="before")
    @classmethod
    def _int_fields_not_bool(cls, value: object, info: ValidationInfo) -> object:
        _reject_bool(value, str(info.field_name))
        return value

    @field_validator("temperature")
    @classmethod
    def _temperature_finite(cls, value: float) -> float:
        return _validate_finite(value, "temperature")

    def hashing_payload(self) -> dict[str, Any]:
        """进入 ForecastArtifactKey 的采样维（§15）；字段顺序由 canonical json 决定。

        kind/contract_version 参与哈希，使契约版本升级自动失效既有 artifact（§15）。
        """
        return {
            "kind": HASH_KIND_SAMPLING,
            "contract_version": FORECAST_CONTRACT_VERSION,
            "seed": self.seed,
            "sample_count": self.sample_count,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }


class ModelMetadata(BaseModel):
    """Forecast 结果的模型 provenance（§9/§14）。

    记录 torch version（编码进 runtime_version）、device、dtype，因为跨设备
    完全 bitwise reproducibility 不被承诺（§9）；device 取 device-class。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: str
    model_id: str
    revision: str
    runtime_version: str
    device: str
    dtype: str
    config_hash: Hash256

    @field_validator("backend", "model_id", "revision", "runtime_version", "device", "dtype")
    @classmethod
    def _nonempty(cls, value: str, info: ValidationInfo) -> str:
        return _require_nonempty(value, str(info.field_name))


class ForecastRequest(BaseModel):
    """Forecast 请求（§8）。

    时间语义不重复实现：由 ResearchTime 校验（+08:00 偏移、cutoff 落在 market_date 当日），
    本模型的 model_validator 直接构造 ResearchTime 并让其错误上抛，保证两处语义永不漂移。
    horizon 单位为 market session 数（非自然日），未来时间轴由 TradingCalendar 生成（§11）。
    lookback_bars 不属于 Request（backend 配置，§8）；本模型 extra="forbid"，误传会显式报错。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    market_date: date
    knowledge_cutoff: datetime
    horizon: int = Field(default=5, ge=1)
    sampling: SamplingConfig

    @field_validator("horizon", mode="before")
    @classmethod
    def _horizon_not_bool(cls, value: object) -> object:
        _reject_bool(value, "horizon")
        return value

    @field_validator("symbol")
    @classmethod
    def _symbol_normalized(cls, value: str) -> str:
        return validate_normalized_symbol(value)

    @model_validator(mode="after")
    def _research_time_consistent(self) -> ForecastRequest:
        ResearchTime(market_date=self.market_date, knowledge_cutoff=self.knowledge_cutoff)
        return self

    @property
    def research_time(self) -> ResearchTime:
        return ResearchTime(market_date=self.market_date, knowledge_cutoff=self.knowledge_cutoff)

    def hashing_payload(self) -> dict[str, Any]:
        """进入 ForecastArtifactKey 的请求维（§15）；kind/contract_version 同 SamplingConfig。"""
        return {
            "kind": HASH_KIND_REQUEST,
            "contract_version": FORECAST_CONTRACT_VERSION,
            "symbol": self.symbol,
            "market_date": self.market_date.isoformat(),
            "knowledge_cutoff": self.knowledge_cutoff.isoformat(),
            "horizon": self.horizon,
            "sampling": self.sampling.hashing_payload(),
        }


# ---------------------------------------------------------------------------
# Forecast Metrics registry（§12/§13）
#
# metric 名称不是自由字符串：必须来自本 registry。新增 / 修改指标数学定义属于
# registry 版本升级（FORECAST_METRIC_REGISTRY_VERSION），并同步 golden 测试与
# ForecastDistribution.metric_definition_version。
# ---------------------------------------------------------------------------

FORECAST_METRIC_REGISTRY_VERSION = "forecast-metrics-v1"

# 旧 artifact 必须仍可反序列化：升级数学定义时把旧版本号追加进来，
# 而不是删除（删除会让历史 ForecastDistribution 无法加载）。
SUPPORTED_FORECAST_METRIC_VERSIONS: tuple[str, ...] = (FORECAST_METRIC_REGISTRY_VERSION,)


class ForecastMetricDefinition(BaseModel):
    """版本化 metric registry 的一项；description 即数学定义（§13）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    unit: str
    description: str


_FORECAST_METRICS: dict[str, ForecastMetricDefinition] = {
    "horizon_return": ForecastMetricDefinition(
        name="horizon_return",
        unit="ratio",
        description="R_h = P_h / P_0 - 1，P_0 = forecast origin close，P_h = horizon last close",
    ),
    "log_return": ForecastMetricDefinition(
        name="log_return",
        unit="ratio",
        description="ln(P_h / P_0)，与 horizon_return 同源但用于加性聚合",
    ),
    "max_drawdown": ForecastMetricDefinition(
        name="max_drawdown",
        unit="ratio",
        description="close path（含 P_0）running max 后的 min(P_t / max_{s<=t} P_s - 1)，<= 0",
    ),
    "path_volatility": ForecastMetricDefinition(
        name="path_volatility",
        unit="ratio",
        description="close path（含 P_0）逐步 log return 的总体标准差（ddof=0）",
    ),
}

FORECAST_METRIC_NAMES: tuple[str, ...] = tuple(_FORECAST_METRICS)


def forecast_metric_definition(name: str) -> ForecastMetricDefinition:
    """按名取 registry 定义；未知名显式失败（§12 不允许 backend 自造 metric 名）。"""
    try:
        return _FORECAST_METRICS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown forecast metric {name!r}; allowed: {sorted(_FORECAST_METRICS)}"
        ) from exc


def _validate_forecast_metric(value: str) -> str:
    forecast_metric_definition(value)
    return value


def _validate_metric_version(value: str) -> str:
    value = _require_nonempty(value, "metric_definition_version")
    if value not in SUPPORTED_FORECAST_METRIC_VERSIONS:
        raise ValueError(
            f"unsupported metric_definition_version {value!r}; "
            f"known versions: {list(SUPPORTED_FORECAST_METRIC_VERSIONS)}"
        )
    return value


ThresholdOperator = Literal["gt", "gte", "lt", "lte"]

ForecastMetricName = Annotated[str, AfterValidator(_validate_forecast_metric)]


# ---------------------------------------------------------------------------
# §11 ForecastSample
# ---------------------------------------------------------------------------


class ForecastPoint(BaseModel):
    """forecast 未来某一 market session 的 OHLCV 一点（§11）。

    timestamp 语义 = 该未来 session 的收盘时刻（+08:00），由 TradingCalendar.next_sessions
    生成（个股停牌不改变 forecast 时间轴）。价格必须是有限实数；量纲与 MarketBar 一致：
    volume = 股，amount = 元。非正价格不在此层拒绝——分布层在计算收益指标时显式失败。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None
    amount: float | None = None

    @field_validator("timestamp")
    @classmethod
    def _timestamp_shanghai_aware(cls, value: datetime) -> datetime:
        return ensure_shanghai_aware(value, "timestamp")

    @field_validator("open", "high", "low", "close")
    @classmethod
    def _prices_finite(cls, value: float, info: ValidationInfo) -> float:
        return _validate_finite(value, str(info.field_name))

    @field_validator("volume", "amount")
    @classmethod
    def _flow_finite_nonnegative(cls, value: float | None, info: ValidationInfo) -> float | None:
        if value is None:
            return None
        _validate_finite(value, str(info.field_name))
        if value < 0:
            raise ValueError(f"{info.field_name} must be non-negative")
        return value


class ForecastSample(BaseModel):
    """单条随机路径（§11）；可独立持久化的研究 Artifact。

    points 用 tuple 而非 list：frozen 模型 + tuple 才能保证构造后不可变。
    sample_id 是 run 内 0-based 序号，排序后与 RawSampleSet 的行一一对应。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_id: int = Field(ge=0)
    points: tuple[ForecastPoint, ...]

    @field_validator("sample_id", mode="before")
    @classmethod
    def _sample_id_not_bool(cls, value: object) -> object:
        _reject_bool(value, "sample_id")
        return value

    @field_validator("points")
    @classmethod
    def _points_nonempty(cls, value: tuple[ForecastPoint, ...]) -> tuple[ForecastPoint, ...]:
        if not value:
            raise ValueError("points must contain at least one ForecastPoint")
        return value


# ---------------------------------------------------------------------------
# §12 ForecastDistribution
# ---------------------------------------------------------------------------


class ThresholdProbability(BaseModel):
    """P(metric <operator> threshold)，operator 显式携带以避免隐含方向（§12）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: ForecastMetricName
    operator: ThresholdOperator
    threshold: float
    probability: float = Field(ge=0, le=1)

    @field_validator("threshold")
    @classmethod
    def _threshold_finite(cls, value: float) -> float:
        return _validate_finite(value, "threshold")

    @field_validator("probability")
    @classmethod
    def _probability_finite(cls, value: float) -> float:
        return _validate_finite(value, "probability")


class QuantileValue(BaseModel):
    """metric 在 quantile 分位上的取值（§12）。quantile 限定 (0, 1) 开区间。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: ForecastMetricName
    quantile: float = Field(gt=0, lt=1)
    value: float

    @field_validator("value")
    @classmethod
    def _value_finite(cls, value: float) -> float:
        return _validate_finite(value, "value")


class ForecastDistribution(BaseModel):
    """跨 samples 的统计摘要（§12）；纯统计对象，provenance 由 ForecastResult 承担（§14）。

    不写死 2% 等阈值：阈值/分位由 DistributionSpec（forecast/distribution.py）显式给出，
    本模型只承载结果并强制 metric 名来自版本化 registry。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    horizon: int = Field(ge=1)
    sample_count: int = Field(ge=1)

    expected_return: float
    median_return: float

    threshold_probabilities: tuple[ThresholdProbability, ...]
    quantiles: tuple[QuantileValue, ...]

    forecast_dispersion: float
    expected_max_drawdown: float
    expected_path_volatility: float

    distribution_spec_version: str
    distribution_spec_hash: Hash256
    metric_definition_version: str

    @field_validator("horizon", "sample_count", mode="before")
    @classmethod
    def _counts_not_bool(cls, value: object, info: ValidationInfo) -> object:
        _reject_bool(value, str(info.field_name))
        return value

    @field_validator(
        "expected_return",
        "median_return",
        "forecast_dispersion",
        "expected_max_drawdown",
        "expected_path_volatility",
    )
    @classmethod
    def _scalars_finite(cls, value: float, info: ValidationInfo) -> float:
        return _validate_finite(value, str(info.field_name))

    @field_validator("distribution_spec_version")
    @classmethod
    def _spec_version_nonempty(cls, value: str) -> str:
        return _require_nonempty(value, "distribution_spec_version")

    @field_validator("metric_definition_version")
    @classmethod
    def _metric_version_known(cls, value: str) -> str:
        return _validate_metric_version(value)

    @model_validator(mode="after")
    def _no_duplicate_entries(self) -> ForecastDistribution:
        threshold_keys = [
            (t.metric, t.operator, t.threshold) for t in self.threshold_probabilities
        ]
        if len(set(threshold_keys)) != len(threshold_keys):
            raise ValueError("threshold_probabilities contains duplicate (metric, operator, threshold)")
        quantile_keys = [(q.metric, q.quantile) for q in self.quantiles]
        if len(set(quantile_keys)) != len(quantile_keys):
            raise ValueError("quantiles contains duplicate (metric, quantile)")
        return self


# ---------------------------------------------------------------------------
# §14 ForecastResult 与 Provenance
# ---------------------------------------------------------------------------


class SamplingMetadata(BaseModel):
    """实际生效的采样参数（§14）；与 SamplingConfig 分开是因为前者是契约真源，
    后者是「run 实际使用了什么」的记录，二者进入 ForecastResult 供重放审计。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = Field(ge=0, le=MAX_SEED)
    sample_count: int = Field(ge=1)
    temperature: float = Field(gt=0)
    top_k: int = Field(ge=0)
    top_p: float = Field(gt=0, le=1)

    @field_validator("seed", "sample_count", "top_k", mode="before")
    @classmethod
    def _int_fields_not_bool(cls, value: object, info: ValidationInfo) -> object:
        _reject_bool(value, str(info.field_name))
        return value

    @classmethod
    def from_config(cls, config: SamplingConfig) -> SamplingMetadata:
        """SamplingConfig → 结果元数据；保证 seed 等字段单点流转，不靠调用方拼装。"""
        return cls(
            seed=config.seed,
            sample_count=config.sample_count,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
        )


class ForecastResult(BaseModel):
    """一次 forecast 的完整产物 + provenance（§14）。

    契约约束：samples 条数 == distribution.sample_count，每条 points 长度 == distribution.horizon；
    input_data_hash 绑定输入快照，artifact_id 绑定缓存键（§15，由 cache 层填充）。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str

    market_date: date
    knowledge_cutoff: datetime

    samples: tuple[ForecastSample, ...]
    distribution: ForecastDistribution

    model: ModelMetadata
    sampling: SamplingMetadata

    input_data_hash: Hash256
    artifact_id: str

    @field_validator("symbol")
    @classmethod
    def _symbol_normalized(cls, value: str) -> str:
        return validate_normalized_symbol(value)

    @field_validator("artifact_id")
    @classmethod
    def _artifact_id_nonempty(cls, value: str) -> str:
        return _require_nonempty(value, "artifact_id")

    @model_validator(mode="after")
    def _consistent(self) -> ForecastResult:
        ResearchTime(market_date=self.market_date, knowledge_cutoff=self.knowledge_cutoff)
        if self.sampling.sample_count != self.distribution.sample_count:
            raise ValueError(
                f"sampling.sample_count {self.sampling.sample_count} != "
                f"distribution.sample_count {self.distribution.sample_count}; "
                "the two provenance records describe the same run and must agree"
            )
        if len(self.samples) != self.distribution.sample_count:
            raise ValueError(
                f"samples length {len(self.samples)} != distribution.sample_count "
                f"{self.distribution.sample_count}"
            )
        for sample in self.samples:
            if len(sample.points) != self.distribution.horizon:
                raise ValueError(
                    f"sample {sample.sample_id} has {len(sample.points)} points != "
                    f"distribution.horizon {self.distribution.horizon}"
                )
        return self
