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
from typing import Annotated, Any

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
from kronos_ai.domain.time import ResearchTime

FORECAST_CONTRACT_VERSION = "forecast-contract-v1"

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
