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

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kronos_ai.domain.symbols import validate_normalized_symbol
from kronos_ai.domain.time import ResearchTime, ensure_shanghai_aware

FORECAST_CONTRACT_VERSION = "forecast-contract-v1"

# torch.Generator.manual_seed 接受 int64；限定非负可移植范围，负 seed 无额外语义
MAX_SEED = 2**63 - 1


class SamplingConfig(BaseModel):
    """采样参数（§8）；CPU / MPS / CUDA 同一 config 语义一致。"""

    model_config = ConfigDict(frozen=True)

    seed: int = Field(ge=0, le=MAX_SEED)
    sample_count: int = Field(default=64, ge=1)
    temperature: float = Field(default=1.0, gt=0)
    top_k: int = Field(default=0, ge=0)
    top_p: float = Field(default=0.9, gt=0, le=1)

    @field_validator("temperature")
    @classmethod
    def _temperature_finite(cls, value: float) -> float:
        if value in (float("inf"), float("-inf")):
            raise ValueError("temperature must be finite")
        return value

    def hashing_payload(self) -> dict[str, Any]:
        """进入 ForecastArtifactKey 的采样维（§15）；字段顺序由 canonical json 决定。"""
        return {
            "seed": self.seed,
            "sample_count": self.sample_count,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }


class ForecastRequest(BaseModel):
    """Forecast 请求（§8）。

    时间语义与 ResearchTime 一致：knowledge_cutoff 必须 +08:00 且落在 market_date 当日。
    horizon 单位为 market session 数（非自然日），未来时间轴由 TradingCalendar 生成（§11）。
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    market_date: date
    knowledge_cutoff: datetime
    horizon: int = Field(default=5, ge=1)
    sampling: SamplingConfig

    @field_validator("symbol")
    @classmethod
    def _symbol_normalized(cls, value: str) -> str:
        return validate_normalized_symbol(value)

    @field_validator("knowledge_cutoff")
    @classmethod
    def _cutoff_shanghai_aware(cls, value: datetime) -> datetime:
        return ensure_shanghai_aware(value, "knowledge_cutoff")

    @model_validator(mode="after")
    def _cutoff_on_market_date(self) -> ForecastRequest:
        if self.knowledge_cutoff.date() != self.market_date:
            raise ValueError(
                "knowledge_cutoff must fall on market_date "
                f"(market_date={self.market_date}, cutoff={self.knowledge_cutoff})"
            )
        return self

    @property
    def research_time(self) -> ResearchTime:
        return ResearchTime(market_date=self.market_date, knowledge_cutoff=self.knowledge_cutoff)

    def hashing_payload(self) -> dict[str, Any]:
        """进入 ForecastArtifactKey 的请求维（§15）。"""
        return {
            "symbol": self.symbol,
            "market_date": self.market_date.isoformat(),
            "knowledge_cutoff": self.knowledge_cutoff.isoformat(),
            "horizon": self.horizon,
            "sampling": self.sampling.hashing_payload(),
        }
