"""Market Contract（基线文档 §7）。

时间戳语义（versioned，DoD 30）：
- MarketBar.timestamp = 该 bar 所属交易 session 的收盘时刻（15:00 Asia/Shanghai）。
  统一取收盘时刻，使 available_at <= knowledge_cutoff 拥有确定比较基准。
- MarketBar.available_at = 数据源实际可提供该 bar 的最早时间，必须 >= timestamp。
- 单位约定：volume = 股，amount = 元（Provider 层负责换算）。

不变式（frozen 模型 + tuple bars，构造后不可变）：
- 所有校验在构造时执行一次；data_hash 由内容自动计算（也可显式传入以供校验）。
- MarketHistory 是 point-in-time 快照：任何 bar 的 available_at 不得超过
  knowledge_cutoff，任何 bar 的交易日不得超过 market_date（§29 Leakage Guard）。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from typing import Annotated, Any

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    ValidationInfo,
    computed_field,
    field_validator,
    model_validator,
)

from kronos_ai.domain.hashing import sha256_hex
from kronos_ai.domain.symbols import validate_normalized_symbol
from kronos_ai.domain.time import MARKET_SESSION_CLOSE, ensure_shanghai_aware

MARKET_CONTRACT_VERSION = "market-contract-v1"


def _validate_shanghai(value: datetime) -> datetime:
    return ensure_shanghai_aware(value, "datetime")


ShanghaiDatetime = Annotated[datetime, AfterValidator(_validate_shanghai)]


def _check_finite(value: float, field: str) -> float:
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{field} must be finite")
    return value


class MarketBar(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    timestamp: ShanghaiDatetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None
    amount: float | None = None
    trade_status: str | None = None
    adjustment_mode: str
    available_at: ShanghaiDatetime

    @field_validator("symbol")
    @classmethod
    def _symbol_normalized(cls, value: str) -> str:
        return validate_normalized_symbol(value)

    @field_validator("timestamp")
    @classmethod
    def _timestamp_is_session_close(cls, value: datetime) -> datetime:
        if value.time() != MARKET_SESSION_CLOSE:
            raise ValueError(f"timestamp must be the session close 15:00+08:00, got {value.time()}")
        return value

    @field_validator("open", "high", "low", "close")
    @classmethod
    def _price_positive_finite(cls, value: float, info: ValidationInfo) -> float:
        _check_finite(value, str(info.field_name))
        if value <= 0:
            raise ValueError(f"{info.field_name} must be > 0")
        return value

    @field_validator("volume", "amount")
    @classmethod
    def _quantity_nonnegative(cls, value: float | None, info: ValidationInfo) -> float | None:
        if value is None:
            return None
        _check_finite(value, str(info.field_name))
        if value < 0:
            raise ValueError(f"{info.field_name} must be >= 0")
        return value

    @field_validator("adjustment_mode")
    @classmethod
    def _adjustment_mode_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("adjustment_mode must be non-empty")
        return value

    @model_validator(mode="after")
    def _bar_invariants(self) -> MarketBar:
        if self.high < max(self.open, self.close):
            raise ValueError("high must be >= max(open, close)")
        if self.low > min(self.open, self.close):
            raise ValueError("low must be <= min(open, close)")
        if self.available_at < self.timestamp:
            raise ValueError("available_at must be >= timestamp")
        return self

    def hashing_payload(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "amount": self.amount,
            "trade_status": self.trade_status,
            "adjustment_mode": self.adjustment_mode,
            "available_at": self.available_at.isoformat(),
        }


class MarketHistory(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    market_date: date
    knowledge_cutoff: ShanghaiDatetime
    bars: tuple[MarketBar, ...]
    provider: str
    dataset_version: str

    @field_validator("symbol")
    @classmethod
    def _symbol_normalized(cls, value: str) -> str:
        return validate_normalized_symbol(value)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def data_hash(self) -> str:
        """内容 hash：读取时实时派生，运行期任何内容变化都必然反映在值中。"""
        return compute_market_history_hash(
            symbol=self.symbol,
            market_date=self.market_date,
            knowledge_cutoff=self.knowledge_cutoff,
            provider=self.provider,
            dataset_version=self.dataset_version,
            bars=self.bars,
        )

    @model_validator(mode="after")
    def _history_invariants(self) -> MarketHistory:
        if not self.bars:
            raise ValueError("bars must not be empty")

        timestamps = [bar.timestamp for bar in self.bars]
        if timestamps != sorted(timestamps):
            raise ValueError("bars must be sorted ascending by timestamp")
        if len(set(timestamps)) != len(timestamps):
            raise ValueError("bars must have unique timestamps")

        for bar in self.bars:
            if bar.symbol != self.symbol:
                raise ValueError(f"bar symbol {bar.symbol!r} != history symbol {self.symbol!r}")
            if bar.available_at > self.knowledge_cutoff:
                raise ValueError(
                    "leakage guard: bar available_at "
                    f"{bar.available_at.isoformat()} > knowledge_cutoff "
                    f"{self.knowledge_cutoff.isoformat()}"
                )
            if bar.timestamp.date() > self.market_date:
                raise ValueError(
                    f"leakage guard: bar timestamp {bar.timestamp.date()} "
                    f"is after market_date {self.market_date}"
                )
        return self


def compute_market_history_hash(
    *,
    symbol: str,
    market_date: date,
    knowledge_cutoff: datetime,
    provider: str,
    dataset_version: str,
    bars: Sequence[MarketBar],
) -> str:
    """MarketHistory.data_hash 的唯一定义；所有参与消费的字段都进入 payload。

    golden 测试锚定本函数的输出（tests/unit/test_market.py），
    修改 payload 构成会使既有 artifact 的 hash 失效，属于契约变更。
    """
    payload: dict[str, Any] = {
        "kind": "market_history",
        "contract_version": MARKET_CONTRACT_VERSION,
        "symbol": symbol,
        "market_date": market_date.isoformat(),
        "knowledge_cutoff": knowledge_cutoff.isoformat(),
        "provider": provider,
        "dataset_version": dataset_version,
        "bars": [bar.hashing_payload() for bar in bars],
    }
    return sha256_hex(payload)
