"""Point-in-time regime 分类（基线文档 §49 Robustness、§29、§48）。

§49 要求 benchmark 结论按 regime 分组给出（Bull / Bear / Sideways / High Vol / Low Vol），
否则「模型在哪些市场状态下有效」无法回答——一个整体指标可能由单一 regime 主导。

本模块的**唯一**设计约束是「只能用 origin 当时已知的信息」：分类输入是
:class:`~kronos_ai.domain.market.MarketHistory` 的 bars（provider 已按 knowledge_cutoff
截断，bar.available_at 全部 <= cutoff），因此 regime 不可能是未来信息的函数。这条约束
由构造方式保证，而不是靠调用方自觉：:func:`classify_regime` 只接受 MarketHistory，
拿不到「未来 bar」这种参数。

口径是版本化的（:data:`REGIME_SPEC_VERSION`）：趋势窗口、波动窗口、阈值改变都会改变
分组结果，从而改变每一个 regime 分组里的指标口径，必须随报告一起被指认。

两个轴彼此独立（§49 的五个名字其实是两个轴）:

```text
trend      BULL | SIDEWAYS | BEAR     累计 log 收益 vs ±threshold
volatility HIGH_VOL | LOW_VOL         已实现波动 vs threshold
```

分轴而不是合成一个五值枚举，是因为「高波动 + 上涨」与「低波动 + 上涨」在证据上应当
可区分；合成单一枚举会让其中一个维度在报表里被平均掉。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kronos_ai.domain.hashing import sha256_hex
from kronos_ai.domain.market import MarketHistory
from kronos_ai.errors import InsufficientHistoryError

REGIME_SPEC_VERSION = "regime-spec-v1"

TrendRegime = Literal["BULL", "SIDEWAYS", "BEAR"]
VolatilityRegime = Literal["HIGH_VOL", "LOW_VOL"]

TREND_REGIMES: tuple[TrendRegime, ...] = ("BEAR", "SIDEWAYS", "BULL")
VOLATILITY_REGIMES: tuple[VolatilityRegime, ...] = ("LOW_VOL", "HIGH_VOL")


class RegimeSpec(BaseModel):
    """regime 分类的版本化配置（§49）。

    阈值是**收益空间**的：``trend_threshold`` 是「累计 log 收益绝对值超过它才算趋势」的
    门槛，``volatility_threshold`` 是「逐步 log 收益总体标准差超过它即高波动」的门槛。
    两者都是无量纲比率，因此对价格水平不敏感（不因股价从 5 元涨到 50 元而改变分组）。

    默认值不是「最优参数」，而是**声明式的粗分类**：它们的作用是把证据分成可比较的桶，
    而不是充当交易信号。任何改动都必须升版本，因为它会改变历史报告的分组含义。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = REGIME_SPEC_VERSION
    trend_window_sessions: int = Field(default=20, ge=2)
    volatility_window_sessions: int = Field(default=20, ge=2)
    trend_threshold: float = Field(default=0.02, gt=0)
    volatility_threshold: float = Field(default=0.015, gt=0)

    @field_validator("version")
    @classmethod
    def _version_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("version must be non-empty")
        return value

    @field_validator("trend_threshold", "volatility_threshold")
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError(f"threshold must be finite, got {value!r}")
        return value

    @model_validator(mode="after")
    def _windows_are_sane(self) -> RegimeSpec:
        if self.trend_window_sessions < 2:
            raise ValueError("trend_window_sessions must be >= 2 to compute a return")
        if self.volatility_window_sessions < 2:
            raise ValueError("volatility_window_sessions must be >= 2 to compute a std")
        return self

    @property
    def required_bars(self) -> int:
        """分类一条 history 所需的最少 bar 数（两个窗口都要能被填满）。"""
        return max(self.trend_window_sessions, self.volatility_window_sessions)

    def hashing_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": self.version,
            "trend_window_sessions": self.trend_window_sessions,
            "volatility_window_sessions": self.volatility_window_sessions,
            "trend_threshold": self.trend_threshold,
            "volatility_threshold": self.volatility_threshold,
        }
        assert set(payload) == set(type(self).model_fields), (
            "RegimeSpec.hashing_payload must cover every model field"
        )
        return payload


DEFAULT_REGIME_SPEC = RegimeSpec()


def regime_spec_hash(spec: RegimeSpec = DEFAULT_REGIME_SPEC) -> str:
    """RegimeSpec 的确定性 hash；进 benchmark 报告，使分组口径可被指认。"""
    return sha256_hex({"kind": "regime_spec", "spec": spec.hashing_payload()})


class RegimeClassification(BaseModel):
    """一个 origin 的 regime 标签（两个轴）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trend: TrendRegime
    volatility: VolatilityRegime
    trend_return: float
    realized_volatility: float
    spec_version: str = REGIME_SPEC_VERSION


def _log_returns(closes: Sequence[float], window: int) -> list[float]:
    """末 ``window`` 根 close 的逐步 log 收益（长度 ``window - 1``）。"""
    tail = closes[-window:]
    returns: list[float] = []
    for previous, current in pairwise(tail):
        if previous <= 0 or current <= 0:
            raise InsufficientHistoryError(
                "regime classification requires positive closes to take log returns; "
                f"got {previous!r} -> {current!r}"
            )
        returns.append(math.log(current / previous))
    return returns


def classify_regime(
    history: MarketHistory,
    *,
    spec: RegimeSpec = DEFAULT_REGIME_SPEC,
) -> RegimeClassification:
    """把一条 point-in-time history 分类成 ``(trend, volatility)``。

    ``history`` 由 provider 按 knowledge_cutoff 截断，因此本函数天然只能看到 origin 当时
    已发布的信息（§29）；它没有接收「未来 bar」的参数，未来信息在类型层就进不来。

    bar 数不足 ``spec.required_bars`` 时抛 :class:`InsufficientHistoryError`：静默用更短
    的窗口会让同一个 regime 标签在不同 origin 上对应不同口径。
    """
    closes = [bar.close for bar in history.bars]
    if len(closes) < spec.required_bars:
        raise InsufficientHistoryError(
            f"regime classification needs at least {spec.required_bars} bars "
            f"(trend_window={spec.trend_window_sessions}, "
            f"volatility_window={spec.volatility_window_sessions}), got {len(closes)} "
            f"for {history.symbol}"
        )
    if any(close <= 0 for close in closes):
        raise InsufficientHistoryError(
            f"regime classification requires positive closes for {history.symbol}"
        )
    trend_return = math.log(closes[-1] / closes[-spec.trend_window_sessions])
    if trend_return > spec.trend_threshold:
        trend: TrendRegime = "BULL"
    elif trend_return < -spec.trend_threshold:
        trend = "BEAR"
    else:
        trend = "SIDEWAYS"

    returns = _log_returns(closes, spec.volatility_window_sessions)
    mean = math.fsum(returns) / len(returns)
    variance = math.fsum((value - mean) ** 2 for value in returns) / len(returns)
    volatility = math.sqrt(variance)
    volatility_regime: VolatilityRegime = (
        "HIGH_VOL" if volatility > spec.volatility_threshold else "LOW_VOL"
    )

    return RegimeClassification(
        trend=trend,
        volatility=volatility_regime,
        trend_return=trend_return,
        realized_volatility=volatility,
        spec_version=spec.version,
    )


__all__ = [
    "DEFAULT_REGIME_SPEC",
    "REGIME_SPEC_VERSION",
    "TREND_REGIMES",
    "VOLATILITY_REGIMES",
    "RegimeClassification",
    "RegimeSpec",
    "TrendRegime",
    "VolatilityRegime",
    "classify_regime",
    "regime_spec_hash",
]
