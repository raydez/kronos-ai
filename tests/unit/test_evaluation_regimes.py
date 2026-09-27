"""Regime 分类单元测试（RX-KAI-019，基线文档 §49 / §29）。

覆盖：

1. **口径正确**：趋势（BULL / BEAR / SIDEWAYS）与波动（HIGH_VOL / LOW_VOL）的边界行为，
   含「恰好等于阈值」的归属（`>` 判据，阈值点归入较温和的一侧）；
2. **point-in-time 结构性保证**：分类函数只接收 MarketHistory，没有「未来 bar」参数，
   因此未来信息在类型层进不来；并且它读取的只是已截断的 bars；
3. **显式失败**：bar 数不足、非正价格；
4. **版本治理**：hashing_payload 覆盖全部字段、hash 对每个字段敏感。
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import pytest
from pydantic import ValidationError

from kronos_ai.domain.hashing import is_sha256_hex
from kronos_ai.domain.market import MarketBar, MarketHistory
from kronos_ai.domain.time import CN_TZ, MARKET_SESSION_CLOSE
from kronos_ai.errors import InsufficientHistoryError
from kronos_ai.evaluation.regimes import (
    DEFAULT_REGIME_SPEC,
    REGIME_SPEC_VERSION,
    RegimeSpec,
    classify_regime,
    regime_spec_hash,
)

SYMBOL = "600000"
MARKET_DATE = date(2026, 9, 25)


def _sessions(count: int) -> tuple[date, ...]:
    days: list[date] = []
    current = MARKET_DATE - timedelta(days=count * 2 + 10)
    while current <= MARKET_DATE:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return tuple(days[-count:])


def history(closes: list[float], *, cutoff_hour: int = 18) -> MarketHistory:
    sessions = _sessions(len(closes))
    bars = tuple(
        MarketBar(
            symbol=SYMBOL,
            timestamp=datetime.combine(day, MARKET_SESSION_CLOSE, tzinfo=CN_TZ),
            open=close,
            high=close * 1.01,
            low=close * 0.99,
            close=close,
            trade_status="1",
            adjustment_mode="raw",
            available_at=datetime.combine(day, MARKET_SESSION_CLOSE, tzinfo=CN_TZ),
        )
        for day, close in zip(sessions, closes, strict=True)
    )
    return MarketHistory(
        symbol=SYMBOL,
        market_date=MARKET_DATE,
        knowledge_cutoff=datetime(2026, 9, 25, cutoff_hour, 0, tzinfo=CN_TZ),
        bars=bars,
        provider="test-fixture",
        dataset_version="test-dataset-v1",
    )


class TestTrendAxis:
    def test_rising_series_is_bull(self) -> None:
        # 21 根 bar 才能满足默认 20 窗口的趋势收益（closes[-1] / closes[-20]）
        closes = [10.0 * (1.01**i) for i in range(21)]
        result = classify_regime(history(closes))
        assert result.trend == "BULL"
        assert result.trend_return > DEFAULT_REGIME_SPEC.trend_threshold

    def test_falling_series_is_bear(self) -> None:
        closes = [10.0 * (0.99**i) for i in range(21)]
        result = classify_regime(history(closes))
        assert result.trend == "BEAR"
        assert result.trend_return < -DEFAULT_REGIME_SPEC.trend_threshold

    def test_flat_series_is_sideways(self) -> None:
        closes = [10.0] * 21
        result = classify_regime(history(closes))
        assert result.trend == "SIDEWAYS"
        assert result.trend_return == pytest.approx(0.0)

    def test_exact_threshold_belongs_to_sideways(self) -> None:
        """阈值点用 ``>`` 判据：恰好等于阈值不算趋势（口径必须唯一确定）。

        阈值取「实测 log 收益」本身，避免用 `exp/log` 往返构造出 5.000000000000002e-2
        这类浮点尾差，把「判据是 > 还是 >=」测成浮点误差测试。
        """
        closes = [10.0] * 19 + [10.0, 10.0 * math.exp(0.05)]
        ratio = math.log(closes[-1] / closes[-2])
        spec = RegimeSpec(trend_window_sessions=2, trend_threshold=ratio)
        assert classify_regime(history(closes), spec=spec).trend == "SIDEWAYS"
        # 阈值略低于实测收益时必须翻成 BULL：证明上一条不是因为「判据永远返回 SIDEWAYS」
        tighter = RegimeSpec(trend_window_sessions=2, trend_threshold=ratio - 1e-12)
        assert classify_regime(history(closes), spec=tighter).trend == "BULL"


class TestVolatilityAxis:
    def test_high_volatility_series(self) -> None:
        closes = [10.0]
        for index in range(20):
            closes.append(closes[-1] * (1.05 if index % 2 == 0 else 0.95))
        result = classify_regime(history(closes))
        assert result.volatility == "HIGH_VOL"

    def test_low_volatility_series(self) -> None:
        closes = [10.0 * (1.0001**i) for i in range(21)]
        result = classify_regime(history(closes))
        assert result.volatility == "LOW_VOL"
        assert result.realized_volatility < DEFAULT_REGIME_SPEC.volatility_threshold


class TestExplicitFailure:
    def test_insufficient_bars(self) -> None:
        with pytest.raises(InsufficientHistoryError):
            classify_regime(history([10.0] * 5))

    def test_non_positive_close(self) -> None:
        """非正 close 不经 MarketBar 校验时（model_construct 绕过）也必须显式失败。

        正常构造路径上 MarketBar 已经拒绝非正价格，因此这里刻意绕过校验，验证分类器
        自身不依赖上游校验（ADR-010 的 defense-in-depth）。
        """
        valid = history([10.0] * 21)
        forged_bar = valid.bars[-1].model_copy(update={"close": -1.0})
        forged = MarketHistory.model_construct(
            symbol=valid.symbol,
            market_date=valid.market_date,
            knowledge_cutoff=valid.knowledge_cutoff,
            bars=(*valid.bars[:-1], forged_bar),
            provider=valid.provider,
            dataset_version=valid.dataset_version,
        )
        with pytest.raises(InsufficientHistoryError):
            classify_regime(forged)

    def test_spec_requires_positive_thresholds(self) -> None:
        with pytest.raises(ValidationError):
            RegimeSpec(trend_threshold=0.0)


class TestPointInTime:
    def test_classification_has_no_future_parameter(self) -> None:
        """结构性保证：分类只接受 history，不存在「未来 bar」入参（§29）。"""
        import inspect

        signature = inspect.signature(classify_regime)
        assert set(signature.parameters) == {"history", "spec"}

    def test_only_uses_supplied_bars(self) -> None:
        """同一根 bar 序列在「不同 cutoff」下给出同一分类（cutoff 不是输入）。"""
        closes = [10.0 * (1.01**i) for i in range(21)]
        early = classify_regime(history(closes, cutoff_hour=15))
        late = classify_regime(history(closes, cutoff_hour=18))
        assert early == late


class TestVersioning:
    def test_hashing_payload_covers_all_fields(self) -> None:
        payload = DEFAULT_REGIME_SPEC.hashing_payload()
        assert set(payload) == set(RegimeSpec.model_fields)
        assert payload["version"] == REGIME_SPEC_VERSION

    def test_hash_is_sha256_and_sensitive(self) -> None:
        base = regime_spec_hash(DEFAULT_REGIME_SPEC)
        assert is_sha256_hex(base)
        variants = [
            RegimeSpec(trend_window_sessions=21),
            RegimeSpec(volatility_window_sessions=21),
            RegimeSpec(trend_threshold=0.03),
            RegimeSpec(volatility_threshold=0.02),
            RegimeSpec(version="regime-spec-v2"),
        ]
        for variant in variants:
            assert regime_spec_hash(variant) != base

    def test_hash_is_stable(self) -> None:
        assert regime_spec_hash(RegimeSpec()) == regime_spec_hash(RegimeSpec())


def test_default_spec_windows_are_satisfiable() -> None:
    """默认 spec 的 required_bars 必须与两个窗口一致（否则默认值自相矛盾）。"""
    assert DEFAULT_REGIME_SPEC.required_bars == max(
        DEFAULT_REGIME_SPEC.trend_window_sessions, DEFAULT_REGIME_SPEC.volatility_window_sessions
    )
