"""ForecastService：请求构造、provider 调用、backend 委派与显式校验（RX-KAI-014）。

不加载任何模型：backend 用 stub 替换，只验证编排契约。
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, cast

import pytest

from kronos_ai.data.calendar import StaticTradingCalendar
from kronos_ai.domain.forecast import (
    ForecastRequest,
    ForecastResult,
    SamplingConfig,
)
from kronos_ai.domain.market import MarketHistory
from kronos_ai.domain.time import CN_TZ
from kronos_ai.errors import ConfigurationError
from kronos_ai.forecast.base import ForecastBackend
from kronos_ai.forecast.service import ForecastService

MARKET_DATE = date(2026, 9, 25)
CUTOFF = datetime(2026, 9, 25, 18, 0, tzinfo=CN_TZ)
SAMPLING = SamplingConfig(seed=11, sample_count=4)


class StubProvider:
    def __init__(self, history: MarketHistory) -> None:
        self._history = history
        self.calls: list[tuple[str, date, datetime, int]] = []
        self.closed = 0

    def get_history(
        self, symbol: str, market_date: date, knowledge_cutoff: datetime, lookback_bars: int
    ) -> MarketHistory:
        self.calls.append((symbol, market_date, knowledge_cutoff, lookback_bars))
        return self._history

    def close(self) -> None:
        self.closed += 1


class StubBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[MarketHistory, ForecastRequest, bool]] = []

    @property
    def name(self) -> str:
        return "stub"

    def forecast(
        self, history: MarketHistory, request: ForecastRequest, *, force: bool = False
    ) -> ForecastResult:
        self.calls.append((history, request, force))
        return cast(ForecastResult, object())


@pytest.fixture
def history(make_history: Any) -> MarketHistory:
    return cast(MarketHistory, make_history(n_bars=16))


def test_service_orchestrates_provider_and_backend(history: MarketHistory) -> None:
    provider = StubProvider(history)
    backend = StubBackend()
    service = ForecastService(provider, backend, lookback_bars=16)

    service.run(
        symbol="600000",
        market_date=MARKET_DATE,
        knowledge_cutoff=CUTOFF,
        sampling=SAMPLING,
        horizon=3,
    )

    assert provider.calls == [("600000", MARKET_DATE, CUTOFF, 16)]
    assert len(backend.calls) == 1
    called_history, request, force = backend.calls[0]
    assert called_history is history
    assert request.horizon == 3
    assert request.sampling == SAMPLING
    assert force is False


def test_service_forwards_force(history: MarketHistory) -> None:
    backend = StubBackend()
    service = ForecastService(StubProvider(history), backend, lookback_bars=8)
    service.run(
        symbol="600000",
        market_date=MARKET_DATE,
        knowledge_cutoff=CUTOFF,
        sampling=SAMPLING,
        force=True,
    )
    assert backend.calls[0][2] is True


def test_request_is_consistent_with_provider_call(history: MarketHistory) -> None:
    provider = StubProvider(history)
    backend = StubBackend()
    ForecastService(provider, backend, lookback_bars=16).run(
        symbol="600000",
        market_date=MARKET_DATE,
        knowledge_cutoff=CUTOFF,
        sampling=SAMPLING,
    )
    symbol, market_date, cutoff, _ = provider.calls[0]
    request = backend.calls[0][1]
    assert (symbol, market_date, cutoff) == (
        request.symbol,
        request.market_date,
        request.knowledge_cutoff,
    )


@pytest.mark.parametrize("lookback", [0, -1])
def test_invalid_lookback_rejected(history: MarketHistory, lookback: int) -> None:
    with pytest.raises(ConfigurationError, match="lookback_bars"):
        ForecastService(StubProvider(history), StubBackend(), lookback_bars=lookback)


def test_close_releases_provider(history: MarketHistory) -> None:
    provider = StubProvider(history)
    service = ForecastService(provider, StubBackend(), lookback_bars=16)
    service.close()
    assert provider.closed == 1


def test_close_is_noop_without_provider_close(history: MarketHistory) -> None:
    class BareProvider:
        def get_history(
            self, symbol: str, market_date: date, knowledge_cutoff: datetime, lookback_bars: int
        ) -> MarketHistory:
            return history

    # 无状态 provider 没有 close：close() 不下发、不报错
    ForecastService(BareProvider(), StubBackend(), lookback_bars=16).close()


def test_stub_backend_satisfies_protocol() -> None:
    assert isinstance(StubBackend(), ForecastBackend)


def test_static_calendar_does_not_satisfy_backend_protocol() -> None:
    calendar = StaticTradingCalendar(exchange="SSE", source="t", sessions=(MARKET_DATE,))
    assert not isinstance(calendar, ForecastBackend)
