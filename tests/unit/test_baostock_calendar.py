"""BaoStockTradingCalendarLoader：把 query_trade_dates 如实搬成 TradingCalendar（RX-KAI-014）。

覆盖点（对应 ADR-009 与 spike §1 的已知边界）：

- 只保留 ``is_trading_day == "1"`` 的日期，顺序稳定、去重；
- 超出已发布覆盖时服务端返回 0 行 → 显式 ``ProviderError``，绝不返回空日历；
- ``is_trading_day`` 非 0/1、行字段缺失、重复日期 → ``DataQualityError``；
- 装载结果满足 :class:`TradingCalendar` 契约（is_session / next_sessions / 覆盖不足抛错）。
"""

from __future__ import annotations

from datetime import date
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from kronos_ai.data.calendar import TradingCalendar
from kronos_ai.errors import CalendarError, ConfigurationError, DataQualityError, ProviderError
from kronos_ai.infrastructure.providers.baostock import BaoStockTradingCalendarLoader

FIELDS = ["calendar_date", "is_trading_day"]
PAGE_ROWS = 2000


class FakeTradeDateResult:
    """最小 ResultData：fields + next/get_row_data 游标（语义同 0.9.4）。"""

    def __init__(
        self,
        rows: list[list[str]],
        *,
        error_code: str = "0",
        error_msg: str = "",
    ) -> None:
        self.fields = list(FIELDS)
        self._rows = rows
        self.error_code = error_code
        self.error_msg = error_msg
        self._cursor = 0

    @property
    def data(self) -> list[list[str]]:
        return self._rows

    @property
    def cur_row_num(self) -> int:
        return self._cursor

    def next(self) -> bool:
        return self._cursor < len(self._rows)

    def get_row_data(self) -> list[str]:
        values = self._rows[self._cursor]
        self._cursor += 1
        return values


class FakeCalendarModule:
    """query_trade_dates 返回固定结果；其余按 0.9.4 模块布局补齐。"""

    def __init__(self, result: FakeTradeDateResult, *, page_rows: int = PAGE_ROWS) -> None:
        self._result = result
        self.common = SimpleNamespace(
            contants=SimpleNamespace(BAOSTOCK_PER_PAGE_COUNT=page_rows),
            context=SimpleNamespace(default_socket=SimpleNamespace(close=lambda: None)),
        )
        self.login_calls = 0
        self.logout_calls = 0
        self.requests: list[tuple[str, str]] = []

    def login(self) -> FakeTradeDateResult:
        self.login_calls += 1
        return FakeTradeDateResult([])

    def logout(self) -> None:
        self.logout_calls += 1

    def query_trade_dates(self, *, start_date: str, end_date: str) -> FakeTradeDateResult:
        self.requests.append((start_date, end_date))
        return self._result


def calendar_rows(*entries: tuple[str, str]) -> list[list[str]]:
    return [[day, flag] for day, flag in entries]


def test_loader_keeps_only_trading_days() -> None:
    module = FakeCalendarModule(
        FakeTradeDateResult(
            calendar_rows(
                ("2026-09-24", "1"),
                ("2026-09-25", "1"),
                ("2026-09-26", "0"),
                ("2026-09-27", "0"),
                ("2026-09-28", "1"),
            )
        )
    )
    loader = BaoStockTradingCalendarLoader(baostock_module=_as_module(module))
    calendar = loader.load(start=date(2026, 9, 24), end=date(2026, 9, 28))
    loader.close()

    assert isinstance(calendar, TradingCalendar)
    assert calendar.exchange == "SSE"
    assert calendar.source == "baostock:query_trade_dates"
    assert calendar.sessions == (date(2026, 9, 24), date(2026, 9, 25), date(2026, 9, 28))
    assert calendar.is_session(date(2026, 9, 25))
    assert not calendar.is_session(date(2026, 9, 26))
    assert module.requests == [("2026-09-24", "2026-09-28")]
    assert module.login_calls == 1
    assert module.logout_calls == 1


def test_next_sessions_beyond_coverage_raises_calendar_error() -> None:
    module = FakeCalendarModule(FakeTradeDateResult(calendar_rows(("2026-09-25", "1"))))
    loader = BaoStockTradingCalendarLoader(baostock_module=_as_module(module))
    calendar = loader.load(start=date(2026, 9, 25), end=date(2026, 9, 25))
    loader.close()

    with pytest.raises(CalendarError):
        calendar.next_sessions(date(2026, 9, 25), 3)


def test_empty_response_is_explicit_provider_error() -> None:
    """跨年静默空（error_code=0、0 行）必须显式失败，不得产出空日历。"""
    module = FakeCalendarModule(FakeTradeDateResult([]), page_rows=1)
    loader = BaoStockTradingCalendarLoader(baostock_module=_as_module(module))
    with pytest.raises(ProviderError, match="no rows"):
        loader.load(start=date(2026, 12, 30), end=date(2027, 1, 5))
    loader.close()


def test_rows_without_trading_days_is_provider_error() -> None:
    module = FakeCalendarModule(
        FakeTradeDateResult(calendar_rows(("2026-10-03", "0"), ("2026-10-04", "0")))
    )
    loader = BaoStockTradingCalendarLoader(baostock_module=_as_module(module))
    with pytest.raises(ProviderError, match="no trading sessions"):
        loader.load(start=date(2026, 10, 3), end=date(2026, 10, 4))
    loader.close()


def test_unknown_trading_flag_is_data_quality_error() -> None:
    module = FakeCalendarModule(FakeTradeDateResult(calendar_rows(("2026-09-25", "2"))))
    loader = BaoStockTradingCalendarLoader(baostock_module=_as_module(module))
    with pytest.raises(DataQualityError, match="is_trading_day"):
        loader.load(start=date(2026, 9, 25), end=date(2026, 9, 25))
    loader.close()


def test_malformed_date_is_data_quality_error() -> None:
    module = FakeCalendarModule(FakeTradeDateResult(calendar_rows(("2026/09/25", "1"))))
    loader = BaoStockTradingCalendarLoader(baostock_module=_as_module(module))
    with pytest.raises(DataQualityError, match="malformed"):
        loader.load(start=date(2026, 9, 25), end=date(2026, 9, 25))
    loader.close()


def test_duplicate_calendar_date_is_data_quality_error() -> None:
    module = FakeCalendarModule(
        FakeTradeDateResult(calendar_rows(("2026-09-25", "1"), ("2026-09-25", "1")))
    )
    loader = BaoStockTradingCalendarLoader(baostock_module=_as_module(module))
    with pytest.raises(DataQualityError, match="repeats"):
        loader.load(start=date(2026, 9, 25), end=date(2026, 9, 25))
    loader.close()


def test_inverted_range_rejected_before_login() -> None:
    module = FakeCalendarModule(FakeTradeDateResult(calendar_rows(("2026-09-25", "1"))))
    loader = BaoStockTradingCalendarLoader(baostock_module=_as_module(module))
    with pytest.raises(ConfigurationError, match="must be <="):
        loader.load(start=date(2026, 9, 26), end=date(2026, 9, 25))
    loader.close()
    assert module.login_calls == 0


def _as_module(fake: Any) -> ModuleType:
    return fake  # type: ignore[return-value]
