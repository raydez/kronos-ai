"""TradingCalendar 契约与 session 语义（RX-KAI-005，基线文档 §6.3 / §6.6 / §11）。

fixture 为合成的 session 序列（非真实交易日历）；真实日历由 RX-KAI-006 spike
校准后接入，见 docs/decisions/ADR-009。
"""

from datetime import date, datetime

import pytest
from pydantic import ValidationError

from kronos_ai.data.calendar import StaticTradingCalendar, TradingCalendar
from kronos_ai.errors import CalendarError, ConfigurationError

# 合成序列：3/2(Mon)–3/6(Fri)，跳过周末与一个合成的 5 日休市窗口 3/9–3/13，
# 3/16(Mon)–3/20(Fri) 恢复。刻意不同于「工作日推导」的结果，用于验证时间轴真源。
SESSIONS = (
    date(2026, 3, 2),
    date(2026, 3, 3),
    date(2026, 3, 4),
    date(2026, 3, 5),
    date(2026, 3, 6),
    date(2026, 3, 16),
    date(2026, 3, 17),
    date(2026, 3, 18),
    date(2026, 3, 19),
    date(2026, 3, 20),
)
FIRST = SESSIONS[0]
LAST = SESSIONS[-1]
CLOSURE_DAY = date(2026, 3, 11)


def make_calendar(**overrides: object) -> StaticTradingCalendar:
    fields: dict[str, object] = {
        "exchange": "SSE",
        "source": "synthetic-fixture",
        "sessions": SESSIONS,
    }
    fields.update(overrides)
    return StaticTradingCalendar(**fields)  # type: ignore[arg-type]


class TestIsSession:
    def test_session_days(self) -> None:
        cal = make_calendar()
        assert cal.is_session(FIRST)
        assert cal.is_session(LAST)

    def test_weekend_inside_coverage_is_not_session(self) -> None:
        cal = make_calendar()
        assert not cal.is_session(date(2026, 3, 7))  # Sat
        assert not cal.is_session(date(2026, 3, 8))  # Sun

    def test_closure_window_inside_coverage_is_not_session(self) -> None:
        assert not make_calendar().is_session(CLOSURE_DAY)

    @pytest.mark.parametrize("day", [date(2026, 3, 1), date(2026, 3, 21)])
    def test_outside_coverage_raises(self, day: date) -> None:
        with pytest.raises(CalendarError, match="outside calendar coverage"):
            make_calendar().is_session(day)

    def test_datetime_rejected(self) -> None:
        stamp = datetime(2026, 3, 16, 15, 0)
        with pytest.raises(ConfigurationError, match="must be a date, not datetime"):
            make_calendar().is_session(stamp)  # type: ignore[arg-type]


class TestNextSessions:
    def test_excludes_market_date(self) -> None:
        assert make_calendar().next_sessions(date(2026, 3, 5), 1) == [date(2026, 3, 6)]

    def test_skips_weekend(self) -> None:
        assert make_calendar().next_sessions(date(2026, 3, 6), 1) == [date(2026, 3, 16)]

    def test_horizon_across_closure_window(self) -> None:
        # 工作日推导会给出 3/9–3/12（落在合成休市窗口内），日历必须给出真实 session
        assert make_calendar().next_sessions(date(2026, 3, 6), 4) == [
            date(2026, 3, 16),
            date(2026, 3, 17),
            date(2026, 3, 18),
            date(2026, 3, 19),
        ]

    def test_full_horizon(self) -> None:
        assert make_calendar().next_sessions(FIRST, 5) == list(SESSIONS[1:6])

    def test_market_date_must_be_session(self) -> None:
        cal = make_calendar()
        for day in (date(2026, 3, 7), CLOSURE_DAY):
            with pytest.raises(CalendarError, match="not a market session"):
                cal.next_sessions(day, 1)

    @pytest.mark.parametrize("count", [0, -1])
    def test_nonpositive_count_rejected(self, count: int) -> None:
        with pytest.raises(ConfigurationError, match="count must be >= 1"):
            make_calendar().next_sessions(date(2026, 3, 5), count)

    def test_market_date_outside_coverage_raises(self) -> None:
        with pytest.raises(CalendarError, match="outside calendar coverage"):
            make_calendar().next_sessions(date(2026, 3, 21), 1)

    def test_datetime_market_date_rejected(self) -> None:
        stamp = datetime(2026, 3, 5, 15, 0)
        with pytest.raises(ConfigurationError, match="must be a date, not datetime"):
            make_calendar().next_sessions(stamp, 1)  # type: ignore[arg-type]

    def test_insufficient_coverage_raises(self) -> None:
        # 3/19 之后只剩 1 个 session，要求 2 个必须显式失败而非截断
        with pytest.raises(CalendarError, match="coverage ends at"):
            make_calendar().next_sessions(date(2026, 3, 19), 2)

    def test_last_session_has_no_next(self) -> None:
        with pytest.raises(CalendarError, match="coverage ends at"):
            make_calendar().next_sessions(LAST, 1)


class TestConstruction:
    @pytest.mark.parametrize(
        "sessions",
        [
            (),
            (SESSIONS[1], SESSIONS[0]),
            (SESSIONS[0], SESSIONS[1], SESSIONS[1]),
        ],
    )
    def test_invalid_sequences_rejected(self, sessions: tuple[date, ...]) -> None:
        with pytest.raises(ValidationError):
            make_calendar(sessions=sessions)

    def test_provenance_is_required(self) -> None:
        with pytest.raises(ValidationError):
            StaticTradingCalendar(sessions=SESSIONS)  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            make_calendar(source="  ")
        with pytest.raises(ValidationError):
            make_calendar(exchange="HKEX")

    def test_coverage(self) -> None:
        assert make_calendar().coverage == (FIRST, LAST)

    def test_exchange_bounded(self) -> None:
        assert make_calendar(exchange="BSE").exchange == "BSE"

    def test_frozen(self) -> None:
        cal = make_calendar()
        with pytest.raises(ValidationError):
            cal.exchange = "SZSE"  # type: ignore[misc]

    def test_sessions_input_is_normalized_to_tuple(self) -> None:
        cal = make_calendar(sessions=list(SESSIONS))
        assert isinstance(cal.sessions, tuple)


class TestValidationBypass:
    """model_construct / model_copy 绕过校验时必须显式失败，不得抛裸异常。"""

    def test_construct_without_sessions(self) -> None:
        cal = StaticTradingCalendar.model_construct(exchange="SSE", source="x", sessions=())
        with pytest.raises(CalendarError, match="no sessions"):
            _ = cal.coverage
        with pytest.raises(CalendarError, match="no sessions"):
            cal.is_session(FIRST)
        with pytest.raises(CalendarError, match="no sessions"):
            cal.next_sessions(FIRST, 1)


class TestProtocolConformance:
    def test_satisfies_trading_calendar(self) -> None:
        assert isinstance(make_calendar(), TradingCalendar)

    def test_returns_fresh_list(self) -> None:
        assert isinstance(make_calendar().next_sessions(FIRST, 2), list)
