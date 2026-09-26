"""BaoStock Provider 显式失败语义（RX-KAI-004，ADR-010）。"""

from datetime import date, datetime, time
from typing import Any

import pandas as pd
import pytest

from kronos_ai.data.base import MarketDataProvider
from kronos_ai.domain.time import SHANGHAI
from kronos_ai.errors import (
    ConfigurationError,
    DataQualityError,
    InsufficientHistoryError,
    ProviderError,
)
from kronos_ai.infrastructure.providers.baostock import (
    BAO_STOCK_PUBLISHED_AT,
    BaoStockProvider,
    to_baostock_code,
)

MD = date(2026, 9, 25)
CUTOFF = datetime(2026, 9, 25, 18, 0, tzinfo=SHANGHAI)
INTRADAY_CUTOFF = datetime(2026, 9, 25, 14, 30, tzinfo=SHANGHAI)

COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount", "tradestatus"]


def row(
    day: str,
    close: str = "10.0",
    tradestatus: str = "1",
    volume: str = "1000000",
    amount: str = "10000000.0",
) -> dict[str, str]:
    return {
        "date": day,
        "open": "9.0",
        "high": "10.5",
        "low": "8.5",
        "close": close,
        "volume": volume,
        "amount": amount,
        "tradestatus": tradestatus,
    }


def frame(*rows: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=COLUMNS)


class FakeResult:
    def __init__(
        self, error_code: str = "0", error_msg: str = "", data: pd.DataFrame | None = None
    ) -> None:
        self.error_code = error_code
        self.error_msg = error_msg
        self._data = data if data is not None else pd.DataFrame()

    def get_data(self) -> pd.DataFrame:
        return self._data


def ok(*rows: dict[str, str]) -> FakeResult:
    return FakeResult(data=frame(*rows))


class FakeBaostock:
    """按调用顺序消费 query_results，最后一个结果重复用于后续查询。"""

    def __init__(
        self,
        login_result: FakeResult | None = None,
        query_results: list[FakeResult | None] | None = None,
    ) -> None:
        self.login_result = login_result if login_result is not None else FakeResult()
        self._query_results: list[FakeResult | None] = query_results or [FakeResult()]
        self.login_calls = 0
        self.logout_calls = 0
        self.queries: list[dict[str, Any]] = []

    def login(self) -> FakeResult:
        self.login_calls += 1
        return self.login_result

    def logout(self) -> None:
        self.logout_calls += 1

    def query_history_k_data_plus(
        self,
        code: str,
        fields: str,
        start_date: str = "",
        end_date: str = "",
        frequency: str = "",
        adjustflag: str = "3",
    ) -> FakeResult | None:
        index = min(len(self.queries), len(self._query_results) - 1)
        self.queries.append(
            {
                "code": code,
                "fields": fields,
                "start_date": start_date,
                "end_date": end_date,
                "frequency": frequency,
                "adjustflag": adjustflag,
            }
        )
        return self._query_results[index]


def provider(fake: FakeBaostock, **kwargs: Any) -> BaoStockProvider:
    return BaoStockProvider(baostock_module=fake, **kwargs)


class TestToBaostockCode:
    @pytest.mark.parametrize(
        ("symbol", "expected"),
        [
            ("600000", "sh.600000"),
            ("688981", "sh.688981"),
            ("900901", "sh.900901"),
            ("000001", "sz.000001"),
            ("300750", "sz.300750"),
            ("200011", "sz.200011"),
            ("830799", "bj.830799"),
            ("920002", "bj.920002"),
        ],
    )
    def test_mapping(self, symbol: str, expected: str) -> None:
        assert to_baostock_code(symbol) == expected

    def test_unknown_board_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="cannot map"):
            to_baostock_code("999999")

    def test_prefixed_input_rejected(self) -> None:
        # 前缀形式只允许出现在 get_history 入口，不允许穿透到映射函数
        with pytest.raises(ConfigurationError, match="cannot map"):
            to_baostock_code("sh.600000")


class TestHappyPath:
    def sample(self) -> FakeResult:
        return ok(
            row("2026-09-22"),
            row("2026-09-23", tradestatus="0"),
            row("2026-09-24"),
            row("2026-09-25"),
        )

    def test_history_contents(self) -> None:
        fake = FakeBaostock(query_results=[self.sample()])
        history = provider(fake).get_history("600000", MD, CUTOFF, lookback_bars=3)
        assert history.symbol == "600000"
        # 停牌日 09-23 不是 valid bar，应只剩 09-22/24/25
        assert [b.timestamp.day for b in history.bars] == [22, 24, 25]
        assert history.bars[-1].trade_status == "normal"
        assert history.bars[-1].adjustment_mode == "raw"
        assert history.bars[0].volume == 1_000_000.0
        assert history.bars[0].amount == 10_000_000.0
        assert history.provider == "baostock"
        assert history.dataset_version == "baostock-v1"
        assert len(history.data_hash) == 64

    def test_available_at_is_publish_time(self) -> None:
        assert time(18, 0) == BAO_STOCK_PUBLISHED_AT
        fake = FakeBaostock(query_results=[self.sample()])
        history = provider(fake).get_history("600000", MD, CUTOFF, lookback_bars=1)
        bar = history.bars[-1]
        assert bar.timestamp == datetime(2026, 9, 25, 15, 0, tzinfo=SHANGHAI)
        assert bar.available_at == CUTOFF

    def test_query_parameters(self) -> None:
        fake = FakeBaostock(query_results=[self.sample()])
        provider(fake).get_history("600000", MD, CUTOFF, lookback_bars=3)
        assert len(fake.queries) == 1
        query = fake.queries[0]
        assert query["code"] == "sh.600000"
        assert query["end_date"] == "2026-09-25"
        assert query["start_date"] == "2026-08-26"
        assert query["frequency"] == "d"
        assert query["adjustflag"] == "3"

    def test_intraday_cutoff_excludes_today(self) -> None:
        fake = FakeBaostock(query_results=[self.sample()])
        history = provider(fake).get_history("600000", MD, INTRADAY_CUTOFF, lookback_bars=3)
        assert [b.timestamp.day for b in history.bars] == [22, 24]

    def test_lookback_truncates_to_latest(self) -> None:
        fake = FakeBaostock(query_results=[self.sample()])
        history = provider(fake).get_history("600000", MD, CUTOFF, lookback_bars=2)
        assert [b.timestamp.day for b in history.bars] == [24, 25]

    def test_blank_volume_and_amount_become_none(self) -> None:
        fake = FakeBaostock(query_results=[ok(row("2026-09-25", volume="", amount=""))])
        history = provider(fake).get_history("600000", MD, CUTOFF, lookback_bars=1)
        assert history.bars[0].volume is None
        assert history.bars[0].amount is None

    def test_prefixed_symbol_normalized(self) -> None:
        fake = FakeBaostock(query_results=[self.sample()])
        history = provider(fake).get_history("sh.600000", MD, CUTOFF, lookback_bars=3)
        assert history.symbol == "600000"

    def test_qfq_mode_propagates(self) -> None:
        fake = FakeBaostock(query_results=[self.sample()])
        history = provider(fake, adjust_flag="2").get_history("600000", MD, CUTOFF, lookback_bars=3)
        assert history.bars[0].adjustment_mode == "qfq"
        assert fake.queries[0]["adjustflag"] == "2"


class TestWindowExpansion:
    def test_expands_until_enough_bars(self) -> None:
        fake = FakeBaostock(
            query_results=[ok(row("2026-09-25")), ok(row("2026-09-24"), row("2026-09-25"))]
        )
        history = provider(fake).get_history("600000", MD, CUTOFF, lookback_bars=2)
        assert [b.timestamp.day for b in history.bars] == [24, 25]
        assert [q["start_date"] for q in fake.queries] == ["2026-08-26", "2026-07-27"]

    def test_short_history_returned_when_expansion_exhausted(self) -> None:
        # 新股/长期停牌：如实返回已有 bar，由上层按 min_history_bars 决策
        fake = FakeBaostock(query_results=[ok(row("2026-09-25"))])
        history = provider(fake).get_history("600000", MD, CUTOFF, lookback_bars=5)
        assert len(history.bars) == 1
        assert len(fake.queries) == 4

    def test_no_valid_bars_raises(self) -> None:
        fake = FakeBaostock(query_results=[ok()])
        with pytest.raises(InsufficientHistoryError, match="no valid bars"):
            provider(fake).get_history("600000", MD, CUTOFF, lookback_bars=3)

    def test_all_suspended_raises(self) -> None:
        fake = FakeBaostock(query_results=[ok(row("2026-09-25", tradestatus="0"))])
        with pytest.raises(InsufficientHistoryError, match="no valid bars"):
            provider(fake).get_history("600000", MD, CUTOFF, lookback_bars=1)

    def test_nonpositive_close_row_skipped(self) -> None:
        fake = FakeBaostock(query_results=[ok(row("2026-09-24"), row("2026-09-25", close="0.0"))])
        history = provider(fake).get_history("600000", MD, CUTOFF, lookback_bars=1)
        assert [b.timestamp.day for b in history.bars] == [24]


class TestExplicitFailures:
    def test_login_failure_raises_provider_error(self) -> None:
        fake = FakeBaostock(login_result=FakeResult(error_code="1", error_msg="bad credentials"))
        with pytest.raises(ProviderError, match="login failed"):
            provider(fake).get_history("600000", MD, CUTOFF, lookback_bars=1)

    def test_query_returning_none_raises_provider_error(self) -> None:
        fake = FakeBaostock(query_results=[None])
        with pytest.raises(ProviderError, match="returned None"):
            provider(fake).get_history("600000", MD, CUTOFF, lookback_bars=1)

    def test_query_error_raises_provider_error(self) -> None:
        fake = FakeBaostock(
            query_results=[FakeResult(error_code="10001", error_msg="network error")]
        )
        with pytest.raises(ProviderError, match="network error"):
            provider(fake).get_history("600000", MD, CUTOFF, lookback_bars=1)

    @pytest.mark.parametrize(
        "bad_row",
        [
            row("2026-09-25", close="abc"),
            row("2026-09-25", tradestatus="9"),
            row("2026-13-45"),
            {**row("2026-09-25"), "high": "8.0"},
        ],
    )
    def test_bad_row_raises_data_quality_error(self, bad_row: dict[str, str]) -> None:
        fake = FakeBaostock(query_results=[ok(bad_row)])
        with pytest.raises(DataQualityError, match="malformed row"):
            provider(fake).get_history("600000", MD, CUTOFF, lookback_bars=1)

    def test_unknown_adjust_flag_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="adjust_flag"):
            BaoStockProvider(adjust_flag="9", baostock_module=FakeBaostock())

    def test_nonpositive_lookback_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="lookback_bars"):
            provider(FakeBaostock()).get_history("600000", MD, CUTOFF, lookback_bars=0)

    def test_invalid_symbol_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="invalid symbol"):
            provider(FakeBaostock()).get_history("abc", MD, CUTOFF, lookback_bars=1)

    def test_unmappable_symbol_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="cannot map"):
            provider(FakeBaostock()).get_history("999999", MD, CUTOFF, lookback_bars=1)


class TestSessionLifecycle:
    def test_login_is_lazy_until_first_query(self) -> None:
        fake = FakeBaostock()
        p = provider(fake)
        assert fake.login_calls == 0
        p.close()
        assert fake.login_calls == 0
        assert fake.logout_calls == 0

    def test_session_shared_across_instances_and_refcounted(self) -> None:
        fake = FakeBaostock(query_results=[ok(row("2026-09-25"))])
        p1 = provider(fake)
        p1.get_history("600000", MD, CUTOFF, lookback_bars=1)
        assert fake.login_calls == 1

        p2 = provider(fake)
        p2.get_history("600000", MD, CUTOFF, lookback_bars=1)
        assert fake.login_calls == 1

        p1.close()
        assert fake.logout_calls == 0
        p2.close()
        assert fake.logout_calls == 1

    def test_close_is_idempotent(self) -> None:
        fake = FakeBaostock(query_results=[ok(row("2026-09-25"))])
        p = provider(fake)
        p.get_history("600000", MD, CUTOFF, lookback_bars=1)
        p.close()
        p.close()
        assert fake.logout_calls == 1

    def test_context_manager_closes(self) -> None:
        fake = FakeBaostock(query_results=[ok(row("2026-09-25"))])
        with provider(fake) as p:
            p.get_history("600000", MD, CUTOFF, lookback_bars=1)
        assert fake.logout_calls == 1

    def test_use_after_close_rejected(self) -> None:
        fake = FakeBaostock(query_results=[ok(row("2026-09-25"))])
        p = provider(fake)
        p.get_history("600000", MD, CUTOFF, lookback_bars=1)
        p.close()
        with pytest.raises(ConfigurationError, match="closed"):
            p.get_history("600000", MD, CUTOFF, lookback_bars=1)

    def test_satisfies_market_data_provider_protocol(self) -> None:
        assert isinstance(provider(FakeBaostock()), MarketDataProvider)
