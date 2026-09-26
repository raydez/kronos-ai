from datetime import date, datetime

import pandas as pd
import pytest

from kronos_ai.domain.time import SHANGHAI
from kronos_ai.errors import DataQualityError, ProviderError
from kronos_ai.infrastructure.providers.baostock import BaoStockProvider, to_baostock_code

MD = date(2026, 9, 25)
CUTOFF = datetime(2026, 9, 25, 18, 0, tzinfo=SHANGHAI)

COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount", "tradestatus"]


def row(day: str, close: str = "10.0", tradestatus: str = "1", volume: str = "1000000") -> dict:
    return {
        "date": day,
        "open": "9.0",
        "high": "10.5",
        "low": "8.5",
        "close": close,
        "volume": volume,
        "amount": "10000000.0",
        "tradestatus": tradestatus,
    }


def frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=COLUMNS)


class FakeResult:
    def __init__(
        self, error_code: str = "0", error_msg: str = "", data: pd.DataFrame | None = None
    ):
        self.error_code = error_code
        self.error_msg = error_msg
        self._data = data if data is not None else pd.DataFrame()

    def get_data(self) -> pd.DataFrame:
        return self._data


class FakeBaostock:
    def __init__(
        self, login_result: FakeResult | None = None, query_result: FakeResult | None = None
    ):
        self.login_result = login_result or FakeResult()
        self.query_result = query_result
        self.login_calls = 0
        self.logout_calls = 0
        self.queries: list[dict] = []

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
        return self.query_result


def provider(fake: FakeBaostock) -> BaoStockProvider:
    return BaoStockProvider(baostock_module=fake)


class TestToBaostockCode:
    @pytest.mark.parametrize(
        ("symbol", "expected"),
        [
            ("600000", "sh.600000"),
            ("688981", "sh.688981"),
            ("000001", "sz.000001"),
            ("300750", "sz.300750"),
            ("830799", "bj.830799"),
            ("920002", "bj.920002"),
        ],
    )
    def test_mapping(self, symbol: str, expected: str) -> None:
        assert to_baostock_code(symbol) == expected

    def test_unknown_board_rejected(self) -> None:
        with pytest.raises(ProviderError, match="cannot map"):
            to_baostock_code("999999")


class TestBaoStockProvider:
    def sample_data(self) -> pd.DataFrame:
        return frame(
            [
                row("2026-09-22"),
                row("2026-09-23", tradestatus="0"),
                row("2026-09-24"),
                row("2026-09-25"),
            ]
        )

    def test_happy_path(self) -> None:
        fake = FakeBaostock(query_result=FakeResult(data=self.sample_data()))
        history = provider(fake).get_history("600000", MD, CUTOFF, lookback_bars=3)
        assert history.symbol == "600000"
        # 停牌日 09-23 不是 valid bar，应只剩 09-22/24/25
        assert [b.timestamp.day for b in history.bars] == [22, 24, 25]
        assert history.bars[-1].trade_status == "normal"
        assert history.bars[-1].adjustment_mode == "raw"
        assert history.bars[0].volume == 1_000_000.0
        assert history.provider == "baostock"
        assert len(history.data_hash) == 64

    def test_intraday_cutoff_excludes_today(self) -> None:
        fake = FakeBaostock(query_result=FakeResult(data=self.sample_data()))
        cutoff = datetime(2026, 9, 25, 14, 30, tzinfo=SHANGHAI)
        history = provider(fake).get_history("600000", MD, cutoff, lookback_bars=3)
        assert [b.timestamp.day for b in history.bars] == [22, 24]

    def test_lookback_truncates_to_latest(self) -> None:
        fake = FakeBaostock(query_result=FakeResult(data=self.sample_data()))
        history = provider(fake).get_history("600000", MD, CUTOFF, lookback_bars=2)
        assert [b.timestamp.day for b in history.bars] == [24, 25]

    def test_login_failure_raises(self) -> None:
        fake = FakeBaostock(login_result=FakeResult(error_code="1", error_msg="bad credentials"))
        with pytest.raises(ProviderError, match="login failed"):
            provider(fake).get_history("600000", MD, CUTOFF, lookback_bars=3)

    def test_none_query_result_raises(self) -> None:
        fake = FakeBaostock(query_result=None)  # type: ignore[arg-type]
        with pytest.raises(ProviderError, match="returned None"):
            provider(fake).get_history("600000", MD, CUTOFF, lookback_bars=3)

    def test_query_error_raises(self) -> None:
        fake = FakeBaostock(query_result=FakeResult(error_code="10001", error_msg="network error"))
        with pytest.raises(ProviderError, match="network error"):
            provider(fake).get_history("600000", MD, CUTOFF, lookback_bars=3)

    def test_empty_data_raises(self) -> None:
        fake = FakeBaostock(query_result=FakeResult(data=frame([])))
        with pytest.raises(ProviderError, match="no data"):
            provider(fake).get_history("600000", MD, CUTOFF, lookback_bars=3)

    def test_malformed_row_raises(self) -> None:
        bad = row("2026-09-24", close="abc")
        fake = FakeBaostock(query_result=FakeResult(data=frame([bad])))
        with pytest.raises(DataQualityError, match="malformed row"):
            provider(fake).get_history("600000", MD, CUTOFF, lookback_bars=1)

    def test_unknown_adjust_flag_rejected(self) -> None:
        with pytest.raises(ProviderError, match="adjust_flag"):
            BaoStockProvider(adjust_flag="9", baostock_module=FakeBaostock())

    def test_qfq_mode_propagates(self) -> None:
        fake = FakeBaostock(query_result=FakeResult(data=self.sample_data()))
        p = BaoStockProvider(adjust_flag="2", baostock_module=fake)
        history = p.get_history("600000", MD, CUTOFF, lookback_bars=3)
        assert history.bars[0].adjustment_mode == "qfq"

    def test_blank_volume_becomes_none(self) -> None:
        data = frame([row("2026-09-24", volume="")])
        fake = FakeBaostock(query_result=FakeResult(data=data))
        history = provider(fake).get_history("600000", MD, CUTOFF, lookback_bars=1)
        assert history.bars[0].volume is None

    def test_prefixed_symbol_normalized(self) -> None:
        fake = FakeBaostock(query_result=FakeResult(data=self.sample_data()))
        history = provider(fake).get_history("sh.600000", MD, CUTOFF, lookback_bars=3)
        assert history.symbol == "600000"

    def test_zero_lookback_rejected(self) -> None:
        fake = FakeBaostock()
        with pytest.raises(DataQualityError, match="lookback_bars"):
            provider(fake).get_history("600000", MD, CUTOFF, lookback_bars=0)

    def test_satisfies_market_data_provider_protocol(self) -> None:
        from kronos_ai.data.base import MarketDataProvider
        from kronos_ai.infrastructure.providers.baostock import BaoStockProvider as P

        assert isinstance(P(baostock_module=FakeBaostock()), MarketDataProvider)
