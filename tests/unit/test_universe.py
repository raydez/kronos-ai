"""Point-in-Time universe 快照与历史成分股加载器（RX-KAI-007，ADR-008）。

Fake 按 spike 核实的服务端语义建模（``docs/spike/baostock-capability.md`` §3）：
``date=`` 返回当时有效的名单、每行带 ``updateDate``；早于可用起点返回空名单；
**未来日期静默返回最新名单**且 ``data.date`` 原样回显（无法据此识别 clamp）。
"""

from __future__ import annotations

import threading
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from kronos_ai.data.universe import (
    EXPECTED_MEMBER_COUNT,
    UNIVERSE_ID_HS300,
    UNIVERSE_ID_ZZ500,
    UNIVERSE_PIT_POLICY_VERSION,
    UniverseLoader,
    UniverseSnapshot,
)
from kronos_ai.domain.time import SHANGHAI
from kronos_ai.errors import (
    ConfigurationError,
    DataQualityError,
    ProviderError,
    UniverseError,
)
from kronos_ai.infrastructure.providers.baostock import (
    BaoStockProvider,
    BaoStockUniverseLoader,
)

LATEST_UPDATE = date(2026, 9, 21)  # spike 观察值：两个指数最新名单的 updateDate
HS300_AVAILABLE_FROM = date(2006, 1, 4)  # spike 探测边界：首个非空探针
ZZ500_AVAILABLE_FROM = date(2007, 1, 31)

CUTOFF = datetime(2026, 9, 25, 18, 0, tzinfo=SHANGHAI)
FIELDS = ["updateDate", "code", "code_name"]

PAGE_ROWS = 2000  # baostock 0.9.4 common.contants.BAOSTOCK_PER_PAGE_COUNT


def members(count: int, first: int = 600000) -> list[str]:
    return [f"{first + offset:06d}" for offset in range(count)]


HS300_MEMBERS = members(EXPECTED_MEMBER_COUNT[UNIVERSE_ID_HS300])
ZZ500_MEMBERS = members(EXPECTED_MEMBER_COUNT[UNIVERSE_ID_ZZ500], first=1)


class FakeResult:
    """模拟 0.9.4 ``ResultData``（成分股查询结果 ≤ 单页，无需翻页路径）。"""

    def __init__(
        self,
        rows: list[list[str]] | None = None,
        fields: list[str] | None = None,
        error_code: str = "0",
        error_msg: str = "",
        date_echo: str = "",
    ) -> None:
        self.fields = list(FIELDS if fields is None else fields)
        self.error_code = error_code
        self.error_msg = error_msg
        self.date = date_echo  # 服务端原样回显请求日期（含未来日期），不提示 clamp
        self._rows = list(rows or [])
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
        if self._cursor < len(self._rows):
            values = self._rows[self._cursor]
            self._cursor += 1
            return values
        return []


class FakeSocket:
    def __init__(self) -> None:
        self.closed = False
        self._released = threading.Event()

    def close(self) -> None:
        self.closed = True
        self._released.set()

    def spin_until_closed(self) -> None:
        self._released.wait(timeout=10)


def rows_for(codes: list[str], update_date: date) -> list[list[str]]:
    return [[update_date.isoformat(), code, f"名{code}"] for code in codes]


class FakeBaostock:
    """按指数持有 (代码列表, 可用起点)，返回与请求日期一致的名单语义。"""

    def __init__(
        self,
        *,
        hs300: tuple[list[str], date] = (HS300_MEMBERS, HS300_AVAILABLE_FROM),
        zz500: tuple[list[str], date] = (ZZ500_MEMBERS, ZZ500_AVAILABLE_FROM),
        response_rows: Any = None,
        response_fields: list[str] | None = None,
        response_error: tuple[str, str] | None = None,
        expose_page_count: bool = True,
        hang_query: bool = False,
    ) -> None:
        self._indices = {"query_hs300_stocks": hs300, "query_zz500_stocks": zz500}
        self._response_rows = response_rows
        self._response_fields = response_fields
        self._response_error = response_error
        self.socket = FakeSocket()
        contants = (
            SimpleNamespace(BAOSTOCK_PER_PAGE_COUNT=PAGE_ROWS)
            if expose_page_count
            else SimpleNamespace()
        )
        self.common = SimpleNamespace(
            contants=contants, context=SimpleNamespace(default_socket=self.socket)
        )
        self.hang_query = hang_query
        self.login_calls = 0
        self.logout_calls = 0
        self.calls: list[tuple[str, str]] = []
        self.last_result: FakeResult | None = None

    def login(self) -> FakeResult:
        self.login_calls += 1
        return FakeResult()

    def logout(self) -> None:
        self.logout_calls += 1

    def __getattr__(self, name: str) -> Any:
        if name not in ("query_hs300_stocks", "query_zz500_stocks"):
            raise AttributeError(name)

        def query(date: str = "") -> FakeResult | None:
            self.calls.append((name, date))
            if self.hang_query:
                self.socket.spin_until_closed()
            if self._response_error is not None:
                code, msg = self._response_error
                self.last_result = FakeResult(error_code=code, error_msg=msg, date_echo=date)
                return self.last_result
            if self._response_rows is not None:
                self.last_result = FakeResult(
                    rows=self._response_rows, fields=self._response_fields, date_echo=date
                )
                return self.last_result
            codes, available_from = self._indices[name]
            self.last_result = FakeResult(
                rows=self._rows_for(codes, available_from, date),
                fields=self._response_fields,
                date_echo=date,
            )
            return self.last_result

        return query

    @staticmethod
    def _rows_for(codes: list[str], available_from: date, day: str) -> list[list[str]]:
        if day == "":  # 缺省 = 最新名单
            return rows_for(codes, LATEST_UPDATE)
        requested = date.fromisoformat(day)
        if requested < available_from:
            return []  # 早于可用起点：空名单，error_code 仍为 0
        if requested >= LATEST_UPDATE:
            # 未来日期被静默 clamp 到最新名单（spike §3 的泄漏形态）
            return rows_for(codes, LATEST_UPDATE)
        # 历史请求：名单是「最近一次修订」的版本，修订日 ≤ 请求日
        return rows_for(codes, requested - timedelta(days=1))

    # 供跨类会话共享测试使用：最小可用的日线查询
    def query_history_k_data_plus(self, *args: Any, **kwargs: Any) -> FakeResult:
        fields = ["date", "open", "high", "low", "close", "volume", "amount", "tradestatus"]
        return FakeResult(
            rows=[["2026-09-25", "9.0", "10.5", "8.5", "10.0", "1000", "10000", "1"]],
            fields=fields,
        )


def loader(fake: FakeBaostock, **kwargs: Any) -> BaoStockUniverseLoader:
    return BaoStockUniverseLoader(baostock_module=fake, **kwargs)  # type: ignore[arg-type]


class TestUniverseSnapshotContract:
    def snapshot(self, **overrides: Any) -> UniverseSnapshot:
        fields: dict[str, Any] = {
            "universe_id": UNIVERSE_ID_HS300,
            "effective_date": date(2020, 6, 30),
            "symbols": tuple(sorted(HS300_MEMBERS)),
            "source": "baostock:query_hs300_stocks",
            "version": "baostock-v1",
            "update_date": date(2020, 6, 29),
        }
        fields.update(overrides)
        return UniverseSnapshot(**fields)

    def test_canonical_snapshot(self) -> None:
        snapshot = self.snapshot()
        assert snapshot.member_count == EXPECTED_MEMBER_COUNT[UNIVERSE_ID_HS300]
        assert snapshot.symbols[0] == "600000"

    def test_unsorted_symbols_rejected(self) -> None:
        with pytest.raises(ValueError, match="sorted"):
            self.snapshot(symbols=tuple(reversed(sorted(HS300_MEMBERS))))

    def test_duplicate_symbols_rejected(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            self.snapshot(symbols=("600000", "600000"))

    def test_prefixed_symbol_rejected(self) -> None:
        with pytest.raises(ValueError, match="normalized"):
            self.snapshot(symbols=("sh.600000",))

    def test_empty_symbols_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            self.snapshot(symbols=())

    def test_list_revised_after_request_date_rejected(self) -> None:
        # 晚于请求时点才修订的名单当日尚未生效：用它就是未来函数
        with pytest.raises(ValueError, match="after effective_date"):
            self.snapshot(effective_date=date(2020, 6, 28), update_date=date(2020, 6, 29))

    def test_frozen(self) -> None:
        snapshot = self.snapshot()
        with pytest.raises(ValueError):
            snapshot.universe_id = "other"  # type: ignore[misc]


class TestLoaderHappyPath:
    def test_hs300_snapshot_matches_response_revision(self) -> None:
        fake = FakeBaostock()
        snapshot = loader(fake).load(UNIVERSE_ID_HS300, date(2020, 6, 30), CUTOFF)
        assert fake.calls == [("query_hs300_stocks", "2020-06-30")]
        assert snapshot.universe_id == UNIVERSE_ID_HS300
        assert snapshot.effective_date == date(2020, 6, 30)
        assert snapshot.update_date == date(2020, 6, 29)
        assert snapshot.source == "baostock:query_hs300_stocks"
        assert snapshot.version == "baostock-v1"
        assert snapshot.symbols == tuple(sorted(HS300_MEMBERS))

    def test_zz500_uses_its_own_query_and_size(self) -> None:
        fake = FakeBaostock()
        snapshot = loader(fake).load(UNIVERSE_ID_ZZ500, date(2008, 6, 30), CUTOFF)
        assert fake.calls == [("query_zz500_stocks", "2008-06-30")]
        assert snapshot.member_count == EXPECTED_MEMBER_COUNT[UNIVERSE_ID_ZZ500]
        assert snapshot.source == "baostock:query_zz500_stocks"

    def test_same_day_cutoff_is_allowed(self) -> None:
        fake = FakeBaostock()
        cutoff = datetime(2020, 6, 30, 18, 0, tzinfo=SHANGHAI)
        snapshot = loader(fake).load(UNIVERSE_ID_HS300, date(2020, 6, 30), cutoff)
        assert snapshot.effective_date == date(2020, 6, 30)

    def test_dataset_version_is_recorded(self) -> None:
        fake = FakeBaostock()
        snapshot = loader(fake, dataset_version="baostock-v2").load(
            UNIVERSE_ID_HS300, date(2020, 6, 30), CUTOFF
        )
        assert snapshot.version == "baostock-v2"

    def test_satisfies_universe_loader_protocol(self) -> None:
        assert isinstance(loader(FakeBaostock()), UniverseLoader)


class TestLoaderPointInTimeGuards:
    def test_before_availability_is_explicit_not_empty(self) -> None:
        # ZZ500 的可用起点比 HS300 晚一年（spike §3）：不得用最新名单替代
        fake = FakeBaostock()
        with pytest.raises(UniverseError, match="do not substitute"):
            loader(fake).load(UNIVERSE_ID_ZZ500, date(2006, 6, 20), CUTOFF)

    def test_error_message_carries_index_specific_probe_bounds(self) -> None:
        fake = FakeBaostock()
        with pytest.raises(UniverseError, match=r"empty at 2007-01-04, non-empty at 2007-01-31"):
            loader(fake).load(UNIVERSE_ID_ZZ500, date(2007, 1, 4), CUTOFF)
        fake = FakeBaostock()
        with pytest.raises(UniverseError, match=r"empty at 2005-12-30, non-empty at 2006-01-04"):
            loader(fake).load(UNIVERSE_ID_HS300, date(2005, 6, 30), CUTOFF)

    def test_future_date_is_rejected_before_hitting_the_server(self) -> None:
        # 服务端对未来日期静默返回最新名单（error_code=0、data.date 原样回显），
        # 唯一可靠的防线是调用侧的 effective_date <= cutoff 当日检查
        fake = FakeBaostock()
        with pytest.raises(ConfigurationError, match="beyond the cutoff cannot be known"):
            loader(fake).load(UNIVERSE_ID_HS300, date(2027, 6, 30), CUTOFF)
        assert fake.calls == []

    def test_unknown_universe_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="unknown universe_id"):
            loader(FakeBaostock()).load("csi1000", date(2020, 6, 30), CUTOFF)

    def test_datetime_effective_date_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="must be a date"):
            loader(FakeBaostock()).load(
                UNIVERSE_ID_HS300,
                datetime(2020, 6, 30, 15, 0),
                CUTOFF,  # type: ignore[arg-type]
            )

    def test_string_effective_date_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="must be a date"):
            loader(FakeBaostock()).load(
                UNIVERSE_ID_HS300,
                "2020-06-30",  # type: ignore[arg-type]
                CUTOFF,
            )

    def test_non_datetime_cutoff_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="must be a datetime"):
            loader(FakeBaostock()).load(
                UNIVERSE_ID_HS300,
                date(2020, 6, 30),
                date(2020, 6, 30),  # type: ignore[arg-type]
            )

    def test_naive_cutoff_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="timezone-aware"):
            loader(FakeBaostock()).load(
                UNIVERSE_ID_HS300,
                date(2020, 6, 30),
                datetime(2020, 6, 30, 18, 0),  # type: ignore[arg-type]
            )

    def test_non_shanghai_cutoff_rejected(self) -> None:
        # +14:00 的偏移会让 .date() 漂移一天（2020-06-30 18:00+14:00 的本地日期已是
        # 07-01），若按 offset 换算就会放过未来 universe，因此必须显式拒绝
        utc_plus_14 = datetime(2020, 6, 30, 18, 0, tzinfo=timezone(timedelta(hours=14)))
        fake = FakeBaostock()
        with pytest.raises(ConfigurationError, match=r"Asia/Shanghai \(\+08:00\)"):
            loader(fake).load(UNIVERSE_ID_HS300, date(2020, 6, 30), utc_plus_14)
        assert fake.calls == []


class TestSameDayRevisionPolicy:
    def test_policy_version_is_pinned(self) -> None:
        # 语义变更（如收紧为严格早于 cutoff 当日）必须升版本，
        # 否则历史 run 无法分辨自己用的是哪版规则（ADR-008 §4）
        assert UNIVERSE_PIT_POLICY_VERSION == "universe-pit-policy-v1"

    def test_same_day_revision_is_accepted(self) -> None:
        # v1 的乐观假设：名单修订的日内发布时刻未知，update_date == effective_date
        # 视为当日 cutoff 之前已知（spike 中 2008-06-30 探针即此形态）
        same_day = date(2020, 6, 30)
        fake = FakeBaostock(response_rows=rows_for(HS300_MEMBERS, same_day))
        snapshot = loader(fake).load(
            UNIVERSE_ID_HS300, same_day, datetime(2020, 6, 30, 18, 0, tzinfo=SHANGHAI)
        )
        assert snapshot.update_date == snapshot.effective_date == same_day


class TestLoaderDataQuality:
    def test_short_list_rejected(self) -> None:
        fake = FakeBaostock(response_rows=rows_for(HS300_MEMBERS[:-1], LATEST_UPDATE))
        with pytest.raises(DataQualityError, match="exactly 300"):
            loader(fake).load(UNIVERSE_ID_HS300, date(2020, 6, 30), CUTOFF)

    def test_duplicate_codes_rejected(self) -> None:
        duplicated = [*HS300_MEMBERS[:299], HS300_MEMBERS[0]]
        fake = FakeBaostock(response_rows=rows_for(duplicated, LATEST_UPDATE))
        with pytest.raises(DataQualityError, match="duplicate"):
            loader(fake).load(UNIVERSE_ID_HS300, date(2020, 6, 30), CUTOFF)

    def test_mixed_update_dates_rejected(self) -> None:
        rows = rows_for(HS300_MEMBERS[:150], LATEST_UPDATE) + rows_for(
            HS300_MEMBERS[150:], date(2020, 6, 29)
        )
        fake = FakeBaostock(response_rows=rows)
        with pytest.raises(DataQualityError, match="distinct updateDate"):
            loader(fake).load(UNIVERSE_ID_HS300, date(2020, 6, 30), CUTOFF)

    def test_revision_after_requested_date_rejected(self) -> None:
        fake = FakeBaostock(response_rows=rows_for(HS300_MEMBERS, LATEST_UPDATE))
        with pytest.raises(DataQualityError, match="violated the snapshot contract"):
            loader(fake).load(UNIVERSE_ID_HS300, date(2020, 6, 30), CUTOFF)

    def test_unparsable_update_date_rejected(self) -> None:
        rows = [["2020/06/29", code, "名"] for code in HS300_MEMBERS]
        fake = FakeBaostock(response_rows=rows)
        with pytest.raises(DataQualityError, match="unparsable updateDate"):
            loader(fake).load(UNIVERSE_ID_HS300, date(2020, 6, 30), CUTOFF)

    def test_malformed_code_rejected(self) -> None:
        rows = rows_for(HS300_MEMBERS, LATEST_UPDATE)
        rows[3][1] = "6000000"
        fake = FakeBaostock(response_rows=rows)
        with pytest.raises(DataQualityError, match="malformed code"):
            loader(fake).load(UNIVERSE_ID_HS300, date(2020, 6, 30), CUTOFF)

    def test_missing_update_date_column_rejected(self) -> None:
        rows = [[code, "名"] for code in HS300_MEMBERS]
        fake = FakeBaostock(response_rows=rows, response_fields=["code", "code_name"])
        with pytest.raises(DataQualityError, match="lacks column"):
            loader(fake).load(UNIVERSE_ID_HS300, date(2020, 6, 30), CUTOFF)


class TestLoaderTransportFailures:
    def test_server_error_code_is_provider_error(self) -> None:
        fake = FakeBaostock(response_error=("10001001", "网络接收超时"))
        with pytest.raises(ProviderError, match=r"query_hs300_stocks failed.*网络接收超时"):
            loader(fake).load(UNIVERSE_ID_HS300, date(2020, 6, 30), CUTOFF)

    def test_missing_field_metadata_rejected(self) -> None:
        fake = FakeBaostock(response_rows=[], response_fields=[])
        with pytest.raises(ProviderError, match="no field metadata"):
            loader(fake).load(UNIVERSE_ID_HS300, date(2020, 6, 30), CUTOFF)

    def test_hanging_query_times_out_and_breaks_socket(self) -> None:
        fake = FakeBaostock(hang_query=True)
        instance = loader(fake, hard_timeout_seconds=0.05)
        with pytest.raises(ProviderError, match="hard timeout"):
            instance.load(UNIVERSE_ID_HS300, date(2020, 6, 30), CUTOFF)
        assert fake.socket.closed is True
        # break_socket 作废登录态：下一次调用必须重新 login 而不是复用死会话
        fake.hang_query = False
        instance.load(UNIVERSE_ID_HS300, date(2020, 6, 30), CUTOFF)
        assert fake.login_calls == 2

    def test_missing_page_count_constant_is_explicit_failure(self) -> None:
        with pytest.raises(ProviderError, match="BAOSTOCK_PER_PAGE_COUNT"):
            loader(FakeBaostock(expose_page_count=False))

    def test_nonpositive_timeout_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="hard_timeout_seconds"):
            loader(FakeBaostock(), hard_timeout_seconds=0)

    def test_use_after_close_rejected(self) -> None:
        fake = FakeBaostock()
        instance = loader(fake)
        instance.load(UNIVERSE_ID_HS300, date(2020, 6, 30), CUTOFF)
        instance.close()
        instance.close()
        assert fake.logout_calls == 1
        with pytest.raises(ConfigurationError, match="closed"):
            instance.load(UNIVERSE_ID_HS300, date(2020, 6, 30), CUTOFF)


class TestSessionSharingWithPriceProvider:
    def test_price_provider_and_loader_share_one_login(self) -> None:
        fake = FakeBaostock()
        provider = BaoStockProvider(baostock_module=fake)  # type: ignore[arg-type]
        universe = loader(fake)
        provider.get_history("600000", date(2026, 9, 25), CUTOFF, lookback_bars=1)
        universe.load(UNIVERSE_ID_HS300, date(2020, 6, 30), CUTOFF)
        assert fake.login_calls == 1
        provider.close()
        assert fake.logout_calls == 0
        universe.close()
        assert fake.logout_calls == 1
