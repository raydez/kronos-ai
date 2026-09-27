"""BaoStock Provider 显式失败语义（RX-KAI-004，ADR-010）。

Fake 模拟 baostock 0.9.4 ``ResultData`` 的完整迭代协议（``fields`` + 分页游标
``next()`` / ``get_row_data()`` / ``data`` / ``cur_row_num``），且刻意**不提供**
``get_data()``：0.9.4 该方法用已移除的 ``DataFrame.append`` 合并翻页结果，超过单页
（2000 行）即抛 AttributeError，provider 不得依赖它（RX-KAI-006 评审 Major 1）。
翻页失败形态按 ``data/resultset.py`` 源码建模（评审 C1）。
"""

from __future__ import annotations

import socket
import threading
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from typing import Any

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
    BAOSTOCK_HARD_TIMEOUT_SECONDS,
    BAOSTOCK_SOCKET_TIMEOUT_SECONDS,
    BaoStockProvider,
    to_baostock_code,
)

MD = date(2026, 9, 25)
CUTOFF = datetime(2026, 9, 25, 18, 0, tzinfo=SHANGHAI)
INTRADAY_CUTOFF = datetime(2026, 9, 25, 14, 30, tzinfo=SHANGHAI)

COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount", "tradestatus"]

PAGE_ROWS = 2000  # baostock 0.9.4 common.contants.BAOSTOCK_PER_PAGE_COUNT


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


def day(offset: int) -> str:
    return (date(2015, 1, 1) + timedelta(days=offset)).isoformat()


class FakeResult:
    """模拟 0.9.4 ``ResultData`` 的分页游标（读 ``data/resultset.py`` 源码）。

    只有当前页**正好**满载 ``PAGE_ROWS`` 行时才请求下一页。下一页请求失败于
    ``send_msg``（socket 超时/断连）时**不置 error_code** 直接返回 False，唯一痕迹是
    游标停在整页；服务端错误码则写入 ``error_code`` 后返回 False。``page_*`` 参数按
    「第 N 次翻页请求」（1 起）注入这两种失败与客户端异常。
    """

    def __init__(
        self,
        error_code: str = "0",
        error_msg: str = "",
        rows: list[list[str]] | None = None,
        fields: list[str] | None = None,
        page_rows: int = PAGE_ROWS,
        page_none_at: int | None = None,
        page_error_at: tuple[int, str, str] | None = None,
        page_raise_at: tuple[int, Exception] | None = None,
    ) -> None:
        self.error_code = error_code
        self.error_msg = error_msg
        self.fields = list(COLUMNS if fields is None else fields)
        self._rows = list(rows or [])
        self._page = 0
        self._cursor = 0
        self._page_rows = page_rows
        self._page_none_at = page_none_at
        self._page_error_at = page_error_at
        self._page_raise_at = page_raise_at
        self.page_requests = 0

    @property
    def data(self) -> list[list[str]]:
        start = self._page * self._page_rows
        return self._rows[start : start + self._page_rows]

    @property
    def cur_row_num(self) -> int:
        return self._cursor

    def next(self) -> bool:
        """与 0.9.4 一致：只回答「还有没有」，**不**推进游标。"""
        if self._cursor < len(self.data):
            return True
        if len(self.data) < self._page_rows:  # 不满一页：正常收尾，不请求下一页
            return False
        return self._request_next_page()

    def _request_next_page(self) -> bool:
        request_index = self.page_requests + 1
        self.page_requests += 1
        if self._page_none_at == request_index:  # send_msg → None：静默截断
            return False
        if self._page_error_at is not None and self._page_error_at[0] == request_index:
            _, self.error_code, self.error_msg = self._page_error_at
            return False
        if self._page_raise_at is not None and self._page_raise_at[0] == request_index:
            raise self._page_raise_at[1]
        self._page += 1
        self._cursor = 0
        return self._cursor < len(self.data)

    def get_row_data(self) -> list[str]:
        """与 0.9.4 一致：返回当前行并推进游标；越界返回空 list 而不抛异常。"""
        if self._cursor < len(self.data):
            values = self.data[self._cursor]
            self._cursor += 1
            return values
        return []

    def fresh(self) -> FakeResult:
        """真实 API 每次查询返回新的 ``ResultData``；游标/页号不得跨查询复用。"""
        return FakeResult(
            error_code=self.error_code,
            error_msg=self.error_msg,
            rows=self._rows,
            fields=self.fields,
            page_rows=self._page_rows,
            page_none_at=self._page_none_at,
            page_error_at=self._page_error_at,
            page_raise_at=self._page_raise_at,
        )


class FakeSocket:
    """模拟 ``baostock.common.context.default_socket``。

    provider 的硬超时只能靠关闭会话 socket 打断 ``send_msg`` 的 EOF 空转
    （EOF 时 ``recv`` 立即返回 ``b""``，socket 超时永不触发）。
    """

    def __init__(self) -> None:
        self.closed = False
        self._released = threading.Event()

    def close(self) -> None:
        self.closed = True
        self._released.set()

    def spin_until_closed(self) -> None:
        """空转直到 socket 被关闭；真实库在 EOF 时**永不**自行返回。"""
        self._released.wait(timeout=10)


def to_values(r: dict[str, str]) -> list[str]:
    """按 0.9.4 ``fields`` 顺序把 dict 行摊平成 ``get_row_data()`` 的 list。"""
    return [r[column] for column in COLUMNS]


def ok(*rows: dict[str, str]) -> FakeResult:
    return FakeResult(rows=[to_values(r) for r in rows])


def full_page(page_rows: int = PAGE_ROWS) -> list[list[str]]:
    """一整页（默认 ``PAGE_ROWS`` 行）合法行：翻页请求的触发器。"""
    return [to_values(row(day(offset))) for offset in range(page_rows)]


class FakeBaostock:
    """按调用顺序消费 query_results，最后一个结果重复用于后续查询。

    ``common.contants.BAOSTOCK_PER_PAGE_COUNT`` 与 ``common.context.default_socket``
    按 0.9.4 的模块布局提供：前者是 provider 页长的唯一来源（缺失须显式失败），
    后者用于验证硬超时以关闭 socket 打断空转。
    """

    def __init__(
        self,
        login_result: FakeResult | None = None,
        query_results: list[FakeResult | None] | None = None,
        login_error: Exception | None = None,
        *,
        page_rows: int = PAGE_ROWS,
        expose_page_count: bool = True,
        hang_query: bool = False,
        hang_login: bool = False,
    ) -> None:
        self.login_result = login_result if login_result is not None else FakeResult()
        self.login_error = login_error
        self._query_results: list[FakeResult | None] = query_results or [FakeResult()]
        self.socket = FakeSocket()
        contants = (
            SimpleNamespace(BAOSTOCK_PER_PAGE_COUNT=page_rows)
            if expose_page_count
            else SimpleNamespace()
        )
        self.common = SimpleNamespace(
            contants=contants, context=SimpleNamespace(default_socket=self.socket)
        )
        self.hang_query = hang_query
        self.hang_login = hang_login
        self.login_calls = 0
        self.logout_calls = 0
        self.login_timeouts: list[float | None] = []
        self.queries: list[dict[str, Any]] = []
        self.last_result: FakeResult | None = None

    def login(self) -> FakeResult:
        self.login_calls += 1
        self.login_timeouts.append(socket.getdefaulttimeout())
        if self.hang_login:
            self.socket.spin_until_closed()
        if self.login_error is not None:
            raise self.login_error
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
        if self.hang_query:
            self.socket.spin_until_closed()
        result = self._query_results[index]
        self.last_result = result.fresh() if result is not None else None
        return self.last_result


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
        fake = FakeBaostock(query_results=[self.sample()])
        history = provider(fake).get_history("600000", MD, CUTOFF, lookback_bars=1)
        bar = history.bars[-1]
        assert bar.timestamp == datetime(2026, 9, 25, 15, 0, tzinfo=SHANGHAI)
        # 用字面量钉住 18:00：该值是**未核实**的乐观假设（能力报告 §6），改动它必须
        # 同步 knowledge_cutoff_policy 版本与文档，故不引用 BAO_STOCK_PUBLISHED_AT 自证
        assert bar.available_at == datetime(2026, 9, 25, 18, 0, tzinfo=SHANGHAI)
        assert bar.available_at <= CUTOFF

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


class TestPaginationContract:
    """RX-KAI-006 评审 Major 1 / C1 的回归测试。

    0.9.4 的 ``ResultData.get_data()`` 依赖 pandas 已移除的 ``DataFrame.append``，
    且单页只 2000 行，长窗口必然翻页并抛 AttributeError（不是 ProviderError）。
    provider 必须走 ``next()`` / ``get_row_data()``，并把翻页失败转成 ProviderError。
    """

    def test_does_not_rely_on_get_data(self) -> None:
        # FakeResult 刻意不提供 get_data()：provider 一旦调用就会以 AttributeError 失败
        assert not hasattr(ok(row("2026-09-25")), "get_data")

    def test_page_size_comes_from_library(self) -> None:
        # 页长必须从库常量读取：假库声明 3 行/页时，跨页与收尾都按 3 行工作
        many = [row(day(offset)) for offset in range(4)]
        page_3 = [to_values(r) for r in many]
        fake = FakeBaostock(
            query_results=[FakeResult(rows=page_3, page_rows=3)],
            page_rows=3,
        )
        history = provider(fake).get_history("600000", MD, CUTOFF, lookback_bars=len(many))
        assert len(history.bars) == 4
        assert fake.last_result is not None
        assert fake.last_result.page_requests == 1

    def test_truncation_detected_at_library_page_size(self) -> None:
        fake = FakeBaostock(
            query_results=[FakeResult(rows=full_page(3), page_rows=3, page_none_at=1)],
            page_rows=3,
        )
        with pytest.raises(ProviderError, match=r"truncated.*full page"):
            provider(fake).get_history("600000", MD, CUTOFF, lookback_bars=1)

    def test_missing_page_count_constant_is_explicit_failure(self) -> None:
        # 页长读不到就显式失败：硬编码会在库改页长时让截断检测静默失效
        with pytest.raises(ProviderError, match="BAOSTOCK_PER_PAGE_COUNT"):
            provider(FakeBaostock(expose_page_count=False))

    def test_collects_across_page_boundary(self) -> None:
        # 2001 行 = 整页 + 1：provider 必须发出第 2 次分页请求
        many = [row(day(offset)) for offset in range(PAGE_ROWS + 1)]
        fake = FakeBaostock(query_results=[ok(*many)])
        history = provider(fake).get_history("600000", MD, CUTOFF, lookback_bars=len(many))
        assert len(history.bars) == PAGE_ROWS + 1
        assert history.bars[0].timestamp.date() == date(2015, 1, 1)
        assert history.bars[-1].timestamp.date() == date(2015, 1, 1) + timedelta(days=PAGE_ROWS)
        assert fake.last_result is not None
        assert fake.last_result.page_requests == 1

    def test_exact_page_multiple_ends_after_empty_page(self) -> None:
        # 4000 行：第 2 页满载后仍请求第 3 页并取到空页；空页收尾不是截断
        many = [row(day(offset)) for offset in range(2 * PAGE_ROWS)]
        fake = FakeBaostock(query_results=[ok(*many)])
        history = provider(fake).get_history("600000", MD, CUTOFF, lookback_bars=len(many))
        assert len(history.bars) == 2 * PAGE_ROWS
        assert fake.last_result is not None
        assert fake.last_result.page_requests == 2

    def test_silent_truncation_after_full_page_is_detected(self) -> None:
        # send_msg 返回 None 时 0.9.4 不置 error_code，唯一痕迹是游标停在整页
        fake = FakeBaostock(query_results=[FakeResult(rows=full_page(), page_none_at=1)])
        with pytest.raises(ProviderError, match=r"truncated.*full page"):
            provider(fake).get_history("600000", MD, CUTOFF, lookback_bars=1)

    def test_page_request_error_code_is_detected(self) -> None:
        fake = FakeBaostock(
            query_results=[
                FakeResult(
                    rows=full_page(),
                    page_error_at=(1, "10002008", "网络接收超时"),
                )
            ]
        )
        with pytest.raises(ProviderError, match=r"page request failed.*网络接收超时"):
            provider(fake).get_history("600000", MD, CUTOFF, lookback_bars=1)

    def test_paging_error_becomes_provider_error(self) -> None:
        fake = FakeBaostock(
            query_results=[
                FakeResult(
                    rows=full_page(),
                    page_raise_at=(1, RuntimeError("connection reset")),
                )
            ]
        )
        with pytest.raises(ProviderError, match=r"paging failed.*connection reset"):
            provider(fake).get_history("600000", MD, CUTOFF, lookback_bars=1)

    def test_missing_field_metadata_rejected(self) -> None:
        fake = FakeBaostock(query_results=[FakeResult(rows=[], fields=[])])
        with pytest.raises(ProviderError, match="no field metadata"):
            provider(fake).get_history("600000", MD, CUTOFF, lookback_bars=1)

    def test_row_value_count_mismatch_rejected(self) -> None:
        fake = FakeBaostock(query_results=[FakeResult(rows=[["2026-09-25", "9.0"]])])
        with pytest.raises(DataQualityError, match="values for 8 fields"):
            provider(fake).get_history("600000", MD, CUTOFF, lookback_bars=1)


class TestHardTimeout:
    """RX-KAI-006 评审 M2：库没有超时 API，EOF 空转只能由硬超时打断。

    假库的 ``spin_until_closed`` 模拟 ``send_msg`` 的 ``while True: recv()``：
    EOF 时立即返回 ``b""``，socket 超时永不触发，只有关闭 socket 才能退出。
    """

    def test_hanging_query_times_out_and_breaks_socket(self) -> None:
        fake = FakeBaostock(query_results=[ok(row("2026-09-25"))], hang_query=True)
        p = provider(fake, hard_timeout_seconds=0.05)
        with pytest.raises(ProviderError, match=r"hard timeout"):
            p.get_history("600000", MD, CUTOFF, lookback_bars=1)
        assert fake.socket.closed is True  # 关闭 socket 是唯一的打断手段
        # break_socket 作废登录态：下一次调用必须重新 login 而不是复用死会话
        fake.hang_query = False
        p.get_history("600000", MD, CUTOFF, lookback_bars=1)
        assert fake.login_calls == 2

    def test_hanging_login_times_out(self) -> None:
        fake = FakeBaostock(hang_login=True, query_results=[ok(row("2026-09-25"))])
        p = provider(fake, hard_timeout_seconds=0.05)
        with pytest.raises(ProviderError, match=r"hard timeout"):
            p.get_history("600000", MD, CUTOFF, lookback_bars=1)
        assert fake.socket.closed is True

    def test_nonpositive_timeout_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="hard_timeout_seconds"):
            provider(FakeBaostock(), hard_timeout_seconds=0.0)

    def test_default_timeout_constant_is_pinned(self) -> None:
        # 常量是本层与运行时长之间的契约：改动必须是有意为之
        assert BAOSTOCK_HARD_TIMEOUT_SECONDS == 60.0


class TestSessionLifecycle:
    def test_login_is_lazy_until_first_query(self) -> None:
        fake = FakeBaostock()
        p = provider(fake)
        assert fake.login_calls == 0
        p.close()
        assert fake.login_calls == 0
        assert fake.logout_calls == 0

    def test_login_bounds_socket_timeout_and_restores_default(self) -> None:
        # M2：login 期间收紧进程默认 socket 超时，使会话 socket 的 recv 有界
        previous = socket.getdefaulttimeout()
        fake = FakeBaostock(query_results=[ok(row("2026-09-25"))])
        provider(fake).get_history("600000", MD, CUTOFF, lookback_bars=1)
        assert fake.login_timeouts == [BAOSTOCK_SOCKET_TIMEOUT_SECONDS]
        assert socket.getdefaulttimeout() == previous

    def test_login_raise_is_wrapped_and_timeout_restored(self) -> None:
        previous = socket.getdefaulttimeout()
        fake = FakeBaostock(login_error=RuntimeError("connect failed"))
        with pytest.raises(ProviderError, match="login raised"):
            provider(fake).get_history("600000", MD, CUTOFF, lookback_bars=1)
        assert socket.getdefaulttimeout() == previous

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
