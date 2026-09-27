"""Explicit-failure BaoStock Provider（RX-KAI-004，基线文档 §18 / ADR-010）。

v1 的 5 处 `_generate_fallback_data` 静默 fallback 全部不迁移：
任何数据源失败都抛 ProviderError，由上层显式处理。

语义约定：
- 输入/输出 symbol 均为规范 6 位代码；`sh.`/`sz.`/`bj.` 前缀仅在适配层出现；
- BaoStock 日线 volume 单位为股、amount 单位为元，与 domain 约定一致，无需换算；
- available_at 采用常量 BAO_STOCK_PUBLISHED_AT（当日 18:00+08:00）。RX-KAI-006 能力
  spike（docs/spike/baostock-capability.md §6）只能确认「已完成 session 的 bar 事后可取」
  （artifact: checks.publication_lag），无法观测确切发布时刻；该常量是未核实的假设，
  且方向是**乐观**而非保守：若真实发布晚于 18:00，同日晚间 run 会用到尚未发布的 bar。
  其语义由版本化的 knowledge_cutoff_policy（cutoff-policy-v1）承载，不得当作已验证事实；
- 停牌日（tradestatus=0）与无效价格（close<=0）行不是 valid bar，不进入
  MarketHistory；无法识别的交易状态是数据缺陷，显式抛 DataQualityError；
- 原始行级数据由 append-only raw snapshot（RX-KAI-015）负责留痕；
- 有效 bar 数少于 lookback_bars（新股/长期停牌）时如实返回，由上层按
  min_history_bars 策略抛 InsufficientHistoryError；窗口内无任何有效 bar
  时本层直接抛 InsufficientHistoryError。

分页与失败语义（读 0.9.4 源码核实，详见 `_fetch_window` / `_drain_pages`）：
- 不使用 `ResultData.get_data()`（依赖已移除的 `DataFrame.append`）；
- 查询与翻页抛出的任何非 KronosAIError 异常都包装为 ProviderError；
- 翻页途中服务端错误码、以及 `send_msg` 静默失败留下的「整页游标」都显式失败；
- 单页行数从库常量 `common.contants.BAOSTOCK_PER_PAGE_COUNT` 读取（缺失即失败），
  不硬编码——否则库改页长会让截断检测静默失效。

会话说明：baostock 登录态是进程级全局单会话，由模块级引用计数管理
（最后一个使用者 close 时才 logout），查询以会话锁串行化。库没有任何超时 API，
且 `send_msg` 的 `while True: recv()` 在连接被对端关闭时永久空转（EOF 立即返回
`b""`，socket 超时永不触发——这是 spike 观察到的挂死形态，见报告 §0），因此本层
施加两层上界：

1. login 期间临时收紧进程默认 socket 超时，使会话 socket 终身带上界，覆盖
   「对端静默不收不发」的阻塞等待；
2. login 与每次取数都在 daemon 线程内执行并施加硬超时（`_run_bounded`）；超时即
   关闭会话 socket，让空转循环以 `OSError` 收场、作废登录态，并抛 ProviderError。

同一模块还提供 `BaoStockUniverseLoader`（RX-KAI-007）：指数成分股查询与日线共享
上述会话、锁与超时策略；其 PIT 规则（空名单、可用起点、未来日期静默 clamp）见该类
docstring 与 ADR-008。
"""

from __future__ import annotations

import contextlib
import importlib
import socket
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from types import ModuleType
from typing import Any, Self, cast

from pydantic import ValidationError

from kronos_ai.data.base import MarketDataProvider  # noqa: F401  (re-export contract)
from kronos_ai.data.calendar import Exchange, StaticTradingCalendar
from kronos_ai.data.universe import (
    EXPECTED_MEMBER_COUNT,
    UNIVERSE_ID_HS300,
    UNIVERSE_ID_ZZ500,
    UniverseSnapshot,
)
from kronos_ai.domain.market import MarketBar, MarketHistory
from kronos_ai.domain.symbols import normalize_symbol
from kronos_ai.domain.time import CN_TZ, ensure_shanghai_aware
from kronos_ai.errors import (
    ConfigurationError,
    DataQualityError,
    InsufficientHistoryError,
    KronosAIError,
    ProviderError,
    UniverseError,
)

BAO_STOCK_PUBLISHED_AT = time(18, 0)

# socket 超时上界（秒）：覆盖「连接被静默丢弃、对端不收不发」的阻塞等待。
# baostock 0.9.4 没有任何超时 API；会话 socket 在 login 时创建、长期复用，
# 因此收紧 login 期间的进程默认超时即让该 socket 终身带上界。
BAOSTOCK_SOCKET_TIMEOUT_SECONDS = 30.0

# 硬超时（秒）：覆盖 socket 超时管不到的 EOF 空转（见模块 docstring）。spike 实测
# 单次查询最慢约 7s，60s 留足余量，同时把「无限挂起」限制为有限等待。
BAOSTOCK_HARD_TIMEOUT_SECONDS = 60.0

ADJUST_FLAG_TO_MODE: dict[str, str] = {
    "3": "raw",
    "1": "hfq",
    "2": "qfq",
}

TRADE_STATUS_MAP: dict[str, str] = {
    "1": "normal",
    "0": "suspended",
}

K_LINE_FIELDS = "date,open,high,low,close,volume,amount,tradestatus"

_A_SHARE_FIRST_TRADE_DAY = date(1990, 12, 19)  # spike 核实：query_trade_dates 首个 session


def to_baostock_code(symbol: str) -> str:
    """规范 6 位代码 → BaoStock 前缀代码；未知板块显式失败而非猜测。

    输入必须是已规范化代码（无 sh./sz./bj. 前缀）；无法映射属于调用方输入
    错误，抛 ConfigurationError 而非 ProviderError。
    """
    if symbol.startswith(("600", "601", "603", "605", "688", "689", "900")):
        return f"sh.{symbol}"
    if symbol.startswith(("000", "001", "002", "003", "200", "300", "301", "302")):
        return f"sz.{symbol}"
    if symbol.startswith(("43", "83", "87", "92")):
        return f"bj.{symbol}"
    raise ConfigurationError(f"cannot map symbol {symbol!r} to a BaoStock exchange prefix")


def _resolve_per_page_count(module: ModuleType) -> int:
    """单页行数必须来自库常量；缺失（版本不符）时显式失败而非猜测。"""
    contants = getattr(getattr(module, "common", None), "contants", None)
    value = getattr(contants, "BAOSTOCK_PER_PAGE_COUNT", None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ProviderError(
            "baostock does not expose common.contants.BAOSTOCK_PER_PAGE_COUNT; "
            "paging integrity cannot be verified (unsupported baostock version?)"
        )
    return value


def _run_bounded[T](
    work: Callable[[], T],
    *,
    label: str,
    timeout: float,
    on_timeout: Callable[[], None],
) -> T:
    """在 daemon 线程内执行可能永久阻塞的库调用，硬超时后显式失败。

    socket 超时只在「无数据可读」时触发；连接被对端关闭（EOF）时 ``recv`` 立即返回
    ``b""``，``send_msg`` 的 ``while True: recv()`` 会一直空转，超时永不生效。
    硬超时只能由调用方施加：超时后 ``on_timeout`` 关闭会话 socket，使空转循环以
    ``OSError`` 收场、僵尸线程自行结束，调用方得到 ProviderError 而不是无限等待。
    """
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = work()
        except BaseException as exc:  # 线程内任何异常都必须回传给调用线程
            box["error"] = exc

    worker = threading.Thread(target=target, name=f"kronos-baostock:{label}", daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        on_timeout()
        raise ProviderError(
            f"baostock {label} exceeded the {timeout:g}s hard timeout; "
            "the session socket was closed to break the client's recv spin"
        )
    if "error" in box:
        raise box["error"]
    return cast("T", box["value"])


def _login_with_bounded_timeout(module: ModuleType) -> Any:
    """login 期间临时收紧进程默认 socket 超时，给会话 socket 一个等待上界。

    ``socket.setdefaulttimeout`` 只影响此后**新建**的 socket；baostock 在 login 里
    创建会话 socket 并长期复用，因此临时收紧即可让它终身带超时，无需触碰库内部
    对象。副作用是同一时刻其他线程新建的 socket 会继承该默认值（30s 为保守上界），
    故 login 一结束立即恢复。

    超时本身仍会被 ``socketutil.send_msg`` 吞掉并表现为「无响应」（见
    ``_drain_pages`` 的静默截断检测），因此这里是**上界**而非错误信号；login 阶段的
    失败则可检出（connect 异常、``BSERR_RECVSOCK_FAIL``）。
    """
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(BAOSTOCK_SOCKET_TIMEOUT_SECONDS)
    try:
        return module.login()
    except Exception as exc:  # login 内部的 UnboundLocalError 等不得逃逸为裸异常
        raise ProviderError(f"baostock login raised: {exc!r}") from exc
    finally:
        socket.setdefaulttimeout(previous)


class _BaostockSession:
    """进程级 baostock 会话的引用计数管理：多实例共享、最后一个离开才 logout。"""

    def __init__(self, module: ModuleType) -> None:
        self.module = module
        self.lock = threading.RLock()
        self._refs = 0
        self._logged_in = False

    def acquire(self, hard_timeout: float) -> None:
        """登记一个持有者并确保已登录（每个 provider 实例只调用一次）。"""
        with self.lock:
            if not self._logged_in:
                self._login(hard_timeout)
            self._refs += 1

    def ensure(self, hard_timeout: float) -> None:
        """确保会话可用：硬超时 ``break_socket`` 作废登录态后由这里重新 login。"""
        with self.lock:
            if not self._logged_in:
                self._login(hard_timeout)

    def release(self) -> None:
        with self.lock:
            if self._refs > 0:
                self._refs -= 1
            if self._refs == 0 and self._logged_in:
                self.module.logout()
                self._logged_in = False

    def _login(self, hard_timeout: float) -> None:
        """调用方必须已持有 ``self.lock``；失败/超时都显式抛出。"""
        result = _run_bounded(
            lambda: _login_with_bounded_timeout(self.module),
            label="login",
            timeout=hard_timeout,
            on_timeout=self.break_socket,
        )
        if getattr(result, "error_code", "1") != "0":
            raise ProviderError(f"baostock login failed: {getattr(result, 'error_msg', 'unknown')}")
        self._logged_in = True

    def break_socket(self) -> None:
        """硬超时后关闭会话 socket 并作废登录态（只应由超时路径调用）。

        关闭 socket 是打断 ``while True: recv()`` 空转的唯一手段（见 ``_run_bounded``）；
        登录态随之作废，下一次使用会重新 login。
        """
        context = getattr(getattr(self.module, "common", None), "context", None)
        sock = getattr(context, "default_socket", None)
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.close()
        self._logged_in = False


_SESSIONS: dict[int, _BaostockSession] = {}
_SESSIONS_LOCK = threading.Lock()


def _session_for(module: ModuleType) -> _BaostockSession:
    with _SESSIONS_LOCK:
        key = id(module)
        session = _SESSIONS.get(key)
        if session is None or session.module is not module:
            session = _BaostockSession(module)
            _SESSIONS[key] = session
        return session


def _page_state_is_truncated(result: Any, per_page: int) -> bool:
    """游标停在整页 = ``send_msg`` 静默失败的唯一可检痕迹（见 ``_drain_pages``）。"""
    data = getattr(result, "data", None)
    cursor = getattr(result, "cur_row_num", None)
    if not isinstance(data, list) or not isinstance(cursor, int):
        return False
    return len(data) == per_page and cursor >= len(data)


def _drain_pages(
    result: Any, subject: str, fields: list[str], per_page: int
) -> list[dict[str, str]]:
    """耗尽 ``ResultData`` 游标并检测两类翻页失败（读 0.9.4 源码核实）。

    - 服务端错误码：``next()`` 先置 ``error_code`` 再返回 False，循环后复查可见；
    - ``send_msg`` 静默失败（socket 超时/对端断开/页号异常）：``next()`` 直接
      返回 False 且**不置 error_code**。其唯一痕迹是游标停在整页
      （``cur_row_num == len(data) == 单页行数``）——正常结束只有三种形态：
      空结果、非整页收尾、或整页后再取到空页（``data`` 清空）。因此
      「非空 + 恰好整页 + 游标到头」必然是截断，显式失败而非交给上层当短结果。
    """
    rows: list[dict[str, str]] = []
    while True:
        has_next = result.next()
        if not has_next:
            if getattr(result, "error_code", "0") != "0":
                raise ProviderError(
                    f"baostock page request failed for {subject}: "
                    f"{getattr(result, 'error_msg', 'unknown')}"
                )
            if _page_state_is_truncated(result, per_page):
                raise ProviderError(
                    f"baostock paging truncated for {subject} at {len(rows)} rows: "
                    "a full page was followed by an empty response with no error code"
                )
            return rows
        values = list(result.get_row_data())
        if len(values) != len(fields):
            raise DataQualityError(
                f"baostock returned {len(values)} values for {len(fields)} fields for {subject}"
            )
        rows.append(dict(zip(fields, values, strict=True)))


def _run_query(
    query: Callable[[], Any], *, label: str, subject: str, per_page: int
) -> list[dict[str, str]]:
    """执行一次 ``query_*`` 调用并收集全部行；失败一律显式（ADR-010）。

    ``label`` 是查询族名（``query`` / ``query_hs300_stocks`` …），只进错误消息。
    """
    try:
        result = query()
    except KronosAIError:
        raise
    except Exception as exc:  # 客户端本地异常（IndexError/JSONDecodeError 等）
        raise ProviderError(f"baostock {label} raised for {subject}: {exc!r}") from exc
    if result is None:  # 入参被客户端拒绝时直接返回 None，不是错误对象
        raise ProviderError(f"baostock {label} returned None for {subject}")
    if getattr(result, "error_code", "1") != "0":
        raise ProviderError(
            f"baostock {label} failed for {subject}: {getattr(result, 'error_msg', 'unknown')}"
        )
    try:
        fields = list(getattr(result, "fields", ()))
        if not fields:
            raise ProviderError(f"baostock returned no field metadata for {subject}")
        return _drain_pages(result, subject, fields, per_page)
    except KronosAIError:
        raise
    except Exception as exc:
        raise ProviderError(f"baostock paging failed for {subject}: {exc!r}") from exc


class _BaostockSessionClient:
    """baostock 会话持有者的公共骨架：登录、引用计数、关闭语义与硬超时配置。

    ``BaoStockProvider`` 与 ``BaoStockUniverseLoader`` 只叠加各自的查询逻辑；会话
    相关的修复（硬超时断 socket、断线后重登录）只应在这里发生一次，避免两份副本
    各自漂移。
    """

    def __init__(
        self,
        *,
        baostock_module: ModuleType | None,
        hard_timeout_seconds: float,
    ) -> None:
        if hard_timeout_seconds <= 0:
            raise ConfigurationError(
                f"hard_timeout_seconds must be > 0, got {hard_timeout_seconds!r}"
            )
        self._hard_timeout = hard_timeout_seconds
        self._bs = (
            baostock_module if baostock_module is not None else importlib.import_module("baostock")
        )
        self._per_page_count = _resolve_per_page_count(self._bs)
        self._session = _session_for(self._bs)
        self._session_acquired = False
        self._closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._session_acquired:
            self._session.release()

    def _ensure_session(self) -> None:
        if self._closed:
            raise ConfigurationError(
                f"{type(self).__name__} is closed; create a new instance to query"
            )
        if not self._session_acquired:
            self._session.acquire(self._hard_timeout)
            self._session_acquired = True
        else:
            self._session.ensure(self._hard_timeout)


class BaoStockProvider(_BaostockSessionClient):
    def __init__(
        self,
        *,
        adjust_flag: str = "3",
        dataset_version: str = "baostock-v1",
        baostock_module: ModuleType | None = None,
        hard_timeout_seconds: float = BAOSTOCK_HARD_TIMEOUT_SECONDS,
    ) -> None:
        if adjust_flag not in ADJUST_FLAG_TO_MODE:
            raise ConfigurationError(f"unknown adjust_flag: {adjust_flag!r}")
        self._adjust_flag = adjust_flag
        self._adjustment_mode = ADJUST_FLAG_TO_MODE[adjust_flag]
        self._dataset_version = dataset_version
        super().__init__(baostock_module=baostock_module, hard_timeout_seconds=hard_timeout_seconds)

    def get_history(
        self,
        symbol: str,
        market_date: date,
        knowledge_cutoff: datetime,
        lookback_bars: int,
    ) -> MarketHistory:
        try:
            symbol = normalize_symbol(symbol)
        except ValueError as exc:
            raise ConfigurationError(f"invalid symbol: {symbol!r}") from exc
        if lookback_bars < 1:
            raise ConfigurationError(f"lookback_bars must be >= 1, got {lookback_bars}")

        code = to_baostock_code(symbol)
        bars = self._collect_bars(code, symbol, market_date, knowledge_cutoff, lookback_bars)

        return MarketHistory(
            symbol=symbol,
            market_date=market_date,
            knowledge_cutoff=knowledge_cutoff,
            bars=tuple(bars),
            provider="baostock",
            dataset_version=self._dataset_version,
        )

    def _collect_bars(
        self,
        code: str,
        symbol: str,
        market_date: date,
        knowledge_cutoff: datetime,
        lookback_bars: int,
    ) -> list[MarketBar]:
        span_days = max(30, lookback_bars * 2)
        bars: list[MarketBar] = []
        for _ in range(4):
            start = market_date - timedelta(days=span_days)
            rows = self._fetch_window(code, start, market_date)
            bars = self._parse_bars(rows, symbol, knowledge_cutoff)
            if len(bars) >= lookback_bars:
                return bars[-lookback_bars:]
            span_days *= 2
            if start <= _A_SHARE_FIRST_TRADE_DAY:
                break
        if not bars:
            raise InsufficientHistoryError(
                f"baostock returned no valid bars for {code} up to {market_date}"
            )
        return bars

    def _fetch_window(self, code: str, start: date, end: date) -> list[dict[str, str]]:
        """按页收集行数据；不使用 ``ResultData.get_data()``。

        0.9.4 的 ``get_data()`` 用已从 pandas 移除的 ``DataFrame.append`` 合并翻页结果，
        单股窗口超过一页（``common.contants.BAOSTOCK_PER_PAGE_COUNT`` 行）即抛
        AttributeError；官方 demo 的 ``next()`` / ``get_row_data()`` 路径没有该问题。
        查询 + 翻页整体施加硬超时（见 ``_run_bounded``），超时关闭会话 socket。
        """
        self._ensure_session()
        with self._session.lock:
            return _run_bounded(
                lambda: _run_query(
                    lambda: self._bs.query_history_k_data_plus(
                        code,
                        K_LINE_FIELDS,
                        start_date=start.isoformat(),
                        end_date=end.isoformat(),
                        frequency="d",
                        adjustflag=self._adjust_flag,
                    ),
                    label="query",
                    subject=code,
                    per_page=self._per_page_count,
                ),
                label=f"query {code}",
                timeout=self._hard_timeout,
                on_timeout=self._session.break_socket,
            )

    def _parse_bars(
        self, rows: list[dict[str, str]], symbol: str, knowledge_cutoff: datetime
    ) -> list[MarketBar]:
        bars: list[MarketBar] = []
        for idx, row in enumerate(rows):
            bar_day_text = str(row.get("date", ""))
            try:
                bar_day = date.fromisoformat(bar_day_text)
                status = TRADE_STATUS_MAP[str(row["tradestatus"]).strip()]
                close = float(row["close"])
            except (KeyError, TypeError, ValueError) as exc:
                raise DataQualityError(
                    f"baostock returned malformed row #{idx} (date={bar_day_text!r}) "
                    f"for {symbol}: {exc}"
                ) from exc

            if status == "suspended" or close <= 0:
                continue

            timestamp = datetime.combine(bar_day, time(15, 0), tzinfo=CN_TZ)
            available_at = datetime.combine(bar_day, BAO_STOCK_PUBLISHED_AT, tzinfo=CN_TZ)
            if available_at > knowledge_cutoff:
                continue

            try:
                bars.append(
                    MarketBar(
                        symbol=symbol,
                        timestamp=timestamp,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=close,
                        volume=self._optional_float(row.get("volume")),
                        amount=self._optional_float(row.get("amount")),
                        trade_status=status,
                        adjustment_mode=self._adjustment_mode,
                        available_at=available_at,
                    )
                )
            except (KeyError, TypeError, ValueError, ValidationError) as exc:
                raise DataQualityError(
                    f"baostock returned malformed row #{idx} (date={bar_day_text!r}) "
                    f"for {symbol}: {exc}"
                ) from exc
        return bars

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        """缺失/空串/非正数 → None（domain 允许 volume/amount 为 None）。"""
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None


@dataclass(frozen=True)
class _IndexSpec:
    """单个指数的成分股查询配置与 spike 核实过的可用性边界。

    ``last_empty_probe`` / ``first_nonempty_probe`` 是 RX-KAI-006 spike 的**探测边界**
    （见 ``checks.index_constituents``），不是可用起点本身：真实起点在两者之间，
    未逐日探测，因此只用来构造错误消息，不参与任何判定。
    """

    universe_id: str
    query_method: str
    last_empty_probe: date
    first_nonempty_probe: date


# HS300 与 ZZ500 的可用深度不同（spike §3），不得共用起点，也不得相互默认。
_INDEX_SPECS: dict[str, _IndexSpec] = {
    UNIVERSE_ID_HS300: _IndexSpec(
        universe_id=UNIVERSE_ID_HS300,
        query_method="query_hs300_stocks",
        last_empty_probe=date(2005, 12, 30),
        first_nonempty_probe=date(2006, 1, 4),
    ),
    UNIVERSE_ID_ZZ500: _IndexSpec(
        universe_id=UNIVERSE_ID_ZZ500,
        query_method="query_zz500_stocks",
        last_empty_probe=date(2007, 1, 4),
        first_nonempty_probe=date(2007, 1, 31),
    ),
}


def _require_plain_date(value: object, field: str) -> None:
    """必须是 date 而不是 datetime 或别的类型。

    datetime 是 date 的子类，误传会在日期比较处按时间部分比较；非日期类型则会在
    比较处抛裸 TypeError——两者都必须在边界上变成 ConfigurationError。
    """
    if isinstance(value, datetime) or not isinstance(value, date):
        raise ConfigurationError(f"{field} must be a date, got {value!r}")


def _parse_constituents(
    rows: list[dict[str, str]], spec: _IndexSpec, effective_date: date
) -> tuple[tuple[str, ...], date]:
    """成分股行 → (规范代码元组, 名单修订日)；任何不符合契约的响应显式失败。"""
    missing = [name for name in ("updateDate", "code") if name not in rows[0]]
    if missing:
        raise DataQualityError(
            f"baostock {spec.query_method} response lacks column(s) {missing}; "
            "the list revision date cannot be established"
        )
    update_dates = sorted({row["updateDate"] for row in rows})
    if len(update_dates) != 1:
        # 名单是原子修订：多个修订日意味着拼接了不同版本的名单，无法证明 PIT 语义
        raise DataQualityError(
            f"baostock {spec.universe_id} constituents at {effective_date} carry "
            f"{len(update_dates)} distinct updateDate values {update_dates[:3]}; "
            "a single uniformly-revised list was expected"
        )
    try:
        update_date = date.fromisoformat(update_dates[0])
    except ValueError as exc:
        raise DataQualityError(
            f"baostock returned unparsable updateDate {update_dates[0]!r} for {spec.universe_id}"
        ) from exc

    symbols: list[str] = []
    for idx, row in enumerate(rows):
        raw_code = row["code"]
        try:
            symbols.append(normalize_symbol(raw_code))
        except ValueError as exc:
            raise DataQualityError(
                f"baostock returned malformed code #{idx} {raw_code!r} for {spec.universe_id}"
            ) from exc
    if len(set(symbols)) != len(symbols):
        raise DataQualityError(
            f"baostock {spec.universe_id} constituents at {effective_date} contain duplicate codes"
        )
    expected = EXPECTED_MEMBER_COUNT[spec.universe_id]
    if len(symbols) != expected:
        raise DataQualityError(
            f"baostock returned {len(symbols)} {spec.universe_id} constituents at "
            f"{effective_date}; exactly {expected} were expected"
        )
    return tuple(sorted(symbols)), update_date


class BaoStockUniverseLoader(_BaostockSessionClient):
    """历史成分股加载器（RX-KAI-007，ADR-008）；与价格 Provider 共用会话与超时策略。

    PIT 规则（spike §3 的泄漏结论，ADR-008 固化）：

    - ``date=`` 在早于可用起点时返回空名单 → ``UniverseError``，绝不用最新名单替代；
    - 未来日期会被服务端静默 clamp 到最新名单且 ``error_code=0``（``data.date`` 原样
      回显，无法据此识别），因此由本层拒绝 ``effective_date`` 晚于 cutoff 当日的调用；
    - ``update_date`` 晚于 ``effective_date`` 的快照不是「当时有效」的名单，显式失败。
    """

    def __init__(
        self,
        *,
        dataset_version: str = "baostock-v1",
        baostock_module: ModuleType | None = None,
        hard_timeout_seconds: float = BAOSTOCK_HARD_TIMEOUT_SECONDS,
    ) -> None:
        self._dataset_version = dataset_version
        super().__init__(baostock_module=baostock_module, hard_timeout_seconds=hard_timeout_seconds)

    def load(
        self,
        universe_id: str,
        effective_date: date,
        knowledge_cutoff: datetime,
    ) -> UniverseSnapshot:
        spec = _INDEX_SPECS.get(universe_id)
        if spec is None:
            known = ", ".join(sorted(_INDEX_SPECS))
            raise ConfigurationError(f"unknown universe_id {universe_id!r}; known: {known}")
        _require_plain_date(effective_date, "effective_date")
        if not isinstance(knowledge_cutoff, datetime):
            raise ConfigurationError(
                f"knowledge_cutoff must be a datetime, got {type(knowledge_cutoff).__name__}"
            )
        try:
            # cutoff 的日期部分必须按北京时间解读：非 +08:00 的偏移会让 .date() 漂移到
            # 另一天，从而放过「未来 universe」（§5 的 ResearchTime 同此约束）
            ensure_shanghai_aware(knowledge_cutoff, "knowledge_cutoff")
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
        if effective_date > knowledge_cutoff.date():
            # 服务端对未来日期静默返回最新名单（spike §3），这类调用必须死在调用侧
            raise ConfigurationError(
                f"effective_date {effective_date} is after knowledge_cutoff "
                f"{knowledge_cutoff.isoformat()}; a universe beyond the cutoff cannot be known"
            )

        rows = self._collect_constituents(spec, effective_date)
        if not rows:
            raise UniverseError(
                f"baostock has no {spec.universe_id} constituents at {effective_date}: the "
                f"constituent history does not reach back this far (spike bounds: empty at "
                f"{spec.last_empty_probe}, non-empty at {spec.first_nonempty_probe}; the exact "
                f"first available date is unknown) — do not substitute the current list"
            )
        symbols, update_date = _parse_constituents(rows, spec, effective_date)
        try:
            return UniverseSnapshot(
                universe_id=spec.universe_id,
                effective_date=effective_date,
                symbols=symbols,
                source=f"baostock:{spec.query_method}",
                version=self._dataset_version,
                update_date=update_date,
            )
        except ValidationError as exc:
            raise DataQualityError(
                f"baostock {spec.universe_id} constituents at {effective_date} violated the "
                f"snapshot contract: {exc}"
            ) from exc

    def _collect_constituents(self, spec: _IndexSpec, effective_date: date) -> list[dict[str, str]]:
        self._ensure_session()
        subject = f"{spec.universe_id}@{effective_date.isoformat()}"
        with self._session.lock:
            return _run_bounded(
                lambda: _run_query(
                    lambda: getattr(self._bs, spec.query_method)(date=effective_date.isoformat()),
                    label=spec.query_method,
                    subject=subject,
                    per_page=self._per_page_count,
                ),
                label=f"universe {subject}",
                timeout=self._hard_timeout,
                on_timeout=self._session.break_socket,
            )


class BaoStockTradingCalendarLoader(_BaostockSessionClient):
    """从 ``query_trade_dates`` 装载生产 TradingCalendar（§6.6；ADR-009）。

    只如实搬运服务端给出的 session 序列，不做工作日规则推导。

    已知边界（spike §1，artifact: checks.trade_dates）：

    - 覆盖 ``1990-12-19`` 起；**跨年不可用**：请求超出已发布范围时服务端 ``error_code=0``
      但返回 0 行（静默空）。装载器把空结果转为显式 :class:`ProviderError`，不返回空日历；
    - 因此年末 origin 且 horizon 跨年时，``next_sessions`` 会以覆盖不足显式失败，而不是
      被静默缩短——这正是 ADR-009 要求的语义。
    """

    def __init__(
        self,
        *,
        baostock_module: ModuleType | None = None,
        hard_timeout_seconds: float = BAOSTOCK_HARD_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__(
            baostock_module=baostock_module, hard_timeout_seconds=hard_timeout_seconds
        )

    def load(
        self,
        *,
        start: date,
        end: date,
        exchange: Exchange = "SSE",
    ) -> StaticTradingCalendar:
        _require_plain_date(start, "start")
        _require_plain_date(end, "end")
        if start > end:
            raise ConfigurationError(f"start {start} must be <= end {end}")
        rows = self._collect_trade_dates(start, end)
        if not rows:
            raise ProviderError(
                f"baostock query_trade_dates returned no rows for {start}..{end}: coverage "
                "does not reach this range (the endpoint silently returns empty beyond the "
                "published calendar, see ADR-009)"
            )
        sessions: list[date] = []
        seen: set[date] = set()
        for idx, row in enumerate(rows):
            try:
                day = date.fromisoformat(str(row["calendar_date"]))
                is_trading = str(row["is_trading_day"]).strip()
            except (KeyError, TypeError, ValueError) as exc:
                raise DataQualityError(
                    f"baostock returned malformed trade-date row #{idx}: {exc}"
                ) from exc
            if is_trading not in {"0", "1"}:
                raise DataQualityError(
                    f"baostock trade-date row #{idx} has unknown is_trading_day "
                    f"{is_trading!r}"
                )
            if day in seen:
                raise DataQualityError(
                    f"baostock trade-date response repeats calendar_date {day}"
                )
            seen.add(day)
            if is_trading == "1":
                sessions.append(day)
        if not sessions:
            raise ProviderError(
                f"baostock query_trade_dates returned no trading sessions for {start}..{end}"
            )
        return StaticTradingCalendar(
            exchange=exchange,
            source="baostock:query_trade_dates",
            sessions=tuple(sorted(sessions)),
        )

    def _collect_trade_dates(self, start: date, end: date) -> list[dict[str, str]]:
        self._ensure_session()
        subject = f"trade_dates {start}..{end}"
        with self._session.lock:
            return _run_bounded(
                lambda: _run_query(
                    lambda: self._bs.query_trade_dates(
                        start_date=start.isoformat(), end_date=end.isoformat()
                    ),
                    label="query_trade_dates",
                    subject=subject,
                    per_page=self._per_page_count,
                ),
                label=subject,
                timeout=self._hard_timeout,
                on_timeout=self._session.break_socket,
            )
