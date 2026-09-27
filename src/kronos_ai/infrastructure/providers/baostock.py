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
"""

from __future__ import annotations

import contextlib
import importlib
import socket
import threading
from collections.abc import Callable
from datetime import date, datetime, time, timedelta
from types import ModuleType
from typing import Any, cast

from pydantic import ValidationError

from kronos_ai.data.base import MarketDataProvider  # noqa: F401  (re-export contract)
from kronos_ai.domain.market import MarketBar, MarketHistory
from kronos_ai.domain.symbols import normalize_symbol
from kronos_ai.domain.time import CN_TZ
from kronos_ai.errors import (
    ConfigurationError,
    DataQualityError,
    InsufficientHistoryError,
    KronosAIError,
    ProviderError,
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


class BaoStockProvider:
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
        if hard_timeout_seconds <= 0:
            raise ConfigurationError(
                f"hard_timeout_seconds must be > 0, got {hard_timeout_seconds!r}"
            )
        self._adjust_flag = adjust_flag
        self._adjustment_mode = ADJUST_FLAG_TO_MODE[adjust_flag]
        self._dataset_version = dataset_version
        self._hard_timeout = hard_timeout_seconds
        self._bs = (
            baostock_module if baostock_module is not None else importlib.import_module("baostock")
        )
        self._per_page_count = _resolve_per_page_count(self._bs)
        self._session = _session_for(self._bs)
        self._session_acquired = False
        self._closed = False

    def __enter__(self) -> BaoStockProvider:
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
            raise ConfigurationError("BaoStockProvider is closed; create a new instance to query")
        if not self._session_acquired:
            self._session.acquire(self._hard_timeout)
            self._session_acquired = True
        else:
            self._session.ensure(self._hard_timeout)

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
                lambda: self._query_and_drain(code, start, end),
                label=f"query {code}",
                timeout=self._hard_timeout,
                on_timeout=self._session.break_socket,
            )

    def _query_and_drain(self, code: str, start: date, end: date) -> list[dict[str, str]]:
        try:
            result = self._bs.query_history_k_data_plus(
                code,
                K_LINE_FIELDS,
                start_date=start.isoformat(),
                end_date=end.isoformat(),
                frequency="d",
                adjustflag=self._adjust_flag,
            )
        except KronosAIError:
            raise
        except Exception as exc:  # 客户端本地异常（IndexError/JSONDecodeError 等）
            raise ProviderError(f"baostock query raised for {code}: {exc!r}") from exc
        if result is None:  # 入参被客户端拒绝时直接返回 None，不是错误对象
            raise ProviderError(f"baostock query returned None for {code}")
        if getattr(result, "error_code", "1") != "0":
            raise ProviderError(
                f"baostock query failed for {code}: {getattr(result, 'error_msg', 'unknown')}"
            )
        try:
            fields = list(getattr(result, "fields", ()))
            if not fields:
                raise ProviderError(f"baostock returned no field metadata for {code}")
            return self._drain_pages(result, code, fields)
        except KronosAIError:
            raise
        except Exception as exc:
            raise ProviderError(f"baostock paging failed for {code}: {exc!r}") from exc

    def _drain_pages(self, result: Any, code: str, fields: list[str]) -> list[dict[str, str]]:
        """耗尽 ``ResultData`` 游标并检测两类翻页失败（读 0.9.4 源码核实）。

        - 服务端错误码：``next()`` 先置 ``error_code`` 再返回 False，循环后复查可见；
        - ``send_msg`` 静默失败（socket 超时/对端断开/页号异常）：``next()`` 直接
          返回 False 且**不置 error_code**。其唯一痕迹是游标停在整页
          （``cur_row_num == len(data) == 单页行数``）——正常结束只有三种形态：
          空结果、非整页收尾、或整页后再取到空页（``data`` 清空）。因此
          「非空 + 恰好整页 + 游标到头」必然是截断，显式失败而非交给上层当短历史。
        """
        rows: list[dict[str, str]] = []
        while True:
            has_next = result.next()
            if not has_next:
                if getattr(result, "error_code", "0") != "0":
                    raise ProviderError(
                        f"baostock page request failed for {code}: "
                        f"{getattr(result, 'error_msg', 'unknown')}"
                    )
                if _page_state_is_truncated(result, self._per_page_count):
                    raise ProviderError(
                        f"baostock paging truncated for {code} at {len(rows)} rows: "
                        "a full page was followed by an empty response with no error code"
                    )
                return rows
            values = list(result.get_row_data())
            if len(values) != len(fields):
                raise DataQualityError(
                    f"baostock returned {len(values)} values for {len(fields)} fields for {code}"
                )
            rows.append(dict(zip(fields, values, strict=True)))

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
