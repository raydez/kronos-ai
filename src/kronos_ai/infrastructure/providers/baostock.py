"""Explicit-failure BaoStock Provider（RX-KAI-004，基线文档 §18 / ADR-010）。

v1 的 5 处 `_generate_fallback_data` 静默 fallback 全部不迁移：
任何数据源失败都抛 ProviderError，由上层显式处理。

语义约定：
- 输入/输出 symbol 均为规范 6 位代码；`sh.`/`sz.`/`bj.` 前缀仅在适配层出现；
- BaoStock 日线 volume 单位为股、amount 单位为元，与 domain 约定一致，无需换算；
- available_at 采用保守常量 BAO_STOCK_PUBLISHED_AT（当日 18:00+08:00），
  该时刻须由 BaoStock 能力 spike（RX-KAI-006）核实后固化为版本化常量；
- 停牌日（tradestatus=0）与无效价格行不是 valid bar，不进入 MarketHistory，
  原始行级数据由 append-only raw snapshot（RX-KAI-015）负责留痕。

会话说明：baostock 的登录态是进程级全局单会话，本 Provider 非线程安全，
并发 worker 必须各自持有一个实例。
"""

from __future__ import annotations

import importlib
import logging
from datetime import date, datetime, time, timedelta
from types import ModuleType
from typing import Any, cast

import pandas as pd

from kronos_ai.data.base import MarketDataProvider  # noqa: F401  (re-export contract)
from kronos_ai.domain.market import (
    MarketBar,
    MarketHistory,
    compute_market_history_hash,
)
from kronos_ai.domain.symbols import normalize_symbol
from kronos_ai.domain.time import SHANGHAI
from kronos_ai.errors import DataQualityError, ProviderError

logger = logging.getLogger(__name__)

BAO_STOCK_PUBLISHED_AT = time(18, 0)

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

_A_SHARE_FIRST_TRADE_DAY = date(1990, 12, 19)


def to_baostock_code(symbol: str) -> str:
    """规范 6 位代码 → BaoStock 前缀代码；未知板块显式失败而非猜测。"""
    if symbol.startswith(("600", "601", "603", "605", "688", "689", "900")):
        return f"sh.{symbol}"
    if symbol.startswith(("000", "001", "002", "003", "200", "300", "301", "302")):
        return f"sz.{symbol}"
    if symbol.startswith(("43", "83", "87", "92")):
        return f"bj.{symbol}"
    raise ProviderError(f"cannot map symbol {symbol!r} to a BaoStock exchange prefix")


class BaoStockProvider:
    def __init__(
        self,
        *,
        adjust_flag: str = "3",
        dataset_version: str = "baostock-v1",
        baostock_module: ModuleType | None = None,
    ) -> None:
        if adjust_flag not in ADJUST_FLAG_TO_MODE:
            raise ProviderError(f"unknown adjust_flag: {adjust_flag!r}")
        self._adjust_flag = adjust_flag
        self._adjustment_mode = ADJUST_FLAG_TO_MODE[adjust_flag]
        self._dataset_version = dataset_version
        self._bs = (
            baostock_module if baostock_module is not None else importlib.import_module("baostock")
        )
        self._logged_in = False

    def __enter__(self) -> BaoStockProvider:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self._logged_in:
            self._bs.logout()
            self._logged_in = False

    def _ensure_logged_in(self) -> None:
        if self._logged_in:
            return
        result = self._bs.login()
        if getattr(result, "error_code", "1") != "0":
            raise ProviderError(f"baostock login failed: {getattr(result, 'error_msg', 'unknown')}")
        self._logged_in = True

    def get_history(
        self,
        symbol: str,
        market_date: date,
        knowledge_cutoff: datetime,
        lookback_bars: int,
    ) -> MarketHistory:
        symbol = normalize_symbol(symbol)
        if lookback_bars < 1:
            raise DataQualityError(f"lookback_bars must be >= 1, got {lookback_bars}")

        code = to_baostock_code(symbol)
        bars = self._collect_bars(code, symbol, market_date, knowledge_cutoff, lookback_bars)

        return MarketHistory(
            symbol=symbol,
            market_date=market_date,
            knowledge_cutoff=knowledge_cutoff,
            bars=bars,
            provider="baostock",
            dataset_version=self._dataset_version,
            data_hash=compute_market_history_hash(
                symbol=symbol,
                market_date=market_date,
                knowledge_cutoff=knowledge_cutoff,
                provider="baostock",
                dataset_version=self._dataset_version,
                bars=bars,
            ),
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
        for _ in range(4):
            start = market_date - timedelta(days=span_days)
            frame = self._fetch_window(code, start, market_date)
            bars = self._parse_bars(frame, symbol, knowledge_cutoff)
            if len(bars) >= lookback_bars:
                return bars[-lookback_bars:]
            span_days *= 2
            if start <= _A_SHARE_FIRST_TRADE_DAY:
                break
        if not bars:
            raise ProviderError(f"baostock returned no data for {code}")
        return bars

    def _fetch_window(self, code: str, start: date, end: date) -> pd.DataFrame:
        self._ensure_logged_in()
        result = self._bs.query_history_k_data_plus(
            code,
            K_LINE_FIELDS,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            frequency="d",
            adjustflag=self._adjust_flag,
        )
        if result is None:
            raise ProviderError(f"baostock query returned None for {code}")
        if getattr(result, "error_code", "1") != "0":
            raise ProviderError(
                f"baostock query failed for {code}: {getattr(result, 'error_msg', 'unknown')}"
            )
        return cast(pd.DataFrame, result.get_data())

    def _parse_bars(
        self, frame: pd.DataFrame, symbol: str, knowledge_cutoff: datetime
    ) -> list[MarketBar]:
        if frame is None or frame.empty:
            return []

        bars: list[MarketBar] = []
        for row in frame.to_dict("records"):
            status = TRADE_STATUS_MAP.get(str(row.get("tradestatus", "")).strip())
            try:
                close = float(row["close"])
                bar_day = date.fromisoformat(str(row["date"]))
            except (KeyError, TypeError, ValueError):
                raise DataQualityError(
                    f"baostock returned malformed row for {symbol}: {row!r}"
                ) from None

            timestamp = datetime.combine(bar_day, time(15, 0), tzinfo=SHANGHAI)
            available_at = datetime.combine(bar_day, BAO_STOCK_PUBLISHED_AT, tzinfo=SHANGHAI)

            if status == "suspended" or close <= 0:
                continue
            if available_at > knowledge_cutoff:
                continue

            volume = self._optional_float(row.get("volume"))
            amount = self._optional_float(row.get("amount"))
            bars.append(
                MarketBar(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=close,
                    volume=volume,
                    amount=amount,
                    trade_status=status,
                    adjustment_mode=self._adjustment_mode,
                    available_at=available_at,
                )
            )
        return bars

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None
