"""MarketDataProvider Protocol（基线文档 §18）。

Core 同步接口；CLI / worker 直接调用，Web 编排层如需异步在外层用 executor 调度，
不要求 Provider 变成 async。
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Protocol, runtime_checkable

from kronos_ai.domain.market import MarketHistory


@runtime_checkable
class MarketDataProvider(Protocol):
    def get_history(
        self,
        symbol: str,
        market_date: date,
        knowledge_cutoff: datetime,
        lookback_bars: int,
    ) -> MarketHistory:
        """返回截至 knowledge_cutoff 可见的最近行情窗口。

        - symbol: 规范 6 位代码；
        - market_date: 研究对象交易日（origin）；
        - knowledge_cutoff: PIT 截止时刻，晚于它的 bar 不得出现；
        - lookback_bars: 请求的有效 bar 数量上限，实际可少于该值（新股）。

        数据源失败必须显式抛错（ProviderError），禁止任何 synthetic fallback。
        """
        ...
