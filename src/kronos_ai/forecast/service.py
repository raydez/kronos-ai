"""ForecastService：Forecast use case 协调器（基线文档 §17 / §34）。

职责只有编排：

```text
MarketDataProvider.get_history(lookback_bars)
        ↓ point-in-time MarketHistory
ForecastBackend.forecast(history, request)
        ↓
ForecastResult
```

不承担：数据源实现（§18）、推理细节（§17）、缓存策略（§15，在 backend 内）。
"""

from __future__ import annotations

from datetime import date, datetime

from kronos_ai.data.base import MarketDataProvider
from kronos_ai.domain.forecast import ForecastRequest, ForecastResult, SamplingConfig
from kronos_ai.errors import ConfigurationError
from kronos_ai.forecast.base import ForecastBackend


class ForecastService:
    """把 point-in-time 数据读取与 backend 推理串起来；同步、可注入。"""

    def __init__(
        self,
        provider: MarketDataProvider,
        backend: ForecastBackend,
        *,
        lookback_bars: int,
    ) -> None:
        if lookback_bars < 1:
            raise ConfigurationError(f"lookback_bars must be >= 1, got {lookback_bars}")
        self._provider = provider
        self._backend = backend
        self._lookback_bars = lookback_bars

    @property
    def backend(self) -> ForecastBackend:
        return self._backend

    @property
    def lookback_bars(self) -> int:
        return self._lookback_bars

    def close(self) -> None:
        """释放 provider 持有的进程级资源（如 BaoStock 会话引用计数）。

        无状态 provider 没有 ``close`` 时无操作；调用方（CLI / benchmark worker）
        应在结束一个 use case 后显式释放，避免长循环中泄漏会话。
        """
        close = getattr(self._provider, "close", None)
        if callable(close):
            close()

    def run(
        self,
        *,
        symbol: str,
        market_date: date,
        knowledge_cutoff: datetime,
        sampling: SamplingConfig,
        horizon: int = 5,
        force: bool = False,
    ) -> ForecastResult:
        request = ForecastRequest(
            symbol=symbol,
            market_date=market_date,
            knowledge_cutoff=knowledge_cutoff,
            horizon=horizon,
            sampling=sampling,
        )
        history = self._provider.get_history(
            symbol=request.symbol,
            market_date=request.market_date,
            knowledge_cutoff=request.knowledge_cutoff,
            lookback_bars=self._lookback_bars,
        )
        return self._backend.forecast(history, request, force=force)
