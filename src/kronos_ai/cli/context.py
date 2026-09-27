"""CLI 运行时装配：把命令行参数解析成可执行组件（§32.1 / §34）。

重依赖（baostock / torch / huggingface_hub）全部在函数内延迟 import：``kronos-ai --help``
与纯参数解析不应触发模型运行时加载。

装配结果是可注入的：:func:`build_forecast_service` 是默认工厂，测试与集成脚本可传入
自己的工厂构造合成 provider / backend，无需网络与权重。
"""

from __future__ import annotations

import os
from argparse import Namespace
from datetime import date, datetime
from pathlib import Path

from kronos_ai.cli.format import calendar_span
from kronos_ai.data.adjustment import AdjustmentMode
from kronos_ai.forecast.service import ForecastService
from kronos_ai.registry import RuntimeRegistry

# BaoStock adjustflag：3=不复权，1=后复权，2=前复权（RX-KAI-004/006）
ADJUST_MODE_TO_FLAG: dict[AdjustmentMode, str] = {"raw": "3", "hfq": "1", "qfq": "2"}

DEFAULT_ARTIFACTS_DIR = Path("artifacts")
INDEX_DB_FILENAME = "index.sqlite3"


def default_artifacts_dir() -> Path:
    """artifact root（§33）；``KRONOS_AI_ARTIFACTS_DIR`` 可覆盖。"""
    return Path(os.environ.get("KRONOS_AI_ARTIFACTS_DIR", str(DEFAULT_ARTIFACTS_DIR)))


def index_db_path_for(artifacts_dir: Path) -> Path:
    """§30 的 SQLite run registry + artifact index 默认位置。"""
    return artifacts_dir / INDEX_DB_FILENAME


def build_forecast_service(args: Namespace, cutoff: datetime) -> ForecastService:
    """默认装配：BaoStock 日历 + BaoStock provider + Kronos runtime + 可选文件缓存。

    ``cutoff`` 由调用方（handler）先按 §5 policy 解析并传入，避免 cutoff 在两处
    各推导一次而产生分叉。
    """
    from kronos_ai.forecast.backends.kronos.backend import KronosForecastBackend
    from kronos_ai.forecast.backends.kronos.runtime import (
        KronosRuntime,
        KronosRuntimeConfig,
    )
    from kronos_ai.forecast.cache import FileSystemForecastCache
    from kronos_ai.infrastructure.providers.baostock import (
        BaoStockProvider,
        BaoStockTradingCalendarLoader,
    )

    market_date = date.fromisoformat(args.market_date)
    start, end = calendar_span(market_date, args.horizon)
    calendar_loader = BaoStockTradingCalendarLoader()
    try:
        calendar = calendar_loader.load(start=start, end=end, exchange="SSE")
    finally:
        calendar_loader.close()

    provider = BaoStockProvider(adjust_flag=ADJUST_MODE_TO_FLAG[args.adjust])
    runtime_kwargs: dict[str, object] = {"device": args.device, "dtype": args.dtype}
    if args.lookback_bars is not None:
        runtime_kwargs["lookback_bars"] = args.lookback_bars
    runtime = KronosRuntime.load(KronosRuntimeConfig(**runtime_kwargs))  # type: ignore[arg-type]

    cache = FileSystemForecastCache(args.cache_dir) if args.cache_dir else None
    backend = KronosForecastBackend(runtime, calendar=calendar, cache=cache)
    # §17：装配在 composition root；registry 只做名字 → 实例解析，未注册名字显式失败
    registry = RuntimeRegistry()
    registry.register_forecast_backend(backend)
    selected = registry.get_forecast_backend(getattr(args, "backend", backend.name))
    return ForecastService(provider, selected, lookback_bars=runtime.lookback_bars)
