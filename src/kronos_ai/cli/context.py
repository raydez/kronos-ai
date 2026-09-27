"""CLI 运行时装配：把命令行参数解析成可执行组件（§32.1 / §34）。

重依赖（baostock / torch / huggingface_hub）全部在函数内延迟 import：``kronos-ai --help``
与纯参数解析不应触发模型运行时加载。

装配结果是可注入的：:func:`build_forecast_service` 是默认工厂，测试与集成脚本可传入
自己的工厂构造合成 provider / backend，无需网络与权重。benchmark 命令同理，默认装配是
:func:`build_benchmark_context`，测试可整体替换 :class:`BenchmarkContext`。
"""

from __future__ import annotations

import os
from argparse import Namespace
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from kronos_ai.cli.format import calendar_span
from kronos_ai.config import ExperimentConfig
from kronos_ai.data.adjustment import AdjustmentMode
from kronos_ai.data.base import MarketDataProvider
from kronos_ai.domain.time import CN_TZ
from kronos_ai.errors import ConfigurationError
from kronos_ai.evaluation.benchmark import LabelDataProvider
from kronos_ai.evaluation.dataset import WalkForwardDataset
from kronos_ai.forecast.base import ForecastBackend
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


def resolve_data_coverage(
    sessions: Sequence[date],
    *,
    today: date,
    end_session: date,
    horizon_sessions: int,
) -> date:
    """已发布数据的覆盖末端；尾部不足以给最后一个 origin 打分时显式失败（§41 / ADR-023 §5）。

    ``sessions`` 是**交易日历装载到的全部 session**（含 ``today`` 之后已发布的未来 session：
    交易所日历是已知的，而 provider 只发布到 ``today``）。两件事必须分开：

    - **数据覆盖末端** = ``sessions`` 中 ``<= today`` 的最后一个 session（provider 已发布到哪）
    - **窗口末端** = ``dataset.end_session``（最后一个 origin，其 label 落在窗口之外）

    把覆盖末端取成窗口末端，最后一个 horizon 的 origin 会被全部判成
    ``INSUFFICIENT_FUTURE_BARS``——那不是「数据没有」，而是装配把自己的窗口当成了数据的边界。
    因此本函数要求覆盖末端**至少覆盖到最后一个 origin 的最后一个 label session**：一次正式
    run 的每个 considered origin 都要能被评分，否则就是拿一份尾部缺 label 的报告去比模型。

    ``end_session`` 必须是真实的 market session：dataset 的窗口是
    ``[start_session, end_session]`` 里 session 的切片，非 session 的上界会把**实际窗口**
    悄悄提前到更早的一个 session，而 config 仍写着它当初写下的那个日期——这正是
    「配置说的窗口与跑的窗口不是同一个」的静默版本。这条校验是**特意补回**的：旧实现在这里
    调用 ``calendar.next_sessions(end_session, horizon)``，顺带继承了它的「非 session 即报错」；
    改写成纯函数后若不显式写出来，这份严格度会静默消失（见 ADR-023 §7.1）。

    抽成纯函数（只依赖 session 序列 / today / end_session / horizon）是为了让数据覆盖相关的
    四条分支（+ 两条输入校验分支）都能用合成日历直接测，而不必起 BaoStock。
    """
    if horizon_sessions < 1:
        raise ConfigurationError(f"horizon_sessions must be >= 1, got {horizon_sessions}")
    if not sessions:
        raise ConfigurationError("calendar has no sessions; cannot resolve data coverage")
    if end_session not in sessions:
        if end_session > sessions[-1]:
            raise ConfigurationError(
                f"dataset.end_session {end_session} lies beyond the loaded calendar coverage "
                f"(which ends at {sessions[-1]}): load the window the config describes, or move "
                "end_session back inside the calendar"
            )
        if end_session < sessions[0]:
            raise ConfigurationError(
                f"dataset.end_session {end_session} precedes the loaded calendar coverage "
                f"(which starts at {sessions[0]}): the window is empty on this calendar"
            )
        raise ConfigurationError(
            f"dataset.end_session {end_session} is not a market session in the loaded calendar "
            "(window bound must be a session): a non-session bound shifts the effective window "
            "to an earlier session while the config still claims the date it wrote"
        )
    published = tuple(day for day in sessions if day <= today)
    if not published:
        raise ConfigurationError(
            f"calendar coverage starts at {sessions[0]}, after today ({today}): no published "
            "session is available, so neither histories nor labels can be sourced"
        )
    data_coverage_end = published[-1]
    future = tuple(day for day in sessions if day > end_session)
    if len(future) < horizon_sessions:
        raise ConfigurationError(
            f"the loaded calendar holds only {len(future)} session(s) after the window bound "
            f"({end_session}) — fewer than the {horizon_sessions} label session(s) the last "
            "origin needs; move dataset.end_session back so the label window fits inside the "
            "calendar coverage"
        )
    required_coverage_end = future[horizon_sessions - 1]
    if data_coverage_end < required_coverage_end:
        raise ConfigurationError(
            f"published data coverage ends at {data_coverage_end}, but the last origin "
            f"({end_session}) needs labels through {required_coverage_end}; move "
            "dataset.end_session back so every considered origin can be scored "
            "(never run a formal benchmark whose tail has no labels)"
        )
    return data_coverage_end


@dataclass(frozen=True)
class BenchmarkContext:
    """一次 benchmark run 所需的全部装配结果（可被测试整体替换）。

    把装配结果做成一个值而不是散落的全局对象，使 ``kronos-ai benchmark forecast`` 的编排
    逻辑可被注入式测试：测试提供合成 dataset / provider / baseline backend，走同一条 handler。

    ``backend_identities`` 是每个 backend 的 §15 artifact identity（baseline 是公式版本，
    Kronos 是 model_id / revision / device / dtype / config_hash）；它进 run metadata，
    因此「这次比的是哪几个具体模型」可被追溯，而不是只留一个 backend 名（§32）。
    """

    dataset: WalkForwardDataset
    provider: MarketDataProvider
    label_provider: LabelDataProvider
    backends: dict[str, ForecastBackend]
    backend_identities: dict[str, dict[str, str]]


def build_benchmark_context(config: ExperimentConfig, args: Namespace) -> BenchmarkContext:
    """默认装配：BaoStock 日历 / provider / label provider + 配置声明的 backends。

    时间轴只来自 TradingCalendar（§6.6）：先按 ``[start_session, end_session]`` 装载日历，
    再切片；切片长度必须**恰好**等于 §27 要求的 session 数，否则拒绝运行——静默截断会让
    dataset_hash 与报告里的样本量对不上。

    **数据覆盖末端与窗口末端是两件事**：窗口的最后一个 session 是最后一个 origin，它的
    label 落在窗口之外；若把覆盖末端取成窗口末端，最后一个 horizon 的 origin 会被全部判成
    ``INSUFFICIENT_FUTURE_BARS``——那不是「数据没有」，而是装配把自己的窗口当成了数据的边界。
    因此日历装载到「今天」（CN 时区）为止，覆盖末端取其中已发布（``<= today``）的最后一个
    session，并要求它**至少覆盖到窗口最后一个 origin 的最后一个 label session**：一次正式
    run 的每个 considered origin 都要能被评分，否则就是拿一份尾部缺 label 的报告去比模型。
    覆盖末端进 metadata 与 ``metrics.json``（数据环境事实，不进 ``report_hash``，理由见
    ADR-023 §5）。这段判定本身在 :func:`resolve_data_coverage` 里，三条失败分支都可离线单测。

    backend 装配是**按名惰性**的：只比较 baseline 的 run 不会加载 Kronos 权重
    （ADR-014 §8 的 defer 项），而声明了 ``kronos`` 的 run 必须真的拿到 Kronos runtime，
    未知名显式失败。
    """
    from kronos_ai.evaluation.baselines import (
        BASELINE_NAMES,
        build_baseline,
    )
    from kronos_ai.evaluation.benchmark import LabelDataProvider  # noqa: F401  (契约注释)
    from kronos_ai.evaluation.dataset import build_walk_forward_dataset
    from kronos_ai.forecast.backends.kronos.backend import KronosForecastBackend
    from kronos_ai.forecast.backends.kronos.runtime import (
        KronosRuntime,
        KronosRuntimeConfig,
    )
    from kronos_ai.forecast.cache import FileSystemForecastCache
    from kronos_ai.infrastructure.providers.baostock import (
        BaoStockLabelBarProvider,
        BaoStockProvider,
        BaoStockTradingCalendarLoader,
    )

    policy = config.label_policy()
    lookback_bars = config.forecast.lookback_bars
    adjust_flag = ADJUST_MODE_TO_FLAG[config.data.adjustment]
    today = datetime.now(CN_TZ).date()
    loader = BaoStockTradingCalendarLoader()
    try:
        calendar = loader.load(
            start=config.dataset.start_session,
            end=max(config.dataset.end_session, today),
            exchange="SSE",
        )
    finally:
        loader.close()
    # 覆盖末端与「最后一个 origin 的 label 窗口必须已存在」的判定是纯函数（可单测）
    data_coverage_end = resolve_data_coverage(
        calendar.sessions,
        today=today,
        end_session=config.dataset.end_session,
        horizon_sessions=policy.horizon_sessions,
    )
    sessions = tuple(
        day
        for day in calendar.sessions
        if config.dataset.start_session <= day <= config.dataset.end_session
    )
    required = config.required_sessions()
    if len(sessions) != required:
        raise ConfigurationError(
            f"calendar slice {config.dataset.start_session}..{config.dataset.end_session} has "
            f"{len(sessions)} sessions but the configured plan requires exactly {required} "
            "(lookback_bars + Σsegments + embargo × gaps); fix the window or the plan, "
            "never truncate silently"
        )
    dataset = build_walk_forward_dataset(
        calendar=calendar,
        sessions=sessions,
        symbols=config.dataset.symbols,
        plan=config.dataset.to_plan(),
        label_policy=policy,
        lookback_bars=lookback_bars,
        cutoff_policy=config.knowledge_cutoff_policy,
    )
    provider = BaoStockProvider(adjust_flag=adjust_flag)
    label_provider = BaoStockLabelBarProvider(
        data_coverage_end=data_coverage_end, adjust_flag=adjust_flag
    )
    cache = FileSystemForecastCache(args.cache_dir) if args.cache_dir else None

    backends: dict[str, ForecastBackend] = {}
    identities: dict[str, dict[str, str]] = {}
    runtime: KronosRuntime | None = None
    for name in config.benchmark.backends:
        if name in BASELINE_NAMES:
            baseline = build_baseline(
                name, calendar=calendar, lookback_bars=lookback_bars, cache=cache
            )
            backends[name] = baseline
            identities[name] = dict(baseline.identity())
            continue
        if name == config.forecast.backend:
            if runtime is None:
                kwargs: dict[str, object] = {
                    "device": config.runtime.device,
                    "dtype": config.runtime.dtype,
                    "lookback_bars": lookback_bars,
                }
                if config.forecast.model is not None:
                    kwargs["model_id"] = config.forecast.model
                runtime = KronosRuntime.load(KronosRuntimeConfig(**kwargs))  # type: ignore[arg-type]
            backends[name] = KronosForecastBackend(runtime, calendar=calendar, cache=cache)
            identities[name] = dict(runtime.artifact_identity())
            continue
        raise ConfigurationError(
            f"benchmark backend {name!r} is neither a known baseline {list(BASELINE_NAMES)} "
            f"nor the configured forecast backend {config.forecast.backend!r}"
        )
    return BenchmarkContext(
        dataset=dataset,
        provider=provider,
        label_provider=label_provider,
        backends=backends,
        backend_identities=identities,
    )
