"""Forecast Benchmark v1 端到端回归（RX-KAI-019，基线文档 §41 / §48 / §49 / §33）。

本文件证明 benchmark runner 的**编排**是对的，且不依赖网络 / 模型权重：

```text
合成 TradingCalendar → WalkForwardDataset（§27，含 embargo）
        + 合成 PIT provider（§18，只给到 origin 当日）
        + 显式 label bar（含一个停牌 origin）
        + naive baseline（§19，纯 CPU）
        ↓
ForecastEvalRecord × backends → §49 指标 → report.md / metrics.json / forecast.parquet
```

断言覆盖：记录数 = origin × backend、指标确实被算出来（且停牌 origin 不进指标）、
pilot 截断被显式记录、未知 backend 显式失败、同输入两次 run 的 ``report_hash`` 相同、
§33 的 artifact 目录真的落盘并被 SQLite artifact index 登记。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from kronos_ai.data.calendar import StaticTradingCalendar
from kronos_ai.domain.forecast import SamplingConfig
from kronos_ai.domain.hashing import is_sha256_hex
from kronos_ai.domain.market import MarketBar, MarketHistory
from kronos_ai.domain.run import RUN_STATUSES
from kronos_ai.domain.time import CN_TZ, MARKET_SESSION_CLOSE
from kronos_ai.errors import ConfigurationError, InsufficientHistoryError
from kronos_ai.evaluation.baselines import build_baseline
from kronos_ai.evaluation.benchmark import (
    BENCHMARK_VERSION,
    ConstantLabelProvider,
    ForecastBenchmarkResult,
    ForecastBenchmarkSpec,
    build_benchmark_run_metadata,
    evaluated_origins_by_symbol,
    run_forecast_benchmark,
)
from kronos_ai.evaluation.dataset import (
    DEFAULT_LABEL_POLICY,
    WalkForwardDataset,
    build_walk_forward_dataset,
)
from kronos_ai.evaluation.regimes import RegimeSpec, regime_spec_hash
from kronos_ai.evaluation.report import (
    metrics_payload,
    render_markdown_report,
    write_forecast_benchmark_artifacts,
)
from kronos_ai.evaluation.walk_forward import SegmentSpec, WalkForwardPlan
from kronos_ai.infrastructure.persistence.artifact_store import ArtifactStore
from kronos_ai.infrastructure.persistence.run_registry import RunRecord, RunRegistry
from kronos_ai.infrastructure.persistence.schema import open_database

pytestmark = pytest.mark.regression

SYMBOL_A = "600000"
SYMBOL_B = "000001"
LOOKBACK = 6
HORIZON = DEFAULT_LABEL_POLICY.horizon_sessions
EMBARGO = DEFAULT_LABEL_POLICY.embargo_sessions
SEGMENTS = (
    SegmentSpec(name="train", length_sessions=4),
    SegmentSpec(name="test", length_sessions=3),
)
SEGMENT_SESSIONS = sum(segment.length_sessions for segment in SEGMENTS)
TOTAL_SESSIONS = LOOKBACK + SEGMENT_SESSIONS + EMBARGO
BACKENDS = ("last_value", "drift")

#: 本 fixture 的 lookback（6）小于 §49 默认 regime 窗口（20）。fixture 用短窗口是**为了
#: 让回归跑得快**，不是为了放松口径：窗口仍是显式的 RegimeSpec，由 runner 拒绝任何
#: 「lookback < regime 窗口」的组合（见 TestRunner.test_regime_spec_wider_than_lookback_...），
#: 因此「静默用更短窗口」不可能发生。
REGIME = RegimeSpec(trend_window_sessions=5, volatility_window_sessions=5)

SAMPLING = SamplingConfig(seed=11, sample_count=4)


def weekdays(start: date, end: date) -> tuple[date, ...]:
    days: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return tuple(days)


CALENDAR_SESSIONS = weekdays(date(2026, 6, 1), date(2027, 6, 30))


def calendar() -> StaticTradingCalendar:
    return StaticTradingCalendar(exchange="SSE", source="test-fixture", sessions=CALENDAR_SESSIONS)


def session_close(day: date) -> datetime:
    return datetime.combine(day, MARKET_SESSION_CLOSE, tzinfo=CN_TZ)


def make_bar(symbol: str, day: date, close: float, previous: float | None = None) -> MarketBar:
    open_ = close if previous is None else previous
    return MarketBar(
        symbol=symbol,
        timestamp=session_close(day),
        open=open_,
        high=max(open_, close) + 0.05,
        low=min(open_, close) - 0.05,
        close=close,
        volume=1_000_000.0,
        amount=10_000_000.0,
        trade_status="1",
        adjustment_mode="raw",
        available_at=session_close(day),
    )


def close_for(symbol: str, index: int) -> float:
    """确定性合成收盘价：两个 symbol 斜率不同，使 baseline 表现可区分。"""
    slope = 0.05 if symbol == SYMBOL_A else -0.03
    return 20.0 + slope * index


def sessions_upto(day: date) -> tuple[date, ...]:
    return tuple(item for item in CALENDAR_SESSIONS if item <= day)


class SyntheticProvider:
    """PIT provider：只为 ``(symbol, market_date)`` 提供到该日为止的窗口。

    两个结构性守卫（都是 §29 的泄漏检查）：

    - ``knowledge_cutoff`` 必须落在 ``market_date`` 当日——runner 若把「未来 cutoff」
      传进来，这里立刻失败；
    - 返回的 bar 序列以 ``market_date`` 结尾，绝不含之后的 session。
    """

    def __init__(self) -> None:
        self.requests: list[tuple[str, date]] = []

    def get_history(
        self, symbol: str, market_date: date, knowledge_cutoff: datetime, lookback_bars: int
    ) -> MarketHistory:
        if knowledge_cutoff.date() != market_date:
            raise ConfigurationError(
                "synthetic PIT provider received a knowledge_cutoff outside market_date: "
                f"{knowledge_cutoff.isoformat()} vs {market_date}"
            )
        self.requests.append((symbol, market_date))
        history_days = sessions_upto(market_date)[-lookback_bars:]
        if len(history_days) != lookback_bars:
            raise ConfigurationError("test fixture calendar too short for lookback_bars")
        bars: list[MarketBar] = []
        previous: float | None = None
        for index, day in enumerate(history_days):
            close = close_for(symbol, index)
            bars.append(make_bar(symbol, day, close, previous))
            previous = close
        if bars[-1].timestamp.date() > market_date:
            raise ConfigurationError("synthetic PIT provider produced a future bar")
        return MarketHistory(
            symbol=symbol,
            market_date=market_date,
            knowledge_cutoff=knowledge_cutoff,
            bars=tuple(bars),
            provider="synthetic",
            dataset_version="synthetic-v1",
        )


def build_dataset(*, sessions: tuple[date, ...] | None = None) -> WalkForwardDataset:
    timeline = CALENDAR_SESSIONS[:TOTAL_SESSIONS] if sessions is None else sessions
    return build_walk_forward_dataset(
        calendar=calendar(),
        sessions=timeline,
        symbols=(SYMBOL_A, SYMBOL_B),
        plan=WalkForwardPlan(segments=SEGMENTS),
        label_policy=DEFAULT_LABEL_POLICY,
        lookback_bars=LOOKBACK,
    )


def origins(dataset: WalkForwardDataset):
    return [origin for segment in dataset.segments for origin in segment.origins]


def build_labels(
    dataset: WalkForwardDataset,
    *,
    suspended: tuple[str, date] | None = None,
    drop_last: int = 0,
    coverage_end: date | None = None,
) -> ConstantLabelProvider:
    """为每个 origin 造 label bar。

    ``drop_last`` 少给最后若干个 session 的 bar（模拟停牌 / 数据末端），``suspended``
    指定一个 origin 在窗口中途缺 bar，``coverage_end`` 用于把数据末端提前。
    """
    bars: dict[tuple[str, date], tuple[MarketBar, ...]] = {}
    for origin in origins(dataset):
        entries: list[MarketBar] = []
        previous: float | None = None
        window = list(origin.label_sessions)
        if drop_last:
            window = window[:-drop_last]
        for step, day in enumerate(origin.label_sessions):
            if day not in window:
                continue
            if (origin.symbol, origin.market_date) == suspended and step == 1:
                previous = None
                continue
            close = close_for(origin.symbol, 100 + step)
            entries.append(make_bar(origin.symbol, day, close, previous))
            previous = close
        bars[(origin.symbol, origin.market_date)] = tuple(entries)
    resolved_end = coverage_end or CALENDAR_SESSIONS[TOTAL_SESSIONS + HORIZON + 5]
    return ConstantLabelProvider(bars=bars, data_coverage_end=resolved_end)


def backends_for(dataset: WalkForwardDataset):
    return {
        name: build_baseline(name, calendar=calendar(), lookback_bars=LOOKBACK) for name in BACKENDS
    }


def run(
    dataset: WalkForwardDataset | None = None,
    *,
    spec: ForecastBenchmarkSpec | None = None,
    labels: ConstantLabelProvider | None = None,
    provider: SyntheticProvider | None = None,
    backends: dict[str, object] | None = None,
    clock: Callable[[], float] | None = None,
):
    resolved_dataset = dataset or build_dataset()
    return run_forecast_benchmark(
        dataset=resolved_dataset,
        provider=provider or SyntheticProvider(),
        label_provider=labels or build_labels(resolved_dataset),
        backends=backends or backends_for(resolved_dataset),
        spec=spec or ForecastBenchmarkSpec(backends=BACKENDS, regime=REGIME),
        sampling=SAMPLING,
        clock=clock or (lambda: 0.0),
    )


class TestRunner:
    def test_records_cover_every_origin_and_backend(self) -> None:
        dataset = build_dataset()
        expected_origins = len(origins(dataset))
        result = run(dataset)
        assert result.evaluated_origins == expected_origins
        assert result.considered_origins == expected_origins
        assert result.truncated is False
        assert len(result.records) == expected_origins * len(BACKENDS)
        assert {record.backend for record in result.records} == set(BACKENDS)
        for record in result.records:
            assert is_sha256_hex(record.artifact_id)
            assert record.latency_ms >= 0

    def test_metrics_are_produced_for_every_backend(self) -> None:
        result = run()
        for backend in BACKENDS:
            entry = result.metrics_for(backend, group="all")
            assert entry is not None
            assert entry.sample_count == result.evaluated_origins
            assert entry.labeled_count > 0
            assert entry.mae is not None
            assert entry.rmse is not None
            assert entry.direction_accuracy is not None
            assert entry.quantile_coverage is not None
            assert entry.coverage_nominal == pytest.approx(0.9)

    def test_regime_slices_are_present(self) -> None:
        result = run()
        axes = {entry.group.axis for entry in result.metrics}
        assert axes == {"overall", "trend", "volatility"}
        assert result.regime_spec_hash == regime_spec_hash(REGIME)

    def test_regime_spec_wider_than_lookback_is_explicit_failure(self) -> None:
        """lookback 短于 regime 窗口时必须拒绝，不能「安静地用更短窗口」分类。"""
        with pytest.raises(ConfigurationError, match="regime spec needs at least"):
            run(spec=ForecastBenchmarkSpec(backends=BACKENDS))

    def test_suspended_origin_is_counted_but_not_scored(self) -> None:
        dataset = build_dataset()
        first_test_origin = dataset.segments[1].origins[0]
        target = (first_test_origin.symbol, first_test_origin.market_date)
        result = run(dataset, labels=build_labels(dataset, suspended=target))
        statuses = {record.label_status for record in result.records}
        assert "SUSPENDED" in statuses
        last_value = result.metrics_for("last_value", group="all")
        assert last_value is not None
        assert last_value.labeled_count < last_value.sample_count

    def test_origin_limit_marks_pilot_subset(self) -> None:
        result = run(spec=ForecastBenchmarkSpec(backends=BACKENDS, origin_limit=3, regime=REGIME))
        assert result.evaluated_origins == 3
        assert result.truncated is True
        assert result.considered_origins > result.evaluated_origins
        assert len(result.records) == 3 * len(BACKENDS)

    def test_unknown_backend_is_explicit_failure(self) -> None:
        dataset = build_dataset()
        with pytest.raises(ConfigurationError, match="not provided"):
            run_forecast_benchmark(
                dataset=dataset,
                provider=SyntheticProvider(),
                label_provider=build_labels(dataset),
                backends=backends_for(dataset),
                spec=ForecastBenchmarkSpec(backends=("last_value", "kronos"), regime=REGIME),
                sampling=SAMPLING,
                clock=lambda: 0.0,
            )

    def test_report_hash_is_deterministic(self) -> None:
        first = run()
        second = run()
        assert first.report_hash == second.report_hash
        assert is_sha256_hex(first.report_hash)
        assert first.version == BENCHMARK_VERSION

    def test_report_hash_ignores_wall_clock_latency(self) -> None:
        """真实时钟下的两次 run 必须给出同一个 report_hash（延迟不进结论 hash）。"""
        ticks = iter(range(0, 10_000))

        def clock() -> float:
            return next(ticks) / 1000.0

        first = run(clock=clock)
        second = run(clock=clock)
        assert first.report_hash == second.report_hash
        # 延迟确实被记录、且两次不同（否则上面的相等是「延迟恒为 0」造出来的假象）
        assert [record.latency_ms for record in first.records] != [
            record.latency_ms for record in second.records
        ]
        assert {record.latency_ms for record in first.records} != {0.0}

    def test_backend_result_for_another_origin_is_explicit_failure(self) -> None:
        """backend 交出别的 origin 的预测时必须失败，而不是把两条证据拼成一条。"""
        dataset = build_dataset()
        baselines = backends_for(dataset)

        class Misaligned:
            name = "last_value"

            def forecast(self, history, request, *, force: bool = False):  # type: ignore[no-untyped-def]
                result = baselines["last_value"].forecast(history, request)
                return result.model_copy(
                    update={"market_date": request.market_date + timedelta(days=1)}
                )

        with pytest.raises(ConfigurationError, match="market_date"):
            run(
                dataset,
                backends={"last_value": Misaligned()},
                spec=ForecastBenchmarkSpec(backends=("last_value",), regime=REGIME),
            )

    def test_backend_name_mismatch_is_explicit_failure(self) -> None:
        """backend 自我申报的名字与被注册的名字不一致时必须失败。"""
        dataset = build_dataset()
        baselines = backends_for(dataset)
        with pytest.raises(ConfigurationError, match=r"model\.backend"):
            run(
                dataset,
                # 注册名与实现自报的名字（drift）不同：record 会被归到一个没人要求过的名下
                backends={"not_drift": baselines["drift"]},
                spec=ForecastBenchmarkSpec(backends=("not_drift",), regime=REGIME),
            )

    def test_short_history_is_explicit_failure(self) -> None:
        """history 不足 lookback_bars 时 runner 自己也要拒绝（不靠 backend 自觉）。"""
        dataset = build_dataset()

        class ShortProvider(SyntheticProvider):
            def get_history(self, symbol, market_date, knowledge_cutoff, lookback_bars):  # type: ignore[no-untyped-def]
                history = super().get_history(symbol, market_date, knowledge_cutoff, lookback_bars)
                return history.model_copy(update={"bars": history.bars[:-1]})

        with pytest.raises(InsufficientHistoryError, match="lookback_bars"):
            run(dataset, provider=ShortProvider())

    def test_pilot_truncation_follows_session_then_symbol_order(self) -> None:
        """``origin_limit`` 是段内 ``(market_date, symbol)`` 顺序下的前 N 个 origin（ADR-023 §2.1）。"""
        result = run(spec=ForecastBenchmarkSpec(backends=BACKENDS, origin_limit=3, regime=REGIME))
        symbols = {record.symbol for record in result.records}
        assert symbols == {SYMBOL_A, SYMBOL_B}
        # 最早的 session 上的两个 symbol 都进来了，第三个名额给下一个 session
        first_day = min(record.market_date for record in result.records)
        assert sum(1 for record in result.records if record.market_date == first_day) == 2 * len(
            BACKENDS
        )
        assert sorted(evaluated_origins_by_symbol(result).values()) == [1, 2]

    def test_pilot_below_symbol_count_shrinks_universe_visibly(self) -> None:
        """``origin_limit`` 小于 symbol 数时 universe **确实会缩**——缩水必须能从结果里读出来。"""
        dataset = build_dataset()
        earliest = min(origin.market_date for origin in dataset.segments[0].origins)
        result = run(
            dataset, spec=ForecastBenchmarkSpec(backends=BACKENDS, origin_limit=1, regime=REGIME)
        )
        assert result.evaluated_origins == 1
        # 顺序是 (market_date, symbol)，因此进来的只有排序最前的那个 symbol
        assert evaluated_origins_by_symbol(result) == {SYMBOL_B: 1}
        assert {record.symbol for record in result.records} == {SYMBOL_B}
        # 人选的是 train 段**最早**的 session：不是最后一条（排除倒序），也不是「每个 symbol
        # 保底一个」（那会得到 2 个 origin）
        assert {record.market_date for record in result.records} == {earliest}

    def test_report_hash_changes_with_origin_limit(self) -> None:
        full = run()
        pilot = run(spec=ForecastBenchmarkSpec(backends=BACKENDS, origin_limit=3, regime=REGIME))
        assert full.report_hash != pilot.report_hash

    def test_report_hash_changes_with_regime_spec(self) -> None:
        """regime 口径是报告身份的一部分：换窗口必须换 report_hash（§49）。"""
        base = run()
        rewindowed = run(
            spec=ForecastBenchmarkSpec(
                backends=BACKENDS,
                regime=RegimeSpec(trend_window_sessions=4, volatility_window_sessions=5),
            )
        )
        assert base.regime_spec_hash != rewindowed.regime_spec_hash
        assert base.report_hash != rewindowed.report_hash

    def test_metrics_payload_exposes_scope_and_latency(self) -> None:
        result = run()
        payload = metrics_payload(result)
        assert payload["considered_origins"] == result.considered_origins
        assert payload["truncated"] is False
        assert payload["report_hash"] == result.report_hash
        assert set(payload["latency_ms"]) == set(BACKENDS)
        assert payload["data_coverage_end"] == result.data_coverage_end.isoformat()

    def test_report_hash_is_insensitive_to_surplus_data_coverage(self) -> None:
        """覆盖末端「多出来的部分」不进 report_hash：它没改变任何一条 label 判定。

        它仍然进 metadata / metrics.json（可追溯「算在覆盖到哪一天的数据上」），但把环境
        事实算进结论 hash，会让同一次 run 第二天重跑就换 hash。
        """
        dataset = build_dataset()
        base = run(dataset, labels=build_labels(dataset))
        later = run(
            dataset,
            labels=build_labels(
                dataset, coverage_end=CALENDAR_SESSIONS[TOTAL_SESSIONS + HORIZON + 30]
            ),
        )
        assert base.data_coverage_end != later.data_coverage_end
        assert [record.label_status for record in base.records] == [
            record.label_status for record in later.records
        ]
        assert base.report_hash == later.report_hash

    def test_data_coverage_end_must_reach_every_evaluated_origin(self) -> None:
        """覆盖末端早于被评估的 origin 是构建错误，不是「没有 label」。"""
        dataset = build_dataset()
        result = run(dataset)
        with pytest.raises(ValueError, match="data_coverage_end"):
            ForecastBenchmarkResult(
                **{
                    **result.model_dump(),
                    "data_coverage_end": result.records[0].market_date - timedelta(days=1),
                }
            )

    def test_insufficient_future_bars_status(self) -> None:
        dataset = build_dataset()
        last_origin_day = max(day for day in CALENDAR_SESSIONS[:TOTAL_SESSIONS])
        short = build_labels(dataset, coverage_end=last_origin_day)
        result = run(dataset, labels=short)
        statuses = {record.label_status for record in result.records}
        assert "INSUFFICIENT_FUTURE_BARS" in statuses
        # 只对**最后一段**（test）断言：数据末端之外没有 label 的正是它；跨段汇总里还混着
        # 更早的 train origin，把两者混为一谈会读错「这一段有没有证据」。
        entry = result.metrics_for("last_value", segment="test", group="all")
        assert entry is not None
        assert entry.labeled_count == 0
        assert entry.mae is None


class TestResultInvariants:
    """``ForecastBenchmarkResult`` 的三条不变量必须在**失败分支**上也被验证。"""

    def dump(self) -> dict:
        return run().model_dump()

    def test_truncated_flag_must_match_origin_counts(self) -> None:
        payload = self.dump()
        payload["truncated"] = True
        with pytest.raises(ValueError, match="truncated"):
            ForecastBenchmarkResult(**payload)

    def test_evaluated_cannot_exceed_considered(self) -> None:
        payload = self.dump()
        payload["evaluated_origins"] = payload["considered_origins"] + 1
        payload["truncated"] = False
        with pytest.raises(ValueError, match="exceeds considered"):
            ForecastBenchmarkResult(**payload)

    def test_run_level_metric_is_required_per_backend(self) -> None:
        payload = self.dump()
        payload["metrics"] = [entry for entry in payload["metrics"] if entry["segment"] is not None]
        with pytest.raises(ValueError, match="run-level"):
            ForecastBenchmarkResult(**payload)

    def test_run_level_sample_count_must_match_records(self) -> None:
        payload = self.dump()
        for entry in payload["metrics"]:
            if entry["segment"] is None and entry["group"]["axis"] == "overall":
                entry["sample_count"] -= 1
                entry["labeled_count"] -= 1
                break
        else:  # pragma: no cover - 上面的 payload 一定含 run 级 overall
            raise AssertionError("fixture produced no run-level overall metric")
        with pytest.raises(ValueError, match="same evidence"):
            ForecastBenchmarkResult(**payload)

    def test_backend_without_records_is_rejected(self) -> None:
        """spec 声明了却没产出记录的 backend 不能被当成「跑了但没数字」。"""
        payload = self.dump()
        payload["spec"]["backends"] = [*payload["spec"]["backends"], "ghost"]
        with pytest.raises(ValueError, match="ghost"):
            ForecastBenchmarkResult(**payload)


class TestArtifacts:
    def test_artifacts_are_written_and_indexed(self, tmp_path: Path) -> None:
        result = run()
        database = open_database(tmp_path / "index.sqlite3")
        try:
            store = ArtifactStore(tmp_path / "artifacts", database)
            registry = RunRegistry(database)
            run_id = registry.new_run_id()
            metadata = build_benchmark_run_metadata(
                result=result,
                run_id=run_id,
                git_commit=None,
                config_hash="0" * 64,
            )
            now = datetime(2026, 9, 27, 12, 0, tzinfo=CN_TZ)
            registry.register(
                RunRecord(
                    run_id=run_id,
                    kind="benchmark_forecast",
                    status="running",
                    created_at=now,
                    updated_at=now,
                    config_hash="0" * 64,
                    dataset_hash=result.dataset_hash,
                    run_dir=f"runs/{run_id}",
                    metadata=metadata,
                )
            )
            written = write_forecast_benchmark_artifacts(
                store=store,
                run_id=run_id,
                result=result,
                metadata=metadata,
                config_text="version: experiment-config-v1\n",
            )
            assert {record.name for record in written} == {
                "report",
                "metrics",
                "metadata",
                "config",
                "forecast",
            }
            report_text = store.read_text(run_id, "report")
            assert "# Forecast Benchmark v1" in report_text
            assert result.report_hash in report_text
            metrics = store.read_json(run_id, "metrics")
            assert metrics["report_hash"] == result.report_hash
            assert metrics["dataset_hash"] == result.dataset_hash
            frame = store.read_parquet(run_id, "forecast")
            assert len(frame) == len(result.records)
            assert set(frame["backend"]) == set(BACKENDS)
            assert registry.get(run_id).status in RUN_STATUSES
        finally:
            database.close()

    def test_missing_metrics_render_as_not_available(self) -> None:
        """报告必须把 ``None`` 渲染成 ``n/a``（不能渲染成 0）。"""
        dataset = build_dataset()
        last_origin_day = max(day for day in CALENDAR_SESSIONS[:TOTAL_SESSIONS])
        result = run(dataset, labels=build_labels(dataset, coverage_end=last_origin_day))
        markdown = render_markdown_report(result)
        assert "n/a" in markdown
        last_value = result.metrics_for("last_value", segment="test", group="all")
        assert last_value is not None
        assert last_value.mae is None
