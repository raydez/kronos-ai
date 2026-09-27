"""``kronos-ai benchmark forecast`` 的 CLI 层测试（RX-KAI-019，§33 / §34 / §41）。

这里测的是**编排**，而不是指标数学（后者在 tests/unit/test_evaluation_forecast_metrics.py，
端到端在 tests/regression/test_forecast_benchmark_pipeline.py）：

```text
--config <yaml> → ExperimentConfig → （注入的）context → run → §33 artifact 目录 + registry
```

装配（BaoStock 日历 / provider / Kronos 权重）通过 ``context_factory`` 注入合成世界，
因此本文件不需要网络、不需要模型权重，也不需要真实交易日历；被测的是 handler 自己：
参数流转、run 生命周期（running → succeeded / failed）、artifact 与 registry 的对应关系、
``--json`` 输出内容。

``build_benchmark_context``（真实装配）不在这里测：它要求 BaoStock 与 Kronos 权重，
属 integration 范畴。
"""

from __future__ import annotations

import json
from argparse import Namespace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from kronos_ai.cli import commands
from kronos_ai.cli.context import BenchmarkContext, resolve_data_coverage
from kronos_ai.cli.main import build_parser
from kronos_ai.config import ExperimentConfig
from kronos_ai.data.calendar import StaticTradingCalendar
from kronos_ai.domain.market import MarketBar, MarketHistory
from kronos_ai.domain.time import CN_TZ, MARKET_SESSION_CLOSE
from kronos_ai.errors import ConfigurationError
from kronos_ai.evaluation.baselines import build_baseline
from kronos_ai.evaluation.benchmark import ConstantLabelProvider
from kronos_ai.evaluation.dataset import (
    DEFAULT_LABEL_POLICY,
    build_walk_forward_dataset,
)
from kronos_ai.evaluation.walk_forward import SegmentSpec
from kronos_ai.infrastructure.persistence.run_registry import RunRegistry
from kronos_ai.infrastructure.persistence.schema import open_database

SYMBOL = "600000"
LOOKBACK = 6
SEGMENTS = (
    SegmentSpec(name="train", length_sessions=2),
    SegmentSpec(name="test", length_sessions=2),
)
ORIGIN_SESSIONS = sum(segment.length_sessions for segment in SEGMENTS)
TOTAL_SESSIONS = (
    LOOKBACK + ORIGIN_SESSIONS + DEFAULT_LABEL_POLICY.embargo_sessions * (len(SEGMENTS) - 1)
)


def weekdays(start: date, end: date) -> tuple[date, ...]:
    days: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return tuple(days)


CALENDAR_SESSIONS = weekdays(date(2026, 6, 1), date(2027, 6, 30))
WINDOW = CALENDAR_SESSIONS[:TOTAL_SESSIONS]


def session_close(day: date) -> datetime:
    return datetime.combine(day, MARKET_SESSION_CLOSE, tzinfo=CN_TZ)


def make_bar(day: date, close: float, previous: float | None) -> MarketBar:
    open_ = close if previous is None else previous
    return MarketBar(
        symbol=SYMBOL,
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


def calendar() -> StaticTradingCalendar:
    return StaticTradingCalendar(exchange="SSE", source="test-fixture", sessions=CALENDAR_SESSIONS)


class SyntheticProvider:
    """只给出到 ``market_date`` 为止的 PIT 窗口；含一条泄漏守卫。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, date, int]] = []

    def get_history(
        self, symbol: str, market_date: date, knowledge_cutoff: datetime, lookback_bars: int
    ) -> MarketHistory:
        if knowledge_cutoff.date() != market_date:
            raise AssertionError("runner must pass a cutoff inside market_date")
        self.calls.append((symbol, market_date, lookback_bars))
        days = tuple(day for day in CALENDAR_SESSIONS if day <= market_date)[-lookback_bars:]
        bars: list[MarketBar] = []
        previous: float | None = None
        for index, day in enumerate(days):
            close = 20.0 + 0.05 * index
            bars.append(make_bar(day, close, previous))
            previous = close
        return MarketHistory(
            symbol=symbol,
            market_date=market_date,
            knowledge_cutoff=knowledge_cutoff,
            bars=tuple(bars),
            provider="synthetic",
            dataset_version="synthetic-v1",
        )


def build_dataset(config: ExperimentConfig):
    return build_walk_forward_dataset(
        calendar=calendar(),
        sessions=WINDOW,
        symbols=config.dataset.symbols,
        plan=config.dataset.to_plan(),
        label_policy=config.label_policy(),
        lookback_bars=config.forecast.lookback_bars,
        cutoff_policy=config.knowledge_cutoff_policy,
    )


def build_labels(dataset: Any) -> ConstantLabelProvider:
    bars: dict[tuple[str, date], tuple[MarketBar, ...]] = {}
    for segment in dataset.segments:
        for origin in segment.origins:
            entries: list[MarketBar] = []
            previous: float | None = None
            for step, day in enumerate(origin.label_sessions):
                close = 20.0 + 0.05 * (100 + step)
                entries.append(make_bar(day, close, previous))
                previous = close
            bars[(origin.symbol, origin.market_date)] = tuple(entries)
    return ConstantLabelProvider(bars=bars, data_coverage_end=CALENDAR_SESSIONS[-1])


def make_context(config: ExperimentConfig) -> BenchmarkContext:
    dataset = build_dataset(config)
    backends = {
        name: build_baseline(name, calendar=calendar(), lookback_bars=config.forecast.lookback_bars)
        for name in config.benchmark.backends
    }
    return BenchmarkContext(
        dataset=dataset,
        provider=SyntheticProvider(),
        label_provider=build_labels(dataset),
        backends=backends,
        backend_identities={name: dict(backend.identity()) for name, backend in backends.items()},
    )


def write_config(
    tmp_path: Path, *, backends: str = "[last_value]", adjustment: str = "raw"
) -> Path:
    path = tmp_path / "benchmark.yaml"
    path.write_text(
        "\n".join(
            [
                "version: experiment-config-v1",
                "data:",
                f"  adjustment: {adjustment}",
                "forecast:",
                "  backend: kronos",
                f"  lookback_bars: {LOOKBACK}",
                "  sampling: {seed: 11, sample_count: 4}",
                "dataset:",
                # symbol 必须带引号：YAML 里 600000 会被解析成 int，config 显式拒绝
                f'  symbols: ["{SYMBOL}"]',
                f"  start_session: {WINDOW[0].isoformat()}",
                f"  end_session: {WINDOW[-1].isoformat()}",
                "  segments:",
                "    - {name: train, length_sessions: 2}",
                "    - {name: test, length_sessions: 2}",
                "benchmark:",
                f"  backends: {backends}",
                # regime 窗口必须 <= lookback，否则 runner 显式拒绝（不静默用更短窗口）
                "regime:",
                "  trend_window_sessions: 5",
                "  volatility_window_sessions: 5",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def benchmark_args(tmp_path: Path, config: Path, *, as_json: bool = False) -> Namespace:
    argv = [
        "benchmark",
        "forecast",
        "--config",
        str(config),
        "--artifacts-dir",
        str(tmp_path / "artifacts"),
        "--index-db",
        str(tmp_path / "artifacts" / "index.sqlite3"),
    ]
    if as_json:
        argv.append("--json")
    return build_parser().parse_args(argv)


def test_benchmark_forecast_writes_artifacts_and_registry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = write_config(tmp_path)
    args = benchmark_args(tmp_path, config_path)

    assert (
        commands.run_benchmark_forecast(args, context_factory=lambda cfg, _a: make_context(cfg))
        == 0
    )
    stdout = capsys.readouterr().out
    assert "benchmark" in stdout
    assert "last_value" in stdout
    # 文本摘要打印的是**跨段汇总**（run 级）那一条，不是某一段的数字
    assert "origins           4 evaluated / 4 considered" in stdout

    database = open_database(tmp_path / "artifacts" / "index.sqlite3")
    try:
        registry = RunRegistry(database)
        runs = registry.list_runs(kind="benchmark_forecast")
        assert len(runs) == 1
        record = runs[0]
        assert record.status == "succeeded"
        assert record.dataset_hash
        assert record.config_hash
        assert record.metadata["kind"] == "benchmark_forecast"
        assert record.metadata["evaluated_origins"] == 4
        assert record.metadata["truncated"] is False
        # 环境 / 数据身份事实进 metadata（不进 report_hash，见 ADR-023 §5）
        assert record.metadata["adjustment"] == "raw"
        assert record.metadata["evaluated_origins_by_symbol"] == {SYMBOL: 4}
        assert set(record.metadata["backend_identities"]) == {"last_value"}
        assert record.metadata["environment"]["python"]

        from kronos_ai.infrastructure.persistence.artifact_store import ArtifactStore

        store = ArtifactStore(tmp_path / "artifacts", database)
        names = {item.name for item in store.list_artifacts(record.run_id)}
        assert names == {"report", "metrics", "metadata", "config", "forecast"}
        metrics = store.read_json(record.run_id, "metrics")
        assert metrics["report_hash"] == record.metadata["report_hash"]
        assert metrics["truncated"] is False
    finally:
        database.close()


def test_benchmark_forecast_json_output_matches_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = write_config(tmp_path, backends="[last_value, drift]")
    args = benchmark_args(tmp_path, config_path, as_json=True)

    assert (
        commands.run_benchmark_forecast(args, context_factory=lambda cfg, _a: make_context(cfg))
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["metadata"]["backends"] == ["last_value", "drift"]
    assert payload["metrics"]["report_hash"] == payload["metadata"]["report_hash"]
    assert set(payload["metrics"]["latency_ms"]) == {"last_value", "drift"}


def test_metrics_artifact_holds_run_level_and_per_segment_slices(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    args = benchmark_args(tmp_path, config_path)
    commands.run_benchmark_forecast(args, context_factory=lambda cfg, _a: make_context(cfg))

    database = open_database(tmp_path / "artifacts" / "index.sqlite3")
    try:
        from kronos_ai.infrastructure.persistence.artifact_store import ArtifactStore

        store = ArtifactStore(tmp_path / "artifacts", database)
        run_id = RunRegistry(database).list_runs(kind="benchmark_forecast")[0].run_id
        metrics = store.read_json(run_id, "metrics")

    finally:
        database.close()
    overall = [
        entry
        for entry in metrics["metrics"]
        if entry["group"]["axis"] == "overall" and entry["segment"] is None
    ]
    assert len(overall) == 1
    assert overall[0]["sample_count"] == 4
    per_segment = {entry["segment"] for entry in metrics["metrics"] if entry["segment"] is not None}
    assert per_segment == {"train", "test"}


def test_artifact_write_failure_marks_run_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run 在 registry 里必须留下 ``failed`` 记录与错误消息，而不是「什么都没有」。"""
    config_path = write_config(tmp_path)
    args = benchmark_args(tmp_path, config_path)

    def boom(**_kwargs: Any) -> Any:
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(commands, "write_forecast_benchmark_artifacts", boom)
    with pytest.raises(RuntimeError, match="disk on fire"):
        commands.run_benchmark_forecast(args, context_factory=lambda cfg, _a: make_context(cfg))

    database = open_database(tmp_path / "artifacts" / "index.sqlite3")
    try:
        runs = RunRegistry(database).list_runs(kind="benchmark_forecast")
    finally:
        database.close()
    assert len(runs) == 1
    assert runs[0].status == "failed"
    assert runs[0].error is not None and "disk on fire" in runs[0].error


def test_adjustment_from_config_reaches_run_metadata(tmp_path: Path) -> None:
    """复权口径来自 config（数据身份），必须随 run 归档：命令行 flag 已经取消。"""
    config_path = write_config(tmp_path, adjustment="hfq")
    args = benchmark_args(tmp_path, config_path)
    commands.run_benchmark_forecast(args, context_factory=lambda cfg, _a: make_context(cfg))

    database = open_database(tmp_path / "artifacts" / "index.sqlite3")
    try:
        runs = RunRegistry(database).list_runs(kind="benchmark_forecast")
    finally:
        database.close()
    assert runs[0].metadata["adjustment"] == "hfq"


def test_config_secret_is_rejected_before_any_run(tmp_path: Path) -> None:
    path = tmp_path / "leaky.yaml"
    path.write_text("api_key: nope\n", encoding="utf-8")
    args = benchmark_args(tmp_path, path)
    with pytest.raises(Exception, match="secret"):
        commands.run_benchmark_forecast(args, context_factory=lambda cfg, _a: make_context(cfg))
    assert not (tmp_path / "artifacts" / "index.sqlite3").exists()


# ---------------------------------------------------------------------------
# 覆盖守卫（resolve_data_coverage）——纯函数，各条分支都能离线覆盖
# ---------------------------------------------------------------------------


class TestResolveDataCoverage:
    """装配层必须拒绝「尾部还没有 label」的窗口，而不是把它跑成空结论（§41 / ADR-023 §5）。"""

    def test_returns_last_published_session(self) -> None:
        sessions = weekdays(date(2026, 6, 1), date(2026, 12, 31))
        today = date(2026, 7, 15)
        coverage = resolve_data_coverage(
            sessions, today=today, end_session=date(2026, 7, 1), horizon_sessions=5
        )
        # 日历知道 7/15 之后的 session，但已发布的只到 today 的那一天
        assert coverage == max(day for day in sessions if day <= today)

    def test_tail_without_labels_is_rejected(self) -> None:
        sessions = weekdays(date(2026, 6, 1), date(2026, 12, 31))
        # today 只比窗口末端晚一天：horizon=5 的 label 窗口还不存在
        with pytest.raises(ConfigurationError, match="needs labels through"):
            resolve_data_coverage(
                sessions,
                today=date(2026, 7, 2),
                end_session=date(2026, 7, 1),
                horizon_sessions=5,
            )

    def test_no_published_session_is_rejected(self) -> None:
        sessions = weekdays(date(2026, 6, 1), date(2026, 12, 31))
        with pytest.raises(ConfigurationError, match="no published session"):
            resolve_data_coverage(
                sessions,
                today=date(2026, 5, 29),
                end_session=date(2026, 6, 30),
                horizon_sessions=5,
            )

    def test_window_beyond_calendar_coverage_is_rejected(self) -> None:
        # 日历就停在窗口末端：连 label 的时间轴都排不出来（对应 next_sessions 的覆盖不足）
        sessions = weekdays(date(2026, 6, 1), date(2026, 7, 1))
        with pytest.raises(ConfigurationError, match="fewer than the"):
            resolve_data_coverage(
                sessions,
                today=date(2026, 7, 1),
                end_session=date(2026, 7, 1),
                horizon_sessions=5,
            )

    def test_non_session_window_bound_is_rejected(self) -> None:
        """非 session 的 end_session 会把实际窗口悄悄提前，必须显式失败而不是照跑。"""
        sessions = weekdays(date(2026, 6, 1), date(2026, 12, 31))
        saturday = date(2026, 7, 4)
        assert saturday not in sessions
        with pytest.raises(ConfigurationError, match="is not a market session"):
            resolve_data_coverage(
                sessions,
                today=date(2026, 8, 1),
                end_session=saturday,
                horizon_sessions=5,
            )

    def test_window_bound_outside_loaded_calendar_is_rejected(self) -> None:
        """非 session 之外的两种成因要各自说清：超出装载覆盖 / 早于装载覆盖。"""
        sessions = weekdays(date(2026, 6, 1), date(2026, 12, 31))
        with pytest.raises(ConfigurationError, match="beyond the loaded calendar coverage"):
            resolve_data_coverage(
                sessions,
                today=date(2026, 7, 15),
                end_session=date(2027, 1, 6),
                horizon_sessions=5,
            )
        with pytest.raises(ConfigurationError, match="precedes the loaded calendar coverage"):
            resolve_data_coverage(
                sessions,
                today=date(2026, 7, 15),
                end_session=date(2026, 5, 20),
                horizon_sessions=5,
            )

    def test_non_positive_horizon_is_rejected(self) -> None:
        sessions = weekdays(date(2026, 6, 1), date(2026, 12, 31))
        with pytest.raises(ConfigurationError, match="horizon_sessions must be >= 1"):
            resolve_data_coverage(
                sessions,
                today=date(2026, 7, 15),
                end_session=date(2026, 7, 1),
                horizon_sessions=0,
            )
