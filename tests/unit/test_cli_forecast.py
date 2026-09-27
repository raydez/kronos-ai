"""CLI（§34）：参数解析、cutoff policy 推导、forecast 命令装配、run show。

forecast 端到端用 tiny 随机权重 runtime（不下载权重、不联网）驱动真实的
KronosForecastBackend，验证 CLI → ForecastService → Backend → ForecastResult 全链路，
以及输出摘要 / --json / --output 落盘。
"""

from __future__ import annotations

import json
from argparse import Namespace
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest

from kronos_ai.cli import commands
from kronos_ai.cli.context import ADJUST_MODE_TO_FLAG
from kronos_ai.cli.format import calendar_span
from kronos_ai.cli.main import build_parser, main
from kronos_ai.data.calendar import StaticTradingCalendar
from kronos_ai.domain.market import MarketHistory
from kronos_ai.domain.time import CN_TZ, CUTOFF_POLICY_VERSION
from kronos_ai.errors import ArtifactError, ProviderError
from kronos_ai.forecast.backends.kronos.backend import KronosForecastBackend
from kronos_ai.forecast.cache import FileSystemForecastCache
from kronos_ai.forecast.service import ForecastService
from kronos_ai.infrastructure.persistence.artifact_store import ArtifactStore
from kronos_ai.infrastructure.persistence.run_registry import RunRecord, RunRegistry
from kronos_ai.infrastructure.persistence.schema import open_database
from kronos_ai.infrastructure.providers.baostock import ADJUST_FLAG_TO_MODE

MARKET_DATE = date(2026, 9, 25)
LOOKBACK = 16


class StaticProvider:
    """只返回既有合成 history；校验 service 用请求上下文正确调用 provider。"""

    def __init__(self, history: MarketHistory) -> None:
        self._history = history
        self.calls: list[tuple[str, date, datetime, int]] = []

    def get_history(
        self, symbol: str, market_date: date, knowledge_cutoff: datetime, lookback_bars: int
    ) -> MarketHistory:
        self.calls.append((symbol, market_date, knowledge_cutoff, lookback_bars))
        return self._history


def make_service_factory(
    runtime: Any, calendar: StaticTradingCalendar, history: MarketHistory
) -> Any:
    provider = StaticProvider(history)

    def _factory(args: Namespace, cutoff: datetime) -> ForecastService:
        backend = KronosForecastBackend(runtime, calendar=calendar)
        return ForecastService(provider, backend, lookback_bars=LOOKBACK)

    return _factory


def forecast_args(**overrides: Any) -> Namespace:
    parser = build_parser()
    argv = ["forecast", "600000", "--market-date", MARKET_DATE.isoformat()]
    for key, value in overrides.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            argv.append(flag)
        elif value is False or value is None:
            continue
        else:
            argv += [flag, str(value)]
    return parser.parse_args(argv)


def test_parser_defaults() -> None:
    args = forecast_args()
    assert args.symbol == "600000"
    assert args.horizon == 5
    assert args.samples == 64
    assert args.seed == 42
    assert args.temperature == 1.0
    assert args.top_k == 0
    assert args.top_p == 0.9
    assert args.adjust == "raw"
    assert args.device == "auto"
    assert args.dtype == "auto"
    assert args.cutoff_policy == "same_day_evening"
    assert args.knowledge_cutoff is None
    assert args.force is False


def test_adjust_flag_mapping_matches_provider_contract() -> None:
    # CLI 的 mode→flag 必须是 provider flag→mode 的逆映射（真源在 provider）
    assert {mode: flag for flag, mode in ADJUST_FLAG_TO_MODE.items()} == ADJUST_MODE_TO_FLAG


def test_calendar_span_covers_horizon() -> None:
    start, end = calendar_span(MARKET_DATE, 5)
    assert start < MARKET_DATE < end
    assert (end - MARKET_DATE).days >= 45


def test_default_cutoff_is_same_day_evening() -> None:
    args = forecast_args()
    cutoff = commands.resolve_cutoff(args)
    assert cutoff == datetime(2026, 9, 25, 18, 0, tzinfo=CN_TZ)


def test_market_close_policy() -> None:
    args = forecast_args(cutoff_policy="market_close")
    assert commands.resolve_cutoff(args) == datetime(2026, 9, 25, 15, 0, tzinfo=CN_TZ)


def test_explicit_cutoff_overrides_policy() -> None:
    args = forecast_args(knowledge_cutoff="2026-09-25T16:30:00+08:00")
    assert commands.resolve_cutoff(args) == datetime(2026, 9, 25, 16, 30, tzinfo=CN_TZ)
    # 显式 cutoff 落在 market_date 之外属于调用方错误
    bad = forecast_args(knowledge_cutoff="2026-09-24T16:30:00+08:00")
    with pytest.raises(ValueError, match="market_date"):
        commands.resolve_cutoff(bad)


def test_explicit_policy_without_value_is_rejected() -> None:
    args = forecast_args(cutoff_policy="explicit")
    with pytest.raises(ValueError, match="explicit"):
        commands.resolve_cutoff(args)


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip()


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "usage" in capsys.readouterr().out.lower()


def test_domain_error_maps_to_exit_code_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """KronosAIError 是已知的领域失败：退出码 1、消息进 stderr，不打印 traceback。"""

    def boom(_args: Namespace) -> int:
        raise ProviderError("baostock exploded")

    monkeypatch.setattr(commands, "run_forecast", boom)
    argv = ["forecast", "600000", "--market-date", MARKET_DATE.isoformat()]
    assert main(argv) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "baostock exploded" in captured.err


@pytest.mark.parametrize("as_json", [False, True])
def test_run_forecast_end_to_end(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    tiny_runtime_factory: Any,
    session_calendar: StaticTradingCalendar,
    make_history: Any,
    as_json: bool,
) -> None:
    runtime = tiny_runtime_factory(lookback_bars=LOOKBACK)
    history: MarketHistory = make_history(n_bars=LOOKBACK)
    factory = make_service_factory(runtime, session_calendar, history)
    output = tmp_path / "result.json"

    args = forecast_args(samples=4, horizon=3, seed=20260927, json=as_json, output=str(output))
    assert commands.run_forecast(args, service_factory=factory) == 0
    stdout = capsys.readouterr().out

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["symbol"] == "600000"
    assert payload["distribution"]["horizon"] == 3
    assert payload["distribution"]["sample_count"] == 4
    assert payload["artifact_id"]
    if as_json:
        summary = json.loads(stdout)
        assert summary["artifact_id"] == payload["artifact_id"]
        assert summary["expected_return"] == pytest.approx(
            payload["distribution"]["expected_return"]
        )
        assert summary["knowledge_cutoff"]["policy"] == "same_day_evening"
    else:
        assert "600000" in stdout
        assert "artifact_id" in stdout
        # §5 / ADR-012：文本输出也必须带 policy 名 + 参数 + 版本，不能只有时刻
        assert "cutoff_policy" in stdout
        assert "same_day_evening" in stdout
        assert CUTOFF_POLICY_VERSION in stdout
        assert "cutoff_parameters" in stdout


def test_run_forecast_passes_lookback_and_cutoff_to_provider(
    tiny_runtime_factory: Any,
    session_calendar: StaticTradingCalendar,
    make_history: Any,
) -> None:
    runtime = tiny_runtime_factory(lookback_bars=LOOKBACK)
    history: MarketHistory = make_history(n_bars=LOOKBACK)
    provider = StaticProvider(history)
    backend = KronosForecastBackend(runtime, calendar=session_calendar)
    service = ForecastService(provider, backend, lookback_bars=LOOKBACK)

    args = forecast_args(samples=2, horizon=2)
    commands.run_forecast(args, service_factory=lambda _args, _cutoff: service)

    assert provider.calls == [
        ("600000", MARKET_DATE, datetime(2026, 9, 25, 18, 0, tzinfo=CN_TZ), LOOKBACK)
    ]


def test_run_forecast_uses_cache_when_configured(
    tmp_path: Path,
    tiny_runtime_factory: Any,
    session_calendar: StaticTradingCalendar,
    make_history: Any,
) -> None:
    """--cache-dir 时第二次同参调用命中缓存（artifact 自包含，不重跑推理）。"""
    runtime = tiny_runtime_factory(lookback_bars=LOOKBACK)
    history: MarketHistory = make_history(n_bars=LOOKBACK)

    cache = FileSystemForecastCache(tmp_path / "cache")
    backend = KronosForecastBackend(runtime, calendar=session_calendar, cache=cache)
    provider = StaticProvider(history)
    service = ForecastService(provider, backend, lookback_bars=LOOKBACK)

    calls = 0
    original = backend._sampler.decode_raw_samples  # 探针：证明第二次未重跑推理

    def counting(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    backend._sampler.decode_raw_samples = counting  # type: ignore[method-assign]

    def factory(_args: Namespace, _cutoff: datetime) -> ForecastService:
        return service

    args = forecast_args(samples=3, horizon=2, seed=7)
    commands.run_forecast(args, service_factory=factory)
    files_first = sorted(p.name for p in cache.root.rglob("*.json"))
    commands.run_forecast(args, service_factory=factory)
    files_second = sorted(p.name for p in cache.root.rglob("*.json"))
    assert files_first == files_second
    assert len(files_first) == 1
    assert calls == 1


def test_run_forecast_force_bypasses_cache(
    tmp_path: Path,
    tiny_runtime_factory: Any,
    session_calendar: StaticTradingCalendar,
    make_history: Any,
) -> None:
    """`--force` 必须从 CLI 一路透传到底层：绕过读缓存、重新推理（§15）。"""
    runtime = tiny_runtime_factory(lookback_bars=LOOKBACK)
    history = make_history(n_bars=LOOKBACK)
    cache = FileSystemForecastCache(tmp_path / "cache")
    backend = KronosForecastBackend(runtime, calendar=session_calendar, cache=cache)
    service = ForecastService(StaticProvider(history), backend, lookback_bars=LOOKBACK)

    calls = 0
    original = backend._sampler.decode_raw_samples

    def counting(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    backend._sampler.decode_raw_samples = counting  # type: ignore[method-assign]

    factory = lambda _args, _cutoff: service  # noqa: E731
    commands.run_forecast(forecast_args(samples=3, horizon=2, seed=7), service_factory=factory)
    commands.run_forecast(
        forecast_args(samples=3, horizon=2, seed=7, force=True), service_factory=factory
    )
    assert calls == 2


def _seed_run(
    root: Path,
    run_id: str,
    *,
    kind: str = "forecast",
    status: str = "succeeded",
    with_metadata: bool = True,
) -> None:
    """在临时 artifact root 登记一个 run，并按需写入 metadata artifact（§30/§32）。"""
    database = open_database(root / "index.sqlite3")
    created = datetime(2026, 9, 25, 18, 0, tzinfo=CN_TZ)
    try:
        registry = RunRegistry(database)
        store = ArtifactStore(root, database)
        registry.register(
            RunRecord(
                run_id=run_id,
                kind=kind,
                status="pending",
                created_at=created,
                updated_at=created,
                config_hash="a" * 64,
                dataset_hash="b" * 64,
                dedup_key=f"{kind}:{run_id}",
                run_dir=f"runs/{run_id}",
            )
        )
        if with_metadata:
            store.write_json(run_id, "metadata", {"run_id": run_id, "kind": kind})
        if status != "pending":
            registry.update_status(run_id, status, now=datetime(2026, 9, 25, 18, 5, tzinfo=CN_TZ))
    finally:
        database.close()


def test_run_show_reports_registered_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_id = "01ARZ3NDEKTSV4RRFFQ69G5FA0"
    _seed_run(tmp_path, run_id)
    parser = build_parser()
    args = parser.parse_args(["run", "show", run_id, "--artifacts-dir", str(tmp_path), "--json"])
    assert commands.run_show(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run"]["run_id"] == run_id
    assert payload["run"]["status"] == "succeeded"
    assert payload["run"]["kind"] == "forecast"
    assert payload["metadata"] == {"run_id": run_id, "kind": "forecast"}
    assert [item["name"] for item in payload["artifacts"]] == ["metadata"]


def test_run_show_text_output_lists_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_id = "01ARZ3NDEKTSV4RRFFQ69G5FA0"
    _seed_run(tmp_path, run_id)
    parser = build_parser()
    args = parser.parse_args(["run", "show", run_id, "--artifacts-dir", str(tmp_path)])
    assert commands.run_show(args) == 0
    out = capsys.readouterr().out
    assert run_id in out
    assert "metadata" in out


def test_run_show_missing_run_raises(tmp_path: Path) -> None:
    # 库存在但 run 不存在 -> 显式失败；库不存在时的失败见 test_run_read_only_commands_...
    _seed_run(tmp_path, "01ARZ3NDEKTSV4RRFFQ69G5FA9")
    parser = build_parser()
    args = parser.parse_args(
        ["run", "show", "01ARZ3NDEKTSV4RRFFQ69G5FA0", "--artifacts-dir", str(tmp_path)]
    )
    with pytest.raises(ArtifactError, match="not registered"):
        commands.run_show(args)


def test_run_read_only_commands_do_not_create_store(tmp_path: Path) -> None:
    parser = build_parser()
    list_args = parser.parse_args(["run", "list", "--artifacts-dir", str(tmp_path)])
    show_args = parser.parse_args(
        ["run", "show", "01ARZ3NDEKTSV4RRFFQ69G5FA0", "--artifacts-dir", str(tmp_path)]
    )
    with pytest.raises(ArtifactError, match="run registry not found"):
        commands.run_list(list_args)
    with pytest.raises(ArtifactError, match="run registry not found"):
        commands.run_show(show_args)
    assert not (tmp_path / "index.sqlite3").exists()


def test_run_list_filters_and_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _seed_run(tmp_path, "01ARZ3NDEKTSV4RRFFQ69G5FA0", kind="forecast", status="succeeded")
    _seed_run(tmp_path, "01ARZ3NDEKTSV4RRFFQ69G5FA1", kind="forecast", status="failed")
    _seed_run(tmp_path, "01ARZ3NDEKTSV4RRFFQ69G5FA2", kind="benchmark", status="succeeded")
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "list",
            "--kind",
            "forecast",
            "--status",
            "succeeded",
            "--artifacts-dir",
            str(tmp_path),
            "--json",
        ]
    )
    assert commands.run_list(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [item["run_id"] for item in payload] == ["01ARZ3NDEKTSV4RRFFQ69G5FA0"]


def test_run_list_rejects_unknown_status(tmp_path: Path) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "list", "--status", "nope", "--artifacts-dir", str(tmp_path)])


def test_module_entrypoint_exposed(capsys: pytest.CaptureFixture[str]) -> None:
    import kronos_ai.__main__ as entry

    assert entry.main(["--version"]) == 0
    assert capsys.readouterr().out.strip()
