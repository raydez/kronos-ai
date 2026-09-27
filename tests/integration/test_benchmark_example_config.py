"""示例配置的**真实**可用性（RX-KAI-019；``integration`` 标记，默认不跑）。

离线用例只能证明「配置能被解析」；示例配置最容易坏的地方是**窗口与真实交易日历对不上**
（session 数不等于 lookback + Σsegments + embargo，或窗口末端之后没有足够的已发布 session
给 label）。这两条只有对着 BaoStock 的 ``query_trade_dates`` 才能验证，因此本文件是
integration 用例：默认 ``pytest -m 'not integration'`` 不执行，需要时显式运行。

```bash
python -m pytest tests/integration/test_benchmark_example_config.py -m integration -q
```

如果示例窗口在真实日历上不再成立（节假日调整、配置漂移），本用例会失败——这正是它存在的
意义：示例不允许静默腐坏。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from kronos_ai.cli import commands
from kronos_ai.cli.main import build_parser
from kronos_ai.config import gate_criteria_path, load_experiment_config
from kronos_ai.domain.time import CN_TZ
from kronos_ai.evaluation.gate import load_gate_criteria
from kronos_ai.infrastructure.persistence.run_registry import RunRegistry
from kronos_ai.infrastructure.persistence.schema import open_database
from kronos_ai.infrastructure.providers.baostock import BaoStockTradingCalendarLoader

pytestmark = [pytest.mark.integration, pytest.mark.benchmark]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "configs" / "benchmark-forecast.yaml"

#: gate 用例必须复现**示例判据所针对的证据量**：判据的门槛是证据量门槛（逐组 / 整体），
#: 而证据量由窗口几何（symbols × 段长）决定。因此这个模板从示例 config 里取 symbols /
#: lookback / 段长，只把 backends 换成 baseline-only（不加载 Kronos 权重）。
GATE_WINDOW = """\
version: experiment-config-v1
data:
  adjustment: raw
forecast:
  backend: kronos
  lookback_bars: {lookback}
  sampling: {{seed: 42, sample_count: 4}}
dataset:
  symbols: [{symbols}]
  start_session: {start}
  end_session: {end}
  segments:
{segments}
benchmark:
  backends: [last_value, drift]
regime:
  trend_window_sessions: 20
  volatility_window_sessions: 20
gate:
  criteria_file: gate.yaml
"""


#: 示例里的 backends 含 kronos（要权重）；本用例只跑 baseline，验证的是「窗口 / 日历 /
#: 装配 / artifact」这条链路，不是模型结论。
BASELINE_ONLY = """\
version: experiment-config-v1
data:
  adjustment: raw
forecast:
  backend: kronos
  lookback_bars: 32
  sampling: {{seed: 42, sample_count: 4}}
dataset:
  symbols: ["600000"]
  start_session: {start}
  end_session: {end}
  segments:
    - {{name: train, length_sessions: 25}}
    - {{name: test, length_sessions: 16}}
benchmark:
  backends: [last_value, drift]
regime:
  trend_window_sessions: 20
  volatility_window_sessions: 20
"""


def test_example_config_window_matches_real_calendar() -> None:
    """示例配置的窗口必须在真实 SSE 日历上**恰好**是它声明的 session 数。"""
    config, _ = load_experiment_config(EXAMPLE_CONFIG)
    loader = BaoStockTradingCalendarLoader()
    try:
        calendar = loader.load(
            start=config.dataset.start_session,
            end=max(config.dataset.end_session, datetime.now(CN_TZ).date()),
            exchange="SSE",
        )
    finally:
        loader.close()
    sessions = [
        day
        for day in calendar.sessions
        if config.dataset.start_session <= day <= config.dataset.end_session
    ]
    assert len(sessions) == config.required_sessions()
    # 窗口末端之后至少还有 horizon 个已发布 session，最后一个 origin 才有 label
    published_end = max(day for day in calendar.sessions if day <= datetime.now(CN_TZ).date())
    needed_end = calendar.next_sessions(
        config.dataset.end_session, config.label_policy().horizon_sessions
    )[-1]
    assert published_end >= needed_end


def test_example_window_runs_end_to_end_with_baselines(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """同窗口、baseline-only 的真实 run：装配 → runner → artifact → registry 全链路。"""
    config, _ = load_experiment_config(EXAMPLE_CONFIG)
    config_path = tmp_path / "baseline-only.yaml"
    config_path.write_text(
        BASELINE_ONLY.format(
            start=config.dataset.start_session.isoformat(),
            end=config.dataset.end_session.isoformat(),
        ),
        encoding="utf-8",
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark",
            "forecast",
            "--config",
            str(config_path),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--json",
        ]
    )
    assert commands.run_benchmark_forecast(args) == 0
    payload = json.loads(capsys.readouterr().out)
    # 1 symbol × (25 train + 16 test) origins，全部被评分（覆盖末端守卫的目的）
    assert payload["metadata"]["considered_origins"] == 41
    assert payload["metadata"]["evaluated_origins"] == 41
    assert payload["metadata"]["truncated"] is False
    assert payload["metadata"]["evaluated_origins_by_symbol"] == {"600000": 41}
    assert payload["metadata"]["adjustment"] == "raw"
    run_id = payload["run_id"]
    database = open_database(tmp_path / "artifacts" / "index.sqlite3")
    try:
        record = RunRegistry(database).get(run_id)
    finally:
        database.close()
    assert record.status == "succeeded"
    assert record.metadata["report_hash"] == payload["metrics"]["report_hash"]


def test_example_gate_criteria_are_satisfiable_on_the_real_window(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """示例判据的门槛必须能被**真实窗口**满足（RX-KAI-020，§42）。

    这条用例守的是「预注册的判据在真实数据上还能不能判」：判据的分组门槛是**证据量**门槛，
    它只依赖窗口与日历（以及**证据切片**），不依赖任何模型的表现——因此可以用 baseline-only
    的真实 run 来验证，而不必加载 Kronos 权重。窗口几何照抄示例 config（symbols × 段长），
    否则验的就不是示例判据所针对的那批 origin。如果窗口漂移导致某个 regime 分组不再有足够
    origin，本用例会失败：那说明 `configs/gate-criteria-forecast-v1.yaml` 的门槛必须在
    **下一次正式 run 之前**重新预注册，而不是让它悄悄变成 InsufficientEvidenceError。
    """
    example, _ = load_experiment_config(EXAMPLE_CONFIG)
    criteria_path = gate_criteria_path(example, EXAMPLE_CONFIG)
    assert criteria_path is not None
    criteria, _ = load_gate_criteria(criteria_path)

    config_path = tmp_path / "baseline-only-with-gate.yaml"
    criteria_file = tmp_path / "gate.yaml"
    criteria_file.write_text(
        "\n".join(
            [
                "version: forecast-gate-criteria-v1",
                "candidate: last_value",
                f"metric: {criteria.metric}",
                "baselines: [drift]",
                f"grouping_axis: {criteria.grouping_axis}",
                # 证据切片也照抄：判决数的是哪些 origin 必须在同一批 origin 上可判
                "evidence_segments: [" + ", ".join(criteria.evidence_segments) + "]",
                # 采用示例判据的证据门槛：这正是被测的东西
                f"min_groups_with_positive_increment: {criteria.min_groups_with_positive_increment}",
                f"min_paired_samples_per_group: {criteria.min_paired_samples_per_group}",
                f"min_paired_samples_overall: {criteria.min_paired_samples_overall}",
                f"confidence_level: {criteria.confidence_level}",
                f"bootstrap_iterations: {criteria.bootstrap_iterations}",
                f"bootstrap_seed: {criteria.bootstrap_seed}",
                f"max_seconds_per_origin: {criteria.max_seconds_per_origin}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    # 窗口几何照抄示例 config：门槛是证据量门槛，换个 symbols 数就换了证据量
    config_path.write_text(
        GATE_WINDOW.format(
            lookback=example.forecast.lookback_bars,
            symbols=", ".join(f'"{symbol}"' for symbol in example.dataset.symbols),
            start=example.dataset.start_session.isoformat(),
            end=example.dataset.end_session.isoformat(),
            segments="\n".join(
                f"    - {{name: {segment.name}, length_sessions: {segment.length_sessions}}}"
                for segment in example.dataset.segments
            ),
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "benchmark",
            "forecast",
            "--config",
            str(config_path),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--json",
        ]
    )
    # 证据不足时 evaluate_gate 会抛 InsufficientEvidenceError（run 失败），
    # 因此这里能跑到 verdict 就说明示例门槛在真实窗口上可判。
    assert commands.run_benchmark_forecast(args) == 0
    payload = json.loads(capsys.readouterr().out)
    gate = payload["gate"]
    assert gate is not None
    assert gate["verdict"] in {"GO", "CONDITIONAL", "REPLACE"}
    assert gate["criteria_hash"] == payload["metadata"]["gate_criteria_hash"]
    # 判决覆盖的段就是判据声明的段（不是「跑出来是哪几段就算哪几段」）
    assert gate["evidence_segments"] == list(criteria.evidence_segments)
    assert len(gate["groups"]) == 3
    for entry in gate["groups"]:
        assert entry["adequate_evidence"] is True, entry
        assert entry["sample_count"] >= criteria.min_paired_samples_per_group
    assert gate["overall"]["adequate_evidence"] is True
    assert gate["overall"]["sample_count"] >= criteria.min_paired_samples_overall
