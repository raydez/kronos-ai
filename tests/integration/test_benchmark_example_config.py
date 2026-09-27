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
from kronos_ai.config import load_experiment_config
from kronos_ai.domain.time import CN_TZ
from kronos_ai.infrastructure.persistence.run_registry import RunRegistry
from kronos_ai.infrastructure.persistence.schema import open_database
from kronos_ai.infrastructure.providers.baostock import BaoStockTradingCalendarLoader

pytestmark = [pytest.mark.integration, pytest.mark.benchmark]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "configs" / "benchmark-forecast.yaml"

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
