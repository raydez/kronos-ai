"""``kronos-ai`` CLI（基线文档 §34：CLI First）。

第一阶段优先命令：

```bash
kronos-ai forecast 600000 --market-date 2026-09-25 --samples 64 --seed 42
kronos-ai run show <run-id>        # 由 RX-KAI-015 落地
kronos-ai benchmark forecast --config configs/benchmark-forecast.yaml   # RX-KAI-019
```

本模块只做参数解析与分发；use case 编排在 :class:`ForecastService`，推理在
ForecastBackend，数据在 MarketDataProvider。CLI 不复制任何领域逻辑。

未传 ``--knowledge-cutoff`` 时按 §5 默认 ``same_day_evening``（market_date 当日
18:00+08:00）推导，policy 名与版本进入输出摘要，不允许静默改变历史 run 的复现状态。
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from typing import Protocol, get_args

from kronos_ai.cli import commands
from kronos_ai.domain.time import KnowledgeCutoffPolicy
from kronos_ai.errors import KronosAIError

__all__ = ["build_parser", "main"]

# argparse choices 与 §5 policy Literal 同源，避免名单在两处漂移
_CUTOFF_POLICIES = get_args(KnowledgeCutoffPolicy)


class _Command(Protocol):
    def __call__(self, args: argparse.Namespace) -> int: ...


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kronos-ai",
        description="Kronos-AI v2 — Financial Forecast & Probabilistic Decision Research Platform",
    )
    parser.add_argument("--version", action="store_true", help="print version and exit")
    sub = parser.add_subparsers(dest="command")

    forecast = sub.add_parser("forecast", help="run a single point-in-time forecast")
    forecast.add_argument("symbol", help="normalized 6-digit A-share code, e.g. 600000")
    forecast.add_argument("--market-date", required=True, help="origin trading day (YYYY-MM-DD)")
    forecast.add_argument(
        "--knowledge-cutoff",
        default=None,
        help="ISO-8601 +08:00 cutoff; defaults to same_day_evening (18:00+08:00)",
    )
    forecast.add_argument(
        "--cutoff-policy",
        default="same_day_evening",
        choices=_CUTOFF_POLICIES,
        help="versioned knowledge_cutoff policy when --knowledge-cutoff is absent",
    )
    forecast.add_argument(
        "--backend",
        default="kronos",
        help="forecast backend name resolved via RuntimeRegistry (§17)",
    )
    forecast.add_argument("--horizon", type=int, default=5, help="future market sessions")
    forecast.add_argument("--samples", type=int, default=64, help="raw sample count")
    forecast.add_argument("--seed", type=int, default=42)
    forecast.add_argument("--temperature", type=float, default=1.0)
    forecast.add_argument("--top-k", type=int, default=0)
    forecast.add_argument("--top-p", type=float, default=0.9)
    forecast.add_argument("--lookback-bars", type=int, default=None)
    forecast.add_argument("--adjust", choices=("raw", "hfq", "qfq"), default="raw")
    forecast.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    forecast.add_argument("--dtype", choices=("auto", "float32"), default="auto")
    forecast.add_argument(
        "--cache-dir",
        default=None,
        help="forecast artifact cache root; omit to disable caching",
    )
    forecast.add_argument("--force", action="store_true", help="bypass the read cache")
    forecast.add_argument("--output", default=None, help="write ForecastResult JSON to this file")
    forecast.add_argument("--json", action="store_true", help="print machine-readable summary")
    forecast.set_defaults(handler=commands.run_forecast)

    run = sub.add_parser("run", help="inspect stored runs")
    run_sub = run.add_subparsers(dest="run_command")
    show = run_sub.add_parser("show", help="show run metadata")
    show.add_argument("run_id")
    show.add_argument("--runs-dir", default=None)
    show.add_argument("--json", action="store_true")
    show.set_defaults(handler=commands.run_show)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "version", False):
        from kronos_ai import __version__

        print(__version__)
        return 0
    handler: Callable[[argparse.Namespace], int] | None = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 2
    try:
        return handler(args)
    except KronosAIError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
