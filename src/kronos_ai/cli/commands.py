"""CLI 命令处理函数（§34）。

每个 handler 是 ``(Namespace) -> int``（进程退出码）。重活委托给 use case / service；
handler 只负责参数校验、调用与输出。
"""

from __future__ import annotations

import json
from argparse import Namespace
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Any

from kronos_ai.cli.context import build_forecast_service, default_runs_dir
from kronos_ai.cli.format import (
    cutoff_policy_from_args,
    print_forecast,
    sampling_from_args,
    write_forecast_result,
)
from kronos_ai.domain.time import resolve_knowledge_cutoff
from kronos_ai.errors import ArtifactError
from kronos_ai.forecast.service import ForecastService

ForecastServiceFactory = Callable[[Namespace, datetime], ForecastService]


def resolve_cutoff(args: Namespace) -> datetime:
    """按 §5 policy 从参数推导 knowledge_cutoff（唯一推导点）。"""
    market_date = date.fromisoformat(args.market_date)
    explicit = (
        datetime.fromisoformat(args.knowledge_cutoff)
        if getattr(args, "knowledge_cutoff", None) is not None
        else None
    )
    return resolve_knowledge_cutoff(market_date, cutoff_policy_from_args(args), explicit)


def run_forecast(
    args: Namespace,
    *,
    service_factory: ForecastServiceFactory = build_forecast_service,
) -> int:
    cutoff = resolve_cutoff(args)
    service = service_factory(args, cutoff)
    try:
        result = service.run(
            symbol=args.symbol,
            market_date=date.fromisoformat(args.market_date),
            knowledge_cutoff=cutoff,
            sampling=sampling_from_args(args),
            horizon=args.horizon,
            force=bool(args.force),
        )
    finally:
        service.close()
    if args.output:
        write_forecast_result(result, Path(args.output))
    print_forecast(result, args=args, cutoff=cutoff)
    return 0


def run_show(args: Namespace) -> int:
    runs_dir = Path(args.runs_dir) if args.runs_dir else default_runs_dir()
    metadata_path = runs_dir / args.run_id / "metadata.json"
    if not metadata_path.is_file():
        raise ArtifactError(f"run {args.run_id!r} not found under {runs_dir}")
    try:
        metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"run metadata at {metadata_path} is not valid JSON: {exc}") from exc
    if getattr(args, "json", False):
        print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    else:
        for key, value in metadata.items():
            print(f"{key:28} {value}")
    return 0
