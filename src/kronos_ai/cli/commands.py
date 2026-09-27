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
from typing import cast

from kronos_ai.cli.context import (
    build_forecast_service,
    default_artifacts_dir,
    index_db_path_for,
)
from kronos_ai.cli.format import (
    cutoff_policy_from_args,
    format_run_record,
    print_forecast,
    sampling_from_args,
    write_forecast_result,
)
from kronos_ai.domain.run import RunStatus
from kronos_ai.domain.time import resolve_knowledge_cutoff
from kronos_ai.errors import ArtifactError
from kronos_ai.forecast.service import ForecastService
from kronos_ai.infrastructure.persistence.artifact_store import ArtifactStore
from kronos_ai.infrastructure.persistence.run_registry import RunRegistry
from kronos_ai.infrastructure.persistence.schema import open_database
from kronos_ai.infrastructure.persistence.sqlite import SQLiteDatabase

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


def _open_persistence(
    args: Namespace, *, create: bool = True
) -> tuple[SQLiteDatabase, ArtifactStore, RunRegistry]:
    """打开 §30 的 SQLite index 与 §33 的 artifact root（stdio 命令共用入口）。

    只读命令（``run show`` / ``run list``）传 ``create=False``：库不存在时显式失败，
    而不是顺手建出一个空 store，让 ``--help`` 之外的查询无副作用。
    """
    root = (
        Path(args.artifacts_dir)
        if getattr(args, "artifacts_dir", None)
        else default_artifacts_dir()
    )
    db_path = Path(args.index_db) if getattr(args, "index_db", None) else index_db_path_for(root)
    if not create and not db_path.exists():
        raise ArtifactError(
            f"run registry not found at {db_path}; run a forecast or benchmark first"
        )
    database = open_database(db_path)
    return database, ArtifactStore(root, database), RunRegistry(database)


def run_show(args: Namespace) -> int:
    database, store, registry = _open_persistence(args, create=False)
    try:
        record = registry.get(args.run_id)
        artifacts = store.list_artifacts(args.run_id)
        metadata = (
            store.read_json(args.run_id, "metadata")
            if store.find_artifact(args.run_id, "metadata") is not None
            else None
        )
        if getattr(args, "json", False):
            payload = {
                "run": record.model_dump(mode="json"),
                "artifacts": [item.model_dump(mode="json") for item in artifacts],
                "metadata": metadata,
            }
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        else:
            print(format_run_record(record))
            if metadata is not None:
                print("-- metadata --")
                print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
            print(f"-- artifacts ({len(artifacts)}) --")
            for item in artifacts:
                print(f"  {item.name:20} {item.media_type:38} {item.size_bytes:>10} B")
    finally:
        database.close()
    return 0


def run_list(args: Namespace) -> int:
    database, _store, registry = _open_persistence(args, create=False)
    try:
        status = getattr(args, "status", None)
        records = registry.list_runs(
            kind=getattr(args, "kind", None),
            status=cast(RunStatus | None, status),
            limit=getattr(args, "limit", None),
        )
        if getattr(args, "json", False):
            print(
                json.dumps(
                    [record.model_dump(mode="json") for record in records],
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        else:
            for record in records:
                print(format_run_record(record))
    finally:
        database.close()
    return 0
