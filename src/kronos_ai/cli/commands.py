"""CLI 命令处理函数（§34）。

每个 handler 是 ``(Namespace) -> int``（进程退出码）。重活委托给 use case / service；
handler 只负责参数校验、调用与输出。
"""

from __future__ import annotations

import json
import platform
from argparse import Namespace
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import cast

from kronos_ai.cli.context import (
    BenchmarkContext,
    build_benchmark_context,
    build_forecast_service,
    default_artifacts_dir,
    index_db_path_for,
)
from kronos_ai.cli.format import (
    cutoff_policy_from_args,
    format_benchmark_summary,
    format_run_record,
    print_forecast,
    sampling_from_args,
    write_forecast_result,
)
from kronos_ai.config import ExperimentConfig, gate_criteria_path, load_experiment_config
from kronos_ai.domain.run import RunStatus
from kronos_ai.domain.time import CN_TZ, resolve_knowledge_cutoff
from kronos_ai.errors import ArtifactError
from kronos_ai.evaluation.benchmark import (
    build_benchmark_run_metadata,
    run_forecast_benchmark,
)
from kronos_ai.evaluation.gate import (
    GateCriteria,
    evaluate_gate,
    gate_payload,
    load_gate_criteria,
    summarise_gate,
)
from kronos_ai.evaluation.report import (
    metrics_payload,
    write_forecast_benchmark_artifacts,
)
from kronos_ai.forecast.service import ForecastService
from kronos_ai.infrastructure.persistence.artifact_store import (
    ArtifactStore,
    run_dir_relative,
)
from kronos_ai.infrastructure.persistence.run_registry import RunRecord, RunRegistry
from kronos_ai.infrastructure.persistence.schema import open_database
from kronos_ai.infrastructure.persistence.sqlite import SQLiteDatabase

ForecastServiceFactory = Callable[[Namespace, datetime], ForecastService]
BenchmarkContextFactory = Callable[[ExperimentConfig, Namespace], BenchmarkContext]


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


def git_commit() -> str | None:
    """读取当前 git commit（§32 可追溯性）；读取失败返回 ``None``，不编造。

    容器 / 打包环境里没有 ``.git`` 是正常情况：报告里该字段显式为 ``null``，
    读报告的人知道「无法追溯」而不是「追溯到了某个值」。
    """
    import subprocess

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def environment_info() -> dict[str, str]:
    """§32 ``runtime`` 维：进程环境事实（python / torch 版本），读不到就不写。

    这些值不进 ``report_hash``（环境事实，理由见 ADR-023 §5），但必须随 run 归档：同一份
    config 在 torch 版本不同时跑出的数字未必可比。
    """
    info = {"python": platform.python_version()}
    try:
        import torch

        info["torch"] = str(torch.__version__)
    except ImportError:
        pass
    return info


def run_benchmark_forecast(
    args: Namespace,
    *,
    context_factory: BenchmarkContextFactory = build_benchmark_context,
) -> int:
    """``kronos-ai benchmark forecast --config <yaml>``（§34 / §41 / §49 / §42）。

    编排顺序：装载 config（含 gate 判据的预注册校验）→ 装配 context → **跑 benchmark**
    → 按预注册判据做 gate 判决 → 用结果建 metadata → 登记 run → 落盘 artifact。
    run_id 只有在拿到结果后才分配，因此：

    ```text
    benchmark 阶段失败（provider / backend / history 不足）→ 不留 run 记录（CLI 报错并退出 1）
    落盘阶段失败                                        → 留 ``failed`` 记录 + 错误消息
    ```

    这是刻意的：run 记录的身份由 ``dataset_hash`` / ``report_hash`` 定义，而这两个只有跑完才
    存在；一个没有结果的 run 没有可追溯的身份，把它登记成 ``failed`` 反而会让「run 目录」
    指向不存在的产物。中途失败的可见性由 CLI 的退出码与 stderr 承担。

    config 声明了 ``gate`` 时，判据在**跑之前**就被读取并校验（§42 的预注册），判决随
    artifact 归档（``gate.json`` + 判据原文）；证据不足会以
    :class:`~kronos_ai.errors.InsufficientEvidenceError` 显式失败，而不是给出一个
    看起来像结论的 REPLACE。
    """
    config, raw_text = load_experiment_config(args.config)
    criteria_path = gate_criteria_path(config, args.config)
    gate_criteria: GateCriteria | None = None
    gate_criteria_text: str | None = None
    if criteria_path is not None:
        gate_criteria, gate_criteria_text = load_gate_criteria(criteria_path)
    context = context_factory(config, args)
    spec = config.benchmark_spec()
    result = run_forecast_benchmark(
        dataset=context.dataset,
        provider=context.provider,
        label_provider=context.label_provider,
        backends=context.backends,
        spec=spec,
        sampling=config.sampling,
    )
    gate = None if gate_criteria is None else evaluate_gate(result, gate_criteria)

    database, store, registry = _open_persistence(args)
    try:
        run_id = registry.new_run_id()
        metadata = build_benchmark_run_metadata(
            result=result,
            run_id=run_id,
            git_commit=git_commit(),
            config_hash=config.config_hash,
            adjustment=config.data.adjustment,
            backend_identities=context.backend_identities,
            environment=environment_info(),
            extra=None
            if gate is None
            else {
                "gate_criteria_version": gate.criteria_version,
                "gate_criteria_hash": gate.criteria_hash,
                "gate_verdict": gate.verdict,
                "gate_hash": gate.gate_hash,
            },
        )
        now = datetime.now(CN_TZ)
        registry.register(
            RunRecord(
                run_id=run_id,
                kind="benchmark_forecast",
                status="running",
                created_at=now,
                updated_at=now,
                config_hash=config.config_hash,
                dataset_hash=result.dataset_hash,
                run_dir=run_dir_relative(run_id),
                metadata=metadata,
            )
        )
        try:
            write_forecast_benchmark_artifacts(
                store=store,
                run_id=run_id,
                result=result,
                metadata=metadata,
                config_text=raw_text,
                gate=gate,
                gate_criteria_text=gate_criteria_text,
            )
        except Exception as exc:
            registry.update_status(
                run_id, "failed", error=f"{type(exc).__name__}: {exc}", now=datetime.now(CN_TZ)
            )
            raise
        registry.update_status(run_id, "succeeded", now=datetime.now(CN_TZ))

        if getattr(args, "json", False):
            payload = {
                "run_id": run_id,
                "config_hash": config.config_hash,
                "metadata": metadata,
                "metrics": metrics_payload(result),
                "gate": None if gate is None else gate_payload(gate),
            }
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        else:
            print(format_benchmark_summary(result, run_id=run_id))
            if gate is not None:
                print(summarise_gate(gate))
    finally:
        database.close()
    return 0
