"""Run registry：run 生命周期与元数据的 SQLite 单一真源（§30–§32；ADR-013）。

- run_id 是 :mod:`kronos_ai.domain.ids` 的 ULID（§31）；
- status 迁移由 :data:`kronos_ai.domain.run.RUN_STATUS_TRANSITIONS` 约束，终态不可再变；
- ``dedup_key`` 上的唯一索引提供 §35（API Job Mode Idempotency-Key）的幂等提交：
  同一 key 重复 submit 返回既有 run，不新建 job；
- 完整 §32 metadata 以 canonical JSON 原样存放（``metadata_json``），registry 只索引
  查询用得到的列（kind / status / config_hash / dataset_hash / run_dir）。
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from kronos_ai.domain.hashing import canonical_json, is_sha256_hex
from kronos_ai.domain.ids import UlidGenerator, validate_ulid
from kronos_ai.domain.run import RUN_STATUS_TRANSITIONS, RUN_STATUSES, RunStatus
from kronos_ai.domain.time import CN_TZ, ensure_shanghai_aware
from kronos_ai.errors import ArtifactError, ConfigurationError
from kronos_ai.infrastructure.persistence.artifact_store import run_dir_relative
from kronos_ai.infrastructure.persistence.sqlite import SQLiteDatabase

RUN_REGISTRY_CONTRACT_VERSION = "run-registry-v1"

_COLUMNS = (
    "run_id, kind, status, created_at, updated_at, config_hash, dataset_hash, "
    "dedup_key, run_dir, error, metadata_json"
)
_INSERT_SQL = f"INSERT INTO runs ({_COLUMNS}) VALUES ({', '.join('?' * 11)})"
_SELECT_SQL = f"SELECT {_COLUMNS} FROM runs"


def _default_clock() -> datetime:
    return datetime.now(CN_TZ)


def _require_nonempty(value: str, field: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field} must be non-empty")
    return stripped


class RunRecord(BaseModel):
    """run registry 的一行（§32 的索引化视图；完整 metadata 在 ``metadata``）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    kind: str
    status: RunStatus
    created_at: datetime
    updated_at: datetime

    config_hash: str | None = None
    dataset_hash: str | None = None
    dedup_key: str | None = None

    run_dir: str
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("run_id")
    @classmethod
    def _run_id_ulid(cls, value: str) -> str:
        try:
            return validate_ulid(value)
        except ConfigurationError as exc:  # pydantic 只把 ValueError 归为 ValidationError
            raise ValueError(str(exc)) from exc

    @field_validator("kind", "run_dir")
    @classmethod
    def _nonempty(cls, value: str, info: ValidationInfo) -> str:
        return _require_nonempty(value, str(info.field_name))

    @field_validator("config_hash", "dataset_hash")
    @classmethod
    def _sha256(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is not None and not is_sha256_hex(value):
            raise ValueError(f"{info.field_name} must be a 64-char lowercase sha256 hex digest")
        return value

    @field_validator("dedup_key")
    @classmethod
    def _dedup_key_nonempty(cls, value: str | None) -> str | None:
        return None if value is None else _require_nonempty(value, "dedup_key")

    @field_validator("metadata")
    @classmethod
    def _metadata_json_serializable(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            canonical_json(value)
        except TypeError as exc:
            raise ValueError(f"run metadata must be JSON-serializable: {exc}") from exc
        return value

    @field_validator("created_at", "updated_at")
    @classmethod
    def _timestamps_shanghai(cls, value: datetime, info: ValidationInfo) -> datetime:
        return ensure_shanghai_aware(value, str(info.field_name))

    @model_validator(mode="after")
    def _updated_after_created(self) -> RunRecord:
        if self.updated_at < self.created_at:
            raise ValueError(
                f"updated_at {self.updated_at.isoformat()} precedes created_at "
                f"{self.created_at.isoformat()}"
            )
        return self

    @model_validator(mode="after")
    def _run_dir_matches_run_id(self) -> RunRecord:
        expected = run_dir_relative(self.run_id)
        if self.run_dir != expected:
            raise ValueError(
                f"run_dir must be {expected!r} for run_id {self.run_id!r}, got {self.run_dir!r}"
            )
        return self


class RunRegistry:
    """SQLite run registry；所有写入串行化（由 :class:`SQLiteDatabase` 保证）。"""

    def __init__(
        self,
        database: SQLiteDatabase,
        *,
        id_generator: UlidGenerator | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._database = database
        self._ids = id_generator if id_generator is not None else UlidGenerator()
        self._clock = clock if clock is not None else _default_clock
        database.bind_contract("run-registry", RUN_REGISTRY_CONTRACT_VERSION)

    @property
    def database(self) -> SQLiteDatabase:
        return self._database

    def new_run_id(self) -> str:
        """分配一个新的 ULID run_id（未落库；由 :meth:`register` 持久化）。"""
        return self._ids.new()

    def register(self, record: RunRecord) -> RunRecord:
        """严格插入；run_id 或 dedup_key 冲突即显式失败。"""
        with self._database.transaction() as connection:
            try:
                connection.execute(_INSERT_SQL, _record_params(record))
            except sqlite3.IntegrityError as exc:
                raise ArtifactError(f"run {record.run_id!r} conflicts with an existing run: {exc}") from exc
        return record

    def submit(self, record: RunRecord) -> tuple[RunRecord, bool]:
        """幂等提交（§35）：dedup_key 已存在时返回既有 run，返回 ``(record, created)``。

        先查后插存在竞态；插入冲突时按 dedup_key 重查一次，命中即视为并发提交的同一 job。
        """
        if record.dedup_key is not None:
            existing = self.find_by_dedup_key(record.dedup_key)
            if existing is not None:
                return existing, False
        try:
            self.register(record)
        except ArtifactError:
            if record.dedup_key is not None:
                existing = self.find_by_dedup_key(record.dedup_key)
                if existing is not None:
                    return existing, False
            raise
        return record, True

    def get(self, run_id: str) -> RunRecord:
        """按 run_id 取记录；不存在即显式失败（不做隐式默认）。"""
        record = self.find(run_id)
        if record is None:
            raise ArtifactError(f"run {run_id!r} is not registered")
        return record

    def find(self, run_id: str) -> RunRecord | None:
        """按 run_id 查（大小写规范化后查询，见 :func:`validate_ulid`）。"""
        canonical = validate_ulid(run_id)
        row = self._database.query_one(f"{_SELECT_SQL} WHERE run_id = ?", (canonical,))
        return None if row is None else _row_to_record(row)

    def find_by_dedup_key(self, dedup_key: str) -> RunRecord | None:
        key = _require_nonempty(dedup_key, "dedup_key")
        row = self._database.query_one(f"{_SELECT_SQL} WHERE dedup_key = ?", (key,))
        return None if row is None else _row_to_record(row)

    def update_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        error: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        now: datetime | None = None,
    ) -> RunRecord:
        """按状态机迁移 status；同状态重复写入幂等，非法迁移显式失败。

        读取 + 迁移在同一个 ``BEGIN IMMEDIATE`` 事务内完成，且 ``UPDATE`` 带
        ``status = <旧值>`` 守卫：并发下即使两个调用基于同一旧状态迁移也只有一个
        能成功，另一个显式失败，而不是互相覆盖或绕过终态不可变。
        """
        if status not in RUN_STATUSES:
            raise ConfigurationError(f"unknown run status: {status!r}")
        validate_ulid(run_id)
        updated_at = ensure_shanghai_aware(now, "now") if now is not None else self._clock()
        with self._database.transaction() as connection:
            row = connection.execute(f"{_SELECT_SQL} WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                raise ArtifactError(f"run {run_id!r} is not registered")
            current = _row_to_record(row)
            if status != current.status and status not in RUN_STATUS_TRANSITIONS[current.status]:
                raise ConfigurationError(
                    f"illegal run status transition {current.status!r} -> {status!r} "
                    f"for run {run_id!r}"
                )
            if updated_at < current.created_at:
                raise ConfigurationError(
                    f"updated_at {updated_at.isoformat()} precedes created_at "
                    f"{current.created_at.isoformat()}"
                )
            merged_metadata = dict(current.metadata)
            if metadata is not None:
                merged_metadata.update(metadata)
            new_error = (
                None if status == "succeeded" else (error if error is not None else current.error)
            )
            cursor = connection.execute(
                "UPDATE runs SET status = ?, updated_at = ?, error = ?, metadata_json = ? "
                "WHERE run_id = ? AND status = ?",
                (
                    status,
                    updated_at.isoformat(),
                    new_error,
                    canonical_json(merged_metadata),
                    run_id,
                    current.status,
                ),
            )
            if cursor.rowcount != 1:
                raise ArtifactError(
                    f"run {run_id!r} changed concurrently during update_status; retry"
                )
            updated = current.model_copy(
                update={
                    "status": status,
                    "updated_at": updated_at,
                    "error": new_error,
                    "metadata": merged_metadata,
                }
            )
        return updated

    def list_runs(
        self,
        *,
        kind: str | None = None,
        status: RunStatus | None = None,
        limit: int | None = None,
    ) -> list[RunRecord]:
        if status is not None and status not in RUN_STATUSES:
            raise ConfigurationError(f"unknown run status: {status!r}")
        if limit is not None and (isinstance(limit, bool) or limit < 1):
            raise ConfigurationError(f"limit must be a positive int, got {limit!r}")
        clauses: list[str] = []
        parameters: list[Any] = []
        if kind is not None:
            clauses.append("kind = ?")
            parameters.append(kind)
        if status is not None:
            clauses.append("status = ?")
            parameters.append(status)
        sql = _SELECT_SQL
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC, run_id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            parameters.append(limit)
        return [_row_to_record(row) for row in self._database.query(sql, parameters)]


def _record_params(record: RunRecord) -> tuple[Any, ...]:
    return (
        record.run_id,
        record.kind,
        record.status,
        record.created_at.isoformat(),
        record.updated_at.isoformat(),
        record.config_hash,
        record.dataset_hash,
        record.dedup_key,
        record.run_dir,
        record.error,
        canonical_json(record.metadata),
    )


def _row_to_record(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        run_id=row["run_id"],
        kind=row["kind"],
        status=row["status"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        config_hash=row["config_hash"],
        dataset_hash=row["dataset_hash"],
        dedup_key=row["dedup_key"],
        run_dir=row["run_dir"],
        error=row["error"],
        metadata=json.loads(row["metadata_json"]),
    )


__all__ = ["RUN_REGISTRY_CONTRACT_VERSION", "RunRecord", "RunRegistry"]
