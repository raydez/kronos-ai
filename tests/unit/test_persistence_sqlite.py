"""SQLite 持久化底座（§30；ADR-013）：WAL、schema 版本、串行化事务、约束。"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path

import pytest

from kronos_ai.domain.run import RUN_STATUSES
from kronos_ai.errors import ArtifactError
from kronos_ai.infrastructure.persistence.schema import (
    PERSISTENCE_SCHEMA_STATEMENTS,
    PERSISTENCE_SCHEMA_VERSION,
    open_database,
)
from kronos_ai.infrastructure.persistence.sqlite import SCHEMA_META_TABLE, SQLiteDatabase

_RUN_INSERT = (
    "INSERT INTO runs (run_id, kind, status, created_at, updated_at, run_dir, metadata_json) "
    "VALUES (?, ?, ?, ?, ?, ?, ?)"
)


@pytest.fixture
def database(tmp_path: Path) -> Iterator[SQLiteDatabase]:
    db = open_database(tmp_path / "index.sqlite3")
    try:
        yield db
    finally:
        db.close()


def test_file_database_uses_wal(database: SQLiteDatabase) -> None:
    mode = database.connection.execute("PRAGMA journal_mode").fetchone()[0]
    assert str(mode).lower() == "wal"
    assert database.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_memory_database_is_supported(tmp_path: Path) -> None:
    database = SQLiteDatabase(
        ":memory:",
        schema_version=PERSISTENCE_SCHEMA_VERSION,
        schema_statements=PERSISTENCE_SCHEMA_STATEMENTS,
    )
    try:
        assert database.path is None
        assert database.query_one(f"SELECT key FROM {SCHEMA_META_TABLE}") is not None
    finally:
        database.close()


def test_reopen_with_same_version_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite3"
    open_database(path).close()
    reopened = open_database(path)
    try:
        tables = {
            row["name"]
            for row in reopened.query("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert {"runs", "artifacts", "snapshots", SCHEMA_META_TABLE} <= tables
    finally:
        reopened.close()


def test_schema_version_mismatch_fails_explicitly(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite3"
    open_database(path).close()
    with pytest.raises(ArtifactError, match="schema_version"):
        SQLiteDatabase(
            path,
            schema_version="some-other-version",
            schema_statements=PERSISTENCE_SCHEMA_STATEMENTS,
        )


def test_negative_busy_timeout_rejected_before_creating_file(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite3"
    with pytest.raises(ArtifactError, match="busy_timeout_ms"):
        SQLiteDatabase(
            path,
            schema_version=PERSISTENCE_SCHEMA_VERSION,
            schema_statements=PERSISTENCE_SCHEMA_STATEMENTS,
            busy_timeout_ms=-1,
        )
    assert not path.exists()


def test_nested_transaction_is_rejected(database: SQLiteDatabase) -> None:
    with ExitStack() as stack, pytest.raises(ArtifactError, match="cannot be nested"):
        stack.enter_context(database.transaction())
        stack.enter_context(database.transaction())
    assert database.connection.in_transaction is False


def test_transaction_rolls_back_on_error(database: SQLiteDatabase) -> None:
    row = ("01ARZ3NDEKTSV4RRFFQ69G5FA0", "forecast", "pending", "t", "t", "runs/x", "{}")
    with pytest.raises(RuntimeError), database.transaction() as connection:
        connection.execute(_RUN_INSERT, row)
        raise RuntimeError("boom")
    assert database.query_one("SELECT 1 FROM runs WHERE run_id = ?", (row[0],)) is None


def test_query_one_returns_none_when_absent(database: SQLiteDatabase) -> None:
    assert database.query_one("SELECT 1 FROM runs WHERE run_id = ?", ("missing",)) is None


def test_status_check_constraint_uses_domain_statuses(database: SQLiteDatabase) -> None:
    for index, status in enumerate(RUN_STATUSES):
        run_id = f"01ARZ3NDEKTSV4RRFFQ69G5FA{index}"
        with database.transaction() as connection:
            connection.execute(
                _RUN_INSERT, (run_id, "forecast", status, "t", "t", f"runs/{run_id}", "{}")
            )
    with pytest.raises(sqlite3.IntegrityError), database.transaction() as connection:
        connection.execute(
            _RUN_INSERT,
            ("01ARZ3NDEKTSV4RRFFQ69G5FAZ", "forecast", "bogus", "t", "t", "runs/x", "{}"),
        )


def test_dedup_key_unique_index_allows_multiple_nulls(database: SQLiteDatabase) -> None:
    with database.transaction() as connection:
        connection.execute(
            _RUN_INSERT,
            ("01ARZ3NDEKTSV4RRFFQ69G5FA0", "forecast", "pending", "t", "t", "runs/a", "{}"),
        )
        connection.execute(
            _RUN_INSERT,
            ("01ARZ3NDEKTSV4RRFFQ69G5FA1", "forecast", "pending", "t", "t", "runs/b", "{}"),
        )
        connection.execute(
            "UPDATE runs SET dedup_key = 'dup' WHERE run_id = ?",
            ("01ARZ3NDEKTSV4RRFFQ69G5FA0",),
        )
    with pytest.raises(sqlite3.IntegrityError), database.transaction() as connection:
        connection.execute(
            "UPDATE runs SET dedup_key = 'dup' WHERE run_id = ?",
            ("01ARZ3NDEKTSV4RRFFQ69G5FA1",),
        )


def test_context_manager_closes(tmp_path: Path) -> None:
    with open_database(tmp_path / "index.sqlite3") as database:
        assert database.query_one("SELECT 1") is not None
    with pytest.raises(sqlite3.ProgrammingError):
        database.query("SELECT 1")


def test_bind_contract_persists_and_rejects_mismatch(database: SQLiteDatabase) -> None:
    database.bind_contract("widget", "widget-v1")
    database.bind_contract("widget", "widget-v1")  # 幂等
    row = database.query_one(
        f"SELECT value FROM {SCHEMA_META_TABLE} WHERE key = ?", ("widget",)
    )
    assert row is not None and row["value"] == "widget-v1"
    with pytest.raises(ArtifactError, match="binds widget"):
        database.bind_contract("widget", "widget-v2")


def test_concurrent_cold_start_of_fresh_database(tmp_path: Path) -> None:
    """Major 2 回归：多个进程/线程同时冷启动一个全新库不得抛未捕获的 sqlite3 异常。"""
    path = tmp_path / "index.sqlite3"
    barrier = threading.Barrier(6)
    failures: list[BaseException] = []
    lock = threading.Lock()

    def open_and_close() -> None:
        barrier.wait()
        try:
            database = open_database(path)
        except BaseException as exc:  # pragma: no cover - 回归失败路径
            with lock:
                failures.append(exc)
            return
        database.close()

    threads = [threading.Thread(target=open_and_close) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert failures == []
    final = open_database(path)
    try:
        row = final.query_one(
            f"SELECT value FROM {SCHEMA_META_TABLE} WHERE key = ?", ("schema_version",)
        )
        assert row is not None and row["value"] == PERSISTENCE_SCHEMA_VERSION
    finally:
        final.close()


def test_schema_version_mismatch_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite3"
    open_database(path).close()
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            f"UPDATE {SCHEMA_META_TABLE} SET value = ? WHERE key = ?",
            ("persistence-schema-v0", "schema_version"),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(ArtifactError, match="schema_version"):
        open_database(path)


def test_negative_busy_timeout_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="busy_timeout_ms"):
        SQLiteDatabase(
            tmp_path / "index.sqlite3",
            schema_version=PERSISTENCE_SCHEMA_VERSION,
            schema_statements=PERSISTENCE_SCHEMA_STATEMENTS,
            busy_timeout_ms=-1,
        )
