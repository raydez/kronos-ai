"""SQLite schema（§30：run registry + artifact index + raw snapshot index）。

run registry 与 artifact index 共用一个数据库文件（§30 把两者都归入 SQLite），因此
schema 版本、DDL、连接在这里集中定义：:func:`open_database` 是唯一装配点，避免
registry 与 store 各自建表后版本漂移。
"""

from __future__ import annotations

from pathlib import Path

from kronos_ai.domain.run import RUN_STATUSES
from kronos_ai.infrastructure.persistence.sqlite import SQLiteDatabase

PERSISTENCE_SCHEMA_VERSION = "persistence-schema-v1"

_STATUS_CHECK = ", ".join(f"'{status}'" for status in RUN_STATUSES)

RUNS_TABLE_DDL = f"""
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ({_STATUS_CHECK})),
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    config_hash   TEXT,
    dataset_hash  TEXT,
    dedup_key     TEXT,
    run_dir       TEXT NOT NULL,
    error         TEXT,
    metadata_json TEXT NOT NULL
)
"""

# 同一 dedup_key 只允许一个 run（§35 的幂等提交）；NULL 不参与唯一约束。
RUNS_DEDUP_INDEX_DDL = (
    "CREATE UNIQUE INDEX IF NOT EXISTS runs_dedup_key_idx "
    "ON runs(dedup_key) WHERE dedup_key IS NOT NULL"
)
RUNS_KIND_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS runs_kind_created_idx ON runs(kind, created_at DESC)"
)
RUNS_STATUS_INDEX_DDL = "CREATE INDEX IF NOT EXISTS runs_status_idx ON runs(status)"

ARTIFACTS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS artifacts (
    run_id        TEXT NOT NULL,
    name          TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    media_type    TEXT NOT NULL,
    sha256        TEXT NOT NULL,
    size_bytes    INTEGER NOT NULL CHECK (size_bytes >= 0),
    row_count     INTEGER CHECK (row_count IS NULL OR row_count >= 0),
    schema_hash   TEXT,
    created_at    TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    PRIMARY KEY (run_id, name)
)
"""
ARTIFACTS_RUN_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS artifacts_run_idx ON artifacts(run_id)"
)

# §30：raw provider response 是 append-only 快照，按 dataset_version 归档、永不覆盖。
SNAPSHOTS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS snapshots (
    dataset_version TEXT NOT NULL,
    name            TEXT NOT NULL,
    relative_path   TEXT NOT NULL,
    media_type      TEXT NOT NULL,
    sha256          TEXT NOT NULL,
    size_bytes      INTEGER NOT NULL CHECK (size_bytes >= 0),
    created_at      TEXT NOT NULL,
    metadata_json   TEXT NOT NULL,
    PRIMARY KEY (dataset_version, name)
)
"""

PERSISTENCE_SCHEMA_STATEMENTS: tuple[str, ...] = (
    RUNS_TABLE_DDL,
    RUNS_DEDUP_INDEX_DDL,
    RUNS_KIND_INDEX_DDL,
    RUNS_STATUS_INDEX_DDL,
    ARTIFACTS_TABLE_DDL,
    ARTIFACTS_RUN_INDEX_DDL,
    SNAPSHOTS_TABLE_DDL,
)


def open_database(path: Path | str) -> SQLiteDatabase:
    """按当前 schema 版本打开（必要时初始化）持久化数据库。"""
    return SQLiteDatabase(
        path,
        schema_version=PERSISTENCE_SCHEMA_VERSION,
        schema_statements=PERSISTENCE_SCHEMA_STATEMENTS,
    )
