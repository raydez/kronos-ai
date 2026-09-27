"""Artifact Store（§30–§33；ADR-013）：Parquet/JSON/YAML/text、完整性、append-only 快照。"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from kronos_ai.domain.hashing import sha256_bytes
from kronos_ai.domain.ids import encode_ulid
from kronos_ai.domain.time import CN_TZ
from kronos_ai.errors import ArtifactError, ConfigurationError
from kronos_ai.infrastructure.persistence.artifact_store import (
    MEDIA_TYPE_JSON,
    MEDIA_TYPE_MARKDOWN,
    MEDIA_TYPE_PARQUET,
    ArtifactStore,
    frame_schema_hash,
    run_dir_relative,
)
from kronos_ai.infrastructure.persistence.schema import open_database
from kronos_ai.infrastructure.persistence.sqlite import SQLiteDatabase

RUN_ID = encode_ulid(1_700_000_000_000, b"\x01" * 10)
OTHER_RUN_ID = encode_ulid(1_700_000_001_000, b"\x02" * 10)


@pytest.fixture
def database(tmp_path: Path) -> Iterator[SQLiteDatabase]:
    db = open_database(tmp_path / "artifacts" / "index.sqlite3")
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def store(tmp_path: Path, database: SQLiteDatabase) -> ArtifactStore:
    return ArtifactStore(tmp_path / "artifacts", database)


def test_write_json_round_trip_and_index(store: ArtifactStore) -> None:
    record = store.write_json(RUN_ID, "metadata", {"kind": "forecast", "n": 3})
    assert record.media_type == MEDIA_TYPE_JSON
    assert record.relative_path == f"{run_dir_relative(RUN_ID)}/metadata.json"
    assert record.size_bytes > 0
    assert len(record.sha256) == 64
    assert store.read_json(RUN_ID, "metadata") == {"kind": "forecast", "n": 3}
    assert store.artifact_path(RUN_ID, "metadata").is_file()


def test_write_yaml_round_trip(store: ArtifactStore) -> None:
    store.write_yaml(RUN_ID, "config", {"seed": 7, "nested": {"a": [1, 2]}})
    assert store.read_yaml(RUN_ID, "config") == {"seed": 7, "nested": {"a": [1, 2]}}
    assert store.get_artifact(RUN_ID, "config").media_type == "application/yaml"


def test_write_text_is_markdown(store: ArtifactStore) -> None:
    record = store.write_text(RUN_ID, "report", "# Title\n")
    assert record.media_type == MEDIA_TYPE_MARKDOWN
    assert record.relative_path.endswith("report.md")
    assert store.read_text(RUN_ID, "report") == "# Title\n"


def test_write_bytes_keeps_filename_and_media_type(store: ArtifactStore) -> None:
    store.write_bytes(
        RUN_ID, "raw", b"\x00\x01", filename="raw.bin", media_type="application/octet-stream"
    )
    assert store.read_bytes(RUN_ID, "raw") == b"\x00\x01"
    assert store.artifact_path(RUN_ID, "raw").name == "raw.bin"


def test_write_parquet_round_trip_and_metadata(store: ArtifactStore) -> None:
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    record = store.write_parquet(RUN_ID, "universe", frame)
    assert record.media_type == MEDIA_TYPE_PARQUET
    assert record.row_count == 3
    assert record.schema_hash == frame_schema_hash(frame)
    pd.testing.assert_frame_equal(store.read_parquet(RUN_ID, "universe"), frame)
    assert store.artifact_path(RUN_ID, "universe").is_file()


def test_write_once_rejects_duplicate_name(store: ArtifactStore) -> None:
    store.write_json(RUN_ID, "metadata", {"a": 1})
    with pytest.raises(ArtifactError, match="write-once"):
        store.write_json(RUN_ID, "metadata", {"a": 2})
    # 原内容不被覆盖。
    assert store.read_json(RUN_ID, "metadata") == {"a": 1}


def test_get_missing_artifact_raises(store: ArtifactStore) -> None:
    with pytest.raises(ArtifactError, match="has no artifact"):
        store.get_artifact(RUN_ID, "nope")


def test_list_artifacts_sorted_by_name(store: ArtifactStore) -> None:
    store.write_json(RUN_ID, "zeta", {})
    store.write_json(RUN_ID, "alpha", {})
    store.write_json(OTHER_RUN_ID, "metadata", {})
    assert [item.name for item in store.list_artifacts(RUN_ID)] == ["alpha", "zeta"]


def test_read_detects_tampered_file(store: ArtifactStore) -> None:
    store.write_json(RUN_ID, "metadata", {"a": 1})
    path = store.artifact_path(RUN_ID, "metadata")
    path.write_bytes(b'{"a": 2}')
    with pytest.raises(ArtifactError, match="corrupt"):
        store.read_json(RUN_ID, "metadata")
    with pytest.raises(ArtifactError, match="corrupt"):
        store.verify_artifact(RUN_ID, "metadata")


def test_read_detects_missing_file(store: ArtifactStore) -> None:
    store.write_json(RUN_ID, "metadata", {"a": 1})
    store.artifact_path(RUN_ID, "metadata").unlink()
    with pytest.raises(ArtifactError, match="missing"):
        store.read_bytes(RUN_ID, "metadata")


@pytest.mark.parametrize("name", ["../evil", "a/b", ".hidden", "", "a\\b"])
def test_artifact_name_rejects_path_traversal(store: ArtifactStore, name: str) -> None:
    with pytest.raises(ValueError, match="safe path segment"):
        store.write_json(RUN_ID, name, {})


def test_invalid_run_id_raises_configuration_error(store: ArtifactStore) -> None:
    with pytest.raises(ConfigurationError, match="ULID"):
        store.run_dir("not-a-ulid")


def test_ensure_run_is_idempotent(store: ArtifactStore) -> None:
    first = store.ensure_run(RUN_ID)
    second = store.ensure_run(RUN_ID)
    assert first == second
    assert first.is_dir()
    assert first == store.run_dir(RUN_ID)


def test_records_survive_reopen(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    path = root / "index.sqlite3"
    first_db = open_database(path)
    try:
        ArtifactStore(root, first_db).write_json(RUN_ID, "metadata", {"a": 1})
    finally:
        first_db.close()

    second_db = open_database(path)
    try:
        reopened = ArtifactStore(root, second_db)
        assert reopened.read_json(RUN_ID, "metadata") == {"a": 1}
        assert reopened.get_artifact(RUN_ID, "metadata").run_id == RUN_ID
    finally:
        second_db.close()


def test_frame_schema_hash_tracks_columns_and_dtypes() -> None:
    base = pd.DataFrame({"a": [1, 2]})
    renamed = pd.DataFrame({"b": [1, 2]})
    retyped = pd.DataFrame({"a": [1.0, 2.0]})
    assert frame_schema_hash(base) == frame_schema_hash(base.copy())
    assert frame_schema_hash(base) != frame_schema_hash(renamed)
    assert frame_schema_hash(base) != frame_schema_hash(retyped)
    assert len(frame_schema_hash(base)) == 64


# ---- append-only snapshots（§30） -------------------------------------------------


def test_snapshot_first_write_creates(store: ArtifactStore) -> None:
    record, created = store.write_snapshot(
        "dataset-v1", "baostock-2026-09-25.json", b'{"rows": 1}', media_type=MEDIA_TYPE_JSON
    )
    assert created is True
    assert record.dataset_version == "dataset-v1"
    assert record.relative_path == "snapshots/dataset-v1/baostock-2026-09-25.json"
    assert store.read_snapshot("dataset-v1", "baostock-2026-09-25.json") == b'{"rows": 1}'


def test_snapshot_same_content_is_idempotent(store: ArtifactStore) -> None:
    first, created = store.write_snapshot("dataset-v1", "raw.json", b"payload")
    second, created_again = store.write_snapshot("dataset-v1", "raw.json", b"payload")
    assert created is True
    assert created_again is False
    assert second == first
    assert len(store.list_snapshots("dataset-v1")) == 1


def test_snapshot_different_content_is_rejected(store: ArtifactStore) -> None:
    store.write_snapshot("dataset-v1", "raw.json", b"payload")
    with pytest.raises(ArtifactError, match="append-only"):
        store.write_snapshot("dataset-v1", "raw.json", b"changed")
    assert store.read_snapshot("dataset-v1", "raw.json") == b"payload"


def test_snapshot_orphan_file_with_other_content_is_rejected(store: ArtifactStore) -> None:
    directory = store.snapshot_dir("dataset-v1")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "raw.json").write_bytes(b"orphan")
    with pytest.raises(ArtifactError, match="different digest"):
        store.write_snapshot("dataset-v1", "raw.json", b"payload")
    # 拒绝覆盖：旧文件必须原样保留
    assert (directory / "raw.json").read_bytes() == b"orphan"
    assert store.find_snapshot("dataset-v1", "raw.json") is None


def test_snapshot_orphan_file_with_same_content_is_adopted(store: ArtifactStore) -> None:
    """崩溃遗留的未索引孤儿文件：内容一致则收养，不再永久毒化该名字。"""
    directory = store.snapshot_dir("dataset-v1")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "raw.json").write_bytes(b"payload")
    record, created = store.write_snapshot("dataset-v1", "raw.json", b"payload")
    assert created is True
    assert record.sha256 == sha256_bytes(b"payload")
    assert store.read_snapshot("dataset-v1", "raw.json") == b"payload"


def test_snapshot_indexed_but_file_missing_is_corrupt(store: ArtifactStore) -> None:
    record, _ = store.write_snapshot("dataset-v1", "raw.json", b"payload")
    store.snapshot_path("dataset-v1", "raw.json").unlink()
    with pytest.raises(ArtifactError, match="missing"):
        store.write_snapshot("dataset-v1", "raw.json", b"payload")
    assert store.find_snapshot("dataset-v1", "raw.json") == record


def test_snapshot_missing_read_raises(store: ArtifactStore) -> None:
    with pytest.raises(ArtifactError, match="has no snapshot"):
        store.read_snapshot("dataset-v1", "nope")


def test_snapshot_validation(store: ArtifactStore) -> None:
    with pytest.raises(ValueError, match="safe path segment"):
        store.write_snapshot("dataset-v1", "../escape", b"x")
    with pytest.raises(ValueError, match="safe path segment"):
        store.write_snapshot("bad/version", "raw.json", b"x")


def test_snapshot_survives_reopen(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    path = root / "index.sqlite3"
    first_db = open_database(path)
    try:
        store = ArtifactStore(root, first_db)
        store.write_snapshot("dataset-v1", "raw.json", b"payload")
    finally:
        first_db.close()

    second_db = open_database(path)
    try:
        reopened = ArtifactStore(root, second_db)
        assert reopened.read_snapshot("dataset-v1", "raw.json") == b"payload"
        record, created = reopened.write_snapshot("dataset-v1", "raw.json", b"payload")
        assert created is False
        assert record.sha256
    finally:
        second_db.close()


def test_metadata_must_be_json_serializable_leaves_no_orphan(store: ArtifactStore) -> None:
    with pytest.raises(ValueError, match="JSON-serializable"):
        store.write_json(RUN_ID, "metadata", {}, metadata={"bad": {1, 2}})
    assert store.find_artifact(RUN_ID, "metadata") is None
    assert not (store.run_dir(RUN_ID) / "metadata.json").exists()


def test_invalid_media_type_leaves_no_orphan(store: ArtifactStore) -> None:
    with pytest.raises(ConfigurationError, match="media_type"):
        store.write_bytes(RUN_ID, "bad", b"payload", filename="bad.bin", media_type="  ")
    assert store.find_artifact(RUN_ID, "bad") is None
    assert not (store.run_dir(RUN_ID) / "bad.bin").exists()


def test_duplicate_name_with_new_filename_leaves_no_orphan(store: ArtifactStore) -> None:
    """write-once 拒绝路径不得遗留未索引孤儿文件（避免永久占位）。"""
    store.write_bytes(RUN_ID, "x", b"A", filename="a.bin")
    with pytest.raises(ArtifactError, match="write-once"):
        store.write_bytes(RUN_ID, "x", b"B", filename="b.bin")
    assert not (store.run_dir(RUN_ID) / "b.bin").exists()
    assert (store.run_dir(RUN_ID) / "a.bin").read_bytes() == b"A"
    assert store.read_bytes(RUN_ID, "x") == b"A"


def test_json_canonical_encoding_is_stable(store: ArtifactStore) -> None:
    store.write_json(RUN_ID, "metadata", {"b": 2, "a": 1})
    raw = store.artifact_path(RUN_ID, "metadata").read_bytes()
    assert json.loads(raw.decode()) == {"a": 1, "b": 2}
    assert raw.decode() == '{"a":1,"b":2}'


def test_store_root_created_on_init(tmp_path: Path, database: SQLiteDatabase) -> None:
    root = tmp_path / "fresh" / "artifacts"
    ArtifactStore(root, database)
    assert root.is_dir()


def test_clock_is_injectable(tmp_path: Path, database: SQLiteDatabase) -> None:
    fixed = datetime(2026, 9, 25, 18, 0, tzinfo=CN_TZ)
    store = ArtifactStore(tmp_path / "artifacts", database, clock=lambda: fixed)
    record = store.write_json(RUN_ID, "metadata", {})
    assert record.created_at == fixed


def test_concurrent_write_of_same_artifact_has_single_winner(tmp_path: Path) -> None:
    """Major 1 回归：并发写同一 (run_id, name) 时索引与文件 sha256 必须一致。"""
    database = open_database(tmp_path / "index.sqlite3")
    try:
        store = ArtifactStore(tmp_path / "artifacts", database)
        barrier = threading.Barrier(8)
        outcomes: list[object] = []
        lock = threading.Lock()

        def worker(payload: bytes) -> None:
            barrier.wait()
            try:
                record: object = store.write_bytes(RUN_ID, "data", payload, filename="data.bin")
            except ArtifactError as exc:  # 期望：败者拿到显式 write-once 错误
                record = exc
            with lock:
                outcomes.append(record)

        threads = [
            threading.Thread(target=worker, args=(f"payload-{index}".encode(),))
            for index in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        winners = [item for item in outcomes if not isinstance(item, ArtifactError)]
        assert len(winners) == 1
        assert len(outcomes) == 8
        stored = store.artifact_path(RUN_ID, "data").read_bytes()
        record = store.get_artifact(RUN_ID, "data")
        assert record.sha256 == sha256_bytes(stored)
        assert record.size_bytes == len(stored)
        assert store.verify_artifact(RUN_ID, "data") == record
    finally:
        database.close()


def test_concurrent_write_of_same_snapshot_is_idempotent(tmp_path: Path) -> None:
    database = open_database(tmp_path / "index.sqlite3")
    try:
        store = ArtifactStore(tmp_path / "artifacts", database)
        barrier = threading.Barrier(6)
        failures: list[ArtifactError] = []
        created_flags: list[bool] = []
        lock = threading.Lock()

        def worker() -> None:
            barrier.wait()
            try:
                _record, created = store.write_snapshot("dataset-v1", "raw.json", b"payload")
            except ArtifactError as exc:
                with lock:
                    failures.append(exc)
            else:
                with lock:
                    created_flags.append(created)

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert failures == []
        assert len(created_flags) == 6
        # 只有第一个 writer 真正建文件；其余要么被收养（created=True）要么走幂等读。
        assert True in created_flags
        assert store.read_snapshot("dataset-v1", "raw.json") == b"payload"
        assert len(store.list_snapshots("dataset-v1")) == 1
    finally:
        database.close()
