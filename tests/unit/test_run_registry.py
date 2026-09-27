"""Run registry（§30–§32；ADR-013）：注册、幂等提交、状态机、过滤、持久化。"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from kronos_ai.domain.ids import encode_ulid
from kronos_ai.domain.time import CN_TZ
from kronos_ai.errors import ArtifactError, ConfigurationError
from kronos_ai.infrastructure.persistence.run_registry import RunRecord, RunRegistry
from kronos_ai.infrastructure.persistence.schema import open_database
from kronos_ai.infrastructure.persistence.sqlite import SQLiteDatabase

BASE_TIME = datetime(2026, 9, 25, 18, 0, tzinfo=CN_TZ)


def ulid(index: int) -> str:
    return encode_ulid(1_700_000_000_000 + index, bytes([index]) * 10)


def make_record(
    run_id: str,
    *,
    kind: str = "forecast",
    status: str = "pending",
    dedup_key: str | None = None,
    config_hash: str | None = None,
    dataset_hash: str | None = None,
    created_at: datetime | None = None,
) -> RunRecord:
    created = BASE_TIME if created_at is None else created_at
    return RunRecord(
        run_id=run_id,
        kind=kind,
        status=status,  # type: ignore[arg-type]
        created_at=created,
        updated_at=created,
        config_hash=config_hash,
        dataset_hash=dataset_hash,
        dedup_key=dedup_key,
        run_dir=f"runs/{run_id}",
    )


@pytest.fixture
def database(tmp_path: Path) -> Iterator[SQLiteDatabase]:
    db = open_database(tmp_path / "index.sqlite3")
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def registry(database: SQLiteDatabase) -> RunRegistry:
    return RunRegistry(database)


def test_register_and_get_round_trip(registry: RunRegistry) -> None:
    record = make_record(ulid(1), dedup_key="forecast:2026-09-25", config_hash="a" * 64)
    registry.register(record)
    fetched = registry.get(record.run_id)
    assert fetched == record
    assert fetched.run_dir == f"runs/{record.run_id}"


def test_get_unknown_run_raises(registry: RunRegistry) -> None:
    with pytest.raises(ArtifactError, match="not registered"):
        registry.get(ulid(99))
    assert registry.find(ulid(99)) is None


def test_register_duplicate_run_id_raises(registry: RunRegistry) -> None:
    record = make_record(ulid(2))
    registry.register(record)
    with pytest.raises(ArtifactError, match="conflicts"):
        registry.register(make_record(ulid(2), kind="benchmark"))


def test_register_duplicate_dedup_key_raises(registry: RunRegistry) -> None:
    registry.register(make_record(ulid(3), dedup_key="job-1"))
    with pytest.raises(ArtifactError, match="conflicts"):
        registry.register(make_record(ulid(4), dedup_key="job-1"))
    assert registry.find_by_dedup_key("job-1") is not None


def test_submit_is_idempotent_on_dedup_key(registry: RunRegistry) -> None:
    record = make_record(ulid(5), dedup_key="job-idempotent")
    first, created = registry.submit(record)
    assert created is True
    assert first.run_id == record.run_id

    again, created_again = registry.submit(
        make_record(ulid(6), dedup_key="job-idempotent", kind="forecast")
    )
    assert created_again is False
    assert again.run_id == record.run_id
    assert len(registry.list_runs(kind="forecast")) == 1


def test_submit_without_dedup_key_always_creates(registry: RunRegistry) -> None:
    _, first = registry.submit(make_record(ulid(7)))
    _, second = registry.submit(make_record(ulid(8)))
    assert (first, second) == (True, True)


def test_update_status_follows_state_machine(registry: RunRegistry) -> None:
    record = make_record(ulid(9))
    registry.register(record)
    now = BASE_TIME + timedelta(minutes=1)
    running = registry.update_status(record.run_id, "running", now=now)
    assert running.status == "running"
    succeeded = registry.update_status(record.run_id, "succeeded", now=now + timedelta(minutes=1))
    assert succeeded.status == "succeeded"
    assert succeeded.error is None


def test_update_status_rejects_illegal_transition(registry: RunRegistry) -> None:
    registry.register(make_record(ulid(10)))
    registry.update_status(ulid(10), "succeeded")
    with pytest.raises(ConfigurationError, match="illegal run status transition"):
        registry.update_status(ulid(10), "running")
    with pytest.raises(ConfigurationError, match="unknown run status"):
        registry.update_status(ulid(10), "completed")  # type: ignore[arg-type]


def test_update_status_same_status_is_idempotent(registry: RunRegistry) -> None:
    record = make_record(ulid(11))
    registry.register(record)
    now = BASE_TIME + timedelta(minutes=2)
    first = registry.update_status(record.run_id, "running", now=now)
    second = registry.update_status(record.run_id, "running", now=now + timedelta(minutes=1))
    assert second.status == "running"
    assert second.updated_at > first.updated_at


def test_terminal_status_is_immutable(registry: RunRegistry) -> None:
    registry.register(make_record(ulid(12)))
    registry.update_status(ulid(12), "running")
    registry.update_status(ulid(12), "failed", error="boom")
    failed = registry.get(ulid(12))
    assert failed.error == "boom"
    with pytest.raises(ConfigurationError, match="illegal run status transition"):
        registry.update_status(ulid(12), "running")
    assert registry.get(ulid(12)).status == "failed"


def test_succeeded_clears_error_and_merges_metadata(registry: RunRegistry) -> None:
    registry.register(make_record(ulid(13), dedup_key="job-13"))
    registry.update_status(ulid(13), "running", error="transient")
    done = registry.update_status(ulid(13), "succeeded", metadata={"rows": 42})
    assert done.error is None
    assert done.metadata == {"rows": 42}
    assert registry.get(ulid(13)).error is None


def test_update_status_rejects_timestamp_before_created(registry: RunRegistry) -> None:
    registry.register(make_record(ulid(14)))
    with pytest.raises(ConfigurationError, match="precedes created_at"):
        registry.update_status(ulid(14), "running", now=BASE_TIME - timedelta(seconds=1))


def test_list_runs_filters_and_orders(registry: RunRegistry) -> None:
    registry.register(make_record(ulid(20), kind="forecast", created_at=BASE_TIME))
    registry.register(
        make_record(ulid(21), kind="forecast", created_at=BASE_TIME + timedelta(minutes=1))
    )
    registry.register(
        make_record(ulid(22), kind="benchmark", created_at=BASE_TIME + timedelta(minutes=2))
    )
    registry.update_status(ulid(21), "running", now=BASE_TIME + timedelta(minutes=3))

    forecast_runs = registry.list_runs(kind="forecast")
    assert [item.run_id for item in forecast_runs] == [ulid(21), ulid(20)]

    running = registry.list_runs(status="running")
    assert [item.run_id for item in running] == [ulid(21)]

    limited = registry.list_runs(limit=1)
    assert [item.run_id for item in limited] == [ulid(22)]


def test_list_runs_rejects_bad_arguments(registry: RunRegistry) -> None:
    with pytest.raises(ConfigurationError, match="unknown run status"):
        registry.list_runs(status="bogus")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="positive int"):
        registry.list_runs(limit=0)


def test_registry_state_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite3"
    first_db = open_database(path)
    try:
        RunRegistry(first_db).register(make_record(ulid(30), dedup_key="job-30"))
    finally:
        first_db.close()

    second_db = open_database(path)
    try:
        reopened = RunRegistry(second_db).get(ulid(30))
    finally:
        second_db.close()
    assert reopened.dedup_key == "job-30"


def test_record_validation_rejects_bad_fields() -> None:
    with pytest.raises(ValidationError, match="sha256"):
        make_record(ulid(31), config_hash="not-a-hash")
    with pytest.raises(ValidationError, match="non-empty"):
        RunRecord(
            run_id=ulid(31),
            kind="  ",
            status="pending",
            created_at=BASE_TIME,
            updated_at=BASE_TIME,
            run_dir="runs/x",
        )
    with pytest.raises(ValidationError, match="ULID"):
        make_record("not-a-ulid")
    with pytest.raises(ValidationError, match="precedes created_at"):
        RunRecord(
            run_id=ulid(31),
            kind="forecast",
            status="pending",
            created_at=BASE_TIME,
            updated_at=BASE_TIME - timedelta(seconds=1),
            run_dir="runs/x",
        )


def test_record_validation_rejects_non_json_metadata() -> None:
    with pytest.raises(ValidationError, match="JSON-serializable"):
        RunRecord(
            run_id=ulid(32),
            kind="forecast",
            status="pending",
            created_at=BASE_TIME,
            updated_at=BASE_TIME,
            run_dir="runs/x",
            metadata={"bad": {1, 2, 3}},
        )


def test_new_run_id_is_ulid(registry: RunRegistry) -> None:
    from kronos_ai.domain.ids import is_ulid

    assert is_ulid(registry.new_run_id())


def test_record_rejects_run_dir_drift() -> None:
    with pytest.raises(ValidationError, match="run_dir must be"):
        RunRecord(
            run_id=ulid(33),
            kind="forecast",
            status="pending",
            created_at=BASE_TIME,
            updated_at=BASE_TIME,
            run_dir="runs/whatever",
        )


def test_lookup_normalizes_lowercase_run_id(registry: RunRegistry) -> None:
    record = make_record(ulid(34))
    registry.register(record)
    lowered = record.run_id.lower()
    assert lowered != record.run_id
    assert registry.get(lowered).run_id == record.run_id
    assert registry.find(lowered) is not None


def test_update_status_rejects_concurrent_change(
    registry: RunRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """守卫回归：读到的旧状态与 UPDATE 时的实际状态不一致必须显式失败。

    RLock 让同进程不出现这种交错，因此这里直接篡改校验到的状态模拟另一个已提交的 writer。
    """
    from kronos_ai.infrastructure.persistence import run_registry as module

    record = make_record(ulid(35))
    registry.register(record)
    registry.update_status(record.run_id, "running", now=BASE_TIME + timedelta(minutes=1))

    real = module._row_to_record

    def stale(row: object) -> RunRecord:
        current = real(row)  # type: ignore[arg-type]
        return current.model_copy(update={"status": "pending"})

    monkeypatch.setattr(module, "_row_to_record", stale)
    with pytest.raises(ArtifactError, match="changed concurrently"):
        registry.update_status(record.run_id, "failed", now=BASE_TIME + timedelta(minutes=2))
