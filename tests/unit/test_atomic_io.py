"""原子写入原语（§15 / §30 共用）：无半成品、无残留临时文件、崩溃安全语义。"""

from __future__ import annotations

import hashlib
import stat
import threading
from pathlib import Path

import pytest

from kronos_ai.atomic_io import (
    FileDigest,
    atomic_create_bytes,
    atomic_create_file,
    atomic_write_bytes,
    atomic_write_text,
    digest_file,
)


def _temp_files(directory: Path) -> list[Path]:
    return [path for path in directory.rglob("*.tmp")]


def test_atomic_write_bytes_creates_parents_and_exact_content(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deep" / "payload.bin"
    atomic_write_bytes(path, b"hello")
    assert path.read_bytes() == b"hello"
    assert _temp_files(tmp_path) == []


def test_atomic_write_bytes_is_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "payload.json"
    atomic_write_bytes(path, b"{}")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_atomic_write_text_uses_utf8(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    atomic_write_text(path, "中文 ✓")
    assert path.read_text(encoding="utf-8") == "中文 ✓"


def test_atomic_write_bytes_overwrites_atomically(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    atomic_write_bytes(path, b"first")
    atomic_write_bytes(path, b"second")
    assert path.read_bytes() == b"second"
    assert _temp_files(tmp_path) == []


def test_atomic_write_bytes_cleans_temp_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "payload.bin"

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr("kronos_ai.atomic_io.os.replace", boom)
    with pytest.raises(OSError, match="simulated replace failure"):
        atomic_write_bytes(path, b"payload")
    assert path.exists() is False
    assert _temp_files(tmp_path) == []


def test_atomic_create_file_cleans_temp_on_writer_failure(tmp_path: Path) -> None:
    path = tmp_path / "table.parquet"

    def writer(target: Path) -> None:
        target.write_bytes(b"partial")
        raise RuntimeError("simulated writer failure")

    with pytest.raises(RuntimeError, match="simulated writer failure"):
        atomic_create_file(path, writer)
    assert path.exists() is False
    assert _temp_files(tmp_path) == []


def test_atomic_create_bytes_is_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    assert atomic_create_bytes(path, b"first") is True
    assert atomic_create_bytes(path, b"second") is False
    assert path.read_bytes() == b"first"
    assert _temp_files(tmp_path) == []


def test_atomic_create_bytes_does_not_overwrite_existing_without_temp(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"pre-existing")
    assert atomic_create_bytes(path, b"new") is False
    assert path.read_bytes() == b"pre-existing"
    assert _temp_files(tmp_path) == []


def test_atomic_create_bytes_concurrent_single_winner(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    barrier = threading.Barrier(8)
    results: list[bool] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        barrier.wait()
        created = atomic_create_bytes(path, f"payload-{index}".encode())
        with lock:
            results.append(created)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results.count(True) == 1
    assert len(results) == 8
    assert _temp_files(tmp_path) == []


def test_atomic_create_file_returns_digest_and_is_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "table.parquet"
    stamp = atomic_create_file(path, lambda target: target.write_bytes(b"payload"))
    assert stamp == FileDigest(hashlib.sha256(b"payload").hexdigest(), len(b"payload"))
    assert atomic_create_file(path, lambda target: target.write_bytes(b"other")) is None
    assert path.read_bytes() == b"payload"
    assert _temp_files(tmp_path) == []


def test_digest_file_matches_sha256_bytes(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    atomic_write_bytes(path, b"stream me")
    assert digest_file(path) == FileDigest(
        hashlib.sha256(b"stream me").hexdigest(), len(b"stream me")
    )
