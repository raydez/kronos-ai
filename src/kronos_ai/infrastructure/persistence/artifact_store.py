"""Artifact Store：Parquet + SQLite index + append-only 快照（§30–§33；ADR-013）。

职责：
- 每个 run 一个目录 ``<root>/runs/<ULID>/``（§33），写入走「原子写 + index 登记」；
- 表格产物落 Parquet，其它产物落 JSON / YAML / Markdown / 原始字节；
- SQLite ``artifacts`` 表登记每个产物的相对路径、media type、sha256、大小、行数与
  schema hash（§30 artifact index）；
- §30 要求 raw provider response 采用 append-only 快照：``<root>/snapshots/<dataset_version>/``
  首次写入即固定，之后**永不覆盖**——内容相同幂等返回，内容不同显式失败。

边界：本模块只做「给定 run_id / dataset_version 的可靠落盘与登记」，不负责构造
§32 run metadata，也不负责在 forecast/benchmark 流程里创建 run（RX-KAI-019）。
"""

from __future__ import annotations

import io
import json
import re
import sqlite3
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from kronos_ai.atomic_io import atomic_create_bytes, atomic_create_file, digest_file
from kronos_ai.domain.hashing import canonical_json, is_sha256_hex, sha256_bytes, sha256_hex
from kronos_ai.domain.ids import validate_ulid
from kronos_ai.domain.time import CN_TZ, ensure_shanghai_aware
from kronos_ai.errors import ArtifactError, ConfigurationError
from kronos_ai.infrastructure.persistence.sqlite import SQLiteDatabase

if TYPE_CHECKING:  # pandas 只在真正读写 Parquet 时加载（CLI --help 不应触发）
    import pandas as pd

ARTIFACT_STORE_CONTRACT_VERSION = "artifact-store-v1"

RUNS_SUBDIR = "runs"
SNAPSHOTS_SUBDIR = "snapshots"

MEDIA_TYPE_JSON = "application/json"
MEDIA_TYPE_YAML = "application/yaml"
MEDIA_TYPE_PARQUET = "application/vnd.apache.parquet"
MEDIA_TYPE_MARKDOWN = "text/markdown"
MEDIA_TYPE_OCTET_STREAM = "application/octet-stream"

_MAX_TOKEN_LENGTH = 128
# 单段安全 token：首字符字母/数字，此后允许 . _ -；不含路径分隔符，故不可能逃出目录。
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_ARTIFACT_COLUMNS = (
    "run_id, name, relative_path, media_type, sha256, size_bytes, row_count, "
    "schema_hash, created_at, metadata_json"
)
_ARTIFACT_INSERT_SQL = (
    f"INSERT INTO artifacts ({_ARTIFACT_COLUMNS}) VALUES ({', '.join('?' * 10)})"
)
_ARTIFACT_SELECT_SQL = f"SELECT {_ARTIFACT_COLUMNS} FROM artifacts"
_SNAPSHOT_COLUMNS = (
    "dataset_version, name, relative_path, media_type, sha256, size_bytes, "
    "created_at, metadata_json"
)
_SNAPSHOT_INSERT_SQL = (
    f"INSERT INTO snapshots ({_SNAPSHOT_COLUMNS}) VALUES ({', '.join('?' * 8)})"
)
_SNAPSHOT_SELECT_SQL = f"SELECT {_SNAPSHOT_COLUMNS} FROM snapshots"


def _default_clock() -> datetime:
    return datetime.now(CN_TZ)


def _validate_token(value: str, field: str) -> str:
    if not isinstance(value, str) or not _SAFE_TOKEN.match(value) or len(value) > _MAX_TOKEN_LENGTH:
        raise ValueError(
            f"{field} {value!r} must be a single safe path segment "
            f"(letters/digits then [A-Za-z0-9._-], <= {_MAX_TOKEN_LENGTH} chars)"
        )
    return value


def run_dir_relative(run_id: str) -> str:
    """run 目录相对 artifact root 的路径（§33）；registry 与 store 共用。"""
    return f"{RUNS_SUBDIR}/{validate_ulid(run_id)}"


def snapshot_relative_path(dataset_version: str, name: str) -> str:
    """快照文件相对 artifact root 的路径（§30）；append-only 目录布局的唯一真源。"""
    version = _validate_token(dataset_version, "dataset_version")
    filename = _validate_token(name, "snapshot name")
    return f"{SNAPSHOTS_SUBDIR}/{version}/{filename}"


def frame_schema_hash(frame: pd.DataFrame) -> str:
    """表格 schema 指纹（列名 + dtype 顺序），用于 index 层快速判断结构是否变化。

    注意：这不是内容 hash。Parquet 字节含压缩 / 元数据，跨写入不保证逐字节一致，
    因此 :attr:`ArtifactRecord.sha256` 只作为**存储完整性**校验；逻辑内容 hash
    由调用方（如 RX-KAI-017 dataset_hash）按领域语义单独定义。
    """
    return sha256_hex(
        {
            "columns": [str(column) for column in frame.columns],
            "dtypes": [str(dtype) for dtype in frame.dtypes],
        }
    )


def _validate_metadata(value: dict[str, Any]) -> dict[str, Any]:
    try:
        canonical_json(value)
    except TypeError as exc:
        raise ValueError(f"artifact metadata must be JSON-serializable: {exc}") from exc
    return value


def _validated_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """落盘**前**校验 metadata，避免写入一个无法登记到 index 的孤儿文件。"""
    payload = dict(metadata) if metadata is not None else {}
    return _validate_metadata(payload)


class ArtifactRecord(BaseModel):
    """artifact index 的一行（§30）；``relative_path`` 相对 artifact root。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    name: str
    relative_path: str
    media_type: str
    sha256: str
    size_bytes: int = Field(ge=0)
    row_count: int | None = Field(default=None, ge=0)
    schema_hash: str | None = None
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("run_id")
    @classmethod
    def _run_id_ulid(cls, value: str) -> str:
        try:
            return validate_ulid(value)
        except ConfigurationError as exc:  # pydantic 只把 ValueError 归为 ValidationError
            raise ValueError(str(exc)) from exc

    @field_validator("name")
    @classmethod
    def _name_safe(cls, value: str) -> str:
        return _validate_token(value, "artifact name")

    @field_validator("relative_path", "media_type")
    @classmethod
    def _nonempty(cls, value: str, info: ValidationInfo) -> str:
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        return value

    @field_validator("sha256")
    @classmethod
    def _sha256(cls, value: str) -> str:
        if not is_sha256_hex(value):
            raise ValueError("sha256 must be a 64-char lowercase sha256 hex digest")
        return value

    @field_validator("metadata")
    @classmethod
    def _metadata_serializable(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_metadata(value)

    @field_validator("created_at")
    @classmethod
    def _created_shanghai(cls, value: datetime) -> datetime:
        return ensure_shanghai_aware(value, "created_at")


class SnapshotRecord(BaseModel):
    """append-only 快照 index 的一行（§30）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_version: str
    name: str
    relative_path: str
    media_type: str
    sha256: str
    size_bytes: int = Field(ge=0)
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("dataset_version", "name")
    @classmethod
    def _safe_token(cls, value: str, info: ValidationInfo) -> str:
        return _validate_token(value, str(info.field_name))

    @field_validator("relative_path", "media_type")
    @classmethod
    def _nonempty(cls, value: str, info: ValidationInfo) -> str:
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        return value

    @field_validator("sha256")
    @classmethod
    def _sha256(cls, value: str) -> str:
        if not is_sha256_hex(value):
            raise ValueError("sha256 must be a 64-char lowercase sha256 hex digest")
        return value

    @field_validator("metadata")
    @classmethod
    def _metadata_serializable(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_metadata(value)

    @field_validator("created_at")
    @classmethod
    def _created_shanghai(cls, value: datetime) -> datetime:
        return ensure_shanghai_aware(value, "created_at")


class ArtifactStore:
    """§30–§33 的本地 Artifact Store。"""

    def __init__(
        self,
        root: Path | str,
        database: SQLiteDatabase,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._root = Path(root)
        self._database = database
        self._clock = clock if clock is not None else _default_clock
        database.bind_contract("artifact-store", ARTIFACT_STORE_CONTRACT_VERSION)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def database(self) -> SQLiteDatabase:
        return self._database

    # ---- 路径 ----------------------------------------------------------------

    def run_dir(self, run_id: str) -> Path:
        return self._root / RUNS_SUBDIR / validate_ulid(run_id)

    def ensure_run(self, run_id: str) -> Path:
        """创建（幂等）run 目录；run 本身是否已登记由 :class:`RunRegistry` 负责。"""
        directory = self.run_dir(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def snapshot_dir(self, dataset_version: str) -> Path:
        return self._root / SNAPSHOTS_SUBDIR / _validate_token(dataset_version, "dataset_version")

    def artifact_relative_path(self, run_id: str, filename: str) -> str:
        return f"{run_dir_relative(run_id)}/{_validate_token(filename, 'filename')}"

    def _run_artifact_path(self, run_id: str, filename: str) -> Path:
        _validate_token(filename, "filename")
        return self.ensure_run(run_id) / filename

    # ---- run artifacts：写入 --------------------------------------------------

    def write_json(
        self, run_id: str, name: str, payload: Any, *, metadata: Mapping[str, Any] | None = None
    ) -> ArtifactRecord:
        _validate_token(name, "artifact name")
        data = canonical_json(payload).encode("utf-8")
        return self._persist_bytes(
            run_id, name, f"{name}.json", data, media_type=MEDIA_TYPE_JSON, metadata=metadata
        )

    def write_yaml(
        self, run_id: str, name: str, payload: Any, *, metadata: Mapping[str, Any] | None = None
    ) -> ArtifactRecord:
        _validate_token(name, "artifact name")
        data = yaml.safe_dump(payload, sort_keys=True, allow_unicode=True).encode("utf-8")
        return self._persist_bytes(
            run_id, name, f"{name}.yaml", data, media_type=MEDIA_TYPE_YAML, metadata=metadata
        )

    def write_text(
        self, run_id: str, name: str, text: str, *, metadata: Mapping[str, Any] | None = None
    ) -> ArtifactRecord:
        """Markdown / 纯文本产物（§33 ``report.md``）。"""
        _validate_token(name, "artifact name")
        return self._persist_bytes(
            run_id,
            name,
            f"{name}.md",
            text.encode("utf-8"),
            media_type=MEDIA_TYPE_MARKDOWN,
            metadata=metadata,
        )

    def write_bytes(
        self,
        run_id: str,
        name: str,
        data: bytes,
        *,
        filename: str,
        media_type: str = MEDIA_TYPE_OCTET_STREAM,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRecord:
        _validate_token(name, "artifact name")
        return self._persist_bytes(
            run_id, name, filename, data, media_type=media_type, metadata=metadata
        )

    def write_parquet(
        self,
        run_id: str,
        name: str,
        frame: pd.DataFrame,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRecord:
        """表格产物（§33 universe/forecast/samples/states/decisions.parquet）。"""
        _validate_token(name, "artifact name")
        schema_hash = frame_schema_hash(frame)
        return self._persist_file(
            run_id,
            name,
            f"{name}.parquet",
            lambda tmp: frame.to_parquet(tmp, engine="pyarrow", index=False),
            media_type=MEDIA_TYPE_PARQUET,
            row_count=len(frame),
            schema_hash=schema_hash,
            metadata=metadata,
        )

    # ---- run artifacts：读取 --------------------------------------------------

    def find_artifact(self, run_id: str, name: str) -> ArtifactRecord | None:
        row = self._database.query_one(
            f"{_ARTIFACT_SELECT_SQL} WHERE run_id = ? AND name = ?", (run_id, name)
        )
        return None if row is None else _row_to_artifact(row)

    def get_artifact(self, run_id: str, name: str) -> ArtifactRecord:
        record = self.find_artifact(run_id, name)
        if record is None:
            raise ArtifactError(f"run {run_id!r} has no artifact {name!r}")
        return record

    def list_artifacts(self, run_id: str) -> list[ArtifactRecord]:
        validate_ulid(run_id)
        rows = self._database.query(
            f"{_ARTIFACT_SELECT_SQL} WHERE run_id = ? ORDER BY name", (run_id,)
        )
        return [_row_to_artifact(row) for row in rows]

    def artifact_path(self, run_id: str, name: str) -> Path:
        record = self.get_artifact(run_id, name)
        return self._root / record.relative_path

    def read_bytes(self, run_id: str, name: str) -> bytes:
        record = self.get_artifact(run_id, name)
        return self._read_verified(self._root / record.relative_path, record.sha256)

    def read_json(self, run_id: str, name: str) -> Any:
        return json.loads(self.read_bytes(run_id, name).decode("utf-8"))

    def read_yaml(self, run_id: str, name: str) -> Any:
        return yaml.safe_load(self.read_bytes(run_id, name).decode("utf-8"))

    def read_text(self, run_id: str, name: str) -> str:
        return self.read_bytes(run_id, name).decode("utf-8")

    def read_parquet(self, run_id: str, name: str) -> pd.DataFrame:
        import pandas as pd

        return pd.read_parquet(io.BytesIO(self.read_bytes(run_id, name)), engine="pyarrow")

    def verify_artifact(self, run_id: str, name: str) -> ArtifactRecord:
        """校验磁盘内容与 index 记录的 sha256 一致；不一致即 :class:`ArtifactError`。"""
        record = self.get_artifact(run_id, name)
        self._read_verified(self._root / record.relative_path, record.sha256)
        return record

    # ---- snapshots：append-only（§30） ----------------------------------------

    def find_snapshot(self, dataset_version: str, name: str) -> SnapshotRecord | None:
        row = self._database.query_one(
            f"{_SNAPSHOT_SELECT_SQL} WHERE dataset_version = ? AND name = ?",
            (dataset_version, name),
        )
        return None if row is None else _row_to_snapshot(row)

    def list_snapshots(self, dataset_version: str) -> list[SnapshotRecord]:
        rows = self._database.query(
            f"{_SNAPSHOT_SELECT_SQL} WHERE dataset_version = ? ORDER BY name", (dataset_version,)
        )
        return [_row_to_snapshot(row) for row in rows]

    def snapshot_path(self, dataset_version: str, name: str) -> Path:
        return self.snapshot_dir(dataset_version) / _validate_token(name, "snapshot name")

    def read_snapshot(self, dataset_version: str, name: str) -> bytes:
        record = self.find_snapshot(dataset_version, name)
        if record is None:
            raise ArtifactError(f"dataset_version {dataset_version!r} has no snapshot {name!r}")
        return self._read_verified(self._root / record.relative_path, record.sha256)

    def write_snapshot(
        self,
        dataset_version: str,
        name: str,
        data: bytes,
        *,
        media_type: str = MEDIA_TYPE_OCTET_STREAM,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[SnapshotRecord, bool]:
        """写入 append-only 快照；返回 ``(record, created)``。

        - 已存在且内容相同 → 幂等返回既有记录（``created=False``）；
        - 已存在但内容不同 → :class:`ArtifactError`（§30：永不覆盖旧快照）；
        - 磁盘上存在未索引的同名文件（崩溃遗留）→ 内容相同则收养，否则拒绝覆盖。

        返回值里的 ``created`` 意为「本次完成了首次登记」（收养孤儿时为 ``True``），
        而非「本 writer 建立了文件」；并发时只有一个 writer 会建文件，其余转成幂等读。

        落盘走 :func:`atomic_create_bytes`（``os.link`` 独占创建），因此同进程 /
        跨进程并发写同一快照时至多一个 writer 能建立文件，败者转成幂等读，不会互相删除。
        """
        _validate_token(dataset_version, "dataset_version")
        _validate_token(name, "snapshot name")
        digest = sha256_bytes(data)
        existing = self.find_snapshot(dataset_version, name)
        if existing is not None:
            return self._verify_snapshot_idempotent(existing, digest)

        directory = self.snapshot_dir(dataset_version)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        relative_path = snapshot_relative_path(dataset_version, name)
        created = atomic_create_bytes(path, data)
        if not created:
            winner = self.find_snapshot(dataset_version, name)
            if winner is not None:
                return self._verify_snapshot_idempotent(winner, digest)
            # 未索引的孤儿文件：内容一致则收养（崩溃后重试），否则拒绝覆盖。
            if digest_file(path).sha256 != digest:
                raise ArtifactError(
                    f"snapshot {dataset_version}/{name} exists on disk with a different digest; "
                    "append-only snapshots are never overwritten (§30)"
                )
        record = SnapshotRecord(
            dataset_version=dataset_version,
            name=name,
            relative_path=relative_path,
            media_type=_require_media_type(media_type),
            sha256=digest,
            size_bytes=len(data),
            created_at=self._clock(),
            metadata=_validated_metadata(metadata),
        )
        try:
            self._index_snapshot(record)
        except sqlite3.IntegrityError as exc:
            winner = self.find_snapshot(dataset_version, name)
            if winner is not None and winner.sha256 == digest:
                return winner, False
            raise ArtifactError(
                f"snapshot {dataset_version}/{name} conflicts with an existing snapshot: {exc}"
            ) from exc
        return record, True

    def _verify_snapshot_idempotent(
        self, existing: SnapshotRecord, digest: str
    ) -> tuple[SnapshotRecord, bool]:
        """幂等命中：同一 dataset_version/name 只允许同一内容（§30 append-only）。

        这里只确认文件仍在（缺失即 store 损坏）；字节级完整性由 :meth:`read_snapshot`
        在读取时校验，避免 append-only 热路径把大快照整个读回内存。
        """
        if existing.sha256 != digest:
            raise ArtifactError(
                f"snapshot {existing.dataset_version}/{existing.name} already exists with a "
                f"different digest ({existing.sha256}); append-only snapshots are never "
                "overwritten (§30)"
            )
        if not (self._root / existing.relative_path).is_file():
            raise ArtifactError(
                f"snapshot {existing.dataset_version}/{existing.name} is indexed but its file "
                "is missing; store is corrupt"
            )
        return existing, False

    # ---- 内部写入/索引 --------------------------------------------------------

    def _persist_bytes(
        self,
        run_id: str,
        name: str,
        filename: str,
        data: bytes,
        *,
        media_type: str,
        row_count: int | None = None,
        schema_hash: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRecord:
        path = self._run_artifact_path(run_id, filename)
        # 先校验再落盘：metadata / media_type 不合法时不留孤儿文件。
        validated_metadata = _validated_metadata(metadata)
        _require_media_type(media_type)
        digest = sha256_bytes(data)
        created = atomic_create_bytes(path, data)
        if not created:
            raise self._conflict_error(run_id, name, path)
        record = self._make_record(
            run_id, name, filename, media_type, digest, len(data),
            row_count, schema_hash, validated_metadata,
        )
        self._index_artifact(record, path)
        return record

    def _persist_file(
        self,
        run_id: str,
        name: str,
        filename: str,
        writer: Callable[[Path], None],
        *,
        media_type: str,
        row_count: int | None = None,
        schema_hash: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRecord:
        path = self._run_artifact_path(run_id, filename)
        validated_metadata = _validated_metadata(metadata)
        _require_media_type(media_type)
        stamp = atomic_create_file(path, writer)
        if stamp is None:
            raise self._conflict_error(run_id, name, path)
        record = self._make_record(
            run_id, name, filename, media_type, stamp.sha256, stamp.size_bytes,
            row_count, schema_hash, validated_metadata,
        )
        self._index_artifact(record, path)
        return record

    def _make_record(
        self,
        run_id: str,
        name: str,
        filename: str,
        media_type: str,
        digest: str,
        size_bytes: int,
        row_count: int | None,
        schema_hash: str | None,
        metadata: Mapping[str, Any],
    ) -> ArtifactRecord:
        return ArtifactRecord(
            run_id=run_id,
            name=name,
            relative_path=self.artifact_relative_path(run_id, filename),
            media_type=_require_media_type(media_type),
            sha256=digest,
            size_bytes=size_bytes,
            row_count=row_count,
            schema_hash=schema_hash,
            created_at=self._clock(),
            metadata=dict(metadata),
        )

    def _conflict_error(self, run_id: str, name: str, path: Path) -> ArtifactError:
        """独占创建失败时区分「已登记（write-once）」与「未登记孤儿文件」。"""
        if self.find_artifact(run_id, name) is not None:
            return ArtifactError(
                f"run {run_id!r} already has artifact {name!r}; run artifacts are write-once"
            )
        return ArtifactError(
            f"run artifact {path} exists on disk but is not indexed; refusing to overwrite"
        )

    def _index_artifact(self, record: ArtifactRecord, path: Path) -> None:
        # 调用前提：文件由本次 atomic_create_* 独占创建（writer 即本 writer 自己），
        # 因此任何失败都可安全删除自己的文件，不会误删并发胜者的内容。
        try:
            with self._database.transaction() as connection:
                connection.execute(
                    _ARTIFACT_INSERT_SQL,
                    (
                        record.run_id,
                        record.name,
                        record.relative_path,
                        record.media_type,
                        record.sha256,
                        record.size_bytes,
                        record.row_count,
                        record.schema_hash,
                        record.created_at.isoformat(),
                        canonical_json(record.metadata),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            path.unlink(missing_ok=True)
            raise self._conflict_error(record.run_id, record.name, path) from exc
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    def _index_snapshot(self, record: SnapshotRecord) -> None:
        with self._database.transaction() as connection:
            connection.execute(
                _SNAPSHOT_INSERT_SQL,
                (
                    record.dataset_version,
                    record.name,
                    record.relative_path,
                    record.media_type,
                    record.sha256,
                    record.size_bytes,
                    record.created_at.isoformat(),
                    canonical_json(record.metadata),
                ),
            )

    def _read_verified(self, path: Path, expected_sha256: str) -> bytes:
        # 纵深防御：relative_path 来自 index，写侧已做 token 校验，读侧再确认不逃出 root。
        resolved = path.resolve()
        if not resolved.is_relative_to(self._root.resolve()):
            raise ArtifactError(f"artifact path {path} escapes artifact root {self._root}")
        try:
            data = resolved.read_bytes()
        except FileNotFoundError as exc:
            raise ArtifactError(f"artifact file {path} is missing") from exc
        actual = sha256_bytes(data)
        if actual != expected_sha256:
            raise ArtifactError(
                f"artifact file {path} sha256 {actual} != index {expected_sha256}; store is corrupt"
            )
        return data


def _require_media_type(media_type: str) -> str:
    if not isinstance(media_type, str) or not media_type.strip():
        raise ConfigurationError("media_type must be a non-empty string")
    return media_type


def _row_to_artifact(row: sqlite3.Row) -> ArtifactRecord:
    return ArtifactRecord(
        run_id=row["run_id"],
        name=row["name"],
        relative_path=row["relative_path"],
        media_type=row["media_type"],
        sha256=row["sha256"],
        size_bytes=row["size_bytes"],
        row_count=row["row_count"],
        schema_hash=row["schema_hash"],
        created_at=datetime.fromisoformat(row["created_at"]),
        metadata=json.loads(row["metadata_json"]),
    )


def _row_to_snapshot(row: sqlite3.Row) -> SnapshotRecord:
    return SnapshotRecord(
        dataset_version=row["dataset_version"],
        name=row["name"],
        relative_path=row["relative_path"],
        media_type=row["media_type"],
        sha256=row["sha256"],
        size_bytes=row["size_bytes"],
        created_at=datetime.fromisoformat(row["created_at"]),
        metadata=json.loads(row["metadata_json"]),
    )


__all__ = [
    "ARTIFACT_STORE_CONTRACT_VERSION",
    "MEDIA_TYPE_JSON",
    "MEDIA_TYPE_MARKDOWN",
    "MEDIA_TYPE_OCTET_STREAM",
    "MEDIA_TYPE_PARQUET",
    "MEDIA_TYPE_YAML",
    "RUNS_SUBDIR",
    "SNAPSHOTS_SUBDIR",
    "ArtifactRecord",
    "ArtifactStore",
    "SnapshotRecord",
    "frame_schema_hash",
    "run_dir_relative",
    "snapshot_relative_path",
]
