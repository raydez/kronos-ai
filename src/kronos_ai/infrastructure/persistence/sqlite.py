"""SQLite 连接、WAL 与 schema 版本（§30：registry / artifact index 落 SQLite）。

§30 要求「WAL 模式，写入串行化」。WAL 允许一写多读，但 sqlite3 在锁竞争时默认抛
``database is locked``；这里用 ``busy_timeout`` + 进程内可重入锁把「串行访问」变成
契约，而不是让调用方到处重试。schema 版本与各组件契约版本写入 ``schema_meta`` 表：
打开一个版本不符的库时显式失败（§3.2），绝不迁移或猜测。

初始化时会并发打开同一个**全新**库（例如多个 benchmark worker 冷启动）：``PRAGMA
journal_mode = WAL`` 与 ``CREATE TABLE`` 都不受 ``busy_timeout`` 保护，且对 ``schema_meta``
的「先查后插」是 TOCTOU。因此初始化阶段对 ``database is locked`` 做有界重试，并把版本
行改为 ``INSERT ... ON CONFLICT DO NOTHING`` 后再回读，最终仍失败则显式抛
:class:`ArtifactError`（不泄漏原始 ``sqlite3`` 异常）。
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar, cast

from kronos_ai.errors import ArtifactError

SQLITE_BUSY_TIMEOUT_MS = 5000
SCHEMA_META_TABLE = "schema_meta"
_SCHEMA_VERSION_KEY = "schema_version"
_MEMORY_TARGET = ":memory:"

# 初始化阶段的锁竞争重试：覆盖「多个连接同时冷启动同一新库」。总等待约 5s，与
# busy_timeout 同量级；仍失败即视为真实故障。
_INIT_RETRY_ATTEMPTS = 100
_INIT_RETRY_INTERVAL_S = 0.05

_T = TypeVar("_T")


def _is_locked_error(exc: sqlite3.OperationalError) -> bool:
    message = str(exc).lower()
    return "locked" in message or "busy" in message


class SQLiteDatabase:
    """一个 SQLite 文件（或 ``:memory:``）上共享的连接 + 串行化访问。

    ``check_same_thread=False`` 让连接可被不同线程使用，但所有语句都在这把
    :class:`threading.RLock` 下执行，避免并发写互相踩锁；读也走同一把锁（研究型
    工作负载并发很低，正确性优先）。
    """

    def __init__(
        self,
        path: Path | str,
        *,
        schema_version: str,
        schema_statements: Sequence[str],
        busy_timeout_ms: int = SQLITE_BUSY_TIMEOUT_MS,
    ) -> None:
        self._target = str(path)
        if busy_timeout_ms < 0:
            raise ArtifactError(f"busy_timeout_ms must be >= 0, got {busy_timeout_ms}")
        self._path: Path | None = None if self._target == _MEMORY_TARGET else Path(path)
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            self._target, isolation_level=None, check_same_thread=False
        )
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        try:
            self._apply_pragmas(busy_timeout_ms)
            self._ensure_schema(schema_version, schema_statements)
        except BaseException:
            # 打开阶段失败（WAL 被拒 / schema 不符 / 锁竞争超时）不能泄漏连接。
            self._connection.close()
            raise

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    def _retry_locked(self, operation: Callable[[], _T]) -> _T:
        """对 ``database is locked`` 做有界重试，其余 OperationalError 立即包装上抛。"""
        for attempt in range(_INIT_RETRY_ATTEMPTS):
            try:
                return operation()
            except sqlite3.OperationalError as exc:
                if not _is_locked_error(exc) or attempt == _INIT_RETRY_ATTEMPTS - 1:
                    raise ArtifactError(
                        f"SQLite store {self._target} init failed: {exc}"
                    ) from exc
                time.sleep(_INIT_RETRY_INTERVAL_S)
        raise AssertionError("unreachable")  # pragma: no cover

    def _apply_pragmas(self, busy_timeout_ms: int) -> None:
        self._connection.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
        self._connection.execute("PRAGMA foreign_keys = ON")
        mode = self._retry_locked(
            lambda: self._connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
        )
        # §30 明确要求 WAL；文件库若未进入 WAL（某些网络文件系统会静默降级）必须显式失败。
        if self._path is not None and str(mode).lower() != "wal":
            raise ArtifactError(
                f"SQLite store {self._target} refused WAL journal mode (got {mode!r}); "
                "§30 requires WAL for run registry / artifact index"
            )

    def _ensure_schema(self, schema_version: str, statements: Sequence[str]) -> None:
        with self._lock:
            self._retry_locked(
                lambda: self._connection.execute(
                    f"CREATE TABLE IF NOT EXISTS {SCHEMA_META_TABLE} "
                    "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
            )
            if self._read_schema_meta(_SCHEMA_VERSION_KEY) is None:
                self._initialize_schema(schema_version, statements)
            stored = self._read_schema_meta(_SCHEMA_VERSION_KEY)
            if stored != schema_version:
                raise ArtifactError(
                    f"store {self._target} has schema_version {stored!r}, "
                    f"expected {schema_version!r}; refusing to use an incompatible store"
                )

    def _initialize_schema(self, schema_version: str, statements: Sequence[str]) -> None:
        """首建 schema。并发冷启动时用 ``INSERT ... ON CONFLICT DO NOTHING`` 消除 TOCTOU。"""
        def run_statement(statement: str) -> None:
            self._retry_locked(lambda: self._connection.execute(statement))

        self._retry_locked(lambda: self._connection.execute("BEGIN IMMEDIATE"))
        try:
            for statement in statements:
                run_statement(statement)
            self._retry_locked(
                lambda: self._connection.execute(
                    f"INSERT INTO {SCHEMA_META_TABLE} (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO NOTHING",
                    (_SCHEMA_VERSION_KEY, schema_version),
                )
            )
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")

    def _read_schema_meta(self, key: str) -> str | None:
        row = self._connection.execute(
            f"SELECT value FROM {SCHEMA_META_TABLE} WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else str(row["value"])

    def bind_contract(self, name: str, version: str) -> None:
        """把组件契约版本登记进 ``schema_meta``；已登记且不一致时显式失败。

        ADR-013 约定「改契约需升版本常量」；仅定义常量而不校验等于没有治理，因此
        :class:`RunRegistry` / :class:`ArtifactStore` 在构造时调用本方法把版本落库。
        """
        with self._lock, self.transaction() as connection:
            row = connection.execute(
                f"SELECT value FROM {SCHEMA_META_TABLE} WHERE key = ?", (name,)
            ).fetchone()
            if row is None:
                connection.execute(
                    f"INSERT INTO {SCHEMA_META_TABLE} (key, value) VALUES (?, ?)",
                    (name, version),
                )
                return
            if str(row["value"]) != version:
                raise ArtifactError(
                    f"store {self._target} binds {name}={row['value']!r}, "
                    f"code expects {version!r}; refusing to mix incompatible contracts"
                )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """串行化的写事务；不支持嵌套（显式失败而不是静默开启子事务）。"""
        with self._lock:
            if self._connection.in_transaction:
                raise ArtifactError("SQLite transaction cannot be nested")
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
            else:
                self._connection.execute("COMMIT")

    def query(self, sql: str, parameters: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._connection.execute(sql, tuple(parameters)).fetchall())

    def query_one(self, sql: str, parameters: Sequence[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            row = self._connection.execute(sql, tuple(parameters)).fetchone()
        return cast("sqlite3.Row | None", row)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> SQLiteDatabase:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
