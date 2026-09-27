"""崩溃安全的原子文件写入原语（§15 forecast cache / §30 Artifact Store）。

同一套「临时文件 + fsync + 原子 rename + 父目录 fsync」被两处共用：forecast cache
（§15）与 Artifact Store（§30–§33）。集中在此，避免两份实现各自演化出不同的持久化
保证——§15 与 §30 都要求读者永远看不到半成品文件。

语义：
- 临时文件用 ``tempfile.mkstemp``（``O_EXCL`` + 内核保证的随机后缀）创建，同进程 /
  跨进程并发写同一路径也不会互相踩到名字；代价是权限为 0600（本地研究 artifact，
  比默认 0644 更严格）。
- rename 之后 fsync 父目录：只 fsync 文件不保证崩溃后 rename 仍在。

写模式（与 §30 的 write-once 对应）：
- :func:`atomic_write_bytes` / :func:`atomic_write_text`：「覆盖」语义，走 rename；
- :func:`atomic_create_bytes` / :func:`atomic_create_file`：「只创建」语义，走 ``os.link``
  独占创建，已存在则返回 ``False`` / ``None``，绝不覆盖。
"""

from __future__ import annotations

import errno
import hashlib
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

from kronos_ai.errors import ArtifactError

_HASH_CHUNK_BYTES = 1024 * 1024

# 目录 fsync 在部分平台 / 文件系统上不被支持：这些 errno 视为可跳过，其余（如 EIO）
# 必须显式上抛，避免把「rename 未持久化」静默当成成功（§3.2）。
DIR_FSYNC_UNSUPPORTED_ERRNOS: frozenset[int] = frozenset(
    {
        errno.EINVAL,
        getattr(errno, "ENOTSUP", errno.EINVAL),
        errno.EBADF,
        errno.EPERM,
        errno.EACCES,
        errno.EROFS,
    }
)


def fsync_dir(directory: Path) -> None:
    """把 rename 持久化到父目录：只 fsync 文件不保证崩溃后 rename 仍在（§15）。

    平台差异：Windows 无法 open 目录，``O_DIRECTORY`` 仅 POSIX 可用；这类「不支持」
    情形跳过即可（产物本身可重算），但真实的 I/O 错误必须上抛。
    """
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        dir_fd = os.open(directory, flags)
    except OSError as exc:
        if exc.errno in DIR_FSYNC_UNSUPPORTED_ERRNOS:
            return
        raise
    try:
        os.fsync(dir_fd)
    except OSError as exc:
        if exc.errno not in DIR_FSYNC_UNSUPPORTED_ERRNOS:
            raise
    finally:
        os.close(dir_fd)


_FSYNC_UNSUPPORTED_FILE_ERRNOS: frozenset[int] = frozenset(
    {
        errno.EINVAL,
        getattr(errno, "ENOTSUP", errno.EINVAL),
        errno.EBADF,
    }
)


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError as exc:
        if exc.errno not in _FSYNC_UNSUPPORTED_FILE_ERRNOS:
            raise
    finally:
        os.close(fd)


def _new_temp(path: Path) -> tuple[int, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    return fd, Path(tmp_name)


class FileDigest(NamedTuple):
    """落盘文件的存储完整性摘要（sha256 + 字节数）。"""

    sha256: str
    size_bytes: int


def _digest_file(path: Path) -> FileDigest:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
            size += len(chunk)
    return FileDigest(digest.hexdigest(), size)


def digest_file(path: Path) -> FileDigest:
    """流式计算文件摘要；避免把大 Parquet 整个读回内存。"""
    return _digest_file(path)


def _link_exclusive(tmp: Path, path: Path) -> bool:
    """把 *tmp* 硬链接为 *path*；*path* 已存在时返回 ``False``（绝不覆盖）。"""
    try:
        os.link(tmp, path)
    except FileExistsError:
        return False
    except OSError as exc:
        # 不支持硬链接的文件系统上不能静默退化成 rename（会破坏独占语义）。
        raise ArtifactError(f"cannot create {path} exclusively via hard link: {exc}") from exc
    return True


def atomic_create_bytes(path: Path, data: bytes) -> bool:
    """**独占**创建 *path*；已存在时返回 ``False`` 且不改动既有文件（§30 write-once）。

    与 :func:`atomic_write_bytes` 的区别在于语义：后者「覆盖」，本函数「只创建」。
    通过 ``os.link`` 在最终路径上做原子创建，天然抵抗同进程 / 跨进程并发重复写入——
    败者拿不到文件，不会破坏胜者已登记的内容（§30 写入串行化）。
    """
    fd, tmp = _new_temp(path)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if not _link_exclusive(tmp, path):
            return False
        fsync_dir(path.parent)
        return True
    finally:
        tmp.unlink(missing_ok=True)


def atomic_create_file(path: Path, writer: Callable[[Path], None]) -> FileDigest | None:
    """独占地把 ``writer(tmp_path)`` 的产物落到 *path*；已存在时返回 ``None``。

    用于无法先得到 bytes 的写入器（Parquet）。成功时返回落盘文件的
    :class:`FileDigest`，调用方无需再把文件读回内存计算 sha256。
    """
    fd, tmp = _new_temp(path)
    os.close(fd)
    try:
        writer(tmp)
        _fsync_file(tmp)
        if not _link_exclusive(tmp, path):
            return None
        fsync_dir(path.parent)
        # 链接成功后才哈希：败者不做无用功（*tmp* 与 *path* 是同一 inode）。
        return _digest_file(tmp)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """字节写入走「临时文件 + fsync + 原子 rename」；读者只会看到完整文件。"""
    fd, tmp = _new_temp(path)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        fsync_dir(path.parent)
    finally:
        # os.replace 成功后 tmp 已不存在；异常路径下清理半成品。
        tmp.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))
