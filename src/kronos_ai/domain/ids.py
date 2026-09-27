"""ULID run identifiers（基线文档 §31；ADR-013）。

不再使用 ``20260926-001`` 这类「日期 + 进程内序号」ID：它不全局唯一、不可并行生成，
跨 run 排序也依赖本地时钟与目录扫描。ULID（Universally Unique Lexicographically
Sortable Identifier）= 48-bit 毫秒时间戳 + 80-bit 随机数，Crockford Base32 编码成
26 个字符，满足 §31 的 globally unique / sortable / parallel-safe 三条要求。

本实现是自包含的（不引入第三方依赖），并提供可注入时钟 / 随机源的
:class:`UlidGenerator`：它把「同进程同毫秒严格递增」变成契约，便于测试与跨进程
并行场景（跨进程唯一性由 80-bit 随机段保证）。
"""

from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Callable

from kronos_ai.errors import ConfigurationError

# Crockford Base32：去掉了易混淆的 I / L / O / U。
ULID_ALPHABET: str = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
ULID_LENGTH: int = 26
ULID_RANDOM_BYTES: int = 10
ULID_TIMESTAMP_MS_BITS: int = 48
ULID_RANDOM_BITS: int = 80
MAX_ULID_TIMESTAMP_MS: int = (1 << ULID_TIMESTAMP_MS_BITS) - 1
MAX_ULID_RANDOM: int = (1 << ULID_RANDOM_BITS) - 1

_ENCODE: dict[int, str] = dict(enumerate(ULID_ALPHABET))
_DECODE: dict[str, int] = {char: value for value, char in enumerate(ULID_ALPHABET)}
# Crockford 规范：大小写等价（``ulid`` 与 ``ULID`` 均可解析）。
_DECODE.update({char.lower(): value for value, char in enumerate(ULID_ALPHABET)})

_MILLISECONDS_PER_SECOND = 1000
_NANOSECONDS_PER_MILLISECOND = 1_000_000


def _validate_timestamp_ms(timestamp_ms: int) -> int:
    if isinstance(timestamp_ms, bool) or not isinstance(timestamp_ms, int):
        raise ConfigurationError(f"ULID timestamp must be an int, got {type(timestamp_ms).__name__}")
    if not 0 <= timestamp_ms <= MAX_ULID_TIMESTAMP_MS:
        raise ConfigurationError(
            f"ULID timestamp {timestamp_ms} outside [0, {MAX_ULID_TIMESTAMP_MS}] "
            "(48-bit millisecond range)"
        )
    return timestamp_ms


def _validate_randomness(randomness: bytes) -> bytes:
    if len(randomness) != ULID_RANDOM_BYTES:
        raise ConfigurationError(
            f"ULID randomness must be {ULID_RANDOM_BYTES} bytes, got {len(randomness)}"
        )
    return randomness


def encode_ulid(timestamp_ms: int, randomness: bytes) -> str:
    """``(48-bit 毫秒时间戳, 80-bit 随机数)`` → 26 字符 ULID（纯函数，便于测试）。"""
    timestamp_ms = _validate_timestamp_ms(timestamp_ms)
    randomness = _validate_randomness(randomness)
    value = (timestamp_ms << ULID_RANDOM_BITS) | int.from_bytes(randomness, "big")
    return "".join(
        _ENCODE[(value >> shift) & 0x1F] for shift in range(ULID_LENGTH * 5 - 5, -1, -5)
    )


def _decode_value(value: str) -> int:
    if not isinstance(value, str):
        raise ConfigurationError(f"ULID must be a string, got {type(value).__name__}")
    if len(value) != ULID_LENGTH:
        raise ConfigurationError(f"ULID must be {ULID_LENGTH} chars, got {len(value)}: {value!r}")
    decoded = 0
    for char in value:
        try:
            digit = _DECODE[char]
        except KeyError:
            raise ConfigurationError(f"ULID contains invalid Crockford base32 char {char!r}") from None
        decoded = (decoded << 5) | digit
    if decoded >> (ULID_TIMESTAMP_MS_BITS + ULID_RANDOM_BITS):
        # 26 个 5-bit 字符共 130 bit；首字符若 > '7' 会溢出 128-bit 值域。
        raise ConfigurationError(f"ULID {value!r} overflows the 128-bit ULID range")
    return decoded


def validate_ulid(value: str) -> str:
    """校验 ULID 并返回**规范大写形式**；不合法时显式 :class:`ConfigurationError`。

    Crockford Base32 大小写等价，但存储层（SQLite 主键 / 文件路径）大小写敏感，因此
    统一规范化成大写，避免 ``find("01hf7y...")`` 因大小写不匹配而误报「未注册」。
    """
    _decode_value(value)
    return value.upper()


def is_ulid(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        _decode_value(value)
    except ConfigurationError:
        return False
    return True


def decode_ulid_timestamp_ms(value: str) -> int:
    """取回 ULID 中的 48-bit 毫秒时间戳（用于 run 排序 / 时间回溯）。"""
    return _decode_value(value) >> ULID_RANDOM_BITS


def new_ulid(*, timestamp_ms: int | None = None, randomness: bytes | None = None) -> str:
    """生成一个 ULID；显式传入 timestamp / randomness 时完全可复现。"""
    ms = _now_timestamp_ms() if timestamp_ms is None else timestamp_ms
    rnd = secrets.token_bytes(ULID_RANDOM_BYTES) if randomness is None else randomness
    return encode_ulid(ms, rnd)


def _now_timestamp_ms() -> int:
    return time.time_ns() // _NANOSECONDS_PER_MILLISECOND


class UlidGenerator:
    """同进程内严格单调递增的 ULID 生成器（§31 sortable）。

    同一毫秒内递增 80-bit 随机段而不是重掷随机数，保证同进程生成的 ID 严格升序；
    跨进程并行时各自独立掷随机段，唯一性由 80-bit 空间保证（碰撞概率可忽略）。
    同一毫秒生成 ``2**80`` 个 ID 才会耗尽随机段，届时显式抛错而非静默回绕。
    """

    def __init__(
        self,
        *,
        time_source: Callable[[], int] | None = None,
        random_source: Callable[[int], bytes] | None = None,
    ) -> None:
        self._time_source = time_source if time_source is not None else _now_timestamp_ms
        self._random_source = (
            random_source if random_source is not None else secrets.token_bytes
        )
        self._lock = threading.Lock()
        self._last_timestamp_ms: int | None = None
        self._last_random: int = 0

    def new(self) -> str:
        with self._lock:
            timestamp_ms = _validate_timestamp_ms(self._time_source())
            if self._last_timestamp_ms is not None and timestamp_ms <= self._last_timestamp_ms:
                # 时钟回拨或同毫秒：沿用上一毫秒，随机段自增，保证严格递增。
                timestamp_ms = self._last_timestamp_ms
                candidate = self._last_random + 1
                if candidate > MAX_ULID_RANDOM:
                    raise ConfigurationError(
                        "ULID monotonic counter exhausted within a single millisecond"
                    )
            else:
                candidate = int.from_bytes(self._random_source(ULID_RANDOM_BYTES), "big")
            self._last_timestamp_ms = timestamp_ms
            self._last_random = candidate
            return encode_ulid(timestamp_ms, candidate.to_bytes(ULID_RANDOM_BYTES, "big"))
