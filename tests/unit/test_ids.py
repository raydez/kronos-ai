"""ULID（§31；ADR-013）：golden vector、校验边界、单调生成器。"""

from __future__ import annotations

import threading

import pytest

from kronos_ai.domain.ids import (
    MAX_ULID_RANDOM,
    MAX_ULID_TIMESTAMP_MS,
    ULID_LENGTH,
    UlidGenerator,
    decode_ulid_timestamp_ms,
    encode_ulid,
    is_ulid,
    new_ulid,
    validate_ulid,
)
from kronos_ai.errors import ConfigurationError

# 规范 ULID spec 提供的 golden vector（https://github.com/ulid/spec）。
GOLDEN_ULID = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
GOLDEN_TIMESTAMP_MS = 1469922850259
GOLDEN_RANDOMNESS = bytes.fromhex("d6764c61efb99302bd5b")


def test_golden_vector_encodes_and_decodes() -> None:
    assert encode_ulid(GOLDEN_TIMESTAMP_MS, GOLDEN_RANDOMNESS) == GOLDEN_ULID
    assert decode_ulid_timestamp_ms(GOLDEN_ULID) == GOLDEN_TIMESTAMP_MS
    assert len(GOLDEN_ULID) == ULID_LENGTH


def test_validate_normalizes_lowercase_to_canonical_uppercase() -> None:
    lowered = GOLDEN_ULID.lower()
    assert validate_ulid(lowered) == GOLDEN_ULID
    assert is_ulid(lowered) is True


def test_validate_rejects_wrong_length() -> None:
    with pytest.raises(ConfigurationError, match="26 chars"):
        validate_ulid("01ARZ3NDEKTSV4RRFFQ69G5FA")


@pytest.mark.parametrize("char", ["I", "L", "O", "U"])
def test_validate_rejects_ambiguous_crockford_chars(char: str) -> None:
    candidate = "01ARZ3NDEKTSV4RRFFQ69G5FA" + char
    with pytest.raises(ConfigurationError, match="invalid Crockford base32"):
        validate_ulid(candidate)
    assert is_ulid(candidate) is False


def test_validate_rejects_overflow() -> None:
    # 26 个 'Z' 解出 130 bit，超出 128-bit ULID 值域。
    with pytest.raises(ConfigurationError, match="overflows"):
        validate_ulid("Z" * ULID_LENGTH)


def test_encode_rejects_out_of_range_timestamp() -> None:
    with pytest.raises(ConfigurationError, match="48-bit"):
        encode_ulid(MAX_ULID_TIMESTAMP_MS + 1, GOLDEN_RANDOMNESS)
    with pytest.raises(ConfigurationError, match="48-bit"):
        encode_ulid(-1, GOLDEN_RANDOMNESS)
    with pytest.raises(ConfigurationError, match="int"):
        encode_ulid(True, GOLDEN_RANDOMNESS)  # type: ignore[arg-type]


def test_encode_rejects_wrong_randomness_length() -> None:
    with pytest.raises(ConfigurationError, match="10 bytes"):
        encode_ulid(GOLDEN_TIMESTAMP_MS, b"\x00")


def test_new_ulid_is_reproducible_with_injected_inputs() -> None:
    assert new_ulid(timestamp_ms=GOLDEN_TIMESTAMP_MS, randomness=GOLDEN_RANDOMNESS) == GOLDEN_ULID


def test_is_ulid_rejects_non_string() -> None:
    assert is_ulid(None) is False
    assert is_ulid(12345) is False


def test_generator_is_strictly_increasing_within_same_millisecond() -> None:
    generator = UlidGenerator(
        time_source=lambda: 1_700_000_000_000, random_source=lambda _: b"\x00" * 10
    )
    ids = [generator.new() for _ in range(5)]
    assert ids == sorted(ids)
    assert len(set(ids)) == 5
    assert decode_ulid_timestamp_ms(ids[0]) == 1_700_000_000_000


def test_generator_tolerates_clock_going_backwards() -> None:
    times = iter([1_700_000_000_002, 1_700_000_000_001])
    generator = UlidGenerator(time_source=lambda: next(times), random_source=lambda _: b"\x00" * 10)
    first = generator.new()
    second = generator.new()
    assert decode_ulid_timestamp_ms(first) == decode_ulid_timestamp_ms(second) == 1_700_000_000_002
    assert first < second


def test_generator_raises_when_monotonic_counter_exhausted() -> None:
    generator = UlidGenerator(
        time_source=lambda: 1_700_000_000_000, random_source=lambda _: b"\xff" * 10
    )
    first = generator.new()
    assert decode_ulid_timestamp_ms(first) == 1_700_000_000_000
    with pytest.raises(ConfigurationError, match="exhausted"):
        generator.new()


def test_generator_uses_random_source_per_new_millisecond() -> None:
    counter = iter(range(1_000_000, 1_000_010))
    seeds = iter(range(10))

    def random_source(size: int) -> bytes:
        return bytes([next(seeds)]) * size

    generator = UlidGenerator(
        time_source=lambda: 1_700_000_000_000 + next(counter), random_source=random_source
    )
    ids = [generator.new() for _ in range(3)]
    assert ids == sorted(ids)
    assert len(set(ids)) == 3


def test_generator_is_unique_and_ordered_across_threads() -> None:
    generator = UlidGenerator()
    produced: list[str] = []
    lock = threading.Lock()
    barrier = threading.Barrier(8)

    def worker() -> None:
        barrier.wait()
        local = [generator.new() for _ in range(250)]
        with lock:
            produced.extend(local)
        assert local == sorted(local)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(produced) == 2000
    assert len(set(produced)) == 2000


def test_max_random_boundary_round_trips() -> None:
    encoded = encode_ulid(0, MAX_ULID_RANDOM.to_bytes(10, "big"))
    assert validate_ulid(encoded) == encoded
    assert decode_ulid_timestamp_ms(encoded) == 0
