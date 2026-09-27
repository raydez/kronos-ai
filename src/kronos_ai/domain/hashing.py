"""Deterministic canonical hashing for contracts and cache keys."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> str:
    """Stable JSON encoding: sorted keys, no whitespace, UTF-8 content."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_hex(obj: Any) -> str:
    """SHA-256 hex digest over the canonical JSON encoding of *obj*."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    """SHA-256 hex digest over raw bytes (artifact integrity, §30)."""
    return hashlib.sha256(data).hexdigest()


_HEX_DIGITS = frozenset("0123456789abcdef")


def is_sha256_hex(value: object) -> bool:
    """True iff *value* is a 64-char lowercase sha256 hex digest (no whitespace)."""
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX_DIGITS
