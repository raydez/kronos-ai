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
