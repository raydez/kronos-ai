"""Symbol 规范形式（基线文档 §7）。

内部规范 symbol = 6 位 ASCII 数字代码（如 600000）。
sh./sz./bj. 前缀形式仅允许在 Provider Adapter 边界存在，由本模块双向转换。
"""

from __future__ import annotations

import re

SYMBOL_PATTERN = re.compile(r"^[0-9]{6}$")

EXCHANGE_PREFIXES = ("sh", "sz", "bj")


def normalize_symbol(raw: str) -> str:
    """任意边界形式 → 6 位规范代码；不合规显式失败。"""
    text = raw.strip().lower()
    if "." in text:
        prefix, _, code = text.partition(".")
        if prefix not in EXCHANGE_PREFIXES or not SYMBOL_PATTERN.match(code):
            raise ValueError(f"unrecognized symbol: {raw!r}")
        return code
    if SYMBOL_PATTERN.match(text):
        return text
    raise ValueError(f"symbol must be a 6-digit code or sh./sz./bj. prefixed: {raw!r}")


def validate_normalized_symbol(raw: str) -> str:
    """要求输入已是规范形式（拒绝前缀），用于 domain 合同字段。"""
    text = raw.strip()
    if not SYMBOL_PATTERN.match(text):
        raise ValueError(f"symbol must be normalized 6-digit code, got {raw!r}")
    return text


def is_normalized_symbol(symbol: str) -> bool:
    return bool(SYMBOL_PATTERN.match(symbol))
