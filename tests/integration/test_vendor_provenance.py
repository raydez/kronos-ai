"""vendor 目录 provenance 核验（ADR-020 / §37）。

把 UPSTREAM.md 的「逐字节副本（仅 import 头部 + 行尾归一化）」从文档声明变成可执行
检查。文件内分两层：

- **离线（默认门禁）**：重新计算本目录 sha256 并与 UPSTREAM.md 表格比对，
  以及核对表格中声明的上游 sha 与在线核验常量一致——文档与文件任何一侧漂移都失败。
- **在线（`integration` 标记，需网络）**：按 pinned commit 从 GitHub 拉取上游文件，
  逐字节比对（`module.py` 允许 CRLF→LF，`kronos.py` 允许表格中写明的 4 行 import 替换）。

在线部分默认不跑：
    pytest -m integration tests/integration/test_vendor_provenance.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from kronos_ai.forecast.backends.kronos.runtime import VENDOR_UPSTREAM_COMMIT

VENDOR_DIR = Path(__file__).resolve().parents[2] / "src/kronos_ai/forecast/backends/kronos/vendor"
UPSTREAM_MD = VENDOR_DIR / "UPSTREAM.md"
RAW_URL = "https://raw.githubusercontent.com/shiyu-coder/Kronos/{commit}/{path}"

UPSTREAM_SHA256 = {
    "kronos.py": "0a5f90282e2039c2de0771473419715c845def154896dbd0f5747837e6241032",
    "module.py": "a07edbadc0e96804c8158c021bbc6063bb7cc43b34d7fc470d5c8ff2005a409f",
    "LICENSE": "acb2d194d378204e5f2be4dcd24d39ecac437903620c790c3315a96dab388fdc",
}

# UPSTREAM.md 表格「上游文件」列 → 本目录文件名
LOCAL_BY_UPSTREAM_NAME = {
    "model/kronos.py": "kronos.py",
    "model/module.py": "module.py",
    "model/__init__.py": "__init__.py",
    "LICENSE": "LICENSE",
}

KRONOS_IMPORT_PATCH = (
    'import sys\n\nfrom tqdm import trange\n\nsys.path.append("../")\nfrom model.module import *\n',
    "from tqdm import trange\n\nfrom kronos_ai.forecast.backends.kronos.vendor.module import *\n",
)


def fetch(path: str) -> bytes:
    import urllib.request

    url = RAW_URL.format(commit=VENDOR_UPSTREAM_COMMIT, path=path)
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()  # type: ignore[no-any-return]


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def documented_rows() -> dict[str, tuple[str, str]]:
    """解析 UPSTREAM.md 的 sha256 表：{本目录文件名: (上游 sha, 本目录 sha)}。"""
    rows: dict[str, tuple[str, str]] = {}
    for line in UPSTREAM_MD.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 5:
            continue
        upstream_name = cells[0].strip("`")
        if upstream_name in LOCAL_BY_UPSTREAM_NAME:
            rows[LOCAL_BY_UPSTREAM_NAME[upstream_name]] = (cells[1].strip("`"), cells[2].strip("`"))
    return rows


def test_documented_local_hashes_match_files() -> None:
    """本目录文件 sha256 必须与 UPSTREAM.md 表格一致：vendor 文件或表格漂移都算失败。"""
    rows = documented_rows()
    assert set(rows) == set(LOCAL_BY_UPSTREAM_NAME.values()), (
        f"UPSTREAM.md sha256 表不完整：{sorted(rows)}"
    )
    for filename, (_, documented_local) in rows.items():
        actual = sha256((VENDOR_DIR / filename).read_bytes())
        assert actual == documented_local, (
            f"{filename} sha256 {actual} != UPSTREAM.md 记录的 {documented_local}；"
            "改 vendor 文件必须同步更新表格"
        )


def test_documented_upstream_hashes_match_constants() -> None:
    """文档声明的上游 sha 必须与本文件在线核验用的常量一致，否则文档与检查各说各话。"""
    rows = documented_rows()
    for filename, expected in UPSTREAM_SHA256.items():
        assert rows[filename][0] == expected, (
            f"UPSTREAM.md 中 {filename} 的上游 sha 与在线核验常量不一致"
        )


@pytest.mark.integration
def test_module_py_matches_upstream_modulo_line_endings() -> None:
    upstream = fetch("model/module.py")
    assert sha256(upstream) == UPSTREAM_SHA256["module.py"], (
        "upstream module.py changed: the pinned commit must be re-verified, not silently accepted"
    )
    local = (VENDOR_DIR / "module.py").read_bytes()
    assert local == upstream.replace(b"\r\n", b"\n")


@pytest.mark.integration
def test_kronos_py_differs_only_in_import_head() -> None:
    upstream = fetch("model/kronos.py").decode("utf-8")
    assert sha256(upstream.encode("utf-8")) == UPSTREAM_SHA256["kronos.py"], (
        "upstream kronos.py changed: the pinned commit must be re-verified, not silently accepted"
    )
    local = (VENDOR_DIR / "kronos.py").read_text()
    old, new = KRONOS_IMPORT_PATCH
    assert old in upstream, "upstream import head changed; update the documented patch"
    assert local == upstream.replace(old, new, 1)


@pytest.mark.integration
def test_license_is_byte_identical() -> None:
    upstream = fetch("LICENSE")
    assert sha256(upstream) == UPSTREAM_SHA256["LICENSE"]
    assert (VENDOR_DIR / "LICENSE").read_bytes() == upstream
