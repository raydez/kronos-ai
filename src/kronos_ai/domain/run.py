"""Run lifecycle contract（基线文档 §30–§33；ADR-013）。

:data:`RunStatus` 是 run registry（§30 SQLite）里 status 列与状态机迁移的唯一真源；
DDL 的 ``CHECK`` 约束、:meth:`RunRegistry.update_status` 的迁移校验都从它派生，
避免状态名在多处漂移。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, get_args

RunStatus = Literal["pending", "running", "succeeded", "failed", "cancelled"]

RUN_STATUSES: tuple[RunStatus, ...] = get_args(RunStatus)

TERMINAL_RUN_STATUSES: frozenset[RunStatus] = frozenset({"succeeded", "failed", "cancelled"})

# 允许的迁移；终态无出边。同状态重复写入由 update_status 单独放行（幂等）。
RUN_STATUS_TRANSITIONS: Mapping[RunStatus, frozenset[RunStatus]] = {
    "pending": frozenset({"running", "succeeded", "failed", "cancelled"}),
    "running": frozenset({"succeeded", "failed", "cancelled"}),
    "succeeded": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}
