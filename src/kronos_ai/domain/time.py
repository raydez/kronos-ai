"""ResearchTime 时间语义（基线文档 §5）。

market_date = 研究对象对应的交易日
knowledge_cutoff = 系统被允许使用信息的最晚时间（timezone-aware, Asia/Shanghai）
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, field_validator, model_validator

SHANGHAI = ZoneInfo("Asia/Shanghai")
CN_UTC_OFFSET = timedelta(hours=8)

MARKET_SESSION_CLOSE = time(15, 0)
SAME_DAY_EVENING = time(18, 0)

KnowledgeCutoffPolicy = Literal["market_close", "same_day_evening", "explicit"]

CUTOFF_POLICY_VERSION = "cutoff-policy-v1"


def ensure_shanghai_aware(value: datetime, field: str) -> datetime:
    """要求 timezone-aware 且 UTC 偏移为 +08:00（Asia/Shanghai）。"""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    offset = value.utcoffset()
    if offset != CN_UTC_OFFSET:
        raise ValueError(f"{field} must be Asia/Shanghai (+08:00), got offset {offset}")
    return value


def _shanghai_combine(day: date, moment: time) -> datetime:
    return datetime.combine(day, moment, tzinfo=SHANGHAI)


def resolve_knowledge_cutoff(
    market_date: date,
    policy: KnowledgeCutoffPolicy,
    explicit_cutoff: datetime | None = None,
) -> datetime:
    """按版本化 policy 推导 knowledge_cutoff。

    - market_close: market_date 当日收盘 15:00+08:00（盘后研究）
    - same_day_evening: market_date 当日 18:00+08:00（CLI 默认，等待盘后数据更新）
    - explicit: 调用方显式指定（严格复现历史实验状态）

    默认值变更必须升 CUTOFF_POLICY_VERSION，不允许静默改变历史 run 复现状态。
    """
    if policy == "market_close":
        return _shanghai_combine(market_date, MARKET_SESSION_CLOSE)
    if policy == "same_day_evening":
        return _shanghai_combine(market_date, SAME_DAY_EVENING)
    if policy == "explicit":
        if explicit_cutoff is None:
            raise ValueError("explicit policy requires explicit_cutoff")
        ensure_shanghai_aware(explicit_cutoff, "explicit_cutoff")
        if explicit_cutoff.date() != market_date:
            raise ValueError("explicit_cutoff must fall on market_date")
        return explicit_cutoff
    raise ValueError(f"unknown cutoff policy: {policy!r}")


class ResearchTime(BaseModel):
    """研究时间状态：market_date + knowledge_cutoff 双时间轴。"""

    market_date: date
    knowledge_cutoff: datetime

    @field_validator("knowledge_cutoff")
    @classmethod
    def _cutoff_shanghai_aware(cls, value: datetime) -> datetime:
        return ensure_shanghai_aware(value, "knowledge_cutoff")

    @model_validator(mode="after")
    def _cutoff_on_market_date(self) -> ResearchTime:
        if self.knowledge_cutoff.date() != self.market_date:
            raise ValueError(
                "knowledge_cutoff must fall on market_date "
                f"(market_date={self.market_date}, cutoff={self.knowledge_cutoff})"
            )
        return self
