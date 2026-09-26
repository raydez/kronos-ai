"""ResearchTime 时间语义（基线文档 §5）。

market_date = 研究对象对应的交易日
knowledge_cutoff = 系统被允许使用信息的最晚时间（timezone-aware, Asia/Shanghai）

时区约定：统一使用固定 +08:00 偏移（CN_TZ）。A 股数据源一律以北京时间表示
交易时刻，不存在需要按地理位置换算的场景；ZoneInfo("Asia/Shanghai") 在
1986-1991 历史夏令时期间会给出 +09:00，与域校验（要求 +08:00）不对称，
故本模块不采用。
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Literal

from pydantic import BaseModel, field_validator, model_validator

CN_UTC_OFFSET = timedelta(hours=8)
CN_TZ = timezone(CN_UTC_OFFSET, "Asia/Shanghai")
SHANGHAI = CN_TZ

MARKET_SESSION_CLOSE = time(15, 0)
SAME_DAY_EVENING = time(18, 0)

KnowledgeCutoffPolicy = Literal["market_close", "same_day_evening", "explicit"]

CUTOFF_POLICY_VERSION = "cutoff-policy-v1"


def ensure_shanghai_aware(value: datetime, field: str) -> datetime:
    """要求 timezone-aware 且 UTC 偏移为 +08:00（Asia/Shanghai 北京时间）。"""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    offset = value.utcoffset()
    if offset != CN_UTC_OFFSET:
        raise ValueError(f"{field} must be Asia/Shanghai (+08:00), got offset {offset}")
    return value


def _shanghai_combine(day: date, moment: time) -> datetime:
    return datetime.combine(day, moment, tzinfo=CN_TZ)


class CutoffPolicyRecord(BaseModel):
    """Run Metadata 记录项：policy 名 + 参数 + 版本（§5）。"""

    policy: KnowledgeCutoffPolicy
    policy_version: str
    parameters: dict[str, Any]


def cutoff_policy_record(policy: KnowledgeCutoffPolicy) -> CutoffPolicyRecord:
    """policy 的版本化描述；默认值语义变更必须升 CUTOFF_POLICY_VERSION。"""
    parameters: dict[str, Any]
    if policy == "market_close":
        parameters = {"session_close": MARKET_SESSION_CLOSE.isoformat()}
    elif policy == "same_day_evening":
        parameters = {"evening": SAME_DAY_EVENING.isoformat()}
    elif policy == "explicit":
        parameters = {"source": "caller_supplied"}
    else:
        raise ValueError(f"unknown cutoff policy: {policy!r}")
    return CutoffPolicyRecord(
        policy=policy, policy_version=CUTOFF_POLICY_VERSION, parameters=parameters
    )


def resolve_knowledge_cutoff(
    market_date: date,
    policy: KnowledgeCutoffPolicy,
    explicit_cutoff: datetime | None = None,
) -> datetime:
    """按版本化 policy 推导 knowledge_cutoff。

    - market_close: market_date 当日收盘 15:00+08:00（盘后研究）
    - same_day_evening: market_date 当日 18:00+08:00（CLI 默认，等待盘后数据更新）
    - explicit: 调用方显式指定（严格复现历史实验状态；须落在 market_date 当日，
      与 ResearchTime 不变式一致）

    返回值保证满足 ResearchTime 校验（+08:00 偏移、同日）。
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
