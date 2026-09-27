"""Walk-forward 分段与 embargo 空置（基线文档 §27 / §29，ADR-022）。

本模块只做 **session 索引空间** 的算术：给定一条升序 market session 时间轴与分段
计划，算出每段占用的 session，并在相邻段之间空置 ``embargo_sessions`` 个 session。
它不加载行情、不认识 symbol、不构造 label——那些属于
:mod:`kronos_ai.evaluation.dataset`。这样 embargo 规则只有一份实现，可以被
``tests/leakage/`` 直接断言，而不必先把整个 dataset 搭起来。

§27 规则（本模块即该规则的唯一定义）：

```text
前一段最后一个 origin 之后
空置 embargo_sessions 个 session
后一段第一个 origin 才能出现
```

``embargo_sessions`` 的单一真源是 :class:`kronos_ai.evaluation.dataset.LabelPolicy`：
本模块只接收一个整数，不硬编码任何默认值，也不知道 label 的定义。``embargo_sessions
>= horizon_sessions`` 由 LabelPolicy 负责；这里只要求 ``embargo_sessions >= 0``。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from itertools import pairwise
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from kronos_ai.domain.time import SessionDate
from kronos_ai.errors import ConfigurationError, LeakageError

WALK_FORWARD_VERSION = "walk-forward-v1"

SegmentName = Literal["train", "validation", "calibration", "test"]
SEGMENT_NAMES: tuple[SegmentName, ...] = ("train", "validation", "calibration", "test")


class SegmentSpec(BaseModel):
    """一段的声明：名字 + 以 session 计的长度（§27 的 Train/Validation/Calibration/Test）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: SegmentName
    length_sessions: int = Field(ge=1)


class WalkForwardPlan(BaseModel):
    """分段计划（session 单位）；不含 embargo，也不含日期。

    embargo 由 LabelPolicy 提供，日期由调用方给出的时间轴决定：计划只是「每段多少
    session」，因此同一个 plan 可以复用到任何时间窗口上，而 §32.1 的
    ``evaluation.embargo_sessions`` 只会有一条来源。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    segments: tuple[SegmentSpec, ...]

    @field_validator("segments")
    @classmethod
    def _nonempty_unique_in_canonical_order(
        cls, value: tuple[SegmentSpec, ...]
    ) -> tuple[SegmentSpec, ...]:
        if not value:
            raise ValueError("segments must not be empty")
        names = [segment.name for segment in value]
        if len(set(names)) != len(names):
            raise ValueError(f"segment names must be unique, got {names}")
        order = [SEGMENT_NAMES.index(name) for name in names]
        if order != sorted(order):
            raise ValueError(
                f"segments must follow the canonical order {SEGMENT_NAMES}, got {names}"
            )
        return value

    @property
    def origin_sessions(self) -> int:
        """各段长度之和（= 会产出 origin 的 session 数）。"""
        return sum(segment.length_sessions for segment in self.segments)

    @property
    def embargo_gaps(self) -> int:
        """相邻段之间的空置区间数（段数 - 1）。"""
        return len(self.segments) - 1

    def required_sessions(self, *, embargo_sessions: int, history_prefix: int) -> int:
        """在给定 embargo 与 lookback 前置窗口下，时间轴必须拥有的 session 总数。"""
        if embargo_sessions < 0:
            raise ConfigurationError(f"embargo_sessions must be >= 0, got {embargo_sessions}")
        if history_prefix < 0:
            raise ConfigurationError(f"history_prefix must be >= 0, got {history_prefix}")
        return history_prefix + self.origin_sessions + embargo_sessions * self.embargo_gaps


class SegmentSessions(BaseModel):
    """一段落在时间轴上的实际位置：起始下标 + 该段的 session 序列。

    ``start_index`` 是段内第一个 session 在**整条时间轴**（含 lookback 前置窗口）中的
    下标。保留全局下标而不是段内相对下标，是为了让相邻段的间距可以直接相减——
    §29 的 leakage 断言就是一次下标减法。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: SegmentName
    start_index: int = Field(ge=0)
    sessions: tuple[SessionDate, ...]

    @field_validator("sessions")
    @classmethod
    def _nonempty_ascending_unique_dates(cls, value: tuple[date, ...]) -> tuple[date, ...]:
        if not value:
            raise ValueError("sessions must not be empty")
        if list(value) != sorted(value):
            raise ValueError("sessions must be sorted ascending")
        if len(set(value)) != len(value):
            raise ValueError("sessions must be unique")
        return value

    @property
    def end_index(self) -> int:
        return self.start_index + len(self.sessions) - 1

    @property
    def first_session(self) -> date:
        return self.sessions[0]

    @property
    def last_session(self) -> date:
        return self.sessions[-1]

    @property
    def num_sessions(self) -> int:
        return len(self.sessions)

    def gap_before(self, later: SegmentSessions) -> int:
        """``later`` 与 ``self`` 之间**空置**的 session 数（不含段的端点）。

        ``later.start_index - self.end_index - 1``：相邻段（``later`` 紧接 ``self``）为 0。
        """
        return later.start_index - self.end_index - 1


def split_sessions(
    *,
    sessions: Sequence[date],
    plan: WalkForwardPlan,
    embargo_sessions: int,
    history_prefix: int = 0,
) -> tuple[SegmentSessions, ...]:
    """把升序 session 时间轴切成分段，相邻段之间空置 ``embargo_sessions`` 个 session。

    ``sessions`` 的构成被**完整确定**，不做任何隐式截断或补全：

    ```text
    [history_prefix 个 session（首 origin 的 lookback 前置窗口）]
    [plan.segments[0] 的 session]
    [embargo_sessions 个空置 session]
    [plan.segments[1] 的 session]
    ...
    ```

    长度不匹配、时间轴非升序/重复/非 session 语义（``datetime``）都抛
    ``ConfigurationError``：这是调用方的输入错误，宁可拒绝也不要悄悄少切一段。
    """
    if embargo_sessions < 0:
        raise ConfigurationError(f"embargo_sessions must be >= 0, got {embargo_sessions}")
    if history_prefix < 0:
        raise ConfigurationError(f"history_prefix must be >= 0, got {history_prefix}")
    _validate_timeline(sessions)

    required = plan.required_sessions(
        embargo_sessions=embargo_sessions, history_prefix=history_prefix
    )
    if len(sessions) != required:
        raise ConfigurationError(
            f"timeline has {len(sessions)} sessions but the plan requires exactly {required} "
            f"(history_prefix={history_prefix} + origin_sessions={plan.origin_sessions} + "
            f"embargo_sessions={embargo_sessions} * gaps={plan.embargo_gaps})"
        )

    timeline = tuple(sessions)
    segments: list[SegmentSessions] = []
    cursor = history_prefix
    for spec in plan.segments:
        end = cursor + spec.length_sessions
        segments.append(
            SegmentSessions(name=spec.name, start_index=cursor, sessions=timeline[cursor:end])
        )
        cursor = end + embargo_sessions
    return tuple(segments)


def require_no_leakage(
    segments: Sequence[SegmentSessions],
    *,
    embargo_sessions: int,
    horizon_sessions: int,
) -> None:
    """§29 leakage 断言：相邻段之间必须满足 ``前段最后 origin + embargo < 后段第一 origin``。

    断言用的是 ``embargo_sessions`` 而**不是** ``horizon_sessions``：当
    ``embargo_sessions > horizon_sessions`` 时，horizon 断言「必要但不充分」——
    label 窗口不重叠并不代表训练段最后的 label 没有被下一段的输入窗口看到。
    这里额外要求 ``embargo_sessions >= horizon_sessions``（LabelPolicy 已保证，
    本函数再验一次），于是 horizon 断言成为该不变量的推论，不需要单独写一遍。
    """
    if horizon_sessions < 1:
        raise ConfigurationError(f"horizon_sessions must be >= 1, got {horizon_sessions}")
    if embargo_sessions < 0:
        raise ConfigurationError(f"embargo_sessions must be >= 0, got {embargo_sessions}")
    if embargo_sessions < horizon_sessions:
        raise ConfigurationError(
            f"embargo_sessions {embargo_sessions} < horizon_sessions {horizon_sessions}: "
            "label 定义要求 embargo 至少与 horizon 一样长（§27）"
        )

    for earlier, later in pairwise(segments):
        gap = earlier.gap_before(later)
        if gap < embargo_sessions:
            raise LeakageError(
                f"segments {earlier.name!r} -> {later.name!r} leave only {gap} sessions "
                f"between {earlier.last_session} and {later.first_session}, "
                f"but embargo_sessions={embargo_sessions} is required "
                "(前段最后 origin + embargo_sessions < 后段第一 origin)"
            )


def _validate_timeline(sessions: Sequence[date]) -> None:
    if not sessions:
        raise ConfigurationError("sessions must not be empty")
    for day in sessions:
        if isinstance(day, datetime):
            raise ConfigurationError(f"sessions must be dates, not datetimes: {day.isoformat()}")
    if list(sessions) != sorted(sessions):
        raise ConfigurationError("sessions must be sorted ascending")
    if len(set(sessions)) != len(sessions):
        raise ConfigurationError("sessions must be unique")


if TYPE_CHECKING:
    # mypy 结构化校验：签名变更属于契约变更（本模块只依赖 §27 的索引算术）
    _CONTRACT_ANCHOR: tuple[SegmentSessions, ...] = split_sessions(
        sessions=(date(2026, 1, 5), date(2026, 1, 6)),
        plan=WalkForwardPlan(segments=(SegmentSpec(name="train", length_sessions=2),)),
        embargo_sessions=0,
    )
