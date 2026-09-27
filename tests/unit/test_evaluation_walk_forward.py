"""Walk-forward 分段与 embargo 空置（RX-KAI-017，基线文档 §27 / §29，ADR-022）。

覆盖范围（§54.2 要求 leakage guard 有 regression test）：

- 分段算术：段长、全局下标、相邻段之间恰好空置 ``embargo_sessions`` 个 session；
- 拒绝：时间轴长度不符 / 非升序 / 有重复 / 误传 datetime、计划为空 / 段名重复 /
  段名不按 canonical 顺序、负的 embargo 或 history_prefix；
- §29 断言：间距不足抛 ``LeakageError``，且判据是 ``embargo_sessions``
  （本模块不做 label 语义，horizon 断言在 evaluation.dataset 侧）。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest
from pydantic import ValidationError

from kronos_ai.errors import ConfigurationError, LeakageError
from kronos_ai.evaluation.walk_forward import (
    SEGMENT_NAMES,
    WALK_FORWARD_VERSION,
    SegmentSessions,
    SegmentSpec,
    WalkForwardPlan,
    require_no_leakage,
    split_sessions,
)


def weekdays(*, start: date = date(2026, 6, 1), count: int) -> tuple[date, ...]:
    """显式工作日序列（fixture 用，不做规则推导；日历语义见 ADR-009 §6）。"""
    days: list[date] = []
    current = start
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return tuple(days)


TRAIN = 8
VALIDATION = 4
TEST = 4
LOOKBACK = 6
EMBARGO = 5


def plan(*, train: int = TRAIN, validation: int = VALIDATION, test: int = TEST) -> WalkForwardPlan:
    return WalkForwardPlan(
        segments=(
            SegmentSpec(name="train", length_sessions=train),
            SegmentSpec(name="validation", length_sessions=validation),
            SegmentSpec(name="test", length_sessions=test),
        )
    )


def timeline() -> tuple[date, ...]:
    required = LOOKBACK + TRAIN + VALIDATION + TEST + EMBARGO * 2
    return weekdays(count=required)


def test_version_is_pinned() -> None:
    assert WALK_FORWARD_VERSION == "walk-forward-v1"
    assert SEGMENT_NAMES == ("train", "validation", "calibration", "test")


def test_plan_required_sessions_arithmetic() -> None:
    p = plan()
    assert p.origin_sessions == TRAIN + VALIDATION + TEST
    assert p.embargo_gaps == 2
    assert p.required_sessions(embargo_sessions=EMBARGO, history_prefix=LOOKBACK) == len(timeline())


def test_split_places_segments_with_embargo_gaps() -> None:
    sessions = timeline()
    segments = split_sessions(
        sessions=sessions, plan=plan(), embargo_sessions=EMBARGO, history_prefix=LOOKBACK
    )

    assert [segment.name for segment in segments] == ["train", "validation", "test"]
    assert [segment.num_sessions for segment in segments] == [TRAIN, VALIDATION, TEST]
    # 第一段从 lookback 前缀之后开始；下标是全局的
    assert segments[0].start_index == LOOKBACK
    assert segments[0].first_session == sessions[LOOKBACK]
    # 相邻段之间恰好空置 embargo 个 session
    assert segments[0].gap_before(segments[1]) == EMBARGO
    assert segments[1].gap_before(segments[2]) == EMBARGO
    assert segments[1].start_index == segments[0].end_index + EMBARGO + 1
    # 最后一段精确吃掉时间轴末尾（多一个少一个都被 split 拒绝）
    assert segments[-1].end_index == len(sessions) - 1
    # 段内 session 是时间轴的一段，且互不重叠
    for segment, spec in zip(segments, plan().segments, strict=True):
        assert (
            segment.sessions
            == sessions[segment.start_index : segment.start_index + spec.length_sessions]
        )
    assert not (set(segments[0].sessions) & set(segments[1].sessions) & set(segments[2].sessions))


def test_split_without_embargo_is_contiguous() -> None:
    sessions = weekdays(count=LOOKBACK + TEST)
    segments = split_sessions(
        sessions=sessions,
        plan=WalkForwardPlan(segments=(SegmentSpec(name="test", length_sessions=TEST),)),
        embargo_sessions=0,
        history_prefix=LOOKBACK,
    )
    assert len(segments) == 1
    assert segments[0].start_index == LOOKBACK
    assert segments[0].end_index == len(sessions) - 1


@pytest.mark.parametrize("delta", [-1, 1])
def test_split_rejects_wrong_timeline_length(delta: int) -> None:
    sessions = timeline()
    wrong = sessions[:-1] if delta < 0 else (*sessions, sessions[-1] + timedelta(days=1))
    with pytest.raises(ConfigurationError) as excinfo:
        split_sessions(
            sessions=wrong, plan=plan(), embargo_sessions=EMBARGO, history_prefix=LOOKBACK
        )
    message = str(excinfo.value)
    assert f"timeline has {len(wrong)} sessions" in message
    assert f"exactly {len(sessions)}" in message
    assert f"embargo_sessions={EMBARGO}" in message


def test_split_rejects_unsorted_or_duplicate_or_empty_timeline() -> None:
    sessions = timeline()
    with pytest.raises(ConfigurationError, match="sorted ascending"):
        split_sessions(
            sessions=(sessions[1], sessions[0], *sessions[2:]),
            plan=plan(),
            embargo_sessions=EMBARGO,
            history_prefix=LOOKBACK,
        )
    with pytest.raises(ConfigurationError, match="unique"):
        split_sessions(
            sessions=(*sessions[:-1], sessions[-2]),
            plan=plan(),
            embargo_sessions=EMBARGO,
            history_prefix=LOOKBACK,
        )
    with pytest.raises(ConfigurationError, match="must not be empty"):
        split_sessions(sessions=(), plan=plan(), embargo_sessions=EMBARGO, history_prefix=LOOKBACK)


def test_split_rejects_datetime_timeline() -> None:
    sessions = timeline()
    poisoned = (*sessions[:2], datetime(2026, 6, 3, 15, 0), *sessions[3:])
    with pytest.raises(ConfigurationError, match="not datetimes"):
        split_sessions(
            sessions=poisoned, plan=plan(), embargo_sessions=EMBARGO, history_prefix=LOOKBACK
        )


@pytest.mark.parametrize(
    ("embargo", "prefix"),
    [(-1, LOOKBACK), (EMBARGO, -1)],
)
def test_split_rejects_negative_embargo_or_prefix(embargo: int, prefix: int) -> None:
    with pytest.raises(ConfigurationError, match="must be >= 0"):
        split_sessions(
            sessions=timeline(), plan=plan(), embargo_sessions=embargo, history_prefix=prefix
        )


class TestPlanValidation:
    def test_empty_plan_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not be empty"):
            WalkForwardPlan(segments=())

    def test_duplicate_names_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must be unique"):
            WalkForwardPlan(
                segments=(
                    SegmentSpec(name="train", length_sessions=1),
                    SegmentSpec(name="train", length_sessions=1),
                )
            )

    def test_out_of_order_names_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="canonical order"):
            WalkForwardPlan(
                segments=(
                    SegmentSpec(name="test", length_sessions=1),
                    SegmentSpec(name="train", length_sessions=1),
                )
            )

    def test_canonical_subsequence_is_accepted(self) -> None:
        plan_ = WalkForwardPlan(
            segments=(
                SegmentSpec(name="train", length_sessions=1),
                SegmentSpec(name="test", length_sessions=1),
            )
        )
        assert [segment.name for segment in plan_.segments] == ["train", "test"]

    @pytest.mark.parametrize("length", [0, -3])
    def test_non_positive_length_is_rejected(self, length: int) -> None:
        with pytest.raises(ValidationError):
            SegmentSpec(name="train", length_sessions=length)

    def test_plan_is_frozen(self) -> None:
        with pytest.raises(ValidationError):
            plan().segments = ()  # type: ignore[misc]


class TestRequireNoLeakage:
    def segments(self, *, gap: int) -> tuple[SegmentSessions, ...]:
        sessions = weekdays(count=3)
        earlier = SegmentSessions(name="train", start_index=0, sessions=sessions[:2])
        later = SegmentSessions(name="test", start_index=gap + 2, sessions=sessions[2:3])
        return (earlier, later)

    def test_exact_embargo_gap_passes(self) -> None:
        require_no_leakage(self.segments(gap=EMBARGO), embargo_sessions=EMBARGO, horizon_sessions=5)

    def test_single_session_embargo_and_horizon_passes(self) -> None:
        # embargo == horizon == 1：合法下界（embargo 不能为 0，因为 horizon >= 1）
        require_no_leakage(self.segments(gap=1), embargo_sessions=1, horizon_sessions=1)

    def test_zero_embargo_is_rejected_because_labels_need_horizon(self) -> None:
        # embargo=0 在 horizon>=1 时必然泄漏：label 窗口本身就是未来 session
        with pytest.raises(ConfigurationError, match="embargo_sessions 0 < horizon_sessions 1"):
            require_no_leakage(self.segments(gap=0), embargo_sessions=0, horizon_sessions=1)

    def test_gap_below_embargo_raises_leakage_error(self) -> None:
        with pytest.raises(LeakageError) as excinfo:
            require_no_leakage(
                self.segments(gap=EMBARGO - 1), embargo_sessions=EMBARGO, horizon_sessions=5
            )
        message = str(excinfo.value)
        assert "leave only 4 sessions" in message
        assert f"embargo_sessions={EMBARGO} is required" in message

    def test_embargo_below_horizon_is_a_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError, match="embargo_sessions 1 < horizon_sessions 5"):
            require_no_leakage(self.segments(gap=1), embargo_sessions=1, horizon_sessions=5)

    @pytest.mark.parametrize(("embargo", "horizon"), [(-1, 1), (1, 0)])
    def test_invalid_arguments_are_rejected(self, embargo: int, horizon: int) -> None:
        with pytest.raises(ConfigurationError):
            require_no_leakage(
                self.segments(gap=5), embargo_sessions=embargo, horizon_sessions=horizon
            )


class TestSegmentSessionsValidation:
    def test_empty_sessions_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not be empty"):
            SegmentSessions(name="train", start_index=0, sessions=())

    def test_datetime_sessions_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not datetime"):
            SegmentSessions(
                name="train",
                start_index=0,
                sessions=(datetime(2026, 6, 1, 15, 0),),  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize(
        "sessions",
        [
            (date(2026, 6, 2), date(2026, 6, 1)),
            (date(2026, 6, 1), date(2026, 6, 1)),
        ],
    )
    def test_unsorted_or_duplicate_sessions_are_rejected(self, sessions: tuple[date, ...]) -> None:
        with pytest.raises(ValidationError):
            SegmentSessions(name="train", start_index=0, sessions=sessions)

    def test_negative_start_index_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SegmentSessions(name="test", start_index=-1, sessions=(date(2026, 6, 1),))

    def test_accessors(self) -> None:
        day = date(2026, 6, 1)
        segment = SegmentSessions(name="train", start_index=3, sessions=(day,))
        assert segment.first_session == segment.last_session == day
        assert segment.end_index == 3
        assert segment.num_sessions == 1
