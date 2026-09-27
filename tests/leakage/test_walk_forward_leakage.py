"""§29 Leakage guard 回归测试（RX-KAI-017）。

这些用例证明「按 horizon 断言」是必要的但不充分：当 ``gap == horizon`` 而
``gap < embargo_sessions`` 时，horizon 断言会放过，只有 embargo 断言能拦住。
数据集模型层的手工构造路径（绕过 builder）也必须拦得住。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from kronos_ai.data.calendar import StaticTradingCalendar
from kronos_ai.domain.market import MarketBar
from kronos_ai.domain.time import CN_TZ, MARKET_SESSION_CLOSE, resolve_knowledge_cutoff
from kronos_ai.errors import ConfigurationError, DataQualityError, LeakageError
from kronos_ai.evaluation.dataset import (
    DEFAULT_LABEL_POLICY,
    DatasetSegment,
    ForecastOrigin,
    WalkForwardDataset,
    build_label,
)
from kronos_ai.evaluation.walk_forward import (
    SegmentSessions,
    SegmentSpec,
    WalkForwardPlan,
    require_no_leakage,
    split_sessions,
)

pytestmark = pytest.mark.leakage

SYMBOL = "600000"
EXCHANGE = "SSE"
SOURCE = "test-fixture"
START = date(2026, 6, 1)
LOOKBACK = 6
EMBARGO_SESSIONS = 5
HORIZON_SESSIONS = DEFAULT_LABEL_POLICY.horizon_sessions


def weekdays(*, count: int) -> tuple[date, ...]:
    days: list[date] = []
    current = START
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return tuple(days)


def calendar(count: int) -> StaticTradingCalendar:
    return StaticTradingCalendar(exchange=EXCHANGE, source=SOURCE, sessions=weekdays(count=count))


def gap_sessions(gap: int) -> tuple[SegmentSessions, ...]:
    """两段之间恰好空 ``gap`` 个 session（gap=0 即首尾相接）。"""
    timeline = weekdays(count=8 + gap + 4)
    train = SegmentSessions(name="train", start_index=0, sessions=timeline[:8])
    test = SegmentSessions(name="test", start_index=8 + gap, sessions=timeline[8 + gap :])
    return (train, test)


def leaky_dataset() -> WalkForwardDataset:
    """时间轴上没有任何 embargo 空隙的样本集（等价于 §27 被绕过）。"""
    total = LOOKBACK + 8 + 4 + 4
    timeline = weekdays(count=total + HORIZON_SESSIONS)
    cal = StaticTradingCalendar(exchange=EXCHANGE, source=SOURCE, sessions=timeline)
    slices = (("train", LOOKBACK, 8), ("validation", LOOKBACK + 8, 4), ("test", LOOKBACK + 12, 4))
    segments = []
    for name, start, length in slices:
        days = timeline[start : start + length]
        origins = tuple(
            ForecastOrigin(
                symbol=SYMBOL,
                market_date=day,
                knowledge_cutoff=resolve_knowledge_cutoff(day, "same_day_evening"),
                label_sessions=tuple(cal.next_sessions(day, HORIZON_SESSIONS)),
            )
            for day in days
        )
        segments.append(
            DatasetSegment(name=name, start_index=start, sessions=days, origins=origins)  # type: ignore[arg-type]
        )
    return WalkForwardDataset(
        version="walk-forward-dataset-v1",
        label_policy=DEFAULT_LABEL_POLICY,
        calendar_exchange=EXCHANGE,
        calendar_source=SOURCE,
        cutoff_policy="same_day_evening",
        lookback_bars=LOOKBACK,
        symbols=(SYMBOL,),
        sessions=timeline[:total],
        segments=tuple(segments),  # type: ignore[arg-type]
    )


class TestRequireNoLeakage:
    def test_horizon_sized_gap_passes_a_horizon_assertion_but_not_embargo(self) -> None:
        # gap == horizon == 3 < embargo == 5：只断言 gap >= horizon 会放过这一段，
        # 但 label 定义（§27/§32.1）要求空置至少 embargo_sessions，必须拦住。
        leaky = gap_sessions(gap=3)
        with pytest.raises(LeakageError, match="leave only 3 sessions"):
            require_no_leakage(leaky, embargo_sessions=EMBARGO_SESSIONS, horizon_sessions=3)

    def test_embargo_sized_gap_passes(self) -> None:
        require_no_leakage(
            gap_sessions(gap=EMBARGO_SESSIONS),
            embargo_sessions=EMBARGO_SESSIONS,
            horizon_sessions=HORIZON_SESSIONS,
        )

    def test_contiguous_segments_are_rejected(self) -> None:
        with pytest.raises(LeakageError, match="leave only 0 sessions"):
            require_no_leakage(
                gap_sessions(gap=0),
                embargo_sessions=EMBARGO_SESSIONS,
                horizon_sessions=HORIZON_SESSIONS,
            )

    def test_embargo_shorter_than_horizon_is_a_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError, match="embargo_sessions 3 < horizon_sessions 5"):
            require_no_leakage(
                gap_sessions(gap=EMBARGO_SESSIONS),
                embargo_sessions=3,
                horizon_sessions=HORIZON_SESSIONS,
            )


class TestDatasetLevelLeakage:
    def test_hand_built_dataset_without_embargo_gaps_raises_leakage_error(self) -> None:
        # LeakageError 必须从 model_validator 直接冒出，而不是被包装成 ValidationError：
        # 数据集完整性不是字段格式问题（§29）
        with pytest.raises(LeakageError, match="leave only 0 sessions"):
            leaky_dataset()

    def test_builder_output_is_leakage_free(self) -> None:
        plan = WalkForwardPlan(
            segments=(
                SegmentSpec(name="train", length_sessions=8),
                SegmentSpec(name="validation", length_sessions=4),
                SegmentSpec(name="test", length_sessions=4),
            )
        )
        total = LOOKBACK + 16 + EMBARGO_SESSIONS * 2
        timeline = weekdays(count=total + HORIZON_SESSIONS)
        split = split_sessions(
            sessions=timeline[:total],
            plan=plan,
            embargo_sessions=EMBARGO_SESSIONS,
            history_prefix=LOOKBACK,
        )
        require_no_leakage(
            split, embargo_sessions=EMBARGO_SESSIONS, horizon_sessions=HORIZON_SESSIONS
        )


class TestLabelLeakage:
    def test_bar_available_before_the_cutoff_is_rejected(self) -> None:
        # MarketBar 的 available_at >= timestamp 不变式使得「label 窗口内的 bar 早于
        # origin cutoff」在正常数据路径下不可达；这里用 model_construct 绕过该不变式，
        # 保证 §29 守卫在异常上游（例如手工拼接的 bar）下依然生效。
        day = weekdays(count=LOOKBACK + 1)[LOOKBACK]
        cal = calendar(LOOKBACK + 1 + HORIZON_SESSIONS)
        origin = ForecastOrigin(
            symbol=SYMBOL,
            market_date=day,
            knowledge_cutoff=resolve_knowledge_cutoff(day, "same_day_evening"),
            label_sessions=tuple(cal.next_sessions(day, HORIZON_SESSIONS)),
        )
        leaked = MarketBar.model_construct(
            symbol=SYMBOL,
            timestamp=datetime.combine(
                origin.label_sessions[0], MARKET_SESSION_CLOSE, tzinfo=CN_TZ
            ),
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            trade_status="1",
            adjustment_mode="raw",
            available_at=origin.knowledge_cutoff,
        )
        with pytest.raises(LeakageError, match="already available at the origin") as caught:
            build_label(
                origin=origin,
                policy=DEFAULT_LABEL_POLICY,
                origin_close=100.0,
                future_bars=(leaked,),
                data_coverage_end=origin.label_sessions[-1],
            )
        # §29 泄漏不是字段格式错误：不能被 pydantic 包成 ValidationError。
        assert not isinstance(caught.value, DataQualityError)
