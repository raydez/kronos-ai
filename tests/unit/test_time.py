from datetime import UTC, date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from kronos_ai.domain.time import (
    CUTOFF_POLICY_VERSION,
    MARKET_SESSION_CLOSE,
    SAME_DAY_EVENING,
    SHANGHAI,
    ResearchTime,
    cutoff_policy_record,
    resolve_knowledge_cutoff,
)

MD = date(2026, 9, 25)


class TestResolveKnowledgeCutoff:
    def test_market_close(self) -> None:
        assert resolve_knowledge_cutoff(MD, "market_close") == datetime(
            2026, 9, 25, 15, 0, tzinfo=SHANGHAI
        )

    def test_same_day_evening(self) -> None:
        assert resolve_knowledge_cutoff(MD, "same_day_evening") == datetime(
            2026, 9, 25, 18, 0, tzinfo=SHANGHAI
        )

    def test_explicit(self) -> None:
        cutoff = datetime(2026, 9, 25, 14, 30, tzinfo=SHANGHAI)
        assert resolve_knowledge_cutoff(MD, "explicit", cutoff) == cutoff

    def test_explicit_requires_value(self) -> None:
        with pytest.raises(ValueError, match="requires explicit_cutoff"):
            resolve_knowledge_cutoff(MD, "explicit")

    def test_explicit_naive_rejected(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            resolve_knowledge_cutoff(MD, "explicit", datetime(2026, 9, 25, 14, 30))

    def test_explicit_other_date_rejected(self) -> None:
        cutoff = datetime(2026, 9, 24, 18, 0, tzinfo=SHANGHAI)
        with pytest.raises(ValueError, match="must fall on market_date"):
            resolve_knowledge_cutoff(MD, "explicit", cutoff)

    def test_unknown_policy_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown cutoff policy"):
            resolve_knowledge_cutoff(MD, "whenever")  # type: ignore[arg-type]

    def test_policy_version_is_pinned(self) -> None:
        assert CUTOFF_POLICY_VERSION == "cutoff-policy-v1"


class TestResearchTime:
    def test_valid(self) -> None:
        rt = ResearchTime(
            market_date=MD, knowledge_cutoff=datetime(2026, 9, 25, 18, 0, tzinfo=SHANGHAI)
        )
        assert rt.market_date == MD

    def test_naive_cutoff_rejected(self) -> None:
        with pytest.raises(ValidationError, match="timezone-aware"):
            ResearchTime(market_date=MD, knowledge_cutoff=datetime(2026, 9, 25, 18, 0))

    def test_non_shanghai_offset_rejected(self) -> None:
        cutoff = datetime(2026, 9, 25, 10, 0, tzinfo=UTC)
        with pytest.raises(ValidationError, match="Asia/Shanghai"):
            ResearchTime(market_date=MD, knowledge_cutoff=cutoff)

    def test_cutoff_on_other_day_rejected(self) -> None:
        cutoff = datetime(2026, 9, 26, 18, 0, tzinfo=SHANGHAI)
        with pytest.raises(ValidationError, match="must fall on market_date"):
            ResearchTime(market_date=MD, knowledge_cutoff=cutoff)

    def test_arbitrary_plus_eight_offset_accepted(self) -> None:
        # 固定 +08:00 即视为 Asia/Shanghai 语义（无 DST）
        tz = timezone(timedelta(hours=8))
        rt = ResearchTime(market_date=MD, knowledge_cutoff=datetime(2026, 9, 25, 18, 0, tzinfo=tz))
        assert rt.knowledge_cutoff.utcoffset() == timedelta(hours=8)


class TestHistoricalDstWindow:
    """1986-1991 中国夏令时：域内固定 +08:00，不使用 ZoneInfo 的历史规则。"""

    def test_zoneinfo_premise(self) -> None:
        # 前提校验：tzdata 在该期间为 +09:00，这正是域内固定 +08:00 的原因
        tz = ZoneInfo("Asia/Shanghai")
        assert datetime(1991, 6, 3, 18, 0, tzinfo=tz).utcoffset() == timedelta(hours=9)

    @pytest.mark.parametrize("day", [date(1988, 7, 1), date(1991, 6, 3)])
    def test_cutoff_stays_plus_eight(self, day: date) -> None:
        cutoff = resolve_knowledge_cutoff(day, "same_day_evening")
        assert cutoff.utcoffset() == timedelta(hours=8)
        assert cutoff.hour == 18
        assert ResearchTime(market_date=day, knowledge_cutoff=cutoff).knowledge_cutoff == cutoff


class TestCutoffPolicyRecord:
    def test_market_close_parameters(self) -> None:
        record = cutoff_policy_record("market_close")
        assert record.policy == "market_close"
        assert record.policy_version == CUTOFF_POLICY_VERSION
        assert record.parameters == {"session_close": MARKET_SESSION_CLOSE.isoformat()}

    def test_same_day_evening_parameters(self) -> None:
        record = cutoff_policy_record("same_day_evening")
        assert record.parameters == {"evening": SAME_DAY_EVENING.isoformat()}

    def test_explicit_parameters(self) -> None:
        assert cutoff_policy_record("explicit").parameters == {"source": "caller_supplied"}

    def test_record_describes_resolved_cutoff(self) -> None:
        # parameters 与 resolve 结果不允许漂移
        record = cutoff_policy_record("same_day_evening")
        resolved = resolve_knowledge_cutoff(MD, "same_day_evening")
        assert resolved.strftime("%H:%M:%S") == record.parameters["evening"]

    def test_unknown_policy_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown cutoff policy"):
            cutoff_policy_record("whenever")  # type: ignore[arg-type]
