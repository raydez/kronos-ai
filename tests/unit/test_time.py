from datetime import UTC, date, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from kronos_ai.domain.time import (
    CUTOFF_POLICY_VERSION,
    SHANGHAI,
    ResearchTime,
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
