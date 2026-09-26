from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from kronos_ai.domain.market import (
    MARKET_CONTRACT_VERSION,
    MarketBar,
    MarketHistory,
    compute_market_history_hash,
)
from kronos_ai.domain.time import SHANGHAI

MD = date(2026, 9, 25)
CUTOFF = datetime(2026, 9, 25, 18, 0, tzinfo=SHANGHAI)


def bar(
    day: date,
    close: float = 10.0,
    symbol: str = "600000",
    available_at: datetime | None = None,
) -> MarketBar:
    ts = datetime(day.year, day.month, day.day, 15, 0, tzinfo=SHANGHAI)
    return MarketBar(
        symbol=symbol,
        timestamp=ts,
        open=9.0,
        high=10.5,
        low=8.5,
        close=close,
        volume=1_000_000.0,
        amount=10_000_000.0,
        trade_status="1",
        adjustment_mode="raw",
        available_at=available_at or datetime(day.year, day.month, day.day, 18, 0, tzinfo=SHANGHAI),
    )


def mutate(b: MarketBar, **changes: object) -> MarketBar:
    """Rebuild a bar through full validation (model_copy bypasses validators)."""
    return MarketBar(**{**b.model_dump(), **changes})


def history(bars: list[MarketBar], **overrides: object) -> MarketHistory:
    fields: dict[str, object] = {
        "symbol": "600000",
        "market_date": MD,
        "knowledge_cutoff": CUTOFF,
        "bars": bars,
        "provider": "baostock",
        "dataset_version": "test-dataset-v1",
    }
    fields.update(overrides)
    computed = compute_market_history_hash(
        symbol=fields["symbol"],
        market_date=fields["market_date"],
        knowledge_cutoff=fields["knowledge_cutoff"],
        provider=fields["provider"],
        dataset_version=fields["dataset_version"],
        bars=bars,
    )
    fields.setdefault("data_hash", computed)
    return MarketHistory(**fields)  # type: ignore[arg-type]


class TestMarketBar:
    def test_valid(self) -> None:
        b = bar(MD)
        assert b.timestamp.time().hour == 15

    def test_timestamp_must_be_session_close(self) -> None:
        ts = datetime(2026, 9, 25, 9, 31, tzinfo=SHANGHAI)
        with pytest.raises(ValidationError, match="session close"):
            mutate(bar(MD), timestamp=ts)

    def test_naive_timestamp_rejected(self) -> None:
        naive = datetime(2026, 9, 25, 15, 0)
        with pytest.raises(ValidationError, match="timezone-aware"):
            mutate(bar(MD), timestamp=naive)

    def test_available_at_before_timestamp_rejected(self) -> None:
        early = datetime(2026, 9, 25, 14, 0, tzinfo=SHANGHAI)
        with pytest.raises(ValidationError, match="available_at must be >= timestamp"):
            bar(MD, available_at=early)

    def test_leakage_available_at_after_cutoff_rejected(self) -> None:
        late = datetime(2026, 9, 25, 19, 0, tzinfo=SHANGHAI)
        b = bar(MD, available_at=late)
        with pytest.raises(ValidationError, match="leakage guard"):
            history([b])

    def test_ohlc_consistency(self) -> None:
        with pytest.raises(ValidationError, match="high"):
            mutate(bar(MD), high=9.5)
        with pytest.raises(ValidationError, match="low"):
            mutate(bar(MD), low=9.4)

    def test_nonpositive_price_rejected(self) -> None:
        with pytest.raises(ValidationError, match="> 0"):
            mutate(bar(MD), close=0.0)

    def test_non_shanghai_bar_offset_rejected(self) -> None:

        ts = datetime(2026, 9, 25, 7, 0, tzinfo=UTC)
        with pytest.raises(ValidationError, match="Asia/Shanghai"):
            mutate(bar(MD), timestamp=ts)

    def test_prefixed_symbol_rejected(self) -> None:
        with pytest.raises(ValidationError, match="6-digit"):
            bar(MD, symbol="sh.600000")


class TestMarketHistory:
    def test_valid(self) -> None:
        h = history([bar(date(2026, 9, 24)), bar(MD)])
        assert len(h.bars) == 2

    def test_unsorted_bars_rejected(self) -> None:
        with pytest.raises(ValidationError, match="sorted"):
            history([bar(MD), bar(date(2026, 9, 24))])

    def test_duplicate_timestamps_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unique"):
            history([bar(MD), bar(MD)])

    def test_symbol_mismatch_rejected(self) -> None:
        with pytest.raises(ValidationError, match="bar symbol"):
            history([bar(MD, symbol="000001")])

    def test_empty_bars_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not be empty"):
            history([])

    def test_data_hash_mismatch_rejected(self) -> None:
        with pytest.raises(ValidationError, match="data_hash mismatch"):
            history([bar(MD)], data_hash="deadbeef")

    def test_data_hash_is_content_deterministic(self) -> None:
        h1 = history([bar(date(2026, 9, 24)), bar(MD)])
        h2 = history([bar(date(2026, 9, 24)), bar(MD)])
        assert h1.data_hash == h2.data_hash

    def test_hash_changes_with_content(self) -> None:
        h1 = history([bar(MD)])
        h2 = history([bar(MD, close=10.1)])
        assert h1.data_hash != h2.data_hash

    def test_contract_version_pinned(self) -> None:
        assert MARKET_CONTRACT_VERSION == "market-contract-v1"
