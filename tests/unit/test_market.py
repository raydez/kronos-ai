from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from kronos_ai.domain.market import (
    MARKET_CONTRACT_VERSION,
    MarketBar,
    MarketHistory,
    compute_market_history_hash,
)
from kronos_ai.domain.time import CN_TZ

MD = date(2026, 9, 25)
CUTOFF = datetime(2026, 9, 25, 18, 0, tzinfo=CN_TZ)

# golden 常量锚定 compute_market_history_hash 的 payload 构成：
# 修改 payload（增删字段/改版本）必须同步更新，否则视为契约破坏。
GOLDEN_SINGLE_HASH = "c590463ac1e5a86d86f5eae10640e6b18974848ccca09d88f2fc9f606684bce9"
GOLDEN_DOUBLE_HASH = "8c0442390c0b815f720bbfa45775ff60596befc98c78d4a19a87d9c397978c5c"


def bar(
    day: date,
    close: float = 10.0,
    symbol: str = "600000",
    available_at: datetime | None = None,
) -> MarketBar:
    ts = datetime(day.year, day.month, day.day, 15, 0, tzinfo=CN_TZ)
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
        available_at=available_at or datetime(day.year, day.month, day.day, 18, 0, tzinfo=CN_TZ),
    )


def mutate(b: MarketBar, **changes: object) -> MarketBar:
    """Rebuild a bar through full validation (model_copy bypasses validators)."""
    return MarketBar(**{**b.model_dump(), **changes})


def history(bars: list[MarketBar], **overrides: object) -> MarketHistory:
    fields: dict[str, object] = {
        "symbol": "600000",
        "market_date": MD,
        "knowledge_cutoff": CUTOFF,
        "bars": tuple(bars),
        "provider": "baostock",
        "dataset_version": "test-dataset-v1",
    }
    fields.update(overrides)
    return MarketHistory(**fields)  # type: ignore[arg-type]


class TestMarketBar:
    def test_valid(self) -> None:
        b = bar(MD)
        assert b.timestamp.time().hour == 15

    def test_timestamp_must_be_session_close(self) -> None:
        ts = datetime(2026, 9, 25, 9, 31, tzinfo=CN_TZ)
        with pytest.raises(ValidationError, match="session close"):
            mutate(bar(MD), timestamp=ts)

    def test_timestamp_close_precision_is_exact(self) -> None:
        ts = datetime(2026, 9, 25, 15, 0, 30, tzinfo=CN_TZ)
        with pytest.raises(ValidationError, match="session close"):
            mutate(bar(MD), timestamp=ts)

    def test_naive_timestamp_rejected(self) -> None:
        naive = datetime(2026, 9, 25, 15, 0)
        with pytest.raises(ValidationError, match="timezone-aware"):
            mutate(bar(MD), timestamp=naive)

    def test_available_at_before_timestamp_rejected(self) -> None:
        early = datetime(2026, 9, 25, 14, 0, tzinfo=CN_TZ)
        with pytest.raises(ValidationError, match="available_at must be >= timestamp"):
            bar(MD, available_at=early)

    def test_available_at_equal_timestamp_accepted(self) -> None:
        ts = datetime(2026, 9, 25, 15, 0, tzinfo=CN_TZ)
        b = bar(MD, available_at=ts)
        assert b.available_at == b.timestamp

    def test_leakage_available_at_after_cutoff_rejected(self) -> None:
        late = datetime(2026, 9, 25, 19, 0, tzinfo=CN_TZ)
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

    def test_fullwidth_digits_rejected(self) -> None:
        with pytest.raises(ValidationError, match="6-digit"):
            bar(MD, symbol="６０００００")

    def test_frozen(self) -> None:
        b = bar(MD)
        with pytest.raises(ValidationError):
            b.close = 11.0  # type: ignore[misc]


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

    def test_future_bar_after_market_date_rejected(self) -> None:
        # cutoff 允许晚于 market_date（宽松场景），未来 bar 仍必须被拦下
        late_cutoff = datetime(2026, 9, 28, 18, 0, tzinfo=CN_TZ)
        with pytest.raises(ValidationError, match="after market_date"):
            history([bar(date(2026, 9, 26))], knowledge_cutoff=late_cutoff)

    def test_history_frozen(self) -> None:
        h = history([bar(MD)])
        with pytest.raises(ValidationError):
            h.bars = ()  # type: ignore[misc]


class TestDataHash:
    def test_golden_single(self) -> None:
        h = history([bar(MD)])
        assert h.data_hash == GOLDEN_SINGLE_HASH

    def test_golden_double(self) -> None:
        h = history([bar(date(2026, 9, 24)), bar(MD)])
        assert h.data_hash == GOLDEN_DOUBLE_HASH

    def test_is_content_deterministic(self) -> None:
        h1 = history([bar(date(2026, 9, 24)), bar(MD)])
        h2 = history([bar(date(2026, 9, 24)), bar(MD)])
        assert h1.data_hash == h2.data_hash

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("open", 9.2),
            ("high", 10.8),
            ("low", 8.6),
            ("close", 10.2),
            ("volume", 2_000_000.0),
            ("amount", 20_000_000.0),
        ],
    )
    def test_bar_numeric_field_changes_hash(self, field: str, value: float) -> None:
        base = history([bar(MD)])
        changed_bar = mutate(bar(MD), **{field: value})
        assert history([changed_bar]).data_hash != base.data_hash

    @pytest.mark.parametrize(
        ("field", "value", "bars"),
        [
            ("symbol", "000001", [bar(MD, symbol="000001")]),
            ("provider", "other", [bar(MD)]),
            ("dataset_version", "other-v2", [bar(MD)]),
            ("market_date", date(2026, 9, 24), [bar(date(2026, 9, 24))]),
            ("knowledge_cutoff", datetime(2026, 9, 25, 19, 0, tzinfo=CN_TZ), [bar(MD)]),
        ],
    )
    def test_history_field_changes_hash(
        self, field: str, value: object, bars: list[MarketBar]
    ) -> None:
        base = history([bar(MD)])
        assert history(bars, **{field: value}).data_hash != base.data_hash

    def test_bar_trade_status_and_mode_change_hash(self) -> None:
        base = history([bar(MD)])
        assert history([mutate(bar(MD), trade_status="suspended")]).data_hash != base.data_hash
        assert history([mutate(bar(MD), adjustment_mode="qfq")]).data_hash != base.data_hash

    def test_hash_cannot_be_injected(self) -> None:
        h = history([bar(MD)], data_hash="deadbeef")
        assert h.data_hash == GOLDEN_SINGLE_HASH

    def test_hash_roundtrip_via_dump(self) -> None:
        h = history([bar(MD)])
        assert MarketHistory.model_validate(h.model_dump()).data_hash == h.data_hash

    def test_contract_version_pinned(self) -> None:
        assert MARKET_CONTRACT_VERSION == "market-contract-v1"

    def test_hash_function_accepts_sequence(self) -> None:
        h = history([bar(MD)])
        assert (
            compute_market_history_hash(
                symbol=h.symbol,
                market_date=h.market_date,
                knowledge_cutoff=h.knowledge_cutoff,
                provider=h.provider,
                dataset_version=h.dataset_version,
                bars=h.bars,
            )
            == h.data_hash
        )
