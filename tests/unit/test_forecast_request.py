from datetime import UTC, date, datetime, timedelta

import pytest
from pydantic import ValidationError

from kronos_ai.domain.forecast import (
    FORECAST_CONTRACT_VERSION,
    MAX_SEED,
    ForecastRequest,
    SamplingConfig,
)
from kronos_ai.domain.hashing import sha256_hex
from kronos_ai.domain.time import CN_TZ, ResearchTime

MD = date(2026, 9, 25)
CUTOFF = datetime(2026, 9, 25, 18, 0, tzinfo=CN_TZ)


def sampling(**overrides: object) -> SamplingConfig:
    fields: dict[str, object] = {"seed": 42}
    fields.update(overrides)
    return SamplingConfig(**fields)  # type: ignore[arg-type]


def request(**overrides: object) -> ForecastRequest:
    fields: dict[str, object] = {
        "symbol": "600000",
        "market_date": MD,
        "knowledge_cutoff": CUTOFF,
        "sampling": sampling(),
    }
    fields.update(overrides)
    return ForecastRequest(**fields)  # type: ignore[arg-type]


class TestSamplingConfig:
    def test_defaults(self) -> None:
        cfg = sampling()
        assert cfg.sample_count == 64
        assert cfg.temperature == 1.0
        assert cfg.top_k == 0
        assert cfg.top_p == 0.9

    def test_seed_required(self) -> None:
        with pytest.raises(ValidationError, match="seed"):
            SamplingConfig()  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        cfg = sampling()
        with pytest.raises(ValidationError):
            cfg.seed = 1  # type: ignore[misc]

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("seed", -1),
            ("seed", MAX_SEED + 1),
            ("sample_count", 0),
            ("temperature", 0.0),
            ("temperature", float("inf")),
            ("top_k", -1),
            ("top_p", 0.0),
            ("top_p", 1.01),
        ],
    )
    def test_rejects_out_of_range(self, field: str, value: object) -> None:
        with pytest.raises(ValidationError):
            sampling(**{field: value})

    def test_boundary_values_accepted(self) -> None:
        cfg = sampling(seed=0, sample_count=1, temperature=1e-9, top_k=0, top_p=1.0)
        assert cfg.seed == 0
        assert cfg.top_p == 1.0
        assert sampling(seed=MAX_SEED).seed == MAX_SEED

    def test_hashing_payload_is_stable_and_sensitive(self) -> None:
        assert sampling().hashing_payload() == sampling().hashing_payload()
        base = sha256_hex(sampling().hashing_payload())
        assert sha256_hex(sampling(seed=43).hashing_payload()) != base
        assert sha256_hex(sampling(sample_count=32).hashing_payload()) != base
        assert sha256_hex(sampling(top_p=0.95).hashing_payload()) != base


class TestForecastRequest:
    def test_valid_defaults(self) -> None:
        req = request()
        assert req.horizon == 5
        assert req.symbol == "600000"

    def test_lookback_bars_is_not_a_request_field(self) -> None:
        # §8：lookback_bars 属于 backend/model 配置，不属于 Request
        assert "lookback_bars" not in ForecastRequest.model_fields

    @pytest.mark.parametrize("symbol", ["sh.600000", "60000", "abc123", ""])
    def test_symbol_must_be_normalized(self, symbol: str) -> None:
        with pytest.raises(ValidationError):
            request(symbol=symbol)

    def test_naive_cutoff_rejected(self) -> None:
        with pytest.raises(ValidationError, match="timezone-aware"):
            request(knowledge_cutoff=datetime(2026, 9, 25, 18, 0))

    def test_non_shanghai_cutoff_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"\+08:00"):
            request(knowledge_cutoff=datetime(2026, 9, 25, 18, 0, tzinfo=UTC))

    def test_cutoff_must_fall_on_market_date(self) -> None:
        with pytest.raises(ValidationError, match="must fall on market_date"):
            request(knowledge_cutoff=CUTOFF + timedelta(days=1))

    def test_horizon_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            request(horizon=0)

    def test_frozen(self) -> None:
        req = request()
        with pytest.raises(ValidationError):
            req.horizon = 10  # type: ignore[misc]

    def test_sampling_is_required(self) -> None:
        with pytest.raises(ValidationError, match="sampling"):
            ForecastRequest(  # type: ignore[call-arg]
                symbol="600000", market_date=MD, knowledge_cutoff=CUTOFF
            )

    def test_research_time_projection(self) -> None:
        req = request()
        assert req.research_time == ResearchTime(market_date=MD, knowledge_cutoff=CUTOFF)

    def test_hashing_payload_structure_and_version(self) -> None:
        payload = request().hashing_payload()
        assert payload["symbol"] == "600000"
        assert payload["market_date"] == "2026-09-25"
        assert payload["horizon"] == 5
        assert payload["sampling"]["seed"] == 42
        assert FORECAST_CONTRACT_VERSION == "forecast-contract-v1"

    def test_hashing_payload_distinguishes_requests(self) -> None:
        base = sha256_hex(request().hashing_payload())
        assert sha256_hex(request(horizon=6).hashing_payload()) != base
        assert sha256_hex(request(sampling=sampling(seed=7)).hashing_payload()) != base
        assert sha256_hex(request(symbol="000001").hashing_payload()) != base
        assert sha256_hex(request().hashing_payload()) == base
