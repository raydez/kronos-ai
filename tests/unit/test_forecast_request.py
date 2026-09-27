from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest
from pydantic import ValidationError

from kronos_ai.domain.forecast import (
    FORECAST_CONTRACT_VERSION,
    MAX_SEED,
    ForecastRequest,
    ModelMetadata,
    SamplingConfig,
)
from kronos_ai.domain.hashing import sha256_hex
from kronos_ai.domain.time import CN_TZ, ResearchTime

MD = date(2026, 9, 25)
CUTOFF = datetime(2026, 9, 25, 18, 0, tzinfo=CN_TZ)

# Golden hashes：hashing_payload 形状的变更检测器。若本测试失败，说明契约 payload
# 变了——必须升级 FORECAST_CONTRACT_VERSION、更新常量，并确认既有 ForecastArtifactKey
# 失效是预期行为（§15），不得直接改常量让测试变绿。
GOLDEN_SAMPLING_HASH = "cfa270082e8a19c758d585a16333427fe228666cb19fd4d17ea59f61d55b8b9d"
GOLDEN_REQUEST_HASH = "ba54d58a4bb608efb6a5172970bbac0522392713fc77fec3573fe02c1a796100"


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

    def test_rejects_extra_field(self) -> None:
        # §8：未知字段（如 backend 侧的 lookback_bars）必须显式失败，不得被静默吞掉
        with pytest.raises(ValidationError, match="lookback_bars"):
            sampling(lookback_bars=256)

    @pytest.mark.parametrize("field", ["seed", "sample_count", "top_k"])
    def test_rejects_bool_for_int_fields(self, field: str) -> None:
        # bool 是 int 子类，宽松模式会把 True 静默变成 1
        with pytest.raises(ValidationError, match="not bool"):
            sampling(**{field: True})

    def test_accepts_numpy_ints(self) -> None:
        cfg = sampling(seed=np.int64(7), sample_count=np.int32(3))
        assert cfg.seed == 7
        assert cfg.sample_count == 3

    def test_hashing_payload_carries_kind_and_contract_version(self) -> None:
        payload = sampling().hashing_payload()
        assert payload["kind"] == "sampling_config"
        assert payload["contract_version"] == FORECAST_CONTRACT_VERSION

    def test_golden_hash(self) -> None:
        assert sha256_hex(sampling().hashing_payload()) == GOLDEN_SAMPLING_HASH

    def test_hashing_payload_is_stable_and_sensitive(self) -> None:
        assert sampling().hashing_payload() == sampling().hashing_payload()
        base = sha256_hex(sampling().hashing_payload())
        assert sha256_hex(sampling(seed=43).hashing_payload()) != base
        assert sha256_hex(sampling(sample_count=32).hashing_payload()) != base
        assert sha256_hex(sampling(temperature=0.7).hashing_payload()) != base
        assert sha256_hex(sampling(top_k=50).hashing_payload()) != base
        assert sha256_hex(sampling(top_p=0.95).hashing_payload()) != base


class TestModelMetadata:
    def metadata(self, **overrides: object) -> ModelMetadata:
        fields: dict[str, object] = {
            "backend": "kronos",
            "model_id": "NeoQuasar/Kronos-small",
            "revision": "901c26c1332695a2a8f243eb2f37243a37bea320",
            "runtime_version": "kronos-runtime-v1/torch-2.14.0",
            "device": "cpu",
            "dtype": "float32",
            "config_hash": "0" * 64,
        }
        fields.update(overrides)
        return ModelMetadata(**fields)  # type: ignore[arg-type]

    def test_valid(self) -> None:
        assert self.metadata().device == "cpu"

    def test_rejects_extra_field(self) -> None:
        with pytest.raises(ValidationError, match="lookback_bars"):
            self.metadata(lookback_bars=256)

    @pytest.mark.parametrize(
        "field", ["backend", "model_id", "revision", "runtime_version", "device", "dtype"]
    )
    def test_rejects_blank_strings(self, field: str) -> None:
        with pytest.raises(ValidationError):
            self.metadata(**{field: "   "})

    def test_strips_surrounding_whitespace(self) -> None:
        # 不去空白会让 " cpu" 与 "cpu" 成为两条本应相同的 provenance 记录
        md = self.metadata(device=" cpu ", model_id="  NeoQuasar/Kronos-small ")
        assert md.device == "cpu"
        assert md.model_id == "NeoQuasar/Kronos-small"

    def test_rejects_malformed_config_hash(self) -> None:
        with pytest.raises(ValidationError, match="sha256"):
            self.metadata(config_hash="deadbeef")


class TestForecastRequest:
    def test_valid_defaults(self) -> None:
        req = request()
        assert req.horizon == 5
        assert req.symbol == "600000"

    def test_lookback_bars_is_not_a_request_field(self) -> None:
        # §8：lookback_bars 属于 backend/model 配置，不属于 Request
        assert "lookback_bars" not in ForecastRequest.model_fields

    def test_lookback_bars_rejected_explicitly(self) -> None:
        with pytest.raises(ValidationError, match="lookback_bars"):
            request(lookback_bars=256)

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

    def test_horizon_rejects_bool(self) -> None:
        with pytest.raises(ValidationError, match="not bool"):
            request(horizon=True)

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
        assert payload["kind"] == "forecast_request"
        assert payload["contract_version"] == FORECAST_CONTRACT_VERSION
        assert payload["symbol"] == "600000"
        assert payload["market_date"] == "2026-09-25"
        assert payload["horizon"] == 5
        assert payload["sampling"]["seed"] == 42

    def test_golden_hash(self) -> None:
        assert sha256_hex(request().hashing_payload()) == GOLDEN_REQUEST_HASH

    def test_hashing_payload_distinguishes_requests(self) -> None:
        base = sha256_hex(request().hashing_payload())
        assert sha256_hex(request(horizon=6).hashing_payload()) != base
        assert sha256_hex(request(sampling=sampling(seed=7)).hashing_payload()) != base
        assert sha256_hex(request(symbol="000001").hashing_payload()) != base
        prev_day = request(
            market_date=date(2026, 9, 24),
            knowledge_cutoff=datetime(2026, 9, 24, 18, 0, tzinfo=CN_TZ),
        )
        assert sha256_hex(prev_day.hashing_payload()) != base
        assert (
            sha256_hex(
                request(
                    knowledge_cutoff=datetime(2026, 9, 25, 15, 0, tzinfo=CN_TZ)
                ).hashing_payload()
            )
            != base
        )
        assert sha256_hex(request().hashing_payload()) == base
