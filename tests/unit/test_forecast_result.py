"""ForecastSample / ForecastDistribution / ForecastResult 契约（基线文档 §11/§12/§14）。"""

from datetime import date, datetime

import pytest
from pydantic import ValidationError

from kronos_ai.domain.forecast import (
    FORECAST_AGGREGATION_DEFINITION_VERSION,
    FORECAST_METRIC_NAMES,
    FORECAST_METRIC_REGISTRY_VERSION,
    SUPPORTED_FORECAST_AGGREGATION_VERSIONS,
    ForecastDistribution,
    ForecastPoint,
    ForecastResult,
    ForecastSample,
    ModelMetadata,
    QuantileValue,
    SamplingConfig,
    SamplingMetadata,
    ThresholdProbability,
    forecast_metric_definition,
)
from kronos_ai.domain.time import CN_TZ

MD = date(2026, 9, 25)
CUTOFF = datetime(2026, 9, 25, 18, 0, tzinfo=CN_TZ)
CLOSE_TS = datetime(2026, 9, 28, 15, 0, tzinfo=CN_TZ)
HASH = "a" * 64
SPEC_HASH = "c" * 64


def point(**overrides: object) -> ForecastPoint:
    fields: dict[str, object] = {
        "timestamp": CLOSE_TS,
        "open": 10.0,
        "high": 10.5,
        "low": 9.8,
        "close": 10.2,
        "volume": 1_000_000.0,
        "amount": 10_000_000.0,
    }
    fields.update(overrides)
    return ForecastPoint(**fields)  # type: ignore[arg-type]


def sample(sample_id: int = 0, horizon: int = 2) -> ForecastSample:
    return ForecastSample(sample_id=sample_id, points=tuple(point() for _ in range(horizon)))


def distribution(**overrides: object) -> ForecastDistribution:
    fields: dict[str, object] = {
        "horizon": 2,
        "sample_count": 3,
        "origin_close": 100.0,
        "expected_return": 0.01,
        "median_return": 0.0,
        "threshold_probabilities": (
            ThresholdProbability(
                metric="horizon_return", operator="gt", threshold=0.02, probability=0.25
            ),
        ),
        "quantiles": (QuantileValue(metric="horizon_return", quantile=0.5, value=0.0),),
        "forecast_dispersion": 0.03,
        "expected_max_drawdown": -0.05,
        "expected_path_volatility": 0.02,
        "distribution_spec_version": "distribution-spec-v1",
        "distribution_spec_hash": SPEC_HASH,
        "metric_definition_version": FORECAST_METRIC_REGISTRY_VERSION,
        "aggregation_definition_version": FORECAST_AGGREGATION_DEFINITION_VERSION,
    }
    fields.update(overrides)
    return ForecastDistribution(**fields)  # type: ignore[arg-type]


def model_metadata() -> ModelMetadata:
    return ModelMetadata(
        backend="kronos",
        model_id="NeoQuasar/Kronos-small",
        revision="deadbeef",
        runtime_version="torch-2.14.0",
        device="cpu",
        dtype="float32",
        config_hash=HASH,
    )


class TestMetricRegistry:
    def test_registry_names_are_versioned(self) -> None:
        assert FORECAST_METRIC_REGISTRY_VERSION == "forecast-metrics-v1"
        assert FORECAST_METRIC_NAMES == (
            "horizon_return",
            "log_return",
            "max_drawdown",
            "path_volatility",
        )

    def test_unknown_metric_explicit_failure(self) -> None:
        with pytest.raises(ValueError, match="unknown forecast metric"):
            forecast_metric_definition("sharpe")

    def test_threshold_metric_must_be_registered(self) -> None:
        with pytest.raises(ValidationError, match="unknown forecast metric"):
            ThresholdProbability(
                metric="p_return_gt_2pct", operator="gt", threshold=0.02, probability=0.3
            )

    def test_quantile_metric_must_be_registered(self) -> None:
        with pytest.raises(ValidationError, match="unknown forecast metric"):
            QuantileValue(metric="alpha", quantile=0.5, value=0.0)


class TestForecastPoint:
    def test_frozen(self) -> None:
        p = point()
        with pytest.raises(ValidationError):
            p.close = 1.0  # type: ignore[misc]

    def test_requires_shanghai_aware(self) -> None:
        with pytest.raises(ValidationError, match="timezone-aware"):
            point(timestamp=datetime(2026, 9, 28, 15, 0))

    def test_rejects_nonposiive_volume(self) -> None:
        with pytest.raises(ValidationError, match="non-negative"):
            point(volume=-1.0)

    def test_optional_flow_fields(self) -> None:
        p = point(volume=None, amount=None)
        assert p.volume is None and p.amount is None

    def test_rejects_non_finite_price(self) -> None:
        with pytest.raises(ValidationError, match="finite"):
            point(close=float("inf"))


class TestForecastSample:
    def test_points_are_immutable_tuple(self) -> None:
        s = sample()
        assert isinstance(s.points, tuple)

    def test_empty_points_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at least one"):
            ForecastSample(sample_id=0, points=())

    def test_negative_sample_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            sample(sample_id=-1)

    def test_bool_sample_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not bool"):
            ForecastSample(sample_id=True, points=(point(),))  # type: ignore[arg-type]


class TestForecastDistribution:
    def test_metric_definition_version_required(self) -> None:
        with pytest.raises(ValidationError, match="metric_definition_version"):
            distribution(metric_definition_version="")

    def test_unknown_metric_definition_version_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unsupported metric_definition_version"):
            distribution(metric_definition_version="forecast-metrics-v999")

    def test_distribution_spec_version_required(self) -> None:
        with pytest.raises(ValidationError, match="distribution_spec_version"):
            distribution(distribution_spec_version="")

    def test_aggregation_definition_version_required(self) -> None:
        with pytest.raises(ValidationError, match="aggregation_definition_version"):
            distribution(aggregation_definition_version="")

    def test_unknown_aggregation_definition_version_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unsupported aggregation_definition_version"):
            distribution(aggregation_definition_version="forecast-aggregation-v999")

    def test_origin_close_required_positive_finite(self) -> None:
        assert distribution().origin_close == 100.0
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(ValidationError, match="origin_close"):
                distribution(origin_close=bad)

    def test_fixture_aggregation_version_is_supported(self) -> None:
        # 夹具必须使用当前受支持的聚合口径版本，否则后续用例会在错误前提下运行
        assert (
            distribution().aggregation_definition_version
            in SUPPORTED_FORECAST_AGGREGATION_VERSIONS
        )

    def test_duplicate_thresholds_rejected(self) -> None:
        dup = ThresholdProbability(
            metric="horizon_return", operator="gt", threshold=0.02, probability=0.25
        )
        with pytest.raises(ValidationError, match="duplicate"):
            distribution(threshold_probabilities=(dup, dup))

    def test_duplicate_quantiles_rejected(self) -> None:
        dup = QuantileValue(metric="horizon_return", quantile=0.5, value=0.0)
        with pytest.raises(ValidationError, match="duplicate"):
            distribution(quantiles=(dup, dup))

    def test_probability_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ThresholdProbability(
                metric="horizon_return", operator="gt", threshold=0.02, probability=1.5
            )

    def test_no_hardcoded_two_percent_threshold(self) -> None:
        """§12/DoD16：schema 层不含写死的 2% 阈值，阈值完全由入参决定。"""
        empty = distribution(threshold_probabilities=(), quantiles=())
        assert empty.threshold_probabilities == () and empty.quantiles == ()


class TestSamplingMetadata:
    def test_from_config_roundtrip(self) -> None:
        config = SamplingConfig(seed=11, sample_count=8, temperature=0.7, top_k=5, top_p=0.8)
        meta = SamplingMetadata.from_config(config)
        assert (meta.seed, meta.sample_count, meta.temperature, meta.top_k, meta.top_p) == (
            11,
            8,
            0.7,
            5,
            0.8,
        )


class TestForecastResult:
    def test_happy_path(self) -> None:
        result = ForecastResult(
            symbol="600000",
            market_date=MD,
            knowledge_cutoff=CUTOFF,
            samples=(sample(0), sample(1), sample(2)),
            distribution=distribution(),
            model=model_metadata(),
            sampling=SamplingMetadata.from_config(SamplingConfig(seed=7, sample_count=3)),
            input_data_hash=HASH,
            artifact_id="forecast-artifact-key",
        )
        assert result.distribution.sample_count == 3

    def test_sample_count_must_match_distribution(self) -> None:
        with pytest.raises(ValidationError, match="sample_count"):
            ForecastResult(
                symbol="600000",
                market_date=MD,
                knowledge_cutoff=CUTOFF,
                samples=(sample(0), sample(1)),  # 2 != 3
                distribution=distribution(),
                model=model_metadata(),
                sampling=SamplingMetadata.from_config(SamplingConfig(seed=7, sample_count=3)),
                input_data_hash=HASH,
                artifact_id="key",
            )

    def test_horizon_must_match_points(self) -> None:
        with pytest.raises(ValidationError, match="horizon"):
            ForecastResult(
                symbol="600000",
                market_date=MD,
                knowledge_cutoff=CUTOFF,
                samples=(sample(0, horizon=3), sample(1, horizon=3), sample(2, horizon=3)),
                distribution=distribution(horizon=2),
                model=model_metadata(),
                sampling=SamplingMetadata.from_config(SamplingConfig(seed=7, sample_count=3)),
                input_data_hash=HASH,
                artifact_id="key",
            )

    def test_duplicate_sample_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="contiguous"):
            ForecastResult(
                symbol="600000",
                market_date=MD,
                knowledge_cutoff=CUTOFF,
                samples=(sample(0), sample(0), sample(2)),  # id 1 缺失、0 重复
                distribution=distribution(),
                model=model_metadata(),
                sampling=SamplingMetadata.from_config(SamplingConfig(seed=7, sample_count=3)),
                input_data_hash=HASH,
                artifact_id="key",
            )

    def test_out_of_order_sample_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="contiguous"):
            ForecastResult(
                symbol="600000",
                market_date=MD,
                knowledge_cutoff=CUTOFF,
                samples=(sample(1), sample(0), sample(2)),
                distribution=distribution(),
                model=model_metadata(),
                sampling=SamplingMetadata.from_config(SamplingConfig(seed=7, sample_count=3)),
                input_data_hash=HASH,
                artifact_id="key",
            )

    def test_cutoff_must_fall_on_market_date(self) -> None:
        with pytest.raises(ValidationError, match="market_date"):
            ForecastResult(
                symbol="600000",
                market_date=date(2026, 9, 24),
                knowledge_cutoff=CUTOFF,
                samples=(sample(0), sample(1), sample(2)),
                distribution=distribution(),
                model=model_metadata(),
                sampling=SamplingMetadata.from_config(SamplingConfig(seed=7, sample_count=3)),
                input_data_hash=HASH,
                artifact_id="key",
            )

    def test_sampling_sample_count_must_match_distribution(self) -> None:
        with pytest.raises(ValidationError, match=r"sampling\.sample_count"):
            ForecastResult(
                symbol="600000",
                market_date=MD,
                knowledge_cutoff=CUTOFF,
                samples=(sample(0), sample(1), sample(2)),
                distribution=distribution(),
                model=model_metadata(),
                sampling=SamplingMetadata.from_config(SamplingConfig(seed=7, sample_count=999)),
                input_data_hash=HASH,
                artifact_id="key",
            )

    def test_artifact_id_required(self) -> None:
        with pytest.raises(ValidationError, match="artifact_id"):
            ForecastResult(
                symbol="600000",
                market_date=MD,
                knowledge_cutoff=CUTOFF,
                samples=(sample(0), sample(1), sample(2)),
                distribution=distribution(),
                model=model_metadata(),
                sampling=SamplingMetadata.from_config(SamplingConfig(seed=7, sample_count=3)),
                input_data_hash=HASH,
                artifact_id="  ",
            )
