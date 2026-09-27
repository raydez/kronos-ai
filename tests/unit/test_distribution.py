"""ForecastDistribution 指标数学与 raw→domain 转换（基线文档 §11/§12/§13）。"""

from __future__ import annotations

import math
from datetime import date, datetime

import numpy as np
import pytest
from pydantic import ValidationError

from kronos_ai.domain.forecast import (
    FORECAST_METRIC_NAMES,
    FORECAST_METRIC_REGISTRY_VERSION,
    ForecastRequest,
    ForecastSample,
    SamplingConfig,
)
from kronos_ai.domain.time import CN_TZ, MARKET_SESSION_CLOSE
from kronos_ai.errors import ModelInferenceError
from kronos_ai.forecast.backends.kronos.sampler import FEATURE_NAMES, KronosSampler, RawSampleSet
from kronos_ai.forecast.distribution import (
    DEFAULT_DISTRIBUTION_SPEC,
    DISTRIBUTION_SPEC_VERSION,
    DistributionSpec,
    QuantileSpec,
    ThresholdSpec,
    _metric_series,
    build_distribution,
    forecast_samples_from_raw,
)

MD = date(2026, 9, 25)
CUTOFF = datetime(2026, 9, 25, 18, 0, tzinfo=CN_TZ)
SESSIONS = (date(2026, 9, 28), date(2026, 9, 29))


def raw_set(closes: list[list[float]], *, origin_close: float = 100.0) -> RawSampleSet:
    """按给定 close 矩阵造 RawSampleSet；其余 feature 用占位正值。"""
    n = len(closes)
    horizon = len(closes[0])
    values = np.zeros((n, horizon, len(FEATURE_NAMES)), dtype=np.float64)
    for i, row in enumerate(closes):
        for j, close in enumerate(row):
            values[i, j] = (close, close + 0.1, close - 0.1, close, 1000.0, 10_000.0)
    return RawSampleSet(
        symbol="600000",
        market_date=MD,
        knowledge_cutoff=CUTOFF,
        horizon=horizon,
        future_sessions=SESSIONS[:horizon],
        feature_names=FEATURE_NAMES,
        values=values,
    )


def pop_std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def path_volatility(closes: list[float], origin_close: float) -> float:
    path = [origin_close, *closes]
    log_returns = [math.log(path[t + 1] / path[t]) for t in range(len(path) - 1)]
    return pop_std(log_returns)


def test_horizon_return_and_thresholds() -> None:
    # closes: [100,110] / [95,90] / [105,120]；P_0=100
    # horizon_return = [0.1, -0.1, 0.2]
    dist = build_distribution(
        raw_set([[100.0, 110.0], [95.0, 90.0], [105.0, 120.0]]),
        origin_close=100.0,
        spec=DEFAULT_DISTRIBUTION_SPEC,
    )

    assert dist.horizon == 2
    assert dist.sample_count == 3
    assert dist.metric_definition_version == FORECAST_METRIC_REGISTRY_VERSION
    assert dist.distribution_spec_version == DEFAULT_DISTRIBUTION_SPEC.version
    assert dist.expected_return == pytest.approx(0.2 / 3)
    assert dist.median_return == pytest.approx(0.1)
    assert dist.forecast_dispersion == pytest.approx(pop_std([0.1, -0.1, 0.2]))

    probs = {(t.metric, t.operator, t.threshold): t.probability for t in dist.threshold_probabilities}
    assert probs[("horizon_return", "gt", -0.02)] == pytest.approx(2 / 3)
    assert probs[("horizon_return", "gt", 0.0)] == pytest.approx(2 / 3)
    assert probs[("horizon_return", "gt", 0.02)] == pytest.approx(2 / 3)
    assert probs[("horizon_return", "lt", -0.02)] == pytest.approx(1 / 3)


def test_quantiles_linear_interpolation() -> None:
    # sorted = [-0.1, 0.1, 0.2]，numpy linear 插值
    dist = build_distribution(
        raw_set([[100.0, 110.0], [95.0, 90.0], [105.0, 120.0]]),
        origin_close=100.0,
        spec=DEFAULT_DISTRIBUTION_SPEC,
    )
    by_q = {q.quantile: q.value for q in dist.quantiles}
    assert by_q[0.05] == pytest.approx(-0.08)
    assert by_q[0.25] == pytest.approx(0.0)
    assert by_q[0.5] == pytest.approx(0.1)
    assert by_q[0.75] == pytest.approx(0.15)
    assert by_q[0.95] == pytest.approx(0.19)


def test_drawdown_and_volatility() -> None:
    closes = [[100.0, 110.0], [95.0, 90.0], [105.0, 120.0]]
    dist = build_distribution(raw_set(closes), origin_close=100.0)

    # MDD: [0, -0.1, 0] → mean -1/30
    assert dist.expected_max_drawdown == pytest.approx(-0.1 / 3)
    expected_vol = sum(path_volatility(row, 100.0) for row in closes) / len(closes)
    assert dist.expected_path_volatility == pytest.approx(expected_vol)


def test_custom_spec_targets_other_metrics() -> None:
    spec = DistributionSpec(
        thresholds=(ThresholdSpec(metric="max_drawdown", operator="lte", threshold=-0.05),),
        quantiles=(QuantileSpec(metric="path_volatility", quantile=0.5),),
    )
    dist = build_distribution(
        raw_set([[100.0, 110.0], [95.0, 90.0], [105.0, 120.0]]),
        origin_close=100.0,
        spec=spec,
    )
    assert len(dist.threshold_probabilities) == 1
    assert dist.threshold_probabilities[0].probability == pytest.approx(1 / 3)
    assert dist.quantiles[0].metric == "path_volatility"


def test_metric_series_covers_registry() -> None:
    """registry 新增指标但漏实现时，_metric_series 必须与 registry 键集一致。"""
    metrics = _metric_series(raw_set([[100.0, 110.0]]), origin_close=100.0)
    assert set(metrics) == set(FORECAST_METRIC_NAMES)


def test_origin_close_must_be_positive() -> None:
    raw = raw_set([[100.0, 110.0]])
    with pytest.raises(ModelInferenceError, match="origin_close"):
        build_distribution(raw, origin_close=0.0)


def test_origin_close_must_be_finite() -> None:
    raw = raw_set([[100.0, 110.0]])
    with pytest.raises(ModelInferenceError, match="finite"):
        build_distribution(raw, origin_close=float("nan"))


def test_build_distribution_requires_close_feature() -> None:
    raw = RawSampleSet(
        symbol="600000",
        market_date=MD,
        knowledge_cutoff=CUTOFF,
        horizon=1,
        future_sessions=(date(2026, 9, 28),),
        feature_names=("open", "high", "low"),
        values=np.full((1, 1, 3), 10.0),
    )
    with pytest.raises(ModelInferenceError, match="lacks 'close'"):
        build_distribution(raw, origin_close=10.0)


def test_gte_lte_operators() -> None:
    spec = DistributionSpec(
        thresholds=(
            ThresholdSpec(metric="horizon_return", operator="gte", threshold=0.0),
            ThresholdSpec(metric="horizon_return", operator="lte", threshold=0.0),
        )
    )
    dist = build_distribution(
        raw_set([[100.0, 110.0], [95.0, 90.0], [105.0, 120.0]]),
        origin_close=100.0,
        spec=spec,
    )
    probs = {t.operator: t.probability for t in dist.threshold_probabilities}
    assert probs["gte"] == pytest.approx(2 / 3)
    assert probs["lte"] == pytest.approx(1 / 3)


class TestDistributionSpecValidation:
    def test_version_defaults(self) -> None:
        assert DistributionSpec().version == DISTRIBUTION_SPEC_VERSION

    def test_duplicate_thresholds_rejected(self) -> None:
        spec = ThresholdSpec(metric="horizon_return", operator="gt", threshold=0.02)
        with pytest.raises(ValidationError, match="duplicate"):
            DistributionSpec(thresholds=(spec, spec))

    def test_duplicate_quantiles_rejected(self) -> None:
        spec = QuantileSpec(metric="horizon_return", quantile=0.5)
        with pytest.raises(ValidationError, match="duplicate"):
            DistributionSpec(quantiles=(spec, spec))


def test_duplicate_feature_name_explicit_failure() -> None:
    raw = RawSampleSet(
        symbol="600000",
        market_date=MD,
        knowledge_cutoff=CUTOFF,
        horizon=1,
        future_sessions=(date(2026, 9, 28),),
        feature_names=("close", "close", "open"),
        values=np.full((1, 1, 3), 10.0),
    )
    with pytest.raises(ModelInferenceError, match="duplicate feature name"):
        forecast_samples_from_raw(raw)


def test_nonpositive_close_rejected() -> None:
    raw = raw_set([[100.0, 0.0]])
    with pytest.raises(ModelInferenceError, match="positive"):
        build_distribution(raw, origin_close=100.0)


class TestForecastSamplesFromRaw:
    def test_points_and_timestamps(self) -> None:
        raw = raw_set([[100.0, 110.0], [95.0, 90.0]])
        samples = forecast_samples_from_raw(raw)

        assert len(samples) == 2
        assert [s.sample_id for s in samples] == [0, 1]
        first = samples[0]
        assert isinstance(first, ForecastSample)
        assert len(first.points) == 2
        expected_ts = [
            datetime.combine(day, MARKET_SESSION_CLOSE, tzinfo=CN_TZ) for day in SESSIONS
        ]
        assert [p.timestamp for p in first.points] == expected_ts
        assert first.points[0].close == pytest.approx(100.0)
        assert first.points[1].close == pytest.approx(110.0)
        assert first.points[0].volume == pytest.approx(1000.0)
        assert first.points[0].amount == pytest.approx(10_000.0)

    def test_missing_close_feature_explicit_failure(self) -> None:
        raw = raw_set([[100.0, 110.0]])
        shallow = RawSampleSet(
            symbol=raw.symbol,
            market_date=raw.market_date,
            knowledge_cutoff=raw.knowledge_cutoff,
            horizon=raw.horizon,
            future_sessions=raw.future_sessions,
            feature_names=("open", "high", "low", "volume"),
            values=np.zeros((1, 2, 4), dtype=np.float64),
        )
        with pytest.raises(ModelInferenceError, match="lacks required features"):
            forecast_samples_from_raw(shallow)


class TestSamplerGenerateSamples:
    def test_generate_samples_matches_calendar(
        self, tiny_runtime_factory: object, session_calendar: object, make_history: object
    ) -> None:
        runtime = tiny_runtime_factory(lookback_bars=8, max_context=512)  # type: ignore[operator]
        sampler = KronosSampler(runtime, calendar=session_calendar)  # type: ignore[arg-type]
        history = make_history()  # type: ignore[operator]
        request = ForecastRequest(
            symbol="600000",
            market_date=MD,
            knowledge_cutoff=CUTOFF,
            horizon=3,
            sampling=SamplingConfig(seed=5, sample_count=4),
        )

        samples = sampler.generate_samples(history, request)

        assert len(samples) == 4
        expected_ts = [
            datetime.combine(day, MARKET_SESSION_CLOSE, tzinfo=CN_TZ)
            for day in session_calendar.next_sessions(MD, 3)
        ]
        for s in samples:
            assert len(s.points) == 3
            assert [p.timestamp for p in s.points] == expected_ts
