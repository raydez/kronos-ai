"""Forecast Benchmark 指标单元测试（RX-KAI-019，基线文档 §49 / §13 / §41）。

覆盖：

1. **数学正确**：MAE / RMSE / direction accuracy / Pearson 相关 / coverage 都用手算值
   （含 ≥2 个等价写法）钉住；口径变更必须让这些用例失败；
2. **缺料是 ``None`` 不是 0**：无 LABELED 记录、相关样本 < 2、方差为 0、分位端点缺失；
3. **分组**：``overall`` / ``trend`` / ``volatility`` 三轴同时产出，计数一致；跨段汇总
   （``segment=None``）与逐段切片各自独立，前者不等于「任意一段」；
4. **契约**：hashing_payload 覆盖全部字段、version 受版本治理、CoverageSpec 的名义覆盖率。
"""

from __future__ import annotations

import math
from datetime import date

import pytest
from pydantic import ValidationError

from kronos_ai.domain.hashing import is_sha256_hex
from kronos_ai.errors import ConfigurationError
from kronos_ai.evaluation.forecast_metrics import (
    DEFAULT_COVERAGE_SPEC,
    DEFERRED_FORECAST_EVAL_METRICS,
    FORECAST_EVAL_METRIC_NAMES,
    FORECAST_EVAL_METRICS_VERSION,
    CoverageSpec,
    ForecastEvalGroup,
    ForecastEvalMetrics,
    ForecastEvalRecord,
    coverage_spec_hash,
    evaluate_forecast_records,
    forecast_eval_metric_definition,
)

MARKET_DATE = date(2026, 9, 25)


def record(
    *,
    symbol: str = "600000",
    backend: str = "last_value",
    segment: str = "test",
    predicted: float = 0.01,
    median: float | None = None,
    realized: float | None = 0.02,
    status: str = "LABELED",
    trend: str = "SIDEWAYS",
    volatility: str = "LOW_VOL",
    coverage: tuple[float | None, float | None] = (-0.05, 0.05),
    day: date = MARKET_DATE,
) -> ForecastEvalRecord:
    lower, upper = coverage
    return ForecastEvalRecord(
        segment=segment,  # type: ignore[arg-type]  # 用例只使用合法 SegmentName
        symbol=symbol,
        market_date=day,
        backend=backend,
        model_revision="baseline-math-v1",
        artifact_id="a" * 64,
        predicted_return=predicted,
        median_return=predicted if median is None else median,
        predicted_direction=_direction(predicted),
        coverage_lower=lower,
        coverage_upper=upper,
        label_status=status,  # type: ignore[arg-type]  # 用例只使用合法 LabelStatus
        realized_return=realized if status == "LABELED" else None,
        realized_direction=_direction(realized) if status == "LABELED" else None,
        trend_regime=trend,  # type: ignore[arg-type]
        volatility_regime=volatility,  # type: ignore[arg-type]
        latency_ms=1.0,
    )


def _direction(value: float | None) -> str:
    """与 LabelPolicy 默认阈值（±2%）同口径，仅用于构造用例。"""
    if value is None:
        return "NEUTRAL"
    if value >= 0.02:
        return "BULLISH"
    if value <= -0.02:
        return "BEARISH"
    return "NEUTRAL"


def overall(records: list[ForecastEvalRecord]) -> ForecastEvalMetrics:
    metrics = evaluate_forecast_records(records)
    matching = [
        entry
        for entry in metrics
        if entry.backend == "last_value"
        and entry.segment == "test"
        and entry.group.axis == "overall"
    ]
    assert len(matching) == 1
    return matching[0]


class TestPointMetrics:
    def test_mae_and_rmse(self) -> None:
        records = [
            record(predicted=0.01, realized=0.04),
            record(predicted=0.03, realized=0.01),
        ]
        entry = overall(records)
        # |0.01-0.04| = 0.03, |0.03-0.01| = 0.02 -> mean 0.025
        assert entry.mae == pytest.approx(0.025)
        # sqrt((0.03^2 + 0.02^2)/2) = sqrt(0.00065)
        assert entry.rmse == pytest.approx(math.sqrt(0.00065))
        assert entry.labeled_count == 2
        assert entry.sample_count == 2

    def test_direction_accuracy(self) -> None:
        records = [
            record(predicted=0.05, realized=0.06),  # BULLISH vs BULLISH -> hit
            record(predicted=0.05, realized=-0.06),  # BULLISH vs BEARISH -> miss
            record(predicted=0.0, realized=0.0),  # NEUTRAL vs NEUTRAL -> hit
            record(predicted=0.0, realized=0.03),  # NEUTRAL vs BULLISH -> miss
        ]
        entry = overall(records)
        assert entry.direction_accuracy == pytest.approx(0.5)

    def test_return_correlation_perfect(self) -> None:
        records = [
            record(predicted=0.01, realized=0.02),
            record(predicted=0.02, realized=0.04),
            record(predicted=0.03, realized=0.06),
        ]
        entry = overall(records)
        assert entry.return_correlation == pytest.approx(1.0)

    def test_return_correlation_inverse(self) -> None:
        records = [
            record(predicted=0.01, realized=0.06),
            record(predicted=0.02, realized=0.04),
            record(predicted=0.03, realized=0.02),
        ]
        entry = overall(records)
        assert entry.return_correlation == pytest.approx(-1.0)


class TestMissingEvidenceIsNone:
    def test_no_labeled_records(self) -> None:
        records = [record(status="SUSPENDED", realized=None)]
        entry = overall(records)
        assert entry.sample_count == 1
        assert entry.labeled_count == 0
        assert entry.mae is None
        assert entry.rmse is None
        assert entry.direction_accuracy is None
        assert entry.return_correlation is None
        assert entry.quantile_coverage is None

    def test_single_labeled_record_has_no_correlation(self) -> None:
        entry = overall([record(predicted=0.01, realized=0.02)])
        assert entry.mae is not None
        assert entry.return_correlation is None

    def test_zero_variance_has_no_correlation(self) -> None:
        records = [
            record(predicted=0.01, realized=0.02),
            record(predicted=0.01, realized=0.03),
        ]
        assert overall(records).return_correlation is None

    def test_coverage_requires_both_bounds(self) -> None:
        records = [record(coverage=(None, None))]
        entry = overall(records)
        assert entry.quantile_coverage is None
        assert entry.coverage_nominal == pytest.approx(0.9)


class TestCoverage:
    def test_inside_and_outside(self) -> None:
        records = [
            record(realized=0.01, coverage=(-0.02, 0.02)),  # inside
            record(realized=0.05, coverage=(-0.02, 0.02)),  # outside
            record(realized=-0.02, coverage=(-0.02, 0.02)),  # boundary counts as inside
            record(realized=0.03, coverage=(-0.02, 0.03)),  # upper boundary inside
        ]
        assert overall(records).quantile_coverage == pytest.approx(0.75)

    def test_partial_bounds_are_rejected_at_construction(self) -> None:
        with pytest.raises(ValidationError):
            record(coverage=(None, 0.02))


class TestGrouping:
    def test_three_axes_with_consistent_counts(self) -> None:
        records = [
            record(trend="BULL", volatility="HIGH_VOL"),
            record(trend="BULL", volatility="LOW_VOL"),
            record(trend="BEAR", volatility="LOW_VOL"),
        ]
        metrics = evaluate_forecast_records(records)
        per_segment = [entry for entry in metrics if entry.segment == "test"]
        labels = {entry.group.label for entry in per_segment}
        assert labels == {
            "all",
            "trend=BULL",
            "trend=BEAR",
            "volatility=HIGH_VOL",
            "volatility=LOW_VOL",
        }
        by_group = {entry.group.label: entry for entry in per_segment}
        assert by_group["all"].sample_count == 3
        assert by_group["trend=BULL"].sample_count == 2
        assert by_group["trend=BEAR"].sample_count == 1
        assert by_group["volatility=HIGH_VOL"].sample_count == 1
        assert by_group["volatility=LOW_VOL"].sample_count == 2
        # 同一批记录另有一条跨段汇总（segment=None），三轴同样齐备
        run_level = [entry for entry in metrics if entry.segment is None]
        assert {entry.group.label for entry in run_level} == labels
        assert len(metrics) == 2 * len(per_segment)

    def test_run_level_aggregates_across_segments(self) -> None:
        """run 级（``segment=None``）是跨段汇总，不是「随便挑一段」。"""
        records = [
            record(segment="train", predicted=0.01, realized=0.02),
            record(segment="train", predicted=0.02, realized=0.04),
            record(segment="test", predicted=0.03, realized=0.06),
        ]
        metrics = evaluate_forecast_records(records)
        run_level = [
            entry for entry in metrics if entry.segment is None and entry.group.axis == "overall"
        ]
        assert len(run_level) == 1
        assert run_level[0].sample_count == 3
        assert run_level[0].labeled_count == 3
        train = [
            entry for entry in metrics if entry.segment == "train" and entry.group.axis == "overall"
        ]
        assert [entry.sample_count for entry in train] == [2]
        # 产出顺序确定：run 级排在逐段之前（backend → segment → axis）
        assert metrics.index(run_level[0]) < metrics.index(train[0])

    def test_grouping_is_stable_and_multi_backend(self) -> None:
        records = [
            record(backend="drift", symbol="600000"),
            record(backend="last_value", symbol="000001"),
        ]
        first = evaluate_forecast_records(records)
        second = evaluate_forecast_records(records)
        assert [entry.hashing_payload() for entry in first] == [
            entry.hashing_payload() for entry in second
        ]
        assert {entry.backend for entry in first} == {"drift", "last_value"}
        # 不同 backend 不得混进同一个切片
        for entry in first:
            assert entry.sample_count == 1

    def test_group_label_render(self) -> None:
        assert ForecastEvalGroup(axis="overall", value="all").label == "all"
        assert ForecastEvalGroup(axis="trend", value="BULL").label == "trend=BULL"


class TestVersionGovernance:
    def test_metric_registry_lookup(self) -> None:
        assert forecast_eval_metric_definition("mae").unit == "ratio"
        with pytest.raises(ConfigurationError):
            forecast_eval_metric_definition("sharpe")  # §50 明确移除交易层指标

    def test_registry_names_and_deferred(self) -> None:
        assert set(FORECAST_EVAL_METRIC_NAMES) == {
            "mae",
            "rmse",
            "direction_accuracy",
            "return_correlation",
            "quantile_coverage",
        }
        assert DEFERRED_FORECAST_EVAL_METRICS == ("crps",)

    def test_metrics_version_pinned(self) -> None:
        entry = overall([record()])
        assert entry.version == FORECAST_EVAL_METRICS_VERSION
        with pytest.raises(ValidationError):
            ForecastEvalMetrics(
                backend="last_value",
                segment="test",
                group=ForecastEvalGroup(axis="overall", value="all"),
                sample_count=1,
                labeled_count=1,
                version="forecast-eval-metrics-v999",
            )

    def test_hashing_payload_covers_all_fields(self) -> None:
        entry = overall([record()])
        assert set(entry.hashing_payload()) == set(ForecastEvalMetrics.model_fields)
        # latency_ms 是环境事实（墙钟），刻意不进 report hash：否则同一条命令跑两次就会
        # 换 hash 而所有指标逐位相同（见 ForecastEvalRecord.hashing_payload 的 docstring）。
        assert set(record().hashing_payload()) == set(ForecastEvalRecord.model_fields) - {
            "latency_ms"
        }

    def test_coverage_spec(self) -> None:
        assert DEFAULT_COVERAGE_SPEC.nominal_coverage == pytest.approx(0.9)
        assert set(DEFAULT_COVERAGE_SPEC.hashing_payload()) == set(CoverageSpec.model_fields)
        assert is_sha256_hex(coverage_spec_hash(DEFAULT_COVERAGE_SPEC))
        other = CoverageSpec(lower_quantile=0.1, upper_quantile=0.9)
        assert coverage_spec_hash(other) != coverage_spec_hash(DEFAULT_COVERAGE_SPEC)
        with pytest.raises(ValidationError):
            CoverageSpec(lower_quantile=0.9, upper_quantile=0.1)
