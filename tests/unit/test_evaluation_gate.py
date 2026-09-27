"""Forecast Backend Go / Replace Gate 的单测（RX-KAI-020，§19 / §42）。

被测的是**判决规则**，而不是任何真实模型的表现：所有证据都是合成记录，因此每个案例都
可以手算。分四组：

1. ``TestGateCriteria``：判据形态（§42 的预注册）与它的 hash 稳定性；
2. ``TestSliceEvidence`` / ``TestGateVerdict``：证据与判决的**不变量**（判决必须是证据的
   函数，不能手工拼装）；
3. ``TestEvaluateGate``：GO / CONDITIONAL / REPLACE 三类判决、逐 regime 分组、compute
   护栏、最强 baseline 的选择、配对口径（非 LABELED / 缺一边 / 重复记录）；
4. ``TestGateArtifacts``：``gate.json`` 与 report.md 段落的内容。

证据不足（配对样本低于预注册门槛）必须抛 :class:`InsufficientEvidenceError` 而不是给出
REPLACE——「没测出来」不是「更差」。
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from kronos_ai.domain.forecast import SamplingConfig
from kronos_ai.errors import ConfigurationError, InsufficientEvidenceError
from kronos_ai.evaluation.benchmark import ForecastBenchmarkResult, ForecastBenchmarkSpec
from kronos_ai.evaluation.forecast_metrics import (
    DEFAULT_COVERAGE_SPEC,
    ForecastEvalRecord,
    coverage_spec_hash,
    evaluate_forecast_records,
)
from kronos_ai.evaluation.gate import (
    CI_METHOD,
    CI_QUANTILE_METHOD,
    GATE_CRITERIA_VERSION,
    GATE_METRICS,
    GATE_VERSION,
    GateCriteria,
    GateSliceEvidence,
    GateVerdict,
    evaluate_gate,
    gate_payload,
    load_gate_criteria,
    render_gate_markdown,
    summarise_gate,
)
from kronos_ai.evaluation.regimes import RegimeSpec, regime_spec_hash

CANDIDATE = "kronos"
BASELINE = "last_value"
SYMBOL = "600000"
REGIME = RegimeSpec(trend_window_sessions=5, volatility_window_sessions=5)
SAMPLING = SamplingConfig(seed=11, sample_count=4)
LABEL_POLICY_VERSION = "label-policy-v1"

TREND_GROUPS = ("BEAR", "SIDEWAYS", "BULL")


def session(index: int) -> date:
    return date(2026, 1, 5) + timedelta(days=index)


def criteria(**overrides: object) -> GateCriteria:
    """一份合法的判据；用例只覆盖自己关心的字段（其余保持最宽松的合法取值）。"""
    payload: dict[str, object] = {
        "version": GATE_CRITERIA_VERSION,
        "candidate": CANDIDATE,
        "metric": "direction_accuracy",
        "baselines": (BASELINE,),
        "grouping_axis": "trend",
        "evidence_segments": ("test",),
        "min_groups_with_positive_increment": 3,
        "min_paired_samples_per_group": 4,
        "min_paired_samples_overall": 12,
        "confidence_level": 0.95,
        "bootstrap_iterations": 200,
        "bootstrap_seed": 7,
        "max_seconds_per_origin": 1.0,
    }
    payload.update(overrides)
    return GateCriteria(**payload)  # type: ignore[arg-type]


def record(
    *,
    backend: str,
    market_date: date,
    correct: bool,
    trend: str = "BULL",
    volatility: str = "LOW_VOL",
    label_status: str = "LABELED",
    latency_ms: float = 1.0,
    predicted_return: float = 0.01,
    realized_return: float | None = None,
    symbol: str = SYMBOL,
    segment: str = "test",
) -> ForecastEvalRecord:
    """一条评估记录；``correct`` 决定预测方向是否与实现方向一致（gate 的 direction 口径）。"""
    if label_status == "LABELED":
        realized_direction = "BULLISH" if correct else "BEARISH"
        return ForecastEvalRecord(
            segment=segment,  # type: ignore[arg-type]
            symbol=symbol,
            market_date=market_date,
            backend=backend,
            model_revision="test-rev-1",
            artifact_id="test-artifact",
            predicted_return=predicted_return,
            median_return=predicted_return,
            predicted_direction="BULLISH",
            label_status="LABELED",
            realized_return=0.02 if realized_return is None else realized_return,
            realized_direction=realized_direction,
            trend_regime=trend,  # type: ignore[arg-type]
            volatility_regime=volatility,  # type: ignore[arg-type]
            latency_ms=latency_ms,
        )
    return ForecastEvalRecord(
        segment=segment,  # type: ignore[arg-type]
        symbol=symbol,
        market_date=market_date,
        backend=backend,
        model_revision="test-rev-1",
        artifact_id="test-artifact",
        predicted_return=predicted_return,
        median_return=predicted_return,
        predicted_direction="BULLISH",
        label_status=label_status,  # type: ignore[arg-type]
        trend_regime=trend,  # type: ignore[arg-type]
        volatility_regime=volatility,  # type: ignore[arg-type]
        latency_ms=latency_ms,
    )


def make_result(
    records: list[ForecastEvalRecord],
    *,
    backends: tuple[str, ...] = (CANDIDATE, BASELINE),
) -> ForecastBenchmarkResult:
    """把记录装成一次 run 的结果（指标走真实聚合路径，不手写指标）。"""
    ordered = tuple(records)
    origins = {(item.symbol, item.market_date) for item in ordered}
    coverage_end = max(item.market_date for item in ordered)
    return ForecastBenchmarkResult(
        dataset_hash="a" * 64,
        label_policy_version=LABEL_POLICY_VERSION,
        lookback_bars=6,
        horizon_sessions=5,
        data_coverage_end=coverage_end,
        spec=ForecastBenchmarkSpec(backends=backends, regime=REGIME),
        sampling=SAMPLING,
        records=ordered,
        metrics=evaluate_forecast_records(ordered),
        considered_origins=len(origins),
        evaluated_origins=len(origins),
        truncated=False,
        regime_spec_hash=regime_spec_hash(REGIME),
        coverage_spec_hash=coverage_spec_hash(DEFAULT_COVERAGE_SPEC),
    )


def direction_result(
    per_group: dict[str, int],
    *,
    candidate_correct: dict[str, bool],
    baseline_correct: bool | dict[str, bool] = False,
    latency_ms: float = 1.0,
    baseline_latency_ms: float = 1.0,
) -> ForecastBenchmarkResult:
    """每个 regime 分组各造 ``per_group[group]`` 个 origin。

    ``candidate_correct[group]`` / ``baseline_correct[group]`` 决定两侧在该分组是否全部答对
    （默认 baseline 全错），因此「候选更好」的增量可以逐组手算。
    """
    records: list[ForecastEvalRecord] = []
    index = 0
    for group, count in per_group.items():
        baseline_group_correct = (
            baseline_correct if isinstance(baseline_correct, bool) else baseline_correct[group]
        )
        for _ in range(count):
            day = session(index)
            index += 1
            records.append(
                record(
                    backend=CANDIDATE,
                    market_date=day,
                    correct=candidate_correct[group],
                    trend=group,
                    latency_ms=latency_ms,
                )
            )
            records.append(
                record(
                    backend=BASELINE,
                    market_date=day,
                    correct=baseline_group_correct,
                    trend=group,
                    latency_ms=baseline_latency_ms,
                )
            )
    return make_result(records)


def revalidate(base: GateVerdict, **overrides: object) -> GateVerdict:
    """改字段后**重新校验**（``model_copy`` 不跑校验，因此不能用来测不变量）。"""
    payload = base.model_dump()
    payload.update(overrides)
    return GateVerdict.model_validate(payload)


class TestGateCriteria:
    def test_metric_whitelist_has_a_direction_for_every_entry(self) -> None:
        from kronos_ai.evaluation.gate import GATE_METRIC_DIRECTIONS

        assert set(GATE_METRICS) == set(GATE_METRIC_DIRECTIONS)
        assert GATE_METRIC_DIRECTIONS["direction_accuracy"] is True
        assert GATE_METRIC_DIRECTIONS["mae"] is False

    def test_metric_without_monotone_direction_is_rejected(self) -> None:
        """return_correlation / quantile_coverage 不进 gate：方向不唯一的指标不可证伪。"""
        for metric in ("return_correlation", "quantile_coverage", "sharpe"):
            with pytest.raises(ValidationError, match="not gate-eligible"):
                criteria(metric=metric)

    def test_candidate_cannot_be_its_own_baseline(self) -> None:
        with pytest.raises(ValidationError, match="must not contain the candidate"):
            criteria(baselines=(CANDIDATE, BASELINE))

    def test_empty_baselines_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="baselines must not be empty"):
            criteria(baselines=())

    def test_unsatisfiable_group_requirement_is_rejected(self) -> None:
        """trend 轴只有 3 个取值；要求 4 个正增量分组等于把结论预定成 REPLACE。"""
        with pytest.raises(ValidationError, match="positive trend groups"):
            criteria(min_groups_with_positive_increment=4)
        # 同一份要求放到 volatility（2 个取值）上同样不成立
        with pytest.raises(ValidationError, match="positive volatility groups"):
            criteria(grouping_axis="volatility", min_groups_with_positive_increment=3)

    def test_run_level_minimum_must_not_be_weaker_than_group_minimum(self) -> None:
        with pytest.raises(ValidationError, match="must be >= min_paired_samples_per_group"):
            criteria(min_paired_samples_per_group=10, min_paired_samples_overall=5)

    def test_unknown_version_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unknown gate criteria version"):
            criteria(version="forecast-gate-criteria-v2")

    def test_evidence_segments_are_mandatory(self) -> None:
        """「哪些 origin 算证据」是判据形态的一部分：缺了它根本构造不出判据。"""
        payload = criteria().model_dump()
        del payload["evidence_segments"]
        with pytest.raises(ValidationError, match="Field required"):
            GateCriteria.model_validate(payload)

    def test_empty_or_unknown_evidence_segments_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="evidence_segments must not be empty"):
            criteria(evidence_segments=())
        with pytest.raises(ValidationError, match="unknown segment names"):
            criteria(evidence_segments=("holdout",))

    def test_evidence_segments_are_canonicalised(self) -> None:
        """书写顺序不换 hash：判据身份只看「哪几段」，不看排列。"""
        shuffled = criteria(evidence_segments=("test", "train"))
        canonical = criteria(evidence_segments=("train", "test"))
        assert shuffled.evidence_segments == ("train", "test")
        assert shuffled.criteria_hash == canonical.criteria_hash

    def test_hash_ignores_key_order_and_comments(self, tmp_path: Path) -> None:
        body = """
version: forecast-gate-criteria-v1
candidate: kronos
metric: direction_accuracy
baselines: [last_value]
grouping_axis: trend
evidence_segments: [test]
min_groups_with_positive_increment: 3
min_paired_samples_per_group: 4
min_paired_samples_overall: 12
confidence_level: 0.95
bootstrap_iterations: 200
bootstrap_seed: 7
max_seconds_per_origin: 1.0
"""
        first = tmp_path / "a.yaml"
        first.write_text(body, encoding="utf-8")
        reordered = body.replace(
            "candidate: kronos\nmetric: direction_accuracy\n",
            "# 判据理由写在这里也不影响 hash\nmetric: direction_accuracy\ncandidate: kronos\n",
        )
        second = tmp_path / "b.yaml"
        second.write_text(reordered, encoding="utf-8")

        left, _ = load_gate_criteria(first)
        right, _ = load_gate_criteria(second)
        assert left.criteria_hash == right.criteria_hash

    def test_hash_changes_with_any_field(self) -> None:
        """事后放宽阈值必须换掉 criteria_hash：这是「预注册」可检查的形式。"""
        base = criteria()
        relaxed = criteria(max_seconds_per_origin=999.0)
        looser_ci = criteria(confidence_level=0.8)
        other_metric = criteria(metric="mae")
        # 换证据切片同样换 critera_hash：事后把 train 池进来也是可见的
        other_slice = criteria(evidence_segments=("train", "test"))
        assert base.criteria_hash != relaxed.criteria_hash
        assert base.criteria_hash != looser_ci.criteria_hash
        assert base.criteria_hash != other_metric.criteria_hash
        assert base.criteria_hash != other_slice.criteria_hash

    def test_missing_file_is_explicit_failure(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="gate criteria not found"):
            load_gate_criteria(tmp_path / "nope.yaml")

    def test_non_mapping_yaml_is_explicit_failure(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- 1\n- 2\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="must be a mapping"):
            load_gate_criteria(path)

    def test_empty_yaml_is_explicit_failure(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("# 只有注释\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="is empty"):
            load_gate_criteria(path)

    def test_broken_yaml_is_explicit_failure(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.yaml"
        path.write_text("candidate: [unclosed\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="not valid YAML"):
            load_gate_criteria(path)


class TestSliceEvidence:
    def test_adequate_flag_is_derived_not_free(self) -> None:
        with pytest.raises(ValidationError, match="adequate_evidence must be exactly"):
            GateSliceEvidence(
                slice="all", required_samples=4, sample_count=5, adequate_evidence=False
            )

    def test_single_sample_carries_no_estimates(self) -> None:
        with pytest.raises(ValidationError, match="every estimate must be None"):
            GateSliceEvidence(
                slice="all",
                required_samples=4,
                sample_count=1,
                candidate_metric=1.0,
                adequate_evidence=False,
            )

    def test_interval_cannot_be_inverted(self) -> None:
        with pytest.raises(ValidationError, match="ci_lower"):
            GateSliceEvidence(
                slice="all",
                required_samples=4,
                sample_count=4,
                candidate_metric=1.0,
                baseline_metric=0.0,
                increment=1.0,
                ci_lower=2.0,
                ci_upper=1.0,
                adequate_evidence=True,
            )

    def test_missing_estimate_on_a_computable_slice_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="an estimate is missing"):
            GateSliceEvidence(
                slice="all",
                required_samples=4,
                sample_count=4,
                candidate_metric=1.0,
                adequate_evidence=True,
            )


class TestBootstrapSeeds:
    def test_each_slice_gets_its_own_derived_seed(self) -> None:
        """区间种子由 ``sha256(seed:slice)`` 派生：切片之间不共用重采样（ADR-024 §2）。

        直接钉住这个派生，而不是透过区间间接观察：所有切片共用一个种子时判决仍会「跑通」，
        但某个分组的区间就依赖别的切片的重采样——那是静默的口径变化（评审实测：把 slice
        从种子里去掉，53 条用例全绿）。
        """
        from kronos_ai.evaluation.gate import _slice_seed

        names = ("all", "BEAR", "SIDEWAYS", "BULL")
        seeds = {name: _slice_seed(20260927, name) for name in names}
        assert len(set(seeds.values())) == len(names)
        assert seeds["all"] == _slice_seed(20260927, "all")  # 确定性
        assert seeds["all"] != _slice_seed(20260928, "all")  # 判据的种子进派生
        assert all(seed >= 0 for seed in seeds.values())


class TestReasonCodes:
    def test_every_reason_code_has_prose_and_no_extra_prose(self) -> None:
        """码表与散文表必须一一对应：少一条就是渲染时的 KeyError。"""
        from kronos_ai.evaluation.gate import _GATE_REASON_CODES, _REASON_PROSE

        assert set(_GATE_REASON_CODES) == set(_REASON_PROSE)

    def test_every_reason_code_is_reachable_by_the_rule(self) -> None:
        """规则必须能产出码表里的每一个码（否则码表里有永远不会出现的死码）。"""
        from kronos_ai.evaluation.gate import _GATE_REASON_CODES, _reason_codes

        reachable: set[str] = set()
        for overall_positive in (True, False):
            for positive_groups in ((), ("BEAR",), ("BEAR", "SIDEWAYS", "BULL")):
                for compute_within_budget in (True, False):
                    reachable.update(
                        _reason_codes(
                            overall_positive=overall_positive,
                            positive_groups=positive_groups,
                            min_groups=3,
                            compute_within_budget=compute_within_budget,
                        )
                    )
        assert reachable == set(_GATE_REASON_CODES)


class TestEvaluateGate:
    def test_go_when_every_group_and_overall_show_positive_increment(self) -> None:
        result = direction_result(
            {"BEAR": 5, "SIDEWAYS": 5, "BULL": 5},
            candidate_correct={"BEAR": True, "SIDEWAYS": True, "BULL": True},
        )
        verdict = evaluate_gate(result, criteria())
        assert verdict.verdict == "GO"
        assert verdict.version == GATE_VERSION
        assert verdict.strongest_baseline == BASELINE
        assert verdict.positive_groups == TREND_GROUPS
        assert verdict.overall.increment == pytest.approx(1.0)
        assert verdict.overall.ci_lower == pytest.approx(1.0)
        assert verdict.compute_within_budget is True
        assert verdict.reason_codes == tuple(sorted(verdict.reason_codes))
        assert "COMPUTE_BUDGET_WITHIN_LIMIT" in verdict.reason_codes
        assert verdict.report_hash == result.report_hash
        assert verdict.criteria_hash == criteria().criteria_hash

    def test_conditional_when_only_some_groups_show_positive_increment(self) -> None:
        result = direction_result(
            {"BEAR": 5, "SIDEWAYS": 5, "BULL": 5},
            candidate_correct={"BEAR": True, "SIDEWAYS": False, "BULL": False},
        )
        verdict = evaluate_gate(result, criteria())
        assert verdict.verdict == "CONDITIONAL"
        assert verdict.positive_groups == ("BEAR",)
        # 整体增量为正（BEAR 组全对、其余全错），但分组数不达门槛
        assert verdict.overall.positive is True
        assert "REGIMES_WITH_POSITIVE_INCREMENT_BELOW_MIN" in verdict.reason_codes

    def test_conditional_when_compute_budget_is_exceeded(self) -> None:
        """§19 把 compute cost 列为判断维度：护栏超限时不允许 GO。"""
        result = direction_result(
            {"BEAR": 5, "SIDEWAYS": 5, "BULL": 5},
            candidate_correct={"BEAR": True, "SIDEWAYS": True, "BULL": True},
            latency_ms=3000.0,
            baseline_latency_ms=30.0,
        )
        verdict = evaluate_gate(result, criteria(max_seconds_per_origin=1.0))
        assert verdict.compute_within_budget is False
        assert verdict.compute_seconds_per_origin == pytest.approx(3.0)
        assert verdict.verdict == "CONDITIONAL"
        assert "COMPUTE_BUDGET_EXCEEDED" in verdict.reason_codes

    def test_replace_when_no_group_shows_positive_increment(self) -> None:
        """候选在每个分组都不如 baseline：无稳定增量 → REPLACE（§42）。"""
        result = direction_result(
            {"BEAR": 5, "SIDEWAYS": 5, "BULL": 5},
            candidate_correct={"BEAR": False, "SIDEWAYS": False, "BULL": False},
            baseline_correct=True,
        )
        verdict = evaluate_gate(result, criteria())
        assert verdict.verdict == "REPLACE"
        assert verdict.positive_groups == ()
        assert verdict.overall.increment == pytest.approx(-1.0)
        assert "NO_REGIME_SHOWS_POSITIVE_INCREMENT" in verdict.reason_codes
        assert sorted(verdict.available_backends) == sorted((BASELINE, CANDIDATE))

    def test_insufficient_overall_evidence_is_not_a_replace(self) -> None:
        result = direction_result(
            {"BEAR": 3, "SIDEWAYS": 3, "BULL": 3},
            candidate_correct={"BEAR": False, "SIDEWAYS": False, "BULL": False},
        )
        # 9 个配对 origin < 12：候选看起来「全错」，但样本量不达门槛
        with pytest.raises(InsufficientEvidenceError, match="below the pre-registered minimum"):
            evaluate_gate(result, criteria())

    def test_insufficient_group_evidence_is_not_a_replace(self) -> None:
        result = direction_result(
            {"BEAR": 2, "SIDEWAYS": 2, "BULL": 2},
            candidate_correct={"BEAR": False, "SIDEWAYS": False, "BULL": False},
        )
        with pytest.raises(InsufficientEvidenceError, match="reach the pre-registered minimum"):
            evaluate_gate(
                result,
                criteria(min_paired_samples_per_group=4, min_paired_samples_overall=6),
            )

    def test_baseline_without_any_metric_is_insufficient_evidence(self) -> None:
        """baseline 一条 LABELED 记录都没有 → 比较集合不完整，不是「候选更好」。"""
        records = [
            record(backend=CANDIDATE, market_date=session(index), correct=True)
            for index in range(5)
        ] + [
            record(
                backend=BASELINE,
                market_date=session(index),
                correct=True,
                label_status="SUSPENDED",
            )
            for index in range(5)
        ]
        with pytest.raises(InsufficientEvidenceError, match="comparison set is incomplete"):
            evaluate_gate(
                make_result(records),
                criteria(min_paired_samples_overall=2, min_paired_samples_per_group=2),
            )

    def test_candidate_outside_the_run_is_a_configuration_error(self) -> None:
        result = direction_result(
            {"BEAR": 5, "SIDEWAYS": 5, "BULL": 5},
            candidate_correct={"BEAR": True, "SIDEWAYS": True, "BULL": True},
        )
        with pytest.raises(ConfigurationError, match="is not among the run's backends"):
            evaluate_gate(result, criteria(candidate="chronos"))

    def test_baseline_outside_the_run_is_a_configuration_error(self) -> None:
        result = direction_result(
            {"BEAR": 5, "SIDEWAYS": 5, "BULL": 5},
            candidate_correct={"BEAR": True, "SIDEWAYS": True, "BULL": True},
        )
        with pytest.raises(ConfigurationError, match="are not among the run's backends"):
            evaluate_gate(result, criteria(baselines=("drift",)))

    def test_duplicate_records_for_one_origin_are_rejected(self) -> None:
        """同一 backend 在同一 origin 有两条记录时配对证据没有唯一定义。

        重复检查**跨全部段**做：这里两条候选记录分别在 `train` / `test` 段，而判据只看
        `test` 段——段是互斥的时间切分，同一个 origin 出现在两段里本身就是记录集合的错，
        不能靠「判决只数某一段」把它藏起来。
        """
        records = [
            record(backend=CANDIDATE, market_date=session(0), correct=True, segment="train"),
            record(backend=CANDIDATE, market_date=session(0), correct=True, segment="test"),
            record(backend=BASELINE, market_date=session(0), correct=False, segment="train"),
        ]
        with pytest.raises(ConfigurationError, match="more than one record"):
            evaluate_gate(
                make_result(records),
                criteria(min_paired_samples_overall=2, min_paired_samples_per_group=2),
            )

    def test_unlabeled_and_unpaired_origins_do_not_enter_the_pairs(self) -> None:
        """非 LABELED 与缺一边的 origin 不进配对；配对数量如实写进证据。"""
        records: list[ForecastEvalRecord] = []
        # session 0..3：两侧都 LABELED → 4 个配对
        for index in range(4):
            day = session(index)
            records.append(record(backend=CANDIDATE, market_date=day, correct=True))
            records.append(record(backend=BASELINE, market_date=day, correct=False))
        # session 4：候选没有 label（数据末端）→ 无法配对
        records.append(
            record(
                backend=CANDIDATE,
                market_date=session(4),
                correct=True,
                label_status="INSUFFICIENT_FUTURE_BARS",
            )
        )
        records.append(record(backend=BASELINE, market_date=session(4), correct=False))
        # session 5：baseline 停牌 → 无法配对
        records.append(record(backend=CANDIDATE, market_date=session(5), correct=True))
        records.append(
            record(
                backend=BASELINE,
                market_date=session(5),
                correct=False,
                label_status="SUSPENDED",
            )
        )
        result = make_result(records)
        verdict = evaluate_gate(
            result,
            criteria(
                min_paired_samples_overall=4,
                min_paired_samples_per_group=2,
                min_groups_with_positive_increment=1,
            ),
        )
        assert verdict.overall.sample_count == 4
        assert verdict.overall.increment == pytest.approx(1.0)

    def test_strongest_baseline_is_chosen_by_the_same_metric(self) -> None:
        """mae 是 lower-is-better：最强的 baseline 是 mae 最小的那个。"""
        records: list[ForecastEvalRecord] = []
        for index in range(6):
            day = session(index)
            records.append(record(backend=CANDIDATE, market_date=day, correct=True))
            # last_value 误差 0.02；drift 误差 0.01（更准）
            records.append(
                record(
                    backend=BASELINE,
                    market_date=day,
                    correct=True,
                    predicted_return=0.02,
                    realized_return=0.0,
                )
            )
            records.append(
                record(
                    backend="drift",
                    market_date=day,
                    correct=True,
                    predicted_return=0.01,
                    realized_return=0.0,
                )
            )
        result = make_result(records, backends=(CANDIDATE, BASELINE, "drift"))
        # 候选的 mae 也是 0.01（predicted 0.01 / realized 0.02），与 drift 并列
        verdict = evaluate_gate(
            result,
            criteria(
                metric="mae",
                baselines=(BASELINE, "drift"),
                min_paired_samples_overall=6,
                min_paired_samples_per_group=2,
                min_groups_with_positive_increment=1,
            ),
        )
        # 并列时按名字取字典序最小者，规则必须是确定的
        assert verdict.strongest_baseline == "drift"

    def test_mae_increment_is_oriented_so_positive_means_better(self) -> None:
        """mae 越小越好，因此增量 = baseline_mae − candidate_mae。"""
        records: list[ForecastEvalRecord] = []
        for index in range(6):
            day = session(index)
            records.append(
                record(
                    backend=CANDIDATE,
                    market_date=day,
                    correct=True,
                    predicted_return=0.0,
                    realized_return=0.0,
                )
            )
            records.append(
                record(
                    backend=BASELINE,
                    market_date=day,
                    correct=True,
                    predicted_return=0.10,
                    realized_return=0.0,
                )
            )
        result = make_result(records)
        verdict = evaluate_gate(
            result,
            criteria(
                metric="mae",
                min_paired_samples_overall=6,
                min_paired_samples_per_group=2,
                min_groups_with_positive_increment=1,
            ),
        )
        assert verdict.higher_is_better is False
        assert verdict.overall.candidate_metric == pytest.approx(0.0)
        assert verdict.overall.baseline_metric == pytest.approx(0.10)
        assert verdict.overall.increment == pytest.approx(0.10)
        assert verdict.verdict == "GO"

    def test_rmse_increment_is_computed_after_aggregation(self) -> None:
        """rmse 的逐记录取值是平方误差，增量必须是 rmse 之差（不是 mse 之差）。"""
        records: list[ForecastEvalRecord] = []
        candidate_errors = (1.0, 3.0)
        baseline_errors = (2.0, 2.0)
        for index in range(4):
            day = session(index)
            group = TREND_GROUPS[0] if index < 2 else TREND_GROUPS[1]
            records.append(
                record(
                    backend=CANDIDATE,
                    market_date=day,
                    correct=True,
                    predicted_return=0.0,
                    realized_return=candidate_errors[index % 2],
                    trend=group,
                )
            )
            records.append(
                record(
                    backend=BASELINE,
                    market_date=day,
                    correct=True,
                    predicted_return=0.0,
                    realized_return=baseline_errors[index % 2],
                    trend=group,
                )
            )
        result = make_result(records)
        verdict = evaluate_gate(
            result,
            criteria(
                metric="rmse",
                min_paired_samples_overall=4,
                min_paired_samples_per_group=2,
                min_groups_with_positive_increment=1,
            ),
        )
        assert verdict.overall.candidate_metric == pytest.approx(math.sqrt(5.0))
        assert verdict.overall.baseline_metric == pytest.approx(2.0)
        assert verdict.overall.increment == pytest.approx(2.0 - math.sqrt(5.0))

    def test_verdict_is_reproducible_and_bound_to_the_criteria(self) -> None:
        result = direction_result(
            {"BEAR": 5, "SIDEWAYS": 5, "BULL": 5},
            candidate_correct={"BEAR": True, "SIDEWAYS": True, "BULL": True},
        )
        first = evaluate_gate(result, criteria())
        second = evaluate_gate(result, criteria())
        assert first.gate_hash == second.gate_hash
        # 换一份判据（例如放宽阈值）→ 判决身份随之变化，事后替换判据不可能悄无声息
        relaxed = evaluate_gate(result, criteria(max_seconds_per_origin=2.0))
        assert relaxed.criteria_hash != first.criteria_hash
        assert relaxed.gate_hash != first.gate_hash
        # bootstrap 种子不同 → 区间可能不同，因此判决身份也不同
        reseeded = evaluate_gate(result, criteria(bootstrap_seed=99))
        assert reseeded.gate_hash != first.gate_hash

    def test_group_evidence_covers_every_axis_value(self) -> None:
        """没有任何记录的分组也要出现在证据里（缺证据可见，而不是被静默丢弃）。"""
        result = direction_result(
            {"BEAR": 5, "SIDEWAYS": 5},
            candidate_correct={"BEAR": True, "SIDEWAYS": True},
        )
        verdict = evaluate_gate(
            result,
            criteria(min_paired_samples_overall=10, min_groups_with_positive_increment=2),
        )
        assert [group.slice for group in verdict.groups] == list(TREND_GROUPS)
        missing = next(group for group in verdict.groups if group.slice == "BULL")
        assert missing.sample_count == 0
        assert missing.increment is None
        assert missing.adequate_evidence is False

    def test_only_the_pre_registered_evidence_segment_counts(self) -> None:
        """只看 test 段与把 train 池进来是两个不同的判决（RX-KAI-020 评审 M1）。

        同一个 run、同一份判据形态，唯一区别是 ``evidence_segments``：train 段上候选全错、
        test 段上候选全对，因此池化会给出与只看 test 相反的结论。这正是它必须在 run 前冻结
        的原因——否则「判决数的是哪些 origin」就成了实现细节。
        """
        records: list[ForecastEvalRecord] = []
        for index in range(8):  # train：候选全错、baseline 全对
            day = session(index)
            records.append(
                record(backend=CANDIDATE, market_date=day, correct=False, segment="train")
            )
            records.append(record(backend=BASELINE, market_date=day, correct=True, segment="train"))
        for index in range(8, 12):  # test：候选全对、baseline 全错
            day = session(index)
            records.append(record(backend=CANDIDATE, market_date=day, correct=True, segment="test"))
            records.append(record(backend=BASELINE, market_date=day, correct=False, segment="test"))
        result = make_result(records)

        test_only = evaluate_gate(
            result,
            criteria(
                evidence_segments=("test",),
                min_paired_samples_overall=4,
                min_paired_samples_per_group=2,
                min_groups_with_positive_increment=1,
            ),
        )
        assert test_only.evidence_segments == ("test",)
        assert test_only.overall.sample_count == 4
        assert test_only.overall.increment == pytest.approx(1.0)
        assert test_only.verdict == "GO"

        pooled = evaluate_gate(
            result,
            criteria(
                evidence_segments=("train", "test"),
                min_paired_samples_overall=4,
                min_paired_samples_per_group=2,
                min_groups_with_positive_increment=1,
            ),
        )
        assert pooled.overall.sample_count == 12
        # 4/12 正确 vs 8/12 正确
        assert pooled.overall.increment == pytest.approx(-1.0 / 3.0)
        assert pooled.verdict == "REPLACE"

    def test_criteria_segment_without_records_is_a_configuration_error(self) -> None:
        """判据声明的段在本次 run 里没有记录 → 配置与 run 不匹配，不给出更弱的判决。

        ``validation`` 是合法的段名，但本次 run 没有这一段：这属于「判据说的是另一批
        origin」，必须在选比较对象之前就失败（而不是退化成 InsufficientEvidenceError）。
        """
        result = direction_result(
            {"BEAR": 5, "SIDEWAYS": 5, "BULL": 5},
            candidate_correct={"BEAR": True, "SIDEWAYS": True, "BULL": True},
        )
        with pytest.raises(ConfigurationError, match="have no records in this run"):
            evaluate_gate(result, criteria(evidence_segments=("validation",)))

    def test_strongest_baseline_is_chosen_on_the_evidence_slice(self) -> None:
        """「最强」在**判决用的那批 origin**上选出：换切片会换比较对象（也换判决）。

        last_value 在 train 上更准、在 test 上更差，drift 相反；30 比 4 的样本量让池化口径
        选出 last_value，而只看 test 段必须选 drift。
        """
        records: list[ForecastEvalRecord] = []
        for index in range(20):
            day = session(index)
            records.append(
                record(
                    backend=CANDIDATE,
                    market_date=day,
                    correct=True,
                    predicted_return=0.0,
                    realized_return=0.0,
                    segment="train",
                )
            )
            records.append(
                record(
                    backend=BASELINE,
                    market_date=day,
                    correct=True,
                    predicted_return=0.001,
                    realized_return=0.0,
                    segment="train",
                )
            )
            records.append(
                record(
                    backend="drift",
                    market_date=day,
                    correct=True,
                    predicted_return=0.5,
                    realized_return=0.0,
                    segment="train",
                )
            )
        for index in range(20, 24):
            day = session(index)
            records.append(
                record(
                    backend=CANDIDATE,
                    market_date=day,
                    correct=True,
                    predicted_return=0.0,
                    realized_return=0.0,
                    segment="test",
                )
            )
            records.append(
                record(
                    backend=BASELINE,
                    market_date=day,
                    correct=True,
                    predicted_return=0.5,
                    realized_return=0.0,
                    segment="test",
                )
            )
            records.append(
                record(
                    backend="drift",
                    market_date=day,
                    correct=True,
                    predicted_return=0.001,
                    realized_return=0.0,
                    segment="test",
                )
            )
        result = make_result(records, backends=(CANDIDATE, BASELINE, "drift"))

        test_only = evaluate_gate(
            result,
            criteria(
                metric="mae",
                baselines=(BASELINE, "drift"),
                evidence_segments=("test",),
                min_paired_samples_overall=4,
                min_paired_samples_per_group=2,
                min_groups_with_positive_increment=1,
            ),
        )
        assert test_only.strongest_baseline == "drift"

        pooled = evaluate_gate(
            result,
            criteria(
                metric="mae",
                baselines=(BASELINE, "drift"),
                evidence_segments=("train", "test"),
                min_paired_samples_overall=4,
                min_paired_samples_per_group=2,
                min_groups_with_positive_increment=1,
            ),
        )
        assert pooled.strongest_baseline == "last_value"

    def test_baseline_without_evidence_in_the_slice_is_insufficient(self) -> None:
        """比较集合里有一个在证据切片上没有记录 → 「最强」没有定义，不判决。"""
        records: list[ForecastEvalRecord] = []
        for index in range(6):
            day = session(index)
            records.append(record(backend=CANDIDATE, market_date=day, correct=True))
            records.append(record(backend=BASELINE, market_date=day, correct=False))
            records.append(record(backend="drift", market_date=day, correct=False, segment="train"))
        result = make_result(records, backends=(CANDIDATE, BASELINE, "drift"))
        with pytest.raises(InsufficientEvidenceError, match="comparison set is incomplete"):
            evaluate_gate(
                result,
                criteria(
                    baselines=(BASELINE, "drift"),
                    min_paired_samples_overall=2,
                    min_paired_samples_per_group=2,
                    min_groups_with_positive_increment=1,
                ),
            )


class TestGateVerdict:
    def test_positive_groups_must_be_derived_from_the_evidence(self) -> None:
        result = direction_result(
            {"BEAR": 5, "SIDEWAYS": 5, "BULL": 5},
            candidate_correct={"BEAR": True, "SIDEWAYS": True, "BULL": True},
        )
        verdict = evaluate_gate(result, criteria())
        with pytest.raises(ValidationError, match="must be exactly the adequate"):
            revalidate(verdict, positive_groups=("BEAR",))

    def test_verdict_must_follow_the_evidence(self) -> None:
        """跑完之后手改判决字段（REPLACE → GO）在构造期就失败。"""
        result = direction_result(
            {"BEAR": 5, "SIDEWAYS": 5, "BULL": 5},
            candidate_correct={"BEAR": False, "SIDEWAYS": False, "BULL": False},
        )
        verdict = evaluate_gate(result, criteria())
        assert verdict.verdict == "REPLACE"
        with pytest.raises(ValidationError, match="does not follow from the evidence"):
            revalidate(verdict, verdict="GO")

    def test_compute_flag_must_match_the_observed_value(self) -> None:
        result = direction_result(
            {"BEAR": 5, "SIDEWAYS": 5, "BULL": 5},
            candidate_correct={"BEAR": True, "SIDEWAYS": True, "BULL": True},
        )
        verdict = evaluate_gate(result, criteria())
        with pytest.raises(ValidationError, match="compute_within_budget must be exactly"):
            revalidate(verdict, compute_within_budget=False)

    def test_override_of_min_groups_is_ignored_by_the_rule(self) -> None:
        """门槛是判决自带的字段：改成「1 个分组就够」时 GO 仍然成立，改大则不再成立。"""
        result = direction_result(
            {"BEAR": 5, "SIDEWAYS": 5, "BULL": 5},
            candidate_correct={"BEAR": True, "SIDEWAYS": False, "BULL": False},
        )
        verdict = evaluate_gate(result, criteria(min_groups_with_positive_increment=1))
        assert verdict.verdict == "GO"
        with pytest.raises(ValidationError, match="does not follow from the evidence"):
            revalidate(verdict, min_groups_with_positive_increment=3)

    def test_unknown_reason_code_is_rejected(self) -> None:
        result = direction_result(
            {"BEAR": 5, "SIDEWAYS": 5, "BULL": 5},
            candidate_correct={"BEAR": True, "SIDEWAYS": True, "BULL": True},
        )
        verdict = evaluate_gate(result, criteria())
        with pytest.raises(ValidationError, match="unknown gate reason codes"):
            revalidate(verdict, reason_codes=("LOOKS_GOOD",))

    def test_judged_and_comparison_backends_must_be_in_the_run(self) -> None:
        """候选与比较对象都必须是本次 run 真的跑过的 backend，且比较对象不能是候选自己。"""
        result = direction_result(
            {"BEAR": 5, "SIDEWAYS": 5, "BULL": 5},
            candidate_correct={"BEAR": True, "SIDEWAYS": True, "BULL": True},
        )
        verdict = evaluate_gate(result, criteria())
        with pytest.raises(ValidationError, match="a backend that did not run cannot be judged"):
            revalidate(verdict, candidate="chronos")
        with pytest.raises(ValidationError, match="must not be the candidate itself"):
            revalidate(verdict, strongest_baseline=CANDIDATE)
        with pytest.raises(ValidationError, match="not among available_backends"):
            revalidate(verdict, strongest_baseline="chronos")

    def test_groups_must_cover_the_axis_they_claim(self) -> None:
        """分组名必须恰好是该分组轴的取值（换名字或换轴都拒绝），且顺序是规范序。"""
        result = direction_result(
            {"BEAR": 5, "SIDEWAYS": 5, "BULL": 5},
            candidate_correct={"BEAR": False, "SIDEWAYS": False, "BULL": False},
        )
        verdict = evaluate_gate(result, criteria())
        renamed = tuple(
            group.model_copy(update={"slice": name})
            for group, name in zip(verdict.groups, ("BEAR", "SIDEWAYS", "TMT"), strict=True)
        )
        with pytest.raises(ValidationError, match="must be exactly the trend axis values"):
            revalidate(verdict, groups=renamed)
        with pytest.raises(ValidationError, match="must be exactly the volatility axis values"):
            revalidate(verdict, grouping_axis="volatility")
        reordered = (verdict.groups[2], verdict.groups[0], verdict.groups[1])
        with pytest.raises(ValidationError, match="canonical order"):
            revalidate(verdict, groups=reordered)

    def test_evidence_segments_must_be_known_and_canonical(self) -> None:
        result = direction_result(
            {"BEAR": 5, "SIDEWAYS": 5, "BULL": 5},
            candidate_correct={"BEAR": True, "SIDEWAYS": True, "BULL": True},
        )
        verdict = evaluate_gate(result, criteria())
        with pytest.raises(ValidationError, match="must not be empty"):
            revalidate(verdict, evidence_segments=())
        with pytest.raises(ValidationError, match="unknown segments in evidence_segments"):
            revalidate(verdict, evidence_segments=("holdout",))
        with pytest.raises(ValidationError, match="canonical segment order"):
            revalidate(verdict, evidence_segments=("test", "train"))

    def test_hashes_must_be_sha256(self) -> None:
        result = direction_result(
            {"BEAR": 5, "SIDEWAYS": 5, "BULL": 5},
            candidate_correct={"BEAR": True, "SIDEWAYS": True, "BULL": True},
        )
        verdict = evaluate_gate(result, criteria())
        with pytest.raises(ValidationError, match="sha256 hex digest"):
            revalidate(verdict, report_hash="not-a-hash")


class TestGateArtifacts:
    def test_payload_carries_evidence_and_identity(self) -> None:
        result = direction_result(
            {"BEAR": 5, "SIDEWAYS": 5, "BULL": 5},
            candidate_correct={"BEAR": True, "SIDEWAYS": False, "BULL": True},
        )
        verdict = evaluate_gate(result, criteria(min_groups_with_positive_increment=2))
        payload = gate_payload(verdict)
        assert payload["kind"] == "forecast_gate_verdict"
        assert payload["verdict"] == "GO"
        assert payload["criteria_hash"] == verdict.criteria_hash
        assert payload["gate_hash"] == verdict.gate_hash
        assert payload["report_hash"] == result.report_hash
        assert payload["ci_method"] == CI_METHOD
        assert payload["ci_quantile_method"] == CI_QUANTILE_METHOD
        assert payload["positive_groups"] == list(verdict.positive_groups)
        assert [group["slice"] for group in payload["groups"]] == list(TREND_GROUPS)
        assert payload["overall"]["positive"] is True
        assert payload["compute_within_budget"] is True
        # 判决数的是哪些 origin 必须能从产物本身读出来，而不是回到判据文件去猜
        assert payload["evidence_segments"] == ["test"]

    def test_markdown_section_states_the_verdict_and_the_evidence(self) -> None:
        result = direction_result(
            {"BEAR": 5, "SIDEWAYS": 5, "BULL": 5},
            candidate_correct={"BEAR": False, "SIDEWAYS": False, "BULL": False},
        )
        verdict = evaluate_gate(result, criteria())
        lines = render_gate_markdown(verdict)
        text = "\n".join(lines)
        assert verdict.verdict in text
        assert verdict.criteria_hash in text
        assert "NO_REGIME_SHOWS_POSITIVE_INCREMENT" in text
        assert "- evidence_segments: test" in text
        # 没有证据的切片渲染 n/a，而不是 0
        assert "| BEAR |" in text

    def test_summary_line_names_candidate_and_baseline(self) -> None:
        result = direction_result(
            {"BEAR": 5, "SIDEWAYS": 5, "BULL": 5},
            candidate_correct={"BEAR": True, "SIDEWAYS": True, "BULL": True},
        )
        verdict = evaluate_gate(result, criteria())
        summary = summarise_gate(verdict)
        assert "GO" in summary
        assert CANDIDATE in summary and BASELINE in summary

    def test_gate_verdict_is_not_constructible_by_hand_when_inconsistent(self) -> None:
        """手工拼一个「证据全负但判决 GO」的对象必须失败（不变量是最后一道闸）。"""
        evidence = GateSliceEvidence(
            slice="all",
            required_samples=10,
            sample_count=10,
            candidate_metric=0.4,
            baseline_metric=0.6,
            increment=-0.2,
            ci_lower=-0.3,
            ci_upper=-0.1,
            adequate_evidence=True,
        )
        with pytest.raises(ValidationError, match="does not follow from the evidence"):
            GateVerdict(
                criteria_version=GATE_CRITERIA_VERSION,
                criteria_hash="b" * 64,
                report_hash="c" * 64,
                dataset_hash="d" * 64,
                candidate=CANDIDATE,
                strongest_baseline=BASELINE,
                metric="direction_accuracy",
                higher_is_better=True,
                grouping_axis="trend",
                evidence_segments=("test",),
                available_backends=(BASELINE, CANDIDATE),
                confidence_level=0.95,
                bootstrap_iterations=200,
                bootstrap_seed=7,
                min_groups_with_positive_increment=3,
                min_paired_samples_per_group=2,
                min_paired_samples_overall=10,
                overall=evidence,
                groups=(
                    evidence.model_copy(update={"slice": "BEAR"}),
                    evidence.model_copy(update={"slice": "SIDEWAYS"}),
                    evidence.model_copy(update={"slice": "BULL"}),
                ),
                positive_groups=(),
                compute_seconds_per_origin=1.0,
                compute_budget_seconds_per_origin=2.0,
                compute_within_budget=True,
                verdict="GO",
                reason_codes=("NO_REGIME_SHOWS_POSITIVE_INCREMENT",),
            )
