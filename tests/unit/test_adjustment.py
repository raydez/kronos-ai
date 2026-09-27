import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from kronos_ai.data.adjustment import (
    ADJUSTMENT_POLICY_VERSION,
    BAOSTOCK_FACTOR_SOURCE,
    DEFAULT_FORECAST_ADJUSTMENT_MODE,
    AdjustFactorRecord,
    AdjustFactorSeries,
    AdjustmentPolicy,
    default_forecast_adjustment_policy,
)

ARTIFACT_PATH = (
    Path(__file__).resolve().parents[2] / "docs" / "spike" / "baostock-capability-raw.json"
)


def spike_artifact() -> dict[str, Any]:
    """spike 落盘的原始证据（能力报告要求每条事实都能从这里读出）。"""
    with ARTIFACT_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


# 真实因子表片段（sh.600000，2026-09-27 经 query_adjust_factor 取得）：
# 逐行可在 docs/spike/baostock-capability-raw.json 的
# checks.adjust_factor.factor_table 中核对（TestFixtureMatchesSpikeArtifact 强制该关联）；
# 末条 foreAdjustFactor = 1.0 证明整表以最新公司行动归一化（能力报告 §2）。
REAL_FACTORS = (
    ("2000-07-06", 0.075297, 1.006502),
    ("2015-06-23", 0.471008, 6.295967),
    ("2020-07-23", 0.771461, 10.312133),
    ("2022-07-21", 0.855421, 11.434431),
    ("2026-07-16", 1.000000, 13.367013),
)

# golden 标的日（artifact: checks.factor_price_mapping.sample_day）：
# 用观测到的 hfq/qfq 收盘反推因子，验证 fixture 与真实复权序列一致。
GOLDEN_DAY = date(2023, 1, 3)
GOLDEN_RAW_CLOSE = 7.23
GOLDEN_HFQ_CLOSE = 82.67093613
GOLDEN_QFQ_CLOSE = 6.18469383
GOLDEN_EX_DATE = date(2022, 7, 21)


def real_series(**overrides: object) -> AdjustFactorSeries:
    records = tuple(
        AdjustFactorRecord(ex_date=date.fromisoformat(day), fore_factor=fore, back_factor=back)
        for day, fore, back in REAL_FACTORS
    )
    fields: dict[str, object] = {
        "symbol": "600000",
        "records": records,
        "source": BAOSTOCK_FACTOR_SOURCE,
        "dataset_version": "baostock-factors-20260926",
    }
    fields.update(overrides)
    return AdjustFactorSeries(**fields)  # type: ignore[arg-type]


class TestAdjustmentPolicy:
    def test_raw_needs_no_factor_source(self) -> None:
        policy = AdjustmentPolicy(mode="raw")
        assert policy.requires_factor_snapshot is False
        assert policy.factor_source is None

    def test_raw_rejects_factor_source(self) -> None:
        with pytest.raises(ValidationError, match="must not declare a factor_source"):
            AdjustmentPolicy(mode="raw", factor_source=BAOSTOCK_FACTOR_SOURCE)

    @pytest.mark.parametrize("mode", ["hfq", "qfq"])
    def test_adjusted_modes_require_factor_source(self, mode: str) -> None:
        with pytest.raises(ValidationError, match="requires a factor_source"):
            AdjustmentPolicy(mode=mode)  # type: ignore[arg-type]
        with pytest.raises(ValidationError, match="requires a factor_source"):
            AdjustmentPolicy(mode=mode, factor_source="   ")  # type: ignore[arg-type]

    @pytest.mark.parametrize("mode", ["hfq", "qfq"])
    def test_adjusted_modes_with_source(self, mode: str) -> None:
        policy = AdjustmentPolicy(mode=mode, factor_source=BAOSTOCK_FACTOR_SOURCE)  # type: ignore[arg-type]
        assert policy.requires_factor_snapshot is True
        assert policy.policy_version == ADJUSTMENT_POLICY_VERSION

    def test_unknown_mode_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AdjustmentPolicy(mode="none")  # type: ignore[arg-type]

    def test_empty_version_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must be non-empty"):
            AdjustmentPolicy(mode="raw", policy_version="  ")

    def test_run_metadata(self) -> None:
        policy = AdjustmentPolicy(mode="hfq", factor_source=BAOSTOCK_FACTOR_SOURCE)
        assert policy.run_metadata() == {
            "adjustment_policy_version": "adjustment-policy-v1",
            "parameters": {"mode": "hfq", "factor_source": BAOSTOCK_FACTOR_SOURCE},
        }

    def test_run_metadata_raw(self) -> None:
        assert AdjustmentPolicy(mode="raw").run_metadata() == {
            "adjustment_policy_version": "adjustment-policy-v1",
            "parameters": {"mode": "raw", "factor_source": None},
        }

    def test_frozen(self) -> None:
        policy = AdjustmentPolicy(mode="raw")
        with pytest.raises(ValidationError):
            policy.mode = "hfq"  # type: ignore[misc]

    def test_default_is_raw(self) -> None:
        # ADR-007：forecast 默认输入为 raw（PIT 稳定，不依赖因子快照）
        assert DEFAULT_FORECAST_ADJUSTMENT_MODE == "raw"
        policy = default_forecast_adjustment_policy()
        assert policy.mode == "raw"
        assert policy.requires_factor_snapshot is False

    def test_policy_version_pinned(self) -> None:
        assert ADJUSTMENT_POLICY_VERSION == "adjustment-policy-v1"


class TestAdjustFactorRecord:
    def test_valid(self) -> None:
        record = AdjustFactorRecord(
            ex_date=date(2026, 7, 16), fore_factor=1.0, back_factor=13.367013
        )
        assert record.ex_date == date(2026, 7, 16)

    @pytest.mark.parametrize("field", ["fore_factor", "back_factor"])
    @pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
    def test_invalid_factors_rejected(self, field: str, value: float) -> None:
        fields = {"ex_date": date(2026, 7, 16), "fore_factor": 1.0, "back_factor": 1.0}
        fields[field] = value
        with pytest.raises(ValidationError, match=field):
            AdjustFactorRecord(**fields)  # type: ignore[arg-type]

    def test_datetime_ex_date_rejected(self) -> None:
        from datetime import datetime

        with pytest.raises(ValidationError):
            AdjustFactorRecord(
                ex_date=datetime(2026, 7, 16, 15, 0),  # type: ignore[arg-type]
                fore_factor=1.0,
                back_factor=1.0,
            )

    def test_frozen(self) -> None:
        record = AdjustFactorRecord(ex_date=date(2026, 7, 16), fore_factor=1.0, back_factor=1.0)
        with pytest.raises(ValidationError):
            record.back_factor = 2.0  # type: ignore[misc]


class TestAdjustFactorSeries:
    def test_valid_and_coverage(self) -> None:
        series = real_series()
        assert series.coverage == (date(2000, 7, 6), date(2026, 7, 16))
        assert len(series.records) == len(REAL_FACTORS)

    def test_symbol_must_be_normalized(self) -> None:
        with pytest.raises(ValidationError, match="6-digit"):
            real_series(symbol="sh.600000")

    @pytest.mark.parametrize("field", ["source", "dataset_version"])
    def test_provenance_required(self, field: str) -> None:
        with pytest.raises(ValidationError, match="must be non-empty"):
            real_series(**{field: " "})

    def test_empty_records_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not be empty"):
            real_series(records=())

    def test_unsorted_records_rejected(self) -> None:
        reversed_records = tuple(reversed(real_series().records))
        with pytest.raises(ValidationError, match="sorted"):
            real_series(records=reversed_records)

    def test_duplicate_ex_dates_rejected(self) -> None:
        records = real_series().records
        with pytest.raises(ValidationError, match="unique"):
            real_series(records=(records[0], records[0]))

    def test_frozen(self) -> None:
        series = real_series()
        with pytest.raises(ValidationError):
            series.source = "other"  # type: ignore[misc]


class TestFactorsAsof:
    def test_before_first_ex_date_returns_none(self) -> None:
        assert real_series().factors_asof(date(2000, 7, 5)) is None

    def test_first_ex_date_is_inclusive(self) -> None:
        record = real_series().factors_asof(date(2000, 7, 6))
        assert record is not None
        assert record.ex_date == date(2000, 7, 6)

    def test_on_ex_date_is_inclusive(self) -> None:
        record = real_series().factors_asof(date(2020, 7, 23))
        assert record is not None
        assert record.back_factor == pytest.approx(10.312133)

    def test_between_ex_dates_uses_previous(self) -> None:
        record = real_series().factors_asof(date(2023, 1, 3))
        assert record is not None
        assert record.ex_date == date(2022, 7, 21)

    def test_after_last_ex_date_uses_last(self) -> None:
        record = real_series().factors_asof(date(2026, 9, 26))
        assert record is not None
        assert record.ex_date == date(2026, 7, 16)
        assert record.fore_factor == 1.0

    def test_factors_reproduce_verified_adjusted_closes(self) -> None:
        # 能力报告 §2 的映射规则（1745 个标的日、0 失配）：
        # hfq(D)/raw(D) == back_factor、qfq(D)/raw(D) == fore_factor
        # 用 artifact 记录的 raw/hfq/qfq 收盘独立复算，避免只读 fixture 的空转断言。
        record = real_series().factors_asof(GOLDEN_DAY)
        assert record is not None
        assert record.ex_date == GOLDEN_EX_DATE
        assert pytest.approx(record.back_factor, rel=1e-9) == GOLDEN_HFQ_CLOSE / GOLDEN_RAW_CLOSE
        assert pytest.approx(record.fore_factor, rel=1e-9) == GOLDEN_QFQ_CLOSE / GOLDEN_RAW_CLOSE

    def test_table_is_normalized_to_latest_action(self) -> None:
        # spike 事实（artifact: checks.factor_normalization）：6 个被测标的的
        # 末条 foreAdjustFactor 恒为 1.0 —— 整表相对查询时刻重算，非 PIT 稳定。
        normalization = spike_artifact()["checks"]["factor_normalization"]
        assert normalization["all_latest_fore_unity"] is True
        assert len(normalization["codes"]) == 6
        for code, entry in normalization["codes"].items():
            assert float(entry["latest_fore_factor"]) == 1.0, code
            assert entry["latest_fore_is_unity"] is True, code
        # fixture 自身也必须满足同一性质（否则 fixture 与实测脱节）
        assert real_series().records[-1].fore_factor == 1.0


class TestFixtureMatchesSpikeArtifact:
    """手写 fixture 只允许是 artifact 的抄录，禁止与实测漂移（评审 m4）。"""

    def test_factor_fixture_rows_come_from_artifact(self) -> None:
        adjust = spike_artifact()["checks"]["adjust_factor"]
        index = {name: position for position, name in enumerate(adjust["fields"])}
        table = {
            row[index["dividOperateDate"]]: (
                row[index["foreAdjustFactor"]],
                row[index["backAdjustFactor"]],
            )
            for row in adjust["factor_table"]
        }
        for day, fore, back in REAL_FACTORS:
            assert day in table, day
            assert float(table[day][0]) == pytest.approx(fore, rel=1e-9), day
            assert float(table[day][1]) == pytest.approx(back, rel=1e-9), day

    def test_golden_constants_come_from_artifact(self) -> None:
        sample = spike_artifact()["checks"]["factor_price_mapping"]["sample_day"]
        assert sample["code"] == "sh.600000"
        assert date.fromisoformat(sample["day"]) == GOLDEN_DAY
        assert sample["closes"]["raw"] == pytest.approx(GOLDEN_RAW_CLOSE)
        assert sample["closes"]["hfq"] == pytest.approx(GOLDEN_HFQ_CLOSE)
        assert sample["closes"]["qfq"] == pytest.approx(GOLDEN_QFQ_CLOSE)
        assert date.fromisoformat(sample["applicable_ex_date"]) == GOLDEN_EX_DATE
