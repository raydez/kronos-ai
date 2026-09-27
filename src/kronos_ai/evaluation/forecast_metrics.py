"""Forecast Benchmark 指标（基线文档 §41 / §48 / §49 / §13）。

§41 把 Phase 2 的产出定义为 `Forecast Benchmark v1`：同一批 point-in-time origin 上，
Kronos 与 naive baseline 用**同一份样本路径与同一份指标代码**比较，再据此做 §42 的
Go / Replace 判断。本模块只做「从逐 origin 记录算出 §49 指标」这一件事：它不加载数据、
不认识 backend 实现、也不写 artifact（runner 在 :mod:`kronos_ai.evaluation.benchmark`）。

可比较性是这里唯一的设计目标：

- **点预测口径显式且版本化**：每条记录的模型点预测取 ``ForecastDistribution``
  的 ``expected_return``（§12 的跨样本均值），不是「某个 backend 自己挑的数字」；
- **方向口径与 label 同源**：预测方向由 §28 的 ``direction_label`` 用**同一份
  LabelPolicy** 从点预测导出，因此 direction accuracy 是「同一阈值下的分类一致率」，
  而不是两套涨跌口径的比较；
- **缺料是 ``None`` 而不是 0**：某个分组里没有 ``LABELED`` 记录、或只有 1 个有效点
  （相关系数无定义）时，指标为 ``None``。用 0 冒充「没有证据」会让 gate 把
  「样本不足」读成「表现很差」（ADR-010）。

§49 的指标清单在本模块的落点：

```text
MAE                mean |predicted − realized|
RMSE               sqrt(mean (predicted − realized)^2)
Direction Accuracy 预测方向 == 实际方向的比例（同 LabelPolicy 阈值）
Return Correlation 预测与实际 horizon 收益的 Pearson 相关系数
Quantile Coverage  实际值落入预测中心区间的比例（区间由 CoverageSpec 版本化）
CRPS               显式 defer（见 DEFERRED_FORECAST_EVAL_METRICS）
```

Robustness 分组（Bull / Bear / Sideways / High Vol / Low Vol）不在这里定义：regime 是
point-in-time 分类（只能用 origin 当日及之前的信息），由
:mod:`kronos_ai.evaluation.regimes` 版本化地给出，本模块只按传入的 regime 标签切片。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kronos_ai.domain.hashing import sha256_hex
from kronos_ai.domain.time import SessionDate
from kronos_ai.errors import ConfigurationError
from kronos_ai.evaluation.dataset import DirectionLabel, LabelStatus
from kronos_ai.evaluation.regimes import TrendRegime, VolatilityRegime
from kronos_ai.evaluation.walk_forward import SegmentName

FORECAST_EVAL_METRICS_VERSION = "forecast-eval-metrics-v1"
COVERAGE_SPEC_VERSION = "coverage-spec-v1"

#: §49 已列但**本期不实现**的指标。显式登记，而不是让它悄悄缺席：
#: CRPS 需要一个版本化的 sample-based CRPS 定义（含结的处理与区间积分口径），
#: 属后续任务；在没有定义之前产出 CRPS 数字等于产出不可解释的指标。
DEFERRED_FORECAST_EVAL_METRICS: tuple[str, ...] = ("crps",)


def _require_finite(value: float, field: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return value


class CoverageSpec(BaseModel):
    """Quantile Coverage 的中心区间定义（§49）。

    区间端点必须是 ForecastDistribution 里**实际存在**的分位（由 DistributionSpec 请求），
    否则该 backend 的 coverage 为 ``None``；这里只声明口径，不替 backend 补分位。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = COVERAGE_SPEC_VERSION
    lower_quantile: float = Field(default=0.05, gt=0, lt=1)
    upper_quantile: float = Field(default=0.95, gt=0, lt=1)

    @model_validator(mode="after")
    def _ordered(self) -> CoverageSpec:
        if self.lower_quantile >= self.upper_quantile:
            raise ValueError(
                f"lower_quantile {self.lower_quantile} must be < upper_quantile "
                f"{self.upper_quantile}"
            )
        return self

    @field_validator("version")
    @classmethod
    def _version_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("version must be non-empty")
        return value

    @property
    def nominal_coverage(self) -> float:
        """名义覆盖率 = ``upper − lower``（区间是中心区间，不是单尾）。"""
        return self.upper_quantile - self.lower_quantile

    def hashing_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": self.version,
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
        }
        assert set(payload) == set(type(self).model_fields), (
            "CoverageSpec.hashing_payload must cover every model field"
        )
        return payload


DEFAULT_COVERAGE_SPEC = CoverageSpec()


def coverage_spec_hash(spec: CoverageSpec) -> str:
    """CoverageSpec 的确定性 hash；进 benchmark 报告，使「换了区间口径」可被指认。"""
    return sha256_hex(
        {
            "kind": "coverage_spec",
            "forecast_eval_metrics_version": FORECAST_EVAL_METRICS_VERSION,
            "spec": spec.hashing_payload(),
        }
    )


class ForecastEvalMetricDefinition(BaseModel):
    """版本化指标定义（§49）：description 即数学口径，不允许由调用方自述。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    unit: str
    description: str
    #: 该指标需要的输入；runner 据此决定能否计算（缺料的指标为 None）。
    requires: tuple[str, ...]


_FORECAST_EVAL_METRICS: dict[str, ForecastEvalMetricDefinition] = {
    "mae": ForecastEvalMetricDefinition(
        name="mae",
        unit="ratio",
        description="mean |predicted_return − realized_return|，仅用 LABELED 记录",
        requires=("predicted_return", "realized_return"),
    ),
    "rmse": ForecastEvalMetricDefinition(
        name="rmse",
        unit="ratio",
        description="sqrt(mean (predicted_return − realized_return)^2)，仅用 LABELED 记录",
        requires=("predicted_return", "realized_return"),
    ),
    "direction_accuracy": ForecastEvalMetricDefinition(
        name="direction_accuracy",
        unit="ratio",
        description=(
            "预测方向 == 实际方向的记录比例；两侧均由同一 LabelPolicy.direction_thresholds "
            "从收益映射而来（预测侧对 expected_return 施加 direction_label）"
        ),
        requires=("predicted_direction", "realized_direction"),
    ),
    "return_correlation": ForecastEvalMetricDefinition(
        name="return_correlation",
        unit="ratio",
        description=(
            "predicted_return 与 realized_return 的 Pearson 相关系数；样本 < 2 或任一维 "
            "方差为 0 时无定义（None），不用 0 冒充"
        ),
        requires=("predicted_return", "realized_return"),
    ),
    "quantile_coverage": ForecastEvalMetricDefinition(
        name="quantile_coverage",
        unit="ratio",
        description=(
            "realized_return 落入 [q_lower, q_upper] 预测区间的比例；区间端点缺失的记录不 "
            "参与计算，全部缺失则该组指标为 None（名义覆盖率见 CoverageSpec）"
        ),
        requires=("coverage_lower", "coverage_upper", "realized_return"),
    ),
}

FORECAST_EVAL_METRIC_NAMES: tuple[str, ...] = tuple(_FORECAST_EVAL_METRICS)


def forecast_eval_metric_definition(name: str) -> ForecastEvalMetricDefinition:
    """按名取 registry 定义；未知名显式失败（不允许调用方自造指标名）。"""
    try:
        return _FORECAST_EVAL_METRICS[name]
    except KeyError as exc:
        raise ConfigurationError(
            f"unknown forecast eval metric {name!r}; allowed: {sorted(_FORECAST_EVAL_METRICS)}"
        ) from exc


class ForecastEvalRecord(BaseModel):
    """一个 ``(segment, symbol, market_date, backend)`` 的评估记录（runner 的产物）。

    ``label_status != "LABELED"`` 的记录仍保留（覆盖率要能统计），但 ``realized_*``
    必须为空：把停牌 / 数据末端当成「0 收益」会让指标系统性地偏向乐观。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    segment: SegmentName
    symbol: str
    market_date: SessionDate
    backend: str
    model_revision: str
    artifact_id: str

    predicted_return: float
    median_return: float
    predicted_direction: DirectionLabel

    coverage_lower: float | None = None
    coverage_upper: float | None = None

    label_status: LabelStatus
    realized_return: float | None = None
    realized_direction: DirectionLabel | None = None

    trend_regime: TrendRegime
    volatility_regime: VolatilityRegime

    latency_ms: float = Field(ge=0)

    @field_validator(
        "predicted_return",
        "median_return",
        "coverage_lower",
        "coverage_upper",
        "realized_return",
    )
    @classmethod
    def _finite(cls, value: float | None) -> float | None:
        if value is None:
            return None
        return _require_finite(value, "value")

    @field_validator("backend", "model_revision", "artifact_id")
    @classmethod
    def _nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field must be non-empty")
        return value

    @model_validator(mode="after")
    def _label_and_coverage_consistent(self) -> ForecastEvalRecord:
        if self.label_status == "LABELED":
            if self.realized_return is None or self.realized_direction is None:
                raise ValueError(
                    "LABELED records must carry realized_return and realized_direction"
                )
        elif self.realized_return is not None or self.realized_direction is not None:
            raise ValueError(
                f"{self.label_status} records must not carry realized return/direction "
                "(证据不足不是 0 收益)"
            )
        (lower, upper) = (self.coverage_lower, self.coverage_upper)
        if (lower is None) != (upper is None):
            raise ValueError(
                "coverage_lower and coverage_upper must be both set or both absent; "
                "a half-specified interval would silently change coverage semantics"
            )
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(f"coverage_lower {lower} must be <= coverage_upper {upper}")
        return self

    @property
    def is_labeled(self) -> bool:
        return self.label_status == "LABELED"

    def hashing_payload(self) -> dict[str, Any]:
        """进 benchmark report hash 的规范 payload。

        ``latency_ms`` **刻意缺席**：它是 run 当时的墙钟观测（环境事实），不是结论的一部分。
        把它算进去，同一条命令跑两次就会得到不同 ``report_hash``，而每一个指标都逐位相同
        ——「同一份结论被复现」就不可判定（与 ``git_commit`` / ``data_coverage_end`` 同一个
        理由，见 ADR-023 §5）。延迟本身照旧进 ``metrics.json`` 与 ``forecast.parquet``。
        """
        payload: dict[str, Any] = {
            "segment": self.segment,
            "symbol": self.symbol,
            "market_date": self.market_date.isoformat(),
            "backend": self.backend,
            "model_revision": self.model_revision,
            "artifact_id": self.artifact_id,
            "predicted_return": self.predicted_return,
            "median_return": self.median_return,
            "predicted_direction": self.predicted_direction,
            "coverage_lower": self.coverage_lower,
            "coverage_upper": self.coverage_upper,
            "label_status": self.label_status,
            "realized_return": self.realized_return,
            "realized_direction": self.realized_direction,
            "trend_regime": self.trend_regime,
            "volatility_regime": self.volatility_regime,
        }
        hashed_fields = set(type(self).model_fields) - {"latency_ms"}
        assert set(payload) == hashed_fields, (
            "ForecastEvalRecord.hashing_payload must cover every model field except latency_ms "
            "(see its docstring for why it is metadata-only)"
        )
        return payload


GroupAxis = Literal["overall", "trend", "volatility"]

#: 聚合切片的键：``(backend, segment, axis, value)``；``segment=None`` 为跨段汇总。
GroupKey = tuple[str, SegmentName | None, GroupAxis, str]


class ForecastEvalGroup(BaseModel):
    """一个切片：``axis=overall`` 时不分组，否则按对应 regime 轴取值。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    axis: GroupAxis
    value: str

    @field_validator("value")
    @classmethod
    def _value_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value

    @property
    def label(self) -> str:
        return self.value if self.axis == "overall" else f"{self.axis}={self.value}"


OVERALL_GROUP = ForecastEvalGroup(axis="overall", value="all")


class ForecastEvalMetrics(BaseModel):
    """一个 ``(backend, segment, group)`` 的 §49 指标集合。

    ``sample_count`` 是该切片的**全部**记录数，``labeled_count`` 是其中真正参与指标计算的
    记录数；两者一起回答「这个数字是从多少条证据上算出来的」。

    ``segment=None`` 是**跨段汇总**（整次 run 的口径），不是一个「没写段名」的缺省值：
    §49 的标题指标（MAE / direction accuracy / coverage）是整次 run 的数字，而逐段数字
    回答「结论是否只在某一段成立」。两者同时产出、各自有明确的 ``segment``，因此
    「这个数字覆盖哪些 session」永远可以从 artifact 读出来，不必去猜。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: str
    #: ``None`` = 跨段汇总（run 级）；否则为 dataset 里的段名。
    segment: SegmentName | None = None
    group: ForecastEvalGroup = OVERALL_GROUP

    sample_count: int = Field(ge=0)
    labeled_count: int = Field(ge=0)

    mae: float | None = None
    rmse: float | None = None
    direction_accuracy: float | None = None
    return_correlation: float | None = None
    quantile_coverage: float | None = None

    coverage_nominal: float | None = None
    version: str = FORECAST_EVAL_METRICS_VERSION

    @field_validator("version")
    @classmethod
    def _version_matches(cls, value: str) -> str:
        if value != FORECAST_EVAL_METRICS_VERSION:
            raise ValueError(
                f"unknown forecast eval metrics version {value!r}; "
                f"this build emits {FORECAST_EVAL_METRICS_VERSION!r}"
            )
        return value

    @field_validator("mae", "rmse", "direction_accuracy", "return_correlation", "quantile_coverage")
    @classmethod
    def _metric_finite(cls, value: float | None) -> float | None:
        return None if value is None else _require_finite(value, "metric")

    @model_validator(mode="after")
    def _counts_consistent(self) -> ForecastEvalMetrics:
        if self.labeled_count > self.sample_count:
            raise ValueError(
                f"labeled_count {self.labeled_count} exceeds sample_count {self.sample_count}"
            )
        return self

    def hashing_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "backend": self.backend,
            "segment": self.segment,
            "group": {"axis": self.group.axis, "value": self.group.value},
            "sample_count": self.sample_count,
            "labeled_count": self.labeled_count,
            "mae": self.mae,
            "rmse": self.rmse,
            "direction_accuracy": self.direction_accuracy,
            "return_correlation": self.return_correlation,
            "quantile_coverage": self.quantile_coverage,
            "coverage_nominal": self.coverage_nominal,
            "version": self.version,
        }
        assert set(payload) == set(type(self).model_fields), (
            "ForecastEvalMetrics.hashing_payload must cover every model field"
        )
        return payload


def _labeled(records: Iterable[ForecastEvalRecord]) -> list[ForecastEvalRecord]:
    return [record for record in records if record.is_labeled]


def _mae(records: Sequence[ForecastEvalRecord]) -> float | None:
    if not records:
        return None
    total = math.fsum(abs(record.predicted_return - _realized(record)) for record in records)
    return total / len(records)


def _rmse(records: Sequence[ForecastEvalRecord]) -> float | None:
    if not records:
        return None
    total = math.fsum((record.predicted_return - _realized(record)) ** 2 for record in records)
    return math.sqrt(total / len(records))


def _direction_accuracy(records: Sequence[ForecastEvalRecord]) -> float | None:
    if not records:
        return None
    hits = sum(1 for record in records if record.predicted_direction == record.realized_direction)
    return hits / len(records)


def _return_correlation(records: Sequence[ForecastEvalRecord]) -> float | None:
    """Pearson 相关系数；样本 < 2 或任一维方差为 0 时无定义（返回 ``None``）。"""
    if len(records) < 2:
        return None
    predicted = [record.predicted_return for record in records]
    realized = [_realized(record) for record in records]
    predicted_mean = math.fsum(predicted) / len(predicted)
    realized_mean = math.fsum(realized) / len(realized)
    covariance = math.fsum(
        (p - predicted_mean) * (r - realized_mean) for p, r in zip(predicted, realized, strict=True)
    )
    predicted_var = math.fsum((p - predicted_mean) ** 2 for p in predicted)
    realized_var = math.fsum((r - realized_mean) ** 2 for r in realized)
    if predicted_var <= 0 or realized_var <= 0:
        return None
    return covariance / math.sqrt(predicted_var * realized_var)


def _quantile_coverage(records: Sequence[ForecastEvalRecord]) -> tuple[float | None, int]:
    """返回 ``(coverage, used)``；``used`` 是同时带区间端点与实际收益的记录数。"""
    usable = [
        record
        for record in records
        if record.coverage_lower is not None and record.coverage_upper is not None
    ]
    if not usable:
        return None, 0
    inside = 0
    for record in usable:
        assert record.coverage_lower is not None and record.coverage_upper is not None
        if record.coverage_lower <= _realized(record) <= record.coverage_upper:
            inside += 1
    return inside / len(usable), len(usable)


def _realized(record: ForecastEvalRecord) -> float:
    """LABELED 记录的 realized_return；非 LABELED 记录调用本函数是调用方 bug。"""
    assert record.realized_return is not None, "_realized must only be called on LABELED records"
    return record.realized_return


def evaluate_forecast_records(
    records: Sequence[ForecastEvalRecord],
    *,
    coverage: CoverageSpec = DEFAULT_COVERAGE_SPEC,
) -> tuple[ForecastEvalMetrics, ...]:
    """把逐 origin 记录聚合成 §49 指标集合。

    每个 backend 产出两类切片：

    ```text
    segment=None     跨段汇总（run 级口径），三轴都有
    segment=<name>   逐段（train / validation / calibration / test），三轴都有
    ```

    跨段汇总是**显式**的：没有它，「本次 run 的 MAE」只能靠读的人自己把逐段数字加权平均，
    或（更糟）拿第一段的数字当整体；CLI 摘要与报告标题行正是要这个数字。

    产出顺序是确定的（backend → segment → 三轴，run 级排在段之前），使报告可逐行 diff；
    同一份记录集合永远给出同一份指标集合。
    """
    grouped: dict[GroupKey, list[ForecastEvalRecord]] = {}
    for record in records:
        keys: tuple[GroupKey, ...] = (
            (record.backend, None, "overall", "all"),
            (record.backend, None, "trend", record.trend_regime),
            (record.backend, None, "volatility", record.volatility_regime),
            (record.backend, record.segment, "overall", "all"),
            (record.backend, record.segment, "trend", record.trend_regime),
            (record.backend, record.segment, "volatility", record.volatility_regime),
        )
        for key in keys:
            grouped.setdefault(key, []).append(record)

    metrics: list[ForecastEvalMetrics] = []
    for key in sorted(grouped, key=_group_sort_key):
        backend, segment, axis, value = key
        bucket = grouped[key]
        labeled = _labeled(bucket)
        coverage_value, _used = _quantile_coverage(labeled)
        metrics.append(
            ForecastEvalMetrics(
                backend=backend,
                segment=segment,
                group=ForecastEvalGroup(axis=axis, value=value),
                sample_count=len(bucket),
                labeled_count=len(labeled),
                mae=_mae(labeled),
                rmse=_rmse(labeled),
                direction_accuracy=_direction_accuracy(labeled),
                return_correlation=_return_correlation(labeled),
                quantile_coverage=coverage_value,
                coverage_nominal=coverage.nominal_coverage,
            )
        )
    return tuple(metrics)


def _group_sort_key(
    key: GroupKey,
) -> tuple[str, str, str, str]:
    """确定性排序键：run 级（``segment is None``）排在逐段之前，其余按名字。

    直接 ``sorted(grouped)`` 会因为 ``None`` 与 ``str`` 不可比较而报错——把排序键显式写
    出来，既避开该错误，也让产出顺序成为契约的一部分。
    """
    backend, segment, axis, value = key
    return (backend, "" if segment is None else segment, axis, value)


__all__ = [
    "COVERAGE_SPEC_VERSION",
    "DEFAULT_COVERAGE_SPEC",
    "DEFERRED_FORECAST_EVAL_METRICS",
    "FORECAST_EVAL_METRICS_VERSION",
    "FORECAST_EVAL_METRIC_NAMES",
    "OVERALL_GROUP",
    "CoverageSpec",
    "ForecastEvalGroup",
    "ForecastEvalMetricDefinition",
    "ForecastEvalMetrics",
    "ForecastEvalRecord",
    "coverage_spec_hash",
    "evaluate_forecast_records",
    "forecast_eval_metric_definition",
]
