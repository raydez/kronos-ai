"""Forecast Backend Go / Replace Gate（基线文档 §19、§42；ADR-024）。

§19 的问题是「Kronos 是否值得继续作为默认 Forecast Backend」，而不是「Kronos 是否必然
有效」；§42 把回答形式固定下来：**判据必须在任何正式 benchmark run 之前预注册**，随
run_id 一起归档，且必须可证伪。本模块实现这套判据与判决。

三条设计约束，缺一条判决就不可信（ADR-024 §背景）：

1. **判据先于数据**。:class:`GateCriteria` 是版本化文档，其 canonical hash
   （:attr:`GateCriteria.criteria_hash`）随 run 归档；判决 :class:`GateVerdict` 绑定
   ``criteria_hash`` 与 ``report_hash``。任何「跑完再挑指标 / 再放宽阈值」都会换掉
   ``criteria_hash``，而归档的原文仍在 run 目录里——事后替换是**可见**的。
2. **增量必须带区间**。判决不用单点数字：每个切片（整体 + 每个 regime 分组）都给出
   逐 origin **配对** bootstrap 的百分位区间，只看区间下界是否大于 0（§42 的例子）。
3. **证据不足 ≠ 表现很差**。配对样本量低于预注册门槛时抛
   :class:`~kronos_ai.errors.InsufficientEvidenceError`，而不是给出 REPLACE
   （ADR-010 在评估侧的实例）。

「哪些 origin 算证据」（:attr:`GateCriteria.evidence_segments`）、「跟谁比」
（``baseline_selection`` 选出的最强 baseline）与「怎么分组」都锚定在**同一批证据段**上：
分组门槛、候选与比较对象的比较区间都只数这批段里的 origin。一处需要说清的口径差别是
「选比较对象」与「判决」的集合：前者按每个 baseline **自己**在这批段里的 ``LABELED``
记录聚合，后者按它与候选**配对**成功的 origin（配对集合可以更小，差额写在证据里）。

判决规则（完整、无遗漏，且被 :class:`GateVerdict` 的不变量复核）::

    GO          整体增量 CI 下界 > 0 且 (>= min_groups 个 regime 分组下界 > 0)
                且 compute 在预注册护栏内
    CONDITIONAL 整体增量下界 > 0 或 (>= 1 个 regime 分组下界 > 0)   —— 其余情形
    REPLACE     整体下界 <= 0 且没有任何一个 regime 分组下界 > 0

``CONDITIONAL`` 携带「在哪些 regime 有效」（§42：后续按 regime 使用）。整体条件是必须的：
默认 backend 服务于整个 universe，只在部分 regime 有效不构成「继续作为默认」。

口径都是版本化的：指标方向表（:data:`GATE_METRIC_DIRECTIONS`）、CI 方法
（:data:`CI_METHOD`）、区间取值法（:data:`CI_QUANTILE_METHOD`）任一改变都会改变判决的
含义，必须同时升版本并重跑依赖它的正式 run。
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kronos_ai.domain.hashing import is_sha256_hex, sha256_hex
from kronos_ai.errors import ConfigurationError, InsufficientEvidenceError
from kronos_ai.evaluation.benchmark import ForecastBenchmarkResult
from kronos_ai.evaluation.forecast_metrics import ForecastEvalRecord
from kronos_ai.evaluation.regimes import TREND_REGIMES, VOLATILITY_REGIMES
from kronos_ai.evaluation.walk_forward import SEGMENT_NAMES

GATE_VERSION = "forecast-gate-v1"
GATE_CRITERIA_VERSION = "forecast-gate-criteria-v1"

#: 区间口径：**逐 origin 配对**的百分位 bootstrap（§42 的「bootstrap 95% CI」）。
#: 配对是关键：候选与 baseline 必须在**同一批** origin 上比较，否则区间里会混进样本
#: 差异，而样本差异不是模型的增量。
CI_METHOD = "paired_bootstrap_percentile"

#: 区间端点取值法：最近秩法（与 §49 Compute 分位口径同一套约定，见
#: :func:`kronos_ai.evaluation.compute_metrics.percentile`）。取最近秩而不是线性插值，
#: 是为了让两个端点都是**真实出现过的一次重采样结果**。
CI_QUANTILE_METHOD = "nearest_rank"

GateMetric = Literal["direction_accuracy", "mae", "rmse"]
GateGroupingAxis = Literal["trend", "volatility"]
GateOutcome = Literal["GO", "CONDITIONAL", "REPLACE"]

#: 允许进 gate 的指标白名单及方向（``higher_is_better``）。**增量**的正负号统一为
#: 「正数 = 候选更好」，因此 mae / rmse 的方向在这里翻转，而不是靠读的人记得。
#:
#: 刻意排除两个 §49 指标：
#:
#: - ``return_correlation``：相关系数大不等于预测更准（方向可能整体反了），
#:   「更好的相关系数」没有唯一的单调方向；
#: - ``quantile_coverage``：它是校准属性，「离名义覆盖率更近更好」不是可累加的增量。
#:
#: 把它们排除而不是允许「越大越好」，是因为 gate 的判据必须可证伪；方向不唯一的指标
#: 会让同一个数字支持两种相反的结论。
GATE_METRIC_DIRECTIONS: Mapping[str, bool] = {
    "direction_accuracy": True,
    "mae": False,
    "rmse": False,
}
GATE_METRICS: tuple[str, ...] = tuple(GATE_METRIC_DIRECTIONS)

_AXIS_VALUES: Mapping[str, tuple[str, ...]] = {
    "trend": TREND_REGIMES,
    "volatility": VOLATILITY_REGIMES,
}

#: 理由码词表：判决只携带**码**，散文由 renderer 生成，因此 ``gate.json`` 可逐位 diff。
_GATE_REASON_CODES: tuple[str, ...] = (
    "COMPUTE_BUDGET_EXCEEDED",
    "COMPUTE_BUDGET_WITHIN_LIMIT",
    "NO_REGIME_SHOWS_POSITIVE_INCREMENT",
    "OVERALL_INCREMENT_CI_LOWER_BOUND_NOT_POSITIVE",
    "OVERALL_INCREMENT_CI_LOWER_BOUND_POSITIVE",
    "REGIMES_WITH_POSITIVE_INCREMENT_BELOW_MIN",
    "REGIMES_WITH_POSITIVE_INCREMENT_MEETS_MIN",
)

_REASON_PROSE: Mapping[str, str] = {
    # 「整体」是**判据声明的证据切片内**的全部 origin（``slice=all``），不是跨段池化：
    # 池化 train 与只看 test 是两个不同的判决（见 evidence_segments）。
    "OVERALL_INCREMENT_CI_LOWER_BOUND_POSITIVE": "整体（证据切片内的全部 origin）增量的 CI 下界 > 0",
    "OVERALL_INCREMENT_CI_LOWER_BOUND_NOT_POSITIVE": (
        "整体（证据切片内的全部 origin）增量的 CI 下界 <= 0"
    ),
    "REGIMES_WITH_POSITIVE_INCREMENT_MEETS_MIN": "达到预注册门槛的正增量 regime 分组数",
    "REGIMES_WITH_POSITIVE_INCREMENT_BELOW_MIN": "正增量 regime 分组数未达预注册门槛",
    "NO_REGIME_SHOWS_POSITIVE_INCREMENT": "没有任何一个 regime 分组出现正增量下界",
    "COMPUTE_BUDGET_WITHIN_LIMIT": "每 origin 平均耗时在预注册护栏内",
    "COMPUTE_BUDGET_EXCEEDED": "每 origin 平均耗时超出预注册护栏（不可 GO）",
}

_OUTCOME_PROSE: Mapping[str, str] = {
    "GO": "Kronos 继续作为默认 ForecastBackend（§42 GO）",
    "CONDITIONAL": "Kronos 只在部分 regime 有效：后续按 regime 使用（§42 CONDITIONAL）",
    "REPLACE": (
        "Kronos 无稳定增量：降级为 benchmark backend，并引入其他 Forecast Backend"
        "（§42 REPLACE；平台建设不终止）"
    ),
}


class GateCriteria(BaseModel):
    """预注册的 gate 判据（§42）。字段是**判据形态**，不是结论。

    §42：「判据的具体数值可以在 Cost Probe 完成后、正式 run 之前确定，但判据形态（指标、
    比较对象、分组方式、置信区间）必须在 run 前冻结」。本模型把形态做成必填字段：
    缺少指标 / 比较对象 / 分组方式 / **证据切片** / 置信区间 / compute 护栏的判据根本无法
    构造，因此不可能「先跑再想」。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = GATE_CRITERIA_VERSION

    #: 被判定的 backend（通常 ``kronos``）。
    candidate: str

    #: 唯一主判据指标；必须是 :data:`GATE_METRICS` 之一（方向由白名单给出）。
    metric: GateMetric

    #: 比较对象**集合**：判据说「相对最强 statistical baseline」，而「最强」由**同一指标**
    #: 在**同一批证据段**（:attr:`evidence_segments`）上选出（:attr:`baseline_selection`）。
    #: 集合必须预注册，否则「最强 baseline」可以在跑完之后再挑；集合里放哪些名字是作者的
    #: 选择，但它会被原样归档，因此「挑了谁当垫脚石」永远可查。
    baselines: tuple[str, ...]
    baseline_selection: Literal["strongest"] = "strongest"

    grouping_axis: GateGroupingAxis = "trend"

    #: **哪些段的 origin 算证据**（必填，无默认）。「哪些 origin 算证据」至少和分组方式一样
    #: 是承重的判据形态：同一份 run、同一份判据，只看 test 段与把 train 池进来会给出不同的
    #: 分组样本量与不同的判决（实测：示例窗口 test 段 BEAR 只有 3 个 origin，池化后有 5 个）。
    #: 因此它必须与指标 / 比较对象 / 分组方式一起在 run 前冻结。值必须是本次 run 的
    #: ``dataset.segments`` 的子集（``load_experiment_config`` 在 run 前校验）。
    evidence_segments: tuple[str, ...]

    #: 至少要有多少个 regime 分组出现正的增量下界（§42 的例子是 >= 3）。
    min_groups_with_positive_increment: int = Field(ge=1)
    #: 逐分组 / 整体切片的最少配对样本数；低于它的分组不参与判决（并在判决里显式标出）。
    min_paired_samples_per_group: int = Field(ge=2)
    min_paired_samples_overall: int = Field(ge=2)

    confidence_level: float = Field(gt=0.5, lt=1)
    bootstrap_iterations: int = Field(ge=100)
    bootstrap_seed: int = Field(ge=0)

    #: §19 把 compute cost 列为判断维度之一：代价护栏是判据的一部分，不是跑完的观后感。
    #: 观测值是候选 backend 在本次 run 里**每个 origin 的平均墙钟秒数**。
    max_seconds_per_origin: float = Field(gt=0)

    @field_validator("candidate")
    @classmethod
    def _candidate_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("criteria candidate must be non-empty")
        return value.strip()

    @field_validator("baselines")
    @classmethod
    def _baselines_nonempty_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(name.strip() for name in value)
        if not cleaned:
            raise ValueError(
                "criteria baselines must not be empty: a gate without a comparison object "
                "cannot be falsified (§42)"
            )
        if any(not name for name in cleaned):
            raise ValueError("criteria baselines entries must be non-empty")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError(f"criteria baselines must be unique, got {list(cleaned)}")
        return cleaned

    @field_validator("evidence_segments")
    @classmethod
    def _segments_nonempty_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        # 段名不写成 ``Literal`` / ``SegmentName``：那样 pydantic 会先报一句
        # 「Input should be 'train', 'validation', ...」，读不到「哪些段算证据是判据形态」
        # 这层理由（与 :attr:`metric` 用 ``mode="before"`` 同一个考虑）。
        cleaned = tuple(name.strip() for name in value)
        if not cleaned:
            raise ValueError(
                "criteria evidence_segments must not be empty: which origins count as evidence "
                "is part of the criterion, it cannot be left to the implementation"
            )
        unknown = sorted(set(cleaned) - set(SEGMENT_NAMES))
        if unknown:
            raise ValueError(f"unknown segment names in evidence_segments: {unknown}")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError(f"evidence_segments must be unique, got {list(cleaned)}")
        # 规范顺序（§27 的段顺序），因此书写顺序不换 hash
        return tuple(sorted(cleaned, key=SEGMENT_NAMES.index))

    @field_validator("metric", mode="before")
    @classmethod
    def _metric_allowed(cls, value: Any) -> Any:
        """白名单在 Literal 之前校验，好让错误消息说明**为什么**这个指标不合格。

        （``mode="before"`` 是必要的：pydantic 会先做 Literal 检查，那个消息只说
        「Input should be ...」，读不到「方向不唯一 → 不可证伪」这层理由。）
        """
        if value not in GATE_METRIC_DIRECTIONS:
            raise ValueError(
                f"metric {value!r} is not gate-eligible; allowed: {list(GATE_METRICS)} "
                "(a metric without a single monotone direction cannot support a falsifiable "
                "criterion, see GATE_METRIC_DIRECTIONS)"
            )
        return value

    @field_validator("confidence_level", "max_seconds_per_origin")
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError(f"criteria value must be finite, got {value!r}")
        return value

    @model_validator(mode="after")
    def _internally_consistent(self) -> GateCriteria:
        if self.version != GATE_CRITERIA_VERSION:
            raise ValueError(
                f"unknown gate criteria version {self.version!r}; this build emits "
                f"{GATE_CRITERIA_VERSION!r}"
            )
        if self.candidate in self.baselines:
            raise ValueError(
                f"criteria baselines must not contain the candidate {self.candidate!r}; "
                "a backend cannot be its own comparison object"
            )
        axis_values = _AXIS_VALUES[self.grouping_axis]
        if self.min_groups_with_positive_increment > len(axis_values):
            raise ValueError(
                f"criteria require {self.min_groups_with_positive_increment} positive "
                f"{self.grouping_axis} groups but the axis only has {len(axis_values)} "
                f"({list(axis_values)}); an unsatisfiable criterion would be a REPLACE by "
                "construction"
            )
        if self.min_paired_samples_overall < self.min_paired_samples_per_group:
            raise ValueError(
                f"min_paired_samples_overall ({self.min_paired_samples_overall}) must be >= "
                f"min_paired_samples_per_group ({self.min_paired_samples_per_group}); a "
                "run-level threshold weaker than a group-level one is incoherent"
            )
        return self

    @property
    def higher_is_better(self) -> bool:
        return GATE_METRIC_DIRECTIONS[self.metric]

    @property
    def axis_values(self) -> tuple[str, ...]:
        """该分组轴上所有可能的取值（规范顺序），用于「每个 regime 都要有证据」的遍历。"""
        return _AXIS_VALUES[self.grouping_axis]

    def hashing_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": self.version,
            "candidate": self.candidate,
            "metric": self.metric,
            "baselines": list(self.baselines),
            "baseline_selection": self.baseline_selection,
            "grouping_axis": self.grouping_axis,
            "evidence_segments": list(self.evidence_segments),
            "min_groups_with_positive_increment": self.min_groups_with_positive_increment,
            "min_paired_samples_per_group": self.min_paired_samples_per_group,
            "min_paired_samples_overall": self.min_paired_samples_overall,
            "confidence_level": self.confidence_level,
            "bootstrap_iterations": self.bootstrap_iterations,
            "bootstrap_seed": self.bootstrap_seed,
            "max_seconds_per_origin": self.max_seconds_per_origin,
        }
        assert set(payload) == set(type(self).model_fields), (
            "GateCriteria.hashing_payload must cover every model field"
        )
        return payload

    @property
    def criteria_hash(self) -> str:
        """判据的 canonical hash（进 run metadata 与判决；注释 / 键顺序不影响）。"""
        return sha256_hex({"kind": "forecast_gate_criteria", "criteria": self.hashing_payload()})


def load_gate_criteria(path: Path | str) -> tuple[GateCriteria, str]:
    """读取 gate criteria 文档，返回 ``(criteria, raw_text)``；非法即拒绝（§42）。

    ``raw_text`` 会随 run 原样归档：canonical hash 覆盖语义，原文让人能读懂当时的判据
    （含注释里写下的理由），两者在同一次 run 里并存，事后替换必然对不上。
    """
    criteria_path = Path(path)
    if not criteria_path.is_file():
        raise ConfigurationError(
            f"gate criteria not found: {criteria_path}; §42 要求判据在任何正式 benchmark run "
            "之前预注册，缺少判据文档的 run 不允许启动"
        )
    raw_text = criteria_path.read_text(encoding="utf-8")
    try:
        payload = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"gate criteria {criteria_path} is not valid YAML: {exc}") from exc
    if payload is None:
        raise ConfigurationError(f"gate criteria {criteria_path} is empty")
    if not isinstance(payload, dict):
        raise ConfigurationError(
            f"gate criteria {criteria_path} must be a mapping, got {type(payload).__name__}"
        )
    try:
        criteria = GateCriteria.model_validate(payload)
    except Exception as exc:  # pydantic ValidationError
        raise ConfigurationError(f"invalid gate criteria {criteria_path}: {exc}") from exc
    return criteria, raw_text


class GateSliceEvidence(BaseModel):
    """一个切片（整体或某个 regime 分组）的配对证据。

    ``None`` 一律表示「没有证据」，不是 0：``sample_count < 2`` 时连区间都算不出来，
    此时指标全部为 ``None`` 且 ``adequate_evidence`` 为 ``False``。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    slice: str
    required_samples: int = Field(ge=2)
    sample_count: int = Field(ge=0)

    candidate_metric: float | None = None
    baseline_metric: float | None = None
    increment: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None

    adequate_evidence: bool

    @field_validator(
        "candidate_metric",
        "baseline_metric",
        "increment",
        "ci_lower",
        "ci_upper",
    )
    @classmethod
    def _finite(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not math.isfinite(value):
            raise ValueError(f"gate evidence values must be finite, got {value!r}")
        return value

    @model_validator(mode="after")
    def _evidence_consistent(self) -> GateSliceEvidence:
        if self.adequate_evidence != (self.sample_count >= self.required_samples):
            raise ValueError(
                "adequate_evidence must be exactly the statement "
                "sample_count >= required_samples; it is a derived fact, not a free flag"
            )
        estimates = (
            self.candidate_metric,
            self.baseline_metric,
            self.increment,
            self.ci_lower,
            self.ci_upper,
        )
        if self.sample_count < 2:
            if any(value is not None for value in estimates):
                raise ValueError(
                    f"slice {self.slice!r} has {self.sample_count} paired samples; every "
                    "estimate must be None (no interval can be computed from one sample)"
                )
            return self
        if any(value is None for value in estimates):
            raise ValueError(
                f"slice {self.slice!r} has {self.sample_count} paired samples but an estimate "
                "is missing; metrics and evidence must come from the same paired set"
            )
        assert self.ci_lower is not None and self.ci_upper is not None
        if self.ci_lower > self.ci_upper:
            raise ValueError(
                f"slice {self.slice!r} has ci_lower {self.ci_lower} > ci_upper {self.ci_upper}"
            )
        return self

    @property
    def positive(self) -> bool:
        """该切片是否支持「候选更好」：证据充分且区间下界 > 0（§42 的判据形态）。"""
        return self.adequate_evidence and self.ci_lower is not None and self.ci_lower > 0.0


class GateVerdict(BaseModel):
    """一次 gate 判决（§19 / §42），绑定判据与它判定的那份结论。

    判决是**自洽的**：它带上判据里的门槛（分组数、配对数、护栏），因此任何读到
    ``gate.json`` 的人都能自己复核「这个 GO 是不是按预注册规则推出来的」，不必去找
    当时的 criteria 文件（原文也在同一个 run 目录里）。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = GATE_VERSION
    criteria_version: str
    criteria_hash: str
    report_hash: str
    dataset_hash: str

    candidate: str
    strongest_baseline: str
    metric: str
    higher_is_better: bool
    grouping_axis: str
    #: 判决覆盖的段（§27 的段名，规范顺序）。读者不必回到判据文件才知道「这个判决数的是哪些
    #: origin」——池化 train 段与只看 test 段是两个不同的判决。
    evidence_segments: tuple[str, ...]
    available_backends: tuple[str, ...]

    confidence_level: float
    bootstrap_iterations: int
    bootstrap_seed: int
    min_groups_with_positive_increment: int = Field(ge=1)
    min_paired_samples_per_group: int = Field(ge=2)
    min_paired_samples_overall: int = Field(ge=2)
    ci_method: str = CI_METHOD
    ci_quantile_method: str = CI_QUANTILE_METHOD

    overall: GateSliceEvidence
    groups: tuple[GateSliceEvidence, ...]
    positive_groups: tuple[str, ...]

    compute_seconds_per_origin: float = Field(ge=0)
    compute_budget_seconds_per_origin: float = Field(gt=0)
    compute_within_budget: bool

    verdict: GateOutcome
    reason_codes: tuple[str, ...]

    @field_validator("criteria_hash", "report_hash", "dataset_hash")
    @classmethod
    def _hashes(cls, value: str) -> str:
        if not is_sha256_hex(value):
            raise ValueError(f"expected a sha256 hex digest, got {value!r}")
        return value

    @field_validator("version")
    @classmethod
    def _version_matches(cls, value: str) -> str:
        if value != GATE_VERSION:
            raise ValueError(f"unknown gate version {value!r}; this build emits {GATE_VERSION!r}")
        return value

    @field_validator("criteria_version")
    @classmethod
    def _criteria_version_matches(cls, value: str) -> str:
        if value != GATE_CRITERIA_VERSION:
            raise ValueError(
                f"unknown gate criteria version {value!r}; this build emits "
                f"{GATE_CRITERIA_VERSION!r}"
            )
        return value

    @field_validator("metric")
    @classmethod
    def _metric_known(cls, value: str) -> str:
        if value not in GATE_METRIC_DIRECTIONS:
            raise ValueError(f"unknown gate metric {value!r}; allowed: {list(GATE_METRICS)}")
        return value

    @field_validator("grouping_axis")
    @classmethod
    def _axis_known(cls, value: str) -> str:
        if value not in _AXIS_VALUES:
            raise ValueError(f"unknown grouping axis {value!r}; allowed: {sorted(_AXIS_VALUES)}")
        return value

    @field_validator("ci_method")
    @classmethod
    def _ci_method_known(cls, value: str) -> str:
        if value != CI_METHOD:
            raise ValueError(f"unknown ci_method {value!r}; this build emits {CI_METHOD!r}")
        return value

    @field_validator("ci_quantile_method")
    @classmethod
    def _ci_quantile_method_known(cls, value: str) -> str:
        if value != CI_QUANTILE_METHOD:
            raise ValueError(
                f"unknown ci_quantile_method {value!r}; this build emits {CI_QUANTILE_METHOD!r}"
            )
        return value

    @field_validator("evidence_segments")
    @classmethod
    def _segments_known(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("evidence_segments must not be empty")
        unknown = sorted(set(value) - set(SEGMENT_NAMES))
        if unknown:
            raise ValueError(f"unknown segments in evidence_segments: {unknown}")
        if tuple(sorted(value, key=SEGMENT_NAMES.index)) != tuple(value):
            raise ValueError("evidence_segments must be in the canonical segment order")
        return value

    @field_validator("higher_is_better")
    @classmethod
    def _direction_known(cls, value: bool) -> bool:
        if not isinstance(value, bool):
            raise ValueError(f"higher_is_better must be a bool, got {value!r}")
        return value

    @field_validator(
        "confidence_level", "compute_seconds_per_origin", "compute_budget_seconds_per_origin"
    )
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError(f"gate verdict values must be finite, got {value!r}")
        return value

    @field_validator("reason_codes")
    @classmethod
    def _known_codes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        unknown = sorted(set(value) - set(_GATE_REASON_CODES))
        if unknown:
            raise ValueError(f"unknown gate reason codes: {unknown}")
        if len(set(value)) != len(value):
            raise ValueError(f"gate reason codes must be unique, got {list(value)}")
        if tuple(sorted(value)) != tuple(value):
            raise ValueError("gate reason codes must be sorted for a diff-friendly artifact")
        if not value:
            raise ValueError("a verdict must carry at least one reason code")
        return value

    @model_validator(mode="after")
    def _verdict_follows_evidence(self) -> GateVerdict:
        """判决必须**由证据推出**：结论是证据的函数，不允许手工拼装。

        这条不变量让「跑完之后改一个字段把 REPLACE 写成 GO」在构造期就失败。
        """
        derived_positive = tuple(group.slice for group in self.groups if group.positive)
        if derived_positive != self.positive_groups:
            raise ValueError(
                f"positive_groups {list(self.positive_groups)} must be exactly the adequate "
                f"groups whose CI lower bound > 0 ({list(derived_positive)})"
            )
        if self.compute_within_budget != (
            self.compute_seconds_per_origin <= self.compute_budget_seconds_per_origin
        ):
            raise ValueError(
                "compute_within_budget must be exactly "
                "compute_seconds_per_origin <= compute_budget_seconds_per_origin"
            )
        if self.higher_is_better != GATE_METRIC_DIRECTIONS[self.metric]:
            raise ValueError(
                f"higher_is_better {self.higher_is_better} contradicts the direction whitelist "
                f"for metric {self.metric!r} ({GATE_METRIC_DIRECTIONS[self.metric]})"
            )
        overall_positive = self.overall.positive
        groups_meet_min = len(derived_positive) >= self.min_groups_with_positive_increment
        if overall_positive and groups_meet_min and self.compute_within_budget:
            expected: GateOutcome = "GO"
        elif overall_positive or derived_positive:
            expected = "CONDITIONAL"
        else:
            expected = "REPLACE"
        if self.verdict != expected:
            raise ValueError(
                f"verdict {self.verdict!r} does not follow from the evidence: overall positive="
                f"{overall_positive}, positive groups={list(derived_positive)}, "
                f"compute_within_budget={self.compute_within_budget} imply {expected!r}"
            )
        expected_codes = _reason_codes(
            overall_positive=overall_positive,
            positive_groups=derived_positive,
            min_groups=self.min_groups_with_positive_increment,
            compute_within_budget=self.compute_within_budget,
        )
        if tuple(self.reason_codes) != expected_codes:
            raise ValueError(
                f"reason_codes {list(self.reason_codes)} do not describe this evidence; the "
                f"codes follow from the same facts as the verdict, so they must be "
                f"{list(expected_codes)}"
            )
        return self

    @model_validator(mode="after")
    def _identity_fields_consistent(self) -> GateVerdict:
        """比较对象与分组名也要自洽（手工拼装的 ``gate.json`` 在这些地方同样立刻失败）。

        `evaluate_gate` 不可能产出这些组合（候选与比较对象来自本次 run、分组名来自判据的
        分组轴），但「判决字段之间互相矛盾」不该只在**某些**字段上被挡住——ADR-024 §7 的
        清单以这里为准。刻意**不**校验的三件事：``evidence_segments``（它是判据带来的输入，
        判决字段之间推不出它）、点估计是否落在自己的区间内（百分位区间不保证点估计一定在内，
        当不变量会在极端样本上误拒）、``required_samples`` 与门槛字段的对应关系。
        """
        if self.candidate not in self.available_backends:
            raise ValueError(
                f"candidate {self.candidate!r} is not among available_backends "
                f"{list(self.available_backends)}; a backend that did not run cannot be judged"
            )
        if self.strongest_baseline == self.candidate:
            raise ValueError(
                "strongest_baseline must not be the candidate itself; a backend cannot be its "
                "own comparison object"
            )
        if self.strongest_baseline not in self.available_backends:
            raise ValueError(
                f"strongest_baseline {self.strongest_baseline!r} is not among available_backends "
                f"{list(self.available_backends)}; a comparison object that did not run cannot "
                "support a comparison"
            )
        axis_values = _AXIS_VALUES[self.grouping_axis]
        slices = tuple(group.slice for group in self.groups)
        if slices != axis_values:
            raise ValueError(
                f"groups must be exactly the {self.grouping_axis} axis values "
                f"{list(axis_values)} in canonical order, got {list(slices)}"
            )
        return self

    def hashing_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": self.version,
            "criteria_version": self.criteria_version,
            "criteria_hash": self.criteria_hash,
            "report_hash": self.report_hash,
            "dataset_hash": self.dataset_hash,
            "candidate": self.candidate,
            "strongest_baseline": self.strongest_baseline,
            "metric": self.metric,
            "higher_is_better": self.higher_is_better,
            "grouping_axis": self.grouping_axis,
            "evidence_segments": list(self.evidence_segments),
            "available_backends": list(self.available_backends),
            "confidence_level": self.confidence_level,
            "bootstrap_iterations": self.bootstrap_iterations,
            "bootstrap_seed": self.bootstrap_seed,
            "min_groups_with_positive_increment": self.min_groups_with_positive_increment,
            "min_paired_samples_per_group": self.min_paired_samples_per_group,
            "min_paired_samples_overall": self.min_paired_samples_overall,
            "ci_method": self.ci_method,
            "ci_quantile_method": self.ci_quantile_method,
            "overall": self.overall.model_dump(mode="json"),
            "groups": [group.model_dump(mode="json") for group in self.groups],
            "positive_groups": list(self.positive_groups),
            "compute_seconds_per_origin": self.compute_seconds_per_origin,
            "compute_budget_seconds_per_origin": self.compute_budget_seconds_per_origin,
            "compute_within_budget": self.compute_within_budget,
            "verdict": self.verdict,
            "reason_codes": list(self.reason_codes),
        }
        assert set(payload) == set(type(self).model_fields), (
            "GateVerdict.hashing_payload must cover every model field"
        )
        return payload

    @property
    def gate_hash(self) -> str:
        """判决内容的 sha256：同一次 run + 同一份判据必须给出同一个判决身份。"""
        return sha256_hex({"kind": "forecast_gate_verdict", "verdict": self.hashing_payload()})


@dataclass(frozen=True)
class PairedObservation:
    """同一 origin 上候选与 baseline 的配对观测（配对是 CI 口径的前提）。"""

    symbol: str
    market_date: date
    candidate_value: float
    baseline_value: float


def _nearest_rank(sorted_values: Sequence[float], q: float) -> float:
    """最近秩法分位（有符号）：``sorted_values`` 升序，``0 < q <= 1``。

    与 :func:`kronos_ai.evaluation.compute_metrics.percentile` 同一套取值约定，但允许
    负值（bootstrap 的增量区间本来就可以跨 0），因此不共用那个要求非负的实现。
    """
    if not sorted_values:
        raise ConfigurationError("nearest_rank needs at least one value")
    if not 0 < q <= 1:
        raise ConfigurationError(f"nearest_rank quantile must be in (0, 1], got {q}")
    rank = max(1, math.ceil(q * len(sorted_values)))
    return sorted_values[min(rank, len(sorted_values)) - 1]


def _slice_seed(seed: int, slice_name: str) -> int:
    """每个切片一个独立且确定的 RNG 种子：切片的区间不依赖遍历顺序。"""
    digest = hashlib.sha256(f"{seed}:{slice_name}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _aggregate(values: Sequence[float], indices: Sequence[int], metric: str) -> float:
    """在重采样下标上聚合一个 backend 的逐记录取值。

    ``rmse`` 的逐记录取值是**平方误差**，因此聚合是 ``sqrt(mean(.))``：增量必须在
    「先聚合、再相减」的口径下比较（``rmse_candidate - rmse_baseline``），而不是把逐条
    平方误差之差平均起来——后者是 MSE 之差，不是 RMSE 之差。
    """
    total = math.fsum(values[index] for index in indices) / len(indices)
    return math.sqrt(total) if metric == "rmse" else total


def _oriented_increment(candidate: float, baseline: float, *, higher_is_better: bool) -> float:
    """方向归一化后的增量：正数**永远**表示候选更好。"""
    return candidate - baseline if higher_is_better else baseline - candidate


def _bootstrap_interval(
    observations: Sequence[PairedObservation],
    *,
    metric: str,
    higher_is_better: bool,
    confidence_level: float,
    iterations: int,
    seed: int,
    slice_name: str,
) -> tuple[float, float]:
    """增量均值的配对 bootstrap 百分位区间（含端点，最近秩法）。

    重采样单位是 **origin**（而不是单条记录），且每个 replicate 用**同一组下标**同时
    聚合两侧——这正是「配对」的含义；按记录独立重采样会把 symbol / 时间上的重复计成
    独立证据，得到的区间不再回答「同一批时点上候选是否更好」。
    """
    rng = random.Random(_slice_seed(seed, slice_name))
    candidate_values = [observation.candidate_value for observation in observations]
    baseline_values = [observation.baseline_value for observation in observations]
    count = len(observations)
    replicates: list[float] = []
    for _ in range(iterations):
        indices = [rng.randrange(count) for _ in range(count)]
        replicates.append(
            _oriented_increment(
                _aggregate(candidate_values, indices, metric),
                _aggregate(baseline_values, indices, metric),
                higher_is_better=higher_is_better,
            )
        )
    replicates.sort()
    alpha = 1.0 - confidence_level
    return (
        _nearest_rank(replicates, alpha / 2.0),
        _nearest_rank(replicates, 1.0 - alpha / 2.0),
    )


def _metric_value(record: ForecastEvalRecord, metric: str) -> float | None:
    """单条记录的指标取值；无定义时返回 ``None``（不做任何填补）。"""
    if record.label_status != "LABELED":
        return None
    if metric == "direction_accuracy":
        if record.realized_direction is None:
            return None
        return 1.0 if record.predicted_direction == record.realized_direction else 0.0
    if record.realized_return is None:
        return None
    error = record.predicted_return - record.realized_return
    if metric == "mae":
        return abs(error)
    if metric == "rmse":
        return error * error
    raise ConfigurationError(f"metric {metric!r} has no per-record definition")


def _pick_group_value(record: ForecastEvalRecord, axis: str) -> str:
    if axis == "trend":
        return record.trend_regime
    if axis == "volatility":
        return record.volatility_regime
    raise ConfigurationError(f"unknown grouping axis {axis!r}")


def _slice_records(
    result: ForecastBenchmarkResult,
    *,
    segments: Sequence[str],
    axis: str,
    group_value: str | None,
) -> list[ForecastEvalRecord]:
    """判据声明的**证据切片**里的记录（段 + 可选的 regime 分组）。

    段过滤在这里、而不是在指标聚合里：判决的样本量必须与 ``evidence_segments`` 一致，
    否则「这个判决数的是哪些 origin」就要靠读实现来回答（那正是判据形态必须冻结的东西）。
    """
    allowed = set(segments)
    selected = [record for record in result.records if record.segment in allowed]
    if group_value is None:
        return selected
    return [record for record in selected if _pick_group_value(record, axis) == group_value]


def _reject_duplicate_origins(records: Sequence[ForecastEvalRecord]) -> None:
    """记录集合必须自洽：``(backend, symbol, market_date)`` 只能出现一次。

    检查**跨全部段**地做，而不是只查判决的那一段：段是互斥的时间切分，同一个 origin 出现
    在两个段里本身就是记录集合的错；而一旦有重复，任何派生量（配对、区间、分组样本量）都
    会随「取哪一条」变化——那是实现细节，不是判据。重复的判定**先于** label 过滤：一条
    ``LABELED`` 加一条 ``SUSPENDED`` 同样是重复，不能靠「哪条有指标值」来决定用哪条。
    """
    seen: dict[tuple[str, str, date], str] = {}
    for record in records:
        key = (record.backend, record.symbol, record.market_date)
        first = seen.get(key)
        if first is not None:
            raise ConfigurationError(
                f"backend {record.backend!r} has more than one record for "
                f"{record.symbol}@{record.market_date} (segments {first!r} and "
                f"{record.segment!r}); paired evidence requires exactly one record per backend "
                "and origin"
            )
        seen[key] = record.segment
    return None


def _pair(
    records: Sequence[ForecastEvalRecord],
    *,
    candidate: str,
    baseline: str,
    metric: str,
) -> tuple[PairedObservation, ...]:
    """把切片里的记录按 ``(symbol, market_date)`` 配对（§42 的配对 bootstrap 前提）。

    - 只有 ``LABELED`` 的记录参与：停牌 / 数据末端不是「0 收益」（§49 / ADR-023 §3）；
    - 缺一边的 origin **不进配对**（不补 0、不取中位数），但配对数量会写进证据里，
      因此「有多少 origin 真的被比较过」永远可见；
    - 同一 ``(backend, origin)`` 的重复记录由 :func:`_reject_duplicate_origins` 在判决入口
      一次性拒掉（跨段检查，含 ``SUSPENDED`` 一类无指标值的记录）。
    """
    candidate_values: dict[tuple[str, date], float] = {}
    baseline_values: dict[tuple[str, date], float] = {}
    for record in records:
        if record.backend not in (candidate, baseline):
            continue
        value = _metric_value(record, metric)
        if value is None:
            continue
        target = candidate_values if record.backend == candidate else baseline_values
        target[(record.symbol, record.market_date)] = value
    paired: list[PairedObservation] = []
    for key in sorted(
        set(candidate_values) & set(baseline_values), key=lambda item: (item[1], item[0])
    ):
        paired.append(
            PairedObservation(
                symbol=key[0],
                market_date=key[1],
                candidate_value=candidate_values[key],
                baseline_value=baseline_values[key],
            )
        )
    return tuple(paired)


def _slice_evidence(
    observations: Sequence[PairedObservation],
    *,
    slice_name: str,
    required_samples: int,
    criteria: GateCriteria,
) -> GateSliceEvidence:
    if len(observations) < 2:
        return GateSliceEvidence(
            slice=slice_name,
            required_samples=required_samples,
            sample_count=len(observations),
            adequate_evidence=False,
        )
    higher_is_better = criteria.higher_is_better
    candidate_values = [observation.candidate_value for observation in observations]
    baseline_values = [observation.baseline_value for observation in observations]
    every_index = list(range(len(observations)))
    candidate_metric = _aggregate(candidate_values, every_index, criteria.metric)
    baseline_metric = _aggregate(baseline_values, every_index, criteria.metric)
    increment = _oriented_increment(
        candidate_metric, baseline_metric, higher_is_better=higher_is_better
    )
    ci_lower, ci_upper = _bootstrap_interval(
        observations,
        metric=criteria.metric,
        higher_is_better=higher_is_better,
        confidence_level=criteria.confidence_level,
        iterations=criteria.bootstrap_iterations,
        seed=criteria.bootstrap_seed,
        slice_name=slice_name,
    )
    return GateSliceEvidence(
        slice=slice_name,
        required_samples=required_samples,
        sample_count=len(observations),
        candidate_metric=candidate_metric,
        baseline_metric=baseline_metric,
        increment=increment,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        adequate_evidence=len(observations) >= required_samples,
    )


def _select_strongest_baseline(result: ForecastBenchmarkResult, criteria: GateCriteria) -> str:
    """按**同一指标**在**预注册的证据切片**上选最强 baseline（§42：比较对象的选法必须在跑前定下）。

    选法与判决必须用同一批证据段：判据说「相对最强 baseline 更好」，如果「最强」是在另一批
    样本（例如把 train 池进来）上算出来的称号，判决里的比较对象就未必是本次证据上最强的那个。
    证据切片在 run 前冻结，因此这里不需要额外的自由度。

    聚合口径是「每个 baseline 自己在这批证据里的 ``LABELED`` 记录」：与候选的**配对**是判决与
    区间的口径（选出的对象再与候选逐 origin 配对），不是选比较对象的口径。

    预注册的 baseline 里只要有一个在证据切片上没有可比的 ``LABELED`` 记录，就不判决：比较
    集合不完整时「最强」没有定义，这是缺证据，不是「候选更好」。
    """
    slice_records = _slice_records(
        result, segments=criteria.evidence_segments, axis=criteria.grouping_axis, group_value=None
    )
    scored: list[tuple[float, str]] = []
    absent: list[str] = []
    for name in criteria.baselines:
        values = [
            value
            for record in slice_records
            if record.backend == name
            for value in (_metric_value(record, criteria.metric),)
            if value is not None
        ]
        if not values:
            absent.append(name)
            continue
        scored.append((_aggregate(values, range(len(values)), criteria.metric), name))
    if absent:
        raise InsufficientEvidenceError(
            f"pre-registered baselines {absent} have no LABELED record in the evidence segments "
            f"{list(criteria.evidence_segments)}; the comparison set is incomplete, so a "
            f"{criteria.metric} comparison against it is missing evidence, not a REPLACE"
        )
    higher_is_better = criteria.higher_is_better
    oriented = [(-value if higher_is_better else value, name) for value, name in scored]
    oriented.sort()
    return oriented[0][1]


def _reason_codes(
    *,
    overall_positive: bool,
    positive_groups: Sequence[str],
    min_groups: int,
    compute_within_budget: bool,
) -> tuple[str, ...]:
    codes = {
        "OVERALL_INCREMENT_CI_LOWER_BOUND_POSITIVE"
        if overall_positive
        else "OVERALL_INCREMENT_CI_LOWER_BOUND_NOT_POSITIVE",
        "REGIMES_WITH_POSITIVE_INCREMENT_MEETS_MIN"
        if len(positive_groups) >= min_groups
        else "REGIMES_WITH_POSITIVE_INCREMENT_BELOW_MIN",
        "COMPUTE_BUDGET_WITHIN_LIMIT" if compute_within_budget else "COMPUTE_BUDGET_EXCEEDED",
    }
    if not positive_groups:
        codes.add("NO_REGIME_SHOWS_POSITIVE_INCREMENT")
    return tuple(sorted(codes))


def evaluate_gate(result: ForecastBenchmarkResult, criteria: GateCriteria) -> GateVerdict:
    """按预注册判据评估一次 benchmark run（§19 / §42）。

    失败语义：候选或比较对象不在本次 run 的 backend 里、记录自相矛盾（同一 origin 有两条
    记录）→ :class:`ConfigurationError`；配对证据低于预注册门槛 →
    :class:`InsufficientEvidenceError`（「没测出来」不等于「更差」）。
    判据本身不可满足（例如要求 3 个正增量分组，而候选连 3 个分组的证据都不够）同样属于
    证据不足，而不是 REPLACE。
    """
    # 记录集合自洽性先查：重复的 (backend, origin) 会让下列每个派生量都随「取哪一条」变化，
    # 而「取哪一条」不在判据里。跨段检查（不只看证据段），理由见 _reject_duplicate_origins。
    _reject_duplicate_origins(result.records)
    if criteria.candidate not in result.backends:
        raise ConfigurationError(
            f"gate candidate {criteria.candidate!r} is not among the run's backends "
            f"{list(result.backends)}; judging a backend that did not run is not a comparison"
        )
    missing = sorted(set(criteria.baselines) - set(result.backends))
    if missing:
        raise ConfigurationError(
            f"criteria baselines {missing} are not among the run's backends {list(result.backends)}"
        )
    # 判据声明的段必须在本次 run 里真的存在：段名写错会静默缩小证据（甚至只留空切片），
    # 那是「判据说一套、判决做一套」。missing 段显式失败，而不是给出一个更弱的判决。
    # 这一条必须在**选比较对象之前**查：判据指向一个本次 run 里不存在的段属于配置与 run
    # 不匹配（ConfigurationError），而不是「缺证据」。
    present_segments = {record.segment for record in result.records}
    absent = [name for name in criteria.evidence_segments if name not in present_segments]
    if absent:
        raise ConfigurationError(
            f"criteria evidence_segments {absent} have no records in this run "
            f"(present: {sorted(present_segments)}); the judgement would silently rest on a "
            "different evidence slice than the pre-registered one"
        )

    strongest = _select_strongest_baseline(result, criteria)
    evidence_segments = criteria.evidence_segments
    overall_observations = _pair(
        _slice_records(
            result, segments=evidence_segments, axis=criteria.grouping_axis, group_value=None
        ),
        candidate=criteria.candidate,
        baseline=strongest,
        metric=criteria.metric,
    )
    if len(overall_observations) < criteria.min_paired_samples_overall:
        raise InsufficientEvidenceError(
            f"paired evidence for {criteria.candidate} vs {strongest} is "
            f"{len(overall_observations)} origins, below the pre-registered minimum "
            f"{criteria.min_paired_samples_overall}; a verdict here would read as a result "
            "while it is a lack of evidence — collect more origins instead of loosening the "
            "criteria (criteria_hash "
            f"{criteria.criteria_hash})"
        )
    overall = _slice_evidence(
        overall_observations,
        slice_name="all",
        required_samples=criteria.min_paired_samples_overall,
        criteria=criteria,
    )

    groups = tuple(
        _slice_evidence(
            _pair(
                _slice_records(
                    result,
                    segments=evidence_segments,
                    axis=criteria.grouping_axis,
                    group_value=value,
                ),
                candidate=criteria.candidate,
                baseline=strongest,
                metric=criteria.metric,
            ),
            slice_name=value,
            required_samples=criteria.min_paired_samples_per_group,
            criteria=criteria,
        )
        for value in criteria.axis_values
    )
    with_evidence = [group for group in groups if group.adequate_evidence]
    if len(with_evidence) < criteria.min_groups_with_positive_increment:
        raise InsufficientEvidenceError(
            f"only {len(with_evidence)} {criteria.grouping_axis} group(s) reach the "
            f"pre-registered minimum of {criteria.min_paired_samples_per_group} paired "
            f"origins (needed: {criteria.min_groups_with_positive_increment} groups with "
            f"evidence); groups={[(group.slice, group.sample_count) for group in groups]}. "
            "没有足够的分组证据就不判决：这是缺证据，不是 REPLACE（§42 要求判据可证伪）"
        )

    candidate_latencies = [
        record.latency_ms for record in result.records if record.backend == criteria.candidate
    ]
    # 走到这里时候选至少有 2 条配对过的记录，所以这个分支通常不可达；保留它是为了让
    # 「候选一条记录都没有」有明确的失败语义，而不是除零错误（那是实现细节的崩溃，
    # 不是「缺证据」）。
    if not candidate_latencies:
        raise InsufficientEvidenceError(
            f"the candidate {criteria.candidate!r} has no records, so its compute cost cannot "
            "be observed against the pre-registered budget"
        )
    compute_seconds_per_origin = math.fsum(candidate_latencies) / len(candidate_latencies) / 1000.0
    compute_within_budget = compute_seconds_per_origin <= criteria.max_seconds_per_origin

    positive_groups = tuple(group.slice for group in groups if group.positive)
    overall_positive = overall.positive
    if (
        overall_positive
        and len(positive_groups) >= criteria.min_groups_with_positive_increment
        and compute_within_budget
    ):
        verdict: GateOutcome = "GO"
    elif overall_positive or positive_groups:
        verdict = "CONDITIONAL"
    else:
        verdict = "REPLACE"

    return GateVerdict(
        criteria_version=criteria.version,
        criteria_hash=criteria.criteria_hash,
        report_hash=result.report_hash,
        dataset_hash=result.dataset_hash,
        candidate=criteria.candidate,
        strongest_baseline=strongest,
        metric=criteria.metric,
        higher_is_better=criteria.higher_is_better,
        grouping_axis=criteria.grouping_axis,
        evidence_segments=tuple(evidence_segments),
        available_backends=tuple(sorted(result.backends)),
        confidence_level=criteria.confidence_level,
        bootstrap_iterations=criteria.bootstrap_iterations,
        bootstrap_seed=criteria.bootstrap_seed,
        min_groups_with_positive_increment=criteria.min_groups_with_positive_increment,
        min_paired_samples_per_group=criteria.min_paired_samples_per_group,
        min_paired_samples_overall=criteria.min_paired_samples_overall,
        overall=overall,
        groups=groups,
        positive_groups=positive_groups,
        compute_seconds_per_origin=compute_seconds_per_origin,
        compute_budget_seconds_per_origin=criteria.max_seconds_per_origin,
        compute_within_budget=compute_within_budget,
        verdict=verdict,
        reason_codes=_reason_codes(
            overall_positive=overall_positive,
            positive_groups=positive_groups,
            min_groups=criteria.min_groups_with_positive_increment,
            compute_within_budget=compute_within_budget,
        ),
    )


def _slice_evidence_payload(evidence: GateSliceEvidence) -> dict[str, Any]:
    return {
        **evidence.model_dump(mode="json"),
        "positive": evidence.positive,
    }


def gate_payload(verdict: GateVerdict) -> dict[str, Any]:
    """``gate.json`` 的内容（§33 的 artifact 之一）。

    每个数字都能追到证据与判据：``criteria_hash`` 指向归档的判据原文，``report_hash``
    指向被判决的那份结论，``gate_hash`` 是判决自身身份。
    """
    return {
        "kind": "forecast_gate_verdict",
        "gate_version": verdict.version,
        "gate_hash": verdict.gate_hash,
        "criteria_version": verdict.criteria_version,
        "criteria_hash": verdict.criteria_hash,
        "report_hash": verdict.report_hash,
        "dataset_hash": verdict.dataset_hash,
        "candidate": verdict.candidate,
        "strongest_baseline": verdict.strongest_baseline,
        "available_backends": list(verdict.available_backends),
        "metric": verdict.metric,
        "higher_is_better": verdict.higher_is_better,
        "grouping_axis": verdict.grouping_axis,
        "evidence_segments": list(verdict.evidence_segments),
        "ci_method": verdict.ci_method,
        "ci_quantile_method": verdict.ci_quantile_method,
        "confidence_level": verdict.confidence_level,
        "bootstrap_iterations": verdict.bootstrap_iterations,
        "bootstrap_seed": verdict.bootstrap_seed,
        "min_groups_with_positive_increment": verdict.min_groups_with_positive_increment,
        "min_paired_samples_per_group": verdict.min_paired_samples_per_group,
        "min_paired_samples_overall": verdict.min_paired_samples_overall,
        "overall": _slice_evidence_payload(verdict.overall),
        "groups": [_slice_evidence_payload(group) for group in verdict.groups],
        "positive_groups": list(verdict.positive_groups),
        "compute_seconds_per_origin": verdict.compute_seconds_per_origin,
        "compute_budget_seconds_per_origin": verdict.compute_budget_seconds_per_origin,
        "compute_within_budget": verdict.compute_within_budget,
        "verdict": verdict.verdict,
        "reason_codes": list(verdict.reason_codes),
    }


def render_gate_markdown(verdict: GateVerdict) -> list[str]:
    """``report.md`` 的 Gate 段落（§42）：判决 + 证据表 + 理由，全部来自判决对象。"""
    lines: list[str] = [
        "## §19 / §42 Gate — Go / Replace",
        "",
        f"- verdict: **{verdict.verdict}** — {_OUTCOME_PROSE[verdict.verdict]}",
        f"- candidate: {verdict.candidate}",
        f"- strongest pre-registered baseline (by {verdict.metric}): {verdict.strongest_baseline}",
        f"- metric: {verdict.metric} ({'higher' if verdict.higher_is_better else 'lower'} is better)",
        f"- grouping_axis: {verdict.grouping_axis}",
        f"- evidence_segments: {', '.join(verdict.evidence_segments)}",
        f"- ci_method: {verdict.ci_method} ({verdict.ci_quantile_method}), "
        f"level={verdict.confidence_level}, iterations={verdict.bootstrap_iterations}, "
        f"seed={verdict.bootstrap_seed}",
        f"- criteria_hash: `{verdict.criteria_hash}`",
        f"- gate_hash: `{verdict.gate_hash}`",
        f"- report_hash: `{verdict.report_hash}`",
        f"- available_backends: {', '.join(verdict.available_backends)}",
        "",
        f"Increment = {'candidate − baseline' if verdict.higher_is_better else 'baseline − candidate'}"
        "（正数一律表示候选更好）；`insufficient` 表示该切片的配对样本数低于预注册门槛，"
        "不参与判决。",
        "",
        "| slice | paired n | required | candidate | baseline | increment | CI low | CI high | positive |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    slices = (verdict.overall, *verdict.groups)
    for evidence in slices:
        if evidence.sample_count < 2:
            cells = " | ".join("n/a" for _ in range(5))
        else:
            assert evidence.candidate_metric is not None
            assert evidence.baseline_metric is not None
            assert evidence.increment is not None
            assert evidence.ci_lower is not None
            assert evidence.ci_upper is not None
            cells = " | ".join(
                f"{value:.6f}"
                for value in (
                    evidence.candidate_metric,
                    evidence.baseline_metric,
                    evidence.increment,
                    evidence.ci_lower,
                    evidence.ci_upper,
                )
            )
        lines.append(
            f"| {evidence.slice} | {evidence.sample_count} | {evidence.required_samples} | "
            f"{cells} | "
            f"{'yes' if evidence.positive else ('insufficient' if not evidence.adequate_evidence else 'no')} |"
        )
    lines += [
        "",
        "| compute (candidate) | observed s/origin | budget s/origin | within budget |",
        "|---|---|---|---|",
        f"| {verdict.candidate} | {verdict.compute_seconds_per_origin:.6f} | "
        f"{verdict.compute_budget_seconds_per_origin:.6f} | {verdict.compute_within_budget} |",
        "",
        "理由（机器可读码 → 读法）：",
        "",
    ]
    for code in verdict.reason_codes:
        lines.append(f"- `{code}` — {_REASON_PROSE[code]}")
    lines.append("")
    return lines


def summarise_gate(verdict: GateVerdict) -> str:
    """CLI 一行的 gate 摘要（判决 + 它相对谁）。"""
    return (
        f"gate {verdict.verdict} — {verdict.candidate} vs {verdict.strongest_baseline} "
        f"({verdict.metric}, {len(verdict.positive_groups)}/"
        f"{len(verdict.groups)} {verdict.grouping_axis} groups positive, "
        f"compute {'ok' if verdict.compute_within_budget else 'over budget'})"
    )


__all__ = [
    "CI_METHOD",
    "CI_QUANTILE_METHOD",
    "GATE_CRITERIA_VERSION",
    "GATE_METRICS",
    "GATE_METRIC_DIRECTIONS",
    "GATE_VERSION",
    "GateCriteria",
    "GateSliceEvidence",
    "GateVerdict",
    "evaluate_gate",
    "gate_payload",
    "load_gate_criteria",
    "render_gate_markdown",
    "summarise_gate",
]
