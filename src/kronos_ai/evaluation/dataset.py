"""Walk-forward dataset（基线文档 §27 / §28 / §29 / §32，ADR-022）。

一次 benchmark 的样本集由三样东西唯一确定：

1. **时间轴**：升序 market session 序列（``TradingCalendar`` 是唯一真源，§6.6）；
2. **分段计划**：每段多少 session（:mod:`kronos_ai.evaluation.walk_forward`）；
3. **LabelPolicy**：label 语义与 embargo 的单一真源（§28）。

三者一起进入 ``dataset_hash``：换 label 定义、换 embargo、换分段都会产生新的
dataset_hash。``embargo_sessions`` 必须在 Phase 2 一次性固化（§27）——事后补 embargo
会让全部历史 dataset_hash 失效——所以它只作为版本化常量出现在
:class:`LabelPolicy` 里，不散落在代码中，也不来自 experiment config 的临时覆盖。

``horizon_sessions`` 与 ForecastSample 的时间轴同源（§11）：origin 的未来 label 横跨
其后的 ``horizon_sessions`` 个 **market session**。个股停牌不改变时间轴，只在 label 侧
显式标记 ``SUSPENDED`` / ``INSUFFICIENT_FUTURE_BARS``（对应本模块的
``suspension_policy`` / ``missing_future_bars_policy``）。

本模块不加载行情、不构造 label 数值：它产出的是「哪些 origin、在哪一段、label 窗口
覆盖哪些 session」，以及把这些固定下来的 ``dataset_hash``。数据不足
``lookback_bars`` 的 origin 在构造期就被拒绝（§17 / §19 的可比性前提，RX-KAI-016）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from kronos_ai.data.calendar import Exchange, TradingCalendar
from kronos_ai.domain.hashing import sha256_hex
from kronos_ai.domain.market import MarketBar, ShanghaiDatetime
from kronos_ai.domain.symbols import validate_normalized_symbol
from kronos_ai.domain.time import (
    KnowledgeCutoffPolicy,
    SessionDate,
    cutoff_policy_record,
    resolve_knowledge_cutoff,
)
from kronos_ai.errors import ConfigurationError, DataQualityError, LeakageError
from kronos_ai.evaluation.walk_forward import (
    WALK_FORWARD_VERSION,
    SegmentName,
    SegmentSessions,
    WalkForwardPlan,
    require_no_leakage,
    split_sessions,
)

LABEL_POLICY_VERSION = "label-policy-v1"
FORECAST_DATASET_VERSION = "walk-forward-dataset-v1"

DEFAULT_HORIZON_SESSIONS = 5
DEFAULT_EMBARGO_SESSIONS = 5

SuspensionPolicy = Literal["mark_suspended"]
MissingFutureBarsPolicy = Literal["mark_insufficient"]
PriceField = Literal["close"]
ReturnDefinition = Literal["close_to_close_simple"]

DEFAULT_SUSPENSION_POLICY: SuspensionPolicy = "mark_suspended"
DEFAULT_MISSING_FUTURE_BARS_POLICY: MissingFutureBarsPolicy = "mark_insufficient"
DEFAULT_PRICE_FIELD: PriceField = "close"
DEFAULT_RETURN_DEFINITION: ReturnDefinition = "close_to_close_simple"

DirectionLabel = Literal["BEARISH", "NEUTRAL", "BULLISH"]
#: §24.1 第一版 Decision Schema 的 Direction 词汇（``UNKNOWN`` 属于决策层「证据不足」，
#: 不是 label 的一类：label 侧用 ``direction is None`` 表达）。
DIRECTION_LABELS: tuple[DirectionLabel, ...] = ("BEARISH", "NEUTRAL", "BULLISH")


class DirectionThreshold(BaseModel):
    """方向 label 的一类（§28 ``direction_thresholds`` 的一项）。

    ``threshold`` 是该 label 的**闭下界**：收益 ``r`` 的 label 是满足
    ``threshold is None or r >= threshold`` 的**最后一项**。第一项必须是
    ``threshold=None``（无下界），于是三分类可以写成不需要 ``-inf`` 哨兵值的显式列表：

    ```text
    BEARISH  threshold=None    #            r < -0.02
    NEUTRAL  threshold=-0.02   # -0.02   <= r <  0.02
    BULLISH  threshold=0.02    #  0.02   <= r
    ```

    用 frozen model 而不是裸 ``dict``：``LabelPolicy`` 是 dataset_hash 的输入，
    可变容器会让「已哈希的策略」在事后被就地改写。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    threshold: float | None

    @field_validator("label")
    @classmethod
    def _label_token(cls, value: str) -> str:
        if not value or not value.isascii() or not value.replace("_", "").isalnum():
            raise ValueError(
                f"label must be an ascii token like 'BEARISH' / 'NEUTRAL', got {value!r}"
            )
        return value

    @field_validator("threshold")
    @classmethod
    def _threshold_finite(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"threshold must be finite, got {value}")
        return value

    def hashing_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"label": self.label, "threshold": self.threshold}
        # 与 LabelPolicy.hashing_payload 同一套完备性约定：加字段忘改这里会让 hash 静默失真。
        assert set(payload) == set(type(self).model_fields), (
            "DirectionThreshold.hashing_payload must cover every model field"
        )
        return payload


#: §13 的分布阈值词汇（±2%）就是方向边界的默认值：同一个平台不应该有两套「涨跌」口径。
DEFAULT_DIRECTION_THRESHOLDS: tuple[DirectionThreshold, ...] = (
    DirectionThreshold(label="BEARISH", threshold=None),
    DirectionThreshold(label="NEUTRAL", threshold=-0.02),
    DirectionThreshold(label="BULLISH", threshold=0.02),
)


class LabelPolicy(BaseModel):
    """§28 版本化 label 定义；``version`` 是主键，语义变更必须新增版本。

    ``embargo_sessions`` 只在这里出现（§32.1）：experiment config 里的
    ``evaluation.embargo_sessions`` 必须与版本化定义一致，否则视为配置错误
    （见 :func:`label_policy_from_experiment_config`）。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    horizon_sessions: int = Field(ge=1)
    embargo_sessions: int = Field(ge=1)
    direction_thresholds: tuple[DirectionThreshold, ...]
    suspension_policy: SuspensionPolicy
    missing_future_bars_policy: MissingFutureBarsPolicy
    price_field: PriceField
    return_definition: ReturnDefinition

    @field_validator("version")
    @classmethod
    def _version_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("version must be non-empty")
        return value

    @field_validator("direction_thresholds")
    @classmethod
    def _thresholds_canonical(
        cls, value: tuple[DirectionThreshold, ...]
    ) -> tuple[DirectionThreshold, ...]:
        if len(value) != len(DIRECTION_LABELS):
            raise ValueError(
                f"direction_thresholds must define exactly the {len(DIRECTION_LABELS)} "
                f"§24.1 Direction labels {DIRECTION_LABELS}, got {len(value)} entries"
            )
        labels = [item.label for item in value]
        if tuple(labels) != DIRECTION_LABELS:
            raise ValueError(
                f"direction_thresholds labels must be {DIRECTION_LABELS} in that order, got {labels}"
            )
        if value[0].threshold is not None:
            raise ValueError(
                f"direction_thresholds[0] ({value[0].label}) must have threshold=None "
                "(无下界); only the first class may be open-ended"
            )
        lower_bounds = [item.threshold for item in value[1:]]
        if any(bound is None for bound in lower_bounds):
            raise ValueError("only the first direction class may have threshold=None")
        if lower_bounds != sorted(lower_bounds):  # type: ignore[type-var]
            raise ValueError(
                f"direction_thresholds lower bounds must be sorted ascending, got {lower_bounds}"
            )
        return value

    @model_validator(mode="after")
    def _embargo_covers_horizon(self) -> LabelPolicy:
        if self.embargo_sessions < self.horizon_sessions:
            raise ValueError(
                f"embargo_sessions {self.embargo_sessions} must be >= "
                f"horizon_sessions {self.horizon_sessions} (§27)"
            )
        return self

    @property
    def direction_interval_map(self) -> Mapping[str, float | None]:
        """label 名 → 其闭下界（首项为 ``None``）；只读视图，供 label builder / 报告使用。"""
        return MappingProxyType({item.label: item.threshold for item in self.direction_thresholds})

    def hashing_payload(self) -> dict[str, Any]:
        """dataset_hash 里的 label 部分；字段增删必须同时改这里（用例守住完备性）。"""
        return {
            "version": self.version,
            "horizon_sessions": self.horizon_sessions,
            "embargo_sessions": self.embargo_sessions,
            "direction_thresholds": [item.hashing_payload() for item in self.direction_thresholds],
            "suspension_policy": self.suspension_policy,
            "missing_future_bars_policy": self.missing_future_bars_policy,
            "price_field": self.price_field,
            "return_definition": self.return_definition,
        }


DEFAULT_LABEL_POLICY = LabelPolicy(
    version=LABEL_POLICY_VERSION,
    horizon_sessions=DEFAULT_HORIZON_SESSIONS,
    embargo_sessions=DEFAULT_EMBARGO_SESSIONS,
    direction_thresholds=DEFAULT_DIRECTION_THRESHOLDS,
    suspension_policy=DEFAULT_SUSPENSION_POLICY,
    missing_future_bars_policy=DEFAULT_MISSING_FUTURE_BARS_POLICY,
    price_field=DEFAULT_PRICE_FIELD,
    return_definition=DEFAULT_RETURN_DEFINITION,
)

#: 版本 → 定义 的唯一注册表。新增 label 语义（不是改数值）时在这里加一条。
LABEL_POLICIES: Mapping[str, LabelPolicy] = MappingProxyType(
    {DEFAULT_LABEL_POLICY.version: DEFAULT_LABEL_POLICY}
)


def resolve_label_policy(policy: LabelPolicy) -> LabelPolicy:
    """把调用方给的 policy 解析为注册表里的版本化定义；未知版本或载荷漂移即拒绝。

    这是 §28 的「label 语义必须版本化」落到代码上的检查点：如果只为改一个数值就
    就地编辑已发布版本的字段，历史 benchmark 的 ``label_policy_version`` 会指向一个
    与当时语义不同的定义，取数时无法解释。因此载荷必须逐字段等于注册表定义。
    """
    canonical = LABEL_POLICIES.get(policy.version)
    if canonical is None:
        known = ", ".join(sorted(LABEL_POLICIES))
        raise ConfigurationError(
            f"unknown label policy version {policy.version!r}; known versions: {known}"
        )
    if policy != canonical:
        raise ConfigurationError(
            f"label policy {policy.version!r} payload differs from the versioned definition; "
            "label 语义/数值变更必须新增 LabelPolicy 版本，而不是改写已发布版本"
        )
    return canonical


def label_policy_from_experiment_config(
    *,
    label_policy_version: str = LABEL_POLICY_VERSION,
    embargo_sessions: int | None = None,
) -> LabelPolicy:
    """§32.1：experiment config 物化为 LabelPolicy；embargo 不一致即拒绝运行。

    ``embargo_sessions`` 的单一真源是 LabelPolicy。若 config 里写了不同的值，
    必须先新增 LabelPolicy 版本（并把新版本登记进 :data:`LABEL_POLICIES`），
    否则同一个版本会对应两种 embargo，dataset_hash 也就无法解释。
    """
    canonical = LABEL_POLICIES.get(label_policy_version)
    if canonical is None:
        known = ", ".join(sorted(LABEL_POLICIES))
        raise ConfigurationError(
            f"unknown label policy version {label_policy_version!r}; known versions: {known}"
        )
    if embargo_sessions is not None and embargo_sessions != canonical.embargo_sessions:
        raise ConfigurationError(
            f"experiment config embargo_sessions={embargo_sessions} differs from "
            f"{canonical.version}'s embargo_sessions={canonical.embargo_sessions}; "
            "embargo 的单一真源是 LabelPolicy（§32.1），必须升版本而不是在 config 里覆盖"
        )
    return canonical


class ForecastOrigin(BaseModel):
    """一个 forecast origin：``(symbol, market_date)`` + 由 policy 推导的时间语义。

    ``knowledge_cutoff`` 由 ``KnowledgeCutoffPolicy`` 推导（§5），``label_sessions``
    是该 origin 未来 label 覆盖的 ``horizon_sessions`` 个 market session（§11 的
    时间轴真源，不是个股自身有效 session）。两者都进 dataset_hash。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    market_date: SessionDate
    knowledge_cutoff: ShanghaiDatetime
    label_sessions: tuple[SessionDate, ...]

    @field_validator("symbol")
    @classmethod
    def _symbol_normalized(cls, value: str) -> str:
        return validate_normalized_symbol(value)

    @field_validator("label_sessions")
    @classmethod
    def _label_sessions_ascending_unique(cls, value: tuple[date, ...]) -> tuple[date, ...]:
        if not value:
            raise ValueError("label_sessions must not be empty")
        if list(value) != sorted(value):
            raise ValueError("label_sessions must be sorted ascending")
        if len(set(value)) != len(value):
            raise ValueError("label_sessions must be unique")
        return value

    @model_validator(mode="after")
    def _origin_timeline(self) -> ForecastOrigin:
        if self.knowledge_cutoff.date() != self.market_date:
            raise ValueError(
                "knowledge_cutoff must fall on market_date "
                f"(market_date={self.market_date}, cutoff={self.knowledge_cutoff})"
            )
        if self.label_sessions[0] <= self.market_date:
            raise ValueError(
                f"label_sessions must lie strictly after market_date {self.market_date}, "
                f"got {self.label_sessions[0]}"
            )
        return self

    def hashing_payload(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "market_date": self.market_date.isoformat(),
            "knowledge_cutoff": self.knowledge_cutoff.isoformat(),
            "label_sessions": [day.isoformat() for day in self.label_sessions],
        }


LabelStatus = Literal["LABELED", "SUSPENDED", "INSUFFICIENT_FUTURE_BARS"]


class ForecastLabel(BaseModel):
    """一个 origin 的 ground truth（§28 + ADR-009 §2）。

    三种状态都是**显式值**，不用「字段缺失」表达：``SUSPENDED`` 是数据事实（horizon
    内有个股缺 bar），``INSUFFICIENT_FUTURE_BARS`` 是数据末端（horizon 越过 provider
    已发布范围），两者在报告里必须能被分开统计（§20），否则「模型差」与「数据缺」混为
    一谈。非 ``LABELED`` 的 label 不参与指标计算，但保留在数据集里以便统计覆盖率。

    ``direction is None`` 对应 §24.1 的 ``UNKNOWN``：决策层需要区分「证据不足」与
    ``NEUTRAL``，label 侧用 ``None`` 表达前者。

    本模型的字段与 :func:`build_label` 实现的就是 policy 的这组版本化取值：
    ``suspension_policy="mark_suspended"``、
    ``missing_future_bars_policy="mark_insufficient"``、``price_field="close"``、
    ``return_definition="close_to_close_simple"``。新增取值必须同时加分支。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    market_date: SessionDate
    label_policy_version: str
    status: LabelStatus
    horizon_return: float | None
    direction: DirectionLabel | None
    observed_sessions: int = Field(ge=0)

    @field_validator("symbol")
    @classmethod
    def _symbol_normalized(cls, value: str) -> str:
        return validate_normalized_symbol(value)

    @field_validator("label_policy_version")
    @classmethod
    def _known_policy_version(cls, value: str) -> str:
        if value not in LABEL_POLICIES:
            raise ValueError(
                f"unknown label policy version {value!r}; known versions: {sorted(LABEL_POLICIES)}"
            )
        return value

    @model_validator(mode="after")
    def _status_payload_consistent(self) -> ForecastLabel:
        if self.status == "LABELED":
            if self.horizon_return is None or self.direction is None:
                raise ValueError("LABELED labels must carry horizon_return and direction")
            if self.observed_sessions < 1:
                raise ValueError("LABELED labels must observe at least one session")
            if self.horizon_return != self.horizon_return or self.horizon_return in (
                float("inf"),
                float("-inf"),
            ):
                raise ValueError(f"horizon_return must be finite, got {self.horizon_return}")
        elif self.horizon_return is not None or self.direction is not None:
            raise ValueError(
                f"{self.status} labels must not carry horizon_return/direction "
                "(证据不足不是 0 收益)"
            )
        return self


def direction_label(horizon_return: float, policy: LabelPolicy) -> DirectionLabel:
    """把 horizon 收益映射到 §24.1 的 Direction label（§28）。

    规则：label 是「满足 ``threshold is None or r >= threshold`` 的最后一项」；
    因为 :class:`DirectionThreshold` 已定死 ``DIRECTION_LABELS`` 顺序与单调下界，
    这条规则就是「按闭下界归档」，边界值归入较高的那一类。
    """
    resolved = resolve_label_policy(policy)
    if horizon_return != horizon_return:
        raise DataQualityError(f"horizon_return must not be NaN, got {horizon_return}")
    label: str = DIRECTION_LABELS[0]
    for entry in resolved.direction_thresholds:
        if entry.threshold is None or horizon_return >= entry.threshold:
            label = entry.label
    # 仅用于类型收窄：LabelPolicy 校验已保证 label ∈ DIRECTION_LABELS
    if label not in DIRECTION_LABELS:
        raise ConfigurationError(
            f"label policy {resolved.version!r} produced direction {label!r}, "
            f"which is not one of {DIRECTION_LABELS}"
        )
    return label


def build_label(
    *,
    origin: ForecastOrigin,
    policy: LabelPolicy,
    origin_close: float,
    future_bars: Sequence[MarketBar],
    data_coverage_end: date,
) -> ForecastLabel:
    """构造一个 origin 的 label（§28 + ADR-009 §2）。

    ``future_bars`` 是该 symbol 在其 label 窗口内的 bar（停牌行允许传入，按缺 bar
    处理）；``data_coverage_end`` 是**数据集全局**的数据末端（provider 已发布的最后
    一个 market session），不是「该个股最后一根 bar」——只有全局口径才能把「停牌」
    与「数据还没发布」分开。

    判定优先级（ADR-009 §2）：

    ```text
    label 窗口越过 data_coverage_end        → INSUFFICIENT_FUTURE_BARS
    窗口内某个 session 无 valid bar          → SUSPENDED
    全部命中                                  → LABELED（close-to-close 收益 + direction）
    ```

    非正 / 非有限 ``origin_close``、跨窗口的 bar、重复 session 都是数据/调用方错误，显式
    失败（ADR-010）；label bar 在 origin 的 ``knowledge_cutoff`` 时点已经可用属 §29 的
    泄漏，抛 :class:`~kronos_ai.errors.LeakageError`。

    该 §29 守卫在正常数据路径上不可达：:class:`~kronos_ai.domain.market.MarketBar`
    保证 ``available_at >= timestamp``，而 label session 严格晚于 ``market_date``，
    所以真实 bar 永远晚于 cutoff。它仍然保留为廉价 defense-in-depth，可被「手工构造
    origin（显式 cutoff 落在 label 窗口之后）」触发（见
    ``tests/leakage/test_walk_forward_leakage.py``）。

    ``price_field`` / ``return_definition`` / ``suspension_policy`` /
    ``missing_future_bars_policy`` 目前各只有一个合法取值；这里显式断言，使未来新增
    字面量时不会静默按旧口径算 label。
    """
    resolved = resolve_label_policy(policy)
    if resolved.price_field != DEFAULT_PRICE_FIELD:
        raise ConfigurationError(
            f"price_field {resolved.price_field!r} is not implemented; "
            f"only {DEFAULT_PRICE_FIELD!r} is supported"
        )
    if resolved.return_definition != DEFAULT_RETURN_DEFINITION:
        raise ConfigurationError(
            f"return_definition {resolved.return_definition!r} is not implemented; "
            f"only {DEFAULT_RETURN_DEFINITION!r} is supported"
        )
    if resolved.suspension_policy != DEFAULT_SUSPENSION_POLICY:
        raise ConfigurationError(
            f"suspension_policy {resolved.suspension_policy!r} is not implemented; "
            f"only {DEFAULT_SUSPENSION_POLICY!r} is supported"
        )
    if resolved.missing_future_bars_policy != DEFAULT_MISSING_FUTURE_BARS_POLICY:
        raise ConfigurationError(
            f"missing_future_bars_policy {resolved.missing_future_bars_policy!r} is not "
            f"implemented; only {DEFAULT_MISSING_FUTURE_BARS_POLICY!r} is supported"
        )
    if origin_close != origin_close or origin_close in (float("inf"), float("-inf")):
        raise DataQualityError(f"origin_close must be finite, got {origin_close}")
    if origin_close <= 0:
        raise DataQualityError(f"origin_close must be > 0, got {origin_close}")
    if isinstance(data_coverage_end, datetime):
        raise ConfigurationError(
            f"data_coverage_end must be a date, not datetime: {data_coverage_end.isoformat()}"
        )

    window = set(origin.label_sessions)
    seen: set[date] = set()
    closes: dict[date, float] = {}
    for bar in future_bars:
        if bar.symbol != origin.symbol:
            raise DataQualityError(
                f"future bar symbol {bar.symbol!r} does not match origin symbol {origin.symbol!r}"
            )
        session = bar.timestamp.date()
        if session not in window:
            raise DataQualityError(
                f"future bar {session} is outside origin {origin.symbol}@{origin.market_date} "
                f"label window {origin.label_sessions}"
            )
        if bar.available_at <= origin.knowledge_cutoff:
            raise LeakageError(
                f"future bar {session} was already available at the origin's knowledge_cutoff "
                f"{origin.knowledge_cutoff.isoformat()}: labels must be future information (§29)"
            )
        # 重复 session 的判定必须先于停牌过滤，否则 (停牌, 有效) 与 (有效, 停牌)
        # 两种顺序会得到不同结果（顺序相关的守卫不是守卫）。
        if session in seen:
            raise DataQualityError(
                f"duplicate future bar for session {session} of symbol {origin.symbol}"
            )
        seen.add(session)
        if bar.trade_status == "0":  # 停牌行按缺 bar 处理（ADR-009：valid bar 只由 provider 构造）
            continue
        closes[session] = bar.close

    observed = len(closes)
    truncated = origin.label_sessions[-1] > data_coverage_end
    if truncated:
        status: LabelStatus = "INSUFFICIENT_FUTURE_BARS"
    elif observed < len(origin.label_sessions):
        status = "SUSPENDED"
    else:
        status = "LABELED"

    horizon_return: float | None = None
    direction: DirectionLabel | None = None
    if status == "LABELED":
        final_close = closes[origin.label_sessions[-1]]
        horizon_return = final_close / origin_close - 1.0
        direction = direction_label(horizon_return, resolved)

    return ForecastLabel(
        symbol=origin.symbol,
        market_date=origin.market_date,
        label_policy_version=resolved.version,
        status=status,
        horizon_return=horizon_return,
        direction=direction,
        observed_sessions=observed,
    )


class DatasetSegment(BaseModel):
    """一段的 origins：``start_index`` 是段内首个 session 在整条时间轴中的下标。

    ``origins`` 的顺序是规范化的 **symbol-major**（symbol 升序在外层，session 升序在
    内层）：dataset_hash 必须与调用方传入的 symbol 书写顺序无关，否则同一份数据会
    因为参数顺序不同而产生两个 hash。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: SegmentName
    start_index: int = Field(ge=0)
    sessions: tuple[SessionDate, ...]
    origins: tuple[ForecastOrigin, ...]

    @field_validator("sessions")
    @classmethod
    def _sessions_ascending_unique(cls, value: tuple[date, ...]) -> tuple[date, ...]:
        if not value:
            raise ValueError("sessions must not be empty")
        if list(value) != sorted(value):
            raise ValueError("sessions must be sorted ascending")
        if len(set(value)) != len(value):
            raise ValueError("sessions must be unique")
        return value

    @model_validator(mode="after")
    def _origins_match_sessions(self) -> DatasetSegment:
        if not self.origins:
            raise ValueError("origins must not be empty")
        segment_sessions = set(self.sessions)
        pairs: set[tuple[str, date]] = set()
        for origin in self.origins:
            if origin.market_date not in segment_sessions:
                raise ValueError(
                    f"origin market_date {origin.market_date} is not in segment {self.name!r}"
                )
            key = (origin.symbol, origin.market_date)
            if key in pairs:
                raise ValueError(f"duplicate origin {key} in segment {self.name!r}")
            pairs.add(key)
        if len(pairs) != len(self.sessions) * len({origin.symbol for origin in self.origins}):
            raise ValueError(
                f"segment {self.name!r} must hold one origin per (symbol, session) pair"
            )
        return self

    @property
    def end_index(self) -> int:
        return self.start_index + len(self.sessions) - 1

    @property
    def first_session(self) -> date:
        return self.sessions[0]

    @property
    def last_session(self) -> date:
        return self.sessions[-1]

    def hashing_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start_index": self.start_index,
            "origins": [origin.hashing_payload() for origin in self.origins],
        }


class WalkForwardDataset(BaseModel):
    """§27 的 point-in-time 样本集：时间轴 + 分段 + LabelPolicy，全部可复现。

    自校验（构造即失败，见 :func:`build_walk_forward_dataset`）：``label_policy`` 必须与
    注册表中的同名版本逐字段一致（§28 「一个版本对应一份定义」）、每个 origin 的 symbol
    必须在 ``symbols`` 里、段的起止下标必须与时间轴切片吻合且首末对齐、相邻段之间必须
    至少空置 ``label_policy.embargo_sessions`` 个 session（§29 的 leakage 断言在这里再走
    一遍，防止绕过 builder 手工拼一个漏数据集的 dataset 再拿去跑 benchmark）。

    注意「至少」：手工构造的 dataset 允许比 embargo 更宽的间隔（多出来的 session 不属于
    任何段，不会泄漏）；builder 产出的则是恰好 embargo。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    label_policy: LabelPolicy
    calendar_exchange: Exchange
    calendar_source: str
    cutoff_policy: KnowledgeCutoffPolicy
    lookback_bars: int = Field(ge=1)
    symbols: tuple[str, ...]
    sessions: tuple[SessionDate, ...]
    segments: tuple[DatasetSegment, ...]

    @field_validator("version")
    @classmethod
    def _known_version(cls, value: str) -> str:
        if value != FORECAST_DATASET_VERSION:
            raise ValueError(
                f"unknown dataset version {value!r}; this build emits {FORECAST_DATASET_VERSION!r}"
            )
        return value

    @field_validator("symbols")
    @classmethod
    def _symbols_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("symbols must not be empty")
        normalized = [validate_normalized_symbol(symbol) for symbol in value]
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"symbols must be unique, got {normalized}")
        if normalized != sorted(normalized):
            raise ValueError(f"symbols must be sorted ascending, got {normalized}")
        return tuple(normalized)

    @field_validator("sessions")
    @classmethod
    def _timeline_ascending_unique(cls, value: tuple[date, ...]) -> tuple[date, ...]:
        if not value:
            raise ValueError("sessions must not be empty")
        if list(value) != sorted(value):
            raise ValueError("sessions must be sorted ascending")
        if len(set(value)) != len(value):
            raise ValueError("sessions must be unique")
        return value

    @model_validator(mode="after")
    def _segments_tile_the_timeline(self) -> WalkForwardDataset:
        if not self.segments:
            raise ValueError("segments must not be empty")
        # §28：version 与定义必须一一对应，手工拼的 dataset 不能声明 v1 却用别的 embargo。
        # ConfigurationError 不是 ValueError，会原样冒泡（与 require_no_leakage 同一口径）。
        resolve_label_policy(self.label_policy)
        symbol_count = len(self.symbols)
        if self.segments[0].start_index != self.lookback_bars:
            raise ValueError(
                f"first segment starts at index {self.segments[0].start_index} but "
                f"lookback_bars={self.lookback_bars} sessions must precede the first origin"
            )
        for segment in self.segments:
            if (
                tuple(self.sessions[segment.start_index : segment.end_index + 1])
                != segment.sessions
            ):
                raise ValueError(
                    f"segment {segment.name!r} sessions do not match the timeline slice "
                    f"[{segment.start_index}, {segment.end_index}]"
                )
            if len(segment.origins) != len(segment.sessions) * symbol_count:
                raise ValueError(
                    f"segment {segment.name!r} has {len(segment.origins)} origins but "
                    f"{len(segment.sessions)} sessions * {symbol_count} symbols are required"
                )
            foreign = sorted({o.symbol for o in segment.origins} - set(self.symbols))
            if foreign:
                raise ValueError(
                    f"segment {segment.name!r} references symbols not in dataset symbols "
                    f"{self.symbols}: {foreign}"
                )
        last = self.segments[-1]
        if last.end_index != len(self.sessions) - 1:
            raise ValueError(
                f"last segment ends at index {last.end_index} but the timeline has "
                f"{len(self.sessions)} sessions"
            )
        # §29：相邻段之间必须空置 embargo_sessions；LeakageError 会直接冒泡（不是
        # ValidationError），因为这是数据集本身的完整性问题，不是字段格式问题。
        require_no_leakage(
            tuple(
                SegmentSessions(name=s.name, start_index=s.start_index, sessions=s.sessions)
                for s in self.segments
            ),
            embargo_sessions=self.label_policy.embargo_sessions,
            horizon_sessions=self.label_policy.horizon_sessions,
        )
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dataset_hash(self) -> str:
        """内容 hash：读取时实时派生（换 label policy / 分段 / 时间轴都会变）。"""
        return compute_dataset_hash(
            version=self.version,
            label_policy=self.label_policy,
            calendar_exchange=self.calendar_exchange,
            calendar_source=self.calendar_source,
            cutoff_policy=self.cutoff_policy,
            lookback_bars=self.lookback_bars,
            symbols=self.symbols,
            sessions=self.sessions,
            segments=self.segments,
        )


def compute_dataset_hash(
    *,
    version: str,
    label_policy: LabelPolicy,
    calendar_exchange: Exchange,
    calendar_source: str,
    cutoff_policy: KnowledgeCutoffPolicy,
    lookback_bars: int,
    symbols: Sequence[str],
    sessions: Sequence[date],
    segments: Sequence[DatasetSegment],
    walk_forward_version: str = WALK_FORWARD_VERSION,
) -> str:
    """``WalkForwardDataset.dataset_hash`` 的唯一定义。

    label policy 以 ``hashing_payload()`` 整体进入 payload（含 ``version``），
    因此 §27 的「dataset_hash 绑定 LabelPolicy 版本」是结构性的，而不是靠调用方
    自觉把版本号拼进去。``walk_forward_version`` 同样是契约的一部分：分段/embargo
    算术规则本身变化（@WALK_FORWARD_VERSION）就会换掉全部 dataset_hash。golden 测试
    锚定本函数的输出（tests/unit/test_evaluation_dataset.py）：修改 payload 构成会使既有
    dataset_hash 失效，属于契约变更。
    """
    payload: dict[str, Any] = {
        "kind": "walk_forward_dataset",
        "dataset_version": version,
        # 分段/embargo 算术的版本（@WALK_FORWARD_VERSION）：改变「一段占哪些 session」的
        # 规则本身会换掉全部 dataset_hash，而不必等到某个字段跟着变。
        "walk_forward_version": walk_forward_version,
        "label_policy": label_policy.hashing_payload(),
        "calendar": {"exchange": calendar_exchange, "source": calendar_source},
        "cutoff_policy": cutoff_policy_record(cutoff_policy).model_dump(mode="json"),
        "lookback_bars": lookback_bars,
        "symbols": list(symbols),
        "sessions": [day.isoformat() for day in sessions],
        "segments": [segment.hashing_payload() for segment in segments],
    }
    # 与 LabelPolicy.hashing_payload 同一套完备性约定：payload 必须覆盖上面列举的每个维度。
    assert set(payload) == {
        "kind",
        "dataset_version",
        "walk_forward_version",
        "label_policy",
        "calendar",
        "cutoff_policy",
        "lookback_bars",
        "symbols",
        "sessions",
        "segments",
    }, "compute_dataset_hash payload must cover every dataset dimension"
    return sha256_hex(payload)


def build_walk_forward_dataset(
    *,
    calendar: TradingCalendar,
    sessions: Sequence[date],
    symbols: Sequence[str],
    plan: WalkForwardPlan,
    label_policy: LabelPolicy,
    lookback_bars: int,
    cutoff_policy: KnowledgeCutoffPolicy = "same_day_evening",
) -> WalkForwardDataset:
    """构造 walk-forward dataset（§27）。

    ``sessions`` 必须**恰好**是：``lookback_bars`` 个前置 session（首个 origin 的
    lookback 窗口）+ 各段 origin session + 相邻段之间的 ``embargo_sessions`` 个空置
    session。多一个少一个都拒绝（ADR-010）：静默取前 N 个会让两次 run 的
    dataset_hash 相同而实际样本不同。

    每个 origin 的 ``label_sessions`` 由 ``calendar.next_sessions`` 推导，因此
    日历覆盖不足以覆盖某个 origin 的 horizon 时会立刻抛 ``CalendarError``（ADR-009 §3），
    而不是产出一个 label 窗口不完整的样本。（``INSUFFICIENT_FUTURE_BARS`` 是另一个口径：
    日历知道 session、但 provider 还没发布到那里，见 ``data_coverage_end``。）

    ``symbols`` 重复会抛 ``ConfigurationError``，不会静默去重；``label_policy`` 先经
    :func:`resolve_label_policy` 校验，未知版本或载荷漂移不会进入 dataset。
    """
    if lookback_bars < 1:
        raise ConfigurationError(f"lookback_bars must be >= 1, got {lookback_bars}")
    if cutoff_policy == "explicit":
        raise ConfigurationError(
            "cutoff_policy='explicit' is not supported by the dataset builder: "
            "per-origin explicit cutoffs are not part of the dataset contract"
        )
    policy = resolve_label_policy(label_policy)
    if not symbols:
        raise ConfigurationError("symbols must not be empty")
    try:
        normalized = [validate_normalized_symbol(s) for s in symbols]
    except ValueError as error:  # 归一化失败是配置错误，不是字段格式问题
        raise ConfigurationError(str(error)) from error
    if len(set(normalized)) != len(normalized):
        # 对比数据集模型的 _symbols_canonical：静默去重会让「传了三个 symbol」与
        # 「传了两个」产生同一个 dataset_hash（ADR-010 显式失败）。
        raise ConfigurationError(f"symbols must be unique, got {normalized}")
    normalized_symbols = tuple(sorted(normalized))

    segment_sessions = split_sessions(
        sessions=sessions,
        plan=plan,
        embargo_sessions=policy.embargo_sessions,
        history_prefix=lookback_bars,
    )
    for day in sessions:
        if not calendar.is_session(day):
            raise ConfigurationError(f"{day} is not a market session of the given calendar")

    segments: list[DatasetSegment] = []
    for segment in segment_sessions:
        origins = tuple(
            ForecastOrigin(
                symbol=symbol,
                market_date=day,
                knowledge_cutoff=resolve_knowledge_cutoff(day, cutoff_policy),
                label_sessions=tuple(calendar.next_sessions(day, policy.horizon_sessions)),
            )
            for symbol in normalized_symbols
            for day in segment.sessions
        )
        segments.append(
            DatasetSegment(
                name=segment.name,
                start_index=segment.start_index,
                sessions=segment.sessions,
                origins=origins,
            )
        )

    return WalkForwardDataset(
        version=FORECAST_DATASET_VERSION,
        label_policy=policy,
        calendar_exchange=calendar.exchange,
        calendar_source=calendar.source,
        cutoff_policy=cutoff_policy,
        lookback_bars=lookback_bars,
        symbols=normalized_symbols,
        sessions=tuple(sessions),
        segments=tuple(segments),
    )


if TYPE_CHECKING:
    # mypy 结构化校验：builder 必须接受 §6.6 的 TradingCalendar（含 exchange/source
    # provenance），并且数据集能只靠契约字段派生 hash。LeakageError 在 dataset 校验
    # 里会直接冒泡，见 WalkForwardDataset 的 docstring。
    from kronos_ai.data.calendar import StaticTradingCalendar
    from kronos_ai.evaluation.walk_forward import SegmentSpec

    _CONTRACT_ANCHOR: WalkForwardDataset = build_walk_forward_dataset(
        calendar=StaticTradingCalendar(
            exchange="SSE", source="type-check", sessions=(date(2026, 1, 5),)
        ),
        sessions=(date(2026, 1, 5),),
        symbols=("600000",),
        plan=WalkForwardPlan(segments=(SegmentSpec(name="test", length_sessions=1),)),
        label_policy=DEFAULT_LABEL_POLICY,
        lookback_bars=1,
    )
