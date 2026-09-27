"""LabelPolicy 与 walk-forward dataset（RX-KAI-017，基线文档 §27 / §28 / §29 / §32）。

覆盖范围：

- 版本化常量与方向词汇（与 §24.1 的 Direction 对齐、与 §13 的 ±2% 阈值同口径）；
- ``LabelPolicy`` 校验（embargo >= horizon、方向类顺序与下界、frozen、extra=forbid）、
  ``hashing_payload`` 完备性、``resolve_label_policy`` / ``label_policy_from_experiment_config``
  的 §28/§32.1 拒绝路径；
- ``build_walk_forward_dataset``：分段、embargo、provenance（exchange/source）、
  origin 时间语义、拒绝路径；
- ``dataset_hash``：golden 值 + 对 label policy / embargo / cutoff policy / lookback /
  时间轴 / symbols 的敏感性（§27「dataset_hash 绑定 LabelPolicy 版本」）；
- dataset 模型自校验（绕过 builder 的路径）；
- ``build_label`` 的三态与守卫（ADR-009 §2）。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import get_args

import pytest
from pydantic import ValidationError

from kronos_ai.data.calendar import StaticTradingCalendar
from kronos_ai.domain.hashing import is_sha256_hex
from kronos_ai.domain.market import MarketBar
from kronos_ai.domain.time import CN_TZ, MARKET_SESSION_CLOSE, resolve_knowledge_cutoff
from kronos_ai.errors import CalendarError, ConfigurationError, DataQualityError
from kronos_ai.evaluation.dataset import (
    DEFAULT_DIRECTION_THRESHOLDS,
    DEFAULT_EMBARGO_SESSIONS,
    DEFAULT_HORIZON_SESSIONS,
    DEFAULT_LABEL_POLICY,
    DIRECTION_LABELS,
    FORECAST_DATASET_VERSION,
    LABEL_POLICIES,
    LABEL_POLICY_VERSION,
    DirectionThreshold,
    ForecastOrigin,
    LabelPolicy,
    MissingFutureBarsPolicy,
    PriceField,
    ReturnDefinition,
    SuspensionPolicy,
    WalkForwardDataset,
    build_label,
    build_walk_forward_dataset,
    compute_dataset_hash,
    direction_label,
    label_policy_from_experiment_config,
    resolve_label_policy,
)
from kronos_ai.evaluation.walk_forward import WALK_FORWARD_VERSION, SegmentSpec, WalkForwardPlan

SYMBOL = "600000"
SYMBOL_B = "000001"
EXCHANGE = "SSE"
SOURCE = "test-fixture"
LOOKBACK = 6
START = date(2026, 6, 1)
PLAN = WalkForwardPlan(
    segments=(
        SegmentSpec(name="train", length_sessions=8),
        SegmentSpec(name="validation", length_sessions=4),
        SegmentSpec(name="test", length_sessions=4),
    )
)
# 6 (lookback) + 8 + 4 + 4 (origins) + 5 * 2 (embargo) = 32
REQUIRED_SESSIONS = 32
# golden 常量锚定 compute_dataset_hash 的 payload 构成（见 tests/unit/test_market.py 同款做法）
GOLDEN_DATASET_HASH = "21b04ae104543462783fb5fe7cb2003eb7e98f047b68406f9ac80185d14849c9"


def weekdays(*, start: date = START, count: int) -> tuple[date, ...]:
    """显式工作日序列（fixture 用，不做规则推导）。"""
    days: list[date] = []
    current = start
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return tuple(days)


def timeline(count: int = REQUIRED_SESSIONS) -> tuple[date, ...]:
    return weekdays(count=count)


def calendar(count: int = REQUIRED_SESSIONS + 10) -> StaticTradingCalendar:
    """日历覆盖到时间轴之后（label 窗口需要 origin 之后的 session）。"""
    return StaticTradingCalendar(exchange=EXCHANGE, source=SOURCE, sessions=weekdays(count=count))


def build_dataset(
    *,
    symbols: tuple[str, ...] = (SYMBOL,),
    plan: WalkForwardPlan = PLAN,
    label_policy: LabelPolicy = DEFAULT_LABEL_POLICY,
    lookback_bars: int = LOOKBACK,
    cutoff_policy: str = "same_day_evening",
    count: int = REQUIRED_SESSIONS,
    cal: StaticTradingCalendar | None = None,
    days: tuple[date, ...] | None = None,
) -> WalkForwardDataset:
    return build_walk_forward_dataset(
        calendar=cal or calendar(),
        sessions=timeline(count) if days is None else days,
        symbols=symbols,
        plan=plan,
        label_policy=label_policy,
        lookback_bars=lookback_bars,
        cutoff_policy=cutoff_policy,  # type: ignore[arg-type]
    )


def origin(
    day: date, *, symbol: str = SYMBOL, horizon: int = DEFAULT_HORIZON_SESSIONS
) -> ForecastOrigin:
    cal = calendar()
    return ForecastOrigin(
        symbol=symbol,
        market_date=day,
        knowledge_cutoff=resolve_knowledge_cutoff(day, "same_day_evening"),
        label_sessions=tuple(cal.next_sessions(day, horizon)),
    )


def bar(
    day: date,
    close: float,
    *,
    symbol: str = SYMBOL,
    trade_status: str | None = "1",
    available_at: datetime | None = None,
) -> MarketBar:
    return MarketBar(
        symbol=symbol,
        timestamp=datetime.combine(day, MARKET_SESSION_CLOSE, tzinfo=CN_TZ),
        open=close,
        high=close,
        low=close,
        close=close,
        trade_status=trade_status,
        adjustment_mode="raw",
        available_at=available_at or datetime.combine(day, MARKET_SESSION_CLOSE, tzinfo=CN_TZ),
    )


def label_bars(
    days: tuple[date, ...], *, symbol: str = SYMBOL, final_close: float = 110.0
) -> tuple[MarketBar, ...]:
    """label 窗口内每个 session 一根 bar，末根 close = final_close（顺序收益率递增）。"""
    return tuple(
        bar(day, final_close if index == len(days) - 1 else 100.0, symbol=symbol)
        for index, day in enumerate(days)
    )


class TestVersionedConstants:
    def test_versions_are_pinned(self) -> None:
        assert LABEL_POLICY_VERSION == "label-policy-v1"
        assert FORECAST_DATASET_VERSION == "walk-forward-dataset-v1"
        assert DEFAULT_HORIZON_SESSIONS == 5
        assert DEFAULT_EMBARGO_SESSIONS == 5
        assert set(LABEL_POLICIES) == {LABEL_POLICY_VERSION}

    def test_direction_labels_match_decision_schema_vocabulary(self) -> None:
        # §24.1 第一版 Direction 词汇；label 侧只用其中三个（UNKNOWN 属决策层）
        assert DIRECTION_LABELS == ("BEARISH", "NEUTRAL", "BULLISH")
        assert tuple(item.label for item in DEFAULT_DIRECTION_THRESHOLDS) == DIRECTION_LABELS
        assert DEFAULT_DIRECTION_THRESHOLDS[0].threshold is None
        assert DEFAULT_DIRECTION_THRESHOLDS[1].threshold == -0.02
        assert DEFAULT_DIRECTION_THRESHOLDS[2].threshold == 0.02

    @pytest.mark.parametrize(
        ("annotation", "expected"),
        [
            (SuspensionPolicy, ("mark_suspended",)),
            (MissingFutureBarsPolicy, ("mark_insufficient",)),
            (PriceField, ("close",)),
            (ReturnDefinition, ("close_to_close_simple",)),
        ],
    )
    def test_policy_vocabularies_are_pinned(
        self, annotation: object, expected: tuple[str, ...]
    ) -> None:
        # policy 取值空间被钉住：新增取值必须同时给 build_label 加分支（否则本用例失败）
        assert get_args(annotation) == expected

    def test_default_policy_payload(self) -> None:
        assert DEFAULT_LABEL_POLICY.version == LABEL_POLICY_VERSION
        assert DEFAULT_LABEL_POLICY.horizon_sessions == DEFAULT_HORIZON_SESSIONS
        assert DEFAULT_LABEL_POLICY.embargo_sessions == DEFAULT_EMBARGO_SESSIONS


class TestLabelPolicy:
    def test_hashing_payload_pins_all_fields(self) -> None:
        payload = DEFAULT_LABEL_POLICY.hashing_payload()
        assert set(payload) == set(LabelPolicy.model_fields)
        assert payload["version"] == LABEL_POLICY_VERSION
        assert payload["direction_thresholds"] == [
            {"label": "BEARISH", "threshold": None},
            {"label": "NEUTRAL", "threshold": -0.02},
            {"label": "BULLISH", "threshold": 0.02},
        ]

    def test_embargo_must_cover_horizon(self) -> None:
        with pytest.raises(ValidationError, match="must be >= horizon_sessions"):
            LabelPolicy(**{**DEFAULT_LABEL_POLICY.model_dump(), "embargo_sessions": 3})

    @pytest.mark.parametrize("field", ["horizon_sessions", "embargo_sessions"])
    @pytest.mark.parametrize("value", [0, -1])
    def test_positive_session_counts(self, field: str, value: int) -> None:
        with pytest.raises(ValidationError):
            LabelPolicy(**{**DEFAULT_LABEL_POLICY.model_dump(), field: value})

    def test_threshold_labels_must_be_the_direction_vocabulary(self) -> None:
        with pytest.raises(ValidationError, match="Direction labels"):
            LabelPolicy(
                **{
                    **DEFAULT_LABEL_POLICY.model_dump(),
                    "direction_thresholds": DEFAULT_DIRECTION_THRESHOLDS[:2],
                }
            )

    def test_threshold_order_is_enforced(self) -> None:
        with pytest.raises(ValidationError, match="in that order"):
            LabelPolicy(
                **{
                    **DEFAULT_LABEL_POLICY.model_dump(),
                    "direction_thresholds": (
                        DirectionThreshold(label="NEUTRAL", threshold=None),
                        DirectionThreshold(label="BEARISH", threshold=-0.02),
                        DirectionThreshold(label="BULLISH", threshold=0.02),
                    ),
                }
            )

    def test_open_ended_class_must_be_first(self) -> None:
        with pytest.raises(ValidationError, match="threshold=None"):
            LabelPolicy(
                **{
                    **DEFAULT_LABEL_POLICY.model_dump(),
                    "direction_thresholds": (
                        DirectionThreshold(label="BEARISH", threshold=-0.02),
                        DirectionThreshold(label="NEUTRAL", threshold=None),
                        DirectionThreshold(label="BULLISH", threshold=0.02),
                    ),
                }
            )

    def test_lower_bounds_must_be_ascending(self) -> None:
        with pytest.raises(ValidationError, match="sorted ascending"):
            LabelPolicy(
                **{
                    **DEFAULT_LABEL_POLICY.model_dump(),
                    "direction_thresholds": (
                        DirectionThreshold(label="BEARISH", threshold=None),
                        DirectionThreshold(label="NEUTRAL", threshold=0.02),
                        DirectionThreshold(label="BULLISH", threshold=-0.02),
                    ),
                }
            )

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_threshold_must_be_finite(self, value: float) -> None:
        with pytest.raises(ValidationError, match="must be finite"):
            DirectionThreshold(label="NEUTRAL", threshold=value)

    def test_label_token_must_be_ascii(self) -> None:
        with pytest.raises(ValidationError, match="ascii token"):
            DirectionThreshold(label="涨", threshold=None)

    def test_policy_is_frozen_and_forbids_extra(self) -> None:
        with pytest.raises(ValidationError):
            DEFAULT_LABEL_POLICY.embargo_sessions = 1  # type: ignore[misc]
        with pytest.raises(ValidationError):
            LabelPolicy(**{**DEFAULT_LABEL_POLICY.model_dump(), "surprise": 1})

    def test_direction_interval_map_is_read_only(self) -> None:
        mapping = DEFAULT_LABEL_POLICY.direction_interval_map
        assert mapping["BEARISH"] is None and mapping["BULLISH"] == 0.02
        with pytest.raises(TypeError):
            mapping["BEARISH"] = -0.5  # type: ignore[index]


class TestResolveLabelPolicy:
    def test_default_policy_resolves(self) -> None:
        assert resolve_label_policy(DEFAULT_LABEL_POLICY) is DEFAULT_LABEL_POLICY

    def test_equal_payload_resolves_to_registered_instance(self) -> None:
        clone = LabelPolicy(**DEFAULT_LABEL_POLICY.model_dump())
        assert resolve_label_policy(clone) is DEFAULT_LABEL_POLICY

    def test_unknown_version_is_rejected(self) -> None:
        drifted = LabelPolicy(**{**DEFAULT_LABEL_POLICY.model_dump(), "version": "label-policy-v9"})
        with pytest.raises(ConfigurationError, match="unknown label policy version"):
            resolve_label_policy(drifted)

    def test_payload_drift_on_published_version_is_rejected(self) -> None:
        # 就地改语义（embargo）而不升版本：历史 dataset_hash 将无法解释，必须拒绝
        drifted = LabelPolicy(**{**DEFAULT_LABEL_POLICY.model_dump(), "embargo_sessions": 6})
        with pytest.raises(
            ConfigurationError, match="payload differs from the versioned definition"
        ):
            resolve_label_policy(drifted)


class TestLabelPolicyFromExperimentConfig:
    def test_default_and_matching_embargo(self) -> None:
        assert label_policy_from_experiment_config() is DEFAULT_LABEL_POLICY
        assert (
            label_policy_from_experiment_config(embargo_sessions=DEFAULT_EMBARGO_SESSIONS)
            is DEFAULT_LABEL_POLICY
        )

    def test_mismatched_embargo_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError) as excinfo:
            label_policy_from_experiment_config(embargo_sessions=3)
        message = str(excinfo.value)
        assert "embargo_sessions=3" in message
        assert "升版本" in message

    def test_unknown_version_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="unknown label policy version"):
            label_policy_from_experiment_config(label_policy_version="label-policy-v9")


class TestDirectionLabel:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0.10, "BULLISH"),
            (0.02, "BULLISH"),  # 边界值归入较高一类
            (0.019999, "NEUTRAL"),
            (0.0, "NEUTRAL"),
            (-0.02, "NEUTRAL"),
            (-0.020001, "BEARISH"),
            (-0.5, "BEARISH"),
        ],
    )
    def test_thresholds(self, value: float, expected: str) -> None:
        assert direction_label(value, DEFAULT_LABEL_POLICY) == expected

    def test_nan_is_rejected(self) -> None:
        with pytest.raises(DataQualityError, match="must not be NaN"):
            direction_label(float("nan"), DEFAULT_LABEL_POLICY)


class TestBuildWalkForwardDataset:
    def test_segments_and_origins(self) -> None:
        dataset = build_dataset()
        assert dataset.version == FORECAST_DATASET_VERSION
        assert dataset.lookback_bars == LOOKBACK
        assert dataset.symbols == (SYMBOL,)
        assert [segment.name for segment in dataset.segments] == ["train", "validation", "test"]
        assert dataset.segments[0].start_index == LOOKBACK
        assert (
            dataset.segments[1].start_index - dataset.segments[0].end_index - 1
            == DEFAULT_EMBARGO_SESSIONS
        )
        origins = [origin_ for segment in dataset.segments for origin_ in segment.origins]
        assert len(origins) == 16
        assert all(origin_.symbol == SYMBOL for origin_ in origins)
        assert all(len(origin_.label_sessions) == DEFAULT_HORIZON_SESSIONS for origin_ in origins)
        assert all(origin_.label_sessions[0] > origin_.market_date for origin_ in origins)
        assert all(origin_.knowledge_cutoff.date() == origin_.market_date for origin_ in origins)

    def test_provenance_and_cutoff_policy_are_recorded(self) -> None:
        dataset = build_dataset(cutoff_policy="market_close")
        assert dataset.calendar_exchange == EXCHANGE
        assert dataset.calendar_source == SOURCE
        assert dataset.cutoff_policy == "market_close"
        assert dataset.segments[0].origins[0].knowledge_cutoff.hour == 15

    def test_symbols_are_canonicalized(self) -> None:
        dataset = build_dataset(symbols=(SYMBOL_B, SYMBOL))
        assert dataset.symbols == (SYMBOL_B, SYMBOL)
        # symbol-major：一个 symbol 的所有 session 连续，且 symbol 升序
        first_segment = dataset.segments[0]
        symbols_seen = [origin_.symbol for origin_ in first_segment.origins]
        assert symbols_seen == [SYMBOL_B] * len(first_segment.sessions) + [SYMBOL] * len(
            first_segment.sessions
        )
        assert [origin_.market_date for origin_ in first_segment.origins[:2]] == list(
            first_segment.sessions[:2]
        )

    def test_two_symbols_share_sessions_and_split_origins(self) -> None:
        dataset = build_dataset(symbols=(SYMBOL, SYMBOL_B))
        for segment in dataset.segments:
            assert {origin_.market_date for origin_ in segment.origins} == set(segment.sessions)

    def test_timeline_length_must_match_plan(self) -> None:
        with pytest.raises(ConfigurationError, match="requires exactly 32"):
            build_dataset(count=REQUIRED_SESSIONS - 1)

    def test_timeline_must_be_sessions_of_the_calendar(self) -> None:
        # 日历缺一个 session（模拟节假日），时间轴里却有它 → 显式失败，不静默顺延
        holiday = date(2026, 6, 4)
        cal = StaticTradingCalendar(
            exchange=EXCHANGE,
            source=SOURCE,
            sessions=tuple(day for day in weekdays(count=REQUIRED_SESSIONS + 10) if day != holiday),
        )
        with pytest.raises(ConfigurationError, match="is not a market session"):
            build_dataset(cal=cal)

    def test_calendar_must_cover_the_label_horizon(self) -> None:
        # 日历恰好只覆盖时间轴：最后一个 origin 的 label 窗口越界 → 显式失败
        with pytest.raises(CalendarError, match="coverage ends at"):
            build_dataset(cal=calendar(REQUIRED_SESSIONS))

    @pytest.mark.parametrize("lookback_bars", [0, -1])
    def test_lookback_bars_must_be_positive(self, lookback_bars: int) -> None:
        with pytest.raises(ConfigurationError, match="lookback_bars must be >= 1"):
            build_dataset(lookback_bars=lookback_bars)

    def test_explicit_cutoff_policy_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="cutoff_policy='explicit'"):
            build_dataset(cutoff_policy="explicit")

    def test_empty_symbols_are_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="symbols must not be empty"):
            build_dataset(symbols=())

    @pytest.mark.parametrize("symbol", ["60000", "sh600000", "60000a"])
    def test_invalid_symbols_are_rejected(self, symbol: str) -> None:
        with pytest.raises(ConfigurationError):
            build_dataset(symbols=(symbol,))

    def test_unknown_label_policy_version_is_rejected(self) -> None:
        drifted = LabelPolicy(**{**DEFAULT_LABEL_POLICY.model_dump(), "version": "label-policy-v9"})
        with pytest.raises(ConfigurationError, match="unknown label policy version"):
            build_dataset(label_policy=drifted)

    def test_duplicate_symbols_are_rejected(self) -> None:
        # 静默去重会让「传了两次同一个 symbol」与「传了一次」共享同一 dataset_hash（ADR-010）。
        with pytest.raises(ConfigurationError, match="symbols must be unique"):
            build_dataset(symbols=(SYMBOL, SYMBOL))

    def test_history_prefix_is_lookback_bars(self) -> None:
        # 时间轴前 lookback_bars 个 session 不是 origin（它们是首个 origin 的 history 前缀）
        dataset = build_dataset()
        assert dataset.segments[0].first_session == dataset.sessions[LOOKBACK]
        assert dataset.sessions[LOOKBACK - 1] not in dataset.segments[0].sessions


class TestDatasetHash:
    def test_golden_hash(self) -> None:
        assert build_dataset().dataset_hash == GOLDEN_DATASET_HASH
        assert is_sha256_hex(GOLDEN_DATASET_HASH)

    def test_hash_is_stable_across_rebuilds(self) -> None:
        assert build_dataset().dataset_hash == build_dataset().dataset_hash

    def test_symbol_order_does_not_change_hash(self) -> None:
        assert (
            build_dataset(symbols=(SYMBOL, SYMBOL_B)).dataset_hash
            == build_dataset(symbols=(SYMBOL_B, SYMBOL)).dataset_hash
        )

    def test_hash_binds_label_policy_payload(self) -> None:
        golden = build_dataset().dataset_hash
        drifted = LabelPolicy(**{**DEFAULT_LABEL_POLICY.model_dump(), "embargo_sessions": 6})
        assert self._hash_with_policy(drifted) != golden
        tighter = LabelPolicy(**{**DEFAULT_LABEL_POLICY.model_dump(), "horizon_sessions": 3})
        assert self._hash_with_policy(tighter) != golden
        thresholds = LabelPolicy(
            **{
                **DEFAULT_LABEL_POLICY.model_dump(),
                "direction_thresholds": (
                    DirectionThreshold(label="BEARISH", threshold=None),
                    DirectionThreshold(label="NEUTRAL", threshold=-0.01),
                    DirectionThreshold(label="BULLISH", threshold=0.01),
                ),
            }
        )
        assert self._hash_with_policy(thresholds) != golden

    def test_hash_binds_calendar_provenance(self) -> None:
        # ADR-009 §4：任何进入 artifact 的时间轴必须能回答「哪个日历、来自哪里」，
        # 因此 exchange / source 必须进 dataset_hash（否则同名时间轴的两个日历会撞 hash）。
        dataset = build_dataset()
        other_exchange = build_dataset(
            cal=StaticTradingCalendar(
                exchange="SZSE", source=SOURCE, sessions=weekdays(count=REQUIRED_SESSIONS + 10)
            )
        )
        other_source = build_dataset(
            cal=StaticTradingCalendar(
                exchange=EXCHANGE,
                source="other-fixture",
                sessions=weekdays(count=REQUIRED_SESSIONS + 10),
            )
        )
        assert other_exchange.dataset_hash != dataset.dataset_hash
        assert other_source.dataset_hash != dataset.dataset_hash

    def test_hash_binds_timeline_and_lookback_and_cutoff(self) -> None:
        dataset = build_dataset()
        # 时间轴整体后移一个 session（lookback 同步缩短以保持总长）
        shifted = self._rebuild(
            sessions=timeline()[1:], lookback_bars=LOOKBACK - 1, count=REQUIRED_SESSIONS - 1
        )
        assert shifted.dataset_hash != dataset.dataset_hash
        assert self._rebuild(cutoff_policy="market_close").dataset_hash != dataset.dataset_hash
        assert (
            self._rebuild(
                lookback_bars=LOOKBACK + 1,
                sessions=weekdays(count=REQUIRED_SESSIONS + 1),
                count=REQUIRED_SESSIONS + 1,
            ).dataset_hash
            != dataset.dataset_hash
        )
        assert self._rebuild(symbols=(SYMBOL, SYMBOL_B)).dataset_hash != dataset.dataset_hash

    def test_hash_binds_walk_forward_version(self) -> None:
        # WALK_FORWARD_VERSION 必须参与 payload：换分段算术规则本身就要换 dataset_hash。
        dataset = build_dataset()

        def digest(walk_forward_version: str) -> str:
            return compute_dataset_hash(
                version=dataset.version,
                label_policy=dataset.label_policy,
                calendar_exchange=dataset.calendar_exchange,
                calendar_source=dataset.calendar_source,
                cutoff_policy=dataset.cutoff_policy,
                lookback_bars=dataset.lookback_bars,
                symbols=dataset.symbols,
                sessions=dataset.sessions,
                segments=dataset.segments,
                walk_forward_version=walk_forward_version,
            )

        assert digest(WALK_FORWARD_VERSION) == GOLDEN_DATASET_HASH == dataset.dataset_hash
        assert digest("walk-forward-v2") != GOLDEN_DATASET_HASH

    def _hash_with_policy(self, policy: LabelPolicy) -> str:
        dataset = build_dataset()
        return compute_dataset_hash(
            version=dataset.version,
            label_policy=policy,
            calendar_exchange=dataset.calendar_exchange,
            calendar_source=dataset.calendar_source,
            cutoff_policy=dataset.cutoff_policy,
            lookback_bars=dataset.lookback_bars,
            symbols=dataset.symbols,
            sessions=dataset.sessions,
            segments=dataset.segments,
        )

    def _rebuild(
        self, *, sessions: tuple[date, ...] | None = None, **overrides: object
    ) -> WalkForwardDataset:
        fields: dict[str, object] = {
            "symbols": (SYMBOL,),
            "plan": PLAN,
            "label_policy": DEFAULT_LABEL_POLICY,
            "lookback_bars": LOOKBACK,
            "cutoff_policy": "same_day_evening",
            "count": REQUIRED_SESSIONS,
        }
        fields.update(overrides)
        if sessions is not None:
            fields["days"] = sessions
            fields["count"] = len(sessions)
        return build_dataset(**fields)  # type: ignore[arg-type]


def payload(dataset: WalkForwardDataset, **overrides: object) -> dict[str, object]:
    """重建 dataset 的构造参数：去掉 computed 的 dataset_hash，再叠加 overrides。"""
    fields = dataset.model_dump(exclude={"dataset_hash"})
    fields.update(overrides)
    return fields


class TestDatasetModelValidation:
    def test_first_segment_must_start_at_lookback(self) -> None:
        dataset = build_dataset()
        with pytest.raises(ValueError, match="must precede the first origin"):
            WalkForwardDataset(**payload(dataset, lookback_bars=LOOKBACK + 1))

    def test_last_segment_must_end_at_timeline_end(self) -> None:
        dataset = build_dataset()
        with pytest.raises(ValueError, match="last segment ends at"):
            WalkForwardDataset(
                **payload(
                    dataset,
                    sessions=(*dataset.sessions, dataset.sessions[-1] + timedelta(days=1)),
                )
            )

    def test_unknown_dataset_version_is_rejected(self) -> None:
        dataset = build_dataset()
        with pytest.raises(ValueError, match="unknown dataset version"):
            WalkForwardDataset(**payload(dataset, version="walk-forward-dataset-v9"))

    def test_origin_count_must_match_sessions(self) -> None:
        dataset = build_dataset()
        segments = [dict(segment) for segment in payload(dataset)["segments"]]
        segments[0]["origins"] = segments[0]["origins"][:-1]
        with pytest.raises(ValueError, match="must hold one origin per"):
            WalkForwardDataset(**payload(dataset, segments=segments))

    def test_duplicate_origin_is_rejected(self) -> None:
        dataset = build_dataset()
        segments = [dict(segment) for segment in payload(dataset)["segments"]]
        origins = list(segments[0]["origins"])
        origins[1] = origins[0]
        segments[0]["origins"] = origins
        with pytest.raises(ValueError, match="duplicate origin"):
            WalkForwardDataset(**payload(dataset, segments=segments))

    def test_drifted_label_policy_is_rejected(self) -> None:
        # §28：模型层也必须走 resolve_label_policy，否则手工 dataset 可以声明 v1 却用别的 embargo。
        dataset = build_dataset()
        drifted = LabelPolicy(**{**DEFAULT_LABEL_POLICY.model_dump(), "embargo_sessions": 6})
        with pytest.raises(
            ConfigurationError, match="payload differs from the versioned definition"
        ):
            WalkForwardDataset(**payload(dataset, label_policy=drifted))

    def test_origin_symbols_must_be_in_dataset_symbols(self) -> None:
        dataset = build_dataset()
        segments = [dict(segment) for segment in payload(dataset)["segments"]]
        segments[0]["origins"] = [
            origin(entry["market_date"], symbol=SYMBOL_B).model_dump()
            for entry in segments[0]["origins"]
        ]
        with pytest.raises(ValueError, match="references symbols not in dataset symbols"):
            WalkForwardDataset(**payload(dataset, segments=segments))

    def test_direction_threshold_payload_covers_all_fields(self) -> None:
        threshold = DEFAULT_DIRECTION_THRESHOLDS[1]
        assert threshold.hashing_payload() == {"label": "NEUTRAL", "threshold": -0.02}

    def test_dataset_is_frozen(self) -> None:
        dataset = build_dataset()
        with pytest.raises(ValidationError):
            dataset.lookback_bars = 1  # type: ignore[misc]

    def test_dataset_hash_is_computed_not_stored(self) -> None:
        payload = build_dataset().model_dump()
        assert "dataset_hash" in payload
        assert payload["dataset_hash"] == GOLDEN_DATASET_HASH


class TestBuildLabel:
    def test_labeled_bullish(self) -> None:
        origin_ = origin(timeline()[LOOKBACK])
        label = build_label(
            origin=origin_,
            policy=DEFAULT_LABEL_POLICY,
            origin_close=100.0,
            future_bars=label_bars(origin_.label_sessions, final_close=110.0),
            data_coverage_end=origin_.label_sessions[-1],
        )
        assert label.status == "LABELED"
        assert label.horizon_return == pytest.approx(0.1)
        assert label.direction == "BULLISH"
        assert label.observed_sessions == DEFAULT_HORIZON_SESSIONS
        assert label.label_policy_version == LABEL_POLICY_VERSION
        assert label.symbol == SYMBOL and label.market_date == origin_.market_date

    @pytest.mark.parametrize(
        ("final_close", "expected"),
        [(110.0, "BULLISH"), (100.0, "NEUTRAL"), (90.0, "BEARISH")],
    )
    def test_labeled_direction(self, final_close: float, expected: str) -> None:
        origin_ = origin(timeline()[LOOKBACK])
        label = build_label(
            origin=origin_,
            policy=DEFAULT_LABEL_POLICY,
            origin_close=100.0,
            future_bars=label_bars(origin_.label_sessions, final_close=final_close),
            data_coverage_end=origin_.label_sessions[-1],
        )
        assert label.direction == expected

    def test_suspended_when_a_session_has_no_bar(self) -> None:
        origin_ = origin(timeline()[LOOKBACK])
        bars = label_bars(origin_.label_sessions)
        label = build_label(
            origin=origin_,
            policy=DEFAULT_LABEL_POLICY,
            origin_close=100.0,
            future_bars=(bars[0], *bars[2:]),
            data_coverage_end=origin_.label_sessions[-1],
        )
        assert label.status == "SUSPENDED"
        assert label.horizon_return is None and label.direction is None
        assert label.observed_sessions == DEFAULT_HORIZON_SESSIONS - 1

    def test_suspended_bar_is_treated_as_missing(self) -> None:
        origin_ = origin(timeline()[LOOKBACK])
        bars = list(label_bars(origin_.label_sessions))
        bars[1] = bar(origin_.label_sessions[1], 100.0, trade_status="0")
        label = build_label(
            origin=origin_,
            policy=DEFAULT_LABEL_POLICY,
            origin_close=100.0,
            future_bars=tuple(bars),
            data_coverage_end=origin_.label_sessions[-1],
        )
        assert label.status == "SUSPENDED"
        assert label.observed_sessions == DEFAULT_HORIZON_SESSIONS - 1

    def test_insufficient_when_horizon_crosses_data_end(self) -> None:
        origin_ = origin(timeline()[LOOKBACK])
        bars = label_bars(origin_.label_sessions)
        label = build_label(
            origin=origin_,
            policy=DEFAULT_LABEL_POLICY,
            origin_close=100.0,
            future_bars=bars,
            data_coverage_end=origin_.label_sessions[-2],
        )
        assert label.status == "INSUFFICIENT_FUTURE_BARS"
        assert label.horizon_return is None and label.direction is None
        # 数据末端优先于停牌判定：即使手上有全部 bar 也不能标注（数据还没发布完）
        assert label.observed_sessions == DEFAULT_HORIZON_SESSIONS

    def test_empty_future_bars_is_suspended_not_zero_return(self) -> None:
        origin_ = origin(timeline()[LOOKBACK])
        label = build_label(
            origin=origin_,
            policy=DEFAULT_LABEL_POLICY,
            origin_close=100.0,
            future_bars=(),
            data_coverage_end=origin_.label_sessions[-1],
        )
        assert label.status == "SUSPENDED"
        assert label.horizon_return is None and label.observed_sessions == 0

    @pytest.mark.parametrize("origin_close", [0.0, -1.0])
    def test_non_positive_origin_close_is_rejected(self, origin_close: float) -> None:
        origin_ = origin(timeline()[LOOKBACK])
        with pytest.raises(DataQualityError, match="origin_close must be > 0"):
            build_label(
                origin=origin_,
                policy=DEFAULT_LABEL_POLICY,
                origin_close=origin_close,
                future_bars=(),
                data_coverage_end=origin_.label_sessions[-1],
            )

    @pytest.mark.parametrize("origin_close", [float("nan"), float("inf")])
    def test_non_finite_origin_close_is_rejected(self, origin_close: float) -> None:
        origin_ = origin(timeline()[LOOKBACK])
        with pytest.raises(DataQualityError, match="must be finite"):
            build_label(
                origin=origin_,
                policy=DEFAULT_LABEL_POLICY,
                origin_close=origin_close,
                future_bars=(),
                data_coverage_end=origin_.label_sessions[-1],
            )

    def test_datetime_data_coverage_end_is_rejected(self) -> None:
        origin_ = origin(timeline()[LOOKBACK])
        with pytest.raises(ConfigurationError, match="must be a date, not datetime"):
            build_label(
                origin=origin_,
                policy=DEFAULT_LABEL_POLICY,
                origin_close=100.0,
                future_bars=(),
                data_coverage_end=datetime(2026, 6, 10, 15, 0, tzinfo=CN_TZ),  # type: ignore[arg-type]
            )

    def test_foreign_symbol_bar_is_rejected(self) -> None:
        origin_ = origin(timeline()[LOOKBACK])
        with pytest.raises(DataQualityError, match="does not match origin symbol"):
            build_label(
                origin=origin_,
                policy=DEFAULT_LABEL_POLICY,
                origin_close=100.0,
                future_bars=(bar(origin_.label_sessions[0], 100.0, symbol=SYMBOL_B),),
                data_coverage_end=origin_.label_sessions[-1],
            )

    def test_out_of_window_bar_is_rejected(self) -> None:
        origin_ = origin(timeline()[LOOKBACK])
        stray = bar(timeline()[LOOKBACK - 1], 100.0)
        with pytest.raises(DataQualityError, match="outside origin"):
            build_label(
                origin=origin_,
                policy=DEFAULT_LABEL_POLICY,
                origin_close=100.0,
                future_bars=(stray,),
                data_coverage_end=origin_.label_sessions[-1],
            )

    def test_duplicate_bar_is_rejected(self) -> None:
        origin_ = origin(timeline()[LOOKBACK])
        bars = label_bars(origin_.label_sessions)
        with pytest.raises(DataQualityError, match="duplicate future bar"):
            build_label(
                origin=origin_,
                policy=DEFAULT_LABEL_POLICY,
                origin_close=100.0,
                future_bars=(bars[0], bars[0], *bars[1:]),
                data_coverage_end=origin_.label_sessions[-1],
            )

    def test_duplicate_bar_is_rejected_regardless_of_suspension_order(self) -> None:
        # 重复 session 的判定不能依赖停牌过滤的顺序：(停牌, 有效) 与 (有效, 停牌) 必须一样拒绝。
        origin_ = origin(timeline()[LOOKBACK])
        session = origin_.label_sessions[1]
        suspended = bar(session, 100.0, trade_status="0")
        valid = bar(session, 100.0)
        for future_bars in ((suspended, valid), (valid, suspended)):
            with pytest.raises(DataQualityError, match="duplicate future bar"):
                build_label(
                    origin=origin_,
                    policy=DEFAULT_LABEL_POLICY,
                    origin_close=100.0,
                    future_bars=future_bars,
                    data_coverage_end=origin_.label_sessions[-1],
                )

    def test_unimplemented_label_policy_payload_is_rejected(self, monkeypatch) -> None:
        # 新登记一个改了 price_field 的版本时，build_label 必须显式拒绝，不能按旧口径静默算。
        origin_ = origin(timeline()[LOOKBACK])
        variant = LabelPolicy.model_construct(
            **{
                **DEFAULT_LABEL_POLICY.model_dump(),
                "version": "label-policy-v2",
                "price_field": "high",
            }
        )
        monkeypatch.setattr(
            "kronos_ai.evaluation.dataset.LABEL_POLICIES",
            {**LABEL_POLICIES, variant.version: variant},
        )
        with pytest.raises(ConfigurationError, match="price_field 'high' is not implemented"):
            build_label(
                origin=origin_,
                policy=variant,
                origin_close=100.0,
                future_bars=(),
                data_coverage_end=origin_.label_sessions[-1],
            )

    def test_unknown_label_policy_version_is_rejected(self) -> None:
        origin_ = origin(timeline()[LOOKBACK])
        drifted = LabelPolicy(**{**DEFAULT_LABEL_POLICY.model_dump(), "version": "label-policy-v9"})
        with pytest.raises(ConfigurationError, match="unknown label policy version"):
            build_label(
                origin=origin_,
                policy=drifted,
                origin_close=100.0,
                future_bars=(),
                data_coverage_end=origin_.label_sessions[-1],
            )

    def test_label_is_deterministic(self) -> None:
        origin_ = origin(timeline()[LOOKBACK])
        kwargs = {
            "origin": origin_,
            "policy": DEFAULT_LABEL_POLICY,
            "origin_close": 100.0,
            "future_bars": label_bars(origin_.label_sessions),
            "data_coverage_end": origin_.label_sessions[-1],
        }
        assert build_label(**kwargs) == build_label(**kwargs)

    def test_labels_do_not_enter_dataset_hash(self) -> None:
        # dataset_hash 只覆盖样本集定义（时间轴 / 分段 / label 语义）；标签数值属于数据，
        # 由 benchmark artifact 负责，因此在同一 dataset 上构 label 不会改变它。
        dataset = build_dataset()
        before = dataset.dataset_hash
        for segment in dataset.segments:
            for origin_ in segment.origins:
                build_label(
                    origin=origin_,
                    policy=dataset.label_policy,
                    origin_close=100.0,
                    future_bars=label_bars(origin_.label_sessions),
                    data_coverage_end=dataset.sessions[-1],
                )
        assert dataset.dataset_hash == before == GOLDEN_DATASET_HASH
