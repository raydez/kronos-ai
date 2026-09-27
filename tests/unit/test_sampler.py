from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest
import torch

from kronos_ai.data.calendar import StaticTradingCalendar
from kronos_ai.domain.forecast import ForecastRequest, SamplingConfig
from kronos_ai.domain.market import MarketBar
from kronos_ai.domain.time import CN_TZ
from kronos_ai.errors import (
    CalendarError,
    ConfigurationError,
    DataQualityError,
    InsufficientHistoryError,
    ModelInferenceError,
)
from kronos_ai.forecast.backends.kronos.runtime import KronosRuntime, KronosRuntimeConfig
from kronos_ai.forecast.backends.kronos.sampler import (
    FEATURE_NAMES,
    KronosSampler,
    RawSampleSet,
    assert_finite_samples,
    build_generator,
    time_stamp_frame,
)
from kronos_ai.forecast.backends.kronos.vendor import calc_time_stamps

MD = date(2026, 9, 25)
CUTOFF = datetime(2026, 9, 25, 18, 0, tzinfo=CN_TZ)
# 注入内存随机初始化模块时没有真实 revision；与 conftest.TINY_PIN 同值（格式合法但显然是假值）
REVISION = "deadbeef" * 5


def runtime(tiny_model: object, tiny_tokenizer: object, **overrides: object) -> KronosRuntime:
    """复用 session 级模块构造 runtime。

    安全性来自 device 固定为 cpu：`.to(cpu, float32)` 是 no-op，不会触发上游
    RotaryPositionalEmbedding 的跨设备缓存问题（见 conftest 的 tiny_runtime_factory
    说明）。需要非 cpu 的用例必须改用 tiny_runtime_factory 新建模块。
    """
    fields: dict[str, object] = {
        "model_revision": REVISION,
        "tokenizer_revision": REVISION,
        "device": "cpu",
        "lookback_bars": 8,
    }
    fields.update(overrides)
    return KronosRuntime(
        model=tiny_model,  # type: ignore[arg-type]
        tokenizer=tiny_tokenizer,  # type: ignore[arg-type]
        config=KronosRuntimeConfig(**fields),  # type: ignore[arg-type]
    )


def request(**overrides: object) -> ForecastRequest:
    fields: dict[str, object] = {
        "symbol": "600000",
        "market_date": MD,
        "knowledge_cutoff": CUTOFF,
        "horizon": 3,
        "sampling": SamplingConfig(seed=7, sample_count=4),
    }
    fields.update(overrides)
    return ForecastRequest(**fields)  # type: ignore[arg-type]


class TestRawSampleSet:
    def test_shape_and_properties(self) -> None:
        values = np.zeros((4, 3, 6))
        raw = RawSampleSet(
            symbol="600000",
            market_date=MD,
            knowledge_cutoff=CUTOFF,
            horizon=3,
            future_sessions=(date(2026, 9, 28), date(2026, 9, 29), date(2026, 9, 30)),
            feature_names=FEATURE_NAMES,
            values=values,
        )
        assert raw.sample_count == 4

    def test_rejects_inconsistent_shapes(self) -> None:
        future = (date(2026, 9, 28), date(2026, 9, 29), date(2026, 9, 30))
        with pytest.raises(ValueError, match="3-dimensional"):
            RawSampleSet("600000", MD, CUTOFF, 3, future, FEATURE_NAMES, np.zeros((4, 3)))
        with pytest.raises(ValueError, match="values horizon"):
            RawSampleSet("600000", MD, CUTOFF, 2, future, FEATURE_NAMES, np.zeros((4, 3, 6)))
        with pytest.raises(ValueError, match="feature width"):
            RawSampleSet("600000", MD, CUTOFF, 3, future, FEATURE_NAMES, np.zeros((4, 3, 5)))
        with pytest.raises(ValueError, match="future_sessions"):
            RawSampleSet("600000", MD, CUTOFF, 3, future[:2], FEATURE_NAMES, np.zeros((4, 3, 6)))

    def test_rejects_non_float64(self) -> None:
        future = (date(2026, 9, 28), date(2026, 9, 29), date(2026, 9, 30))
        with pytest.raises(ValueError, match="float64"):
            RawSampleSet(
                "600000",
                MD,
                CUTOFF,
                3,
                future,
                FEATURE_NAMES,
                np.zeros((4, 3, 6), dtype=np.float32),
            )

    def test_rejects_non_finite_values(self) -> None:
        future = (date(2026, 9, 28), date(2026, 9, 29), date(2026, 9, 30))
        values = np.zeros((4, 3, 6))
        values[2, 1, 3] = np.nan
        with pytest.raises(ValueError, match="finite"):
            RawSampleSet("600000", MD, CUTOFF, 3, future, FEATURE_NAMES, values)

    def test_rejects_empty_sample_axis(self) -> None:
        future = (date(2026, 9, 28), date(2026, 9, 29), date(2026, 9, 30))
        with pytest.raises(ValueError, match="at least one sample"):
            RawSampleSet("600000", MD, CUTOFF, 3, future, FEATURE_NAMES, np.zeros((0, 3, 6)))

    def test_values_are_read_only(self) -> None:
        future = (date(2026, 9, 28), date(2026, 9, 29), date(2026, 9, 30))
        raw = RawSampleSet("600000", MD, CUTOFF, 3, future, FEATURE_NAMES, np.zeros((4, 3, 6)))
        with pytest.raises(ValueError, match="read-only"):
            raw.values[0, 0, 0] = 1.0


class TestTimeStampFrame:
    def test_matches_upstream_calc_time_stamps(self) -> None:
        sessions = [date(2026, 9, 24), date(2026, 9, 25), date(2026, 9, 28)]
        stamps = [
            datetime.combine(day, datetime.min.time(), tzinfo=CN_TZ).replace(hour=15)
            for day in sessions
        ]
        ours = time_stamp_frame(stamps)
        # 上游实现要求 Series（pandas 3.x 的 DatetimeIndex 无 .dt），语义应逐值一致
        upstream = calc_time_stamps(pd.Series(pd.DatetimeIndex(stamps))).values.astype(np.float32)
        assert np.array_equal(ours, upstream)
        assert ours.dtype == np.float32

    def test_column_semantics(self) -> None:
        # 2026-09-25 是周五（weekday=4），15:00
        frame = time_stamp_frame([datetime(2026, 9, 25, 15, 0, tzinfo=CN_TZ)])
        assert frame.tolist() == [[0.0, 15.0, 4.0, 25.0, 9.0]]


class TestGenerator:
    def test_same_seed_same_stream(self) -> None:
        probs = np.linspace(0.1, 0.4, 4).astype(np.float32)

        def draws(gen: torch.Generator) -> list[int]:
            return [
                int(torch.multinomial(torch.from_numpy(probs), 1, generator=gen).item())
                for _ in range(5)
            ]

        assert draws(build_generator(11, "cpu")) == draws(build_generator(11, "cpu"))
        assert draws(build_generator(12, "cpu")) != draws(build_generator(11, "cpu"))

    def test_does_not_touch_global_rng(self) -> None:
        torch.manual_seed(0)
        before = torch.random.get_rng_state()
        build_generator(99, "cpu")
        assert torch.equal(before, torch.random.get_rng_state())


class TestSamplerValidation:
    def test_alignment_symbol_mismatch(
        self,
        tiny_model: object,
        tiny_tokenizer: object,
        session_calendar: StaticTradingCalendar,
        make_history: object,
    ) -> None:
        sampler = KronosSampler(runtime(tiny_model, tiny_tokenizer), calendar=session_calendar)
        history = make_history(symbol="600000")  # type: ignore[operator]
        with pytest.raises(ConfigurationError, match="history symbol"):
            sampler.decode_raw_samples(history, request(symbol="000001"))

    def test_alignment_market_date_mismatch(
        self,
        tiny_model: object,
        tiny_tokenizer: object,
        session_calendar: StaticTradingCalendar,
        make_history: object,
    ) -> None:
        sampler = KronosSampler(runtime(tiny_model, tiny_tokenizer), calendar=session_calendar)
        history = make_history(market_date=date(2026, 9, 24))  # type: ignore[operator]
        with pytest.raises(ConfigurationError, match="history market_date"):
            sampler.decode_raw_samples(history, request())

    def test_alignment_cutoff_mismatch(
        self,
        tiny_model: object,
        tiny_tokenizer: object,
        session_calendar: StaticTradingCalendar,
        make_history: object,
    ) -> None:
        sampler = KronosSampler(runtime(tiny_model, tiny_tokenizer), calendar=session_calendar)
        history = make_history(cutoff=datetime(2026, 9, 25, 15, 0, tzinfo=CN_TZ))  # type: ignore[operator]
        with pytest.raises(ConfigurationError, match="history knowledge_cutoff"):
            sampler.decode_raw_samples(history, request())

    def test_insufficient_history(
        self,
        tiny_model: object,
        tiny_tokenizer: object,
        session_calendar: StaticTradingCalendar,
        make_history: object,
    ) -> None:
        sampler = KronosSampler(runtime(tiny_model, tiny_tokenizer), calendar=session_calendar)
        history = make_history(n_bars=5)  # type: ignore[operator]
        with pytest.raises(InsufficientHistoryError, match="lookback_bars=8"):
            sampler.decode_raw_samples(history, request())

    def test_missing_volume_is_data_quality_error(
        self,
        tiny_model: object,
        tiny_tokenizer: object,
        session_calendar: StaticTradingCalendar,
        make_history: object,
    ) -> None:
        bars = tuple(
            MarketBar(**{**bar.model_dump(), "volume": None})
            for bar in make_history(n_bars=8).bars  # type: ignore[operator]
        )
        history = make_history(bars=bars)  # type: ignore[operator]
        sampler = KronosSampler(runtime(tiny_model, tiny_tokenizer), calendar=session_calendar)
        with pytest.raises(DataQualityError, match="volume/amount"):
            sampler.decode_raw_samples(history, request())

    def test_missing_amount_is_data_quality_error(
        self,
        tiny_model: object,
        tiny_tokenizer: object,
        session_calendar: StaticTradingCalendar,
        make_history: object,
    ) -> None:
        bars = tuple(
            MarketBar(**{**bar.model_dump(), "amount": None})
            for bar in make_history(n_bars=8).bars  # type: ignore[operator]
        )
        history = make_history(bars=bars)  # type: ignore[operator]
        sampler = KronosSampler(runtime(tiny_model, tiny_tokenizer), calendar=session_calendar)
        with pytest.raises(DataQualityError, match="volume/amount"):
            sampler.decode_raw_samples(history, request())

    def test_calendar_coverage_failure_propagates(
        self,
        tiny_model: object,
        tiny_tokenizer: object,
        make_history: object,
    ) -> None:
        # 只覆盖到 market_date 当日的日历：未来 3 个 session 不足
        calendar = StaticTradingCalendar(
            exchange="SSE", source="test-fixture", sessions=(date(2026, 9, 24), date(2026, 9, 25))
        )
        sampler = KronosSampler(runtime(tiny_model, tiny_tokenizer), calendar=calendar)
        with pytest.raises(CalendarError, match="coverage ends at"):
            sampler.decode_raw_samples(make_history(), request())  # type: ignore[operator]

    def test_horizon_guard_precedes_calendar_lookup(
        self,
        tiny_model: object,
        tiny_tokenizer: object,
        make_history: object,
    ) -> None:
        """配置错误与日历覆盖同时失败：报 ConfigurationError，证明 horizon 判定先于日历查询。"""
        calendar = StaticTradingCalendar(
            exchange="SSE", source="test-fixture", sessions=(date(2026, 9, 24), date(2026, 9, 25))
        )
        runtime_ = runtime(tiny_model, tiny_tokenizer, lookback_bars=2, max_context=2)
        sampler = KronosSampler(runtime_, calendar=calendar)
        history = make_history(n_bars=2)  # type: ignore[operator]

        with pytest.raises(ConfigurationError, match="exceeds max_context"):
            sampler.decode_raw_samples(history, request(horizon=3))
        # 同一日历下 horizon 合法时泄漏为 CalendarError：证明上面的顺序是真实顺序
        with pytest.raises(CalendarError, match="coverage ends at"):
            sampler.decode_raw_samples(history, request(horizon=1))

    def test_non_finite_output_fails_explicitly(self) -> None:
        with pytest.raises(ModelInferenceError, match="non-finite"):
            assert_finite_samples(np.array([[np.nan]]), symbol="600000")


class TestSamplerBehaviour:
    def test_returns_raw_samples_with_calendar_axis(
        self,
        tiny_model: object,
        tiny_tokenizer: object,
        session_calendar: StaticTradingCalendar,
        make_history: object,
    ) -> None:
        sampler = KronosSampler(runtime(tiny_model, tiny_tokenizer), calendar=session_calendar)
        raw = sampler.decode_raw_samples(make_history(), request())  # type: ignore[operator]

        assert raw.values.shape == (4, 3, 6)
        assert raw.future_sessions == (date(2026, 9, 28), date(2026, 9, 29), date(2026, 9, 30))
        assert raw.sample_count == 4
        # sample 维必须是真样本而不是同一个值重复：分布信息未被 mean 抹掉（§10）
        assert raw.values.std(axis=0).mean() > 0

    def test_same_seed_same_samples(
        self,
        tiny_model: object,
        tiny_tokenizer: object,
        session_calendar: StaticTradingCalendar,
        make_history: object,
    ) -> None:
        sampler = KronosSampler(runtime(tiny_model, tiny_tokenizer), calendar=session_calendar)
        history = make_history()
        first = sampler.decode_raw_samples(history, request())
        second = sampler.decode_raw_samples(history, request())
        assert np.array_equal(first.values, second.values)

    def test_global_rng_state_untouched(
        self,
        tiny_model: object,
        tiny_tokenizer: object,
        session_calendar: StaticTradingCalendar,
        make_history: object,
    ) -> None:
        """§9：per-run generator 必须覆盖每个 step 的 s1/s2 两次采样，全局 RNG 不变。"""
        sampler = KronosSampler(runtime(tiny_model, tiny_tokenizer), calendar=session_calendar)
        torch.manual_seed(1234)
        before = torch.random.get_rng_state()
        sampler.decode_raw_samples(make_history(), request())  # type: ignore[operator]
        assert torch.equal(before, torch.random.get_rng_state())

    def test_different_seed_different_samples(
        self,
        tiny_model: object,
        tiny_tokenizer: object,
        session_calendar: StaticTradingCalendar,
        make_history: object,
    ) -> None:
        sampler = KronosSampler(runtime(tiny_model, tiny_tokenizer), calendar=session_calendar)
        history = make_history()
        seed_a = sampler.decode_raw_samples(
            history, request(sampling=SamplingConfig(seed=1, sample_count=4))
        )
        seed_b = sampler.decode_raw_samples(
            history, request(sampling=SamplingConfig(seed=2, sample_count=4))
        )
        assert not np.array_equal(seed_a.values, seed_b.values)

    def test_lookback_window_uses_only_recent_bars(
        self,
        tiny_model: object,
        tiny_tokenizer: object,
        session_calendar: StaticTradingCalendar,
        make_history: object,
    ) -> None:
        sampler = KronosSampler(runtime(tiny_model, tiny_tokenizer), calendar=session_calendar)
        long_history = make_history(n_bars=16)
        short_history = make_history(bars=long_history.bars[-8:])
        long_raw = sampler.decode_raw_samples(long_history, request())
        short_raw = sampler.decode_raw_samples(short_history, request())
        # lookback_bars=8：窗口外的更早 bar 不参与（归一化统计量只来自窗口）
        assert np.array_equal(long_raw.values, short_raw.values)

    def test_horizon_respects_calendar_next_sessions(
        self,
        tiny_model: object,
        tiny_tokenizer: object,
        session_calendar: StaticTradingCalendar,
        make_history: object,
    ) -> None:
        sampler = KronosSampler(runtime(tiny_model, tiny_tokenizer), calendar=session_calendar)
        long_horizon = sampler.decode_raw_samples(make_history(), request(horizon=10))
        assert len(long_horizon.future_sessions) == 10
        assert long_horizon.values.shape == (4, 10, 6)
        assert long_horizon.future_sessions == tuple(session_calendar.next_sessions(MD, 10))
