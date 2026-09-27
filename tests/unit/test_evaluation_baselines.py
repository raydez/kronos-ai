"""Forecast baseline（§19）单元测试（RX-KAI-016）。

覆盖四件事：

1. **数学正确性**：每个 baseline 的价格路径与手算/精确有理数期望对齐（golden），
   包括 lookback 窗口截断与 AR(1) 的 OLS 迭代；
2. **契约一致性**：实现 §17 ForecastBackend、产出 §14 provenance、artifact_id 与
   §15 key 完全一致（缓存层会逐维校验，任何漂移都会在这里先炸）；
3. **显式失败**：历史不足、零方差 AR(1)、对齐不一致、非法参数；
4. **可比性前提**：确定性退化分布（dispersion=0、分位全等、阈值概率 ∈ {0,1}），
   且导入 baseline 不加载 torch（benchmark 的廉价参照系必须能在无 GPU 依赖下跑）。
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from kronos_ai.data.calendar import StaticTradingCalendar
from kronos_ai.domain.forecast import ForecastRequest, SamplingConfig, SamplingMetadata
from kronos_ai.domain.hashing import is_sha256_hex
from kronos_ai.domain.market import MarketBar, MarketHistory
from kronos_ai.domain.time import CN_TZ, MARKET_SESSION_CLOSE
from kronos_ai.errors import ConfigurationError, InsufficientHistoryError, ModelInferenceError
from kronos_ai.evaluation.baselines import (
    AR1_BASELINE_NAME,
    BASELINE_NAMES,
    DEFAULT_MOVING_AVERAGE_WINDOW,
    DRIFT_BASELINE_NAME,
    LAST_VALUE_BASELINE_NAME,
    MOVING_AVERAGE_BASELINE_NAME,
    Baseline,
    MovingAverageBaseline,
    build_baseline,
    build_baseline_backends,
)
from kronos_ai.forecast.base import ForecastBackend
from kronos_ai.forecast.cache import FileSystemForecastCache, build_forecast_artifact_key
from kronos_ai.forecast.service import ForecastService
from kronos_ai.registry import RuntimeRegistry

SYMBOL = "600000"
MARKET_DATE = date(2026, 9, 25)
CUTOFF = datetime(2026, 9, 25, 18, 0, tzinfo=CN_TZ)
SAMPLING = SamplingConfig(seed=7, sample_count=4)
LOOKBACK = 32


def _weekdays(start: date, end: date) -> tuple[date, ...]:
    days: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return tuple(days)


def _sessions(count: int) -> tuple[date, ...]:
    """以 MARKET_DATE 结尾、向前数 count 个工作日的序列（升序）。"""
    return _weekdays(MARKET_DATE - timedelta(days=count * 2 + 10), MARKET_DATE)[-count:]


def calendar() -> StaticTradingCalendar:
    """覆盖 past 与 future 两侧：forecast 需要 market_date 之后的 session 时间轴。"""
    return StaticTradingCalendar(
        exchange="SSE",
        source="test-fixture",
        sessions=_weekdays(MARKET_DATE - timedelta(days=900), MARKET_DATE + timedelta(days=900)),
    )


def history(closes: list[float], *, symbol: str = SYMBOL) -> MarketHistory:
    """用给定 close 序列造 history（volume/amount 故意为 None：naive baseline 不需要）。"""
    days = _sessions(len(closes))
    bars = tuple(
        MarketBar(
            symbol=symbol,
            timestamp=datetime.combine(day, MARKET_SESSION_CLOSE, tzinfo=CN_TZ),
            open=value * 0.99,
            high=value * 1.01,
            low=value * 0.98,
            close=value,
            volume=None,
            amount=None,
            trade_status="1",
            adjustment_mode="raw",
            available_at=datetime.combine(day, MARKET_SESSION_CLOSE, tzinfo=CN_TZ),
        )
        for day, value in zip(days, closes, strict=True)
    )
    return MarketHistory(
        symbol=symbol,
        market_date=MARKET_DATE,
        knowledge_cutoff=CUTOFF,
        bars=bars,
        provider="test-fixture",
        dataset_version="test-dataset-v1",
    )


def request(*, horizon: int = 3, sampling: SamplingConfig = SAMPLING) -> ForecastRequest:
    return ForecastRequest(
        symbol=SYMBOL,
        market_date=MARKET_DATE,
        knowledge_cutoff=CUTOFF,
        horizon=horizon,
        sampling=sampling,
    )


def make(name: str, **overrides: object) -> Baseline:
    kwargs: dict[str, object] = {"calendar": calendar(), "lookback_bars": LOOKBACK}
    kwargs.update(overrides)
    return build_baseline(name, **kwargs)  # type: ignore[arg-type]


class TestConstruction:
    def test_all_baselines_satisfy_forecast_backend_protocol(self) -> None:
        for backend in build_baseline_backends(calendar=calendar(), lookback_bars=LOOKBACK):
            assert isinstance(backend, ForecastBackend)

    def test_build_baseline_backends_covers_every_name_in_declared_order(self) -> None:
        backends = build_baseline_backends(calendar=calendar(), lookback_bars=LOOKBACK)
        assert tuple(backend.name for backend in backends) == BASELINE_NAMES
        assert set(BASELINE_NAMES) == {
            LAST_VALUE_BASELINE_NAME,
            DRIFT_BASELINE_NAME,
            MOVING_AVERAGE_BASELINE_NAME,
            AR1_BASELINE_NAME,
        }

    def test_unknown_baseline_name_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="unknown forecast baseline 'nope'"):
            make("nope")

    def test_lookback_bars_must_be_positive(self) -> None:
        with pytest.raises(ConfigurationError, match="lookback_bars must be >= 1"):
            make(LAST_VALUE_BASELINE_NAME, lookback_bars=0)

    def test_lookback_bars_below_min_estimable_bars_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="needs at least 4 bars"):
            make(AR1_BASELINE_NAME, lookback_bars=3)

    def test_direct_moving_average_construction_validates_window(self) -> None:
        # 公开构造器自己也要守门（不能只靠 build_baseline 的前置校验）
        with pytest.raises(ConfigurationError, match="window must be >= 1, got 0"):
            MovingAverageBaseline(calendar=calendar(), lookback_bars=LOOKBACK, window=0)

    def test_moving_average_window_validation(self) -> None:
        with pytest.raises(ConfigurationError, match="window must be >= 1"):
            make(MOVING_AVERAGE_BASELINE_NAME, moving_average_window=0)
        with pytest.raises(ConfigurationError, match="exceeds lookback_bars"):
            make(MOVING_AVERAGE_BASELINE_NAME, moving_average_window=LOOKBACK + 1)
        # 非法窗口无条件校验：非 MA baseline 也不得因「参数用不到」而静默接受
        with pytest.raises(ConfigurationError, match="window must be >= 1"):
            make(DRIFT_BASELINE_NAME, moving_average_window=0)

    def test_moving_average_default_window_is_versioned_constant(self) -> None:
        # golden：默认窗口是接口的一部分（改它就改变了 benchmark 参照系），显式钉住
        assert DEFAULT_MOVING_AVERAGE_WINDOW == 20
        backend = make(MOVING_AVERAGE_BASELINE_NAME)
        assert backend.parameters() == {"window": DEFAULT_MOVING_AVERAGE_WINDOW}

    def test_registry_roundtrip(self) -> None:
        registry = RuntimeRegistry()
        for backend in build_baseline_backends(calendar=calendar(), lookback_bars=LOOKBACK):
            registry.register_forecast_backend(backend)
        assert registry.forecast_backend_names() == tuple(sorted(BASELINE_NAMES))
        assert registry.get_forecast_backend(DRIFT_BASELINE_NAME).name == DRIFT_BASELINE_NAME
        with pytest.raises(ConfigurationError, match="unknown forecast backend"):
            registry.get_forecast_backend("baseline_does_not_exist")


class TestPathMath:
    def test_last_value_repeats_origin_close(self) -> None:
        result = make(LAST_VALUE_BASELINE_NAME, lookback_bars=3).forecast(
            history([10.0, 11.0, 12.0]), request()
        )
        closes = [point.close for point in result.samples[0].points]
        assert closes == [12.0, 12.0, 12.0]
        assert result.distribution.expected_return == 0.0

    def test_drift_projects_log_drift(self) -> None:
        # d = ln(11/10)；P_t = 11 * exp(d·t)
        result = make(DRIFT_BASELINE_NAME, lookback_bars=2).forecast(
            history([10.0, 11.0]), request(horizon=2)
        )
        closes = [point.close for point in result.samples[0].points]
        assert closes == pytest.approx([12.1, 13.31])
        assert result.distribution.expected_return == pytest.approx(1.21 - 1.0)

    def test_drift_uses_only_the_lookback_window(self) -> None:
        """lookback_bars 决定窗口：窗口外的早期价格不得影响漂移。"""
        closes = [1.0, 2.0, 3.0, 10.0, 11.0]
        windowed = make(DRIFT_BASELINE_NAME, lookback_bars=2).forecast(
            history(closes), request(horizon=2)
        )
        full = make(DRIFT_BASELINE_NAME, lookback_bars=5).forecast(
            history(closes), request(horizon=2)
        )
        assert [p.close for p in windowed.samples[0].points] == pytest.approx([12.1, 13.31])
        # 全窗口漂移 d = ln(11)；两者必须不同（否则 lookback_bars 未真正参与计算）
        assert full.samples[0].points[0].close != pytest.approx(12.1)

    def test_moving_average_uses_trailing_window(self) -> None:
        closes = [5.0, 40.0, 20.0, 30.0]
        result = make(
            MOVING_AVERAGE_BASELINE_NAME, lookback_bars=4, moving_average_window=3
        ).forecast(history(closes), request(horizon=2))
        assert [p.close for p in result.samples[0].points] == [30.0, 30.0]

    def test_ar1_iterates_ols_returns(self) -> None:
        """手算 golden：r = (0.01, 0.03, 0.02, 0.05)。

        lagged=(0.01, 0.03, 0.02)，mean=0.02=1/50；current=(0.03, 0.02, 0.05)，
        mean=0.1/3=1/30；Σ(x-x̄)(y-ȳ) = -1e-4，Σ(x-x̄)² = 2e-4
        ⇒ φ = -1/2，截距 c = 1/30 + (1/2)·(1/50) = 1/30 + 1/100 = 13/300。
        以最后观测收益 0.05 为状态起点：r̂₁ = 13/300 - 0.05/2 = 11/600，
        r̂₂ = 13/300 - 11/1200 = 41/1200，累计 11/600 + 41/1200 = 21/400。
        """
        closes = [100.0 * math.exp(x) for x in (0.0, 0.01, 0.04, 0.06, 0.11)]
        result = make(AR1_BASELINE_NAME, lookback_bars=5).forecast(
            history(closes), request(horizon=2)
        )
        origin = closes[-1]
        assert [p.close for p in result.samples[0].points] == pytest.approx(
            [origin * math.exp(11 / 600), origin * math.exp(21 / 400)]
        )

    def test_ar1_direction_follows_negative_slope(self) -> None:
        """φ<0（均值回复）时，正收益之后应给出低于起点的预期（不做符号硬化）。"""
        closes = [100.0 * math.exp(x) for x in (0.0, 0.01, 0.04, 0.06, 0.11)]
        result = make(AR1_BASELINE_NAME, lookback_bars=5).forecast(
            history(closes), request(horizon=1)
        )
        assert result.distribution.expected_return == pytest.approx(math.exp(11 / 600) - 1.0)


class TestFailureModes:
    def test_misaligned_history_is_rejected(self) -> None:
        other = history([10.0, 11.0, 12.0], symbol="000001")
        with pytest.raises(ConfigurationError, match="history symbol"):
            make(LAST_VALUE_BASELINE_NAME, lookback_bars=3).forecast(other, request())

    def test_misaligned_market_date_is_rejected(self) -> None:
        misaligned = ForecastRequest(
            symbol=SYMBOL,
            market_date=MARKET_DATE - timedelta(days=1),
            knowledge_cutoff=CUTOFF - timedelta(days=1),
            horizon=3,
            sampling=SAMPLING,
        )
        with pytest.raises(ConfigurationError, match="history market_date"):
            make(LAST_VALUE_BASELINE_NAME, lookback_bars=3).forecast(
                history([10.0, 11.0, 12.0]), misaligned
            )

    def test_misaligned_knowledge_cutoff_is_rejected(self) -> None:
        misaligned = ForecastRequest(
            symbol=SYMBOL,
            market_date=MARKET_DATE,
            knowledge_cutoff=CUTOFF + timedelta(hours=1),
            horizon=3,
            sampling=SAMPLING,
        )
        with pytest.raises(ConfigurationError, match="history knowledge_cutoff"):
            make(LAST_VALUE_BASELINE_NAME, lookback_bars=3).forecast(
                history([10.0, 11.0, 12.0]), misaligned
            )

    def test_history_shorter_than_lookback_fails_explicitly(self) -> None:
        """与 kronos 同一口径：窗口凑不满就失败，不用短窗口冒充（§6.4）。"""
        with pytest.raises(InsufficientHistoryError, match="3 bars, lookback_bars=32 required"):
            make(LAST_VALUE_BASELINE_NAME).forecast(history([10.0, 11.0, 12.0]), request())
        with pytest.raises(InsufficientHistoryError, match="1 bars, lookback_bars=2 required"):
            make(DRIFT_BASELINE_NAME, lookback_bars=2).forecast(history([10.0]), request())

    def test_moving_average_requires_full_lookback_window(self) -> None:
        # 运行期门槛是 lookback_bars（不是 window）：窗口凑不满就交给 dataset builder 剔除
        with pytest.raises(InsufficientHistoryError, match="3 bars, lookback_bars=32 required"):
            make(MOVING_AVERAGE_BASELINE_NAME, moving_average_window=5).forecast(
                history([10.0, 11.0, 12.0]), request()
            )

    def test_zero_return_variance_ar1_fails(self) -> None:
        flat = [10.0] * 6  # 全部同价 ⇒ 收益恒为 0 ⇒ 斜率不可识别
        with pytest.raises(ModelInferenceError, match="zero return variance"):
            make(AR1_BASELINE_NAME, lookback_bars=6).forecast(history(flat), request())

    def test_non_finite_path_fails_explicitly(self) -> None:
        """+1381 的对数漂移把 exp 推到 overflow（numpy 警告），随后被显式守卫拦下。"""
        with pytest.warns(RuntimeWarning), pytest.raises(ModelInferenceError, match="non-finite"):
            make(DRIFT_BASELINE_NAME, lookback_bars=2).forecast(
                history([1e-300, 1e300]), request(horizon=2)
            )

    def test_non_positive_path_fails_explicitly(self) -> None:
        """对数漂移 -1381 使外推价格下溢到 0；不得把 0 价当成合法预测。"""
        with pytest.raises(ModelInferenceError, match="non-positive"):
            make(DRIFT_BASELINE_NAME, lookback_bars=2).forecast(
                history([1e300, 1e-300]), request(horizon=2)
            )


class TestProvenanceAndDeterminism:
    def test_result_provenance_matches_identity_and_key(self) -> None:
        backend = make(MOVING_AVERAGE_BASELINE_NAME)
        hist = history([float(x) for x in range(10, 50)])
        req = request()
        result = backend.forecast(hist, req)

        identity = backend.identity()
        assert result.model.backend == MOVING_AVERAGE_BASELINE_NAME
        assert result.model.model_id == identity["model_id"] == "baseline:moving_average"
        assert result.model.revision == identity["model_revision"]
        assert result.model.config_hash == identity["config_hash"]
        assert result.model.device == "cpu"
        assert result.input_data_hash == hist.data_hash
        assert result.sampling == SamplingMetadata.from_config(req.sampling)

        key = build_forecast_artifact_key(
            history=hist,
            request=req,
            model_identity=identity,
            calendar=backend.calendar,
            distribution_spec=backend.distribution_spec,
        )
        assert result.artifact_id == key.digest

    def test_repeated_forecast_is_bit_identical(self) -> None:
        backend = make(AR1_BASELINE_NAME)
        hist = history([float(x) for x in range(10, 50)])
        first = backend.forecast(hist, request())
        second = backend.forecast(hist, request())
        assert first.artifact_id == second.artifact_id
        assert json.dumps(first.model_dump(mode="json"), sort_keys=True) == json.dumps(
            second.model_dump(mode="json"), sort_keys=True
        )

    def test_artifact_id_depends_on_baseline_and_sampling(self) -> None:
        hist = history([float(x) for x in range(10, 50)])

        def digest(name: str, **overrides: object) -> str:
            return make(name, **overrides).forecast(hist, request()).artifact_id

        ids = {
            digest(LAST_VALUE_BASELINE_NAME),
            digest(DRIFT_BASELINE_NAME),
            digest(MOVING_AVERAGE_BASELINE_NAME),
            digest(AR1_BASELINE_NAME),
            digest(MOVING_AVERAGE_BASELINE_NAME, moving_average_window=5),
            digest(LAST_VALUE_BASELINE_NAME, lookback_bars=LOOKBACK + 1),
        }
        assert len(ids) == 6
        # 采样参数同样属于身份：sample_count 变化必须产出不同 artifact_id
        other_sampling = make(LAST_VALUE_BASELINE_NAME).forecast(
            hist, request(sampling=SamplingConfig(seed=7, sample_count=3))
        )
        assert other_sampling.artifact_id not in ids

    def test_config_hash_reacts_to_parameters_only(self) -> None:
        assert make(DRIFT_BASELINE_NAME).config_hash() == make(DRIFT_BASELINE_NAME).config_hash()
        assert (
            make(MOVING_AVERAGE_BASELINE_NAME).config_hash()
            != make(MOVING_AVERAGE_BASELINE_NAME, moving_average_window=5).config_hash()
        )
        assert (
            make(LAST_VALUE_BASELINE_NAME).config_hash()
            != make(LAST_VALUE_BASELINE_NAME, lookback_bars=8).config_hash()
        )
        # config_hash 必须是合法 sha256 hex（进 artifact key 前会被逐字校验）
        assert is_sha256_hex(make(LAST_VALUE_BASELINE_NAME).config_hash())

    def test_config_hash_golden(self) -> None:
        """golden：config_hash 的载荷（含 math_version / features）是缓存身份的一部分。

        任何字段增删改都会改变此值；若只因「测试仍然全绿」就放行，两个不同模型会
        在 §15 缓存里互相冒充。
        """
        assert (
            make(DRIFT_BASELINE_NAME).config_hash()
            == "01f7a2f4ced5f023f4f0abf30c00360826c23fa97ab56dcb3959b83d50b27154"
        )
        assert (
            make(MOVING_AVERAGE_BASELINE_NAME).config_hash()
            == "3be8bde44d14d3a3f5e26e648a3dc85149cb8dcf8d12d101db88884720973e28"
        )

    def test_cache_roundtrip_and_force(self, tmp_path: Path) -> None:
        cache = FileSystemForecastCache(tmp_path / "cache")
        backend = make(DRIFT_BASELINE_NAME, lookback_bars=3, cache=cache)
        hist = history([10.0, 11.0, 12.0])

        first = backend.forecast(hist, request())
        paths_after_first = sorted(cache.root.rglob("*.json"))
        assert len(paths_after_first) == 1

        second = backend.forecast(hist, request())
        assert second == first
        assert sorted(cache.root.rglob("*.json")) == paths_after_first

        forced = backend.forecast(hist, request(), force=True)
        assert forced == first
        assert sorted(cache.root.rglob("*.json")) == paths_after_first


class TestDegenerateDistribution:
    def test_samples_are_identical_and_volume_is_absent(self) -> None:
        result = make(DRIFT_BASELINE_NAME, lookback_bars=3).forecast(
            history([10.0, 11.0, 12.0]), request(horizon=2)
        )
        assert len(result.samples) == SAMPLING.sample_count
        assert [sample.sample_id for sample in result.samples] == list(range(SAMPLING.sample_count))
        first = result.samples[0].model_dump()
        for sample in result.samples[1:]:
            assert sample.model_dump() == {**first, "sample_id": sample.sample_id}
        assert result.samples[0].points[0].volume is None
        assert result.samples[0].points[0].amount is None
        # 四个价格特征取同一 close 路径（naive 规则不区分 open/high/low）
        point = result.samples[0].points[0]
        assert (point.open, point.high, point.low, point.close) == (
            point.close,
            point.close,
            point.close,
            point.close,
        )

    def test_distribution_is_a_point_mass(self) -> None:
        result = make(DRIFT_BASELINE_NAME, lookback_bars=3).forecast(
            history([10.0, 11.0, 12.0]), request(horizon=2)
        )
        distribution = result.distribution
        assert distribution.sample_count == SAMPLING.sample_count
        assert distribution.horizon == 2
        assert distribution.forecast_dispersion == 0.0
        assert distribution.median_return == distribution.expected_return
        quantile_values = {q.value for q in distribution.quantiles}
        assert quantile_values == {distribution.expected_return}
        assert all(t.probability in (0.0, 1.0) for t in distribution.threshold_probabilities)
        # horizon_return = +21%（+2% 与 0% 阈值必被跨越，-2% 阈值必然不跨越）
        by_threshold = {
            (t.operator, t.threshold): t.probability for t in distribution.threshold_probabilities
        }
        assert by_threshold[("gt", 0.0)] == 1.0
        assert by_threshold[("gt", 0.02)] == 1.0
        assert by_threshold[("lt", -0.02)] == 0.0


class TestServiceIntegration:
    def test_service_drives_baseline_end_to_end(self) -> None:
        """§17 的价值在此：baseline 与 kronos 在 service 层不可区分。"""
        hist = history([10.0, 11.0, 12.0])

        class _StaticProvider:
            def get_history(
                self, symbol: str, market_date: date, knowledge_cutoff: datetime, lookback_bars: int
            ) -> MarketHistory:
                return hist

            def close(self) -> None:
                return None

        service = ForecastService(
            _StaticProvider(),
            make(LAST_VALUE_BASELINE_NAME, lookback_bars=3),
            lookback_bars=3,
        )
        result = service.run(
            symbol=SYMBOL,
            market_date=MARKET_DATE,
            knowledge_cutoff=CUTOFF,
            sampling=SAMPLING,
            horizon=2,
        )
        assert [point.close for point in result.samples[0].points] == [12.0, 12.0]
        assert result.model.backend == LAST_VALUE_BASELINE_NAME


class TestImportWeight:
    def test_importing_baselines_does_not_load_torch(self) -> None:
        """benchmark 的廉价参照系必须能在无 torch/pandas 的进程里跑（§48）。"""
        code = (
            "import sys; import kronos_ai.evaluation.baselines as b; "
            "assert 'torch' not in sys.modules, 'torch imported'; "
            "assert 'pandas' not in sys.modules, 'pandas imported'; "
            "print(b.BASELINE_NAMES)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert completed.returncode == 0, completed.stderr
        assert "last_value" in completed.stdout

    def test_baseline_forecast_runs_without_torch_or_pandas(self, tmp_path: Path) -> None:
        """不光能 import：完整 forecast（含缓存写入）在无 torch/pandas 进程里跑通。"""
        code = """
import sys
from datetime import date, datetime, timedelta
from kronos_ai.data.calendar import StaticTradingCalendar
from kronos_ai.domain.forecast import ForecastRequest, SamplingConfig
from kronos_ai.domain.market import MarketBar, MarketHistory
from kronos_ai.domain.time import CN_TZ, MARKET_SESSION_CLOSE
from kronos_ai.evaluation.baselines import build_baseline
from kronos_ai.forecast.cache import FileSystemForecastCache

market_date = date(2026, 9, 25)
cutoff = datetime(2026, 9, 25, 18, 0, tzinfo=CN_TZ)
days = [market_date - timedelta(days=offset) for offset in range(4, 0, -1)]
sessions = sorted(
    day for day in (market_date + timedelta(days=offset) for offset in range(-10, 10))
    if day.weekday() < 5
)
calendar = StaticTradingCalendar(exchange="SSE", source="probe", sessions=sessions)
bars = tuple(
    MarketBar(
        symbol="600000",
        timestamp=datetime.combine(day, MARKET_SESSION_CLOSE, tzinfo=CN_TZ),
        open=float(10 + index),
        high=float(10 + index),
        low=float(10 + index),
        close=float(10 + index),
        trade_status="1",
        adjustment_mode="raw",
        available_at=datetime.combine(day, MARKET_SESSION_CLOSE, tzinfo=CN_TZ),
    )
    for index, day in enumerate(days)
)
history = MarketHistory(
    symbol="600000",
    market_date=market_date,
    knowledge_cutoff=cutoff,
    bars=bars,
    provider="probe",
    dataset_version="probe-v1",
)
backend = build_baseline(
    "last_value",
    calendar=calendar,
    lookback_bars=4,
    cache=FileSystemForecastCache(sys.argv[1]),
)
result = backend.forecast(
    history,
    ForecastRequest(
        symbol="600000",
        market_date=market_date,
        knowledge_cutoff=cutoff,
        horizon=2,
        sampling=SamplingConfig(seed=1, sample_count=2),
    ),
)
first = backend.forecast(
    history,
    ForecastRequest(
        symbol="600000",
        market_date=market_date,
        knowledge_cutoff=cutoff,
        horizon=2,
        sampling=SamplingConfig(seed=1, sample_count=2),
    ),
)
assert first == result, "cache roundtrip changed the result"
assert [point.close for point in result.samples[0].points] == [13.0, 13.0]
assert "torch" not in sys.modules, "torch imported"
assert "pandas" not in sys.modules, "pandas imported"
print("ok")
"""
        completed = subprocess.run(
            [sys.executable, "-c", code, str(tmp_path / "cache")],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == "ok"
