"""KronosForecastBackend：§17 契约 + §15 缓存接入（RX-KAI-014）。

用 tiny 随机权重 runtime 驱动真实 sampler；缓存命中不依赖 GPU/网络，属默认回归集。
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest

from kronos_ai.data.calendar import StaticTradingCalendar
from kronos_ai.domain.forecast import (
    ForecastRequest,
    SamplingConfig,
    SamplingMetadata,
)
from kronos_ai.domain.market import MarketHistory
from kronos_ai.domain.time import CN_TZ
from kronos_ai.forecast.backends.kronos.backend import BACKEND_NAME, KronosForecastBackend
from kronos_ai.forecast.base import ForecastBackend
from kronos_ai.forecast.cache import FileSystemForecastCache, build_forecast_artifact_key

pytestmark = pytest.mark.regression

MARKET_DATE = date(2026, 9, 25)
CUTOFF = datetime(2026, 9, 25, 18, 0, tzinfo=CN_TZ)
LOOKBACK = 16
HORIZON = 3
SAMPLING = SamplingConfig(seed=20260927, sample_count=4)


@pytest.fixture
def backend(
    tiny_runtime_factory: Any, session_calendar: StaticTradingCalendar
) -> KronosForecastBackend:
    return KronosForecastBackend(
        tiny_runtime_factory(lookback_bars=LOOKBACK), calendar=session_calendar
    )


@pytest.fixture
def history(make_history: Any) -> MarketHistory:
    return make_history(n_bars=LOOKBACK)  # type: ignore[no-any-return]


def request() -> ForecastRequest:
    return ForecastRequest(
        symbol="600000",
        market_date=MARKET_DATE,
        knowledge_cutoff=CUTOFF,
        horizon=HORIZON,
        sampling=SAMPLING,
    )


def test_backend_has_stable_name_and_satisfies_protocol(
    backend: KronosForecastBackend,
) -> None:
    assert backend.name == BACKEND_NAME
    assert isinstance(backend, ForecastBackend)


def test_identity_delegates_to_runtime(backend: KronosForecastBackend) -> None:
    assert backend.identity() == backend.runtime.artifact_identity()


def test_forecast_assembles_provenance(
    backend: KronosForecastBackend, history: MarketHistory
) -> None:
    req = request()
    key = build_forecast_artifact_key(
        history=history,
        request=req,
        model_identity=backend.identity(),
        calendar=backend.calendar,
    )
    result = backend.forecast(history, req)

    assert result.artifact_id == key.digest
    assert result.input_data_hash == history.data_hash
    assert result.model == backend.runtime.metadata()
    assert result.sampling == SamplingMetadata.from_config(SAMPLING)
    assert result.distribution.horizon == HORIZON
    assert result.distribution.sample_count == SAMPLING.sample_count
    assert len(result.samples) == SAMPLING.sample_count
    assert result.distribution.origin_close == pytest.approx(history.bars[-1].close)


def test_cache_hit_skips_second_inference(
    tmp_path: Path,
    tiny_runtime_factory: Any,
    session_calendar: StaticTradingCalendar,
    history: MarketHistory,
) -> None:
    cache = FileSystemForecastCache(tmp_path / "cache")
    backend = KronosForecastBackend(
        tiny_runtime_factory(lookback_bars=LOOKBACK), calendar=session_calendar, cache=cache
    )
    calls = 0
    original = backend._sampler.decode_raw_samples  # 测试探针：统计 sampler 调用次数

    def counting(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    backend._sampler.decode_raw_samples = counting  # type: ignore[method-assign]

    first = backend.forecast(history, request())
    second = backend.forecast(history, request())

    assert calls == 1
    assert first == second
    assert len(list(cache.root.rglob("*.json"))) == 1


def test_force_recomputes_but_matches(
    tmp_path: Path,
    tiny_runtime_factory: Any,
    session_calendar: StaticTradingCalendar,
    history: MarketHistory,
) -> None:
    cache = FileSystemForecastCache(tmp_path / "cache")
    backend = KronosForecastBackend(
        tiny_runtime_factory(lookback_bars=LOOKBACK), calendar=session_calendar, cache=cache
    )
    calls = 0
    original = backend._sampler.decode_raw_samples

    def counting(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    backend._sampler.decode_raw_samples = counting  # type: ignore[method-assign]

    first = backend.forecast(history, request())
    cached = backend.forecast(history, request())
    forced = backend.forecast(history, request(), force=True)

    # 第二次命中读缓存（不重跑）；force 绕过读缓存必须重新推理一次
    assert calls == 2
    assert cached == first
    assert forced == first


def test_force_is_accepted_without_cache(
    backend: KronosForecastBackend, history: MarketHistory
) -> None:
    # 无缓存实现接受 force 并忽略，保持调用方统一语义
    result = backend.forecast(history, request(), force=True)
    assert result.distribution.sample_count == SAMPLING.sample_count
