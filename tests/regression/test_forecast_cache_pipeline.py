"""端到端回归：KronosSampler → build_distribution → ForecastResult → cached_forecast。

评审发现（RX-KAI-013）：单元测试只用手工 fixture 驱动 ``FileSystemForecastCache``，
从未把「真实推理产物」接进缓存。本文件补上这条链路，验证的是**契约之间的接缝**，
而不是单独某个模块：

- ``build_forecast_artifact_key`` 用的 future_sessions 与 sampler 实际推理用的时间轴一致；
- ``compute`` 回填的 ``artifact_id``/``input_data_hash`` 能通过 ``_verify`` 的逐维校验；
- 第二次调用同一 key 命中磁盘缓存、不再推理（§15「同一 key 不得重复推理」）；
- 落盘 artifact 自包含，换一个 cache 实例（不重跑推理）也能读回同一结果。

缓存命中/落盘不依赖 GPU/网络，因此放在默认执行的 regression 集内（非 integration）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest

from kronos_ai.data.calendar import StaticTradingCalendar
from kronos_ai.domain.forecast import (
    ForecastRequest,
    ForecastResult,
    SamplingConfig,
    SamplingMetadata,
)
from kronos_ai.domain.market import MarketHistory
from kronos_ai.domain.time import CN_TZ
from kronos_ai.forecast.backends.kronos.runtime import KronosRuntime
from kronos_ai.forecast.backends.kronos.sampler import KronosSampler
from kronos_ai.forecast.cache import (
    FileSystemForecastCache,
    ForecastArtifactKey,
    build_forecast_artifact_key,
    cached_forecast,
)
from kronos_ai.forecast.distribution import build_distribution, forecast_samples_from_raw

pytestmark = pytest.mark.regression

MD = date(2026, 9, 25)
CUTOFF = datetime(2026, 9, 25, 18, 0, tzinfo=CN_TZ)
SEED = 20260927
SAMPLE_COUNT = 4
HORIZON = 3
LOOKBACK = 16


@dataclass(frozen=True)
class Pipeline:
    """一次 forecast 的全部输入与两侧实现：真实 runtime + 真实磁盘缓存。"""

    runtime: KronosRuntime
    calendar: StaticTradingCalendar
    history: MarketHistory
    request: ForecastRequest
    key: ForecastArtifactKey
    cache: FileSystemForecastCache

    def compute(self) -> ForecastResult:
        """与未来 CLI 的 ``compute()`` 同构：唯一一次真实推理 + 组装 provenance。"""
        raw = KronosSampler(self.runtime, calendar=self.calendar).decode_raw_samples(
            self.history, self.request
        )
        distribution = build_distribution(raw, origin_close=self.history.bars[-1].close)
        return ForecastResult(
            symbol=self.request.symbol,
            market_date=self.request.market_date,
            knowledge_cutoff=self.request.knowledge_cutoff,
            samples=forecast_samples_from_raw(raw),
            distribution=distribution,
            model=self.runtime.metadata(),
            sampling=SamplingMetadata.from_config(self.request.sampling),
            input_data_hash=self.history.data_hash,
            artifact_id=self.key.digest,
        )


@pytest.fixture
def pipeline(
    tmp_path: Path,
    tiny_runtime_factory: Any,
    session_calendar: StaticTradingCalendar,
    make_history: Any,
) -> Pipeline:
    runtime = tiny_runtime_factory(lookback_bars=LOOKBACK)
    history: MarketHistory = make_history(n_bars=LOOKBACK)
    request = ForecastRequest(
        symbol="600000",
        market_date=MD,
        knowledge_cutoff=CUTOFF,
        horizon=HORIZON,
        sampling=SamplingConfig(seed=SEED, sample_count=SAMPLE_COUNT),
    )
    key = build_forecast_artifact_key(
        history=history,
        request=request,
        model_identity=runtime.artifact_identity(),
        calendar=session_calendar,
    )
    return Pipeline(
        runtime=runtime,
        calendar=session_calendar,
        history=history,
        request=request,
        key=key,
        cache=FileSystemForecastCache(tmp_path / "cache"),
    )


def test_pipeline_result_passes_cache_verification(pipeline: Pipeline) -> None:
    """compute 的产物必须能通过 cache 的身份校验（put 内含 _verify）。"""
    result = pipeline.compute()

    assert result.artifact_id == pipeline.key.digest
    assert result.input_data_hash == pipeline.history.data_hash
    assert result.distribution.origin_close == pytest.approx(pipeline.history.bars[-1].close)
    assert result.distribution.horizon == HORIZON
    assert result.distribution.sample_count == SAMPLE_COUNT
    assert len(result.samples) == SAMPLE_COUNT

    # 不抛异常即代表逐维校验（含 timeline == key.future_sessions）通过
    pipeline.cache.put(pipeline.key, result)
    assert pipeline.cache.path_for(pipeline.key).exists()


def test_cache_hit_skips_second_inference(pipeline: Pipeline) -> None:
    """§15：同一 key 第二次调用必须命中缓存，compute 只执行一次。"""
    calls = 0

    def compute() -> ForecastResult:
        nonlocal calls
        calls += 1
        return pipeline.compute()

    first, first_cached = cached_forecast(pipeline.cache, pipeline.key, compute)
    second, second_cached = cached_forecast(pipeline.cache, pipeline.key, compute)

    assert (first_cached, second_cached) == (False, True)
    assert calls == 1
    assert first == second


def test_force_recomputes_but_stays_consistent(pipeline: Pipeline) -> None:
    """--force 只绕过读缓存：重算结果与首次缓存逐字节等价（同 seed 可复现，§9）。"""
    calls = 0

    def compute() -> ForecastResult:
        nonlocal calls
        calls += 1
        return pipeline.compute()

    first, _ = cached_forecast(pipeline.cache, pipeline.key, compute)
    forced, forced_cached = cached_forecast(pipeline.cache, pipeline.key, compute, force=True)

    assert forced_cached is False
    assert calls == 2
    assert forced == first


def test_artifact_on_disk_is_self_contained(pipeline: Pipeline) -> None:
    """换一个 cache 实例、不重跑推理即可读回同一结果：artifact 自包含、可重放（DoD 13）。"""
    written, _ = cached_forecast(pipeline.cache, pipeline.key, pipeline.compute)

    reopened = FileSystemForecastCache(pipeline.cache.root)
    replayed, was_cached = cached_forecast(
        reopened,
        pipeline.key,
        # compute 若被调用即失败：证明命中来自磁盘 artifact，而非重算
        lambda: pytest.fail("replay must not recompute"),
    )

    assert was_cached is True
    assert replayed == written

    # origin_close 是落盘字段：第三方无需重跑推理即可拿到 P_0 并独立重算指标（§13）
    payload = json.loads(reopened.path_for(pipeline.key).read_text(encoding="utf-8"))
    assert payload["distribution"]["origin_close"] == pytest.approx(pipeline.history.bars[-1].close)
    assert payload["distribution"]["aggregation_definition_version"]
    assert payload["artifact_id"] == pipeline.key.digest
