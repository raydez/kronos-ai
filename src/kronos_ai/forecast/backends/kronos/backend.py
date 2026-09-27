"""KronosForecastBackend：把 runtime / sampler / distribution / cache 组装成 ForecastBackend。

职责（§17 / §15 / RX-KAI-013/014）：

```text
MarketHistory + ForecastRequest
        ↓ build_forecast_artifact_key（同一 calendar，同一 distribution spec）
ForecastArtifactKey
        ↓ cached_forecast（命中即返回，不重复推理）
KronosSampler.decode_raw_samples      ← raw samples（mean 之前）
        ↓ forecast_samples_from_raw
ForecastSample[]
        ↓ build_distribution（§13 指标）
ForecastDistribution
        ↓ 组装 provenance
ForecastResult
```

不承担：数据读取（MarketDataProvider）、CLI 参数解析、run/artifact 落盘（RX-KAI-015）。
"""

from __future__ import annotations

from collections.abc import Mapping

from kronos_ai.data.calendar import TradingCalendar
from kronos_ai.domain.forecast import (
    ForecastRequest,
    ForecastResult,
    SamplingMetadata,
)
from kronos_ai.domain.market import MarketHistory
from kronos_ai.forecast.backends.kronos.runtime import KronosRuntime
from kronos_ai.forecast.backends.kronos.sampler import KronosSampler
from kronos_ai.forecast.cache import (
    ForecastCache,
    build_forecast_artifact_key,
    cached_forecast,
)
from kronos_ai.forecast.distribution import (
    DEFAULT_DISTRIBUTION_SPEC,
    DistributionSpec,
    build_distribution,
    forecast_samples_from_raw,
)

BACKEND_NAME = "kronos"


class KronosForecastBackend:
    """§17 ForecastBackend 的 Kronos 实现；同步、无全局单例。

    ``cache`` 是可注入的（默认不缓存）：缓存键要求模型身份与未来时间轴，二者都由本
    层持有，因此缓存的接入点在这里而不是外部调用方。``force=True`` 只绕过读缓存，
    写入路径仍原子（§15）。
    """

    def __init__(
        self,
        runtime: KronosRuntime,
        *,
        calendar: TradingCalendar,
        cache: ForecastCache | None = None,
        distribution_spec: DistributionSpec = DEFAULT_DISTRIBUTION_SPEC,
    ) -> None:
        self._runtime = runtime
        self._calendar = calendar
        self._sampler = KronosSampler(runtime, calendar=calendar)
        self._cache = cache
        self._distribution_spec = distribution_spec

    @property
    def name(self) -> str:
        return BACKEND_NAME

    @property
    def runtime(self) -> KronosRuntime:
        return self._runtime

    @property
    def calendar(self) -> TradingCalendar:
        return self._calendar

    @property
    def distribution_spec(self) -> DistributionSpec:
        return self._distribution_spec

    def identity(self) -> Mapping[str, str]:
        """§15 的模型维；整体转发 runtime.artifact_identity()，不做子集裁剪。"""
        return self._runtime.artifact_identity()

    def forecast(
        self,
        history: MarketHistory,
        request: ForecastRequest,
        *,
        force: bool = False,
    ) -> ForecastResult:
        key = build_forecast_artifact_key(
            history=history,
            request=request,
            model_identity=self.identity(),
            calendar=self._calendar,
            distribution_spec=self._distribution_spec,
        )
        if self._cache is None:
            return self._compute(key.digest, history, request)
        result, _ = cached_forecast(
            self._cache,
            key,
            lambda: self._compute(key.digest, history, request),
            force=force,
        )
        return result

    def _compute(
        self, artifact_id: str, history: MarketHistory, request: ForecastRequest
    ) -> ForecastResult:
        raw = self._sampler.decode_raw_samples(history, request)
        samples = forecast_samples_from_raw(raw)
        distribution = build_distribution(
            raw,
            origin_close=history.bars[-1].close,
            spec=self._distribution_spec,
        )
        return ForecastResult(
            symbol=request.symbol,
            market_date=request.market_date,
            knowledge_cutoff=request.knowledge_cutoff,
            samples=samples,
            distribution=distribution,
            model=self._runtime.metadata(),
            sampling=SamplingMetadata.from_config(request.sampling),
            input_data_hash=history.data_hash,
            artifact_id=artifact_id,
        )
