"""ForecastBackend 契约（基线文档 §17，ADR-012）。

Core 同步接口：CLI / Benchmark worker 直接调用；FastAPI 在外层 submit job → worker →
sync core（§17 / §3.7），不要求本层 async。

约定：

- ``name`` 是 backend 的稳定标识，进入 Run Metadata（§32 ``forecast_backend``）；
  artifact 身份由 §15 ``ForecastArtifactKey`` 的既有维度决定，不含 backend 名；
- ``forecast`` 消费 point-in-time 的 :class:`MarketHistory` 与 :class:`ForecastRequest`，
  返回带完整 provenance 的 :class:`ForecastResult`；
- 任何失败显式抛错（ADR-010），禁止 synthetic fallback；
- 实现可以有额外的可选关键字参数（例如缓存 ``force``），但必须至少兼容本签名。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from kronos_ai.domain.forecast import ForecastRequest, ForecastResult
from kronos_ai.domain.market import MarketHistory


@runtime_checkable
class ForecastBackend(Protocol):
    @property
    def name(self) -> str:
        """稳定 backend 标识，写入 Run Metadata（§32 ``forecast_backend``）。"""
        ...

    def forecast(
        self, history: MarketHistory, request: ForecastRequest, *, force: bool = False
    ) -> ForecastResult:
        """在 point-in-time history 上产出一次 forecast。

        前置条件：``history`` 与 ``request`` 的 symbol / market_date /
        knowledge_cutoff 必须一致（由实现显式校验，§3.2）。

        ``force`` 是 §15 缓存语义的可选开关：实现若不缓存则接收并忽略，但签名允许
        调用方统一表达「绕过读缓存」。有缓存的实现必须只绕过读取，写入依旧原子。
        """
        ...


if TYPE_CHECKING:
    # mypy 结构化校验：Kronos backend 必须满足 §17 契约
    # （runtime_checkable 的 isinstance 只检查成员存在，不校验签名）
    from kronos_ai.forecast.backends.kronos.backend import KronosForecastBackend

    def _contract_anchor(backend: KronosForecastBackend) -> ForecastBackend:
        return backend
