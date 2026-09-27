"""RuntimeRegistry：多 backend 生命周期管理（基线文档 §17）。

替代 v1 的 ModelManager Singleton：多模型、可测试、可配置、无全局单例耦合。
注册表按名字解析 backend，未注册的名字显式失败（§3.2），不返回 None、不做隐式默认。
"""

from __future__ import annotations

from kronos_ai.errors import ConfigurationError
from kronos_ai.forecast.base import ForecastBackend


class _Registry[T]:
    """名字 → 实例的只读注册表；重复注册显式失败，避免静默覆盖。"""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str, item: T) -> None:
        key = name.strip()
        if not key:
            raise ConfigurationError(f"{self._kind} name must be non-empty")
        if key in self._items:
            raise ConfigurationError(f"{self._kind} {key!r} is already registered")
        self._items[key] = item

    def get(self, name: str) -> T:
        key = name.strip()
        try:
            return self._items[key]
        except KeyError:
            known = ", ".join(sorted(self._items)) or "<none>"
            raise ConfigurationError(
                f"unknown {self._kind} {name!r}; registered: {known}"
            ) from None

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))


class RuntimeRegistry:
    """§17 的 RuntimeRegistry；forecast / decision backend 各自独立命名空间。

    decision 命名空间随 RX-KAI-022 的 :class:`DecisionBackend` 契约接入；本任务
    （RX-KAI-014）先落地 forecast 侧，避免提前耦合尚未定义的决策契约。
    """

    def __init__(self) -> None:
        self._forecast: _Registry[ForecastBackend] = _Registry("forecast backend")

    def register_forecast_backend(self, backend: ForecastBackend) -> None:
        self._forecast.register(backend.name, backend)

    def get_forecast_backend(self, name: str) -> ForecastBackend:
        return self._forecast.get(name)

    def forecast_backend_names(self) -> tuple[str, ...]:
        return self._forecast.names()
