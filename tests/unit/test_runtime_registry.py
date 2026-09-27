"""RuntimeRegistry：按名字解析 backend、重复/未知名字显式失败（RX-KAI-014，§17）。

替代 v1 的 ModelManager Singleton；这里只验证注册表本身，backend 用 stub。
"""

from __future__ import annotations

from datetime import date, datetime
from typing import cast

import pytest

from kronos_ai.domain.forecast import ForecastRequest, ForecastResult, SamplingConfig
from kronos_ai.domain.market import MarketHistory
from kronos_ai.domain.time import CN_TZ
from kronos_ai.errors import ConfigurationError
from kronos_ai.forecast.base import ForecastBackend
from kronos_ai.registry import RuntimeRegistry


class StubBackend:
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def forecast(
        self, history: MarketHistory, request: ForecastRequest, *, force: bool = False
    ) -> ForecastResult:
        return cast(ForecastResult, None)


def test_register_and_get() -> None:
    registry = RuntimeRegistry()
    backend = StubBackend("kronos")
    registry.register_forecast_backend(backend)

    assert registry.get_forecast_backend("kronos") is backend
    assert registry.forecast_backend_names() == ("kronos",)


def test_names_are_sorted() -> None:
    registry = RuntimeRegistry()
    for name in ("zoo", "alpha", "kronos"):
        registry.register_forecast_backend(StubBackend(name))
    assert registry.forecast_backend_names() == ("alpha", "kronos", "zoo")


def test_duplicate_registration_fails() -> None:
    registry = RuntimeRegistry()
    registry.register_forecast_backend(StubBackend("kronos"))
    with pytest.raises(ConfigurationError, match="already registered"):
        registry.register_forecast_backend(StubBackend("kronos"))


def test_unknown_name_lists_registered() -> None:
    registry = RuntimeRegistry()
    registry.register_forecast_backend(StubBackend("kronos"))
    with pytest.raises(ConfigurationError, match="kronos"):
        registry.get_forecast_backend("nope")


def test_unknown_name_with_empty_registry() -> None:
    with pytest.raises(ConfigurationError, match="<none>"):
        RuntimeRegistry().get_forecast_backend("nope")


def test_blank_name_fails() -> None:
    registry = RuntimeRegistry()
    with pytest.raises(ConfigurationError, match="non-empty"):
        registry.register_forecast_backend(StubBackend("   "))


def test_whitespace_is_trimmed_for_lookup() -> None:
    registry = RuntimeRegistry()
    backend = StubBackend("kronos")
    registry.register_forecast_backend(backend)
    assert registry.get_forecast_backend("  kronos  ") is backend


def test_registered_backend_satisfies_protocol() -> None:
    assert isinstance(StubBackend("kronos"), ForecastBackend)


def test_request_fixture_is_valid() -> None:
    # 保证 stub 签名里引用的契约类型可构造（防止 schema 漂移让本文件失去意义）
    request = ForecastRequest(
        symbol="600000",
        market_date=date(2026, 9, 25),
        knowledge_cutoff=datetime(2026, 9, 25, 18, 0, tzinfo=CN_TZ),
        sampling=SamplingConfig(seed=1, sample_count=1),
    )
    assert request.horizon == 5
