"""Explicit-failure exception hierarchy (基线文档 §3.2, ADR-010)."""

from __future__ import annotations


class KronosAIError(Exception):
    """Base class for all kronos-ai domain errors."""


class ProviderError(KronosAIError):
    """Data provider failed; no synthetic fallback is permitted."""


class DataQualityError(KronosAIError):
    """Data violated a domain contract invariant."""


class InsufficientHistoryError(KronosAIError):
    """Symbol lacks the minimum history required by policy (§6.4)."""


class ModelInferenceError(KronosAIError):
    """Forecast backend inference or output parsing failed; no random OHLC."""


class CalendarError(KronosAIError):
    """Trading calendar source failed or a date is outside calendar coverage."""


class ConfigurationError(KronosAIError):
    """Configuration is invalid or internally inconsistent (§32.1)."""
