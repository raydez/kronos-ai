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


class UniverseError(KronosAIError):
    """Universe constituents are unavailable for the requested point in time (§6.2)."""


class ModelLoadError(KronosAIError):
    """Model or tokenizer checkpoint could not be loaded or failed post-load verification.

    Distinct from ModelInferenceError: the failure happens before any inference, and the
    caller must not retry with silently different weights (§17).
    """


class ModelInferenceError(KronosAIError):
    """Forecast backend inference or output parsing failed; no random OHLC."""


class CalendarError(KronosAIError):
    """Trading calendar failed, a date is outside coverage, or it is not a market session."""


class LeakageError(KronosAIError):
    """Future information leaked into an input window or a training segment (§29).

    Distinct from DataQualityError: the data itself is valid, but the point-in-time
    protocol (walk-forward split, embargo, knowledge cutoff) was violated. The caller
    must not repair it by silently dropping samples — the dataset split is wrong.
    """


class ConfigurationError(KronosAIError):
    """Configuration is invalid or internally inconsistent (§32.1)."""


class InsufficientEvidenceError(KronosAIError):
    """There is not enough evidence to answer the question that was asked (§19 / §42).

    Distinct from ConfigurationError (the setup is wrong) and from a negative result (the
    evidence exists and does not support the claim): a Go / Replace verdict computed on too
    few paired origins would read as "the candidate failed" when the truth is "we did not
    test it" (ADR-010 in evaluation form). The caller must collect more evidence, not
    loosen the pre-registered criteria.
    """


class ArtifactError(KronosAIError):
    """A stored artifact is missing, corrupt, or inconsistent with its cache key.

    Distinct from ModelInferenceError: the failure is in persisted state, and the
    caller must not silently fall back to re-inference or synthesized content (§3.2).
    """
