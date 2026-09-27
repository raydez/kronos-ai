"""raw samples → domain artifacts：ForecastSample / ForecastDistribution（基线文档 §11/§12/§13）。

职责边界（§10/§12）：

- ``forecast_samples_from_raw`` 把 :class:`~kronos_ai.forecast.raw.RawSampleSet`
  的 (sample_count, horizon, feature_count) 矩阵转成可持久化的 ForecastSample；
- ``build_distribution`` 在 samples 之上计算 §13 的指标并组装 §12 的 ForecastDistribution。

本模块不执行模型推理、不做缓存、不组装 ForecastResult 的 provenance（缓存/哈希属 §15，
由 RX-KAI-013 承担）。指标数学定义集中在 :mod:`kronos_ai.domain.forecast` 的版本化
metric registry：本模块只实现 registry 中已声明名称的数学式，不新增名称。

``raw`` 参数按结构消费（属性访问），类型仅在 TYPE_CHECKING 下引入，避免
sampler ↔ distribution 的运行时循环 import。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from kronos_ai.domain.forecast import (
    FORECAST_AGGREGATION_DEFINITION_VERSION,
    FORECAST_METRIC_REGISTRY_VERSION,
    ForecastDistribution,
    ForecastMetricName,
    ForecastPoint,
    ForecastSample,
    QuantileValue,
    ThresholdOperator,
    ThresholdProbability,
)
from kronos_ai.domain.hashing import sha256_hex
from kronos_ai.domain.time import CN_TZ, MARKET_SESSION_CLOSE
from kronos_ai.errors import ModelInferenceError

if TYPE_CHECKING:
    from kronos_ai.forecast.raw import RawSampleSet

DISTRIBUTION_SPEC_VERSION = "distribution-spec-v1"

# ForecastPoint 依赖的必需 feature 列；缺失即显式失败（不静默省略）
_REQUIRED_FEATURES: tuple[str, ...] = ("open", "high", "low", "close")


class ThresholdSpec(BaseModel):
    """ForecastDistribution 中一条 ThresholdProbability 的请求（§12）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: ForecastMetricName
    operator: ThresholdOperator
    threshold: float


class QuantileSpec(BaseModel):
    """ForecastDistribution 中一条 QuantileValue 的请求（§12）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: ForecastMetricName
    quantile: float = Field(gt=0, lt=1)


class DistributionSpec(BaseModel):
    """构建 ForecastDistribution 的显式配置（§12）。

    这是「移除写死 2%」的落地方式：阈值/分位不再是 schema 里的常量，而是版本化配置。
    默认值见 :data:`DEFAULT_DISTRIBUTION_SPEC`；调用方可传入自定义 spec。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = DISTRIBUTION_SPEC_VERSION
    thresholds: tuple[ThresholdSpec, ...] = ()
    quantiles: tuple[QuantileSpec, ...] = ()

    @model_validator(mode="after")
    def _no_duplicates(self) -> DistributionSpec:
        threshold_keys = [(t.metric, t.operator, t.threshold) for t in self.thresholds]
        if len(set(threshold_keys)) != len(threshold_keys):
            raise ValueError("thresholds contains duplicate (metric, operator, threshold)")
        quantile_keys = [(q.metric, q.quantile) for q in self.quantiles]
        if len(set(quantile_keys)) != len(quantile_keys):
            raise ValueError("quantiles contains duplicate (metric, quantile)")
        return self

    def hashing_payload(self) -> dict[str, object]:
        """进入 ``distribution_spec_hash`` 的规范 payload。

        spec（阈值/分位）决定 ForecastDistribution 的字段取值，因此它是 forecast
        身份的一部分：同一 key 复用一条用不同 spec 算出的分布属静默错误产物。
        显式展开每个字段，不依赖声明顺序。
        """
        return {
            "spec_version": self.version,
            "thresholds": [
                {
                    "metric": t.metric,
                    "operator": t.operator,
                    "threshold": t.threshold,
                }
                for t in self.thresholds
            ],
            "quantiles": [{"metric": q.metric, "quantile": q.quantile} for q in self.quantiles],
        }


DEFAULT_DISTRIBUTION_SPEC = DistributionSpec(
    thresholds=(
        ThresholdSpec(metric="horizon_return", operator="gt", threshold=-0.02),
        ThresholdSpec(metric="horizon_return", operator="gt", threshold=0.0),
        ThresholdSpec(metric="horizon_return", operator="gt", threshold=0.02),
        ThresholdSpec(metric="horizon_return", operator="lt", threshold=-0.02),
    ),
    quantiles=tuple(
        QuantileSpec(metric="horizon_return", quantile=q) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
    ),
)


def distribution_spec_hash(spec: DistributionSpec = DEFAULT_DISTRIBUTION_SPEC) -> str:
    """DistributionSpec 的确定性 hash（进 ForecastArtifactKey 与 ForecastDistribution）。

    同时折叠 ``FORECAST_METRIC_REGISTRY_VERSION``（指标数学定义）与
    ``FORECAST_AGGREGATION_DEFINITION_VERSION``（跨样本聚合口径）：二者变化都会改变
    分布取值，必须让旧 artifact 失效（§12/§15）。

    版本号唯一来源是 ``spec.version``（在 ``spec.hashing_payload`` 内）；此处不再另写
    模块级 DISTRIBUTION_SPEC_VERSION，避免同一版本号在 payload 中出现两次、且两处
    可能不一致。
    """
    return sha256_hex(
        {
            "kind": "distribution_spec",
            "aggregation_definition_version": FORECAST_AGGREGATION_DEFINITION_VERSION,
            "metric_definition_version": FORECAST_METRIC_REGISTRY_VERSION,
            "spec": spec.hashing_payload(),
        }
    )


def forecast_samples_from_raw(raw: RawSampleSet) -> tuple[ForecastSample, ...]:
    """把 raw sample 矩阵转成 ForecastSample 序列（§11）。

    未来时间轴直接采用 ``raw.future_sessions``（由 TradingCalendar 生成，个股停牌
    不改变时间轴）；feature 列按名索引，缺失必需列显式失败。
    """
    if len(raw.future_sessions) != raw.horizon:
        raise ModelInferenceError(
            f"raw sample set declares horizon {raw.horizon} but carries "
            f"{len(raw.future_sessions)} future sessions"
        )
    index = _feature_index(raw.feature_names)
    missing = [name for name in _REQUIRED_FEATURES if name not in index]
    if missing:
        raise ModelInferenceError(
            f"raw sample feature set {raw.feature_names} lacks required features {missing}"
        )

    samples: list[ForecastSample] = []
    for sample_id in range(raw.sample_count):
        points = tuple(_point_at(raw, sample_id, step, index) for step in range(raw.horizon))
        samples.append(ForecastSample(sample_id=sample_id, points=points))
    return tuple(samples)


def build_distribution(
    raw: RawSampleSet,
    *,
    origin_close: float,
    spec: DistributionSpec = DEFAULT_DISTRIBUTION_SPEC,
) -> ForecastDistribution:
    """在 raw samples 上计算 §13 指标并组装 ForecastDistribution（§12）。

    Args:
        raw: 单次 run 的 raw samples，价格空间，float64。
        origin_close: forecast origin close ``P_0``（通常为 lookback 窗口最后一根
            bar 的 close）。所有收益类指标以它为基准。
        spec: 阈值/分位请求；默认 :data:`DEFAULT_DISTRIBUTION_SPEC`。

    失败语义：``P_0`` 或任一 close 非正/非有限时抛 :class:`ModelInferenceError`
    （§3.2），不产出静默可疑的分布。
    """
    metrics = _metric_series(raw, origin_close=origin_close)

    horizon_returns = metrics["horizon_return"]
    thresholds = tuple(
        ThresholdProbability(
            metric=t.metric,
            operator=t.operator,
            threshold=t.threshold,
            probability=_probability(metrics[t.metric], t.operator, t.threshold),
        )
        for t in spec.thresholds
    )
    quantiles = tuple(
        QuantileValue(
            metric=q.metric,
            quantile=q.quantile,
            value=float(np.quantile(metrics[q.metric], q.quantile, method="linear")),
        )
        for q in spec.quantiles
    )

    return ForecastDistribution(
        horizon=raw.horizon,
        sample_count=raw.sample_count,
        origin_close=float(origin_close),
        # 聚合口径：expected_return/median_return=逐样本 horizon_return 的 mean/median；
        # forecast_dispersion=总体标准差（ddof=0）；max_drawdown/path_volatility 取均值。
        # 改任一聚合方式必须升级 FORECAST_AGGREGATION_DEFINITION_VERSION（§13）。
        expected_return=float(horizon_returns.mean()),
        median_return=float(np.median(horizon_returns)),
        threshold_probabilities=thresholds,
        quantiles=quantiles,
        forecast_dispersion=float(horizon_returns.std()),
        expected_max_drawdown=float(metrics["max_drawdown"].mean()),
        expected_path_volatility=float(metrics["path_volatility"].mean()),
        distribution_spec_version=spec.version,
        distribution_spec_hash=distribution_spec_hash(spec),
        metric_definition_version=FORECAST_METRIC_REGISTRY_VERSION,
        aggregation_definition_version=FORECAST_AGGREGATION_DEFINITION_VERSION,
    )


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _feature_index(feature_names: Sequence[str]) -> dict[str, int]:
    index: dict[str, int] = {}
    for position, name in enumerate(feature_names):
        if name in index:
            raise ModelInferenceError(f"duplicate feature name {name!r} in {feature_names}")
        index[name] = position
    return index


def _point_at(raw: RawSampleSet, sample_id: int, step: int, index: dict[str, int]) -> ForecastPoint:
    row = raw.values[sample_id, step]
    return ForecastPoint(
        timestamp=_session_close(raw.future_sessions[step]),
        open=float(row[index["open"]]),
        high=float(row[index["high"]]),
        low=float(row[index["low"]]),
        close=float(row[index["close"]]),
        volume=float(row[index["volume"]]) if "volume" in index else None,
        amount=float(row[index["amount"]]) if "amount" in index else None,
    )


def _session_close(day: date) -> datetime:
    return datetime.combine(day, MARKET_SESSION_CLOSE, tzinfo=CN_TZ)


def _close_matrix(raw: RawSampleSet) -> np.ndarray:
    index = _feature_index(raw.feature_names)
    if "close" not in index:
        raise ModelInferenceError(f"raw sample feature set {raw.feature_names} lacks 'close'")
    closes = np.asarray(raw.values[:, :, index["close"]], dtype=np.float64)
    if closes.size == 0:
        raise ModelInferenceError("raw sample set contains no close values")
    if not np.isfinite(closes).all():
        raise ModelInferenceError("raw sample close values are non-finite")
    if (closes <= 0).any():
        raise ModelInferenceError(
            "raw sample close values must be positive to compute return metrics"
        )
    return closes


def _metric_series(raw: RawSampleSet, *, origin_close: float) -> dict[str, np.ndarray]:
    """§13 数学定义；所有指标一次算全，供阈值/分位/汇总共用。"""
    if not np.isfinite(origin_close):
        raise ModelInferenceError("origin_close must be finite")
    if origin_close <= 0:
        raise ModelInferenceError("origin_close must be positive to compute return metrics")

    closes = _close_matrix(raw)
    path = np.concatenate([np.full((closes.shape[0], 1), origin_close), closes], axis=1)
    log_path = np.log(path)
    drawdown = path / np.maximum.accumulate(path, axis=1) - 1.0
    # path_volatility = 逐步 log return 的总体标准差（ddof=0，与 registry 描述一致）。
    # horizon == 1 时只有单个 log return，总体标准差定义为 0（非「无波动」的语义主张），
    # 这是刻意且已版本化的口径，不是缺失数据。
    path_volatility = np.diff(log_path, axis=1).std(axis=1)
    return {
        "horizon_return": closes[:, -1] / origin_close - 1.0,
        "log_return": log_path[:, -1] - log_path[:, 0],
        "max_drawdown": drawdown.min(axis=1),
        "path_volatility": path_volatility,
    }


def _probability(values: np.ndarray, op: ThresholdOperator, threshold: float) -> float:
    if op == "gt":
        mask = values > threshold
    elif op == "gte":
        mask = values >= threshold
    elif op == "lt":
        mask = values < threshold
    elif op == "lte":
        mask = values <= threshold
    else:  # pragma: no cover - Literal 已在类型层收窄，保留显式失败
        raise ValueError(f"unknown threshold operator {op!r}")
    return float(mask.mean())
