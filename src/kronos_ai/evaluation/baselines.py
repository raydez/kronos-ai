"""Forecast naive baselines（基线文档 §19 / §41 / §48；ADR-014）。

Phase 2 Go/Replace Gate（§19/§42）要求 Kronos 与「不依赖神经网络的强 baseline」比较：
Last Value、Drift、Moving Average，以及一个简单 statistical baseline。本模块实现这
四个确定性点预测器，让 gate 有可证伪的参照系——没有 baseline 的「模型更好」不可证伪。

设计约束
--------

1. **同一份接口**：每个 baseline 实现 :class:`~kronos_ai.forecast.base.ForecastBackend`
   Protocol，因此能进 :class:`~kronos_ai.registry.RuntimeRegistry`、能被 benchmark
   （§48）与 CLI 以同一方式驱动，也共用 §15 的 artifact key / 缓存语义。
2. **同一份样本与指标路径**：baseline 不自己算 MAE/RMSE，而是产出一个
   :class:`~kronos_ai.forecast.raw.RawSampleSet`（确定性路径重复 sample_count 次），
   再走与 Kronos 完全相同的 ``forecast_samples_from_raw`` → ``build_distribution``。
   两条路径的样本/指标代码是同一份，benchmark 中的差异只可能来自预测本身。
3. **退化分布是刻意的**：确定性点预测的跨样本离散度为 0——quantiles 全等、
   ``prob_close_above_threshold`` ∈ {0, 1}、CRPS 等于 MAE 的单调变换。因此
   quantile coverage / CRPS 对 4 个 naive baseline 不构成证据（§49 以 MAE/RMSE/
   Direction Accuracy / Return Correlation 为主判据）。这里不为了让指标「好看」
   而给 baseline 注入人为噪声（ADR-010）。
4. **不引入随机性**：seed / temperature / top_k / top_p 被接受并记录进 provenance
   （接口一致），但不参与计算；同一输入必须逐位重现同一输出。

baseline 只消费 ``MarketHistory`` 中已存在的 bar（provider 已按 knowledge cutoff 截断），
不做任何未来信息回填。
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from kronos_ai.data.calendar import TradingCalendar
from kronos_ai.domain.forecast import (
    ForecastRequest,
    ForecastResult,
    ModelMetadata,
    SamplingMetadata,
)
from kronos_ai.domain.hashing import sha256_hex
from kronos_ai.domain.market import MarketHistory
from kronos_ai.errors import ConfigurationError, InsufficientHistoryError, ModelInferenceError
from kronos_ai.forecast.base import require_aligned
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
from kronos_ai.forecast.raw import RawSampleSet

__all__ = [
    "AR1_BASELINE_NAME",
    "BASELINE_FEATURES",
    "BASELINE_MATH_VERSION",
    "BASELINE_NAMES",
    "DEFAULT_MOVING_AVERAGE_WINDOW",
    "DRIFT_BASELINE_NAME",
    "LAST_VALUE_BASELINE_NAME",
    "MOVING_AVERAGE_BASELINE_NAME",
    "Ar1Baseline",
    "Baseline",
    "DriftBaseline",
    "LastValueBaseline",
    "MovingAverageBaseline",
    "build_baseline",
    "build_baseline_backends",
]

# baseline 数学定义的版本。任何公式/口径变化都必须递增它：它进入 model_revision 与
# config_hash，从而让旧 artifact 的 key 失效（§15），而不是静默改变历史 run 的含义。
BASELINE_MATH_VERSION = "baseline-math-v1"

# baseline 只用 OHLC：naive 规则不涉及成交量/成交额，声明更窄的特征集可避免用占位
# 数值伪造 volume/amount（那会让 ForecastSample 声称拥有从未观测的数据）。
BASELINE_FEATURES: tuple[str, ...] = ("open", "high", "low", "close")

LAST_VALUE_BASELINE_NAME = "last_value"
DRIFT_BASELINE_NAME = "drift"
MOVING_AVERAGE_BASELINE_NAME = "moving_average"
AR1_BASELINE_NAME = "ar1"

# 稳定顺序：benchmark 报告与配置校验都按此顺序列举 baseline，避免集合遍历顺序漂移。
BASELINE_NAMES: tuple[str, ...] = (
    LAST_VALUE_BASELINE_NAME,
    DRIFT_BASELINE_NAME,
    MOVING_AVERAGE_BASELINE_NAME,
    AR1_BASELINE_NAME,
)

# Moving Average 的默认窗口（20 个交易日 ≈ 一个月）：上游沿用已久的默认值，写死在
# 这里而不是散落在调用方，且进入 config_hash（窗口变化必须失效缓存）。
DEFAULT_MOVING_AVERAGE_WINDOW = 20


class Baseline(ABC):
    """确定性点预测 baseline 的公共骨架。

    子类只需实现 :meth:`Baseline._point_path`（给定 lookback 窗口 close 与 horizon，
    返回 horizon 长度的价格路径）并声明 ``_NAME`` / ``_MIN_ESTIMABLE_BARS``。
    """

    _NAME: ClassVar[str]
    _MIN_ESTIMABLE_BARS: ClassVar[int]

    def __init__(
        self,
        *,
        calendar: TradingCalendar,
        lookback_bars: int,
        distribution_spec: DistributionSpec = DEFAULT_DISTRIBUTION_SPEC,
        cache: ForecastCache | None = None,
    ) -> None:
        if lookback_bars < 1:
            raise ConfigurationError(f"lookback_bars must be >= 1, got {lookback_bars}")
        if lookback_bars < self.min_estimable_bars:
            raise ConfigurationError(
                f"baseline {self._NAME} needs at least {self.min_estimable_bars} bars, "
                f"got lookback_bars={lookback_bars}"
            )
        self._calendar = calendar
        self._lookback_bars = lookback_bars
        self._distribution_spec = distribution_spec
        self._cache = cache

    @property
    def name(self) -> str:
        """backend 名（§17）：也是 ``ModelMetadata.backend`` 与 benchmark 配置里的键。"""
        return self._NAME

    @property
    def lookback_bars(self) -> int:
        return self._lookback_bars

    @property
    def calendar(self) -> TradingCalendar:
        return self._calendar

    @property
    def min_estimable_bars(self) -> int:
        """估计量可识别所需的**配置**下界；子类可覆盖（MovingAverage 需要完整 window 根）。

        注意它与 §6.4 / provider 侧的 ``min_history_bars`` 不是同一个量：后者是「运行期
        history 够不够用」的门槛，在这里由 ``lookback_bars`` 承担。本属性只约束
        ``lookback_bars`` 的合法性（构造期检查，估计量不可识别时拒绝）。
        """
        return self._MIN_ESTIMABLE_BARS

    @property
    def distribution_spec(self) -> DistributionSpec:
        return self._distribution_spec

    def parameters(self) -> Mapping[str, Any]:
        """进入 config_hash 的 baseline 专属参数（子类覆盖）。"""
        return {}

    def identity(self) -> Mapping[str, str]:
        """artifact identity（§15）：与 :meth:`KronosRuntime.artifact_identity` 同形。

        对 baseline 而言「模型」就是公式：``model_revision`` 取数学定义版本，
        ``config_hash`` 覆盖 lookback 窗口与 baseline 参数（窗口从 20 改成 10 必须
        产出不同 artifact_id，否则两个不同模型会在缓存里互相冒充）。
        """
        return {
            "model_id": f"baseline:{self._NAME}",
            "model_revision": BASELINE_MATH_VERSION,
            "runtime_version": f"numpy-{np.__version__}",
            "device_class": "cpu",
            "dtype": "float64",
            "config_hash": self.config_hash(),
        }

    def config_hash(self) -> str:
        return sha256_hex(
            {
                "kind": "baseline_config",
                "math_version": BASELINE_MATH_VERSION,
                "baseline": self._NAME,
                "lookback_bars": self._lookback_bars,
                "features": list(BASELINE_FEATURES),
                "parameters": dict(self.parameters()),
            }
        )

    def forecast(
        self,
        history: MarketHistory,
        request: ForecastRequest,
        *,
        force: bool = False,
    ) -> ForecastResult:
        """产出 §14 ForecastResult（确定性点预测的退化分布）。

        ``force`` 与 cache 语义见缓存模块：baseline 始终同输入同输出，缓存只是省算力，
        不改变结果。
        """
        require_aligned(history, request)
        closes = self._window_closes(history)
        path = self._point_path(closes, request.horizon)
        if not np.isfinite(path).all():
            raise ModelInferenceError(
                f"baseline {self._NAME} produced a non-finite price path for symbol "
                f"{history.symbol}; refusing to propagate corrupt output"
            )
        if (path <= 0.0).any():
            raise ModelInferenceError(
                f"baseline {self._NAME} produced a non-positive price for symbol "
                f"{history.symbol}; refusing to propagate corrupt output"
            )

        future_sessions = tuple(self._calendar.next_sessions(request.market_date, request.horizon))
        raw = RawSampleSet(
            symbol=history.symbol,
            market_date=history.market_date,
            knowledge_cutoff=history.knowledge_cutoff,
            horizon=request.horizon,
            future_sessions=future_sessions,
            feature_names=BASELINE_FEATURES,
            values=self._degenerate_values(path, request.sampling.sample_count),
        )
        key = build_forecast_artifact_key(
            history=history,
            request=request,
            model_identity=self.identity(),
            calendar=self._calendar,
            distribution_spec=self._distribution_spec,
        )

        def compute() -> ForecastResult:
            return self._assemble(
                artifact_id=key.digest,
                raw=raw,
                origin_close=float(closes[-1]),
                history=history,
                request=request,
            )

        if self._cache is None:
            return compute()
        result, _ = cached_forecast(self._cache, key, compute, force=force)
        return result

    @abstractmethod
    def _point_path(self, closes: np.ndarray, horizon: int) -> np.ndarray:
        """给定 lookback 窗口收盘价（正数、按时间升序）与 horizon，返回价格路径 (horizon,)。"""

    def _window_closes(self, history: MarketHistory) -> np.ndarray:
        """lookback 窗口（最后 ``lookback_bars`` 根 bar）的 close。

        与 kronos **同一口径**：必须凑满 ``lookback_bars``，否则显式失败（§6.4）。
        这是从 §19/§42 gate 推导出的前置条件（不是 §19 的原话）：§19 只是把 gate 定义为
        与 naive baseline 的比较，而要预注册、可证伪地比较两个 backend，必须先固定
        「同一窗口、同一信息量」——否则差异可能来自可用数据量而不是预测能力。因此
        数据不足的 origin 必须由 dataset builder（§27，RX-KAI-017 负责保证每个 origin
        都有 ``len(bars) >= lookback_bars``）剔除，而不是让 backend 各自静默降级。
        """
        if len(history.bars) < self._lookback_bars:
            raise InsufficientHistoryError(
                f"symbol {history.symbol} has {len(history.bars)} bars, "
                f"lookback_bars={self._lookback_bars} required"
            )
        bars = history.bars[-self._lookback_bars :]
        return np.asarray([bar.close for bar in bars], dtype=np.float64)

    @staticmethod
    def _degenerate_values(path: np.ndarray, sample_count: int) -> np.ndarray:
        """把确定性路径编码成 raw samples：``sample_count`` 个完全相同的 OHLC 路径。

        OHLC 四列取同一 close 路径（naive 规则不区分开高低），这样
        ``forecast_samples_from_raw`` 产出的 4 个价格特征一致、volume/amount 为 None，
        下游分布与 §13 指标无需为 baseline 特判。
        """
        values = np.empty((sample_count, path.shape[0], len(BASELINE_FEATURES)), dtype=np.float64)
        values[:, :, :] = path[None, :, None]
        return values

    def model_metadata(self) -> ModelMetadata:
        """§14 provenance；由 :meth:`Baseline.identity` 映射而来，不另起一套取值。

        ``ModelMetadata`` 与 artifact key 的字段名不完全同名（``revision`` vs
        ``model_revision``、``device`` vs ``device_class``），映射集中在本方法，
        避免两处各自拼装而分叉。
        """
        identity = self.identity()
        return ModelMetadata(
            backend=self._NAME,
            model_id=identity["model_id"],
            revision=identity["model_revision"],
            runtime_version=identity["runtime_version"],
            device=identity["device_class"],
            dtype=identity["dtype"],
            config_hash=identity["config_hash"],
        )

    def _assemble(
        self,
        *,
        artifact_id: str,
        raw: RawSampleSet,
        origin_close: float,
        history: MarketHistory,
        request: ForecastRequest,
    ) -> ForecastResult:
        samples = forecast_samples_from_raw(raw)
        distribution = build_distribution(
            raw,
            origin_close=origin_close,
            spec=self._distribution_spec,
        )
        return ForecastResult(
            symbol=request.symbol,
            market_date=request.market_date,
            knowledge_cutoff=request.knowledge_cutoff,
            samples=samples,
            distribution=distribution,
            model=self.model_metadata(),
            sampling=SamplingMetadata.from_config(request.sampling),
            input_data_hash=history.data_hash,
            artifact_id=artifact_id,
        )


class LastValueBaseline(Baseline):
    """§19 Last Value：``P_t = P_0``（原点收盘价），全部 horizon 步持平。

    最弱的参照系，但极难被击败的短 horizon 基准（随机游走下它就是最小 MSE 预测）。
    """

    _NAME = LAST_VALUE_BASELINE_NAME
    _MIN_ESTIMABLE_BARS = 1

    def _point_path(self, closes: np.ndarray, horizon: int) -> np.ndarray:
        return np.full(horizon, closes[-1], dtype=np.float64)


class DriftBaseline(Baseline):
    """§19 Drift：对数空间漂移外推。

    ``d = (ln P_N - ln P_1) / (N - 1)``（窗口内平均对数收益），
    ``P_t = P_N * exp(d * t)``。窗口首尾价格决定全部预测，因此对端点异常值敏感——
    这正是 baseline 应当暴露的弱点，不做平滑掩饰。
    """

    _NAME = DRIFT_BASELINE_NAME
    _MIN_ESTIMABLE_BARS = 2

    def _point_path(self, closes: np.ndarray, horizon: int) -> np.ndarray:
        drift = (math.log(float(closes[-1])) - math.log(float(closes[0]))) / (closes.shape[0] - 1)
        steps = np.arange(1, horizon + 1, dtype=np.float64)
        path = np.empty(horizon, dtype=np.float64)
        path[:] = float(closes[-1]) * np.exp(drift * steps)
        return path


class MovingAverageBaseline(Baseline):
    """§19 Moving Average：末 ``window`` 根 close 的算术均值作为全 horizon 的常数路径。"""

    _NAME = MOVING_AVERAGE_BASELINE_NAME
    _MIN_ESTIMABLE_BARS = 1

    def __init__(
        self,
        *,
        calendar: TradingCalendar,
        lookback_bars: int,
        window: int = DEFAULT_MOVING_AVERAGE_WINDOW,
        distribution_spec: DistributionSpec = DEFAULT_DISTRIBUTION_SPEC,
        cache: ForecastCache | None = None,
    ) -> None:
        if window < 1:
            raise ConfigurationError(f"moving_average window must be >= 1, got {window}")
        # window > lookback_bars 时窗口永远无法被满足，均值会静默退化成「尽可能多」的
        # 短窗口；这是配置错误，构造期就拒绝（§3.2）。
        if window > lookback_bars:
            raise ConfigurationError(
                f"moving_average window {window} exceeds lookback_bars {lookback_bars}"
            )
        self._window = window
        super().__init__(
            calendar=calendar,
            lookback_bars=lookback_bars,
            distribution_spec=distribution_spec,
            cache=cache,
        )

    @property
    def window(self) -> int:
        return self._window

    @property
    def min_estimable_bars(self) -> int:
        """``window`` 根 bar 才能算出真正的 w 日均值（配置下界，不足即拒绝构造）。"""
        return max(self._window, self._MIN_ESTIMABLE_BARS)

    def parameters(self) -> Mapping[str, Any]:
        return {"window": self._window}

    def _point_path(self, closes: np.ndarray, horizon: int) -> np.ndarray:
        # lookback_bars >= window 且必须凑满 lookback_bars ⇒ 窗口内至少有 window 根 bar
        mean_close = float(closes[-self._window :].mean())
        return np.full(horizon, mean_close, dtype=np.float64)


class Ar1Baseline(Baseline):
    """§19 简单 statistical baseline：对数收益的一阶自回归（AR(1)）迭代外推。

    在窗口内用 OLS 拟合 ``r_t = c + φ * r_{t-1}``（普通最小二乘，无正则化），
    再以最后一个已观测收益为状态起点迭代出未来收益，累积回价格空间。

    与 Drift 的区别：Drift 只用首尾两点、隐含 φ=1 的均值项；AR(1) 用整窗估计斜率与
    截距。这是「能被击败才算模型有用」的最低门槛之一。

    不做静默稳健化：窗口内滞后收益方差为 0（无法识别斜率）时显式失败，而不是把 φ
    悄悄置 0 冒充估计值；``|φ| > 1`` 的非平稳窗口不做截断，但路径非有限/非正时失败。
    """

    _NAME = AR1_BASELINE_NAME
    # 需要 2 个 (r_{t-1}, r_t) 对才能估计斜率 ⇒ N 根 bar 给出 N-2 对 ⇒ N >= 4
    _MIN_ESTIMABLE_BARS = 4

    def _point_path(self, closes: np.ndarray, horizon: int) -> np.ndarray:
        log_returns = np.diff(np.log(closes))
        lagged = log_returns[:-1]
        current = log_returns[1:]
        lagged_mean = float(lagged.mean())
        current_mean = float(current.mean())
        variance = float(((lagged - lagged_mean) ** 2).sum())
        if variance == 0.0:
            raise ModelInferenceError(
                "cannot estimate the AR(1) slope: lookback window has zero return variance"
            )
        phi = float(((lagged - lagged_mean) * (current - current_mean)).sum()) / variance
        intercept = current_mean - phi * lagged_mean

        log_price = math.log(float(closes[-1]))
        state = float(log_returns[-1])
        path = np.empty(horizon, dtype=np.float64)
        for step in range(horizon):
            state = intercept + phi * state
            log_price += state
            path[step] = math.exp(log_price)
        return path


def build_baseline(
    name: str,
    *,
    calendar: TradingCalendar,
    lookback_bars: int,
    distribution_spec: DistributionSpec = DEFAULT_DISTRIBUTION_SPEC,
    cache: ForecastCache | None = None,
    moving_average_window: int = DEFAULT_MOVING_AVERAGE_WINDOW,
) -> Baseline:
    """按名字构造 baseline；未知名字显式失败（§3.2），不返回 None 或兜底实现。

    ``moving_average_window`` 只对 ``moving_average`` 生效，但合法性**无条件**校验：
    配置里的非法窗口不应因为当前 backend 恰好不是 MA 就静默通过（§3.2）。
    """
    if moving_average_window < 1:
        raise ConfigurationError(f"moving_average window must be >= 1, got {moving_average_window}")
    common: dict[str, Any] = {
        "calendar": calendar,
        "lookback_bars": lookback_bars,
        "distribution_spec": distribution_spec,
        "cache": cache,
    }
    if name == LAST_VALUE_BASELINE_NAME:
        return LastValueBaseline(**common)
    if name == DRIFT_BASELINE_NAME:
        return DriftBaseline(**common)
    if name == MOVING_AVERAGE_BASELINE_NAME:
        return MovingAverageBaseline(window=moving_average_window, **common)
    if name == AR1_BASELINE_NAME:
        return Ar1Baseline(**common)
    raise ConfigurationError(
        f"unknown forecast baseline {name!r}; known baselines: {list(BASELINE_NAMES)}"
    )


def build_baseline_backends(
    *,
    calendar: TradingCalendar,
    lookback_bars: int,
    distribution_spec: DistributionSpec = DEFAULT_DISTRIBUTION_SPEC,
    cache: ForecastCache | None = None,
    moving_average_window: int = DEFAULT_MOVING_AVERAGE_WINDOW,
) -> tuple[Baseline, ...]:
    """按 :data:`BASELINE_NAMES` 顺序构造全部 baseline（benchmark 的固定参照集）。"""
    return tuple(
        build_baseline(
            name,
            calendar=calendar,
            lookback_bars=lookback_bars,
            distribution_spec=distribution_spec,
            cache=cache,
            moving_average_window=moving_average_window,
        )
        for name in BASELINE_NAMES
    )


if TYPE_CHECKING:
    # mypy 结构化校验：四个 baseline 必须满足 §17 ForecastBackend 契约
    # （runtime_checkable 的 isinstance 只检查成员存在，不校验签名）
    from kronos_ai.forecast.base import ForecastBackend

    def _contract_anchor(backend: Baseline) -> ForecastBackend:
        return backend
