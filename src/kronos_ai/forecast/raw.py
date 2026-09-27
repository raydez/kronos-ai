"""Raw sample matrix：一次 forecast 的未聚合样本原料（基线文档 §10/§11）。

``RawSampleSet`` 是「sample 维 reshape 之后、mean 之前」的原始张量（§10 截取点），
形状 ``(sample_count, horizon, len(feature_names))``，价格空间，float64。它属于
forecast 层而不是某个 backend：模型 backend（kronos）与确定性 baseline
（:mod:`kronos_ai.evaluation.baselines`）产出同一种原料，随后走同一条
``forecast_samples_from_raw`` → ``build_distribution`` 路径。

这样做的理由是可比性：Kronos 与 baseline 的样本 / 指标代码路径完全同一，benchmark
里出现的差异只可能来自预测本身，不可能来自两套实现的口径分叉。

只有校验，不做聚合：均值、阈值概率、分位全部属 :mod:`kronos_ai.forecast.distribution`
（§10 明确禁止用 ``mean`` 抹掉分布）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np


@dataclass(frozen=True)
class RawSampleSet:
    """一次采样的 raw samples（未做任何跨样本聚合）。

    values 形状 = (sample_count, horizon, len(feature_names))，价格空间，float64。
    未来时间轴由 future_sessions 给出（market session 序列，非个股 session）。
    """

    symbol: str
    market_date: date
    knowledge_cutoff: datetime
    horizon: int
    future_sessions: tuple[date, ...]
    feature_names: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.values.ndim != 3:
            raise ValueError(f"values must be 3-dimensional; got shape {self.values.shape}")
        if self.values.dtype != np.float64:
            raise ValueError(f"values must be float64; got {self.values.dtype}")
        if not np.isfinite(self.values).all():
            raise ValueError("values must be finite; non-finite samples indicate corrupt output")
        if self.values.shape[1] != self.horizon:
            raise ValueError(
                f"values horizon {self.values.shape[1]} != declared horizon {self.horizon}"
            )
        if self.values.shape[2] != len(self.feature_names):
            raise ValueError(
                f"values feature width {self.values.shape[2]} != "
                f"feature_names length {len(self.feature_names)}"
            )
        if self.values.shape[0] < 1:
            raise ValueError("values must contain at least one sample")
        if len(self.future_sessions) != self.horizon:
            raise ValueError("future_sessions length must equal horizon")
        # frozen dataclass 不阻止对 ndarray 的原地写入；置为只读，保证样本集不可变
        self.values.flags.writeable = False

    @property
    def sample_count(self) -> int:
        return int(self.values.shape[0])
