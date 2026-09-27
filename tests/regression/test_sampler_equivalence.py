"""Sampler 等价性与 raw 截取回归（§10、§54.2；ADR-004）。

核心断言：v2 raw samples 的样本均值必须与上游 `auto_regressive_inference`
在相同 seed 下的输出一致——证明 v2 只截取了 mean 之前的样本，没有改变
随机路径本身。上游不暴露 raw samples，因此该等价性是「截取点正确」的
最直接回归证据。

同文件覆盖 DoD 32 要求的可复现性回归：固定 input + seed + sampling config
→ raw samples 一致；以及 §9 的全局 RNG 不被触碰。
"""

from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from kronos_ai.data.calendar import StaticTradingCalendar
from kronos_ai.domain.forecast import ForecastRequest, SamplingConfig
from kronos_ai.domain.market import MarketHistory
from kronos_ai.domain.time import CN_TZ
from kronos_ai.errors import ConfigurationError
from kronos_ai.forecast.backends.kronos.runtime import KronosRuntime
from kronos_ai.forecast.backends.kronos.sampler import KronosSampler
from kronos_ai.forecast.backends.kronos.vendor import (
    Kronos,
    KronosTokenizer,
    auto_regressive_inference,
    calc_time_stamps,
)

pytestmark = pytest.mark.regression

MD = date(2026, 9, 25)
CUTOFF = datetime(2026, 9, 25, 18, 0, tzinfo=CN_TZ)
SEED = 20260927
SAMPLE_COUNT = 6
HORIZON = 4
LOOKBACK = 12

# 故意本地硬编码而非 import sampler.NORM_EPS：参考实现必须独立于被测实现，
# 改 sampler 的常数不应同时移动比较的两边。值与 vendored 上游一致（1e-5，
# vendor/kronos.py:543 的 (x - x_mean) / (x_std + 1e-5)）。
REFERENCE_NORM_EPS = 1e-5


def make_request(**overrides: object) -> ForecastRequest:
    fields: dict[str, object] = {
        "symbol": "600000",
        "market_date": MD,
        "knowledge_cutoff": CUTOFF,
        "horizon": HORIZON,
        "sampling": SamplingConfig(seed=SEED, sample_count=SAMPLE_COUNT),
    }
    fields.update(overrides)
    return ForecastRequest(**fields)  # type: ignore[arg-type]


def _stamps(days: list[date]) -> pd.Series:
    return pd.Series(
        pd.DatetimeIndex(
            [
                datetime.combine(day, datetime.min.time(), tzinfo=CN_TZ).replace(hour=15)
                for day in days
            ]
        )
    )


def _upstream_mean(
    tiny_tokenizer: KronosTokenizer,
    tiny_model: Kronos,
    history: MarketHistory,
    *,
    lookback: int,
    horizon: int,
    max_context: int,
    sample_count: int,
    seed: int,
    calendar: StaticTradingCalendar,
) -> np.ndarray:
    """独立重建上游输入（与 sampler 相同归一化，但不复用其内部状态）。"""
    # eval 是等价性的前提：vendored s2 cross-attention 用 is_causal_flag =
    # self.training 决定掩码（vendor/module.py:387），train 模式下 s2 logits 与
    # eval 不同。上游 canonical 用法也是 eval（from_pretrained 末尾调用 .eval()，
    # hub_mixin.py:801，行号对应锁定版本 huggingface_hub 2.0.0），
    # 这里必须与之一致，否则比较的是两种不同的推理语义。
    assert not tiny_model.training and not tiny_tokenizer.training
    bars = history.bars[-lookback:]
    x = np.asarray(
        [(b.open, b.high, b.low, b.close, b.volume, b.amount) for b in bars], dtype=np.float64
    )
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_norm = np.clip((x - x_mean) / (x_std + REFERENCE_NORM_EPS), -5.0, 5.0)
    future_sessions = calendar.next_sessions(history.market_date, horizon)

    x_stamp = calc_time_stamps(_stamps([bar.timestamp for bar in bars])).values.astype(np.float32)
    y_stamp = calc_time_stamps(_stamps(future_sessions)).values.astype(np.float32)

    torch.manual_seed(seed)
    upstream_mean = auto_regressive_inference(
        tiny_tokenizer,
        tiny_model,
        torch.from_numpy(x_norm[None].astype(np.float32)),
        torch.from_numpy(x_stamp[None]),
        torch.from_numpy(y_stamp[None]),
        max_context=max_context,
        pred_len=horizon,
        clip=5.0,
        T=1.0,
        top_k=0,
        top_p=0.9,
        sample_count=sample_count,
        verbose=False,
    )
    # 上游 mean 覆盖 history + future（截取由 KronosPredictor.generate 完成），
    # 且在归一化空间；比较前取未来段并统一到价格空间
    price = (
        upstream_mean[:, -horizon:, :].astype(np.float64) * (x_std + REFERENCE_NORM_EPS) + x_mean
    )
    assert price.shape == (1, horizon, 6)  # 上游保留 batch 维
    return price[0]


@pytest.mark.parametrize(
    ("lookback", "max_context", "horizon"),
    [
        (LOOKBACK, 512, HORIZON),  # 常规：total_seq_len 远小于 max_context，窗口不截断
        (LOOKBACK, LOOKBACK, HORIZON),  # lookback == max_context：buffer roll + decode 窗口截断
        (4, 4, 4),  # horizon == 窗口长度：decode 窗口只含未来段（无历史上下文）
        (16, 512, 60),  # horizon 远大于 lookback：解码窗口整体长于输入窗口
    ],
)
def test_raw_samples_mean_equals_upstream_inference(
    tiny_model: Kronos,
    tiny_tokenizer: KronosTokenizer,
    session_calendar: StaticTradingCalendar,
    make_history: Any,
    tiny_runtime_factory: Any,
    lookback: int,
    max_context: int,
    horizon: int,
) -> None:
    history: MarketHistory = make_history(n_bars=lookback)
    request = make_request(horizon=horizon)
    runtime: KronosRuntime = tiny_runtime_factory(lookback_bars=lookback, max_context=max_context)
    raw = KronosSampler(runtime, calendar=session_calendar).decode_raw_samples(history, request)

    upstream_price = _upstream_mean(
        tiny_tokenizer,
        tiny_model,
        history,
        lookback=lookback,
        horizon=horizon,
        max_context=max_context,
        sample_count=SAMPLE_COUNT,
        seed=SEED,
        calendar=session_calendar,
    )
    v2_mean = raw.values.mean(axis=0)

    assert v2_mean.shape == (horizon, 6)
    # 断言比较对象本身非退化（避免"两边都是常数/全零"的假绿）
    assert np.isfinite(upstream_price).all()
    assert upstream_price.std() > 0
    assert v2_mean.std() > 0
    # 固有偏差 = v2 在 float64 上累积 mean 而上游在 float32（实测 ~2e-9 相对），
    # 容差取 1e-7 留约 30× 余量：既能吸收精度差，又足以震红归一化/截取点的结构性错误
    # （实测把 NORM_EPS 从 1e-5 改成 1e-4 即失败）
    np.testing.assert_allclose(v2_mean, upstream_price, rtol=1e-7, atol=1e-9)


def test_horizon_beyond_max_context_is_rejected(
    session_calendar: StaticTradingCalendar,
    make_history: Any,
    tiny_runtime_factory: Any,
) -> None:
    """上游 decode 窗口只含最后 max_context 个 token：horizon 超窗会静默错位，必须显式拒绝。"""
    history: MarketHistory = make_history(n_bars=LOOKBACK)
    runtime: KronosRuntime = tiny_runtime_factory(lookback_bars=LOOKBACK, max_context=16)
    sampler = KronosSampler(runtime, calendar=session_calendar)
    with pytest.raises(ConfigurationError, match="exceeds max_context"):
        sampler.decode_raw_samples(history, make_request(horizon=17))


def test_raw_samples_preserve_distribution(
    session_calendar: StaticTradingCalendar,
    make_history: Any,
    tiny_runtime_factory: Any,
) -> None:
    """mean 路径会抹掉的 sample 维信息必须仍然存在（§10 的核心动机）。"""
    history: MarketHistory = make_history(n_bars=LOOKBACK)
    raw = KronosSampler(
        tiny_runtime_factory(lookback_bars=LOOKBACK), calendar=session_calendar
    ).decode_raw_samples(history, make_request())
    assert raw.values.shape == (SAMPLE_COUNT, HORIZON, 6)
    close_idx = 3
    spread = raw.values[:, :, close_idx].std(axis=0)
    assert (spread > 0).all()
    # 且不是把单条路径重复 sample_count 次
    assert not np.allclose(raw.values, raw.values[:1, :, :], atol=1e-12)


def test_same_seed_bitwise_identical_across_runs(
    session_calendar: StaticTradingCalendar,
    make_history: Any,
    tiny_runtime_factory: Any,
) -> None:
    """§54.2 / DoD 32：固定 input + seed + sampling config → raw samples 一致。"""
    history: MarketHistory = make_history(n_bars=LOOKBACK)
    sampler = KronosSampler(tiny_runtime_factory(lookback_bars=LOOKBACK), calendar=session_calendar)
    first = sampler.decode_raw_samples(history, make_request())
    second = sampler.decode_raw_samples(history, make_request())
    assert np.array_equal(first.values, second.values)


def test_global_rng_state_untouched(
    session_calendar: StaticTradingCalendar,
    make_history: Any,
    tiny_runtime_factory: Any,
) -> None:
    """§9：per-run RNG 不读写全局 torch RNG 状态。"""
    history: MarketHistory = make_history(n_bars=LOOKBACK)
    sampler = KronosSampler(tiny_runtime_factory(lookback_bars=LOOKBACK), calendar=session_calendar)

    torch.manual_seed(1234)
    before = torch.random.get_rng_state()
    sampler.decode_raw_samples(history, make_request())
    assert torch.equal(before, torch.random.get_rng_state())
