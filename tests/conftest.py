"""共享测试夹具：tiny 随机初始化 Kronos 模型（不下载权重、不要求网络）。

tiny 模型参数被压缩到最小可运行规模；所有 dropout = 0 保证采样路径确定性。
另有合成 MarketHistory / TradingCalendar 工厂，避免任何真实行情数据依赖。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import pytest
import torch

from kronos_ai.data.calendar import StaticTradingCalendar
from kronos_ai.domain.market import MarketBar, MarketHistory
from kronos_ai.domain.time import CN_TZ, MARKET_SESSION_CLOSE
from kronos_ai.forecast.backends.kronos.runtime import (
    DeviceChoice,
    KronosRuntime,
    KronosRuntimeConfig,
)
from kronos_ai.forecast.backends.kronos.vendor import Kronos, KronosTokenizer

TINY_S1_BITS = 4
TINY_S2_BITS = 4
TINY_D_MODEL = 16

# 夹具时间轴：2026-09-01（周二）起的工作日序列；market_date = 2026-09-25（周五）
FIXTURE_FIRST_SESSION = date(2026, 9, 1)
FIXTURE_MARKET_DATE = date(2026, 9, 25)
FIXTURE_CUTOFF = datetime(2026, 9, 25, 18, 0, tzinfo=CN_TZ)

# 直接注入随机初始化模块时的占位 revision（40 位 hex，格式合法但显然是假值）
TINY_PIN = "deadbeef" * 5

# 权重初始化种子：Kronos._init_weights 走全局 RNG，固定后可避免断言（如 spread > 0）随权重抽签抖动
TINY_WEIGHT_SEED = 20260927


def build_tiny_tokenizer() -> KronosTokenizer:
    torch.manual_seed(TINY_WEIGHT_SEED)
    return KronosTokenizer(
        d_in=6,
        d_model=TINY_D_MODEL,
        n_heads=2,
        ff_dim=32,
        n_enc_layers=2,
        n_dec_layers=2,
        ffn_dropout_p=0.0,
        attn_dropout_p=0.0,
        resid_dropout_p=0.0,
        s1_bits=TINY_S1_BITS,
        s2_bits=TINY_S2_BITS,
        beta=1.0,
        gamma0=0.1,
        gamma=0.1,
        zeta=0.1,
        group_size=4,
    )


def build_tiny_model() -> Kronos:
    torch.manual_seed(TINY_WEIGHT_SEED)
    return Kronos(
        s1_bits=TINY_S1_BITS,
        s2_bits=TINY_S2_BITS,
        n_layers=1,
        d_model=TINY_D_MODEL,
        n_heads=2,
        ff_dim=32,
        ffn_dropout_p=0.0,
        attn_dropout_p=0.0,
        resid_dropout_p=0.0,
        token_dropout_p=0.0,
        learn_te=True,
    )


def weekdays(start: date, end: date) -> tuple[date, ...]:
    """测试夹具专用：显式工作日序列（生产日历规则不在此定义，见 ADR-009）。"""
    days: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return tuple(days)


def make_bar(
    symbol: str,
    day: date,
    *,
    index: int = 0,
    volume: float | None = None,
    amount: float | None = None,
) -> MarketBar:
    close = 10.0 + index * 0.1 + (index % 5) * 0.07
    open_ = close - 0.05
    return MarketBar(
        symbol=symbol,
        timestamp=datetime.combine(day, MARKET_SESSION_CLOSE, tzinfo=CN_TZ),
        open=open_,
        high=max(open_, close) + 0.12,
        low=min(open_, close) - 0.12,
        close=close,
        volume=1_000_000.0 + index * 1_000.0 if volume is None else volume,
        amount=10_000_000.0 + index * 10_000.0 if amount is None else amount,
        trade_status="1",
        adjustment_mode="raw",
        available_at=datetime.combine(day, MARKET_SESSION_CLOSE, tzinfo=CN_TZ),
    )


@pytest.fixture(scope="session")
def tiny_tokenizer() -> KronosTokenizer:
    # eval 是数值契约的一部分：vendored s2 cross-attention 用 is_causal_flag =
    # self.training 决定掩码（vendor/module.py:387），train 下的 s2 logits 与 eval 不同
    return build_tiny_tokenizer().eval()


@pytest.fixture(scope="session")
def tiny_model() -> Kronos:
    return build_tiny_model().eval()


@pytest.fixture
def tiny_runtime_factory() -> Any:
    """构造独立 runtime：每次新建模块，绝不复用 session 夹具实例。

    复用会踩上游 RotaryPositionalEmbedding 的非 buffer 缓存
    （module.py:292-301：cos/sin 按 seq_len 缓存，模块换 device 后缓存不跟着走，
    同 seq_len 复用会返回旧 device 的张量）。
    device 默认 cpu：等价性回归要与同样跑在 cpu 的上游参考实现对齐（§9 不承诺跨设备
    一致），需要 mps/cuda 的用例显式覆盖。eval 由 KronosRuntime 强制。
    """

    def _make(
        *,
        lookback_bars: int = 256,
        max_context: int = 512,
        device: DeviceChoice = "cpu",
        model_revision: str = TINY_PIN,
        tokenizer_revision: str = TINY_PIN,
        **overrides: Any,
    ) -> KronosRuntime:
        config = KronosRuntimeConfig(
            lookback_bars=lookback_bars,
            max_context=max_context,
            device=device,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            **overrides,
        )
        return KronosRuntime(
            model=build_tiny_model(),
            tokenizer=build_tiny_tokenizer(),
            config=config,
        )

    return _make


@pytest.fixture
def tiny_runtime(tiny_runtime_factory: Any) -> KronosRuntime:
    return tiny_runtime_factory(device="cpu")


@pytest.fixture
def session_calendar() -> StaticTradingCalendar:
    # 覆盖到 2027 年底：回归测试需要 60+ 个未来 session（decode 窗口截断用例）
    sessions = weekdays(FIXTURE_FIRST_SESSION, date(2027, 12, 31))
    return StaticTradingCalendar(exchange="SSE", source="test-fixture", sessions=sessions)


@pytest.fixture
def make_history() -> Any:
    """构造合成 MarketHistory：n_bars 个 session 日线，末 bar 落在 market_date。"""

    def _make(
        *,
        symbol: str = "600000",
        market_date: date = FIXTURE_MARKET_DATE,
        n_bars: int = 32,
        cutoff: datetime = FIXTURE_CUTOFF,
        bars: tuple[MarketBar, ...] | None = None,
        **bar_overrides: Any,
    ) -> MarketHistory:
        if bars is None:
            sessions = weekdays(date(2026, 6, 1), market_date)
            sessions = sessions[-n_bars:]
            bars = tuple(
                make_bar(symbol, day, index=i, **bar_overrides) for i, day in enumerate(sessions)
            )
        return MarketHistory(
            symbol=symbol,
            market_date=market_date,
            knowledge_cutoff=cutoff,
            bars=bars,
            provider="test-fixture",
            dataset_version="test-dataset-v1",
        )

    return _make
