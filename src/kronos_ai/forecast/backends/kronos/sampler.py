"""KronosSampler：raw stochastic sample 截取（基线文档 §10/§11；RX-KAI-010/011，ADR-004/005）。

与上游 `auto_regressive_inference` 的关系（受控 adapter，不放 vendor 目录）：

- 自回归循环逐结构与上游保持一致（缓冲与 roll、stamp 拼接、`encode/decode(half=True)`、
  `temperature` / top-k / top-p 过滤顺序）。显式差异共三处，均为 v2 设计：
  1. 截取点：``tokenizer.decode`` 结果 reshape 出 sample 维之后、``mean(sample)`` 之前，
     返回全部 raw samples（mean 会抹掉分布，§10）
  2. RNG：``torch.multinomial(..., generator=...)`` 使用 per-run generator，
     覆盖每个自回归 step 的 s1 / s2 两次采样调用（§9），不触碰全局 RNG
  3. 精度：窗口归一化统计量与输出用 float64（上游 `predict()` 在 float32 上算
     mean/std，`vendor/kronos.py:541-544`）。相对偏差实测 ~2e-9，等价性回归以
     rtol=1e-7 覆盖（改归一化常数即失败）
- 逐 step ``torch.cuda.empty_cache()``：vendored commit 的 ``auto_regressive_inference``
  中本就没有该调用（见 vendor/kronos.py），v2 不重新引入——这不是差异，是保持原样
- 默认不再有 mean 路径（禁止「sample_count=64 → 原生 predict → mean 伪装成分布」）。

采样参数（seed / temperature / top-k / top-p）与模型身份不在本层携带：raw samples 只是
原料，seed 与 runtime identity 由 RX-KAI-012 的 ForecastSample / ForecastDistribution
连同 §15 的 artifact key 一起承担。

数值契约前提：模块必须处于 **eval 模式**（KronosRuntime 强制）。vendored 的 s2
cross-attention 用 ``is_causal_flag = self.training`` 决定注意力掩码
（vendor/module.py:387，dropout 同理见 :392）：train 与 eval 会给出不同的 s2
logits，等价性断言在 train 模式下不成立。上游 canonical 用法同样落在 eval
（huggingface_hub 的 ``from_pretrained`` 在返回前调用 ``model.eval()``，
hub_mixin.py:801/821，行号对应锁定版本 huggingface_hub 2.0.0），
因此 eval 是复现上游语义的必要条件而非可选项。

内存模型（§16）：单 symbol 调用 batch=1，但前向张量实际是
``batch × sample_count × seq_len × d_model`` 规模，batch 与 sample_count 在内存上相乘。
批量 API 落地前必须由 Cost Probe（RX-KAI-018）给出 batch × sample_count 曲线。

时间轴（§11）：未来时间戳一律来自 TradingCalendar.next_sessions，
个股停牌不改变 forecast 时间轴。lookback 窗口取 history 的最后
``lookback_bars`` 根 bar，不保证末根落在 market_date（停牌日没有 bar）——
陈旧窗口是上游语义的一部分，本层不补造 bar、也不缩短窗口。

价格空间：返回值为反归一化后的原始价格量纲（元 / 股），归一化统计量来自
lookback 窗口本身（逐窗口统计，§8）。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd
import torch

from kronos_ai.data.calendar import TradingCalendar
from kronos_ai.domain.forecast import ForecastRequest, ForecastSample
from kronos_ai.domain.market import MarketBar, MarketHistory
from kronos_ai.domain.time import CN_TZ, MARKET_SESSION_CLOSE
from kronos_ai.errors import (
    ConfigurationError,
    DataQualityError,
    InsufficientHistoryError,
    ModelInferenceError,
)
from kronos_ai.forecast.backends.kronos.rng import RunRNG
from kronos_ai.forecast.backends.kronos.runtime import KronosRuntime
from kronos_ai.forecast.distribution import forecast_samples_from_raw

FEATURE_NAMES: tuple[str, ...] = ("open", "high", "low", "close", "volume", "amount")
TIME_FEATURE_NAMES: tuple[str, ...] = ("minute", "hour", "weekday", "day", "month")

# 与上游 predict() 一致的归一化保护项；上游用 1e-5，不做更改
NORM_EPS = 1e-5


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


def assert_finite_samples(values: np.ndarray, *, symbol: str) -> None:
    """非有限值属于模型输出损坏，显式失败（§3.2），不进入下游指标。"""
    if not np.isfinite(values).all():
        raise ModelInferenceError(
            f"kronos produced non-finite forecast samples for symbol {symbol}; "
            "refusing to propagate corrupt output"
        )


def time_stamp_frame(timestamps: Sequence[datetime]) -> np.ndarray:
    """逐 bar 时间特征 (minute, hour, weekday, day, month)。

    与上游 `calc_time_stamps` 语义一致（已验证等价，见 tests/unit/test_sampler.py），
    但直接作用在 DatetimeIndex 属性上：pandas 3.x 的 DatetimeIndex 没有 .dt accessor，
    上游实现只能接收 Series。
    """
    index = pd.DatetimeIndex(list(timestamps))
    frame = np.empty((len(index), len(TIME_FEATURE_NAMES)), dtype=np.float64)
    frame[:, 0] = index.minute
    frame[:, 1] = index.hour
    frame[:, 2] = index.weekday
    frame[:, 3] = index.day
    frame[:, 4] = index.month
    return frame.astype(np.float32)


class KronosSampler:
    """消费 MarketHistory + ForecastRequest，产出 raw samples（无聚合）。

    不承担：缓存、持久化、分布统计（分别属于 cache / artifact store / distribution）。
    """

    def __init__(self, runtime: KronosRuntime, *, calendar: TradingCalendar) -> None:
        self._runtime = runtime
        self._calendar = calendar

    def decode_raw_samples(self, history: MarketHistory, request: ForecastRequest) -> RawSampleSet:
        """§10 截取点：sample 维 reshape 之后、mean 之前。"""
        self._require_aligned(history, request)
        # 上游 decode 窗口只含最后 max_context 个 token：horizon 超过窗口时未来段会被
        # 截断且与 future_sessions 错位，此处显式拒绝而非产出静默错位的样本。
        # 先于日历查询判定：配置错误不该依赖日历覆盖（也不该在缺日历时被掩盖）
        max_context = self._runtime.config.max_context
        if request.horizon > max_context:
            raise ConfigurationError(
                f"horizon {request.horizon} exceeds max_context {max_context}; "
                "the context window cannot carry the whole forecast horizon"
            )
        bars = self._lookback_window(history)
        future_sessions = tuple(self._calendar.next_sessions(request.market_date, request.horizon))

        x = self._feature_matrix(bars)
        x_mean = x.mean(axis=0)
        x_std = x.std(axis=0)
        x_norm = np.clip((x - x_mean) / (x_std + NORM_EPS), -self._clip, self._clip)

        x_stamp = time_stamp_frame([bar.timestamp for bar in bars])
        y_stamp = time_stamp_frame([self._session_close(day) for day in future_sessions])

        sampling = request.sampling
        run_rng = RunRNG(sampling, device_class=self._runtime.device_class)
        decoded = self._decode_tokens(
            x_norm, x_stamp, y_stamp, sample_count=sampling.sample_count, run_rng=run_rng
        )

        values = decoded * (x_std + NORM_EPS) + x_mean
        assert_finite_samples(values, symbol=history.symbol)

        return RawSampleSet(
            symbol=history.symbol,
            market_date=history.market_date,
            knowledge_cutoff=history.knowledge_cutoff,
            horizon=request.horizon,
            future_sessions=future_sessions,
            feature_names=FEATURE_NAMES,
            values=values,
        )

    def generate_samples(
        self, history: MarketHistory, request: ForecastRequest
    ) -> list[ForecastSample]:
        """§10 推荐接口：在 mean 之前截取 raw samples，转为可持久化 ForecastSample 序列。

        聚合统计（均值/分位/阈值概率）不在此层：raw samples 是研究原料，分布由
        ``forecast.distribution.build_distribution`` 显式构建（§10 禁止 mean 伪装成分布）。
        """
        raw = self.decode_raw_samples(history, request)
        return list(forecast_samples_from_raw(raw))

    @property
    def _clip(self) -> float:
        return self._runtime.config.clip

    @staticmethod
    def _session_close(day: date) -> datetime:
        return datetime.combine(day, MARKET_SESSION_CLOSE, tzinfo=CN_TZ)

    def _require_aligned(self, history: MarketHistory, request: ForecastRequest) -> None:
        if history.symbol != request.symbol:
            raise ConfigurationError(
                f"history symbol {history.symbol!r} != request symbol {request.symbol!r}"
            )
        if history.market_date != request.market_date:
            raise ConfigurationError(
                f"history market_date {history.market_date} != request market_date "
                f"{request.market_date}"
            )
        if history.knowledge_cutoff != request.knowledge_cutoff:
            raise ConfigurationError(
                "history knowledge_cutoff "
                f"{history.knowledge_cutoff.isoformat()} != request knowledge_cutoff "
                f"{request.knowledge_cutoff.isoformat()}"
            )

    def _lookback_window(self, history: MarketHistory) -> tuple[MarketBar, ...]:
        lookback = self._runtime.lookback_bars
        if len(history.bars) < lookback:
            raise InsufficientHistoryError(
                f"symbol {history.symbol} has {len(history.bars)} bars, "
                f"lookback_bars={lookback} required"
            )
        return history.bars[-lookback:]

    @staticmethod
    def _feature_matrix(bars: Sequence[MarketBar]) -> np.ndarray:
        rows = []
        for bar in bars:
            if bar.volume is None or bar.amount is None:
                raise DataQualityError(
                    f"bar {bar.timestamp.isoformat()} for symbol {bar.symbol} lacks "
                    "volume/amount; kronos requires the full feature set"
                )
            rows.append((bar.open, bar.high, bar.low, bar.close, bar.volume, bar.amount))
        return np.asarray(rows, dtype=np.float64)

    def _decode_tokens(
        self,
        x: np.ndarray,
        x_stamp: np.ndarray,
        y_stamp: np.ndarray,
        *,
        sample_count: int,
        run_rng: RunRNG,
    ) -> np.ndarray:
        """单 symbol 自回归采样；返回 (sample_count, horizon, feature_count) 价格空间之外的
        归一化空间张量（尚未反归一化）。"""
        runtime = self._runtime
        dtype = runtime.config.torch_dtype
        device = torch.device(runtime.device_class)
        max_context = runtime.config.max_context
        horizon = y_stamp.shape[0]

        with torch.no_grad():
            x_tensor = torch.from_numpy(x[None]).to(device=device, dtype=dtype)
            x_stamp_tensor = torch.from_numpy(x_stamp[None]).to(device=device, dtype=dtype)
            y_stamp_tensor = torch.from_numpy(y_stamp[None]).to(device=device, dtype=dtype)

            x_tensor = torch.clip(x_tensor, -self._clip, self._clip)
            x_tensor = (
                x_tensor.unsqueeze(1)
                .repeat(1, sample_count, 1, 1)
                .reshape(-1, x_tensor.size(1), x_tensor.size(2))
            )
            x_stamp_tensor = (
                x_stamp_tensor.unsqueeze(1)
                .repeat(1, sample_count, 1, 1)
                .reshape(-1, x_stamp_tensor.size(1), x_stamp_tensor.size(2))
            )
            y_stamp_tensor = (
                y_stamp_tensor.unsqueeze(1)
                .repeat(1, sample_count, 1, 1)
                .reshape(-1, y_stamp_tensor.size(1), y_stamp_tensor.size(2))
            )

            tokenizer = runtime.tokenizer
            model = runtime.model
            x_token = tokenizer.encode(x_tensor, half=True)  # type: ignore[operator]

            initial_seq_len = x_tensor.size(1)
            total_seq_len = initial_seq_len + horizon
            full_stamp = torch.cat([x_stamp_tensor, y_stamp_tensor], dim=1)
            batch_size = x_token[0].size(0)

            generated_pre = x_token[0].new_empty(batch_size, horizon)
            generated_post = x_token[1].new_empty(batch_size, horizon)
            pre_buffer = x_token[0].new_zeros(batch_size, max_context)
            post_buffer = x_token[1].new_zeros(batch_size, max_context)
            buffer_len = min(initial_seq_len, max_context)
            start_idx = max(0, initial_seq_len - max_context)
            pre_buffer[:, :buffer_len] = x_token[0][:, start_idx : start_idx + buffer_len]
            post_buffer[:, :buffer_len] = x_token[1][:, start_idx : start_idx + buffer_len]

            for step in self._step_range(horizon):
                current_seq_len = initial_seq_len + step
                window_len = min(current_seq_len, max_context)
                if current_seq_len <= max_context:
                    input_tokens = [
                        pre_buffer[:, :window_len],
                        post_buffer[:, :window_len],
                    ]
                else:
                    input_tokens = [pre_buffer, post_buffer]

                context_start = max(0, current_seq_len - max_context)
                current_stamp = full_stamp[:, context_start:current_seq_len, :].contiguous()

                s1_logits, context = model.decode_s1(  # type: ignore[operator]
                    input_tokens[0], input_tokens[1], current_stamp
                )
                s1_logits = s1_logits[:, -1, :]
                sample_pre = run_rng.sample_logits(s1_logits)

                s2_logits = model.decode_s2(context, sample_pre)  # type: ignore[operator]
                s2_logits = s2_logits[:, -1, :]
                sample_post = run_rng.sample_logits(s2_logits)

                generated_pre[:, step] = sample_pre.squeeze(-1)
                generated_post[:, step] = sample_post.squeeze(-1)

                if current_seq_len < max_context:
                    pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
                    post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
                else:
                    pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
                    post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
                    pre_buffer[:, -1] = sample_pre.squeeze(-1)
                    post_buffer[:, -1] = sample_post.squeeze(-1)

            full_pre = torch.cat([x_token[0], generated_pre], dim=1)
            full_post = torch.cat([x_token[1], generated_post], dim=1)
            context_start = max(0, total_seq_len - max_context)
            decode_tokens = [
                full_pre[:, context_start:total_seq_len].contiguous(),
                full_post[:, context_start:total_seq_len].contiguous(),
            ]
            z = tokenizer.decode(decode_tokens, half=True)  # type: ignore[operator]

            # decode 窗口长度 = min(total_seq_len, max_context)（上游语义）
            window_seq_len = min(total_seq_len, max_context)
            expected_shape = (batch_size, window_seq_len, len(FEATURE_NAMES))
            if tuple(z.shape) != expected_shape:
                raise ModelInferenceError(
                    f"tokenizer decode shape {tuple(z.shape)} != expected {expected_shape}; "
                    "check tokenizer d_in against sampler feature set"
                )
            # 截取点：sample 维 reshape 之后、mean 之前（§10）
            raw: torch.Tensor = z.reshape(-1, sample_count, z.size(1), z.size(2))
            if raw.size(0) != 1:
                raise ModelInferenceError(
                    f"decode produced {raw.size(0)} symbols; this path is single-symbol "
                    "by construction (batch 维来自单个 history)，批量采样属 §16 的后续任务"
                )
            return raw[0, :, -horizon:, :].detach().to("cpu").to(torch.float64).numpy()

    def _step_range(self, horizon: int) -> Iterable[int]:
        """verbose 配置唯一消费点：只切换进度显示，不影响任何数值路径。"""
        if not self._runtime.config.verbose:
            return range(horizon)

        from tqdm import tqdm

        return tqdm(range(horizon), desc="kronos sampling", leave=False)
