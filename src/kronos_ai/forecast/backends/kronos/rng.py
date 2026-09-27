"""Per-run RNG isolation（基线文档 §9 / §54.2；ADR-005，RX-KAI-011）。

采样随机性的所有权集中在本模块：一次 forecast run 构造一个 :class:`RunRNG`，
之后所有读随机性的 torch 采样算子都必须经由它。这样做的原因是 §9 明确不推荐把
``torch.manual_seed(seed)`` 当作最终方案——它写全局状态，并发 run 会互相污染，
同 seed 也不再决定同结果。

三条约束在接口形态上强制，而不是靠约定：

1. **唯一构造点**：``RunRNG(sampling, device_class)`` 用显式 generator，不读不写全局 RNG；
   seed 不再是可与采样入参分叉的独立参数——run 从 :class:`SamplingConfig` 构造，
   ``sample_logits`` 不再接收 config，两个 seed 源在类型上就不可能不一致；
2. **随机流不可外借**：``generator`` 属性不公开，采样只能走
   :meth:`RunRNG.sample_logits`，调用方无法缓存或复用私有 generator；
3. **每个采样点都计入同一流**：Kronos 自回归每 step 有 s1/s2 两次采样，两次共用
   同一 generator（ADR-005 §2），因此一个 run 的随机流是
   ``horizon × 2`` 次 multinomial 的确定序列。

``sample_logits`` 的 ``temperature`` / top-k / top-p 处理顺序与上游
``sample_from_logits`` 逐点一致（``vendor/kronos.py:370``），差别只在 generator
的所有权（ADR-005）。

可复现性承诺边界（ADR-005 §4）：同 input + 同模型内容 + 同 seed + 同 sampling
config + 同 device-class / dtype / torch 版本 → raw samples 逐位一致；跨设备 /
跨 dtype / 跨 torch 版本不作承诺，这些维由 ``runtime_version`` 与 ``artifact_identity``
记录并进入 artifact key。
"""

from __future__ import annotations

import torch

from kronos_ai.domain.forecast import SamplingConfig
from kronos_ai.errors import ConfigurationError
from kronos_ai.forecast.backends.kronos.vendor import top_k_top_p_filtering


class RunRNG:
    """一次 run 的受控随机流；不共享、不触碰全局 RNG。

    generator 是单下划线约定的私有字段（Python 无真私有）；不提供 accessor，
    采样入口只有 :meth:`sample_logits`。仓库级守卫测试断言 ``torch`` 的随机算子
    （``multinomial`` / ``rand`` / ``randn`` / ``normal`` / ``uniform`` / ``randint`` /
    ``randperm`` / ``bernoulli``）只出现在本模块（vendor 白名单除外），使“绕开随机流”
    在 CI 层可检测。

    seed / temperature / top-k / top-p 全部取自构造期的同一个 :class:`SamplingConfig`：
    采样点的随机参数与 seed 不再有第二个来源，因此不可能出现「run 用 seed=A 采样、
    metadata 记录 seed=B」的分叉（§9/§14）。
    """

    __slots__ = ("_generator", "_sampling")

    def __init__(self, sampling: SamplingConfig, *, device_class: str) -> None:
        try:
            generator = torch.Generator(device=device_class)
        except (RuntimeError, TypeError) as exc:  # 设备后端不支持 generator
            raise ConfigurationError(
                f"cannot create torch.Generator on device {device_class!r}: {exc}"
            ) from exc
        generator.manual_seed(sampling.seed)
        self._generator = generator
        self._sampling = sampling

    @property
    def sampling(self) -> SamplingConfig:
        """本 run 生效的采样参数（含 seed）；只读，供 provenance 单点取用。"""
        return self._sampling

    def sample_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """从 ``logits`` 抽一个 token 下标，形状 ``(..., 1)``。

        应用顺序与上游一致：temperature → top-k/top-p 过滤 → softmax → multinomial。
        采样参数取自构造 :class:`RunRNG` 的同一个 :class:`SamplingConfig`。
        """
        sampling = self._sampling
        scaled = logits / sampling.temperature
        if sampling.top_k > 0 or sampling.top_p < 1.0:
            scaled = top_k_top_p_filtering(scaled, top_k=sampling.top_k, top_p=sampling.top_p)
        probs = torch.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=self._generator)
