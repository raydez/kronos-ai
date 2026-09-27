"""RunRNG 隔离回归（§9、ADR-005，RX-KAI-011）。

断言「随机流所有权」的四条性质：

1. 同 seed → 同流；不同 seed → 不同流；
2. 构造与采样都不读写全局 torch RNG；
3. 并发 run 互不干扰（两个线程交错采样 == 各自串行采样）；
4. 上一 run 的采样不影响下一 run 的结果（无隐藏全局状态）。
"""

from __future__ import annotations

import ast
import threading
from pathlib import Path

import pytest
import torch

from kronos_ai.domain.forecast import SamplingConfig
from kronos_ai.errors import ConfigurationError
from kronos_ai.forecast.backends.kronos.rng import RunRNG

LOGITS = torch.tensor([[2.0, 1.0, 0.5, -1.0]], dtype=torch.float32)


def make_rng(seed: int, *, sample_count: int = 1, top_k: int = 0, top_p: float = 1.0) -> RunRNG:
    # seed 与采样参数来自同一个 SamplingConfig：RunRNG 持有它，sample_logits 不再收 config
    config = SamplingConfig(seed=seed, sample_count=sample_count, top_k=top_k, top_p=top_p)
    return RunRNG(config, device_class="cpu")


def draws(run_rng: RunRNG, *, steps: int = 6) -> list[int]:
    return [int(run_rng.sample_logits(LOGITS).item()) for _ in range(steps)]


class TestRunRNGDeterminism:
    def test_same_seed_same_stream(self) -> None:
        assert draws(make_rng(11)) == draws(make_rng(11))

    def test_different_seed_different_stream(self) -> None:
        assert draws(make_rng(12)) != draws(make_rng(11))

    def test_previous_run_does_not_affect_next(self) -> None:
        """同 seed 的 run 结果不依赖此前跑过什么（无隐藏全局状态）。"""
        baseline = draws(make_rng(42))
        draws(make_rng(7))
        torch.manual_seed(999)
        _ = torch.rand(32)
        assert draws(make_rng(42)) == baseline

    def test_stream_advances_monotonically(self) -> None:
        """每次 sample_logits 恰好消耗一次抽取：前 k 次 == 前 k+1 次的前缀，且序列非常量。"""
        short = draws(make_rng(2026), steps=6)
        long = draws(make_rng(2026), steps=7)
        assert long[:6] == short
        # 如果每次调用都重新 seed，序列会是常数；非退化分布下必须出现变化
        assert len(set(short)) > 1


class TestRunRNGGlobalIsolation:
    def test_construction_does_not_touch_global_rng(self) -> None:
        torch.manual_seed(0)
        before = torch.random.get_rng_state()
        make_rng(99)
        assert torch.equal(before, torch.random.get_rng_state())

    def test_sampling_does_not_touch_global_rng(self) -> None:
        run_rng = make_rng(3)
        torch.manual_seed(1234)
        before = torch.random.get_rng_state()
        draws(run_rng)
        assert torch.equal(before, torch.random.get_rng_state())

    def test_global_seed_does_not_change_stream(self) -> None:
        """全局 seed 是环境状态，不应成为 run 结果的一部分。"""
        torch.manual_seed(1)
        first = draws(make_rng(5))
        torch.manual_seed(2)
        second = draws(make_rng(5))
        assert first == second


class TestRunRNGConcurrency:
    def test_interleaved_threads_match_serial(self) -> None:
        """两个 run 在两个线程中交错采样，结果必须等于各自串行执行。"""
        serial_a = draws(make_rng(101), steps=40)
        serial_b = draws(make_rng(202), steps=40)

        results: dict[str, list[int]] = {}
        barrier = threading.Barrier(2)

        def worker(name: str, seed: int) -> None:
            run_rng = make_rng(seed)
            barrier.wait()
            results[name] = draws(run_rng, steps=40)

        threads = [
            threading.Thread(target=worker, args=("a", 101)),
            threading.Thread(target=worker, args=("b", 202)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert results["a"] == serial_a
        assert results["b"] == serial_b


class TestRunRNGValidation:
    def test_unsupported_device_fails_explicitly(self) -> None:
        with pytest.raises(ConfigurationError, match=r"cannot create torch\.Generator"):
            RunRNG(SamplingConfig(seed=1), device_class="not-a-device")

    def test_sampling_property_is_the_construction_config(self) -> None:
        """seed 只有一个来源：RunRNG 暴露的 config 就是构造它的那一个。"""
        config = SamplingConfig(seed=17, temperature=0.5, top_k=3, top_p=0.8)
        run_rng = RunRNG(config, device_class="cpu")
        assert run_rng.sampling is config


class TestRunRNGSamplingSemantics:
    def test_zero_probability_tokens_never_drawn(self) -> None:
        """top_k=1 时唯一存活 token 必被抽中（过滤顺序与上游一致）。"""
        run_rng = make_rng(7, top_k=1)
        logits = torch.tensor([[0.1, 5.0, 0.2]], dtype=torch.float32)
        for _ in range(20):
            assert int(run_rng.sample_logits(logits).item()) == 1

    def test_output_shape_preserves_leading_dims(self) -> None:
        run_rng = make_rng(7)
        logits = torch.zeros((3, 5), dtype=torch.float32)
        assert run_rng.sample_logits(logits).shape == (3, 1)


class TestMultinomialChokePoint:
    def test_global_rng_draw_only_in_rng_module(self) -> None:
        """§9 守卫：任何隐式读全局 RNG 的采样都只允许在 rng.py（vendor 白名单除外）。

        覆盖 multinomial 之外的 ``rand/randn/normal/uniform/randint/randperm/bernoulli``：
        只盯 multinomial 会漏掉「用 torch.rand 生成随机性」的绕开路径。

        这是 best-effort 的 AST 近似：只识别接收者为 ``torch`` 的属性访问
        （``torch.rand(...)`` 等）。不覆盖 ``from torch import rand; rand(...)``、
        ``torch.rand_like``、``Tensor.uniform_``、``torch.distributions.*.sample()``
        等变体；但也不会误伤 ``random.uniform`` 这类同名属性。
        """
        package_root = Path(__file__).resolve().parents[2] / "src" / "kronos_ai"
        draw_ops = {
            "multinomial",
            "rand",
            "randn",
            "normal",
            "uniform",
            "randint",
            "randperm",
            "bernoulli",
        }
        offenders: list[str] = []
        for path in package_root.rglob("*.py"):
            if "vendor" in path.parts:
                continue  # ADR-020 白名单：上游逐字节副本，不在 v2 路径上
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                # 只认 ``torch.<op>`` 形式的调用/属性访问，避免误伤 random.uniform 等同名属性；
                # 也不误伤文档字符串里的提及。
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr in draw_ops
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "torch"
                ):
                    offenders.append(str(path.relative_to(package_root)))
                    break
        assert offenders == ["forecast/backends/kronos/rng.py"]
