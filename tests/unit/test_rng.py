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


def draws(run_rng: RunRNG, *, steps: int = 6) -> list[int]:
    # SamplingConfig 只提供 temperature/top_k/top_p；seed 在 RunRNG 构造期固定
    config = SamplingConfig(seed=0, sample_count=1, top_p=1.0)
    return [int(run_rng.sample_logits(LOGITS, config).item()) for _ in range(steps)]


class TestRunRNGDeterminism:
    def test_same_seed_same_stream(self) -> None:
        assert draws(RunRNG(seed=11, device_class="cpu")) == draws(
            RunRNG(seed=11, device_class="cpu")
        )

    def test_different_seed_different_stream(self) -> None:
        assert draws(RunRNG(seed=12, device_class="cpu")) != draws(
            RunRNG(seed=11, device_class="cpu")
        )

    def test_previous_run_does_not_affect_next(self) -> None:
        """同 seed 的 run 结果不依赖此前跑过什么（无隐藏全局状态）。"""
        baseline = draws(RunRNG(seed=42, device_class="cpu"))
        draws(RunRNG(seed=7, device_class="cpu"))
        torch.manual_seed(999)
        _ = torch.rand(32)
        assert draws(RunRNG(seed=42, device_class="cpu")) == baseline

    def test_stream_advances_monotonically(self) -> None:
        """每次 sample_logits 恰好消耗一次抽取：前 k 次 == 前 k+1 次的前缀，且序列非常量。"""
        short = draws(RunRNG(seed=2026, device_class="cpu"), steps=6)
        long = draws(RunRNG(seed=2026, device_class="cpu"), steps=7)
        assert long[:6] == short
        # 如果每次调用都重新 seed，序列会是常数；非退化分布下必须出现变化
        assert len(set(short)) > 1


class TestRunRNGGlobalIsolation:
    def test_construction_does_not_touch_global_rng(self) -> None:
        torch.manual_seed(0)
        before = torch.random.get_rng_state()
        RunRNG(seed=99, device_class="cpu")
        assert torch.equal(before, torch.random.get_rng_state())

    def test_sampling_does_not_touch_global_rng(self) -> None:
        run_rng = RunRNG(seed=3, device_class="cpu")
        torch.manual_seed(1234)
        before = torch.random.get_rng_state()
        draws(run_rng)
        assert torch.equal(before, torch.random.get_rng_state())

    def test_global_seed_does_not_change_stream(self) -> None:
        """全局 seed 是环境状态，不应成为 run 结果的一部分。"""
        torch.manual_seed(1)
        first = draws(RunRNG(seed=5, device_class="cpu"))
        torch.manual_seed(2)
        second = draws(RunRNG(seed=5, device_class="cpu"))
        assert first == second


class TestRunRNGConcurrency:
    def test_interleaved_threads_match_serial(self) -> None:
        """两个 run 在两个线程中交错采样，结果必须等于各自串行执行。"""
        serial_a = draws(RunRNG(seed=101, device_class="cpu"), steps=40)
        serial_b = draws(RunRNG(seed=202, device_class="cpu"), steps=40)

        results: dict[str, list[int]] = {}
        barrier = threading.Barrier(2)

        def worker(name: str, seed: int) -> None:
            run_rng = RunRNG(seed=seed, device_class="cpu")
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
            RunRNG(seed=1, device_class="not-a-device")


class TestRunRNGSamplingSemantics:
    def test_zero_probability_tokens_never_drawn(self) -> None:
        """top_k=1 时唯一存活 token 必被抽中（过滤顺序与上游一致）。"""
        config = SamplingConfig(seed=7, sample_count=1, top_k=1)
        run_rng = RunRNG(seed=7, device_class="cpu")
        logits = torch.tensor([[0.1, 5.0, 0.2]], dtype=torch.float32)
        for _ in range(20):
            assert int(run_rng.sample_logits(logits, config).item()) == 1

    def test_output_shape_preserves_leading_dims(self) -> None:
        config = SamplingConfig(seed=7, sample_count=1)
        run_rng = RunRNG(seed=7, device_class="cpu")
        logits = torch.zeros((3, 5), dtype=torch.float32)
        assert run_rng.sample_logits(logits, config).shape == (3, 1)


class TestMultinomialChokePoint:
    def test_multinomial_only_in_rng_module(self) -> None:
        """§9 守卫：随机采样只能在 rng.py 内发生（vendor 是上游副本，白名单除外）。"""
        package_root = Path(__file__).resolve().parents[2] / "src" / "kronos_ai"
        offenders = []
        for path in package_root.rglob("*.py"):
            if "vendor" in path.parts:
                continue  # ADR-020 白名单：上游逐字节副本，不在 v2 路径上
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                # 只认真实调用/属性访问，不误伤文档字符串里的提及
                if isinstance(node, ast.Attribute) and node.attr == "multinomial":
                    offenders.append(str(path.relative_to(package_root)))
                    break
        assert offenders == ["forecast/backends/kronos/rng.py"]
