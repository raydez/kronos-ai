"""Compute cost probe（§16 / §49）单元测试（RX-KAI-018）。

覆盖四件事：

1. **口径正确**：最近秩法分位数、latency summary 的自洽性、throughput 的分母
   （只算被测区间，不含采样与预算检查开销）、artifact bytes 摊到 forecast origin；
2. **§16 的三条硬约束**：组数 ≤ 10（全因子笛卡尔积被结构性拒绝，报告层再挡一次）、
   预算上限必填、超限只产出 ``INFEASIBLE_ON_THIS_DEVICE`` 而不是「跑完了」
   （含末轮撑爆预算的事后复核、MEASURED 必须跑满全部 repeats）；
3. **显式失败（ADR-010）**：RAM 不可观测时拒绝运行、手工删组的报告被拒、
   从 INFEASIBLE 组合外推被拒；
4. **throughput-memory curve**：只含 MEASURED 组合、按 ``forward_width`` 升序、
   一个都没测到就是空曲线；
5. **廉价可跑**：本模块不加载 torch / pandas（cost probe 面向 baseline 也能在 CI 跑），
   且整个 probe 在子进程里可端到端跑通。

预算与时钟全部注入（:class:`_StepClock` / :class:`_ScriptedSampler`），因此除 golden
hash 外所有断言都是确定性的——probe 引擎的正确性不依赖于真实机器的速度。
"""

from __future__ import annotations

import math
import subprocess
import sys
from typing import get_args

import pytest
from pydantic import ValidationError

from kronos_ai.errors import ConfigurationError
from kronos_ai.evaluation.compute_metrics import (
    COMPUTE_METRICS_VERSION,
    MAX_PROBE_CELLS,
    ComputeBudgetReport,
    LatencySummary,
    PilotMatrix,
    ProbeBudget,
    ProbeCell,
    ProbeCellResult,
    ProbeDeviceClass,
    ProbeObservation,
    ProcessResourceSampler,
    ResourcePeak,
    ResourceSample,
    compute_report_hash,
    max_rss_bytes,
    percentile,
    project_full_run_seconds,
    run_cost_probe,
)

#: 确定性报告（:class:`_StepClock` step=0.5s + :class:`_ScriptedSampler` + 固定 matrix）
#: 的 report_hash。改动矩阵 / 预算 / 指标数学都会让它变化。
GOLDEN_REPORT_HASH = "63883e293f48d042ff71c2beee273f20aa68e808760d3b74cb4f9432699b984f"

MATRIX = PilotMatrix(
    device_class="cpu",
    sample_counts=(16, 32),
    batch_sizes=(1, 2),
    symbols=300,
    forecast_origins=100,
)


class _StepClock:
    """每次调用前进固定步长的假时钟（perf_counter 语义：单调、单位秒）。"""

    def __init__(self, step: float = 0.5) -> None:
        self._now = 0.0
        self._step = step

    def __call__(self) -> float:
        value = self._now
        self._now += self._step
        return value


class _ScriptedSampler:
    """按脚本返回资源采样；脚本用尽后重复最后一项。"""

    def __init__(self, samples: list[ResourceSample]) -> None:
        assert samples, "scripted sampler needs at least one sample"
        self._samples = list(samples)
        self.calls = 0

    def sample(self) -> ResourceSample:
        index = min(self.calls, len(self._samples) - 1)
        self.calls += 1
        return self._samples[index]


def _measure(*, forecast_count: int = 1, artifact_bytes: int | None = None):
    """构造一个返回固定观测的 measure。"""

    def _run(cell: ProbeCell) -> ProbeObservation:
        assert cell.batch_size >= 1
        return ProbeObservation(
            forecast_count=forecast_count,
            artifact_bytes=artifact_bytes,
        )

    return _run


def _cell_result_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "cell": {"device_class": "cpu", "sample_count": 16, "batch_size": 1},
        "status": "MEASURED",
        "completed_repeats": 1,
        "wall_clock_seconds": 1.0,
        "resources": {"ram_bytes": 1024},
        "latency": {
            "count": 1,
            "p50_ms": 10.0,
            "p95_ms": 10.0,
            "mean_ms": 10.0,
            "min_ms": 10.0,
            "max_ms": 10.0,
        },
        "throughput_per_second": 100.0,
    }
    payload.update(overrides)
    return payload


class TestPercentile:
    """最近秩法：p95 必须是真实观测到的某一次延迟。"""

    def test_nearest_rank_on_four_samples(self) -> None:
        samples = [1.0, 2.0, 3.0, 4.0]
        assert percentile(samples, 0.25) == 1.0  # ceil(1.0) = 1 -> index 0
        assert percentile(samples, 0.5) == 2.0  # ceil(2.0) = 2 -> index 1
        assert percentile(samples, 0.51) == 3.0  # ceil(2.04) = 3 -> index 2
        assert percentile(samples, 0.95) == 4.0  # ceil(3.8) = 4 -> index 3
        assert percentile(samples, 1.0) == 4.0

    def test_is_order_independent_and_never_interpolates(self) -> None:
        shuffled = [4.0, 1.0, 3.0, 2.0]
        for q in (0.1, 0.5, 0.95, 1.0):
            assert percentile(shuffled, q) in {1.0, 2.0, 3.0, 4.0}
        # 只在其一元素，任何分位都只能是那个元素（插值会给出别的数字）
        assert percentile([7.0], 0.5) == 7.0
        assert percentile([7.0], 0.95) == 7.0

    @pytest.mark.parametrize("q", [0.0, -0.1, 1.5])
    def test_quantile_out_of_range_is_rejected(self, q: float) -> None:
        with pytest.raises(ConfigurationError, match="quantile must be in"):
            percentile([1.0], q)

    @pytest.mark.parametrize("bad", [[], [-1.0], [math.nan], [math.inf]])
    def test_bad_samples_are_rejected(self, bad: list[float]) -> None:
        with pytest.raises(ConfigurationError):
            percentile(bad, 0.5)


class TestLatencySummary:
    def test_from_samples_matches_percentile_definition(self) -> None:
        summary = LatencySummary.from_samples([10.0, 20.0, 30.0, 40.0])
        assert summary.count == 4
        assert summary.p50_ms == 20.0
        assert summary.p95_ms == 40.0
        assert summary.min_ms == 10.0
        assert summary.max_ms == 40.0
        assert summary.mean_ms == 25.0

    def test_empty_samples_are_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="at least one sample"):
            LatencySummary.from_samples([])

    def test_inconsistent_distribution_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="inconsistent"):
            LatencySummary(
                count=2, p50_ms=50.0, p95_ms=10.0, mean_ms=30.0, min_ms=10.0, max_ms=40.0
            )

    def test_mean_outside_min_max_is_rejected(self) -> None:
        """非负样本的均值必然落在 [min, max] 内，超出即为不一致的分布。"""
        with pytest.raises(ValidationError, match=r"mean=100\.0"):
            LatencySummary(
                count=3, p50_ms=10.0, p95_ms=30.0, mean_ms=100.0, min_ms=10.0, max_ms=30.0
            )


class TestDeviceClassContract:
    def test_matches_runtime_device_class(self) -> None:
        """本模块不得 import torch，因此 device-class 字面量在此重述并由本用例钉住。"""
        from kronos_ai.forecast.backends.kronos.runtime import DeviceClass

        assert get_args(ProbeDeviceClass) == get_args(DeviceClass)


class TestPilotMatrix:
    def test_cells_are_sample_major_and_ascending(self) -> None:
        cells = MATRIX.cells
        assert [cell.label for cell in cells] == [
            "cpu/samples=16/batch=1",
            "cpu/samples=16/batch=2",
            "cpu/samples=32/batch=1",
            "cpu/samples=32/batch=2",
        ]
        assert [cell.forward_width for cell in cells] == [16, 32, 32, 64]

    def test_declared_forecasts_is_symbols_times_origins(self) -> None:
        assert MATRIX.declared_forecasts == 300 * 100

    def test_cartesian_product_above_cap_is_rejected(self) -> None:
        """§16 的 4 sample × 3 batch = 12 组示例必须被拒，并提示「裁剪扫描」。"""
        with pytest.raises(ValidationError, match="caps a probe at 10"):
            PilotMatrix(
                device_class="mps",
                sample_counts=(1, 16, 32, 64),
                batch_sizes=(1, 4, 16),
                symbols=50,
                forecast_origins=100,
            )

    def test_exactly_ten_cells_is_allowed(self) -> None:
        matrix = PilotMatrix(
            device_class="mps",
            sample_counts=(1, 8, 16, 32, 64),
            batch_sizes=(1, 4),
            symbols=50,
            forecast_origins=100,
        )
        assert len(matrix.cells) == MAX_PROBE_CELLS == 10

    def test_device_is_locked_to_a_single_value(self) -> None:
        """设备不是扫描轴：多设备要用多个单点 matrix（§16）。"""
        with pytest.raises(ValidationError):
            PilotMatrix(
                device_class=("cpu", "mps"),  # type: ignore[arg-type]
                sample_counts=(16,),
                batch_sizes=(1,),
                symbols=1,
                forecast_origins=1,
            )

    @pytest.mark.parametrize(
        ("axis", "values"),
        [
            ("sample_counts", ()),
            ("batch_sizes", ()),
            ("sample_counts", (16, 16)),
            ("sample_counts", (32, 16)),
            ("batch_sizes", (0, 1)),
        ],
    )
    def test_bad_axis_is_rejected(self, axis: str, values: tuple[int, ...]) -> None:
        kwargs: dict[str, object] = {
            "device_class": "cpu",
            "sample_counts": (16,),
            "batch_sizes": (1,),
            "symbols": 1,
            "forecast_origins": 1,
            axis: values,
        }
        with pytest.raises(ValidationError):
            PilotMatrix(**kwargs)  # type: ignore[arg-type]


class TestProbeBudget:
    def test_caps_have_no_defaults(self) -> None:
        """§16「Probe 不允许无上限运行」：不给上限就构造不出来。"""
        with pytest.raises(ValidationError):
            ProbeBudget()  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            ProbeBudget(wall_clock_seconds_per_cell=1.0)  # type: ignore[call-arg]

    @pytest.mark.parametrize(
        ("wall", "memory"),
        [(0.0, 1), (-1.0, 1), (math.inf, 1), (1.0, 0), (1.0, -1)],
    )
    def test_non_positive_or_infinite_caps_are_rejected(self, wall: float, memory: int) -> None:
        with pytest.raises(ValidationError):
            ProbeBudget(wall_clock_seconds_per_cell=wall, memory_bytes=memory)


class TestResourcePeak:
    def test_exceeded_reports_the_offending_dimension(self) -> None:
        peak = ResourcePeak(ram_bytes=100, vram_bytes=900)
        assert peak.exceeded(500) == "vram peak 900 bytes > memory budget 500 bytes"
        assert ResourcePeak(ram_bytes=900, vram_bytes=100).exceeded(500) == (
            "ram peak 900 bytes > memory budget 500 bytes"
        )

    def test_within_budget_and_unobservable_are_both_none(self) -> None:
        assert ResourcePeak(ram_bytes=100).exceeded(500) is None
        # 不可观测 ≠ 超限；「不可观测」由 _PeakTracker.require_observable 显式失败
        assert ResourcePeak().exceeded(500) is None

    def test_negative_values_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ResourcePeak(ram_bytes=-1)
        with pytest.raises(ValueError, match="non-negative"):
            ResourceSample(vram_bytes=-1)


class TestProcessResourceSampler:
    def test_max_rss_is_positive(self) -> None:
        assert max_rss_bytes() > 1_000_000  # 解释器自身就远超 1MB

    def test_vram_is_none_without_injected_sampler(self) -> None:
        sample = ProcessResourceSampler().sample()
        assert sample.ram_bytes is not None and sample.ram_bytes > 0
        assert sample.vram_bytes is None

    def test_vram_sampler_is_used_when_injected(self) -> None:
        sample = ProcessResourceSampler(vram_sampler=lambda: 42).sample()
        assert sample.vram_bytes == 42


class TestRunCostProbe:
    def test_measures_every_cell_with_injected_clock(self) -> None:
        report = run_cost_probe(
            matrix=MATRIX,
            budget=ProbeBudget(wall_clock_seconds_per_cell=100.0, memory_bytes=10**12),
            repeats_per_cell=3,
            measure=_measure(),
            resource_sampler=_ScriptedSampler([ResourceSample(ram_bytes=1024)]),
            clock=_StepClock(0.5),
        )
        assert report.all_feasible
        assert len(report.cells) == 4
        for cell in report.measured_cells:
            assert cell.completed_repeats == 3
            assert cell.latency is not None
            assert cell.latency.p50_ms == 500.0  # 两次 clock() 调用，步长 0.5s
            assert cell.latency.max_ms == 500.0
            # forecast_count=1、每次迭代 0.5s -> 2 forecast/秒
            assert cell.throughput_per_second == pytest.approx(2.0)
            assert cell.resources.ram_bytes == 1024

    def test_throughput_counts_every_forecast_in_a_batched_iteration(self) -> None:
        report = run_cost_probe(
            matrix=PilotMatrix(
                device_class="cpu",
                sample_counts=(16,),
                batch_sizes=(2,),
                symbols=10,
                forecast_origins=10,
            ),
            budget=ProbeBudget(wall_clock_seconds_per_cell=100.0, memory_bytes=10**12),
            repeats_per_cell=2,
            measure=_measure(forecast_count=2),
            resource_sampler=_ScriptedSampler([ResourceSample(ram_bytes=1)]),
            clock=_StepClock(0.5),
        )
        cell = report.cells[0]
        assert cell.throughput_per_second == pytest.approx(4.0)
        assert report.total_forecasts == 4  # 2 repeats × batch 2

    def test_artifact_bytes_and_cache_hit_ratio_are_aggregated(self) -> None:
        flags = iter([True, False])
        observations: list[ProbeObservation] = [
            ProbeObservation(artifact_bytes=1000, was_cached=next(flags)),
            ProbeObservation(artifact_bytes=2000, was_cached=next(flags)),
            ProbeObservation(artifact_bytes=3000, was_cached=True),
        ]
        counter = {"index": 0}

        def measure(_cell: ProbeCell) -> ProbeObservation:
            observation = observations[counter["index"]]
            counter["index"] += 1
            return observation

        report = run_cost_probe(
            matrix=PilotMatrix(
                device_class="cpu",
                sample_counts=(16,),
                batch_sizes=(1,),
                symbols=5,
                forecast_origins=5,
            ),
            budget=ProbeBudget(wall_clock_seconds_per_cell=100.0, memory_bytes=10**12),
            repeats_per_cell=3,
            measure=measure,
            resource_sampler=_ScriptedSampler([ResourceSample(ram_bytes=1)]),
            clock=_StepClock(0.5),
        )
        cell = report.cells[0]
        assert cell.artifact_bytes_per_forecast == pytest.approx(2000.0)
        assert cell.cache_hit_ratio == pytest.approx(2 / 3)

    def test_absent_optional_observations_stay_none(self) -> None:
        report = run_cost_probe(
            matrix=MATRIX,
            budget=ProbeBudget(wall_clock_seconds_per_cell=100.0, memory_bytes=10**12),
            repeats_per_cell=1,
            measure=_measure(),
            resource_sampler=_ScriptedSampler([ResourceSample(ram_bytes=1)]),
            clock=_StepClock(0.5),
        )
        assert report.cells[0].artifact_bytes_per_forecast is None
        assert report.cells[0].cache_hit_ratio is None

    def test_wall_clock_budget_aborts_the_cell(self) -> None:
        """步长 0.5s、每轮迭代消耗 3 个 tick（检查 / before / after）：第 1、2 轮迭代前的
        elapsed 依次为 0.5 / 2.0，第 3 轮读到 3.5 > 2.5 -> 中止，完成 2 次。"""
        report = run_cost_probe(
            matrix=MATRIX,
            budget=ProbeBudget(wall_clock_seconds_per_cell=2.5, memory_bytes=10**12),
            repeats_per_cell=10,
            measure=_measure(),
            resource_sampler=_ScriptedSampler([ResourceSample(ram_bytes=1)]),
            clock=_StepClock(0.5),
        )
        assert not report.all_feasible
        for cell in report.infeasible_cells:
            assert cell.status == "INFEASIBLE_ON_THIS_DEVICE"
            assert cell.completed_repeats == 2
            assert cell.abort_reason is not None
            assert "wall-clock budget 2.5s per cell exceeded after 2 of 10 repeats" in (
                cell.abort_reason
            )
            # 被截断的运行里算出来的 p50 / throughput 不许出现在报告里（§16）
            assert cell.latency is None
            assert cell.throughput_per_second is None
            assert cell.cache_hit_ratio is None

    def test_memory_budget_aborts_the_cell(self) -> None:
        report = run_cost_probe(
            matrix=MATRIX,
            budget=ProbeBudget(wall_clock_seconds_per_cell=100.0, memory_bytes=500),
            repeats_per_cell=10,
            measure=_measure(),
            resource_sampler=_ScriptedSampler(
                [
                    ResourceSample(ram_bytes=100),  # 迭代前基线
                    ResourceSample(ram_bytes=200),  # 第 1 次迭代后
                    ResourceSample(ram_bytes=1000),  # 第 2 次迭代后 -> 超限
                ]
            ),
            clock=_StepClock(0.5),
        )
        cell = report.cells[0]
        assert cell.completed_repeats == 2
        assert cell.resources.ram_bytes == 1000
        assert cell.abort_reason is not None
        assert "memory budget exceeded after 2 repeats" in cell.abort_reason
        assert "ram peak 1000 bytes > memory budget 500 bytes" in cell.abort_reason

    def test_artifact_bytes_are_reported_per_forecast_origin(self) -> None:
        """§49 的口径是 artifact bytes / forecast origin：batch=4、每轮 4000 bytes 时
        应为 1000，而不是每轮的 4000。"""
        report = run_cost_probe(
            matrix=PilotMatrix(
                device_class="cpu",
                sample_counts=(16,),
                batch_sizes=(4,),
                symbols=8,
                forecast_origins=8,
            ),
            budget=ProbeBudget(wall_clock_seconds_per_cell=100.0, memory_bytes=10**12),
            repeats_per_cell=2,
            measure=_measure(forecast_count=4, artifact_bytes=4000),
            resource_sampler=_ScriptedSampler([ResourceSample(ram_bytes=1)]),
            clock=_StepClock(0.5),
        )
        cell = report.cells[0]
        assert cell.artifact_bytes_per_forecast == pytest.approx(1000.0)
        assert report.throughput_memory_curve[0].forward_width == 64

    def test_wall_clock_overshoot_in_the_final_repeat_is_infeasible(self) -> None:
        """``repeats=1`` 时循环前的检查看不到末轮耗时：事后复核必须把它标成超限
        （§16「任一组合超限」），否则 3600s 的一轮会被报成 MEASURED。"""
        report = run_cost_probe(
            matrix=MATRIX,
            budget=ProbeBudget(wall_clock_seconds_per_cell=1.0, memory_bytes=10**12),
            repeats_per_cell=1,
            measure=_measure(),
            resource_sampler=_ScriptedSampler([ResourceSample(ram_bytes=1)]),
            clock=_StepClock(0.5),
        )
        assert not report.all_feasible
        assert report.measured_cells == ()
        for cell in report.cells:
            assert cell.status == "INFEASIBLE_ON_THIS_DEVICE"
            assert cell.completed_repeats == 1
            assert cell.abort_reason is not None
            assert "wall-clock budget 1.0s per cell exceeded" in cell.abort_reason
            assert "the cell took 2.0s over 1 repeats" in cell.abort_reason
            assert cell.latency is None
            assert cell.throughput_per_second is None

    def test_unobservable_memory_dimension_refuses_to_run(self) -> None:
        with pytest.raises(ConfigurationError, match="no ram observation"):
            run_cost_probe(
                matrix=MATRIX,
                budget=ProbeBudget(wall_clock_seconds_per_cell=100.0, memory_bytes=500),
                repeats_per_cell=1,
                measure=_measure(),
                resource_sampler=_ScriptedSampler([ResourceSample()]),
                clock=_StepClock(0.5),
            )

    def test_vram_only_sampler_is_rejected(self) -> None:
        """只有一个内存上限：观测不到宿主 RAM 就等于对 RAM 无界（vram-only 不够）。"""
        with pytest.raises(ConfigurationError, match="vram-only sampling is not enough"):
            run_cost_probe(
                matrix=MATRIX,
                budget=ProbeBudget(wall_clock_seconds_per_cell=100.0, memory_bytes=10**12),
                repeats_per_cell=1,
                measure=_measure(),
                resource_sampler=_ScriptedSampler([ResourceSample(vram_bytes=2048)]),
                clock=_StepClock(0.5),
            )

    def test_resource_sampler_without_sample_is_rejected(self) -> None:
        """untyped 调用方传入不合契约的采样器时，显式失败而不是等到 AttributeError。"""
        with pytest.raises(ConfigurationError, match=r"must implement ResourceSampler\.sample"):
            run_cost_probe(
                matrix=MATRIX,
                budget=ProbeBudget(wall_clock_seconds_per_cell=1.0, memory_bytes=1),
                repeats_per_cell=1,
                measure=_measure(),
                resource_sampler=object(),  # type: ignore[arg-type]
                clock=_StepClock(0.5),
            )

    def test_repeats_must_be_positive(self) -> None:
        with pytest.raises(ConfigurationError, match="repeats_per_cell must be >= 1"):
            run_cost_probe(
                matrix=MATRIX,
                budget=ProbeBudget(wall_clock_seconds_per_cell=1.0, memory_bytes=1),
                repeats_per_cell=0,
                measure=_measure(),
                resource_sampler=_ScriptedSampler([ResourceSample(ram_bytes=1)]),
                clock=_StepClock(0.5),
            )

    def test_default_sampler_and_real_clock_work(self) -> None:
        report = run_cost_probe(
            matrix=PilotMatrix(
                device_class="cpu",
                sample_counts=(8,),
                batch_sizes=(1,),
                symbols=1,
                forecast_origins=1,
            ),
            budget=ProbeBudget(wall_clock_seconds_per_cell=60.0, memory_bytes=10**12),
            repeats_per_cell=1,
            measure=_measure(),
        )
        assert report.all_feasible
        assert report.total_wall_clock_seconds > 0.0

    def test_non_advancing_clock_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="non-positive elapsed time"):
            run_cost_probe(
                matrix=MATRIX,
                budget=ProbeBudget(wall_clock_seconds_per_cell=100.0, memory_bytes=10**12),
                repeats_per_cell=1,
                measure=_measure(),
                resource_sampler=_ScriptedSampler([ResourceSample(ram_bytes=1)]),
                clock=lambda: 0.0,
            )


class TestComputeBudgetReport:
    def _report(self) -> ComputeBudgetReport:
        return run_cost_probe(
            matrix=MATRIX,
            budget=ProbeBudget(wall_clock_seconds_per_cell=100.0, memory_bytes=10**12),
            repeats_per_cell=2,
            measure=_measure(artifact_bytes=1024),
            resource_sampler=_ScriptedSampler([ResourceSample(ram_bytes=2048)]),
            clock=_StepClock(0.5),
        )

    def test_golden_report_hash(self) -> None:
        assert self._report().report_hash == GOLDEN_REPORT_HASH

    def test_hash_binds_matrix_and_budget(self) -> None:
        report = self._report()
        payload = {
            "matrix": report.matrix,
            "budget": report.budget,
            "repeats_per_cell": report.repeats_per_cell,
            "cells": report.cells,
        }
        digest = compute_report_hash(**payload)  # type: ignore[arg-type]
        assert digest == GOLDEN_REPORT_HASH == report.report_hash

        wider = PilotMatrix(
            device_class="cpu",
            sample_counts=(16, 32),
            batch_sizes=(1, 2),
            symbols=301,
            forecast_origins=100,
        )
        assert (
            compute_report_hash(
                matrix=wider,
                budget=report.budget,
                repeats_per_cell=report.repeats_per_cell,
                cells=report.cells,
            )
            != GOLDEN_REPORT_HASH
        )
        # 换了预算上限就是换了实验条件，hash 必须变
        assert (
            compute_report_hash(
                matrix=report.matrix,
                budget=ProbeBudget(wall_clock_seconds_per_cell=101.0, memory_bytes=10**12),
                repeats_per_cell=report.repeats_per_cell,
                cells=report.cells,
            )
            != GOLDEN_REPORT_HASH
        )

    def test_hash_binds_metrics_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """换了指标数学就算换了实验条件：version 在 hash 载荷里，digest 必须变。"""
        report = self._report()
        assert report.report_hash == GOLDEN_REPORT_HASH
        monkeypatch.setattr(
            "kronos_ai.evaluation.compute_metrics.COMPUTE_METRICS_VERSION",
            "compute-metrics-v2",
        )
        assert report.report_hash != GOLDEN_REPORT_HASH

    def test_measured_cell_must_complete_every_repeat(self) -> None:
        """把只跑了 1/2 次的结果标成 MEASURED（并附上由那 1 次算出的 p50）必须被拒。"""
        report = self._report()
        truncated = (
            report.cells[0].model_copy(update={"completed_repeats": 1}),
            *report.cells[1:],
        )
        with pytest.raises(ValidationError, match="completed 1 of 2 repeats"):
            ComputeBudgetReport(
                matrix=report.matrix,
                budget=report.budget,
                repeats_per_cell=report.repeats_per_cell,
                cells=truncated,
            )

    def test_measured_cell_may_not_overshoot_the_wall_clock_budget(self) -> None:
        """跑满 repeats 但超预算的单元也不能自称 MEASURED（同一类手工报告残口）。"""
        report = self._report()
        overshot = (
            report.cells[0].model_copy(update={"wall_clock_seconds": 3600.0}),
            *report.cells[1:],
        )
        with pytest.raises(ValidationError, match=r"over the 100\.0s per-cell wall-clock budget"):
            ComputeBudgetReport(
                matrix=report.matrix,
                budget=report.budget,
                repeats_per_cell=report.repeats_per_cell,
                cells=overshot,
            )

    def test_report_rejects_matrix_above_the_cell_cap(self) -> None:
        """``model_construct`` 能绕过 PilotMatrix 的组数上限，报告级不变量必须再挡一次。"""
        bypassed = PilotMatrix.model_construct(
            device_class="cpu",
            sample_counts=(1, 2, 3, 4, 5),
            batch_sizes=(1, 2, 3),
            symbols=1,
            forecast_origins=1,
        )
        assert len(bypassed.cells) == 15 > MAX_PROBE_CELLS
        with pytest.raises(ValidationError, match="caps a probe at 10"):
            ComputeBudgetReport(
                matrix=bypassed,
                budget=ProbeBudget(wall_clock_seconds_per_cell=1.0, memory_bytes=1),
                repeats_per_cell=1,
                cells=(),
            )

    def test_throughput_memory_curve_is_width_ordered(self) -> None:
        curve = self._report().throughput_memory_curve
        assert [point.forward_width for point in curve] == [16, 32, 32, 64]
        for point in curve:
            assert point.throughput_per_second == pytest.approx(2.0)
            assert point.ram_bytes == 2048
            assert point.vram_bytes is None

    def test_throughput_memory_curve_is_empty_when_nothing_was_measured(self) -> None:
        report = run_cost_probe(
            matrix=MATRIX,
            budget=ProbeBudget(wall_clock_seconds_per_cell=0.1, memory_bytes=10**12),
            repeats_per_cell=1,
            measure=_measure(),
            resource_sampler=_ScriptedSampler([ResourceSample(ram_bytes=1)]),
            clock=_StepClock(0.5),
        )
        assert report.measured_cells == ()
        assert report.throughput_memory_curve == ()

    def test_report_indexes_and_totals(self) -> None:
        report = self._report()
        assert len(report.measured_cells) == 4
        assert report.infeasible_cells == ()
        # batch 1/2 两组 × 2 repeats：2 + 4 每次 cell，四组共 12
        assert report.total_forecasts == 12
        for cell in report.cells:
            assert cell.latency is not None
            assert cell.wall_clock_seconds >= cell.latency.count * 0.5
        assert report.total_wall_clock_seconds == pytest.approx(
            sum(cell.wall_clock_seconds for cell in report.cells)
        )
        assert report.total_wall_clock_seconds > 0.0

    def test_dropping_a_declared_cell_is_rejected(self) -> None:
        report = self._report()
        with pytest.raises(ValidationError, match="cover the declared pilot matrix"):
            ComputeBudgetReport(
                matrix=report.matrix,
                budget=report.budget,
                repeats_per_cell=report.repeats_per_cell,
                cells=report.cells[:-1],
            )

    def test_unknown_version_is_rejected(self) -> None:
        report = self._report()
        with pytest.raises(ValidationError, match="unknown compute metrics version"):
            ComputeBudgetReport(
                matrix=report.matrix,
                budget=report.budget,
                repeats_per_cell=report.repeats_per_cell,
                cells=report.cells,
                version="compute-metrics-v2",
            )

    def test_reported_version_is_the_versioned_constant(self) -> None:
        assert self._report().version == COMPUTE_METRICS_VERSION

    def test_measured_cell_requires_latency_and_throughput(self) -> None:
        with pytest.raises(ValidationError, match="must report latency and throughput"):
            ProbeCellResult(**_cell_result_payload(latency=None))  # type: ignore[arg-type]

    def test_measured_cell_rejects_abort_reason(self) -> None:
        with pytest.raises(ValidationError, match="must not carry an abort reason"):
            ProbeCellResult(**_cell_result_payload(abort_reason="oops"))  # type: ignore[arg-type]

    def test_infeasible_cell_requires_reason_and_hides_metrics(self) -> None:
        with pytest.raises(ValidationError, match="must carry an abort reason"):
            ProbeCellResult(
                **_cell_result_payload(  # type: ignore[arg-type]
                    status="INFEASIBLE_ON_THIS_DEVICE",
                    latency=None,
                    throughput_per_second=None,
                )
            )
        with pytest.raises(ValidationError, match="must not report metrics"):
            ProbeCellResult(
                **_cell_result_payload(  # type: ignore[arg-type]
                    status="INFEASIBLE_ON_THIS_DEVICE", abort_reason="wall-clock budget exceeded"
                )
            )
        infeasible = ProbeCellResult(
            **_cell_result_payload(  # type: ignore[arg-type]
                status="INFEASIBLE_ON_THIS_DEVICE",
                latency=None,
                throughput_per_second=None,
                artifact_bytes_per_forecast=None,
                cache_hit_ratio=None,
                abort_reason="wall-clock budget exceeded",
            )
        )
        assert infeasible.latency is None
        assert infeasible.completed_repeats == 1


class TestProjectFullRunSeconds:
    def _measured(self) -> ProbeCellResult:
        return run_cost_probe(
            matrix=MATRIX,
            budget=ProbeBudget(wall_clock_seconds_per_cell=100.0, memory_bytes=10**12),
            repeats_per_cell=2,
            measure=_measure(),
            resource_sampler=_ScriptedSampler([ResourceSample(ram_bytes=1)]),
            clock=_StepClock(0.5),
        ).cells[0]

    def test_projection_uses_measured_throughput(self) -> None:
        cell = self._measured()
        assert cell.throughput_per_second == pytest.approx(2.0)
        assert project_full_run_seconds(cell_result=cell, full_forecasts=1000) == pytest.approx(
            500.0
        )
        assert project_full_run_seconds(
            cell_result=cell, full_forecasts=1000, parallelism=4
        ) == pytest.approx(125.0)

    def test_infeasible_cell_cannot_be_projected_from(self) -> None:
        infeasible = ProbeCellResult(
            **_cell_result_payload(  # type: ignore[arg-type]
                status="INFEASIBLE_ON_THIS_DEVICE",
                latency=None,
                throughput_per_second=None,
                abort_reason="wall-clock budget exceeded",
            )
        )
        with pytest.raises(ConfigurationError, match="cannot project from cell"):
            project_full_run_seconds(cell_result=infeasible, full_forecasts=10)

    @pytest.mark.parametrize(("forecasts", "parallelism"), [(0, 1), (10, 0)])
    def test_non_positive_arguments_are_rejected(self, forecasts: int, parallelism: int) -> None:
        with pytest.raises(ConfigurationError):
            project_full_run_seconds(
                cell_result=self._measured(),
                full_forecasts=forecasts,
                parallelism=parallelism,
            )


class TestImportWeight:
    def test_importing_compute_metrics_does_not_load_torch_or_pandas(self) -> None:
        code = (
            "import sys; import kronos_ai.evaluation.compute_metrics as m; "
            "assert 'torch' not in sys.modules, 'torch imported'; "
            "assert 'pandas' not in sys.modules, 'pandas imported'; "
            "assert 'resource' not in sys.modules, 'resource imported eagerly'; "
            "print(m.MAX_PROBE_CELLS, m.COMPUTE_METRICS_VERSION)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == f"10 {COMPUTE_METRICS_VERSION}"

    def test_probe_runs_without_torch_or_pandas(self) -> None:
        """§48：cost probe 要能对 baseline（无 torch/pandas）跑通。"""
        code = """
import sys
from kronos_ai.evaluation.compute_metrics import (
    PilotMatrix, ProbeBudget, ResourceSample, run_cost_probe,
)

class Sampler:
    def sample(self):
        return ResourceSample(ram_bytes=1024)

clock_state = {"now": 0.0}

def clock():
    value = clock_state["now"]
    clock_state["now"] += 0.25
    return value

report = run_cost_probe(
    matrix=PilotMatrix(
        device_class="cpu", sample_counts=(16,), batch_sizes=(1,),
        symbols=2, forecast_origins=2,
    ),
    budget=ProbeBudget(wall_clock_seconds_per_cell=10.0, memory_bytes=10**9),
    repeats_per_cell=2,
    measure=lambda cell: __import__(
        "kronos_ai.evaluation.compute_metrics", fromlist=["ProbeObservation"]
    ).ProbeObservation(),
    resource_sampler=Sampler(),
    clock=clock,
)
assert 'torch' not in sys.modules
assert 'pandas' not in sys.modules
print(report.all_feasible, report.total_forecasts, report.report_hash[:8])
"""
        completed = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.startswith("True 2 ")
