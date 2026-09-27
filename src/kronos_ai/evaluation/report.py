"""Benchmark 报告与 artifact 落盘（基线文档 §33 / §48 / §49；ADR-023）。

§49 要求报告同时给出统计指标与计算成本；§33 要求一次 run 的产物是目录下一组命名文件
（``config.yaml`` / ``metadata.json`` / ``forecast.parquet`` / ``metrics.json`` /
``report.md``）。本模块负责把这批文件从内存态结果写出来，并保证两件事：

1. **报告不重新计算指标**：所有数字都来自 :class:`ForecastBenchmarkResult`（已由
   :func:`~kronos_ai.evaluation.forecast_metrics.evaluate_forecast_records` 算出）。
   如果 renderer 自己算一遍，报告与 metrics.json 就有了两条可能分叉的路径。
2. **缺失显示为缺失**：指标为 ``None`` 时报告写 ``n/a``（并附 ``labeled_count``），
   而不是 0。读报告的人必须能区分「没有证据」与「表现很差」。

落盘经 :class:`~kronos_ai.infrastructure.persistence.artifact_store.ArtifactStore`：写入
原子（临时文件 + fsync + rename）、内容 sha256 进 artifact index，与 §30 的 append-only
快照约定一致。
"""

from __future__ import annotations

from typing import Any

from kronos_ai.domain.hashing import sha256_hex
from kronos_ai.errors import ConfigurationError
from kronos_ai.evaluation.benchmark import ForecastBenchmarkResult, evaluated_origins_by_symbol
from kronos_ai.evaluation.compute_metrics import percentile
from kronos_ai.evaluation.forecast_metrics import ForecastEvalMetrics
from kronos_ai.evaluation.gate import GateVerdict, gate_payload, render_gate_markdown
from kronos_ai.infrastructure.persistence.artifact_store import (
    MEDIA_TYPE_YAML,
    ArtifactRecord,
    ArtifactStore,
)

BENCHMARK_REPORT_VERSION = "forecast-benchmark-report-v1"

#: §49 Compute 维中**本报告不产出**、也不得凭空补上的项。显式登记而不是让它悄悄缺席：
#: cache hit ratio 需要缓存层暴露命中计数（§15 尚未提供 run 级统计），RAM / VRAM peak 与
#: artifact bytes per origin 属 §16 Cost Probe 的测量口径（RX-KAI-018），一次 forecast
#: benchmark run 只如实记录自己观测到的延迟分位。
DEFERRED_BENCHMARK_COMPUTE_METRICS: tuple[str, ...] = (
    "cache_hit_ratio",
    "ram_peak",
    "vram_peak",
    "artifact_bytes_per_forecast_origin",
)

#: §49 Compute 维：报告里列出的延迟分位口径（最近秩法，见 compute_metrics.percentile）。
LATENCY_QUANTILES: tuple[float, ...] = (0.5, 0.95)


def _metric_cell(value: float | None, *, digits: int = 4) -> str:
    """``None`` → ``n/a``；这不是「0」，报告里必须一眼可辨。"""
    return "n/a" if value is None else f"{value:.{digits}f}"


def latency_summary(result: ForecastBenchmarkResult) -> dict[str, dict[str, float]]:
    """按 backend 汇总 §49 的 p50 / p95 延迟（毫秒）。

    只统计记录里实际观测到的调用；没有记录的 backend 直接缺席（不写 0）。
    """
    by_backend: dict[str, list[float]] = {}
    for record in result.records:
        by_backend.setdefault(record.backend, []).append(record.latency_ms)
    summary: dict[str, dict[str, float]] = {}
    for backend, samples in by_backend.items():
        summary[backend] = {
            f"p{int(q * 100)}_ms": percentile(samples, q) for q in LATENCY_QUANTILES
        }
        summary[backend]["forecasts"] = float(len(samples))
    return summary


def metrics_payload(result: ForecastBenchmarkResult) -> dict[str, Any]:
    """``metrics.json`` 的内容（§33 / §49）。

    ``report_hash`` 由结果内容派生：任何影响指标的输入变化都会换掉它，因此「同一份报告
    被复现」是可判定的，而不是靠人眼比对。
    """
    return {
        "kind": "forecast_benchmark_metrics",
        "report_version": BENCHMARK_REPORT_VERSION,
        "benchmark_version": result.version,
        "forecast_eval_metrics_version": result.metrics_version,
        "report_hash": result.report_hash,
        "dataset_hash": result.dataset_hash,
        "label_policy_version": result.label_policy_version,
        "regime_spec_hash": result.regime_spec_hash,
        "coverage_spec_hash": result.coverage_spec_hash,
        "lookback_bars": result.lookback_bars,
        "horizon_sessions": result.horizon_sessions,
        "data_coverage_end": result.data_coverage_end.isoformat(),
        "sampling": result.sampling.model_dump(mode="json"),
        "backends": list(result.backends),
        "considered_origins": result.considered_origins,
        "evaluated_origins": result.evaluated_origins,
        "evaluated_origins_by_symbol": evaluated_origins_by_symbol(result),
        "truncated": result.truncated,
        "metrics": [entry.model_dump(mode="json") for entry in result.metrics],
        "latency_ms": latency_summary(result),
        "deferred_compute_metrics": list(DEFERRED_BENCHMARK_COMPUTE_METRICS),
    }


def records_frame_payload(result: ForecastBenchmarkResult) -> list[dict[str, Any]]:
    """``forecast.parquet`` 的行（逐 ``segment × symbol × origin × backend``）。"""
    rows: list[dict[str, Any]] = []
    for record in result.records:
        rows.append(
            {
                "segment": record.segment,
                "symbol": record.symbol,
                "market_date": record.market_date.isoformat(),
                "backend": record.backend,
                "model_revision": record.model_revision,
                "artifact_id": record.artifact_id,
                "predicted_return": record.predicted_return,
                "median_return": record.median_return,
                "predicted_direction": record.predicted_direction,
                "coverage_lower": record.coverage_lower,
                "coverage_upper": record.coverage_upper,
                "label_status": record.label_status,
                "realized_return": record.realized_return,
                "realized_direction": record.realized_direction,
                "trend_regime": record.trend_regime,
                "volatility_regime": record.volatility_regime,
                "latency_ms": record.latency_ms,
            }
        )
    return rows


def _per_symbol_cell(result: ForecastBenchmarkResult) -> str:
    """每个 symbol 被评估的 origin 数（pilot 是否悄悄缩掉 universe 的证据）。"""
    counts = evaluated_origins_by_symbol(result)
    if not counts:
        return "none (no records)"
    return ", ".join(f"{symbol}: {count}" for symbol, count in counts.items())


def _metrics_table(entries: tuple[ForecastEvalMetrics, ...], title: str) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| backend | segment | group | n | labeled | MAE | RMSE | dir acc | ret corr | cov@90% |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for entry in entries:
        lines.append(
            "| {backend} | {segment} | {group} | {n} | {labeled} | {mae} | {rmse} | {acc} | "
            "{corr} | {cov} |".format(
                backend=entry.backend,
                # segment=None 是跨段汇总，不是「缺一段名」：写 "all" 让标题行可被识别，
                # 而不是留下一个含义不明的 "-"
                segment=entry.segment if entry.segment is not None else "all",
                group=entry.group.label,
                n=entry.sample_count,
                labeled=entry.labeled_count,
                mae=_metric_cell(entry.mae),
                rmse=_metric_cell(entry.rmse),
                acc=_metric_cell(entry.direction_accuracy),
                corr=_metric_cell(entry.return_correlation),
                cov=_metric_cell(entry.quantile_coverage),
            )
        )
    return lines


def render_markdown_report(
    result: ForecastBenchmarkResult,
    *,
    metadata: dict[str, Any] | None = None,
    gate: GateVerdict | None = None,
) -> str:
    """渲染 §48/§49 的人类可读报告（``report.md``）。

    ``gate`` 为 ``None`` 时**不写** Gate 段落：没做判决的 run 不留一个空标题让人以为
    「判决过、结论是空白」（§42 的判决要么有、要么明确缺席）。
    """
    meta = metadata or {}
    lines: list[str] = [
        "# Forecast Benchmark v1",
        "",
        f"- benchmark_version: `{result.version}`",
        f"- forecast_eval_metrics_version: `{result.metrics_version}`",
        f"- report_hash: `{result.report_hash}`",
        f"- dataset_hash: `{result.dataset_hash}`",
        f"- label_policy_version: `{result.label_policy_version}`",
        f"- regime_spec_hash: `{result.regime_spec_hash}`",
        f"- coverage_spec_hash: `{result.coverage_spec_hash}`",
        f"- run_id: `{meta.get('run_id', 'n/a')}`",
        f"- git_commit: `{meta.get('git_commit') or 'n/a'}`",
        f"- config_hash: `{meta.get('config_hash', 'n/a')}`",
        f"- adjustment: `{meta.get('adjustment', 'n/a')}`",
        "",
        "## Scope",
        "",
        f"- lookback_bars: {result.lookback_bars}",
        f"- horizon_sessions: {result.horizon_sessions}",
        f"- data_coverage_end: {result.data_coverage_end.isoformat()}",
        f"- backends: {', '.join(result.backends)}",
        f"- considered_origins: {result.considered_origins}",
        f"- evaluated_origins: {result.evaluated_origins}",
        f"- evaluated_origins_by_symbol: {_per_symbol_cell(result)}",
        f"- truncated (pilot subset): {result.truncated}",
        f"- sampling: `{result.sampling.model_dump(mode='json')}`",
        "",
        "> 指标为 `n/a` 表示该切片没有足够证据（`labeled_count` 见同行的 n/labeled 列），"
        "不是「指标为 0」。",
        "",
    ]
    overall = tuple(entry for entry in result.metrics if entry.group.axis == "overall")
    lines += _metrics_table(overall, "§49 Forecast metrics — overall")
    lines.append("")
    for axis in ("trend", "volatility"):
        entries = tuple(entry for entry in result.metrics if entry.group.axis == axis)
        if entries:
            lines += _metrics_table(entries, f"Robustness — {axis}")
            lines.append("")

    lines += [
        "## §49 Compute",
        "",
        "| backend | p50 (ms) | p95 (ms) | forecasts |",
        "|---|---|---|---|",
    ]
    for backend, values in sorted(latency_summary(result).items()):
        lines.append(
            f"| {backend} | {values['p50_ms']:.3f} | {values['p95_ms']:.3f} | "
            f"{int(values['forecasts'])} |"
        )
    lines += [
        "",
        "> Compute budget / hardware 结论由独立的 Compute Budget Report（§16，RX-KAI-018）"
        "给出；本表只记录本次 run 实际观测到的延迟。",
        "",
        f"> 本报告不产出的 §49 Compute 项：{', '.join(DEFERRED_BENCHMARK_COMPUTE_METRICS)}"
        "（口径见 `deferred_compute_metrics`，不是「测出来是 0」）。",
        "",
    ]
    if gate is not None:
        lines += render_gate_markdown(gate)
    return "\n".join(lines)


def write_forecast_benchmark_artifacts(
    *,
    store: ArtifactStore,
    run_id: str,
    result: ForecastBenchmarkResult,
    metadata: dict[str, Any],
    config_text: str,
    gate: GateVerdict | None = None,
    gate_criteria_text: str | None = None,
) -> tuple[ArtifactRecord, ...]:
    """把一次 benchmark run 的产物写入 §33 的 run 目录。

    写入顺序无关紧要（每个文件都原子落盘 + 单独登记），但内容必须与 ``metadata`` 里的
    hash 一致：``report_hash`` / ``dataset_hash`` / ``config_hash`` 都能在 artifact 里
    找到对应实体。

    ``gate`` / ``gate_criteria_text`` 必须**同时**给出或同时缺席（§42 的预注册：
    判决与它依据的判据原文住在一起，事后替换判据必然与 ``criteria_hash`` 对不上）。
    """
    import pandas as pd

    if (gate is None) != (gate_criteria_text is None):
        raise ConfigurationError(
            "gate verdict and gate criteria text must be archived together: a verdict without "
            "its pre-registered criteria cannot be re-checked (§42)"
        )
    written: list[ArtifactRecord] = []
    written.append(
        store.write_text(
            run_id, "report", render_markdown_report(result, metadata=metadata, gate=gate)
        )
    )
    written.append(store.write_json(run_id, "metrics", metrics_payload(result)))
    written.append(store.write_json(run_id, "metadata", metadata))
    written.append(
        store.write_yaml(
            run_id,
            "config",
            {"config_yaml_sha256": sha256_hex(config_text), "config_yaml": config_text},
        )
    )
    frame = pd.DataFrame(records_frame_payload(result))
    written.append(store.write_parquet(run_id, "forecast", frame))
    if gate is not None:
        assert gate_criteria_text is not None
        written.append(
            store.write_bytes(
                run_id,
                "gate_criteria",
                gate_criteria_text.encode("utf-8"),
                filename="gate_criteria.yaml",
                media_type=MEDIA_TYPE_YAML,
            )
        )
        written.append(store.write_json(run_id, "gate", gate_payload(gate)))
    return tuple(written)


__all__ = [
    "BENCHMARK_REPORT_VERSION",
    "DEFERRED_BENCHMARK_COMPUTE_METRICS",
    "LATENCY_QUANTILES",
    "latency_summary",
    "metrics_payload",
    "records_frame_payload",
    "render_markdown_report",
    "write_forecast_benchmark_artifacts",
]
