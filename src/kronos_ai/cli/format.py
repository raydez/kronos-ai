"""CLI 输出格式化与请求参数衍生（供 forecast / benchmark 命令复用）。

这些函数不触碰网络 / 模型 / 文件系统（除显式写文件），因此可单测。
"""

from __future__ import annotations

import json
from argparse import Namespace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, cast, get_args

from kronos_ai.domain.forecast import ForecastResult, SamplingConfig
from kronos_ai.domain.hashing import canonical_json
from kronos_ai.domain.time import KnowledgeCutoffPolicy, cutoff_policy_record


def sampling_from_args(args: Namespace) -> SamplingConfig:
    return SamplingConfig(
        seed=args.seed,
        sample_count=args.samples,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )


def cutoff_policy_from_args(args: Namespace) -> KnowledgeCutoffPolicy:
    """显式 cutoff 优于 policy 名；两者都不存在时由 argparse 默认值兜底。"""
    if getattr(args, "knowledge_cutoff", None) is not None:
        return "explicit"
    policy = args.cutoff_policy
    # choices 与 Literal 同源，避免 policy 名单在 argparse / format 两处漂移
    if policy not in get_args(KnowledgeCutoffPolicy):
        raise ValueError(f"unknown cutoff policy: {policy!r}")
    return cast(KnowledgeCutoffPolicy, policy)


def cutoff_summary(args: Namespace, cutoff: datetime) -> dict[str, Any]:
    """§5 Run Metadata 需要记录 cutoff policy 名 + 参数 + 版本，而非只记时刻。"""
    record = cutoff_policy_record(cutoff_policy_from_args(args))
    return {
        "policy": record.policy,
        "policy_version": record.policy_version,
        "parameters": record.parameters,
        "resolved": cutoff.isoformat(),
    }


def _positive_probability(result: ForecastResult) -> float | None:
    for entry in result.distribution.threshold_probabilities:
        if entry.metric == "horizon_return" and entry.operator == "gt" and entry.threshold == 0.0:
            return entry.probability
    return None


def format_forecast_summary(result: ForecastResult, *, cutoff: dict[str, Any]) -> str:
    distribution = result.distribution
    lines = [
        f"symbol            {result.symbol}",
        f"market_date       {result.market_date.isoformat()}",
        f"cutoff_policy     {cutoff['policy']}",
        f"cutoff_version    {cutoff['policy_version']}",
        f"cutoff_parameters {json.dumps(cutoff['parameters'], sort_keys=True)}",
        f"knowledge_cutoff  {result.knowledge_cutoff.isoformat()}",
        f"horizon           {distribution.horizon} sessions",
        f"samples           {distribution.sample_count}",
        f"origin_close      {distribution.origin_close:.6f}",
        f"expected_return   {distribution.expected_return:+.6f}",
        f"median_return     {distribution.median_return:+.6f}",
        f"dispersion        {distribution.forecast_dispersion:.6f}",
        f"expected_mdd      {distribution.expected_max_drawdown:+.6f}",
        f"path_volatility   {distribution.expected_path_volatility:.6f}",
    ]
    probability = _positive_probability(result)
    if probability is not None:
        lines.append(f"p(return > 0)     {probability:.6f}")
    lines += [
        f"artifact_id       {result.artifact_id}",
        f"input_data_hash   {result.input_data_hash}",
        f"model             {result.model.model_id}@{result.model.revision[:12]}",
        f"device/dtype      {result.model.device}/{result.model.dtype}",
    ]
    return "\n".join(lines)


def forecast_summary_payload(result: ForecastResult, *, cutoff: dict[str, Any]) -> dict[str, Any]:
    distribution = result.distribution
    return {
        "symbol": result.symbol,
        "market_date": result.market_date.isoformat(),
        "knowledge_cutoff": cutoff,
        "horizon": distribution.horizon,
        "sample_count": distribution.sample_count,
        "origin_close": distribution.origin_close,
        "expected_return": distribution.expected_return,
        "median_return": distribution.median_return,
        "forecast_dispersion": distribution.forecast_dispersion,
        "expected_max_drawdown": distribution.expected_max_drawdown,
        "expected_path_volatility": distribution.expected_path_volatility,
        "metric_definition_version": distribution.metric_definition_version,
        "distribution_spec_version": distribution.distribution_spec_version,
        "distribution_spec_hash": distribution.distribution_spec_hash,
        "artifact_id": result.artifact_id,
        "input_data_hash": result.input_data_hash,
        "model": result.model.model_dump(mode="json"),
        "sampling": result.sampling.model_dump(mode="json"),
    }


def write_forecast_result(result: ForecastResult, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(result.model_dump(mode="json")), encoding="utf-8")


def print_forecast(result: ForecastResult, *, args: Namespace, cutoff: datetime) -> None:
    summary = cutoff_summary(args, cutoff)
    if getattr(args, "json", False):
        print(json.dumps(forecast_summary_payload(result, cutoff=summary)))
    else:
        print(format_forecast_summary(result, cutoff=summary))


def calendar_span(market_date: date, horizon: int) -> tuple[date, date]:
    """为 ``next_sessions`` 装载足够覆盖的日历区间（含跨年缓冲）。

    跨年时 BaoStock 静默返回空 → 装载器显式失败（ADR-009），这里只扩展请求范围，
    不猜测 session。
    """
    return (
        market_date - timedelta(days=7),
        market_date + timedelta(days=max(30, horizon * 3 + 30)),
    )
