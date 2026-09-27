"""A 股复权策略契约（RX-KAI-006，基线文档 §6.1 / ADR-007）。

复权不是一个开关，四个概念必须分开（§6.1）：

```text
Raw OHLC                    ← 原始价格，PIT 稳定，规范存储
Corporate Actions           ← 公司行动明细（本任务不采集）
Adjustment Factor           ← 因子元数据，须 append-only 快照（§30）
Point-in-Time Adjusted Series ← 由前两者派生的视图，随快照可复现
```

数据源事实（`docs/spike/baostock-capability.md`）：

- 因子表随查询时刻归一化（最新记录 foreAdjustFactor 恒为 1.0），因此 provider 直接
  返回的复权序列（adjustflag=1/2）不是 PIT 可复现的，禁止进入 artifact；
- 复权序列完全由「原始价格 + 因子表」决定，映射规则已系统验证：
  ``hfq(D) = raw(D) * back_factor(latest ex_date <= D)``、
  ``qfq(D) = raw(D) * fore_factor(latest ex_date <= D)``；
- 退市标的的因子表可能缺失（静默空），任何调整都必须显式失败而非按 1.0 兜底。

本模块只固化策略契约、版本号与快照元数据校验；价格换算与快照落盘属数据集构建
（RX-KAI-015 及后续），此处不实现，避免在未经 evidence 的地方引入算术。
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from kronos_ai.domain.symbols import validate_normalized_symbol

AdjustmentMode = Literal["raw", "hfq", "qfq"]

ADJUSTMENT_POLICY_VERSION = "adjustment-policy-v1"

DEFAULT_FORECAST_ADJUSTMENT_MODE: AdjustmentMode = "raw"

BAOSTOCK_FACTOR_SOURCE = "baostock:query_adjust_factor"


class AdjustmentPolicy(BaseModel):
    """版本化复权策略：mode + 因子来源 provenance（§32 run metadata）。"""

    model_config = ConfigDict(frozen=True)

    mode: AdjustmentMode
    factor_source: str | None = None
    policy_version: str = ADJUSTMENT_POLICY_VERSION

    @field_validator("policy_version")
    @classmethod
    def _version_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("policy_version must be non-empty")
        return value

    @model_validator(mode="after")
    def _factor_source_matches_mode(self) -> AdjustmentPolicy:
        if self.mode == "raw":
            if self.factor_source is not None:
                raise ValueError("raw mode must not declare a factor_source")
            return self
        if self.factor_source is None or not self.factor_source.strip():
            raise ValueError(
                f"{self.mode} mode requires a factor_source naming the snapshot provenance"
            )
        return self

    @property
    def requires_factor_snapshot(self) -> bool:
        """复权模式必须绑定因子表快照；raw 模式不依赖因子表。"""
        return self.mode != "raw"

    def run_metadata(self) -> dict[str, Any]:
        """§32 Run Metadata 的 adjustment 片段：版本 + 参数，缺一不可复现。"""
        return {
            "adjustment_policy_version": self.policy_version,
            "parameters": {"mode": self.mode, "factor_source": self.factor_source},
        }


def default_forecast_adjustment_policy() -> AdjustmentPolicy:
    """Forecast 默认输入策略：raw（ADR-007）。

    选择 raw 的理由：PIT 稳定、不依赖随查询时刻重算的因子表、不依赖尚未核实的
    盘后发布时刻语义。
    """
    return AdjustmentPolicy(mode=DEFAULT_FORECAST_ADJUSTMENT_MODE)


class AdjustFactorRecord(BaseModel):
    """单次公司行动的因子记录（``query_adjust_factor`` 一行的语义映射）。"""

    model_config = ConfigDict(frozen=True)

    ex_date: date
    fore_factor: float
    back_factor: float

    @field_validator("fore_factor", "back_factor")
    @classmethod
    def _positive_finite(cls, value: float, info: Any) -> float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{info.field_name} must be finite")
        if value <= 0:
            raise ValueError(f"{info.field_name} must be > 0")
        return value


class AdjustFactorSeries(BaseModel):
    """因子表快照元数据：append-only 留痕后的只读视图（§6.1 / §30）。

    快照一旦落盘即不可变；重新拉取产生新 ``dataset_version`` 与新快照，旧快照保留。
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    records: tuple[AdjustFactorRecord, ...]
    source: str
    dataset_version: str

    @field_validator("symbol")
    @classmethod
    def _symbol_normalized(cls, value: str) -> str:
        return validate_normalized_symbol(value)

    @field_validator("source", "dataset_version")
    @classmethod
    def _nonempty(cls, value: str, info: Any) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must be non-empty")
        return value

    @model_validator(mode="after")
    def _records_sorted_unique(self) -> AdjustFactorSeries:
        if not self.records:
            raise ValueError("records must not be empty")
        ex_dates = [record.ex_date for record in self.records]
        if ex_dates != sorted(ex_dates):
            raise ValueError("records must be sorted ascending by ex_date")
        if len(set(ex_dates)) != len(ex_dates):
            raise ValueError("records must have unique ex_dates")
        return self

    @property
    def coverage(self) -> tuple[date, date]:
        return (self.records[0].ex_date, self.records[-1].ex_date)

    def factors_asof(self, day: date) -> AdjustFactorRecord | None:
        """``day`` 当日生效的因子：最近一个 ``ex_date <= day``；早于首个行动则为 None。

        返回 None 表示「该日之前没有任何已记录的公司行动」，调用方不得据此推断
        因子为 1.0（退市标的因子表缺失是静默的，见模块 docstring）。
        """
        applicable: AdjustFactorRecord | None = None
        for record in self.records:
            if record.ex_date > day:
                break
            applicable = record
        return applicable
