"""Evaluation 层（基线文档 §4 / §27 / §41）。

按终态目录约定，本包承载与「评估」相关的能力，与推理（``forecast``）、数据（``data``）
分离：

- :mod:`kronos_ai.evaluation.baselines`：naive baselines（§19），已落地（RX-KAI-016）
- :mod:`kronos_ai.evaluation.walk_forward`：分段与 embargo 空置（§27、§29），已落地（RX-KAI-017）
- :mod:`kronos_ai.evaluation.dataset`：LabelPolicy / walk-forward dataset / label 构造
  （§27、§28，已落地 RX-KAI-017）
- :mod:`kronos_ai.evaluation.compute_metrics`：compute cost probe 与 compute budget
  report（§16、§49，已落地 RX-KAI-018）
- :mod:`kronos_ai.evaluation.forecast_metrics`：§49 forecast 指标口径（已落地 RX-KAI-019）
- :mod:`kronos_ai.evaluation.regimes`：point-in-time regime 分组（§49，已落地 RX-KAI-019）
- :mod:`kronos_ai.evaluation.benchmark`：Forecast Benchmark v1 的 run 编排（§41，已落地
  RX-KAI-019）；契约见 ADR-023
- :mod:`kronos_ai.evaluation.report`：§33 artifact 与人类可读报告（已落地 RX-KAI-019）
- :mod:`kronos_ai.evaluation.gate`：Forecast Backend Go / Replace Gate（§19、§42，已落地
  RX-KAI-020）；判据预注册、配对 bootstrap 区间与 GO / CONDITIONAL / REPLACE 判决，
  契约见 ADR-024
- ``decision_metrics.py`` / ``calibration_metrics.py``：决策层指标（§50，后续任务）

依赖方向是单向的：``evaluation`` → ``forecast`` / ``data`` / ``domain``。因此 baseline
可以复用推理层的接口、样本转换与缓存键，而推理核心不依赖任何评估代码。
"""

from __future__ import annotations
