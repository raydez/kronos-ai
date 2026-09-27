"""Evaluation 层（基线文档 §4 / §27 / §41）。

按终态目录约定，本包承载与「评估」相关的能力，与推理（``forecast``）、数据（``data``）
分离：

- :mod:`kronos_ai.evaluation.baselines`：naive baselines（§19），已落地（RX-KAI-016）
- :mod:`kronos_ai.evaluation.walk_forward`：分段与 embargo 空置（§27、§29），已落地（RX-KAI-017）
- :mod:`kronos_ai.evaluation.dataset`：LabelPolicy / walk-forward dataset / label 构造
  （§27、§28，已落地 RX-KAI-017）
- ``forecast_metrics.py`` / ``compute_metrics.py``：指标与聚合（§13、§49，RX-KAI-019）
- ``report.py``：benchmark 报告（§48、§49）

依赖方向是单向的：``evaluation`` → ``forecast`` / ``data`` / ``domain``。因此 baseline
可以复用推理层的接口、样本转换与缓存键，而推理核心不依赖任何评估代码。
"""

from __future__ import annotations
