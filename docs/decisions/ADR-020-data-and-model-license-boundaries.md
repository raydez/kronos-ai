# ADR-020: Data and Model License Boundaries

- 状态：Accepted
- 日期：2026-09-27
- 对应任务：RX-KAI-009
- 对应基线：`Kronos-AI_v2_Greenfield_Rewrite_设计与落地方案.md` §36（ADR-020）、§37、DoD 27

## 背景

v2 需要 vendor 上游 Kronos 模型代码（RX-KAI-009/010），并使用 HF 发布的模型权重；
同时平台消费 A 股行情数据（BaoStock）。§37 要求：「vendor 前必须实际核验上游仓库
LICENSE（不得凭印象），并在 vendor 目录保留原始 license 声明」「代码仓库默认不提交
原始行情数据」。本 ADR 固化三类边界的事实与规则。

## 决策

### 1. 代码 License（2026-09-27 实际核验）

| 对象 | 事实 | 核验方式 |
| --- | --- | --- |
| 上游仓库 `shiyu-coder/Kronos` | MIT，Copyright (c) 2025 ShiYu | GitHub API `/repos/shiyu-coder/Kronos/license` 返回 `spdx_id: MIT`，LICENSE 原文入库 |
| vendored 副本 | commit `67b630e67f6a18c9e9be918d9b4337c960db1e9a`（2026-04-13） | 见 `src/kronos_ai/forecast/backends/kronos/vendor/UPSTREAM.md` |

规则：

```text
vendor 目录 = upstream 逐字节副本，只允许两类机械偏差，且必须逐条登记在
UPSTREAM.md 并可用 tests/integration/test_vendor_provenance.py 在线复核：
  (a) import 路径适配（kronos.py 的 4 行头部替换）
  (b) 行尾归一化（module.py 上游 CRLF → 本目录 LF；去 CR 后逐字节相同）
另有一个 v2 自写文件 vendor/__init__.py（import shim，无任何数值逻辑），
是「vendor 不放 v2 逻辑」约束下唯一且显式声明的例外。
原始 LICENSE 与 UPSTREAM.md（来源/commit/日期/逐文件 sha256/改动清单）随代码入库
v2 自身逻辑（raw sample 截取、per-run RNG、去 mean）不写入 vendor 目录，
放在受控 adapter（sampler.py），升级上游 = 整目录替换 + 重跑等价性回归
```

核验方式（不靠记忆）：

```text
UPSTREAM.md 给出每个文件的上游 sha256 与本目录 sha256
tests/integration/test_vendor_provenance.py 重新计算本目录 sha256 并与表格比对
（离线，默认门禁运行；vendor 文件或表格被改动都会失败）
同文件按 pinned commit 在线比对上游内容
（集成测试，`-m integration` 手动/CI 触发；上游漂移会被检出）
```

### 2. 模型权重 License

```text
NeoQuasar/Kronos-small            @ 901c26c1332695a2a8f243eb2f37243a37bea320  MIT
NeoQuasar/Kronos-Tokenizer-base   @ 0e0117387f39004a9016484a186a908917e22426  MIT
```

（HF API `cardData.license` 核验，2026-09-27；revision 以 commit hash 固定，见
`runtime.py` 的默认值。）

权重不随代码仓库分发：运行时按 revision 从 HF 拉取或指向本地目录；权重文件、
HF cache 不进入 git。权重 License 与代码 License 分开记录（§37）。

### 3. 行情数据与 Artifact 再分发

```text
默认策略：代码仓库不提交原始行情数据（raw provider response 快照仅本地 append-only）
Benchmark Report 可发布：aggregated metrics / methodology / config
原始 Market Artifact 是否可公开：必须依据数据源条款单独判断，不得由 SDK 开源推断
```

BaoStock 使用条款的再分发结论随 Phase 2 数据处理任务（RX-KAI-017 数据落盘）复核，
若条款不允许再分发，则 artifacts 保持本地、报告只含聚合指标。

## 修订记录

- 2026-09-27（RX-KAI-009 施工期核验）：原文写作「逐字节副本（仅允许 import 路径适配）」，
  实际核验发现两处未登记的偏差——`module.py` 上游为 CRLF、本目录为 LF；`__init__.py`
  是 v2 自写 import shim（上游同名文件是训练侧注册表）。二者均无语义差异，已按上述
  规则登记到 `UPSTREAM.md` 并补上可执行的在线核验测试。原文措辞与实际不一致本身即
  本次修订的动因：provenance 声明必须与仓库实态一致，否则审计结论不可信。

## 后果

- vendor 目录的 license 义务（保留版权与许可声明）显式满足，且可审计。
- 上游代码升级路径明确：替换 vendor 目录 + 重跑 `tests/regression/test_sampler_equivalence.py`。
- 数据/权重/代码三类边界各自记录，DoD 27 可在不依赖记忆的情况下复核。
