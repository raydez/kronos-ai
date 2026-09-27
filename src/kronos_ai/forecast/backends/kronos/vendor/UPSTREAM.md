# Vendored upstream code provenance

本目录是上游 Kronos 模型代码的 vendored 副本（基线文档 §37：vendor 前必须实际核验
上游 LICENSE，并在 vendor 目录保留原始 license 声明）。

## 来源（2026-09-27 实际核验）

| 项 | 值 |
| --- | --- |
| 仓库 | https://github.com/shiyu-coder/Kronos |
| 文件 | `model/kronos.py`、`model/module.py` |
| Commit | `67b630e67f6a18c9e9be918d9b4337c960db1e9a`（master，2026-04-13） |
| 代码 License | MIT（Copyright (c) 2025 ShiYu），原文见本目录 `LICENSE` |
| 权重 | `NeoQuasar/Kronos-small` @ `901c26c1332695a2a8f243eb2f37243a37bea320`（MIT）<br>`NeoQuasar/Kronos-Tokenizer-base` @ `0e0117387f39004a9016484a186a908917e22426`（MIT） |

## 对本副本做的唯一修改

`kronos.py` 头部 import 适配（机械改动，无语义变化）：

```text
- sys.path.append("../")
- from model.module import *
+ from kronos_ai.forecast.backends.kronos.vendor.module import *
```

`module.py` 与上游逐字节一致。

## 不再做的事（v2 纪律）

- 不在此目录内添加 v2 逻辑（raw sample 截取、per-run RNG、去 mean 等）：
  这些属于受控 adapter，位于 `..` 的 `sampler.py`（ADR-004 / ADR-005）。
- 不在本目录内修 bug 或改行为；上游升级 = 重新核验 license + 整目录替换 + 重跑
  采样等价性回归测试（`tests/regression/test_sampler_equivalence.py`）。
