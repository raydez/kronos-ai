# Vendored upstream code provenance

本目录是上游 Kronos 模型代码的 vendored 副本（基线文档 §37：vendor 前必须实际核验
上游 LICENSE，并在 vendor 目录保留原始 license 声明）。规则见 ADR-020。

## 来源（2026-09-27 实际核验）

| 项 | 值 |
| --- | --- |
| 仓库 | https://github.com/shiyu-coder/Kronos |
| Commit | `67b630e67f6a18c9e9be918d9b4337c960db1e9a`（master，2026-04-13） |
| 上游文件 | `model/kronos.py`、`model/module.py`、`model/__init__.py`、`LICENSE` |
| 代码 License | MIT（Copyright (c) 2025 ShiYu），原文见本目录 `LICENSE` |
| 权重 | `NeoQuasar/Kronos-small` @ `901c26c1332695a2a8f243eb2f37243a37bea320`（MIT）<br>`NeoQuasar/Kronos-Tokenizer-base` @ `0e0117387f39004a9016484a186a908917e22426`（MIT） |

## 逐文件核验（sha256）

`tests/integration/test_vendor_provenance.py` 分两层核验：

- 离线（默认门禁）：重新计算本目录 sha256 并与下表比对，同时核对表内上游 sha 与在线
  核验常量一致。下表任何一格漂移都会被检测到——改文件必须同步改表。
- 在线（`-m integration`，需网络）：按上述 commit 从 GitHub 拉取上游文件逐字节比对。

| 文件 | 上游 sha256（@ 上述 commit） | 本目录 sha256 | 行尾 | 变换 |
| --- | --- | --- | --- | --- |
| `model/kronos.py` | `0a5f90282e2039c2de0771473419715c845def154896dbd0f5747837e6241032` | `6689c833f27c97368ffa011d9b5287fcd90bd2458cc3ac12b7ea486a46d87fd3` | LF | import 头部适配（见下） |
| `model/module.py` | `a07edbadc0e96804c8158c021bbc6063bb7cc43b34d7fc470d5c8ff2005a409f` | `a4df669998fa8115ac219b06687abaff2fac0fce9da762007dd142038949eee5` | 上游 CRLF → 本目录 LF | 仅行尾归一化：内容 sha256 与上游去 CR 后逐字节相同 |
| `model/__init__.py` | `f8f856ca3fedadcaac97e196be23d1aeda1c3c9ffe8903d66d43ea3bcac6240c` | `8ba4b713d2c9949f3e19bf153c1e6d4044b2588beefdcf3c08c9f83390fd69c4` | LF | **未采用**：本目录 `__init__.py` 是 v2 自写的 import shim（见下） |
| `LICENSE` | `acb2d194d378204e5f2be4dcd24d39ecac437903620c790c3315a96dab388fdc` | `acb2d194d378204e5f2be4dcd24d39ecac437903620c790c3315a96dab388fdc` | LF | 无（逐字节一致） |

### `kronos.py` 的 import 头部适配（机械改动，无语义变化）

```text
- import sys
-
  from tqdm import trange
-
- sys.path.append("../")
- from model.module import *
+ from kronos_ai.forecast.backends.kronos.vendor.module import *
```

除此 4 行外，`kronos.py` 与上游逐字节一致。

### `__init__.py`：目录内唯一的 v2 自写文件（无逻辑）

上游 `model/__init__.py` 是训练侧注册表（`model_dict` / `get_model_class`），不包含
v2 需要的 `auto_regressive_inference` / `sample_from_logits` / `calc_time_stamps` /
`top_k_top_p_filtering`。本目录的 `__init__.py` 因此由 v2 自写，只做一件事：
把 vendor 内的公开名字按绝对路径重新导出，给调用方一个稳定 import 面
（`from kronos_ai.forecast.backends.kronos.vendor import Kronos, ...`）。

它不含任何数值逻辑，也不改变上游行为；ADR-020 的「vendor 目录不放 v2 逻辑」约束
针对的是行为改动，本文件是该约束下唯一且显式声明的例外。

## 不再做的事（v2 纪律）

- 不在此目录内添加 v2 逻辑（raw sample 截取、per-run RNG、去 mean 等）：
  这些属于受控 adapter，位于同级的 `../sampler.py`（ADR-004 / ADR-005）。
- 不在本目录内修 bug 或改行为；上游升级 = 重新核验 license + 整目录替换 + 重跑
  采样等价性回归测试（`tests/regression/test_sampler_equivalence.py`）+ 更新本文件
  的 sha256 与 `runtime.py` 的 `VENDOR_UPSTREAM_COMMIT`（后者进入 runtime_version，
  漏更新会让旧 artifact 被误判为仍有效）。
