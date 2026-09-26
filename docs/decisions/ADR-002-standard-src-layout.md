# ADR-002: Standard src Layout

- 状态：Accepted
- 日期：2026-09-26
- 对应任务：RX-KAI-002
- 对应基线：`Kronos-AI_v2_Greenfield_Rewrite_设计与落地方案.md` §4、§36（ADR-002）

## 背景

v1 采用根目录 `backend/` 平铺结构，包边界、工具链配置与打包能力缺失，且存在
`PredictionService` / `ModelManager` / `KronosIntegration` God Object。

## 决策

v2 采用标准 src layout（PEP 621 + hatchling + uv）：

```text
pyproject.toml          # 唯一依赖入口（54.1），uv.lock 锁定可复现依赖矩阵
src/kronos_ai/          # 正式包，Phase 1 仅创建所需子集
tests/                  # unit / integration / regression / leakage / benchmark
configs/                # 实验配置（YAML，随 run 归档）
experiments/            # 研究脚本
artifacts/              # run 产物（gitignore）
docs/decisions/         # ADR
```

要点：

1. `requires-python = ">=3.11"`；开发环境锁定 Python 3.12（`.python-version`）。
2. 工具链：pytest、ruff、mypy（pydantic 插件）在 pyproject 内统一配置。
3. 终态目录（§4）中的 `state/ decision/ calibration/ policy/ evaluation/ api/`
   等 Phase 3+ 子包在其对应任务开工时创建，不预先建空目录。
4. v1 `backend/` 在 v2 稳定 merge 回 main 后整体移除（Migration Matrix §38），
   不维护 `backend_v2/` 双目录（§2 禁令）。
5. `.gitignore` 中遗留的全局 `test_*.py` / `*_test.py` 忽略规则已移除，
   `tests/` 为正式交付物必须入库。

## 后果

- 测试、lint、类型检查、打包、console script 都有单一标准入口。
- src layout 强制安装后导入（`uv sync` + editable），杜绝「从仓库根直接 import」
  的路径侥幸，包边界与真实发布形态一致。
