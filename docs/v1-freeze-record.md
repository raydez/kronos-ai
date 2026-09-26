# v1 Freeze Record（RX-KAI-001）

> 冻结对象：Kronos-AI v1（tag `v1.0.0`，commit `a879ba0`）
> 冻结日期：2026-09-26
> 性质：本记录随 v2 分支进入仓库，固化 v1 终态事实，供 v2 施工参考。
> v1 此后仅允许维护：critical bugfix / security / documentation / reproducibility，不再新增核心能力。

## 1. 运行环境

- Python：3.11（Docker 基础镜像 `python:3.11-slim`）
- 运行方式：uvicorn（`backend/main.py`），容器暴露 8000，健康检查 `GET /health`
- 数据目录：`data/models`、`data/cache`、`data/logs`、`data/database`（Dockerfile 内创建）

关键依赖（`backend/requirements.txt`）：

| 包 | 版本约束 |
|---|---|
| fastapi | ==0.104.1 |
| uvicorn[standard] | ==0.24.0 |
| pandas | ==2.1.3 |
| numpy | >=1.26.0 |
| baostock | ==0.8.9 |
| pydantic | ==2.5.0 |
| torch | >=2.0.0 |
| transformers | >=4.30.0 |
| huggingface-hub | >=0.17.0 |
| einops | >=0.7.0 |
| tqdm | >=4.66.0 |

已知缺陷：torch / transformers / huggingface-hub 等核心依赖为范围版本，v1 环境无法精确复现依赖矩阵；v2 通过 lockfile（uv）解决。

## 2. 模型 ID（HuggingFace）

默认加载：`kronos-small`（`model_manager.py` L42）。

| 配置名 | model_id | tokenizer_id | context_length | params |
|---|---|---|---|---|
| kronos-mini | NeoQuasar/Kronos-mini | NeoQuasar/Kronos-Tokenizer-2k | 2048 | 4.1M |
| kronos-small | NeoQuasar/Kronos-small | NeoQuasar/Kronos-Tokenizer-base | 512 | 24.7M |

加载方式：HuggingFace `from_pretrained`，缓存位于 `~/.cache/huggingface/hub`。
已知缺陷：未固定模型 revision，上游权重更新不可追溯；配置注释仍残留「需要替换为实际的模型ID」TODO。
设备：v1 在 `kronos_integration.py` 顶部 monkey-patch 强制 CPU（隐藏 CUDA），无 GPU 路径。

## 3. 当前启动方式

- 本地：`start.sh` / `quick-start.sh`
- Docker：`docker-compose.yml`（backend + frontend），`docker-start.sh` / `docker-stop.sh`

## 4. 已知无效指标（禁止带入 v2，禁止作为 baseline 引用）

| 无效指标 | 位置 | 机理 |
|---|---|---|
| 准确率 ~75%/85%/90% | `README.md` L242 | 无 Benchmark 支撑的宣传数字 |
| `accuracy: 0.85`、`processing_time: "< 2s"` | `kronos_integration.py` L504-506 | `get_model_info()` 硬编码 |
| confidence = `max(0.6, 0.9 - i*0.05)` | `kronos_integration.py` L368、L466 | 按预测序号线性递减的人工公式，与模型输出无关 |

## 5. 已知 invalidating 机制（v2 明确废除，见基线文档 §38 Migration Matrix）

| 机制 | 位置 | 问题 |
|---|---|---|
| `sample_count=1` 单样本预测 | `kronos_integration.py` L353 | 无分布信息，无法支撑 Probabilistic Forecast |
| `freq='B'` 伪交易日历 | `kronos_integration.py` L330、L338 | 普通工作日 ≠ A 股交易日，节假日预测时间轴失真 |
| `_generate_fallback_data` | `stock_service.py` L256-303（5 处调用点） | 数据源失败静默生成随机 K 线 |
| 预测解析失败随机生成 OHLC | `kronos_integration.py` L459-474 | 同上 |
| 同步推理阻塞异步链 | `kronos_integration.py` / `prediction_service.py` | 模型推理直接运行在 async 调用链中 |
| 人工 confidence 与 ModelManager Singleton | `kronos_integration.py` / `model_manager.py` | 见基线文档 §1 不沿用清单 |

## 6. 可迁移资产

- `backend/model/kronos.py`、`backend/model/module.py`：upstream-derived（Kronos 官方实现，MIT），校验后作为 v2 runtime source
- BaoStock 调用逻辑：参考重写，不迁移任何 fallback 路径
- frontend 图表组件：选择性迁移

---

对应基线文档：`Kronos-AI_v2_Greenfield_Rewrite_设计与落地方案.md`
对应任务：RX-KAI-001（基线文档 §39 Phase 0）
