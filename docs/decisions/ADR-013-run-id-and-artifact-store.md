# ADR-013: Run ID 与 Artifact Store

- 状态：Accepted
- 日期：2026-09-26
- 对应任务：RX-KAI-015
- 对应基线：`Kronos-AI_v2_Greenfield_Rewrite_设计与落地方案.md` §30、§31、§32、§33

## 背景

v1 把研究产物散落成 `artifacts/runs/<日期>-<序号>/metadata.json`：run id 依赖本地时钟
与目录扫描（不全局唯一、并行不安全），metadata 只写一个 JSON 文件，没有索引，无法回答
「哪个 run 用了哪个 dataset_version」「哪些 run 失败了」「同一 config 是否已经跑过」。
§30 要求 run registry / artifact index / model / dataset / job metadata 落 SQLite、
表格产物落 Parquet，§31 要求 run id 全局唯一且可排序，§32 要求完整 Run Metadata，
§33 要求固定的目录布局。raw provider response 还必须 append-only（BaoStock 复权因子
有追溯修订，快照是 point-in-time 可复现性的最后一环）。

## 决策

### 1. run id = ULID（§31）

- 自实现 48-bit 毫秒 + 80-bit 随机数的 ULID（`domain/ids.py`），Crockford Base32 编码
  26 字符，不引入第三方依赖。
- `UlidGenerator` 把「同进程同毫秒严格递增」变成契约：同一毫秒内自增 80-bit 随机段而非
  重掷随机数，保证 `sortable`；耗尽随机段显式抛错，绝不静默回绕。跨进程唯一性由
  80-bit 随机空间保证，因此 `parallel-safe`。
- 时钟回拨时沿用上一毫秒并继续自增，不产生「回退」的 id。

### 2. 持久化底座 = 单个 SQLite 文件 + WAL（§30）

- `infrastructure/persistence/sqlite.py` 是唯一连接点：`PRAGMA journal_mode=WAL`
  （文件库未进入 WAL 时显式失败，§30 明确要求 WAL）、`busy_timeout`、
  `foreign_keys=ON`，所有语句在一把可重入锁下执行——把「写入串行化」变成契约而不是
  让调用方到处重试 `database is locked`。
- schema 版本写入 `schema_meta` 表；打开版本不符的库时显式失败（§3.2），不迁移、不猜测。
- 冷启动并发（多个 benchmark worker 同时打开一个全新库）时 `PRAGMA journal_mode=WAL`
  与建表都不受 `busy_timeout` 保护，且「先查 schema_meta 后插入」是 TOCTOU：因此
  初始化阶段对 `database is locked` 做有界重试，版本行用
  `INSERT ... ON CONFLICT(key) DO NOTHING` 后再回读，仍失败则显式抛 `ArtifactError`
  （不泄漏原始 `sqlite3` 异常）。
- `bind_contract(name, version)` 把各组件契约版本登记进 `schema_meta`：契约常量不参与
  校验就等于没有治理，因此 `RunRegistry` / `ArtifactStore` 构造时落库，版本不一致即
  显式失败。
- §30 还提到 model / dataset / job metadata 三张表；本任务只建 `runs` / `artifacts` /
  `snapshots`，model 目录由 RX-KAI-016+ 引入、dataset / job 由 RX-KAI-017 / RX-KAI-029
  引入，届时沿用同一 `schema.py` 与 `PERSISTENCE_SCHEMA_VERSION` 升级路径。
- `transaction()` 用 `BEGIN IMMEDIATE`，**禁止嵌套**（显式报错而非静默开子事务）。
- run registry 与 artifact index **共用同一个 DB 文件**（§30 把两者都归入 SQLite），
  `schema.py::open_database` 是唯一装配点，避免两处建表后版本漂移。

### 3. 状态机单一真源（§30–§32）

`domain/run.py` 的 `RunStatus` / `RUN_STATUS_TRANSITIONS` 是 status 的唯一真源：
DDL 的 `CHECK` 约束、`RunRegistry.update_status` 的迁移校验都从它派生。终态
（`succeeded` / `failed` / `cancelled`）没有出边；同状态重复写入幂等。

### 4. 幂等提交用 dedup_key（§35 前置）

`runs.dedup_key` 上的唯一 partial index 让 `RunRegistry.submit` 具备幂等语义：同一
`dedup_key` 重复提交返回既有 run 而不是新建 job（RX-KAI-029 的异步 API 依赖它）。

### 5. Artifact Store：write-once + index（§30/§33）

- 目录布局固定为 §33 的 `artifacts/runs/<ULID>/`；`run_dir_relative` 是 registry 与
  store 共用的路径真源。
- 每个 run artifact 落盘后登记进 `artifacts` 表：相对路径、media type、sha256、字节数、
  行数、schema hash。**run artifact 是 write-once**：同名重复写入显式失败，绝不覆盖。
- 表格产物（universe / forecast / samples / states / decisions）写 Parquet，
  行数与 `schema_hash`（列名 + dtype 顺序的指纹）一并登记。
- 落盘用 `atomic_io.atomic_create_*`（`os.link` 独占创建）而非 rename 覆盖：同名并发
  写入时至多一个 writer 能建立文件，败者拿到 `False`/`None` 并转成显式 write-once
  错误，**不会删除并发胜者刚写的文件**。索引冲突（`sqlite3.IntegrityError`）时按名
  回查一次，命中则报 write-once，不盲目 unlink。
- 落盘前先校验 `metadata` / `media_type`，并保证「文件由本 writer 独占创建」是
  `_index_artifact` 的前提：因此索引失败时可以安全删除自己的文件，既不留孤儿也不
  误删他人内容。
- **崩溃恢复差距（有意为之）**：若在「文件已落盘、索引未写入」之间崩溃，run artifact
  会拒绝覆盖该孤儿文件（显式失败，符合 §3.2），需人工清理；快照则在内容摘要一致时
  收养。不为 Parquet 做同内容收养，是因为其字节（压缩 / 元数据）不保证可重现。
- `ArtifactRecord.sha256` 是**存储完整性**校验（文件字节摘要），不是逻辑内容 hash：
  Parquet 字节含压缩与元数据，跨写入不逐字节稳定；逻辑内容 hash（如 §32 的
  `dataset_hash`）由领域语义单独定义。
- 所有写入复用 `atomic_io` 的「临时文件 + fsync + 父目录 fsync」：读者永远看不到
  半成品文件——§15 缓存与 §30 store 共用同一套保证，不维护两份实现（覆盖语义走
  `os.replace`，write-once 语义走 `os.link` 独占创建）。
- 读取时重新计算 sha256 与 index 比对，磁盘被篡改即显式失败，不做「尽力而为」。

### 6. raw provider response = append-only 快照（§30）

- `write_snapshot(dataset_version, name, data)` 首次拉取即落盘到
  `artifacts/snapshots/<dataset_version>/`；内容相同幂等返回，**内容不同显式失败**
  （永不覆盖）。
- 幂等分支会校验索引对应的文件仍在（缺失即 store 损坏）；磁盘上存在未索引的同名
  孤儿文件（崩溃遗留）时，内容一致则收养、不一致则拒绝，而不是永久毒化该名字。
- 新数据产生新 `dataset_version` + 新快照，旧快照保留；`.gitignore` 忽略 `artifacts/*`。

### 7. CLI 走 registry（§34）

`run show <run_id>` 读 `RunRegistry` + `ArtifactStore`（不再直接读
`metadata.json` 文件），新增 `run list`（按 kind/status/limit 过滤）；`--artifacts-dir`
替代 `--runs-dir`，index DB 默认 `<artifacts-dir>/index.sqlite3`
（`KRONOS_AI_ARTIFACTS_DIR` 可覆盖 root）。`run show` / `run list` 是只读命令：index DB
不存在时显式失败，而不是顺手建出一个空 store。

`run show --json` 的载荷由 v1 的「扁平 metadata」改为 `{run, artifacts, metadata}`
（对齐 registry + artifact index 两个真源）；这是对 CLI 输出的破坏性变更，依赖旧格式的
脚本需同步更新。

### 8. 本期不做的（显式 defer）

- §30 的 SQLite 清单还提到 model / dataset / job metadata：本任务只建 `runs` / `artifacts`
  / `snapshots` 三张表。job 由 RX-KAI-029、dataset 由 RX-KAI-017、model 身份走 §32 run
  metadata；届时沿用同一 `schema.py` 并升 `PERSISTENCE_SCHEMA_VERSION`。
- §32 的类型化 Run Metadata（`git_commit` / `market_date_range` / `sampling` / `runtime` …）
  本期存为已校验的 JSON mapping（`RunRecord.metadata`），不建 pydantic 模型：在第一个
  真正填充这些字段的 producer（RX-KAI-016 / RX-KAI-019）出现前定义 schema 属投机泛化。
  registry 本期只需保证「可序列化 + 可查询」。

## 后果

- run 具备全局唯一、可排序、并行安全的身份，`artifacts/` 可被交叉对比与回放。
- 「哪次运行、用了什么配置与数据集、产出哪些文件」变成一次 SQL 查询；RX-KAI-019 的
  benchmark 与 §48 的 Go/Replace gate 可以直接消费 registry。
- 快照保证 point-in-time 可复现：即使上游修订历史数据，旧 run 仍能解释自己的输入。
- 迁移成本固定：`PERSISTENCE_SCHEMA_VERSION` / `ARTIFACT_STORE_CONTRACT_VERSION` /
  `RUN_REGISTRY_CONTRACT_VERSION` / `RUN_STATUS_TRANSITIONS` 任一变更都需升级版本常量
  并同步 golden 测试。
