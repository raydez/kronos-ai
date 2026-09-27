# ADR-011: Forecast Artifact Cache

- 状态：Accepted
- 日期：2026-09-27
- 对应任务：RX-KAI-013
- 对应基线：`Kronos-AI_v2_Greenfield_Rewrite_设计与落地方案.md` §8、§14、§15、§30、§36（ADR-011）

## 背景

Kronos 推理是整个平台最贵的操作（§16：成本随 `sample_count` 线性放大）。Benchmark
（§19/§41/§48）与 walk-forward 评估（§27）会对同一 `(symbol, market_date, cutoff,
horizon, model, sampling)` 组合反复求值；没有缓存，全量 Benchmark 会重复支付同一份
推理成本，且「两次跑同一实验」不再可复现（第二次可能用了不同随机路径）。

v1 没有 forecast 级缓存，最接近的是一次性脚本产物。§15 因此把 Forecast 定义为
**昂贵 Artifact**：同一个 `ForecastArtifactKey` 必须命中同一个产物，不得重复推理，
除非调用方显式 `--force`。

难点不在「存一个文件」，而在：

```text
1. key 必须穷尽所有影响输出的身份（漏一个字段 = 静默复用错误产物）
2. 并发 benchmark worker 会同时 miss / 同时写同一 key（§16 的 pilot matrix）
3. 缓存产物必须能被 provenance 审计（DoD12：ForecastResult 记录完整 metadata）
```

## 决策

### 1. `ForecastArtifactKey` 是契约，`digest` 即 `artifact_id`

`ForecastArtifactKey`（`forecast/cache.py`）折叠 §15 的身份维，并补上 §15 字面清单
遗漏但客观影响输出的两维：

```text
symbol, input_data_hash, market_date, knowledge_cutoff, horizon,
future_sessions, origin_close, distribution_spec_hash,
model_id, model_revision, runtime_version, device_class, dtype, config_hash,
seed, sample_count, temperature, top_k, top_p
```

- `input_data_hash` 取 `MarketHistory.data_hash`（读取时实时派生，见 ADR-006/008），
  因此任何输入内容变化都会改变 key。
- `config_hash` 来自 `KronosRuntime.artifact_identity()`，覆盖 `lookback_bars` /
  `max_context` / `clip` / tokenizer revision——`lookback_bars` 决定输入窗口与逐窗口
  归一化统计量（§8），不进 key 会让「改窗口后仍复用旧 forecast」。
- `future_sessions`（§6.6 `TradingCalendar.next_sessions` 的结果）是未来时间轴：
  它既作为模型 stamp 参与推理（日历变 → logits 变），又是 artifact 中
  `ForecastPoint.timestamp` 的取值。两份日历对同一 `(symbol, market_date, horizon)`
  可给出不同 session 序列，因此时间轴不同 = 输出不同，必须进 key。派生必须用
  **与 sampler 同一**的 `calendar.next_sessions`，否则 key 与推理实际时间轴脱钩。
- `origin_close`（P_0，forecast origin close）是全部收益 / 回撤 / 波动指标的锚点，
  因此逐属身份。它由 `history` 唯一确定（lookback 窗口末根 bar 的 close），
  在 `build_forecast_artifact_key` 内从 history 派生而非接收调用方入参——否则同一份
  history 可派生出两个「同 digest、不同 P_0」的 artifact 且都能通过 `_verify`。
- `distribution_spec_hash` 折叠 `DistributionSpec`（阈值/分位，含其 `version`）
  + `FORECAST_METRIC_REGISTRY_VERSION` + `FORECAST_AGGREGATION_DEFINITION_VERSION`。
  同一批 raw samples 在不同 spec 或不同指标/聚合口径下会产出不同
  `ForecastDistribution` 字段；不进 key 会让「改阈值/改聚合后仍返回旧分布」。
- 模型维必须**整体**取自 `artifact_identity()`：缺任一键显式失败
  （`ConfigurationError`），runtime docstring 明确禁止缓存实现挑子集。非字符串值也
  显式失败——否则 `str(None)` 会静默造出一个合法但错误的 identity。
- `digest = sha256(canonical_json(hashing_payload))`，payload 内含
  `FORECAST_ARTIFACT_KEY_VERSION` 与 `FORECAST_CONTRACT_VERSION`：升级契约会自动
  让既有 artifact 全部失效，而不是靠人工清理。
- `digest` 是 `ForecastResult.artifact_id` 的唯一真源（§14）；`artifact_id` 不再是
  自由字符串。

字段构成 / payload / canonical 编码任一变化都会让 `tests/unit/test_forecast_cache.py`
的 golden digest 变红。

### 2. 本地实现：`<digest 前两位>/<digest>.json`

`FileSystemForecastCache(root)` 把 `ForecastResult` 以 canonical json 落盘，
两层分片避免单目录膨胀。相同 key 的重复写入字节完全一致，因此 §15 的
`last-write-wins` 是无害的。

写入路径 = 「临时文件 + `fsync` + 原子 `os.replace`」：读者要么看到旧完整文件、
要么看到新完整文件，不会看到半截 json。临时文件由 `tempfile.mkstemp` 创建
（`O_EXCL` + 内核保证的随机名），因此同进程多线程 / 多进程并发写同一 key 不会
互相踩到对方的临时文件（若靠 `pid + 32-bit 随机数` 命名，碰撞时一个 writer 的清理
会删掉另一个 writer 正在写的文件）。

读取路径对 artifact 做**逐身份维**校验，任一不符即 `ArtifactError`（§3.2，不静默
重新推理，也不静默接受）：

```text
反序列化失败（文件损坏）                    → ArtifactError
artifact_id != key.digest                   → ArtifactError
symbol / market_date / knowledge_cutoff     → ArtifactError
input_data_hash                             → ArtifactError
distribution.horizon / sample_count         → ArtifactError
distribution.origin_close                    → ArtifactError
distribution.distribution_spec_hash         → ArtifactError
distribution.metric_definition_version / aggregation_definition_version → ArtifactError
sampling.{seed, sample_count, temperature, top_k, top_p} → ArtifactError
model.{model_id, revision, runtime_version, device, dtype, config_hash} → ArtifactError
sample 的 point 时间轴 != key.future_sessions → ArtifactError
```

只比 `artifact_id` 是不够的：`compute()` 可能回填了正确的 id 却用了别的日历、别的
分布 spec 或别的采样参数（这正是本任务评审构造出的反例）。写入路径（`put`）走同一
个 `_verify`，因此「推理产物与 key 不符」在落盘前就已失败。逐维失败会一次性汇报所有
不匹配维（拼接为一条 `ArtifactError`），便于定位是哪一层 provenance 漏回填。

`_verify` 校验的是**输入身份与分布锚点**，不重算样本路径与指标值（那需要重跑推理，
违背缓存目的）。因此缓存正确性以「`compute()` 诚实且确定性」为前提：它把可信
实现产出的 `ForecastResult` 绑定到由 `(history, request, calendar, spec)` 派生的 key。
若调用方伪造 `input_data_hash`（例如同末根 close、不同早期 bars），理论上可写入
「同 key 异输出」——这已超出缓存层的职责边界，由 §9 的 determinism 测试与
artifact provenance 审计共同兜底。

### 3. `cached_forecast` 是 §15 的唯一执行点

```python
result, was_cached = cached_forecast(cache, key, compute, force=False)
```

命中直接返回；miss 则调用 `compute()` 恰好一次并原子写入。`force=True` **只绕过读
缓存**，仍然写入（覆盖旧字节，内容确定性下等价）。并发 worker 同时 miss 时会各算
一次（§15 允许），后写者胜出且内容一致。

`compute()` 产出的 `ForecastResult` 必须与 `key` 的每一维身份一致，由 `cache.put`
校验——推理层若忘记回填 provenance，或用了与 key 不符的日历 / 分布 spec / 模型，
缓存不会静默接受一个「身份不明」的产物。

### 4. 存储介质与 §30 的边界

本任务只交付**可注入的 `ForecastCache` Protocol + 本地文件实现**。Parquet 列式
布局、SQLite run registry / artifact index（§30–§33）、ULID Run ID（§31）属
RX-KAI-015；届时 `FileSystemForecastCache` 可被 store 实现替换而不改调用方。

### 5. 单写者假设的显式性

文件实现不引入锁：§15 的并发安全由「原子 rename + 内容确定性」保证，而不是靠
全局互斥。SQLite 侧的写入串行化（WAL）由 RX-KAI-015 承担。

## 后果

- 同一 key 的重复推理在 `cached_forecast` 层被消除；`--force` 是唯一重算入口，
  且必然重写缓存。
- key 契约一旦变更，golden digest 与所有缓存同时失效——这是刻意的，避免新旧产物
  混用。RX-KAI-013 加入 `future_sessions` / `distribution_spec_hash` 属于 key 契约变更，
  同时 `ForecastDistribution` 新增 `distribution_spec_hash` 字段（`ForecastResult`
  schema 变更），当时 `FORECAST_CONTRACT_VERSION` 升到 `forecast-contract-v2`。
- 评审加固（RX-KAI-013 之后）把 `FORECAST_CONTRACT_VERSION` 再升到
  `forecast-contract-v3`：`ForecastDistribution` 新增 `origin_close`（P_0，§13 指标的
  基准，落盘后 artifact 可被第三方独立重算）与 `aggregation_definition_version`
  （跨样本聚合口径的版本）。二者均为必填字段：旧 v2 artifact 因 key 失效+反序列化
  失败而被显式淘汰（版本白名单只保证「旧 metric/aggregation 版本仍可反序列化」，
  不承诺兼容新增必填 schema 字段）。同时 `distribution_spec_hash` 不再重复写模块常量版本号，
  只以 `spec.version` 为准，并折叠聚合版本；`_verify` 增加 metric/aggregation 版本的
  显式逐维比对。golden digest、`test_forecast_request.py` 的两个 golden hash 与
  `test_distribution.py` 的 `distribution_spec_hash` golden 同步更新。
- 二次评审加固把 `FORECAST_ARTIFACT_KEY_VERSION` 升到 `forecast-artifact-key-v2`：
  把 `origin_close`（P_0）纳入 key 与 `_verify`。上一轮虽把 P_0 落盘进 artifact，
  但它未进 key、也未被 `_verify` 比对，因此同一 history 可能产生两个
  「同 digest、不同 origin_close」的 artifact（连带收益/回撤/波动全部不同）且都被
  静默接受——与 `future_sessions` / `distribution_spec_hash` 当初要堵的是同一类洞。
  现在 key 从 history 末根 bar 派生 P_0，`_verify` 再逐维回比，`build_distribution`
  若用了别的 P_0 会在落盘前显式失败。`ForecastResult` 同时强制 sample_id 连续升序
  （§11 run 内 0-based 序号），原子写补父目录 `fsync` 以持久化 rename。
- Benchmark / walk-forward 的成本曲线可归因：命中数 = 省下的推理次数。
- artifact 身份错误从「结果悄悄变了」降级为「构造期显式失败」：key 少字段、模型
  identity 不全、provenance 未回填都会立刻报错。
- 日历 provenance（`TradingCalendar.exchange` / `source`）目前**不**记入 artifact：
  影响输出的部分是未来 session 时间轴，已由 `future_sessions` 进 key。把 exchange/source
  也写进 run metadata（§31/§32）属 RX-KAI-015，不阻塞本任务的缓存正确性。
- `model.backend`（如 `"kronos"`）**故意**不进 key：它是实现名而非身份——同一模型可由不同 backend
  包装，真正区分输出的是 `model_id` + `model_revision` + `runtime_version` + `config_hash`。
  多 backend 共存后（RX-KAI-020 Go/Replace gate）如出现「同 model_id、不同 backend 且实现不同」，
  由各自的 `config_hash` 承担区分。
- 未来引入 Parquet/SQLite store 时，`ForecastArtifactKey` 与 `cached_forecast`
  语义保持不变（缓存的正确性只依赖 key + 原子写，不依赖介质）。
