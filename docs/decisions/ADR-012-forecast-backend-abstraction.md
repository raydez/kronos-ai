# ADR-012: Forecast Backend Abstraction

- 状态：Accepted
- 日期：2026-09-26
- 对应任务：RX-KAI-014
- 对应基线：`Kronos-AI_v2_Greenfield_Rewrite_设计与落地方案.md` §15、§17、§34、§36（ADR-012）

## 背景

v1 把「取数 → 模型加载 → 推理 → 采样 → 指标」揉进
`backend/app/services/kronos_integration.py`（God Object）：模型是进程级单例，取数、
推理、缓存、HTTP 语义耦合在一处，无法替换 backend、无法在不启动 Web 的情况下跑研究、
也无法把「同一请求换一个模型」当成可比较的实验。v2 要求 Core 同步、可注入、可对比
（§3.7、§17），并把 CLI 作为第一阶段主入口（§34）。

## 决策

### 1. 三层职责分离，接口固定

```text
data/     MarketDataProvider.get_history(...) -> MarketHistory        §18
forecast/ ForecastBackend.forecast(history, request) -> ForecastResult §17
forecast/ ForecastService（应用编排：provider → backend）             §17/§34
```

- `ForecastBackend` 是 `runtime_checkable` Protocol，成员为 `name`（只读属性）与
  `forecast(history, request, *, force=False) -> ForecastResult`。`force` 是 §15 缓存
  语义的统一开关：无缓存的实现接收并忽略，有缓存的实现只绕过读缓存、写入仍原子。
- 实现可以有额外的可选关键字参数，但必须至少兼容该签名；`name` 作为属性而非可写字段，
  防止调用方在运行中途改写 backend 身份（它会写入 Run Metadata §32 的
  `forecast_backend`）。`name` **不**进入 §15 的 `ForecastArtifactKey`：该 key 的
  模型身份由 `model_id` / `model_revision` / `runtime_version` 表达（ADR-011）。
- 领域对象（`MarketHistory` / `ForecastRequest` / `ForecastResult`）是层间唯一通货，
  任何一层都不得把 provider 原始行、torch 张量或 pandas 结构泄漏给上层。

### 2. 禁止的形态

- 禁止 `PredictionService` / `KronosIntegration` 之类的 God Object：数据、推理、缓存、
  传输语义不得在同一类型里生长。
- 禁止 `ModelManager` 式进程级 Singleton：backend 生命周期由调用方持有，进程内可以存在
  多个 runtime（多模型 / 多 dtype / 多设备对比）。
- 禁止 silent fallback：模型加载失败、provider 失败、日历覆盖不足一律显式抛错
  （ADR-010），不返回合成结果、不缩短 horizon（ADR-009）。

### 3. RuntimeRegistry（§17）

- 名字 → 实例的注册表，`register_*` / `get_*` / `names`；重复注册、未知名字、空名字
  一律抛 `ConfigurationError`，不返回 `None`、不做隐式默认。
- forecast 与 decision 使用**各自独立**的命名空间，避免 `"rule"` 这类名字在两类
  backend 间碰撞。decision 命名空间随 RX-KAI-022 的 `DecisionBackend` 契约接入。
- 注册表不负责构造：装配（读配置、选 device、加载权重）在 CLI/worker 的 composition
  root，注册表只解析。

### 4. 缓存接入点

`KronosForecastBackend` 是唯一同时持有「模型身份」与「未来时间轴（calendar）」的层，
而 §15 的 `ForecastArtifactKey` 两者都需要。因此缓存作为可注入依赖接在 backend 内
（`cache=None` 即不缓存），而不是让外部调用方拼 key、拼 provenance。这样
`cached_forecast` 的「同一 key 不得重复推理」在 backend 边界可被单测，且 `artifact_id`
由 backend 回填、与 key 天然一致。

### 5. CLI First（§34）

- `kronos-ai forecast`、`kronos-ai benchmark forecast`、`kronos-ai run show` 为第一阶段
  命令；CLI 只做参数解析、装配、输出格式化，不复制领域逻辑。
- 未传 `--knowledge-cutoff` 时默认 `same_day_evening`（market_date 当日 18:00+08:00），
  policy 名 + 参数 + 版本写入输出摘要与 Run Metadata（§5），不允许在输出里只出现时刻。
- 重依赖（baostock / torch / huggingface_hub）在装配阶段延迟 import：`--help` 与参数
  解析不触发模型运行时加载。
- 装配是可注入的（`service_factory`）：测试与集成脚本可注入合成 provider / backend，
  不联网、不下载权重即可跑通 CLI 全链路。

### 6. Core 同步，异步在编排层（ADR-013 前置）

`ForecastBackend.forecast` 与 `ForecastService.run` 都是同步的。Web 编排层（RX-KAI-029）
在外层以 job 形式调度（提交 → 202 → 轮询状态），不要求 Core 变成 async，也不允许
长任务在请求线程内同步阻塞。

## 后果

- 同一 `ForecastRequest` 可以在 registry 里换不同 backend 运行并逐字段比较，为
  §19 / §48 的 benchmark 与 Go/Replace gate 提供前提。
- 模型、缓存、日历、数据源均可在测试中替换，`tests/unit` 能用 stub provider/backend
  覆盖 CLI 与编排，`tests/regression` 用 tiny 权重覆盖真实推理接缝。
- `name`、`force` 属于 `ForecastBackend` 对外契约；`identity()` 是
  `KronosForecastBackend` 的 provenance 访问器（供 §15 artifact key 构造使用；
  Run Metadata 的模型维由 `KronosRuntime.metadata()` 提供）。
  上述任一变更都需升级对应版本常量并同步 golden 测试。
