# Kronos-AI v2｜Greenfield Rewrite 设计与落地方案

> 文档类型：架构与工程落地基线  
> 适用仓库：`raydez/kronos-ai`  
> 升级策略：Same Repository + Architecture Rewrite + Selective Migration  
> 版本：v2 Baseline
> 修订日期：2026-09-26  
> 状态：Kronos-AI v2 唯一施工基线


---

# 1. 核心结论

Kronos-AI v2 继续采用：

> **Same Repository + Greenfield Core**

即：

```text
保留 raydez/kronos-ai
↓
冻结 v1
↓
保留 Git 历史 / Star / Fork / Issues
↓
Greenfield Rewrite v2
↓
Selective Migration
```

不继续沿用：

```text
PredictionService
ModelManager Singleton
KronosIntegration God Object
```

v2 的正式定位为：

> **Financial Forecast & Probabilistic Decision Research Platform**

中文：

> **金融时间序列预测与概率决策研究平台**

长期主链：

```text
Point-in-Time Market Data
        ↓
Forecast Backend
        ↓
Raw Forecast Samples
        ↓
Forecast Distribution
        ↓
Market State
        ↓
Decision Backend
        ↓
Calibration
        ↓
Policy / Abstention
        ↓
Walk-forward Evaluation
        ↓
Benchmark / Research Report
```

平台边界：

v2 负责：

```text
Point-in-Time Market Data
Forecast
Forecast Distribution
State Construction
Typed Decision
Calibration
Policy
Abstention
Evaluation
Backtest Research
Benchmark
```

v2 不负责：

```text
Broker
Order Management
Execution
Account
Position
Portfolio Capital Allocation
Production Trading
```

v2 是研究平台，不是交易平台。

---

# 2. v1 冻结原则

v1 被正式定义为：

> Functional Prototype / Proof of Integration

它已经验证：

```text
A 股数据
↓
Kronos
↓
FastAPI
↓
Web UI
```

但 v1 中存在不适合研究平台继续沿用的机制：

- 人工 confidence；
- 单样本预测；
- 普通工作日代替真实交易日；
- ModelManager Singleton；
- KronosIntegration 多职责耦合；
- README 和 API 中存在无 Benchmark 支撑的性能字段；
- 数据源失败时随机生成 K 线；
- 预测解析失败时随机生成 OHLC；
- 同步模型推理运行在异步调用链中；
- 配置、运行时和业务逻辑边界不清。

因此：

```bash
git tag v1.0.0
git checkout -b v2
```

基线文档与 v1 freeze 记录在 v2 分支首个 commit 进入仓库；tag v1.0.0 的树保持纯 v1 内容，checkout v1.0.0 不会看到任何 v2 文件。

v1 只允许维护：

```text
critical bugfix
security
documentation
reproducibility
```

不再新增核心能力。

分支与合并策略：

```text
v2 branch 上重写
→ 替换仓库实现
→ 稳定后 merge 回 main
```

禁止：

```text
长期保留 backend_v2/ 双目录结构
```

原因：

```text
双架构目录会成为永久负担
新代码永远有"退回旧实现"的借口
```

v1 通过 tag / release / git history 保留，
不通过并行目录保留。

---

# 3. Greenfield Rewrite 的硬性原则

## 3.1 Contract First

先定义：

```text
MarketBar
MarketHistory
UniverseSnapshot
ResearchTime
ForecastRequest
SamplingConfig
ForecastSample
ForecastDistribution
ForecastResult
MarketState
FeatureSet
DecisionSchema
DecisionAnswer
DecisionResult
CalibratedDecision
PolicyResult
LabelPolicy
EvaluationRun
```

实现围绕 Contract 工作。

## 3.2 Fail Explicitly

Research / Benchmark 主链禁止任何静默 synthetic fallback。

错误行为：

```text
Baostock 失败
↓
随机生成 K 线
↓
继续 Forecast
```

禁止。

错误行为：

```text
Kronos 输出解析失败
↓
随机生成 OHLC
↓
继续返回结果
```

禁止。

正确行为：

```text
ProviderError
ModelInferenceError
DataQualityError
InsufficientHistoryError
```

由上层显式处理。

如需模拟数据，仅允许通过：

```text
SyntheticMarketDataProvider
SyntheticForecastBackend
test fixtures
```

显式启用。

## 3.3 Point-in-Time First

任何历史 Benchmark 都必须回答：

> 在当时那个时间点，系统真实能够知道什么？

因此所有数据与状态必须具备：

```text
market_date
knowledge_cutoff
available_at
```

## 3.4 Raw Samples First

Kronos 的多样本不能在进入 Forecast Layer 前被求均值。

必须：

```text
Kronos stochastic decode
↓
Raw Sample 1
Raw Sample 2
...
Raw Sample N
↓
ForecastSample[]
↓
Distribution Builder
```

## 3.5 Reproducibility by Construction

每次 Forecast 必须可追溯：

```text
model revision
input hash
sampling config
seed
device
dtype
code revision
```

## 3.6 Evaluation First

新增模型前先定义：

```text
baseline
dataset
split
metric
acceptance gate
compute budget
```

## 3.7 Core Sync, Orchestration Async

核心研究 Runtime 以同步接口为主：

```text
DataProvider
ForecastBackend
DecisionBackend
Calibrator
Evaluator
```

FastAPI / Job System 在外层按需异步。

避免为了 Web 框架提前把同步推理强行包装成全 async core。

---

# 4. v2 推荐终态目录

> 以下为终态结构；首版只创建 Phase 1 所需子集。

```text
kronos-ai/
├── src/
│   └── kronos_ai/
│       ├── domain/
│       │   ├── market.py
│       │   ├── universe.py
│       │   ├── forecast.py
│       │   ├── state.py
│       │   ├── decision.py
│       │   ├── policy.py
│       │   └── evaluation.py
│       │
│       ├── data/
│       │   ├── base.py
│       │   ├── calendar.py
│       │   ├── adjustment.py
│       │   ├── universe.py
│       │   ├── quality.py
│       │   └── features.py
│       │
│       ├── forecast/
│       │   ├── base.py
│       │   ├── service.py
│       │   ├── distribution.py
│       │   ├── cache.py
│       │   └── backends/
│       │       └── kronos/
│       │           ├── runtime.py
│       │           ├── sampler.py
│       │           └── backend.py
│       │
│       ├── state/
│       │   └── builder.py
│       │
│       ├── decision/
│       │   ├── base.py
│       │   ├── service.py
│       │   ├── capability.py
│       │   └── backends/
│       │       ├── rule.py
│       │       ├── logistic.py
│       │       ├── lightgbm.py
│       │       └── external/
│       │
│       ├── calibration/
│       │   ├── base.py
│       │   ├── temperature.py
│       │   ├── platt.py
│       │   └── isotonic.py
│       │
│       ├── policy/
│       │   ├── engine.py
│       │   └── abstention.py
│       │
│       ├── evaluation/
│       │   ├── dataset.py
│       │   ├── walk_forward.py
│       │   ├── baselines.py
│       │   ├── forecast_metrics.py
│       │   ├── decision_metrics.py
│       │   ├── calibration_metrics.py
│       │   ├── trading_metrics.py
│       │   ├── compute_metrics.py
│       │   └── report.py
│       │
│       ├── infrastructure/
│       │   ├── providers/
│       │   │   └── baostock.py
│       │   ├── persistence/
│       │   ├── jobs/
│       │   └── external/
│       │
│       ├── cli/
│       └── api/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── regression/
│   ├── leakage/
│   └── benchmark/
│
├── configs/
├── experiments/
├── artifacts/
├── docs/
│   ├── architecture.md
│   ├── migration-v1-v2.md
│   └── decisions/
│
├── pyproject.toml
├── docker-compose.yml
└── README.md
```

---

# 5. 时间语义：market_date 与 knowledge_cutoff

v2 不再用单一 `as_of` 表达时间。

定义：

```python
class ResearchTime(BaseModel):
    market_date: date
    knowledge_cutoff: datetime
```

其中：

```text
market_date
= 研究对象对应的交易日

knowledge_cutoff
= 系统被允许使用信息的最晚时间
```

`knowledge_cutoff` 必须：

```text
timezone-aware
Asia/Shanghai
```

例如：

```text
market_date = 2026-09-25

knowledge_cutoff =
2026-09-25T18:00:00+08:00
```

和：

```text
2026-09-25T14:30:00+08:00
```

是不同实验状态。

所有 Feature 必须满足：

```python
available_at <= knowledge_cutoff
```

knowledge_cutoff 本身由 policy 决定，必须版本化：

```text
cutoff_policies:

market_close
= market_date 当日收盘（用于盘后研究场景）

same_day_evening
= market_date 当日固定时刻（如 18:00+08:00，等待数据源盘后更新完成）

explicit
= 调用方显式指定任意时刻（用于严格复现历史实验状态）
```

Run Metadata 必须记录：

```text
knowledge_cutoff_policy = policy 名 + 参数
```

CLI 默认值推导：

```text
未显式指定时
→ same_day_evening
→ market_date 当日 18:00 Asia/Shanghai
```

默认值允许变更，但变更必须升 policy 版本，不允许静默改变历史 run 的复现状态。

---

# 6. A 股 Point-in-Time 数据策略

## 6.0 数据源能力 Spike 前置

本节全部策略建立在数据源真实能力之上。

实现任何 point-in-time 组件之前，必须先完成 BaoStock 能力 spike 并产出报告：

```text
query_hs300_stocks / query_zz500_stocks
→ 历史成分股回溯深度、变更日期粒度

query_adjust_factor
→ 复权因子的 PIT 语义：按历史日期查询，返回的是当时生效因子还是最新重算因子

query_history_k_data_plus
→ tradestatus 字段可用性、停牌日是否有 bar

query_trade_dates
→ 交易日历覆盖范围
```

若某项能力不满足 point-in-time 要求：

```text
先引入补充数据源（如 Tushare index_weight）
再决定实现方案
```

而不是先写抽象层、再在 Phase 2 中途暴露数据缺陷。

该 spike 是 RX-KAI-006 / 007 的前置子任务。

## 6.1 复权不是一个简单开关

不能只写：

```text
前复权 / 后复权 / 不复权
```

必须区分：

```text
Raw OHLC
Corporate Actions
Adjustment Factor
Point-in-Time Adjusted Series
```

推荐保存：

```text
raw market data
+
corporate action metadata
+
adjustment factor metadata
```

具体 Forecast Backend 使用哪种输入，必须通过 ADR 固化。

原因：

历史前复权序列可能因后续公司行动发生变化。

因此 Benchmark 需要保证：

```text
历史输入可复现
```

而不是每次查询得到一套被未来公司行动重算过的价格。

## 6.2 Universe 必须 Point-in-Time

禁止：

```text
用今天的 CSI300 成分股
回测 2020 年
```

定义：

```python
class UniverseSnapshot(BaseModel):
    universe_id: str
    effective_date: date
    symbols: list[str]
    source: str
    version: str
```

每个 Benchmark Date 必须读取当时有效的成分股快照。

## 6.3 停牌策略

必须定义：

```text
market session
stock session
valid bar
```

默认建议：

```text
Forecast horizon
= 市场未来 N 个交易 session
```

Label 构建时若个股停牌导致不足 N 个有效 bar，需要显式标记：

```text
SUSPENDED
INSUFFICIENT_FUTURE_BARS
```

不能默默向后延长。

## 6.4 新股

需要最小历史长度：

```text
min_history_bars
```

历史不足时：

```text
InsufficientHistoryError
```

不自动 padding 随机数据。

## 6.5 退市与 ST

Universe Builder 与 Dataset Builder 必须保留真实历史状态，不因今天退市而从历史样本中删除。

这也是避免幸存者偏差的一部分。

## 6.6 TradingCalendar Contract

日历是 Forecast 时间轴与 Label 时间轴的承重墙，签名必须在基线中固定：

```python
class TradingCalendar(Protocol):

    def next_sessions(
        self,
        market_date: date,
        count: int,
    ) -> list[date]:
        ...

    def is_session(
        self,
        day: date,
    ) -> bool:
        ...
```

支持 SSE / SZSE / BSE。

所有：

```text
ForecastPoint.timestamp
Label horizon
Walk-forward 切点
停牌标记
```

必须经由同一 TradingCalendar 实例生成，禁止各自独立推导。

---

# 7. Market Contract

```python
class MarketBar(BaseModel):
    symbol: str

    # 该 bar 所属交易 session 的收盘时刻，语义见本节末尾
    timestamp: datetime

    open: float
    high: float
    low: float
    close: float

    volume: float | None
    amount: float | None

    trade_status: str | None
    adjustment_mode: str

    available_at: datetime
```

MarketHistory：

```python
class MarketHistory(BaseModel):
    symbol: str
    market_date: date
    knowledge_cutoff: datetime

    bars: list[MarketBar]

    provider: str
    dataset_version: str
    data_hash: str
```

时间戳语义：

```text
MarketBar.timestamp
= 该 bar 所属交易 session 的收盘时刻（15:00 Asia/Shanghai）
```

统一取收盘时刻而不是开盘时刻，是为了让：

```text
available_at <= knowledge_cutoff
```

拥有确定的比较基准。

`available_at` 语义：

```text
数据源实际可提供该 bar 的最早时间
```

且必须满足：

```text
available_at >= timestamp
```

Provider 负责给出真实发布延迟（BaoStock 为盘后更新）。Leakage Guard 依赖 `available_at` 字段本身，而不是"收盘即可见"的假设。

Symbol 规范形式：

```text
内部规范 symbol = 6 位数字代码，如 600000
```

```text
sh.600000 / sz.000001 等带前缀形式
仅允许在 Provider Adapter 边界存在
```

Adapter 负责双向转换；进入 domain 层后不再出现交易所前缀。Cache key、artifact hash、dataset hash 均基于规范 symbol，不受数据源格式差异污染。

行情单位约定：

```text
volume  = 股
amount  = 元
```

Provider 层负责把 BaoStock 的原始单位换算到该约定，domain 层不感知数据源单位差异。

---

# 8. ForecastRequest

```python
class SamplingConfig(BaseModel):
    seed: int

    sample_count: int = 64

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.9


class ForecastRequest(BaseModel):
    symbol: str

    market_date: date
    knowledge_cutoff: datetime

    horizon: int = 5

    sampling: SamplingConfig
```

`lookback_bars` 不在 ForecastRequest 中：

```text
lookback_bars 属于 Forecast Backend / Model 配置
```

但它决定输入窗口与逐窗口归一化统计量，因此必须：

```text
纳入 config_hash
进入 input_data_hash 计算范围
随 ModelMetadata 版本化
```

这里 seed 不允许只存在于 Evaluation Run metadata 中。

它必须进入：

```text
Request
↓
Backend
↓
Sampler
↓
Result Metadata
```

---

# 9. RNG Isolation

不推荐只使用：

```python
torch.manual_seed(seed)
```

作为最终方案。

因为并发运行会共享全局 RNG 状态。

推荐：

```text
per-run RNG
```

实现目标：

```python
generator = torch.Generator(device=device)
generator.manual_seed(seed)
```

然后采样接口显式接收：

```text
generator
```

如果上游 Kronos `sample_from_logits()` 不支持 generator，则 v2 的 Kronos sampler 应建立受控 patch / adapter。

需要新增测试：

```text
same input
+
same model
+
same seed
+
same sampling config
=
same raw forecast samples
```

在同一硬件 / runtime 条件下成立。

同时要记录：

```text
torch version
device
dtype
CUDA/MPS
```

因为跨设备完全 bitwise reproducibility 不应被默认承诺。

---

# 10. Raw Forecast Sample Exposure

这是 v2 的 P0 设计。

Kronos 原生多样本逻辑当前类似：

```text
repeat sample_count
↓
stochastic autoregressive decode
↓
reshape sample dimension
↓
mean(sample dimension)
↓
return mean prediction
```

v2 必须在 mean 之前截取：

```text
raw decoded samples
```

推荐新接口：

```python
class KronosSampler:

    def generate_samples(
        self,
        history: MarketHistory,
        request: ForecastRequest,
    ) -> list[ForecastSample]:
        ...
```

禁止：

```text
sample_count=64
↓
调用原生 predict()
↓
拿到 mean prediction
↓
伪装成 distribution
```

---

# 11. ForecastSample

```python
class ForecastPoint(BaseModel):
    timestamp: datetime

    open: float
    high: float
    low: float
    close: float

    volume: float | None = None
    amount: float | None = None


class ForecastSample(BaseModel):
    sample_id: int
    points: list[ForecastPoint]
```

ForecastSample 必须是可独立持久化的研究 Artifact。

ForecastPoint.timestamp 语义：

```text
= 市场未来第 N 个交易 session
```

由 TradingCalendar.next_sessions 生成。

不是个股自身的有效 session，也不是 pandas 工作日推导：

```text
个股停牌不改变 forecast 样本的时间轴
```

停牌导致的实际 bar 缺失，由 Label Policy 在 label 侧显式标记：

```text
SUSPENDED
INSUFFICIENT_FUTURE_BARS
```

因此 forecast 路径与 label 路径共享同一时间轴语义。

---

# 12. ForecastDistribution

移除写死：

```text
p_return_gt_2pct
p_return_lt_minus_2pct
```

定义：

```python
class ThresholdProbability(BaseModel):
    metric: str
    operator: Literal["gt", "gte", "lt", "lte"]
    threshold: float
    probability: float


class QuantileValue(BaseModel):
    metric: str
    quantile: float
    value: float


class ForecastDistribution(BaseModel):
    horizon: int
    sample_count: int

    expected_return: float
    median_return: float

    threshold_probabilities: list[ThresholdProbability]
    quantiles: list[QuantileValue]

    forecast_dispersion: float
    expected_max_drawdown: float
    expected_path_volatility: float

    metric_definition_version: str
```

`ThresholdProbability.metric` / `QuantileValue.metric` 不允许自由字符串：

```text
metric 名称必须来自版本化 metric registry
```

例如：

```text
horizon_return
log_return
max_drawdown
path_volatility
```

新增 metric 属于 registry 版本升级，不允许各 backend 各自发明名称。

---

# 13. Forecast Metrics 数学定义

## 13.1 Horizon Return

默认：

```text
R_h = P_h / P_0 - 1
```

其中：

```text
P_0 = forecast origin close
P_h = horizon last close
```

## 13.2 Path Drawdown

对每条 Forecast Sample：

```text
close path
↓
running max
↓
drawdown
```

最大回撤：

$$
MDD = \min_t\left(\frac{P_t}{\max_{s \le t} P_s}-1\right)
$$

`expected_max_drawdown` 定义为：

```text
mean(sample-level MDD)
```

## 13.3 Path Volatility

默认定义为 Forecast horizon 内 sample path 的逐步 log return 标准差，再对 samples 求均值。

若未来改为其他定义，必须升级：

```text
metric_definition_version
```

---

# 14. ForecastResult 与 Provenance

不要求在 ForecastDistribution 内重复 symbol 等字段。

使用 Envelope：

```python
class ModelMetadata(BaseModel):
    backend: str
    model_id: str
    revision: str

    runtime_version: str
    device: str
    dtype: str

    config_hash: str


class SamplingMetadata(BaseModel):
    seed: int
    sample_count: int
    temperature: float
    top_k: int
    top_p: float


class ForecastResult(BaseModel):
    symbol: str

    market_date: date
    knowledge_cutoff: datetime

    samples: list[ForecastSample]
    distribution: ForecastDistribution

    model: ModelMetadata
    sampling: SamplingMetadata

    input_data_hash: str
    artifact_id: str
```

这样：

```text
ForecastDistribution
```

保持纯统计对象，

而 provenance 由：

```text
ForecastResult
```

统一承担。

---

# 15. Forecast Cache

v2 必须把 Forecast 当昂贵 Artifact 缓存。

定义：

```text
ForecastArtifactKey =
hash(
  symbol,
  input_data_hash,
  market_date,
  knowledge_cutoff,
  horizon,
  model_id,
  model_revision,
  runtime_version,
  device-class,
  dtype,
  seed,
  sample_count,
  temperature,
  top_k,
  top_p
)
```

同一个 key：

```text
不得重复推理
```

除非：

```text
--force
```

并发写安全：

```text
多 worker 并行 benchmark
→ 同一 cache key 同时 miss
→ 同时推理
→ 同时写入
```

约定：

```text
artifact 写入：临时文件 + 原子 rename
SQLite registry：WAL 模式，写入串行化
重复写入：内容确定性下 last-write-wins 无害
```

`--force` 只绕过读缓存，写入路径仍必须原子。

---

# 16. Benchmark Compute Budget

正式全量 Benchmark 前先执行 Cost Probe。

## Pilot Matrix

例如：

```text
50 stocks
×
100 forecast origins
×
sample_count {1, 16, 32, 64}
×
batch / worker 并行度 {1, 4, 16}
×
CPU / MPS / CUDA
```

禁止全因子笛卡尔积：

```text
先锁定目标设备（默认主力开发机一种）
再扫 sample_count × batch 维度
其他设备仅跑单点验证
```

目标：

```text
probe 总组数 <= 10
```

batch 维度必须显式扫描：

```text
Kronos 原生实现将 batch 与 sample_count 相乘后统一前向
实际内存随 batch × sample_count 增长
```

记录：

```text
forecast/sec
p50 latency
p95 latency
RAM peak
VRAM peak
throughput-memory curve（batch × sample_count 维度）
artifact size
cache hit ratio
```

输出：

```text
Benchmark Compute Budget Report
```

只有完成 Cost Probe 后，才能确定：

```text
full universe
full period
sample_count
hardware requirement
```

不在设计文档中提前声称完整实验一定是小时级、天级或周级。

Probe 自身必须有预算上限：

```text
wall-clock 上限（如每组 4 小时）
内存上限（触发即中止该组合）
```

任一组合超限，报告标记：

```text
INFEASIBLE_ON_THIS_DEVICE
```

Probe 不允许无上限运行。

---

# 17. ForecastBackend

Core 默认同步：

```python
class ForecastBackend(Protocol):

    name: str

    def forecast(
        self,
        history: MarketHistory,
        request: ForecastRequest,
    ) -> ForecastResult:
        ...
```

CLI：

```text
直接调用
```

Benchmark Worker：

```text
直接调用
```

FastAPI：

```text
submit job
↓
worker/executor
↓
sync core
```

避免 model inference 阻塞 HTTP event loop。

多 Backend 生命周期由 RuntimeRegistry 管理：

```python
class RuntimeRegistry:

    def get_forecast_backend(self, name: str) -> ForecastBackend: ...

    def get_decision_backend(self, name: str) -> DecisionBackend: ...
```

替代 v1 Singleton ModelManager：

```text
多模型
可测试
可配置
无全局单例耦合
```

Service 层是 Application Service：

```text
ForecastService
StateService
DecisionService
EvaluationService
```

每个 Service 是 Use Case 协调器，不承担底层技术细节。

禁止重新出现：

```text
PredictionService 式职责模糊的大服务
```

---

# 18. DataProvider

同样采用同步核心接口：

```python
class MarketDataProvider(Protocol):

    def get_history(
        self,
        symbol: str,
        market_date: date,
        knowledge_cutoff: datetime,
        lookback_bars: int,
    ) -> MarketHistory:
        ...
```

如果底层 BaoStock 是阻塞 IO：

- CLI / worker 可直接同步调用；
- Web orchestration 可通过 executor / thread pool 调度；
- 不要求 Domain / Core 变成 async。

---

# 19. Forecast Phase 2 Go / No-Go Gate

Phase 2 的目标不是证明 Kronos 必然有效。

而是判断：

> Kronos 是否值得继续作为默认 Forecast Backend？

至少与：

```text
Last Value
Drift
Moving Average
简单 Statistical Baseline
```

比较。

Gate 不使用单一拍脑袋阈值。

综合判断：

```text
Forecast quality
+
regime consistency
+
confidence interval
+
economic usefulness
+
compute cost
```

如果 Kronos 无法形成稳定增量：

```text
Kronos
↓
降级为 benchmark backend
```

项目不终止。

而是进入：

```text
Forecast Backend replacement
```

候选：

```text
Chronos
TimesFM
Moirai
PatchTST
statistical baseline
future models
```

因此：

```text
Kronos failure
≠
Platform failure
```

---

# 20. MarketState

```python
class MarketState(BaseModel):
    symbol: str

    market_date: date
    knowledge_cutoff: datetime

    market_features: dict[str, float]
    forecast: ForecastDistribution

    schema_version: str
    feature_set_version: str
    builder_version: str

    source_forecast_artifact_id: str
```

不再使用模糊：

```text
state_version
```

`market_features` 不允许是裸 dict：

```text
合法 key 集合、单位、方向
由 feature_set_version 对应的 FeatureSet 契约静态定义
```

Builder 出口必须通过 FeatureSet 校验：

```text
key 完整性：无缺失、无未知 key
数值有限性：no NaN / Inf
```

DecisionBackend 只能依赖 FeatureSet 声明的特征，禁止直接读取底层数据源。

---

# 21. Decision Layer 进入条件

Decision Layer 只有在：

```text
Forecast / State / Dataset / Evaluation
```

稳定后才进入。

第一阶段 Decision Backend：

```text
Rule
Logistic Regression
LightGBM
```

它们负责建立强 baseline。

Typed Decision 不默认进入 v2 Core。

---

# 22. Candidate Typed Decision Backends

当前只定义候选能力矩阵。

| Backend | 类型 | 运行形态 | 是否本地 | 概率来源 | 备注 |
|---|---|---|---|---|---|
| AnyJev | LLM → Decision | local model / adapter | 可 | logits + calibration | 适合研究 calibration |
| Kev | 专用 Decision Model | HTTP / local service | 可 | calibrated decision probability | Python 环境可能独立 |
| Nimble | 开放训练/模型路线 | model/service | 可 | typed scoring | 更偏研究与训练 recipe |
| TypeSafe Jev | 专用 System-One API | external API | 否/依实现 | API probability | Reference benchmark |

v2 不规定必须把全部候选纳入首版 Core。

进入实际开发前，每个 Candidate 必须补充：

```text
dependency
license
runtime
API
authentication
model size
device
latency
output schema
calibration semantics
failure mode
```

---

# 23. External Backend Adapter 原则

禁止：

```text
主环境直接硬依赖所有候选模型
```

优先：

```text
HTTP service
local subprocess
isolated runtime
```

统一：

```python
class ExternalDecisionAdapter(Protocol):

    def decide(
        self,
        state: MarketState,
        schema: DecisionSchema,
    ) -> DecisionResult:
        ...
```

---

# 24. Decision Layer Contracts

## 24.1 第一版 Decision Schema

保持最小，四个维度：

```text
Direction
Signal Validity
Risk Level
Actionability
```

Direction：

```text
BULLISH
NEUTRAL
BEARISH
UNKNOWN
```

Signal Validity：

```text
TRUE
FALSE
```

Risk Level：

```text
1
2
3
4
5
```

Actionability：

```text
CANDIDATE
WATCH
AVOID
ABSTAIN
```

显式不设计：

```text
BUY
SELL
POSITION SIZE
```

研究层不越界到执行层，该边界随 schema 版本固化。

Schema 版本：

```text
default-v1
```

新增维度属于新 schema 版本，不允许原地扩展。

标签语义边界（v2 首版）：

```text
Direction
→ LabelPolicy.direction_thresholds 构造 label，参与训练与评估

Actionability
→ 由 Policy 从 calibrated probability 推导，不需要独立 label

Signal Validity / Risk Level
→ schema 预留维度
→ v2 首版不构造 ground truth、不参与训练与评估
```

禁止为预留维度强行发明标签制造伪 ground truth。

维度启用条件：

```text
LabelPolicy 补构造规则并版本化
+
Decision Schema 升版本
两者缺一不可
```

## 24.2 DecisionAnswer 与 DecisionResult

Calibration 需要知道概率从哪里来。

```python
class DecisionAnswer(BaseModel):
    direction: Literal["BULLISH", "NEUTRAL", "BEARISH", "UNKNOWN"]
    signal_validity: bool | None = None
    risk_level: int | None = None
    actionability: Literal["CANDIDATE", "WATCH", "AVOID", "ABSTAIN"] | None = None


class DecisionScores(BaseModel):
    score_space: Literal[
        "logits",
        "probabilities",
        "calibrated_probabilities"
    ]

    values: dict[str, float]
```

`values` 的 key 语义：

```text
key 必须等于当前 schema_version 的枚举值集合
```

例如 direction 三分类：

```text
{"BULLISH": 1.2, "NEUTRAL": -0.4, "BEARISH": -2.1}
```

枚举含 UNKNOWN 共四值，但示例只有三个 key：

```text
key 是枚举值集合的子集
UNKNOWN 表示模型无法给出判断，无 score，故不出现
```

禁止 backend 各自发明 key 名；Calibrator 与 Policy 依此契约消费。

```python
class DecisionResult(BaseModel):
    backend: str
    model_revision: str

    answer: DecisionAnswer
    scores: DecisionScores

    latency_ms: float

    schema_version: str
```

`answer` 是结构化对象而不是裸字符串：

```text
Calibrator / Policy / Benchmark
依赖确定的字段与枚举
```

## 24.3 CalibratedDecision

```python
class CalibratedDecision(BaseModel):
    decision: DecisionResult

    calibrated_scores: DecisionScores

    calibrator: str
    calibrator_version: str

    fit_provenance: CalibratorFitProvenance


class CalibratorFitProvenance(BaseModel):
    fit_window_start: date
    fit_window_end: date
    label_policy_version: str
    fit_dataset_hash: str
```

约束：

```text
calibrated_scores.score_space = "calibrated_probabilities"
```

Calibrator 跨 run 复用时，fit provenance 回答：

```text
它是在哪些数据上拟合的
用的是哪个版本的 label 定义
```

## 24.4 PolicyResult

```python
class PolicyResult(BaseModel):
    source: CalibratedDecision

    actionability: Literal["CANDIDATE", "WATCH", "AVOID", "ABSTAIN"]

    direction: Literal["BULLISH", "NEUTRAL", "BEARISH", "UNKNOWN"]

    policy_version: str
```

Policy 输出可以推翻 Decision 的原始判断：

```text
BULLISH 0.52
↓
ABSTAIN
```

这正是 Policy 独立于 Decision Model 存在的意义。

---

# 25. Calibrator Contract

```python
class Calibrator(Protocol):

    name: str
    required_score_space: str

    def fit(
        self,
        scores,
        labels,
    ) -> None:
        ...

    def transform(
        self,
        scores,
    ):
        ...
```

能力示例：

```text
Temperature Scaling
requires logits

Platt Scaling
can use binary score / logit

Isotonic Regression
accepts probabilities / scalar score
```

不能假设所有 Backend 都暴露 logits。

---

# 26. Policy / Abstention

链路：

```text
Decision
↓
Calibrated Probability
↓
Policy
↓
Candidate / Watch / Avoid / Abstain
```

Policy 不属于 Decision Model。

Threshold 必须从：

```text
validation
calibration
error budget
```

推导。

核心指标：

```text
Coverage@1%
Coverage@5%
Coverage@10%
Abstention Rate
Confident Error Rate
```

---

# 27. Walk-forward Dataset

每个样本：

```text
Point-in-Time Universe
↓
Market Data available by knowledge_cutoff
↓
Features
↓
Forecast Artifact
↓
MarketState
↓
Future Label
```

禁止：

```text
random shuffle
```

采用：

```text
Train
↓
Validation
↓
Calibration
↓
Test
↓
Roll
```

分段间必须插入 embargo：

```text
embargo_sessions >= horizon_sessions
```

原因：

```text
horizon = 5 时，origin t 的标签横跨未来 5 个 session
若分段边界直接相邻
前段末尾 origin 的标签窗口
与后段开头样本的输入 / 标签窗口在时间轴上重叠
模型在训练标签里"见过"评估期价格
```

规则：

```text
前一段最后一个 origin 之后
空置 embargo_sessions 个 session
后一段第一个 origin 才能出现
```

注意：

```text
Phase 2 naive baseline 不训练
但 Dataset Builder 在 Phase 2 定型
Phase 4 / 5 的 LightGBM / Calibrator 复用同一 dataset
embargo 必须在此处一次性固化
事后补 embargo 会使全部 dataset_hash 失效
```

tests/leakage/ 必须包含标签窗口重叠用例：

```text
任意相邻两段之间
前段最后 origin + embargo_sessions
< 后段第一 origin
```

断言必须用 embargo_sessions 而不是 horizon_sessions：

```text
embargo_sessions > horizon_sessions 时
horizon 断言必要但不充分
```

---

# 28. Label Policy

必须版本化：

```python
class LabelPolicy(BaseModel):
    horizon_sessions: int
    embargo_sessions: int

    direction_thresholds: dict
    suspension_policy: str
    missing_future_bars_policy: str

    price_field: str
    return_definition: str

    version: str
```

`embargo_sessions` 随 LabelPolicy 版本化，禁止在代码中硬编码。

避免未来更换 label 定义后无法解释历史 benchmark。

---

# 29. Leakage Guard

P0 检查：

```text
available_at <= knowledge_cutoff
```

同时检查：

```text
future market data
future corporate actions
future universe membership
future calibration labels
future feature revisions
```

不得泄漏。

专门建立：

```text
tests/leakage/
```

---

# 30. Artifact Store

建议：

```text
Parquet
+
SQLite
```

Parquet：

```text
market snapshots
universe snapshots
forecast samples
states
decisions
benchmark datasets
```

SQLite：

```text
run registry
artifact index
model metadata
dataset metadata
job metadata
```

Raw provider response 采用 append-only 快照：

```text
首次拉取即落盘为快照
之后只读，永不覆盖
```

原因：

```text
BaoStock 历史数据存在追溯修订（复权因子尤甚）
快照是 point-in-time 可复现性的最后一环
```

重新拉取产生新数据时：

```text
新 dataset_version + 新快照
旧快照保留
```

---

# 31. Run ID

不再使用：

```text
20260926-001
```

推荐：

```text
ULID
```

满足：

```text
globally unique
sortable
parallel-safe
```

---

# 32. Run Metadata

```json
{
  "run_id": "01K...",
  "git_commit": "...",
  "config_hash": "...",
  "dataset_hash": "...",

  "market_date_range": "...",
  "knowledge_cutoff_policy": "...",

  "forecast_backend": "kronos",
  "forecast_model_revision": "...",

  "sampling": {
    "seed": 42,
    "sample_count": 64,
    "temperature": 1.0,
    "top_k": 0,
    "top_p": 0.9
  },

  "runtime": {
    "python": "...",
    "torch": "...",
    "device": "...",
    "dtype": "..."
  },

  "decision_backend": null,
  "calibrator": null,

  "universe_version": "...",
  "adjustment_policy_version": "...",
  "label_policy_version": "..."
}
```

## 32.1 配置体系

三层配置：

```text
Pydantic Settings（环境 + 默认值）
+
YAML Experiment Config（可复现实验描述）
+
Environment Secret（仅环境变量）
```

Experiment Config 示例：

```yaml
runtime:
  device: auto
  dtype: auto

forecast:
  backend: kronos
  model: kronos-small
  lookback_bars: 256
  sampling:
    seed: 42
    sample_count: 64
    temperature: 1.0
    top_k: 0
    top_p: 0.9

knowledge_cutoff_policy: same_day_evening

universe:
  source: csi300
  snapshot: point-in-time

evaluation:
  walk_forward: true
  embargo_sessions: 5
```

embargo_sessions 的单一真源是 LabelPolicy：

```text
experiment config 物化为 LabelPolicy 并版本化
dataset_hash 以 LabelPolicy 版本为准
```

若修改了 config 中的 embargo_sessions，
必须同步升 LabelPolicy 版本，否则视为配置错误拒绝运行。

Secret：

```text
API_KEY
HF_TOKEN
```

只从环境变量读取，禁止写入 YAML / 代码 / artifact。

config.yaml 本身作为 run artifact 归档，并参与 config_hash 计算。

---

# 33. Artifact 目录

```text
artifacts/
└── runs/
    └── <ULID>/
        ├── config.yaml
        ├── metadata.json
        ├── universe.parquet
        ├── forecast.parquet
        ├── samples.parquet
        ├── states.parquet
        ├── decisions.parquet
        ├── metrics.json
        └── report.md
```

---

# 34. CLI First

第一阶段优先：

```bash
kronos-ai forecast 600000 \
  --market-date 2026-09-25 \
  --knowledge-cutoff 2026-09-25T18:00:00+08:00 \
  --samples 64 \
  --seed 42
```

未传 `--knowledge-cutoff` 时：

```text
默认 same_day_evening
= market_date 当日 18:00 Asia/Shanghai
```

Benchmark：

```bash
kronos-ai benchmark forecast \
  --config configs/benchmark-forecast.yaml
```

Inspect Artifact：

```bash
kronos-ai run show <run-id>
```

---

# 35. API Job Mode

长任务禁止同步 HTTP 阻塞。

推荐：

```text
POST /api/v2/evaluation/runs
↓
202 Accepted
↓
run_id
```

然后：

```text
GET /api/v2/evaluation/runs/{run_id}
```

返回：

```text
queued
running
completed
failed
```

取消：

```text
POST /api/v2/evaluation/runs/{run_id}/cancel
```

防重复提交：

```text
POST /api/v2/evaluation/runs 携带 Idempotency-Key
同一 key 重复提交返回同一 run_id，不创建新 job
```

理由：

```text
全量 benchmark 是昂贵长任务
误提交两次的代价是双倍计算
```

---

# 36. 新 ADR 清单

```text
ADR-001 Greenfield Rewrite over Incremental Refactor
ADR-002 Standard src Layout
ADR-003 Contract First Architecture
ADR-004 Expose Raw Stochastic Forecast Samples from Kronos
ADR-005 Per-run RNG and Sampling Reproducibility
ADR-006 Point-in-Time Market Data Semantics
ADR-007 A-share Price Adjustment Policy
ADR-008 Point-in-Time Universe Snapshots
ADR-009 Trading Calendar and Suspension Policy
ADR-010 Fail Explicitly — No Silent Synthetic Fallback
ADR-011 Forecast Artifact Cache
ADR-012 Forecast Backend Abstraction
ADR-013 Core Sync, Async Orchestration
ADR-014 Walk-forward Only
ADR-015 Calibration as Platform Capability
ADR-016 Policy and Abstention Separation
ADR-017 Parquet + SQLite Artifact Storage
ADR-018 External Decision Backend Isolation
ADR-019 Benchmark Compute Budget and Pilot Gate
ADR-020 Data License and Artifact Redistribution
ADR-021 Engineering Standards and Quality Gate
ADR-022 Embargoed Walk-forward Splits
```

---

# 37. License 与数据分发

需要分别确认：

```text
source code license
vendored upstream code license
model weight license
dataset terms
market data redistribution
benchmark report redistribution
```

vendored upstream code license：

```text
backend/model/kronos.py 等 vendor 进 v2 的上游代码
vendor 前必须实际核验上游仓库 LICENSE（不得凭印象）
并在 vendor 目录保留原始 license 声明
与模型权重 license 分开记录
```

不得因为：

```text
SDK 是开源
```

推断：

```text
市场数据可任意再分发
```

默认策略：

```text
代码仓库
不提交原始行情数据
```

Benchmark Report 可发布：

```text
aggregated metrics
methodology
config
```

原始 Market Artifact 是否可公开，必须依据数据源条款单独判断。

---

# 38. v1 → v2 Migration Matrix

| v1 模块 | v2 处理 |
|---|---|
| `backend/model/kronos.py` | 校验后作为 upstream-derived runtime source |
| `backend/model/module.py` | 同上 |
| `stock_service.py` | 不迁移 fallback；参考 BaoStock 调用逻辑重写 |
| `_generate_fallback_data` | 删除；仅测试代码允许 synthetic provider |
| `kronos_integration.py` | 不迁移；拆 Runtime / Sampler / Backend |
| `model_manager.py` | 删除 |
| `prediction_service.py` | 删除 |
| `config.py` | 删除 |
| `main.py` | 后期重写 |
| frontend charts | 选择性迁移 |
| old API contracts | 不兼容迁移 |

---

# 39. Phase 0｜Freeze v1

任务：

```text
tag v1.0.0
create v2 branch
commit baseline doc + v1 freeze record（v2 首个 commit）
record environment
record model IDs
record current startup
document known invalid metrics
document synthetic fallback
```

Git 锚点顺序（与 §2 分支策略一致）：

```text
main 上 tag v1.0.0（纯 v1 代码 HEAD）
→ checkout -b v2
→ v2 分支上 commit 基线文档 + v1 freeze 记录
```

理由：

```text
tag v1.0.0 是 v1 系统的冻结锚点，语义必须纯净
checkout v1.0.0 得到的树不包含任何 v2 文件
基线文档的 git 锚点由 v2 分支首个 commit 承担
文档内容即最终基线，只保留最终版本，无需 tag 背书
v2 稳定后 merge 回 main，文档随 v2 进入主线
```

同时 README 标明：

> v1 is a legacy prototype and should not be used for research benchmark claims.

v2 README 首页以可复现性为卖点：

```text
Reproducible Benchmark
Calibrated Decision
Walk-forward Evaluation
```

禁止恢复无实验依据的准确率数字作为首页宣传。

---

# 40. Phase 1｜Greenfield Forecast Core

仅创建：

```text
domain
data
forecast
cli
tests
docs/decisions
```

交付：

```text
MarketHistory
ResearchTime
ForecastRequest
SamplingConfig
KronosRuntime
KronosSampler
KronosForecastBackend
ForecastSample
ForecastDistribution
Forecast Cache
CLI
```

验收：

```text
同一 input + seed + config
→ 同一 raw samples
```

在同一 runtime / device 条件下成立。

---

# 41. Phase 2｜Forecast Evaluation

建设：

```text
Point-in-Time Dataset Builder
Universe Snapshot
Adjustment Policy
Trading Calendar
Walk-forward Runner
Naive Baselines
Forecast Metrics
Compute Cost Probe
Artifact Store
```

Phase 2 输出：

```text
Forecast Benchmark v1
```

以及：

```text
Kronos Go / Replace Decision
```

---

# 42. Phase 2 Gate

可能结果：

## GO

```text
Kronos
继续作为默认 ForecastBackend
```

## CONDITIONAL

```text
Kronos
只在部分 regime / universe 有效
```

则后续按 regime 使用。

## REPLACE

```text
Kronos 无稳定增量
```

则：

```text
降级为 benchmark backend
```

并引入其他 Forecast Backend。

不终止整体平台建设。

Gate 判据必须预注册：

```text
任何正式 benchmark run 之前
提交 gate-criteria 文档
随 run_id 一起归档
```

判据必须可证伪，例如：

```text
Kronos 相对最强 statistical baseline 的 direction accuracy 增量
在 >= 3 个 regime 分组中
bootstrap 95% CI 下界 > 0
```

禁止：

```text
跑完后从多个指标中挑选有利指标
事后放宽阈值
```

判据的具体数值可以在 Cost Probe 完成后、正式 run 之前确定，但判据形态（指标、比较对象、分组方式、置信区间）必须在 run 前冻结。

---

# 43. Phase 3｜Market State

加入：

```text
MarketFeatures
ForecastDistribution
MarketState
Feature Set Versioning
```

仍不接 Jev。

---

# 44. Phase 4｜Decision Baselines

只实现：

```text
Rule
Logistic Regression
LightGBM
```

回答：

> 结构化市场状态能做到什么程度？

---

# 45. Phase 5｜Calibration + Policy

加入：

```text
score_space
calibrator
Brier
ECE
NLL
Coverage
Abstention
```

形成完整：

```text
Decision
→ Probability
→ Calibrated Probability
→ Policy
```

---

# 46. Phase 6｜Typed Decision Research

先做 Capability Matrix。

再选择 1–2 个 Backend 实现。

优先选择标准：

```text
reproducibility
local deployment
probability semantics
calibration
license
runtime isolation
latency
maintenance
```

而不是：

```text
Star
热度
Jev 名称相似度
```

---

# 47. Phase 7｜Semantic Context

只有前面稳定后，再考虑：

```text
news
announcement
sector
macro
event
narrative
```

这是 Typed Decision Model 真正可能比 LightGBM 更有优势的区域。

---

# 48. 第一轮 Benchmark 建议

先小规模：

```text
Universe:
CSI300 point-in-time subset

Forecast Origins:
100–300 sessions

Horizon:
5 sessions

Samples:
16 / 32 / 64

Baselines:
Last Value
Drift
Moving Average
Kronos
```

先验证：

```text
Forecast validity
compute budget
cache behavior
reproducibility
```

再扩大到：

```text
CSI300 + CSI500
```

---

# 49. Benchmark 指标

## Forecast

```text
MAE
RMSE
Direction Accuracy
Return Correlation
Quantile Coverage
CRPS（后续）
```

## Robustness

```text
Bull
Bear
Sideways
High Vol
Low Vol
```

## Compute

```text
p50
p95
forecast/sec
RAM
VRAM
artifact bytes / forecast origin
cache hit ratio
```

---

# 50. Decision Benchmark 指标

统计层指标：

```text
Macro F1
Balanced Accuracy
Brier
ECE
NLL
Coverage@5%
Abstention Rate
Latency
```

经济指标为 signal-level，不依赖仓位与执行假设：

```text
Quantile Portfolio Return Spread（Top / Bottom 分组收益差）
IC / Rank IC 及其 horizon 衰减
```

显式移除：

```text
Sharpe
Max Drawdown
Turnover
```

移除原因：

```text
三者依赖仓位假设与执行成本模型
（A 股佣金、印花税、滑点、T+1）
未定义 CostModel 前算出的数字没有意义
且与 §24.1 不越界执行层原则冲突
```

未来若确需交易层指标：

```text
先定义版本化 ExecutionAssumption / CostModel 契约
再随指标版本引入
```

---

# 51. 首批 RX-KAI 工程任务

```text
RX-KAI-001
Freeze v1 and create v2 branch

RX-KAI-002
Create src-layout Greenfield skeleton

RX-KAI-003
Define ResearchTime and core market contracts

RX-KAI-004
Implement explicit-failure BaoStock provider

RX-KAI-005
Define A-share TradingCalendar and session semantics

RX-KAI-006
Define point-in-time AdjustmentPolicy
前置：BaoStock 能力 spike——复权因子 PIT 语义、历史成分回溯、停牌字段验证，产出数据源能力报告

RX-KAI-007
Define UniverseSnapshot and historical constituents loader
依赖 spike 结论；成分股历史数据不满足 PIT 时先引入补充数据源

RX-KAI-008
Define ForecastRequest + SamplingConfig

RX-KAI-009
Implement KronosRuntime
verbose 默认关闭；lookback_bars 作为模型超参纳入配置版本化与 input hash

RX-KAI-010
Patch/extract Kronos raw stochastic sample path
截取点：sample 维 reshape 之后、mean 之前；移除逐步 torch.cuda.empty_cache()；注意 batch × sample_count 内存相乘；每个自回归 step 有 s1 / s2 两次采样调用，per-run RNG 必须覆盖全部调用点

RX-KAI-011
Implement per-run RNG isolation

RX-KAI-012
Implement ForecastSample + ForecastDistribution

RX-KAI-013
Implement Forecast Artifact Cache

RX-KAI-014
Build Forecast CLI

RX-KAI-015
Build Artifact Store + ULID Run Registry

RX-KAI-016
Implement Forecast baseline models

RX-KAI-017
Build Walk-forward dataset generator
分段边界必须执行 LabelPolicy.embargo_sessions 空置规则，dataset_hash 绑定 LabelPolicy 版本

RX-KAI-018
Build Benchmark Cost Probe

RX-KAI-019
Build Forecast Benchmark v1

RX-KAI-020
Add Forecast Backend Go/Replace Gate

RX-KAI-021
Define MarketState

RX-KAI-022
Define DecisionBackend + DecisionScores

RX-KAI-023
Implement Rule / Logistic / LightGBM

RX-KAI-024
Implement Calibration Layer

RX-KAI-025
Implement Policy + Abstention

RX-KAI-026
Create Typed Decision Backend capability matrix

RX-KAI-027
Implement first selected Typed Decision Backend

RX-KAI-028
Build Decision Benchmark v1

RX-KAI-029
Implement FastAPI Job API

RX-KAI-030
Migrate frontend selectively
```

---

# 52. 推荐施工顺序

```text
001
↓
002
↓
003 / 004 / 005
↓
006 / 007
↓
008
↓
009
↓
010
↓
011
↓
012
↓
013 / 014
↓
015
↓
016 / 017
↓
018
↓
019
↓
020
↓
021
↓
022 / 023
↓
024 / 025
↓
026
↓
027
↓
028
↓
029
↓
030
```

---

# 53. Complexity 标记

建议 Issue 使用：

```text
S
M
L
XL
```

而不是在架构基线中承诺精确人日。

示例：

```text
RX-KAI-002 S
RX-KAI-010 L
RX-KAI-017 L
RX-KAI-019 XL
RX-KAI-026 M
```

---

# 54. 工程规范与 Quality Gate

## 54.1 工程基线

```text
Python 3.12+
type hints 全覆盖
Pydantic v2
pytest
ruff
mypy / pyright
uv
```

不再维护 requirements.txt 作为唯一依赖入口。

## 54.2 PR Quality Gate

所有 PR 至少通过：

```text
lint
type check
unit test
```

核心模块额外要求 regression test：

```text
domain contracts
sampler
leakage guard
label builder
```

sampler 必须包含可复现性回归测试：

```text
固定 input + seed + sampling config
→ raw samples 一致
```

## 54.3 CI 策略

```text
每次 CI：lint / type / unit / leakage
nightly 或手动：smoke benchmark
全量 Benchmark：按需触发，产物入库
```

Leakage 测试属于每次 CI 的强制项，不是可选研究工具。

---

# 55. v2 Definition of Done

Kronos-AI v2 Core 只有满足以下条件才算完成：

1. v1 已冻结并打 Tag；
2. v2 不依赖旧 PredictionService；
3. v2 不依赖旧 ModelManager；
4. Benchmark 不存在静默 synthetic fallback；
5. Market Data 是 point-in-time；
6. Universe 是 point-in-time；
7. 复权策略已版本化；
8. `market_date` 与 `knowledge_cutoff` 已分离；
9. Kronos raw samples 可被暴露；
10. Sampling seed 可从 Request 传入实际 sampler；
11. per-run RNG 已隔离；
12. ForecastResult 记录完整 sampling metadata；
13. Forecast Artifact 可缓存并重放；
14. Benchmark 有 compute budget probe；
15. Kronos 有明确 go / conditional / replace gate；
16. ForecastDistribution 不写死 2% 等阈值；
17. 指标数学定义版本化；
18. MarketState schema / feature / builder 版本分离；
19. Decision baseline 至少包括 Rule / Logistic / LightGBM；
20. Calibration 明确 score space；
21. Policy 支持 Abstention；
22. Typed Decision Backend 通过 capability matrix 后再接入；
23. Walk-forward 是唯一正式 Benchmark 模式；
24. Run 使用 ULID；
25. 所有报告可追溯 git / data / model / config / seed；
26. README 不再展示无可复现实验依据的性能数字；
27. 数据与模型 License / Redistribution 边界已记录；
28. Decision Schema 已版本化且不越界（无 BUY / SELL / 仓位字段）；
29. DecisionResult.answer 是结构化 DecisionAnswer 而非裸字符串；
30. MarketBar / ForecastPoint 时间戳语义已显式定义并纳入 Leakage Guard；
31. Artifact Store 并发写安全（原子写入 + WAL）；
32. PR Quality Gate 与 sampler 可复现性回归测试已建立；
33. Phase 2 Gate 判据已预注册并随 run 归档；
34. Walk-forward 分段间 embargo 已固化于 LabelPolicy 与 Dataset Builder；
35. Decision Benchmark 不含依赖执行假设的交易指标；
36. Signal Validity / Risk Level 为 schema 预留维度，未构造伪标签；
37. Raw provider response 遵循 append-only 快照原则；
38. Phase 0 按 tag v1.0.0 → v2 branch → commit 基线文档顺序完成 git 锚定，且 tag 不含任何 v2 内容。

---

# 56. 最终架构判断

v2 的核心不是：

```text
Kronos + Jev
```

而是：

```text
Point-in-Time Data
↓
Forecast Backend
↓
Raw Stochastic Samples
↓
Forecast Distribution
↓
Market State
↓
Decision Backend
↓
Calibration
↓
Policy
↓
Evaluation
```

Kronos 是：

```text
第一代 Forecast Backend
```

Jev / Kev / AnyJev / Nimble 是：

```text
候选 Typed Decision Backend
```

LightGBM 是：

```text
必须认真对待的强 baseline
```

因此：

```text
Kronos 被替换
```

不推翻平台。

```text
Jev 不如 LightGBM
```

也不推翻平台。

真正需要长期保留的是：

```text
Point-in-Time Data
Contracts
Raw Samples
Reproducibility
Calibration
Abstention
Evaluation
Artifact Lineage
```

这些能力才是 Kronos-AI 从 Demo 变成研究基础设施的核心。

---

# 57. 下一步

正式进入开发前，不再扩大设计范围。

下一步执行：

```text
RX-KAI-001
↓
RX-KAI-010
```

首个里程碑：

> **Kronos-AI v2 Forecast Core**

必须能够完成：

```text
Point-in-Time Market Data
↓
Kronos
↓
Raw 64 Samples
↓
ForecastDistribution
↓
Artifact Cache
↓
CLI
```

并证明：

```text
可重放
可追踪
无随机 fallback
无未来数据泄漏
```

随后再进入：

```text
Forecast Benchmark v1
```

而不是直接进入 Jev 集成。

---

## 附录 A｜首个研究问题

第一阶段只回答：

> **Kronos 在严格 point-in-time、无数据泄漏、无幸存者偏差、可复现采样条件下，对 A 股日线未来 5 个交易 session 是否具有稳定、可重复、具有经济意义的增量预测价值？**

这是 v2 的第一个真正 Gate。

在这个问题没有回答之前，不应扩大到：

```text
Jev
新闻
宏观
Agent
自动交易
```

---

## 附录 B｜第二个研究问题

如果 Forecast Layer 通过 Gate，再回答：

> **Kronos Forecast Distribution + Market State 是否能被简单规则或 LightGBM 有效转化为 Direction / Actionability 决策？**

只有当这个问题被建立后，Typed Decision Model 的比较才有研究意义。

---

## 附录 C｜第三个研究问题

最终再回答：

> **在加入半结构化或非结构化市场上下文后，Typed Probabilistic Decision Model 是否在 Calibration、Coverage、Regime Robustness 或 Semantic Context 利用能力上稳定优于传统 tabular baseline？**

这才是 Jev / Kev / AnyJev / Nimble 真正值得投入的阶段。
