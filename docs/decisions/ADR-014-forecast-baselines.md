# ADR-014: Forecast Naive Baselines

- 状态：Accepted
- 日期：2026-09-26
- 对应任务：RX-KAI-016
- 对应基线：`Kronos-AI_v2_Greenfield_Rewrite_设计与落地方案.md` §4、§13、§17、§19、§41、§48、§49

## 背景

§19 把 Forecast Phase 2 定义为 Go / No-Go Gate：判断 Kronos 是否值得继续作为默认
Forecast Backend，参照系是 Last Value / Drift / Moving Average / 简单 Statistical
Baseline；§42 进一步要求 gate 判据预注册、可证伪。没有 baseline 的「模型有用」不可
证伪：既不知道增量大小，也不知道增量的方向在哪些 regime 下反转。

工程上还有两个必须先解决的问题：

1. 参照系必须能**在廉价环境反复运行**（§48 的 pilot matrix 要有 100–300 个 origin）。
   如果 baseline 与主模型共用一条必须加载 torch 的代码路径，参照系就失去了「廉价」
   的意义，也会把 GPU 依赖带进 benchmark 的每一个 worker；
2. 参照系与主模型必须走**同一份样本转换与指标代码**。若 baseline 自己实现 MAE/分位，
   那么 benchmark 中出现的差异可能来自两套口径分叉，而不是预测本身——这类差异无法
   归因，gate 也就无法据此决策。

## 决策

### 1. 位置与依赖方向（§4）

按 §4 终态目录，baseline 放在 `src/kronos_ai/evaluation/baselines.py`（新增
`kronos_ai.evaluation` 包）。依赖方向固定为 `evaluation` → `forecast` / `data` /
`domain`：baseline 复用推理层契约与缓存键，推理核心不反向依赖任何评估代码。

### 2. 实现 §17 ForecastBackend，四个 baseline 一份实现

- `LastValueBaseline`（`last_value`）：`P_t = P_0`。
- `DriftBaseline`（`drift`）：`d = (ln P_N − ln P_1)/(N − 1)`，`P_t = P_N·exp(d·t)`。
- `MovingAverageBaseline`（`moving_average`）：末 `window` 根 close 的算术均值（默认
  `DEFAULT_MOVING_AVERAGE_WINDOW = 20`），全 horizon 常数路径。
- `Ar1Baseline`（`ar1`）：窗口内 OLS 拟合对数收益 `r_t = c + φ·r_{t−1}`，以最后观测
  收益为状态起点迭代外推，再累积回价格空间。

`BASELINE_NAMES` 是稳定顺序的唯一真源（benchmark 配置校验、报告列顺序都取它），
`build_baseline` / `build_baseline_backends` 按名构造，未知名字显式失败（§3.2）。

### 3. 退化分布：确定性点预测的编码方式

四个 baseline 都不生成随机路径。它们把**同一条确定性路径重复 `sample_count` 次**
编码成一个 `RawSampleSet`，因此结果满足 §14 契约（`samples` 条数 ==
`distribution.sample_count`）并复用 §13 指标与 §15 缓存键，无需任何 baseline 特判。

代价必须写明：跨样本离散度为 0 ⇒ `forecast_dispersion = 0`、分位全等、阈值概率 ∈
{0, 1}、CRPS 退化为 MAE 的单调变换。因此 **quantile coverage / CRPS 对这 4 个
baseline 不构成证据**，§49 的判据要以 MAE / RMSE / Direction Accuracy / Return
Correlation 为主。这里不为了让分布「看起来像分布」而注入人为噪声（ADR-010）：那会
让 baseline 的指标变成噪声参数的函数，gate 判据将不可复现。

### 4. 共用 raw sample 与指标路径

`RawSampleSet` 从 `forecast/backends/kronos/sampler.py` 上移到新的
`forecast/raw.py`（sampler 保留 re-export，行为不变）。理由：它是「sample 维 reshape
之后、mean 之前」的 forecast 层概念（§10），不是 kronos 概念；放在 forecast 层后，
baseline 可以在**不 import torch** 的前提下复用
`forecast_samples_from_raw` → `build_distribution`。

同理，history ↔ request 的一致性前置条件从 `KronosSampler._require_aligned` 上移到
`forecast/base.py::require_aligned`，成为所有 backend 共用的单点校验。

### 5. 身份与版本（§15）

- `model_id = "baseline:<name>"`；`backend = <name>`；`model_revision =
  BASELINE_MATH_VERSION`——对 baseline 而言「模型」就是公式，公式版本就是模型版本。
- `config_hash` 覆盖 `lookback_bars`、特征集与 baseline 参数（如 MA 窗口）：窗口从
  20 改成 10 必须产出不同 `artifact_id`，否则两个不同模型会在缓存里互相冒充。
- `runtime_version = f"numpy-{numpy.__version__}"`，`device_class = "cpu"`，
  `dtype = "float64"`。
- backend 名不进 artifact key（沿用 §15/ADR-012 规则：身份由 key 的既有维度决定）。
- `ModelMetadata` 与 artifact key 的字段名差异（`revision` vs `model_revision`、
  `device` vs `device_class`）在 `Baseline.model_metadata()` 单点映射，避免两处拼装分叉。

### 6. 历史窗口与显式失败（§6.4）

- 估计窗口 = history 的末 `lookback_bars` 根 bar，**与 kronos 同一口径：必须凑满**，
  否则抛 `InsufficientHistoryError`（消息与 `KronosSampler._lookback_window` 逐字一致）。
  这是从 §19/§42 推导出的前置条件，不是 §19 的原话：§19 只把 gate 定义为「与 naive
  baseline 比较」，而要**预注册、可证伪**地比较两个 backend，必须先固定「同一窗口、
  同一信息量」——否则测得的「增量」会混入可用数据量的差异。历史不足的 origin 必须由
  dataset builder（§27）剔除：**RX-KAI-017 必须保证每个 origin 都有
  `len(bars) >= lookback_bars`**，而不是让每个 backend 各自静默降级。
- `min_estimable_bars`（**不叫** `min_history_bars`）是**配置**下界（估计量可识别性）：
  `lookback_bars < min_estimable_bars` 在构造期即拒绝（AR(1) 需 ≥2 个
  `(r_{t−1}, r_t)` 对；MA 需完整 `window`）。它与 §6.4 / provider 侧的
  `min_history_bars` 不是同一个量（后者是「运行期 history 够不够用」，这里由
  `lookback_bars` 承担），命名分开以免两套语义互相冒充。
- `moving_average` 的 `window > lookback_bars` 在构造期即拒绝（窗口永远无法满足）；
  `moving_average_window < 1` 在 `build_baseline` 中**无条件**校验，即使当前 backend
  不是 MA 也不放过（配置里的非法值不应因「参数用不到」而静默通过）。
- AR(1) 在窗口内滞后收益方差为 0 时抛 `ModelInferenceError`（斜率不可识别），不静默
  把 φ 置 0；`|φ| > 1` 的非平稳窗口不做截断，但路径非有限/非正即失败。
- baseline 接受并记录 `seed` / `temperature` / `top_k` / `top_p`（接口一致），但不参与
  计算：同输入必须逐位重现同输出（见 `tests/unit/test_evaluation_baselines.py`）。

### 7. 缓存

缓存是可选注入（`cache=`），语义与 kronos backend 完全一致：`force=True` 只绕过读
缓存，写入仍原子（§15）。baseline 本身确定性，缓存只省算力、不改变结果——这一点由
「缓存前后 `ForecastResult` 相等」的用例守住。

### 8. 本期不做的（显式 defer）

- **CLI `--backend` 接线**：`kronos-ai forecast` 目前无条件构造 Kronos runtime；让
  baseline 可被 CLI 选择需要把 runtime 构造改成按名惰性，属 RX-KAI-019 的 benchmark
  runner 改造范围（benchmark 需要按配置选后端，并避免为廉价 baseline 加载模型）。
  本任务只保证 baseline 可被 `RuntimeRegistry` 与 `ForecastService` 直接驱动。
- **bootstrap 置信区间与 gate 判据**：属 §42 的 Go/Replace 决策任务（RX-KAI-020）。
- **指标聚合（MAE/RMSE/regime 分组）**：属 §49 的 benchmark 任务（RX-KAI-019）；本任务
  只产出可被其消费的 `ForecastResult`。

## 后果

- Go / Replace Gate 有了同接口、同指标、同缓存语义的可证伪参照系；benchmark 的差异
  只可能来自预测，不来自口径。
- baseline 可在无 torch / 无 pandas 的进程中**完整跑完一次 forecast**（含缓存写入与
  命中重放，有子进程用例守住），benchmark 的参照系不必占用 GPU，也不需要为每个
  worker 加载模型权重。
- 新增两处跨模块搬运（`forecast/raw.py`、`forecast/base.py::require_aligned`），
  都是「把 backend 私有概念提到 forecast 层」的一次性成本，避免 baseline 复制样本与
  校验逻辑。注：`forecast/raw.py` 尚未补进 §4 的终态目录清单，属设计文档漂移，
  待下次该章节修订时一并补上。
- 身份治理有 golden 钉子：`config_hash` 的取值（含 `baseline-math-v1` / 特征集 / 参数
  载荷）与 `DEFAULT_MOVING_AVERAGE_WINDOW` 都被 golden 用例钉住，静默修改会让测试
  失败，而不是让不一致的历史 artifact 被重新解释。
- 版本治理：`BASELINE_MATH_VERSION` 或任一 baseline 参数语义变化都必须递增/进
  `config_hash`，否则历史 artifact 会被静默重新解释。
