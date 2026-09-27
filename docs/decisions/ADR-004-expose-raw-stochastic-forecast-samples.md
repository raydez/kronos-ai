# ADR-004: Expose Raw Stochastic Forecast Samples from Kronos

- 状态：Accepted
- 日期：2026-09-27
- 对应任务：RX-KAI-010（截取路径与受控 adapter）；消费契约见 RX-KAI-012
- 对应基线：`Kronos-AI_v2_Greenfield_Rewrite_设计与落地方案.md` §9、§10、§16、§36（ADR-004）

## 背景

Kronos 上游的多路径采样只在内部存在：`auto_regressive_inference`
（`vendor/kronos.py:386`）先 `repeat_interleave(sample_count)` 复制输入，做随机自回归
decode，最后 `z.reshape(batch, sample_count, seq, feat).mean(1)`——**mean 是唯一出口**。
v1 的用法正是「sample_count=64 → 调用原生 predict → 拿 mean 当结论」，采样维信息在
出模型前就被抹掉，平台因此无法回答「这条预测的分布有多宽」这类问题；而 v2 的概率化
决策、calibration（§26）与 abstention（§27）全部以分布为输入。

§10 把「在 mean 之前截取 raw samples」定为 v2 的 P0 设计，并明确禁止
「拿 mean prediction 伪装成 distribution」。本 ADR 固化截取点的精确定义、
v2 接口形态与「截取不改变随机路径」的证明义务。

## 决策

### 1. 截取点：sample 维 reshape 之后、mean 之前

```text
tokenizer.decode(...) → z
↓
z.reshape(-1, sample_count, window_seq_len, feature_count)   ← 此处截取
↓
（上游：.mean(1) → 逐 step 只保留均值）
```

v2 返回 `(sample_count, horizon, feature_count)` 的**全部**原始路径，不做任何跨样本
聚合。`raw[0, :, -horizon:, :]`：第一维恒为 1（单 symbol，见 §5），末维取未来段
（decode 窗口覆盖 history + future，历史段不属于预测输出）。

### 2. 受控 adapter，不改 vendor

实现位于 `src/kronos_ai/forecast/backends/kronos/sampler.py`（`KronosSampler`），
不在 `vendor/` 内改动上游代码（ADR-020）。自回归循环按上游语义重建：缓冲与 roll、
stamp 拼接、`top_k/top_p/temperature` 过滤顺序、`encode/decode(half=True)` 均逐点对齐；
差异共三处，均为显式设计：

```text
1. 截取点（本 ADR）
2. per-run generator 取代全局 RNG（ADR-005）
3. 窗口归一化统计量与输出走 float64（上游 predict() 在 float32 上算 mean/std，
   vendor/kronos.py:541-544）；实测相对偏差 ~2e-9
```

另有一条**不是差异**的说明：vendored commit（`67b630e6…`）的 `auto_regressive_inference`
中本就没有逐 step `torch.cuda.empty_cache()` 调用，v2 保持原样、不重新引入——逐 step
驱逐缓存会把时间花在缓存重建上，而 §16 关注的是 `batch × sample_count` 的显存相乘
（属 RX-KAI-018 Cost Probe 的输入）。

### 3. 输出契约：`RawSampleSet`

```text
values         (sample_count, horizon, feature_count)，float64，价格量纲，只读
future_sessions 来自 TradingCalendar.next_sessions（§11），长度 == horizon
```

- 价格量纲：归一化统计量取自 lookback 窗口本身（`(x - mean) / (std + 1e-5)`，
  与上游 `predict()` 一致，`vendor/kronos.py:543`），截取后按窗口统计量反归一化。
  下游拿到的是元/股，不是 z-score。
- `float64`：批量采样路径上 mean 会放大 float32 的累积误差；raw 样本是上游数值的
  最后一份可信副本，输出即提升精度（截取处先 `.to(torch.float64)`）。
- 只读 + 有限性校验：样本集不可被下游就地改写；非有限值属模型输出损坏，显式失败
  （§3.2），不进入指标。
- 窗口内统计量与 `horizon <= max_context` 为硬前置：decode 窗口只含最后
  `max_context` 个 token，horizon 超窗会静默错位（截断 + 与 `future_sessions` 对不上），
  因此在采样前显式拒绝。

### 4. 接口落点与命名

§10 推荐的接口是 `KronosSampler.generate_samples(history, request) -> list[ForecastSample]`。
RX-KAI-010 先交付其前置：`KronosSampler.decode_raw_samples(history, request) -> RawSampleSet`；
RX-KAI-012 已在其上落地 §10 形态的 `generate_samples`（对 raw 样本做纯转换，
不再触碰采样路径，见下节 §7）。

### 5. 单 symbol 是构造性约束

输入 batch 维固定为 1（单 `MarketHistory`），`sample_count` 通过
`unsqueeze(1).repeat(1, sample_count, ...)` 展开——因此 `sample_count` 与 batch 在
前向张量里相乘（§16）。截取处对 reshape 结果做恒等断言（第一维必须为 1），
batch>1 的批量采样 API 落地前必须显式改造此处，不允许静默只返回第一个 symbol。

### 6. 等价性证明义务（可执行）

「截取不改变随机路径」不能只靠代码审阅。回归测试
（`tests/regression/test_sampler_equivalence.py`）在相同 seed、相同 eval 模块、
相同 max_context 下独立重建上游输入，断言：

```text
mean(v2 raw samples, axis=sample) == auto_regressive_inference(...) 的输出
```

边界覆盖四组（`lookback`/`max_context`/`horizon`）：

```text
12/512/4   常规：窗口不截断
12/12/4    lookback == max_context：buffer roll + decode 窗口截断到 12 token
4/4/4      horizon == 窗口长度：decode 窗口只含未来段（无历史上下文）
16/512/60  horizon 远大于 lookback：解码窗口整体长于输入窗口
```

上游不暴露 raw samples，因此该等价性是「截取点正确」最直接的回归证据（§54.2 / DoD 32）。
容差 rtol=1e-7 的依据：v2 在 float64 上累积 mean、上游在 float32，实测固有相对偏差
约 2e-9（留 30× 余量）；该容差仍有判别力——把归一化常数从 1e-5 改成 1e-4 即全组变红。

## 后果

- 下游（Distribution / Calibration / Label）以原始分布为输入，mean 只能作为
  显式聚合出现在分布对象上，不会在数据通路上「伪装成分布」。
- 采样路径无法改动而不失败：`mean(v2 raw)` 必须仍等于上游输出；修改归一化、
  截取点或循环结构都会让回归变红（容差已按实测收紧，见 §6）。
- 单条 `MarketHistory` 的采样成本 = `sample_count` 倍前向量，成本曲线由 RX-KAI-018
  实测后进入预算约束（§16）。
- 数值等价性以 eval 模式为前提（`vendor/module.py:387` 的
  `is_causal_flag = self.training` 改变 s2 注意力掩码）：上游 canonical 用法同样是
  eval（`from_pretrained` 末尾调用 `.eval()`），KronosRuntime 强制 eval（RX-KAI-009）。

## 7. RX-KAI-012 落地：ForecastSample / ForecastDistribution 与版本化 metric registry

§11–§14 的消费契约已落地，落点与关键决策：

- **契约**（`domain/forecast.py`）：`ForecastPoint` / `ForecastSample`（§11）、
  `ThresholdProbability` / `QuantileValue` / `ForecastDistribution`（§12）、
  `SamplingMetadata` / `ForecastResult`（§14）。`ForecastSample.points` 用 tuple 而非
  list：frozen 模型 + tuple 才真正不可变（list 字段在 frozen 下仍可原地改写）。
- **raw → domain 转换 + 分布构建**（`forecast/distribution.py`）：
  `forecast_samples_from_raw` 按 feature 名索引把 `RawSampleSet` 转成 `ForecastSample`，
  时间轴直接采用 `RawSampleSet.future_sessions`（个股停牌不改变时间轴，§11）；
  `build_distribution(raw, origin_close=…)` 计算 §13 指标并组装 §12 分布对象。
  该模块按结构消费 `RawSampleSet`（类型仅 TYPE_CHECKING 引入），避免
  sampler ↔ distribution 运行时循环 import。
- **§10 接口**：`KronosSampler.generate_samples` 现在返回
  `list[ForecastSample]`，实现是 `list(forecast_samples_from_raw(decode_raw_samples(...)))`，
  采样路径零改动。
- **§12 移除写死 2%**：阈值/分位由版本化 `DistributionSpec` 显式给出
  （`DEFAULT_DISTRIBUTION_SPEC` 只作为默认值，不再是 schema 常量）。
  `ForecastDistribution` 内不含任何写死的阈值字段。
- **§12/§13/DoD17 metric registry 版本化**：`FORECAST_METRIC_REGISTRY_VERSION`
  （`forecast-metrics-v1`）+ `_FORECAST_METRICS` 定义 `horizon_return` / `log_return` /
  `max_drawdown` / `path_volatility`。`ThresholdProbability.metric` 与
  `QuantileValue.metric` 经 `AfterValidator` 强制来自 registry，backend 不能自造名称；
  `ForecastDistribution.metric_definition_version` 经白名单
  （`SUPPORTED_FORECAST_METRIC_VERSIONS`）校验——升级时把旧版本号追加进白名单，
  删除会让历史 artifact 无法反序列化。新增/改数学定义必须升版本。
- **spec 版本可追溯**：`ForecastDistribution.distribution_spec_version` 记录构建时
  使用的 `DistributionSpec.version`（默认 `distribution-spec-v1`），使默认阈值变更
  在产物上可见（也是 RX-KAI-013 cache key 的输入之一）。
- **provenance 单点一致**：`ForecastResult` 交叉校验
  `sampling.sample_count == distribution.sample_count`——同一 run 的两个记录互相矛盾
  属 provenance 缺陷（DoD12），必须在构造时显式失败。
- **分位插值显式化**：`np.quantile(..., method="linear")` 写死，不依赖 numpy 默认值。
- **数学定义显式化**（§13）：收益类指标 `P_0 = forecast origin close`；`max_drawdown`
  与 `path_volatility` 的 close path 含 `P_0`；离散度与波动率用总体标准差（ddof=0，
  样本集即完整经验分布）。`P_0` 或任一 close 非正/非有限 → `ModelInferenceError`（§3.2）。
- **`generate_samples` 不消费 `origin_close`**：分布是独立步骤（需要 `P_0`），
  由调用方显式提供；采样层不持有「分布」语义。
