# ADR-023: Forecast Benchmark v1（Runner / 指标 / 报告契约）

- 状态：Accepted
- 日期：2026-09-27
- 对应任务：RX-KAI-019
- 对应基线：`Kronos-AI_v2_Greenfield_Rewrite_设计与落地方案.md` §13、§16、§27、§28、§29、§32.1、§33、§34、§41、§48、§49、§55

## 背景

§41 把 Phase 2 的产出定义为一次 `Forecast Benchmark v1` run：在**同一批** point-in-time
origin 上跑多个 forecast backend，用同一份样本转换与指标代码比较，产物落盘成可追溯
artifact；§42 再据此做 Kronos 的 Go / Replace 判断。

v1 的 benchmark 有四种失败模式，全部是「数字看起来能用，但其实不可解释」：

1. **未来数据进入特征**：预测输入里混入了 origin 之后的 bar（或 label 窗口与输入窗口重叠），
   指标因此系统性偏乐观；
2. **口径分叉**：每个 backend 自己算 MAE / 分位，差异来自两套实现而不是预测本身；
3. **分母不可解释**：报错或数据不足的 origin 被静默跳过，报告不说自己其实只跑了 20%；
4. **事后调参**：跑完再从多个指标里挑一个好看的、或悄悄缩短样本量/窗口。

本 ADR 固化 RX-KAI-019 引入的四个模块边界与它们之间的契约：

```text
evaluation/benchmark.py          run 编排（origin × backend → 记录）
evaluation/forecast_metrics.py   §49 指标口径（唯一实现）
evaluation/regimes.py            point-in-time regime 分组（版本化）
evaluation/report.py             §33 artifact 与人类可读报告
config.py                        §32.1 Experiment Config（一次 run 的唯一描述）
```

## 决策

### 1. 编排与指标分离，未来数据走独立接口（§29）

runner 只做三件事：按 dataset 给定的段遍历 origin、取 PIT history、调用 backend；所有
指标计算与分组都在 `forecast_metrics` / `regimes` 里。

label 窗口的 ground truth 由**单独的** `LabelDataProvider` 提供，而不是复用 §18 的
`MarketDataProvider`：

```python
class LabelDataProvider(Protocol):
    def get_label_bars(
        self, symbol: str, market_date: date, sessions: Sequence[date]
    ) -> tuple[MarketBar, ...]: ...
```

分成两个对象之后，「哪一行代码碰了未来数据」在 runner 里是一眼可见的；把未来 bar 塞进
同一个 provider，泄漏与否就只能靠调用方自觉。实现（`BaoStockLabelBarProvider`）不得为凑齐
`label_sessions` 而合成 bar——缺失本身就是 `SUSPENDED` 的证据。

分段与 embargo **不在 runner 里重新实现**：`WalkForwardDataset`（§27 / ADR-022）给出
`segments`，runner 只按它遍历，因此不可能出现「runner 自己发明的切分」。

### 2. 记录集合与指标集合都必须自洽（§49）

**跨段汇总是一等的切片。** `evaluate_forecast_records` 对每个 backend 同时产出：

```text
segment=None     跨段汇总（整次 run 口径），overall / trend / volatility 三轴
segment=<name>   逐段切片，overall / trend / volatility 三轴
```

没有跨段汇总，报告标题行的「本次 run 的 MAE」只能靠读的人自己加权平均，或者（更糟）把
排序第一的段当成整体——CLI 摘要正是要这个数字。产出顺序固定为
`backend → segment（run 级在前）→ axis`，报告因此可逐行 diff。

`ForecastBenchmarkResult` 在构造时校验这条不变量：每个 backend 必须有且只有一条
`segment=None` 的 overall 指标，且它的 `sample_count` 正好等于该 backend 的记录数。这样
「指标与记录来自同一份证据」是机器可检查的，而不是靠约定。

`metrics_for(backend, segment=None)` 的语义是**精确**的：`segment=None` 取跨段汇总那一条，
不是「任意一段」。

### 2.1 出口也要验身份，pilot 不能缩掉 universe

runner 对每个 origin 做两件事，缺一不可：

```text
入口：require_aligned(history, request)         —— history 是否就是声明的那个研究时点（§17）
       len(history.bars) >= lookback_bars       —— 不足即 InsufficientHistoryError，不靠 backend 自觉
出口：result.symbol/market_date/knowledge_cutoff == origin.*，且 result.model.backend == 注册名
```

出口校验针对的是一类**artifact 里看不出来**的错配：record 的 symbol / market_date 取自
`ForecastResult` 自身，若 backend 交回别的 origin 的预测，runner 会把两条不同时点的证据拼成
一条，而报告照常出数。这与 §17 的 history ↔ request 是同一类错误，只是位置在出口。

pilot 截断（`origin_limit`，§48）按**段内 `(market_date, symbol)`** 顺序取最早的 origin：
若按 symbol 优先，`origin_limit` 小于 symbol 数时整份报告只会覆盖一个 symbol，而 metadata
里只有一个总数。`evaluated_origins_by_symbol` 因此由记录派生并写进 metadata 与报告——
「pilot 有没有悄悄缩掉 universe」是读得出来的。

### 3. 缺料是 `None`，不是 0

无 `LABELED` 记录、相关样本 < 2、任一维方差为 0、分位端点缺失时，指标为 `None`；报告渲染
`n/a` 并同时给出 `sample_count` / `labeled_count`。用 0 冒充「没有证据」会让 §42 的 gate
把「样本不足」读成「表现很差」（ADR-010 在评估侧的实例）。

覆盖率的分母只含**带区间端点**的记录；区间端点必须来自 `ForecastDistribution.quantiles`
里实际存在的分位（由 `DistributionSpec` 请求），backend 不补分位、报告不替它补。

### 4. regime 分组只能用 origin 当时已知的信息（§29 / §49）

`classify_regime` 的输入类型是 `MarketHistory`（已按 `knowledge_cutoff` 截断），它**拿不到**
「未来 bar」这种参数——未来信息在类型层进不来。两个轴独立：

```text
trend       BULL | SIDEWAYS | BEAR     累计 log 收益 vs ±trend_threshold
volatility  HIGH_VOL | LOW_VOL         逐步 log 收益 std vs volatility_threshold
```

不合成单一五值枚举，因为「高波动 + 上涨」与「低波动 + 上涨」在证据上应当可区分。

`RegimeSpec` 版本化，且 runner **拒绝** `lookback_bars < spec.required_bars` 的组合：静默用
更短窗口会让同一个 regime 标签在不同 origin 上对应不同口径。fixture 用短窗口时必须显式
传入自己的 `RegimeSpec`（回归测试即如此），不允许悄悄改默认值。

### 5. 报告不重算指标，缺失不美化（§33 / §49）

`report.md` 的每个数字都来自 `ForecastBenchmarkResult.metrics`（已算好），renderer 不自己
再算一遍——否则报告与 `metrics.json` 就有了两条可能分叉的路径。`segment=None` 渲染成
`all`（不是含义不明的 `-`），`None` 指标渲染成 `n/a`。

落盘走 `ArtifactStore`（原子写 + sha256 进 artifact index）：

```text
report.md / metrics.json / metadata.json / config.yaml / forecast.parquet
```

`report_hash` 由结果内容的规范化 JSON 派生，因此「同一份结论被复现」是可判定的。
**环境事实不进 `report_hash`**：`latency_ms`（墙钟）、`data_coverage_end`（数据发布到哪一天）、
`git_commit`、python / torch 版本（§32 的 runtime 维）都只进 `metrics.json` / `metadata.json`
/ `report.md`。理由是同一条：把它们算进结论身份，同一条命令第二天（或换一台机器）重跑就会
得到不同 `report_hash`，而每一个指标都逐位相同；它们对结论的影响已经逐条写在记录里
（`label_status` / `predicted_*` / `realized_*`），而记录进 hash。
`config.yaml` 原样归档，`config_hash` 由其**语义**派生。

### 6. 唯一的缩小样本量旋钮是 `origin_limit`，且必须显式记录

§48 的 pilot 允许先跑小样本。runner 只接受 `origin_limit` 这一种截断，并把
`considered_origins` / `evaluated_origins` / `truncated` 一起写进结果与 metadata：报告永远
能回答「这份结论是在多少个 origin 上得出的」，而 `truncated` 被校验为「evaluated <
considered」的等价命题，不能是自由字段。

失败语义：backend 名缺失、history 不足、label 越界、backend 抛错一律让整次 run 失败
（ADR-010），runner 不跳过该 origin 继续——跳过会让分母不可解释。

### 7. Experiment Config 是「一次 run」的唯一描述（§32.1）

`configs/benchmark-forecast.yaml` → `ExperimentConfig` 逐字段校验：

- `config_hash` 覆盖全部字段（注释与键顺序不影响，`dataset.symbols` 也做规范化排序，
  书写顺序不换 hash），随 run 归档；
- YAML 里出现 `api_key` / `token` / `secret` 一类键名**直接拒绝加载**，secret 只从环境
  变量读，因此「把 token 写进 config 再归档进 run 目录」在类型层不可能发生；
- `embargo_sessions` / `horizon_sessions` 的单一真源是 `LabelPolicy`：config 里写了与版本化
  定义不一致的值即拒绝运行（要改必须升 LabelPolicy 版本）；
- **复权口径属于数据身份**（`data.adjustment`，§6.1 / ADR-007）：raw / hfq / qfq 改变每一根
  bar，也就改变每个预测与 label。它写在 config 里（进 `config_hash`、随 config.yaml 归档、
  列进报告 Scope），不在命令行 flag 上——flag 形式会让同一份 config.yaml 对应两个结论，
  而 artifact 里没有任何一处能解释 `report_hash` 为什么变了；
- 时间轴必须来自 TradingCalendar（§6.6）：调用方按 `[start_session, end_session]` 切片后，
  session 数必须**恰好**等于 `lookback_bars + Σsegments + embargo × gaps`，否则拒绝运行。

### 7.1 数据覆盖末端与窗口末端是两件事

窗口的最后一个 session 是最后一个 considered origin，它的 label 落在窗口之外。装配因此：

```text
日历装载到「今天」（CN 时区）
data_coverage_end = 其中已发布（<= today）的最后一个 session
要求 data_coverage_end >= 最后一个 origin 的最后 label session
```

否则拒绝运行（「把 dataset.end_session 往后挪」是唯一的修法）。这条守卫挡掉的是「拿一份尾部
horizon 个 origin 没有 label 的报告去比模型」——那是系统性的乐观缺失，不是数据边界。
`data_coverage_end` 进 metadata 与 `metrics.json`，使「这份报告算在覆盖到哪一天的数据上」
永远可指认。

守卫本身是纯函数 `cli.context.resolve_data_coverage(sessions, *, today, end_session,
horizon_sessions) -> date`：只依赖日历装载到的 session 序列与三个标量，因此不变量可以在
合成日历上直接测，而不必起 BaoStock。四条失败分支各自显式失败（均为 `ConfigurationError`，
消息里带上足够的定位信息）：

```text
end_session 不是 market session   → 窗口上界会把实际窗口悄悄提前，config 仍写着它写的那个日期
end_session 在装载区间之外         → 超出 / 早于日历装载覆盖（各自报出自己的成因与修法）
已发布 session 为空                → 日历还没走到 start_session（histories 与 labels 都取不到）
窗口之后不足 horizon 个 session    → 排不出最后一个 origin 的 label 时间轴
data_coverage_end < 最后 label     → 数据还没发布到那里（把 end_session 往回挪）
```

「`end_session` 必须是 market session」这条是抽函数时**特意补回**的：旧的实现借用
`calendar.next_sessions(end_session, horizon)`，顺带继承了它的「非 session 即报错」，改写为
纯函数后这条校验会静默消失——实测把示例窗口末端改成周六（并相应前移 `start_session` 保持
session 数不变）后，真实 CLI run 会返回 0 并把 41 个 origin 全部评分完，而 `dataset_hash`
只能反映「实际切片」，读报告的人无法知道 config 写的上界不是 session。

把「日历覆盖不足」从 `CalendarError` 改为 `ConfigurationError`（两者都是 `KronosAIError`，
CLI 出口一致）：这里失败的原因是**装配选的窗口**超出了日历装载范围，而 `CalendarError`
描述的是日历自己的契约（`next_sessions` 排不出 count 个 session）。ADR-009 §「错误消息前缀
是调用方可依赖的约定」仍由日历自身保证，`next_sessions` 的行为未变；覆盖守卫的消息里
`coverage ends at` / `outside calendar coverage` 只作为**句子的一部分**出现（第 3 条分支的
「published data coverage ends at …」即以 `published data` 开头），不构成 ADR-009 的保留
前缀，类型也不是 `CalendarError`，按前缀判定的调用方不会误判。这条分支也不再要求调用方
「把日历再往后装」——在 `build_benchmark_context` 里日历已经装到 `max(end_session, today)`，
再往后装既不必要（会落到下一条分支、给出同样的拒绝）也不是这里的修法。

`end_session` 落在装载区间之外时另有两个成因分支，各自给出可执行的定位信息：超出装载覆盖
（`beyond the loaded calendar coverage`）与早于装载覆盖（`precedes the loaded calendar
coverage`）——它们都不是「非 session」，混成一句话会把事实说错。

### 7.2 CLI stdout 只放结果，不放第三方库的进度打印

baostock 的 `login()` / `logout()` / 查询会直接 `print("login success!")`。CLI 的 `--json`
输出同样在 stdout，两者混在一起时 `json.loads(stdout)` 直接失败（实测）。provider 因此在
login / logout / 单次 query 的调用点把第三方库的 stdout 重定向到 stderr：诊断信息保留，
机器可读输出干净。这几处都在会话锁内，重定向不会与并发输出交错。

### 8. 显式 defer

- **CRPS**：§49 已列入，但需要一个版本化的 sample-based CRPS 定义（结的处理、区间积分
  口径）。在定义之前产出 CRPS 数字等于产出不可解释的指标，因此登记在
  `DEFERRED_FORECAST_EVAL_METRICS` 并缺席本期。
- **§49 Compute 的部分项**：`cache_hit_ratio`（需要 §15 缓存层暴露 run 级命中计数）、
  `ram_peak` / `vram_peak` / `artifact_bytes_per_forecast_origin`（属 §16 Cost Probe 的测量
  口径，RX-KAI-018）不在本报告产出，登记在 `DEFERRED_BENCHMARK_COMPUTE_METRICS`；报告只
  记录本次 run 自己观测到的延迟分位。
- **universe → symbols 的解析与 `universe.parquet`**：`universe` 段当前只是声明（`source` /
  `snapshot` 被校验、进 `config_hash`），生效名单是 `dataset.symbols`。用
  `BaoStockUniverseLoader` 把 `source` 解析成 PIT 快照、并对 `dataset.symbols` 做「是否属于
  该快照」的校验，属后续任务（§6.2 / §33）。在那之前不要把 `universe` 读成「名单已被校验」。
- **§19 的 Go / Replace Gate**：属于 RX-KAI-020。判据必须在正式 run 之前预注册并随 run_id
  归档（§42），本 ADR 只保证 gate 需要的输入（逐 backend、逐 regime、带置信区间的增量）
  都能从一次 run 的 artifact 里拿到。

## 后果

- 一次 benchmark run 的结论可以用 `report_hash` 指认，用 `dataset_hash` / `config_hash` /
  `git_commit` 追溯；「数字是怎么算出来的」有唯一实现（`forecast_metrics`）。
- CLI（`kronos-ai benchmark forecast --config ...`）与 API 层将来共用同一条编排路径：
  差异只在装配（`BenchmarkContext`），不在执行。
- 报告不阻止「结论只在一个 regime 成立」：跨段与逐段、三个轴都摆出来，读的人能看见样本量
  （`n` / `labeled`），而不是只看到一个标题数字。
- 指标口径（`FORECAST_EVAL_METRICS_VERSION`）、coverage 口径（`COVERAGE_SPEC_VERSION`）、
  regime 口径（`regime_spec_hash`）任一改变都会换掉 `report_hash`，历史报告不会被悄悄重解释。
- 环境事实（延迟、覆盖末端、git、python / torch）可追溯但不参与 `report_hash`：同一条命令
  的两次运行给出同一个 `report_hash`，是「复现」这件事在真实路径上可验证的含义。
- runner 在入口（history ↔ request ↔ lookback）与出口（result ↔ origin ↔ backend 名）两侧
  都验身份，因此「预测与 label 错配」这类 artifact 里看不出来的错误无法产出报告。
- CRPS、部分 §49 Compute 项、gate 判据、universe 解析的缺失是**显式**的：前三者在代码里
  登记（`DEFERRED_*`），universe 在 ADR 与 config 注释里登记，不会被读成「已经算过、只是没写」。
