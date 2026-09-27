# ADR-019: Benchmark Compute Budget and Pilot Gate

- 状态：Accepted
- 日期：2026-09-26
- 对应任务：RX-KAI-018
- 对应基线：`Kronos-AI_v2_Greenfield_Rewrite_设计与落地方案.md` §16、§41、§48、§49、§55.14

## 背景

§16 把「全量 benchmark 跑得完吗」定义成一个必须先回答的问题：正式全量 run 之前先做
Cost Probe，再由 probe 结果确定 full universe / full period / sample_count / hardware
requirement。这个顺序不能反——先跑全量再看墙钟，等于把机器资源当成测量工具。

v1 的三类失败模式必须在类型层就被挡住：

1. **组合爆炸**：把「设备 × sample_count × batch × universe × period」当笛卡尔积扫。
   §16 明确禁止，并把 probe 总组数上限钉在 10；
2. **无上限运行**：probe 本身没有 wall-clock / 内存上限，「探针」变成一次没有边界的
   全量实验；
3. **静默降级**：拿一个被截断或被跳过组合的运行去算 p50 与 throughput，报告读起来像
   「全部可行」，实际上这台机器跑不动（ADR-010 在 benchmark 侧的实例）。

还有一个纯工程问题：cost probe 面向的参照系（§48 的 baseline）不需要 torch。如果
probe 引擎必须 import torch 才能测「时间与内存」，那么参照系就被拖回了 GPU 依赖。

## 决策

### 1. 引擎 / 口径 / 报告三层，`evaluation/compute_metrics.py` 单点落地

```text
执行引擎     run_cost_probe：逐组跑、查预算、中止超限组合
指标口径     percentile / LatencySummary / throughput / ResourcePeak
报告         ComputeBudgetReport（完整性校验 + report_hash + throughput-memory curve）
```

§49 的指标清单因此都有落点：p50 / p95 / mean / min / max 与 forecast/sec 在
``LatencySummary`` + ``throughput_per_second``，RAM / VRAM peak 在 ``ResourcePeak``，
artifact bytes per forecast origin 与 cache hit ratio 在 ``ProbeCellResult``；§16 的
throughput-memory curve 是一等的 ``ComputeBudgetReport.throughput_memory_curve``
（只含 MEASURED 组合、按 ``forward_width`` 升序），而不是让调用方自己拼。

模块不 import torch / pandas / provider。观测由调用方注入
（``MeasureCallable``），资源由 ``ResourceSampler`` 注入。这样 probe 引擎可以在 CI 里
用假 measure 与假时钟做确定性测试，而真实接线（provider → MarketHistory →
backend.forecast）留在 RX-KAI-019：本任务先把「预算与统计口径」本身做对。

### 2. 禁止全因子笛卡尔积（§16）

- **设备是标量不是轴**：``PilotMatrix.device_class`` 是单值，多设备各自建一个单点
  matrix 验证（§16「先锁定目标设备 / 其他设备仅跑单点验证」）。把它写成 tuple 会被
  pydantic 拒绝，因此「扫设备」在类型层不可能。
- 扫描轴只有 ``sample_counts`` 与 ``batch_sizes``，且
  ``len(sample_counts) × len(batch_sizes) <= MAX_PROBE_CELLS``（= 10）。
  超限抛 ``ValidationError`` 并提示裁剪，**不提供**「悄悄截断到 10 组」的降级：
  5 个 sample_count × 2 个 batch = 10 组是允许的，4 × 3 = 12 必须由人来决定砍哪一维。
- 组合顺序规范化为 ``sample_count`` 外层、``batch_size`` 内层升序，报告因此可逐行
  diff。
- 组数上限是**报告级**不变量：``ComputeBudgetReport`` 在核对 ``cells`` 覆盖之前先复核
  ``len(matrix.cells) <= MAX_PROBE_CELLS``，因为 ``model_construct`` 之类的绕过路径
  可以跳过 ``PilotMatrix`` 的 validator（组数上限不只是 matrix 的属性）。
- ``batch_size`` 是 §16 的 batch 维度（一次前向提交多少个 origin），
  ``ProbeCell.forward_width = batch × sample_count`` 把「内存随两者相乘增长」写成
  可断言的对象，而不是文档里的一句话。

### 3. 不允许无上限运行（§16）

``ProbeBudget`` 的两个字段都没有默认值：不声明 wall-clock 与内存上限就构造不出
budget，也就跑不了 probe。上限的**具体数值**由调用方决定（§16 的示例是每组 4 小时），
不硬编码在此。

- 每次迭代**前**检查 wall-clock；
- 每次迭代**后**检查峰值内存；
- 循环结束、构造 MEASURED 结果**之前**再复核一次总耗时。单次 ``measure`` 调用无法
  被抢占（进程内无解），所以末轮把预算撑爆只能事后发现；若不复核，``repeats_per_cell=1``
  的 3600s 组合就会被报成 MEASURED，与「任一组合超限即标 INFEASIBLE」直接矛盾。
- 触发即中止该组合。

内存预算必须可被观测：采样器报不出**宿主 RAM** 时抛 ``ConfigurationError``，拒绝运行
（vram-only 不够，理由见 §7）。否则「预算守住了」的证据会是「什么都没测到」。

### 4. 超限不是「跑完了」：`INFEASIBLE_ON_THIS_DEVICE`

``ProbeCellStatus`` 只有两个取值：``MEASURED`` 与 ``INFEASIBLE_ON_THIS_DEVICE``
（§16 的标记）。没有 ``SKIPPED``——「跳过」正是把不可行结论藏起来的那类状态。
model validator 强制两者互斥且完备：

- ``MEASURED`` 一定带 ``latency`` 与 ``throughput_per_second``，且没有 ``abort_reason``；
- ``INFEASIBLE_ON_THIS_DEVICE`` 一定带 ``abort_reason``，且**不带**任何从被截断运行
  算出来的指标（``latency`` / ``throughput`` / ``artifact_bytes_per_forecast`` /
  ``cache_hit_ratio`` 全为 ``None``）。

第二条防线是报告级的：``ComputeBudgetReport`` 要求每个 ``MEASURED`` 单元的
``completed_repeats`` **正好等于** ``repeats_per_cell``，且 ``wall_clock_seconds`` 不得
超出该组合的 wall-clock 预算。没有它们，手工把一个只跑了 2/10 次的结果（外带由那 2 次
算出的 p50）、或一个跑满但超预算的单元拼进报告就能通过全部校验——那正是本节声称
不可能的路径。由 pydantic 的 ``model_copy`` / ``model_construct`` 造出的对象不走字段
校验，所以这类跨对象不变量只能在报告层断言。

被中止的组合仍然保留 ``completed_repeats`` 与内存峰值：它们是「为什么不可行」的证据。

### 5. 报告必须完整覆盖声明的 matrix

``ComputeBudgetReport`` 的 model validator 要求 ``cells`` 与 ``PilotMatrix.cells``
逐一相同。手工删掉一个 INFEASIBLE 组合、拼一份「全部可行」的报告会被拒绝——§16 存在
的意义就是让「这台机器跑不动」这个结论无法被安静地抹掉。

``PilotMatrix`` 同时承载全量 run 的规模（``symbols`` × ``forecast_origins``），因为
probe 的产出之一就是外推：§16 要求由 probe 决定 full universe / period。

### 6. 指标口径版本化、可追溯

- ``COMPUTE_METRICS_VERSION`` = ``compute-metrics-v1``，进 ``report_hash``
  （``kind=compute_budget_report``）：换了指标数学与换了机器都会产生不同的 hash。
- §49 的 p50 / p95 用**最近秩法**：``ceil(q·n)`` 处那个真实观测。理由写在
  ``percentile`` 的 docstring 里——p95 会被直接用来判断「跑不跑得完」，能被指到某一次
  具体运行比平滑更重要。为此不用线性插值。
- ``throughput_per_second`` 的分母是**迭代延迟之和**（不含资源采样与预算检查开销），
  分子是本次迭代完成的 forecast 数（``batch_size`` 个 origin 即算 ``batch_size`` 次）。
  两个数字的对应关系写在类型上，而不是靠读代码猜。
- ``artifact_bytes_per_forecast`` 的分母是**报告了产物的那些迭代所完成的 forecast 数**，
  即口径就是字段名：artifact bytes / forecast origin（§49）。用「有产物的迭代次数」
  做分母会在 ``batch_size > 1`` 时静默偏大 ``batch_size`` 倍。
- ``cache_hit_ratio`` / ``artifact_bytes_per_forecast`` 在观测缺失时为 ``None``，
  不是 0：没有落盘 / 没有缓存层的组合不该在报告里显示「0 字节产物」。
- ``LatencySummary`` 的自洽约束覆盖 ``mean``：``min ≤ {p50, p95, mean} ≤ max``。均值
  落在极值区间外不是「奇怪但可接受」，而是样本集合不一致。
- ``project_full_run_seconds`` 是**外推**而非测量，只接受 ``MEASURED`` 的组合，并且
  要求调用方显式给出全量 forecast 数与 worker 并行度。假设（吞吐不随规模变化、并行
  效率不超过 1）写在 docstring 里，是给人看的可证伪前提。

### 7. 资源峰值口径

默认采样器 ``ProcessResourceSampler`` 用 ``getrusage(RUSAGE_SELF).ru_maxrss`` 读本进程
RSS 峰值：macOS 单位是 bytes，Linux 是 kibibytes，本模块按平台换算（ADR-019 记录这一
实测口径；未知平台按 Linux 口径处理，数值更大，更保守）。

这里有一个必须写明的局限：``ru_maxrss`` 是**进程生命周期的高水位线**，单调不降。
它有两个后果：

1. 每个组合读到的值是「到该组合结束为止的进程峰值」，**不会低估**内存占用；
2. 一旦某个组合把进程峰值推过预算，其后每个组合在基线采样处就已超限、首轮即中止
   （级联 ``INFEASIBLE_ON_THIS_DEVICE``）——这是刻意行为：进程确实到过那个水位。

所以需要**按组合归因**的内存维度（§16 throughput-memory curve）时，调用方应注入每次
调用返回**当前**占用的采样器（RX-KAI-019 接 torch 侧统计）；``_PeakTracker`` 对当前值
采样同样成立。

VRAM 由调用方以 ``vram_sampler`` 注入，未注入即 ``None``——纯 CPU 环境上「VRAM 不可
观测」是正确结论，不是数据缺失，更不是 0。反过来，**RAM 必须可观测**：只有一个内存
上限时，观测不到宿主 RAM 就等于对 RAM 无界，``all_feasible`` 会变成一句空话。采样器
报不出 RAM 时该组合拒绝运行（``ConfigurationError``，ADR-010）；vram-only 不够。

## 后果

- Cost Probe 先于全量 benchmark 成为**结构性**约束：报告里任何一个
  ``INFEASIBLE_ON_THIS_DEVICE`` 组合都让 ``all_feasible`` 为假，全量 run 的
  sample_count / batch / universe 只能从 ``MEASURED`` 组合里选。
- 报告与 run 一起落盘（§30 / §33，RX-KAI-019 接 Artifact Store），因此「这批结论是在
  哪台机器上、用什么预算、哪一版指标数学得出的」可由 ``report_hash`` 回答。
- 由于设备与预算只影响 compute 结论、不影响预测内容，**probe 组合不进 Run Metadata 的
  预测身份维度**：它是执行条件，不是 forecast 定义的一部分。
- 本任务不接 CLI / 配置文件：probe 需要真实 provider 接线才有意义，而那部分属
  RX-KAI-019 的 benchmark runner（§34 的 `kronos-ai benchmark forecast`）。在此之前，
  引擎由单元测试与子进程用例保证正确。
