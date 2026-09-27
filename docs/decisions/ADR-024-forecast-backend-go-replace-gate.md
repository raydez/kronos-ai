# ADR-024: Forecast Backend Go / Replace Gate

- 状态：Accepted
- 日期：2026-09-27
- 对应任务：RX-KAI-020
- 对应基线：`Kronos-AI_v2_Greenfield_Rewrite_设计与落地方案.md` §16、§19、§41、§42、§48、§49

## 背景

§19 的问题不是「Kronos 是否有效」，而是「Kronos 是否值得继续作为默认 Forecast Backend」；
§42 把回答形式固定下来，并给出三条硬约束：

```text
判据必须在任何正式 benchmark run 之前预注册，随 run_id 一起归档
判据必须可证伪（指标、比较对象、分组方式、置信区间都要说清）
禁止跑完后从多个指标里挑选有利指标、事后放宽阈值
```

一个「先跑再解释」的 gate 有四种失败模式，每一种都让判决失去意义：

1. **阈值事后定**：看到结果再决定「多少算好」，判据就不再能证伪任何东西；
2. **单点数字**：只看一个 MAE / accuracy 差值，无法区分「真的更好」与「样本噪声」；
3. **分组说了不算**：整体指标可能由单一 regime 或单一 symbol 主导，「在哪些市场状态下有效」
   根本无从回答（§49 的 Robustness 维）；
4. **代价不记账**：一个只在 compute 上不可用的 backend 仍然会被判「值得当默认」，
   而 §19 明确把 compute cost 列为判断维度之一。

本 ADR 固化 RX-KAI-020 引入的判决口径与预注册机制。实现只有一个模块：

```text
evaluation/gate.py    GateCriteria（判据）/ 配对 bootstrap / GateVerdict（判决）
```

## 决策

### 1. 判据是版本化文档，先于数据，随 run 归档

`GateCriteria`（`forecast-gate-criteria-v1`）的字段就是 §42 说的**判据形态**：候选 backend、
唯一主判据指标、比较对象集合与选法、**证据切片**（`evidence_segments`）、分组轴与分组门槛、
样本量门槛、置信区间参数、compute 护栏。形态字段全部必填——缺少其中任何一个的判据根本
无法构造，因此「先跑再想」在类型层就不可能。

**证据切片是判据形态的一部分**（RX-KAI-020 评审 M1）。「哪些 origin 算证据」与「怎么分组」
同样是承重的：同一份 run、同一份判据，只看 `test` 段与把 `train` 池进来会给出不同的分组
样本量与不同的判决——实测示例窗口的 `test` 段 `BEAR` 只有 3 个 origin，池化后有 5 个，
因此任何落在 `(3, 5]` 区间的分组门槛都会让两种口径分别是「不可判」（`test`）与「可判」
（池化）。因此它没有默认值，必须与指标 / 比较对象 / 分组方式一起在 run 前冻结，并且：

```text
load_experiment_config  校验 evidence_segments ⊆ dataset.segments（run 之前，配置层）
evaluate_gate           校验声明的每一段在本次 run 里真的有记录（否则 ConfigurationError）
GateVerdict             携带 evidence_segments，因此「这个判决数的是哪些 origin」在产物里即可读
选最强 baseline         在同一批**证据段**上选出（口径见 §3）
```

判据身份随之覆盖切片：改 `evidence_segments` 会换 `criteria_hash`，所以「事后把 train 池
进来把门槛凑够」和事后放宽阈值一样是**可见**的。

**重新预注册记录**：本判据的第一版没有冻结证据切片（`evidence_segments` 字段尚不存在，
实现把 `train` + `test` 池化），其取值是 `direction_accuracy` / 分组门槛 5 / 整体门槛 32
（该版本从未进仓库，见 §6 的限定）。评审指出后，`configs/gate-criteria-forecast-v1.yaml`
在**任何正式 run 之前**重新预注册：加上 `evidence_segments: [test]`、主判据指标从
`direction_accuracy` 改为 `mae`、**只把分组门槛 5 改成 3**（整体门槛 32 两版相同）。
第一版判据只在示例窗口上被跑过一次，那次判决不作结论；这里的「正式 run」指的是按目标
universe / 窗口冻结判据的 run。

预注册的形式是三层证据同时落盘，缺一层都不成立：

```text
configs/benchmark-forecast.yaml 的 gate.criteria_file   → 这次 run 用的是哪份判据（进 config_hash）
run metadata 的 gate_criteria_hash                       → 判据的 canonical hash（语义口径）
artifact gate_criteria.yaml（原文）+ gate.json（判决）    → 原文 + 判决，两者同住一个 run 目录
```

`report.md` 的 Gate 段落里既有判决也有 `criteria_hash`，因此「这份结论是按哪份判据判的」
可以只读报告回答。

**事前拒绝**：`load_experiment_config` 在装载时就要求判据文件存在、YAML 合法、字段自洽，
并且候选与比较对象都在 `benchmark.backends` 里——否则 run 直接失败。理由是「跑完 100 秒
才发现判据指向一个没跑的 backend」既是浪费，也让失败出现在错误的阶段。

**可追溯性的边界**（显式登记）：artifact store 的写入是 append-only 且每个 artifact 带
sha256 进索引，因此「判决 ↔ 判据原文 ↔ 被判定的结论」三者互相钉住；但**没有**在 run 开始
之前把判据 hash 写到 run 之外的地方（那需要 run_id 先于结果分配，与 ADR-023 的 run_id 分配
时机冲突）。所以强形式的预注册（例如把 criteria hash 提交进 git 并在 run 里记录 commit）
仍属后续工作，见 §9 的 defer。

### 2. 判决只看配对 bootstrap 区间的下界（§42）

指标是**逐 origin 配对**的：候选与最强 baseline 必须在同一批 `(symbol, market_date)` 上
各有一条 `LABELED` 记录，这一对才进证据。配对不是实现细节，而是口径：

```text
非 LABELED（停牌 / 数据末端）      → 不进配对（缺料不是 0，§49）
只有一边有 LABELED 的 origin       → 不进配对（但配对数量会写进证据，缺多少可见）
同一 (backend, origin) 两条记录    → 显式失败（选哪一条都会让 CI 变成实现细节）
```

重复记录的检查在判决入口**跨全部段**做一次，而不是只查证据切片：段是互斥的时间切分，
同一 `(backend, symbol, market_date)` 出现在两个段里本身就是记录集合的错，不能靠「判决
只数某一段」把它藏起来。判定也**先于** label 过滤——一条 `LABELED` 加一条 `SUSPENDED`
同样是重复，不允许「哪条有指标值就用哪条」。

区间口径：`paired_bootstrap_percentile`（逐 origin 重采样，**同一组下标**同时聚合两侧）
+ `nearest_rank` 取值法（端点必须是一次真实出现过的重采样结果，与 §49 Compute 的分位口径
同一套约定）；`0.95 / 2000 / seed` 都写在判据里。每个切片（整体 + 每个 regime 分组）用
`sha256(seed:slice)` 派生的独立种子，因此某个分组的区间不依赖遍历顺序。

判决用**下界**，不是点估计，也不是 p 值：§42 的例子就是「bootstrap 95% CI 下界 > 0」。

**未做多重比较校正**（显式登记）：3 个分组各自 95% 的区间，家族错误率高于 5%。这与 §42
的字面判据一致（每个分组各自给 CI），但报告不宣称整体显著性；若要校正，应作为判据的
新版本引入，而不是在判决里悄悄加一个系数。

### 3. 判定指标白名单与方向（可证伪的前提）

`GATE_METRIC_DIRECTIONS` 是唯一白名单及其方向，增量统一成「正数 = 候选更好」：

```text
direction_accuracy   higher is better
mae                  lower  is better
rmse                 lower  is better（逐记录取值是平方误差，先聚合再开方）
```

两个 §49 指标**不进** gate：`return_correlation`（相关系数大不等于方向对，没有唯一单调
方向）与 `quantile_coverage`（校准属性，「离名义覆盖率更近」不是可累加的增量）。
方向不唯一的指标会让同一个数字支持两种相反的结论，而 §42 要求判据可证伪。

指标白名单的拒绝消息在 `Literal` 校验**之前**给出（`mode="before"`），因此错误信息说的是
「为什么这个指标不合格」，而不是 pydantic 的 `Input should be ...`。

最强 baseline 由**同一指标**在**同一批证据段**（`evidence_segments`）上选出
（`baseline_selection: strongest`），并列时取名字字典序最小者——选法在判据里预注册，
「跑完再挑更强的垫脚石」不可行；用同一批证据段选，是因为判据说「相对最强 baseline 更好」，
若「最强」是在另一批样本上算出的称号，判决里的比较对象未必是本次证据上最强的那个。
选择用的是**每个 baseline 自己**在这批段里的 `LABELED` 记录（不要求与候选配对：配对是
判决与区间的口径，选出的对象之后才与候选逐 origin 配对），因此选择集合可以比配对集合大。
预注册的 baseline 里只要有一个在证据切片上没有可比的 `LABELED` 记录，就抛
`InsufficientEvidenceError`：比较集合不完整时「最强」没有定义，这是缺证据，不是
「候选更好」。

### 4. 逐 regime 分组判定 + 整体条件（GO / CONDITIONAL / REPLACE）

判据必须声明分组轴（`trend` 三组 / `volatility` 两组）与门槛：

```text
min_groups_with_positive_increment  >= 1 个分组的下界 > 0（§42 的例子是 3）
min_paired_samples_per_group / _overall   低于门槛的切片不参与判决（并在结果里标出）
```

要求的分组数超过该轴的取值数时**加载即拒绝**：那等于把结论预定成 REPLACE。
run 级门槛低于分组门槛同样拒绝（不自洽）。

判决规则（完整、无遗漏，并被 `GateVerdict` 在构造期复核）：

```text
GO          整体下界 > 0 且 >= min_groups 个分组下界 > 0 且 compute 在护栏内
CONDITIONAL 整体下界 > 0 或 >= 1 个分组下界 > 0      （其余情形）
REPLACE     整体下界 <= 0 且没有任何一个分组下界 > 0
```

三条都建立在「证据足够判决」之上：整体配对样本数不到 `min_paired_samples_overall`、
或达到 `min_paired_samples_per_group` 的分组数不到 `min_groups_with_positive_increment`
时，`evaluate_gate` 在算判决**之前**就抛 `InsufficientEvidenceError`（§6）。因此 REPLACE
的实际含义是「证据足够判决，且没有一个切片显示出正增量」，没有「没测出来也算 REPLACE」
这条捷径。

整体条件是必须的：默认 backend 服务于整个 universe，只在部分 regime 有效不构成「继续作为
默认」（那正是 CONDITIONAL：「后续按 regime 使用」）。判决里带 `positive_groups` 与每个
分组的证据，因此「在哪些 regime 有效」是读得出来的，而不是从整体数字里猜。

### 5. compute 护栏是判据的一部分（§19）

判据必须给 `max_seconds_per_origin`（候选在每个 origin 上的平均墙钟秒数上限）。观测值来自
本次 run 每条记录的 `latency_ms`，护栏判定写进判决（`compute_within_budget`）。超限时
**不允许 GO**（降为 CONDITIONAL 并给 `COMPUTE_BUDGET_EXCEEDED` 理由码）：一个值得留作
默认的 backend 必须在其代价下可用。

这条护栏与 §16 的 Cost Probe（RX-KAI-018 / ADR-019）不是一回事：Cost Probe 回答「这台机器
跑不跑得完」（可行性，按 device class 测），gate 护栏回答「还值不值得当默认」（判断）。
两者都用延迟，但一个按 device class 立预算、一个按预注册的数值立判据，谁都不替代谁。
延迟是环境事实（不进 `report_hash`），但**进判决**——因此判决身份 `gate_hash` 会随机器变化，
这一点在 §7 里显式说明。

口径上的不对称要写清：`compute_seconds_per_origin` 取候选在**本次 run 全部段**的记录（不只
证据切片）：代价是 backend 的属性，跑 train 段花的机器时间与判决看哪一段无关。因此同一个判决
里「指标与区间数 `evidence_segments`，compute 数整次 run 的记录」是有意的，不是漏过滤。

### 6. 证据不足 ≠ 表现很差

配对样本量低于门槛、可评估的分组数不足、baseline 没有任何该指标的证据时，抛
`InsufficientEvidenceError`（新增错误类型），而不是给出 REPLACE：

```text
「没测出来」与「更差」是两件事；把前者写成后者就是 ADR-010 在评估侧的翻版
```

这条也解释了「判据不可满足」的处理：要求 3 个正增量分组、而只有 2 个分组有足够证据时，
那是**缺证据**（run 失败，提示去收集更多 origin），不是「Kronos 不行」。

示例判据的门槛调整走过一次这个过程：第一版按**池化**口径（`train` + `test`，82 个 origin，
trend 轴 BEAR 5 / SIDEWAYS 27 / BULL 50）把分组门槛定成 5；冻结证据切片后只看 `test` 段
（32 个 origin，BEAR 3 / SIDEWAYS 13 / BULL 16），分组门槛随之改成 3（整体门槛 32 两版
未变）。两次**门槛**调整都只看**证据量**，不看任何模型表现（同一批重注册里还有一次
**指标**变更 `direction_accuracy → mae`，那一步看的是本窗口的指标分辨力——见 §8，不属
「门槛」）。诚实说明：定门槛时看到的证据量分布、
以及第一版判据本身，都没有独立痕迹留在仓库里（第一版从未提交，也不保存 run 前的证据量
探针产物），这部分只是作者声明；可核查的是门槛与 run 的实际分组样本量一起写进了
`gate.json`，读的人可以对出「门槛是否贴着证据量定的」。

### 7. 判决是证据的函数，且绑定它判定的结论

`GateVerdict` 在构造期复核若干件事，任何一件不符就拒绝构造：

```text
positive_groups == 证据里 adequate 且 CI 下界 > 0 的那些分组
compute_within_budget == (观测值 <= 预算)
verdict == §4 的规则给出的那个结果
reason_codes == 由同样这些事实推出的那组码
candidate 与 strongest_baseline ∈ available_backends（且后者 != candidate）
groups 的分组名 == 该 grouping_axis 的取值（规范顺序）
version / criteria_version / metric / grouping_axis / ci_method / ci_quantile_method
  / evidence_segments / higher_is_better 与常量表和白名单交叉校验
```

因此「跑完之后改一个字段把 REPLACE 写成 GO」在构造期就失败。**不校验**的还有三件事，
连同理由：

```text
evidence_segments                   它是判据带来的输入，判决字段之间推不出它（设计上不可能校验）
点估计是否落在自己的区间内          百分位区间不保证点估计一定在内，当不变量会在极端样本上误拒
required_samples 与门槛字段是否一致  evaluate_gate 必然让它们相等，但断言它不增加任何安全
```

这条不变量的强度有边界：它挡住的是**同一份判决内部**的矛盾，不保证判决来自一次真实 run——
手工构造一份字段互相一致、但没有任何 run 支持的 `gate.json` 仍然可能（所以「可核查」依赖的是
run 归档本身：判决、判据原文、被判定的结果同住一个 run 目录，且 artifact store 是
append-only；`increment` / `ci_*` 是凭据，改它们等于伪造一次从未发生的测量）。

判决绑定两个身份：`criteria_hash`（判据）与 `report_hash`（被判定的结论），自身身份是
`gate_hash`。延迟进判决 ⇒ `gate_hash` 会跨机器不同，这是有意的：判决说的是「在那次 run 的
观测代价下，这个 backend 值不值得当默认」，而不是「这台机器快不快」。

### 8. artifact 与配置形态

```text
configs/benchmark-forecast.yaml       gate.criteria_file（路径，相对 config 所在目录解析）
configs/gate-criteria-forecast-v1.yaml 判据文档（含注释里写下的理由）
run 目录                              gate_criteria.yaml（原文）/ gate.json（判决）/ report.md（Gate 段）
run metadata                          gate_criteria_version / gate_criteria_hash / gate_verdict / gate_hash
```

路径相对 config 所在目录解析，因此 `configs/` 里的一组文件可以整体搬动；`config_hash` 里有
判据**路径**（「用了哪份判据」），判据**内容**身份由 metadata 的 `gate_criteria_hash` 承担。
不声明 `gate` 段的 run 不做判决：`gate.json` / `gate_criteria.yaml` 直接缺席，metadata 里
也没有 gate 字段——缺席是明确的，而不是一个空结论（与 §49 的 `n/a` 同一个原则）。

`gate.json` 带派生字段（`kind` / `gate_version` / `gate_hash` 与每个切片的 `positive`），
`GateVerdict` 又是 `extra="forbid"`：因此**回读**归档判决的工具必须先剔除这几个键（判决
原样归档是为了可读与可 diff，不是为了直接喂回模型）。本模块从未发布过，因此同名
`forecast-gate-v1` 下**加 `evidence_segments` 之前**的中间构建产物（缺少该必填字段）不保证
可回读——不做兼容，因为它只存在于本任务开发期的 scratch 里。

示例判据的门槛不是「最优参数」，而是**可判的最低证据量**：先做证据可得性检查
（只看每个 regime 分组有多少 origin，不看任何模型表现），再据此定
`min_paired_samples_per_group = 3`、`min_paired_samples_overall = 32`、
`min_groups_with_positive_increment = 3`，证据切片 `evidence_segments = [test]`。这一步必须
在正式 run 之前完成，且其取值连同理由写在判据文件的注释里（含「没有独立痕迹」这条诚实的
限定，见 §6）。

主判据指标用 `mae` 而不是 `direction_accuracy`，原因是本窗口下后者没有分辨力。实测
（`test` 段，32 个 origin）：5 个 backend 里 4 个——**含候选 kronos 自己**——的
`direction_accuracy` 精确并列在 `23/32 = 0.71875`（只有 `moving_average` 更低，为 0.53125），
因此在 `direction_accuracy` 下候选与「最强 baseline」的数字相同，每个切片的增量都恒为 0、
区间退化成 `[0, 0]`，判决退化成 `REPLACE`——这个结论只反映指标在本窗口没有分辨力，与
「有没有增量」无关，而且选谁当 baseline 都不改变它。同一批记录上 `mae` 是 0.013277
（kronos）/ 0.014837（last_value）/ 0.016194（ar1）/ 0.016491（drift）/ 0.018721
（moving_average），五个数字互不相同。

也就是说：同一个「最强」在两种指标下分别由一个实现细节（并列打破的字典序）和一组真实
数字决定。这也是 §3 的方向白名单存在的意义——换指标要换 `criteria_hash`，是有记录的决定。

## 后果

- 一次判决可以用 `gate_hash` 指认，用 `criteria_hash` 说明依据、用 `report_hash` 追到结论；
  「这份 GO 是按什么问题、什么门槛判出来的」不需要额外解释。
- 事后调参（换指标、放宽阈值、换比较对象、改分组）都会换掉 `criteria_hash`，而归档的判据
  原文仍在同一个 run 目录里——**替换是可见的**。
- 判据形态让「多跑几轮直到出现 GO」这件事至少留下痕迹：每次 run 的判据 hash 与判决都在
  registry 里，反复 run 同一个 `report_hash` 才会给出同一个判决。
- `CONDITIONAL` 的落点是「按 regime 使用」：判决携带每个分组的证据与正增量分组名单，
  后续决策层（Phase 3+）可以据此选择在哪些状态下使用哪个 backend，而不是全有全无。
- 示例窗口（2 symbol × 82 origin，证据切片 `test` = 32 个 origin）跑出的判决是
  `CONDITIONAL`：kronos vs 最强 baseline `last_value`（`mae`），只有 `BULL` 分组的下界 > 0
  （increment +0.003446，CI [0.001017, 0.005653]），`BEAR` / `SIDEWAYS` 的点估计为负且区间
  跨 0，整体 point estimate +0.001560 但下界 −0.000251 未过 0，compute 约 0.37 s/origin
  （两次示例 run 实测 0.365 / 0.383，随机器与负载波动）在护栏内。这**不是** Phase 2 的正式
  结论——示例窗口是演示与回归用的；正式 run 必须按其 universe / 窗口重新预注册判据。
- 没有足够证据时 run 会失败而不是给结论，因此「先跑个小样本看看」必须显式走 §48 的
  `origin_limit` 并在判据里把门槛同步调低——两条记录都会留在 artifact 里。

## 9. 显式 defer

- **强形式的预注册**：把判据 hash 在 run 开始前写进 run 之外的可追溯位置（git 提交 /
  registry 预登记）。当前形式是「run 归档内三者自洽 + artifact store append-only」，
  能发现事后替换，但不能证明「判据在 run 之前就已存在」。
- **universe 维度分组**：§42 说「只在部分 regime / universe 有效」，当前只实现
  trend / volatility 两个 regime 轴；按 symbol / sector / 流动性分组需要 §6.2 的
  PIT universe 快照生效（ADR-023 §8 已登记同一件事）。
- **多重比较校正**：见 §2 末段。
- **economic usefulness（§19 综合判断的第四个维度，第五个是 compute cost）**：任何交易层
  指标都需要版本化的 CostModel / ExecutionAssumption（§50 已把 turnover / net PnL 一类
  指标整体移出），在它定义之前不做「经济上有用」的判决。
- **判据与 §16 Cost Probe 的自动衔接**：让 probe 的结果（device class 的可行性预算）
  自动生成 gate 的 compute 护栏数值，目前由人写进判据——自动化需要先把 device class
  与 run 环境对齐（metadata 里已有 `environment`，但没有 device class 的规范字段）。
- **判决的置信区间敏感性**：判据固定 `bootstrap_iterations` / `seed`；换种子会换
  `gate_hash`（区间可能变），当前不做「多种子稳定性」检查。
- **证据不足时 run 产物被整体丢弃**：判决发生在 run 注册**之前**（`evaluate_gate` 在
  `_open_persistence` 之前调用），因此 `InsufficientEvidenceError` 会让 CLI 直接退出 1，
  这次 run 连 registry 记录都不留（不是「登记成 failed」），已经跑完、已经付费的 benchmark
  结果（`forecast.parquet` / `metrics.json`）全部不落盘，只剩一个「证据不足」的错误。
  重定门槛要再跑一遍——对示例规模无所谓，对正式 run 是真实浪费。当前不改（改法牵涉 run
  生命周期与 registry 状态语义，属于 §33 的 artifact/状态设计），代价是每次重定门槛都要重跑。
- **判据读取的 TOCTOU 窗口**：判据在 `load_experiment_config`（run 前校验）与 CLI 取对象
  （进判决/归档）时**各读一次**。两次读之间文件被替换时，校验的是 A、归档与判决用的是 B
  （毫秒级窗口，且替换会换 `criteria_hash`，因此在产物里可见）。修法是把已读到的判据对象
  与原文沿调用链传下去，当前为保持 `load_experiment_config` 的单一职责（配置装载）而保留
  双读，见 `config.py` 里的注释。

## 已知遗留（不在本次交付范围）

- **`config.yaml` artifact 的 `config_yaml_sha256` 名不副实**（`evaluation/report.py`，RX-KAI-019
  遗留）：它的值是 `sha256_hex(config_text)`，即「配置原文经 canonical JSON 化之后」的哈希，
  **不是**文件字节的 sha256。全仓库只有写入方、没有读者，也不进任何 hash（`config_hash`
  是语义 hash、artifact 完整性走 `ArtifactStore` 的字节 sha256），因此不影响本任务的正确性；
  但用 `shasum configs/benchmark-forecast.yaml` 去核对归档会得到不同的数字。登记在此备查，
  修法（改名 `config_text_canonical_sha256`，或改成字节哈希）属 RX-KAI-019 的后续小修。
