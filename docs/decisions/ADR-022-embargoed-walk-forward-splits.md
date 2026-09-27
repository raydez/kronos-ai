# ADR-022: Embargoed Walk-forward Splits

- 状态：Accepted
- 日期：2026-09-26
- 对应任务：RX-KAI-017
- 对应基线：`Kronos-AI_v2_Greenfield_Rewrite_设计与落地方案.md` §27、§28、§29、§32.1、§36（ADR-022）

## 背景

walk-forward 样本集有三个容易各自发明、又必须彼此一致的东西：**时间轴**、**分段**、
**label 语义**（§27 / §28）。v1 的失败模式是「指标好看但不可信」：训练段的 label
窗口与验证段的输入窗口重叠（没有 embargo），或事后为了「多几个样本」调整切分，于是
同一份代码在不同 run 里产出不同的样本集却共用一个报告标题。

本 ADR 固化三件事：分段与空置（embargo）的唯一实现、label 语义的版本化真源、
以及把这些固定下来的 `dataset_hash`。目标是让「这次 benchmark 用的是哪批样本、
按哪个 label 定义算的」成为一个可以用一个 sha256 回答的问题（§32、§33）。

## 决策

### 1. 模块边界：索引算术与 dataset 语义分离

```text
evaluation/walk_forward.py   纯 session 索引空间算术：分段、embargo 空置、leakage 断言
                             （不认识 symbol / bar / label / 收益率）
evaluation/dataset.py        LabelPolicy、ForecastOrigin、ForecastLabel、
                             WalkForwardDataset、label builder、dataset_hash
```

`dataset.py` 单向依赖 `walk_forward.py`；反过来不成立（§4 的单向依赖约定，与
`evaluation` → `forecast`/`data`/`domain` 的方向一致）。embargo 规则只有一份实现，
且它不依赖任何数据加载代码，因此可以脱离 provider 做穷举式回归测试。

### 2. LabelPolicy 是 embargo 与 label 语义的唯一真源

- `LabelPolicy` 是 frozen model，`version` 是主键；`LABEL_POLICIES` 是
  `{version: definition}` 的只读注册表，`resolve_label_policy` 要求调用方传入的
  policy 与注册表定义**逐字段相等**，否则 `ConfigurationError`。
- `resolve_label_policy` 是**所有**入口的检查点：`build_walk_forward_dataset`、
  `build_label` 以及 `WalkForwardDataset` 的 model validator 都会调它。手工拼一个
  `version="label-policy-v1"` 但 `embargo_sessions` 不是 v1 定义的 dataset 会被拒（§28
  的「一个版本对应一份定义」必须对绕过 builder 的路径也成立）。
- `embargo_sessions` 只出现在 `LabelPolicy` 里（§32.1）。experiment config 的
  `evaluation.embargo_sessions` 必须与所选版本一致（`label_policy_from_experiment_config`），
  不一致即拒绝运行——事后改 embargo 会让全部历史 `dataset_hash` 失效，这是特性而不是
  需要兼容的用法。
- 不变量 `embargo_sessions >= horizon_sessions` 在两处校验：`LabelPolicy`
  （`ValueError`，字段级）与 `require_no_leakage`（`ConfigurationError`，调用级）。

### 3. 分段的语义是「完全确定」，不是「够用就行」

- 段名只能是 `train` / `validation` / `calibration` / `test`，且必须按这个 canonical
  顺序出现（`SegmentName` 的 `get_args` 顺序即契约）。
- 时间轴由 `history_prefix + Σ段长 + embargo × (段数-1)` **唯一确定**：
  `split_sessions` 要求长度精确相等，多一个少一个都抛 `ConfigurationError`。
  静默取前 N 个是最危险的一类 bug——两次 run 的 `dataset_hash` 相同而样本不同。
- `history_prefix` 在 benchmark 里取 `lookback_bars`：首个 origin 的 lookback 窗口
  必须落在时间轴上，否则第一个 origin 天生不可估（§17 前置条件）。
- `lookback_bars` 是 **dataset 级**单一数字，不是 per-backend 配置：§19 的 gate 要求
  所有 backend 在同一窗口、同一信息量下比较（承接 RX-KAI-016 的口径）。

### 4. Leakage guard 断言 embargo，而不是 horizon

`require_no_leakage` 检查的是「前段最后 origin + `embargo_sessions` < 后段第一 origin」。
只看「label 窗口不重叠」（即 `gap >= horizon_sessions`）是必要但不充分的：当
`embargo_sessions > horizon_sessions` 时，训练段末端的 label 与验证段输入窗口之间
仍可能留下可被利用的信息重叠。因此代码里只写 embargo 断言，horizon 断言作为
`embargo >= horizon` 不变量的推论存在（tests/leakage/ 有用例专门证明这一点）。

`LeakageError` 从 `WalkForwardDataset` 的 model validator **直接冒泡**，不被包装成
`ValidationError`：数据集完整性不是字段格式问题，调用方不应当把它当作「某个字段填错
了」来修补（ADR-010）。同一口径也用在 label 侧：label 窗口内出现
`available_at <= origin.knowledge_cutoff` 的 bar 属 §29 泄漏，抛 `LeakageError`。

数据集模型层另外校验：每个 origin 的 symbol 必须在 `symbols` 里；段的起止下标必须与
时间轴切片吻合且首末对齐。手工构造的 dataset 允许比 embargo **更宽**的间隔（多出的
session 不属于任何段，不会泄漏），builder 产出的则是恰好 embargo。

### 5. `dataset_hash` 的载荷

```text
kind=walk_forward_dataset
walk_forward_version,
dataset_version, label_policy(hashing_payload：version + 8 个字段), 
calendar{exchange, source}, cutoff_policy(cutoff_policy_record),
lookback_bars, symbols, sessions(iso), segments[{name, start_index, origins[…]}]      
```

- `walk_forward_version`（分段 / embargo 算术的版本）也进 payload：改变「一段占哪些
  session」的规则本身就换掉全部 `dataset_hash`，不必等到某个字段跟着变。
- 重复 symbol 由 builder 显式拒绝（`ConfigurationError`），不静默去重：否则
  `symbols=("600000", "600000")` 与 `symbols=("600000",)` 会共享同一个 `dataset_hash`
  （ADR-010）。

- label policy 以**整体 payload** 进入（含 `version`），因此「dataset_hash 绑定
  LabelPolicy 版本」是结构性的，而不是靠调用方记得拼版本号。
- 日历 provenance（`exchange` / `source`）必须进 hash：ADR-009 §4 要求任何进入
  artifact 的时间轴都能回答「哪个日历、来自哪里」。为此 `TradingCalendar` Protocol
  增加只读属性 `exchange` / `source`（纯增量，`StaticTradingCalendar` 结构化满足）。
- `origins` 顺序规范化为 **symbol-major**（symbol 升序在外层、session 升序在内层），
  使 `dataset_hash` 与调用方传 symbol 的顺序无关。
- 字段增删属于契约变更：`tests/unit/test_evaluation_dataset.py::TestDatasetHash`
  钉住一个 golden hash，并由 `LabelPolicy.hashing_payload` 的完备性用例（键集合必须
  等于 `model_fields`）守住「加了字段却忘了进 hash」。
- `dataset_hash` 是 `@computed_field @property`，读取时实时派生（与
  `MarketHistory.data_hash` 同构），不落盘、可被篡改的副本不存在。

### 6. Label 侧的三态与方向词汇

- 三态显式取值：`LABELED` / `SUSPENDED` / `INSUFFICIENT_FUTURE_BARS`（ADR-009 §2）。
  非 `LABELED` 的 label **不带** `horizon_return`/`direction`（`None`），绝不用 0 收益
  冒充「证据不足」：否则「模型差」与「数据缺」在报告里混为一谈（§20）。
- 判定优先级：窗口越过 `data_coverage_end` → `INSUFFICIENT_FUTURE_BARS`；窗口内某
  market session 无 valid bar（含 `trade_status="0"` 的停牌行）→ `SUSPENDED`。
  `data_coverage_end` 是**数据集全局**口径（provider 已发布的最后一个 market session），
  不是「该个股最后一根 bar」——只有全局口径才能把停牌与发布滞后分开。
- 同一个 session 出现两根 bar 一律拒绝，且判定不依赖停牌过滤的顺序：
  `(停牌, 有效)` 与 `(有效, 停牌)` 都必须报 `DataQualityError`（顺序相关的守卫不是守卫）。
- 方向词汇直接复用 §24.1 Decision Schema 的 `BEARISH`/`NEUTRAL`/`BULLISH`，
  默认阈值即 §13 的 ±2%，避免平台上出现两套「涨跌」口径。`DirectionThreshold`
  用「闭下界列表」表达三分类（首项 `threshold=None`），边界值归入较高一类；
  不再用裸 `dict`（§28 的 `direction_thresholds: dict` 在实现上物化为有序 frozen model，
  可变容器会让已哈希的策略被就地改写）。
- §24.1 的 `UNKNOWN` 是**决策层**的「证据不足」，不是 label 的一类：label 侧用
  `direction is None` 表达，决策层负责把它映射成 `UNKNOWN`（RX-KAI-02x）。

### 7. session 日期不接受 `datetime`

label / 分段时间轴用 `SessionDate = Annotated[date, BeforeValidator(...)]` 显式拒绝
`datetime`（`date` 的子类），而不是静默 `.date()` 截断——误传 `MarketBar.timestamp`
会让 forecast 时间轴与 label 时间轴错位，这正是 ADR-009 §3 要杜绝的 v1 bug 类型。

## 后果

- 一次 benchmark 的样本集可被一个 `dataset_hash` 唯一确认；报告必须引用它（§48）。
- 新增 label 语义（而不是改数值）必须：新增 `LabelPolicy` 版本 → 注册进
  `LABEL_POLICIES` → 必要时给 `build_label` 加分支。policy 的取值空间由测试用
  `typing.get_args` 钉住，加取值不加分支会让用例失败。
- `evaluation/dataset.py` 是「纯定义 + 校验」：不加载行情、不落盘。标签数值与
  覆盖率统计在 benchmark 层（RX-KAI-019）生成 artifact，并随 run 记录
  `label_policy_version` 与 `dataset_hash`。
- 数据集构造期即失败：日历覆盖不足以支撑 horizon、时间轴与计划不匹配、手工拼接一个
  漏数据集、symbols 重复，都会在生成阶段报错，不会产出「看起来跑通了」的报告。
- 「日历 coverage 不足」与「数据末端」是两个口径（ADR-009 §2）：前者在 dataset 构建期
  抛 `CalendarError`（整个样本集不可构造），后者只把单个 label 标为
  `INSUFFICIENT_FUTURE_BARS`。
