# ADR-009: Trading Calendar and Suspension Policy

- 状态：Accepted
- 日期：2026-09-26
- 对应任务：RX-KAI-005
- 对应基线：`Kronos-AI_v2_Greenfield_Rewrite_设计与落地方案.md` §4、§6.3、§6.6、§11、§36（ADR-009）

## 背景

Forecast 时间轴与 Label 时间轴必须共享同一 session 语义（§11）。v1 以「跳过周末」近似
交易日（`backend/app/services/kronos_integration.py:446` 的 `weekday() >= 5`），既不识别
节假日，也没有停牌判定规则，导致预测时间轴与个股实际可交易日错位，且下游各自推导。
本 ADR 固化术语、判定规则与覆盖范围契约，使 ForecastPoint、Label、Walk-forward 切分、
停牌标记不再各自发明。

## 决策

### 1. 术语（版本化语义）

```text
market session = 交易所开市日，由 TradingCalendar 给出
stock session  = 个股在某个 market session 实际可交易的 session（停牌即缺席）
valid bar      = stock session 的日线 bar，且价格有效
```

valid bar 的构造性判定在 Provider 层执行（RX-KAI-004）：停牌行（tradestatus=0）与
非正价格行不产生 bar。因此 label 侧只能从「market session 上缺 bar」推断停牌，
该推断规则在本 ADR 固化，禁止在 Dataset / Label 实现中各自发明。

### 2. 停牌与 horizon 判定（Label 侧，RX-KAI-017 实现）

```text
horizon 内某个 market session 无 valid bar   → SUSPENDED
horizon 越过数据末端（data_coverage_end）    → INSUFFICIENT_FUTURE_BARS
```

禁止静默向后延长 horizon；个股停牌不改变 forecast 时间轴（§11）。

「日历 coverage 不足」与「数据末端」是两个不同口径（RX-KAI-017 澄清）：

- **日历 coverage 不足**：`calendar.next_sessions(origin, horizon)` 无法给出 horizon 个
  session。这是构建期错误，整个 dataset 构建直接抛 `CalendarError`（§3），不会产生
  一个部分样本。
- **数据末端**（`INSUFFICIENT_FUTURE_BARS`）：日历知道那些 session，但 provider 发布的
  数据还没有到那里。按单个 label 标记，不使构建失败。

落地（RX-KAI-017，`src/kronos_ai/evaluation/dataset.py`）：三态显式取值为
`LABELED` / `SUSPENDED` / `INSUFFICIENT_FUTURE_BARS`，非 `LABELED` 的 label 不带
收益与方向（不用 0 收益冒充「证据不足」）。判定优先级：窗口越过
`data_coverage_end` → `INSUFFICIENT_FUTURE_BARS`；窗口内有 market session 无 valid bar
（含 `trade_status="0"` 的停牌行）→ `SUSPENDED`。`data_coverage_end` 必须是**全局**
口径（provider 已发布的最后一个 market session），不是「该个股最后一根 bar」：
个股停牌与发布滞后只在全局口径下可区分（§29 的 label provenance）。label 窗口内出现
`available_at <= origin.knowledge_cutoff` 的 bar 属 §29 泄漏，抛 `LeakageError`（该路径
在正常数据下不可达，因为 `MarketBar` 保证 `available_at >= timestamp`；保留为
defense-in-depth）。

### 3. 日历契约与覆盖范围

- 签名按 §6.6 固定：`is_session(day) -> bool`、`next_sessions(market_date, count) -> list[date]`；
  `next_sessions` 严格排除 market_date 当日，升序返回。
- 覆盖范围外（早于首个 / 晚于最后一个 session）抛 `CalendarError`，消息以
  `outside calendar coverage` 开头；coverage 不足以产出 count 个 session 抛
  `CalendarError`，消息以 `coverage ends at` 开头。两个前缀是调用方可依赖的约定，
  编排层按前缀区分「日期越界」与「日历尚未覆盖未来」。
- `market_date` 必须是 market session：origin 落在非 session 上抛 `CalendarError`。
- `count < 1` 抛 `ConfigurationError`（调用方输入错误，与 RX-KAI-004 的
  `lookback_bars` 校验一致）。
- 误传 `datetime`（`date` 的子类，如 `MarketBar.timestamp`）抛 `ConfigurationError`，
  不做静默 `.date()` 截断。
- 数据源契约：日历必须覆盖 origin 之后至少 `max horizon` 个 session，否则编排层无法
  生成 ForecastPoint。BaoStock `query_trade_dates` 的实际覆盖范围已由 RX-KAI-006 spike
  核实（`docs/spike/baostock-capability.md` §1）：
  - 覆盖 `1990-12-19`..**当年度末**（如 2026-12-31），区间内每个自然日都有一行，
    节假日为 `is_trading_day=0`；年内未来 session 已发布，**跨年不可用**；
  - 越界区间返回 `error_code=0` + 0 行（静默空），**不是**错误信号；
  - 因此装载器/调用方必须把「静默空」与「覆盖不足」转成显式 `CalendarError`，
    禁止把空结果当作「该区间无交易日」；
  - 年末 origin 且 horizon 跨年是 `INSUFFICIENT_FUTURE_BARS` 的真实来源，
    属数据源边界而非故障，必须显式失败而不缩短 horizon；
  - 取数侧旁证：最近窗口（`checks.publication_lag`，`today=2026-09-27`）内三个标的
    的最后一条 bar 都停在 `2026-09-24`，窗口内的 calendar session 只有
    `09-17`/`09-18`/`09-21`..`09-24`，而 `2026-09-25`（周五，中秋）与周末
    `09-26`/`09-27` 在日历上本就不是 session（见 §6），故
    `sessions_without_bars=[]`，「bar 缺行」与「非交易日」一致，
    未观测到发布滞后（`docs/spike/baostock-capability.md` §6）。

### 4. 实现分工与 provenance

- 契约与语义承载于 `src/kronos_ai/data/calendar.py`（§4 布局）。
- `StaticTradingCalendar` 只回放显式给定的官方 session 序列，不含任何规则推导；
  仅用于测试 fixture、离线回放与显式启用的合成数据源（§3.2）。
- `exchange` / `source` 为必填 provenance，不提供默认值：任何进入 artifact 的时间轴
  必须能回答「哪个日历、来自哪里」（§3.5）。BaoStock 装载器在 RX-KAI-006 结论之后
  落地，`source` 形如 `baostock:query_trade_dates-v1`。
- RX-KAI-017 起，这两个字段同时是 `TradingCalendar` Protocol 的**只读属性**（纯增量
  变更，`StaticTradingCalendar` 已结构化满足）：`build_walk_forward_dataset` 直接读
  `calendar.exchange` / `calendar.source` 写入 `dataset_hash`（ADR-022 §5），
  不再依赖 `getattr` 式的鸭子类型探测。

### 5. 停牌字段实测（RX-KAI-006 spike）

`query_history_k_data_plus` 的 `tradestatus`（`1` 正常 / `0` 停牌）与 `isST` 均可用；
停牌日**仍返回 bar**，形态为 O=H=L=C（延续前收）。202 个样本停牌行中：202 行 flat OHLC、
201 行 close 与前一行相同、201 行 `volume=0`，另有 **1 行 `volume` 为空串**
（`sh.600816`；退市标的 `sz.000003` 的末日 `2002-06-14` 也出现空串 `volume`/`amount`，
见 ADR-007）。因此「valid bar」只能按 `tradestatus` 与价格有效性构造，不能用「该日有无行」
推断停牌，也**不能假定数值字段非空**；`isST` 与 `tradestatus` 正交（ST 可正常交易也可停牌）。

### 6. 日历不可用工作日规则替代（实测反例）

`2026-09-25`（星期五，中秋节）在 BaoStock 日历上是 `is_trading_day=0`，其后
`2026-09-28`..`09-30` 正常开市，`2026-10-01`..`10-07` 为国庆长假。任何「跳过周末」
式的近似都会把 `2026-09-25` 当作 session，从而让 forecast 时间轴与个股实际可交易日
错位——这正是本 ADR 要求统一 `TradingCalendar` 真源、禁止 `pandas` 工作日推导的
直接证据（v1 `kronos_integration.py` 的 `weekday() >= 5` 即属此列）。

## 后果

- Forecast / Label / Walk-forward / 停牌标记共享同一时间轴真源，`pandas` 工作日推导被禁止。
- 日历覆盖不足、origin 非 session 等错误在生成阶段显式失败，不缺省向后延长。
- 日历 `exchange` / `source` 进入 `dataset_hash` 与 Run Metadata，所以「同一批样本」
  的含义包含「同一份日历 provenance」。
- 真实日历接入前，任何基准结果必须能被识别出 `source` 为合成序列，避免被误读为
  真实交易日历结论。
