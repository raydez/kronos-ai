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
horizon 越过日历 coverage 或数据末端         → INSUFFICIENT_FUTURE_BARS
```

禁止静默向后延长 horizon；个股停牌不改变 forecast 时间轴（§11）。

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
  生成 ForecastPoint。BaoStock `query_trade_dates` 的实际覆盖范围（是否包含未来日期）
  由 RX-KAI-006 spike 核实，核实前不得假设其覆盖未来。

### 4. 实现分工与 provenance

- 契约与语义承载于 `src/kronos_ai/data/calendar.py`（§4 布局）。
- `StaticTradingCalendar` 只回放显式给定的官方 session 序列，不含任何规则推导；
  仅用于测试 fixture、离线回放与显式启用的合成数据源（§3.2）。
- `exchange` / `source` 为必填 provenance，不提供默认值：任何进入 artifact 的时间轴
  必须能回答「哪个日历、来自哪里」（§3.5）。BaoStock 装载器在 RX-KAI-006 结论之后
  落地，`source` 形如 `baostock:query_trade_dates-v1`。

## 后果

- Forecast / Label / Walk-forward / 停牌标记共享同一时间轴真源，`pandas` 工作日推导被禁止。
- 日历覆盖不足、origin 非 session 等错误在生成阶段显式失败，不缺省向后延长。
- 真实日历接入前，任何基准结果必须能被识别出 `source` 为合成序列，避免被误读为
  真实交易日历结论。
