# ADR-007: A-share Price Adjustment Policy

- 状态：Accepted
- 日期：2026-09-26
- 对应任务：RX-KAI-006（前置：BaoStock 能力 spike）
- 对应基线：`Kronos-AI_v2_Greenfield_Rewrite_设计与落地方案.md` §6.0、§6.1、§6.4、§29、§30、§32、§36（ADR-007）、§55 DoD #7

## 背景

Kronos 的输入是 OHLC 序列，「用哪一种价格」直接决定预测与回测能否复现。v1
（`backend/`，baostock 0.8.9）**完全没有复权处理**：全仓库既无 `adjustflag`，也无因子
元数据与模式标记，价格口径既未声明也未版本化。§6.1 因此要求区分四个概念，并规定
「具体 Forecast Backend 使用哪种输入，必须通过 ADR 固化」。

RX-KAI-006 的能力 spike（`docs/spike/baostock-capability.md`、原始证据
`docs/spike/baostock-capability-raw.json`）给出三项决定性事实：

1. **因子表随查询时刻归一化**：全部 6 个被测标的最新一条记录的 `foreAdjustFactor`
   恒为 `1.000000`，`backAdjustFactor` 单调累积到该最新公司行动。表中每一行的数值
   都相对「查询时刻的最新行动」而定，接口不存在 as-of 语义。
2. **复权序列完全由「原始价格 + 因子表」决定**：映射规则在 5 个标的、1 745 个标的日上
   零失配：

   ```text
   hfq(D) = raw(D) * backAdjustFactor(最近一个 ex_date <= D)
   qfq(D) = raw(D) * foreAdjustFactor(最近一个 ex_date <= D)
   ```

3. **因子表按「窗口」截断，空结果不等于标的无因子**：`query_adjust_factor` 只有
   ex-date 区间语义，窗口内没有公司行动就返回 0 行且 `error_code=0`。误读示例：退市标的
   `sz.000003` 的因子记录实际有 15 条（`1991-07-03`..`1996-07-29`），而
   `start_date=2000-01-01` 的窗口查询返回 0 行——「0 行」完全由窗口造成，与退市无关。
   两种窗口都在 artifact 里可对照：`checks.symbol_lifecycle.adjust_factor.from_1990.rows=15`
   与 `.from_2000.rows=0`。退市必须靠 `query_stock_basic` 的 `outDate` / `status` 判定
   （`sz.000003`：`outDate=2002-06-14`、`status=0`；在册对照 `sh.600000`：
   `outDate=""`、`status=1`，见 `checks.symbol_lifecycle.stock_basic` 与 `listed_reference`）。

映射规则的单点 golden 值（同一 artifact 可查，供回归测试锚定）：
`sh.600000` `2023-01-03` → `raw=7.23`、`hfq=82.67093613`、`qfq=6.18469383`，
适用因子行 `ex_date=2022-07-21`、`fore=0.855421`、`back=11.434431`；
`hfq/raw=11.434431`、`qfq/raw=0.855421`，与规则精确一致。

由 1 与 2 可推出：provider 直接返回的 `adjustflag=1/2` 序列**不是 PIT 可复现的**
（同一历史日期在不同时刻查询会得到不同价格）；而只要因子表被快照留痕，复权序列
就可以在本地精确重建。

## 决策

### 1. 四个概念分离，规范存储为「原始价格 + 因子表快照」（§6.1）

```text
Raw OHLC                      ← 规范存储，PIT 稳定
Corporate Actions             ← 本任务不采集（见「后果」第 4 条）
Adjustment Factor             ← 只作为元数据留痕，append-only 快照（§30）
Point-in-Time Adjusted Series ← 派生视图，不是存储形态
```

### 2. 默认 forecast 输入 = `raw`

`AdjustmentPolicy(mode="raw")` 为默认，且**不得**声明 `factor_source`。理由：

- PIT 稳定：原始价格是数据源唯一不经重算的口径；
- 不依赖因子表：退市标的、因子表缺失、快照尚未建立时都不会阻塞主流程；
- 不依赖尚未核实的盘后发布时刻语义（能力报告 §6 第 1 条）。

### 3. 复权序列是派生视图，且被 gate 住

- **禁止**把 provider 的 `adjustflag=1/2` 序列直接写入 artifact 或用作基准输入——
  它随查询时刻变化，会让同一实验在不同时间得到不同结果。
- 唯一允许的生成路径：本地原始价格 + **已快照**的因子表，按上文验证过的映射规则换算；
  因子表快照遵循 §30 append-only（重新拉取产生新 `dataset_version`，旧快照保留）。
- 因此 `AdjustmentPolicy` 对 `hfq` / `qfq` 强制要求 `factor_source`（声明快照 provenance），
  缺失即构造失败——把「复权模式必须有因子快照」变成类型层面的事实而非口头约定。

### 4. 版本化

- `ADJUSTMENT_POLICY_VERSION = "adjustment-policy-v1"`，随 `AdjustmentPolicy.run_metadata()`
  进入 §32 Run Metadata 的 `adjustment_policy_version` 字段（对应 DoD #7）。
- 默认值语义变更（例如未来把默认输入改为 `hfq`）必须提升版本号，旧 run 记录保持可解释。

### 5. 显式失败，禁止因子兜底

因子表缺失、为空或未覆盖所需区间时**显式失败**，不得把缺失因子静默视为 `1.0`
（「窗口内 0 行」正是这种陷阱：它既可能是窗口造成的，也可能是数据缺陷，必须显式区分）。
`AdjustFactorSeries.factors_asof(day)` 对「早于首个公司行动」的日期返回 `None` 而不是
`1.0`，调用方必须自行决定语义。

### 6. 实现边界与执法点

本任务（RX-KAI-006）只固化**契约、版本与校验**：`src/kronos_ai/data/adjustment.py` 中的
`AdjustmentPolicy`、`AdjustFactorRecord`、`AdjustFactorSeries`。价格换算的落地与因子表
快照落盘属数据集构建与 artifact store（RX-KAI-015 及后续），届时按本 ADR 的规则实现，
并复用能力报告中的映射规则（含上文的 golden 值）做回归验证。

**执法点说明**：第 3 条的「禁止写入 artifact」是**产物边界**的约束，不是 provider 的
能力限制。`BaoStockProvider(adjust_flag=...)` 保留 `1/2` 透传能力（用于本地换算与
回归对照测试），且 `AdjustmentPolicy` 目前没有生产调用方；因此该禁止条款在 RX-KAI-006
阶段**尚不可被运行时强制**，其执法落在 RX-KAI-015 的 dataset / artifact 写入路径
（届时须拒绝 `adjustment_mode != "raw"` 且未绑定因子表快照的输入），**并须在
`tests/leakage/`（§29 泄漏守卫）中加一条回归测试**：provider 的 `adjustflag=1/2`
序列不得出现在 artifact 或基准输入里。在此之前，任何复权序列都不得进入基准结果，
也不得作为 `ForecastRequest` 的输入。

## 后果

- 基准结果不被因子重算污染：默认 `raw` 的复现只需 raw 快照，无需因子表。
- 复权模式需要额外的因子表快照与一次本地换算；代价换来的是可复现（这是 §6.1 的
  「历史输入可复现」要求）。
- 因子缺失、快照缺失在构造/加载阶段显式失败，不会产出「看起来正常但口径不明」的价格。
- **退市标的的价格与因子都可取，退市信号本身不在价格序列里**：`sz.000003` 的退市窗口
  `2002-06-03`..`2002-06-14` 全量 10 行都是 `tradestatus=0` 的 flat 停牌行
  （`checks.symbol_lifecycle.history.table`），末日 `volume`/`amount` 为空串；仅看 OHLC
  无法区分「长期停牌」与「已退市」。退市必须由 `query_stock_basic` 的
  `outDate`/`status` 判定——该结论交给 RX-KAI-007 的 universe 合约与 §6.5 生存偏差处理
  落地。
- **未采集公司行动明细**（`query_dividend_data` 未探测）：默认 `raw` 路径不需要它；
  若将来复权模式成为默认或需要分红解释，必须先补采并重新评估本 ADR。
- 能力报告 §6 列出的未验证项（盘后发布时刻、跨时间因子重写）不因本 ADR 变为已知，
  仍由快照机制与版本化 cutoff policy 承载。
