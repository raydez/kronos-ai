# BaoStock 数据源能力报告（RX-KAI-006 前置 spike）

- 日期：2026-09-27（含评审修订后的最终 artifact）
- 对应基线：`Kronos-AI_v2_Greenfield_Rewrite_设计与落地方案.md` §6.0（spike 前置）、§6.1–§6.5
- 结论文件（原始证据）：`docs/spike/baostock-capability-raw.json`
- 复现脚本：`experiments/spike_baostock_capability.py`

```text
uv run python experiments/spike_baostock_capability.py --timeout 30 \
    --json docs/spike/baostock-capability-raw.json
```

脚本退出码：`0` = 全部检查 `ok`；`1` = 存在 `error`（检查抛异常，含客户端返回 `None`）；
`2` = 存在 `unverified`（硬超时内无响应）且无 `error`；`3` = 两者都有。
artifact 中 `status` 字段与退出码同义，`errors` / `unverified` 是两个独立列表。
**「无响应」不得当作「能力确认」**，因此任何非 `ok` 都会让脚本显式失败。

每个检查还必须给出 `confirmed`（能力是否被确认，附 `confirm_criterion`），且除
`error_semantics` 外的检查不允许出现任何 `error_code != "0"` 的探针结果：否则该检查
被降级为 `error`。「所有查询都失败、但恰好跑完」不会被计为 `ok`（评审 M5）。
`confirmed` 未通过时的 `reason` 字段给出未确认的判据。

运行环境与证据：客户端 baostock 0.9.4，端点 `public-api.baostock.com:10030`；
本次 11 项检查全部 `ok` 且 `confirmed=true`（`checks_ok=11/11`），`errors` 与
`unverified` 均为空。

**报告纪律**：下文每一条事实都必须能从落盘 artifact 里直接读出（评审 Major 2）。
凡引用数字，均标注其 JSON 路径；artifact 中查不到的推论一律移入 §6「未验证项」。

---

## 0. 前置条件：客户端版本与端点

`query_*` 全部失败的根因是端点迁移，不是网络不可达：

| 客户端 | 端点 | 实测 |
| --- | --- | --- |
| 0.8.9（v1 固定在 `backend/requirements.txt`） | `www.baostock.com:10030` | 端点本身可达：`checks.client_info.legacy_endpoint_probe` → `connect_ok=true`、`elapsed_s=0.007`；「对端立即关闭、客户端在 `util/socketutil.py::send_msg` 的 `while True: recv` 中空转、查询永久挂起」是 **0.8.9 客户端会话**中的观察，升级后无法复现（§6 第 7 条） |
| 0.9.4（本次升级） | `public-api.baostock.com:10030` | 端点可达（`checks.client_info.tcp_probe`，`elapsed_s=0.007`）；登录 `elapsed_s=0.09`（`checks.login`）；各查询检查本次运行 `elapsed_s` 在 `0.14`–`10.19` 之间（最慢为 `checks.index_constituents`）——均为**运行相关**数字，以 artifact 为准 |

其他版本事实：

- 0.9.x 新增 API key（`bs.set_API_key`）。**`set_API_key` 自身不做校验**，artifact 把它的
  源码原样落盘（`checks.client_info.api_key.setter_source`，函数体只有一个恒真的 `if`
  与 `setattr`）；但 **`login()` 会在发出任何 I/O 之前做客户端校验**：调用
  `valid_API_key`（33 位、`bs-` 前缀、Base62 字符集、校验和），不符即返回
  `BSERR_APIKey_FORMAT_INCORRECT = "10001012"`
  （`checks.client_info.api_key.validator_call_site` 与 `.validator_source`，以及
  `.server_error_code_for_bad_format`）。**匿名登录仍可用**，本次全部证据来自匿名会话，
  未使用 VIP 端点 `vip-api.baostock.com`，也未使用 API key。
- 0.9.4 **仍未修复** `send_msg` 的空转逻辑：EOF 与超时都不被当作错误。因此本 spike
  对每个检查施加 daemon 线程硬超时，并在报告中区分 `ok` / `error` / `unverified`。
- 客户端对部分入参做本地校验并**返回 `None`**（不是带 error_code 的结果对象），
  Provider 必须显式处理（见 §5）。
- 0.9.4 把单页行数从 10 000 降到 `BAOSTOCK_PER_PAGE_COUNT = 2000`，且
  `ResultData.get_data()` 用 pandas 已移除的 `DataFrame.append` 合并翻页结果：
  单股窗口超过一页即抛 `AttributeError`。取数必须走官方 demo 的
  `next()` / `get_row_data()` 路径，并且**翻页失败也必须显式失败**——
  `send_msg` 返回空时库不置 `error_code`，唯一痕迹是游标停在整页
  （RX-KAI-006 评审 Major 1 / C1，
  `src/kronos_ai/infrastructure/providers/baostock.py::_drain_pages`）。本 spike 用
  同一判据守卫自身取数（`experiments/spike_baostock_capability.py::_reject_truncation`，
  页长同样取自库常量）：一旦出现静默截断，该检查直接转为 `error`，不会把「少了一批
  行」写进 artifact。

---

## 1. 交易日历 `query_trade_dates` —— 可用，年内覆盖未来、跨年不可用

字段：`calendar_date, is_trading_day`。

| 事实 | 证据（JSON 路径） |
| --- | --- |
| 覆盖 `1990-12-19` .. `2026-12-31` | `checks.trade_dates.rows=13162` / `sessions=8797`（请求 1990-01-01..2027-12-31） |
| 区间内每个自然日都有一行，节假日 `is_trading_day=0` | 单日 `2026-10-01`（国庆）→ `sample_rows=[["2026-10-01","0"]]`（`checks.error_semantics.probes.trade_dates_holiday_single_day`） |
| 年内未来日期已发布 | `covers_future_sessions=true`（`2026-12-31` > 今日） |
| 跨年不可用，且**静默返回空** | 请求 `2027-01-01..2027-12-31` → `error_code=0`、`rows=0`（`checks.trade_dates.beyond_coverage`） |
| 工作日规则会出错：星期五也可能是休市 | `checks.trade_dates.holiday_window`（窗口 `2026-09-24`..`2026-10-12`，19 行）：session 仅 `09-24`、`09-28`..`09-30`、`10-08`、`10-09`、`10-12`；`2026-09-25`（星期五，中秋节）与 `10-01`..`10-07`（国庆）均为 `is_trading_day=0` |

判定：满足 point-in-time（历史 session 在多次采样中保持一致，但单次拉取**不能**证明
「永不重算」，故 §6 仍保留该项）。**覆盖范围足以支撑 ForecastPoint 生成，但不跨年**
——年末 origin 且 horizon 跨年时，日历无法给出 `next_sessions`，必须显式失败而非缩短
horizon。

> 影响 ADR-009：`INSUFFICIENT_FUTURE_BARS` 的一种真实来源是「日历年边界」，
> 不是数据源故障；装载器必须把「静默空」转成显式失败。

---

## 2. 复权因子 `query_adjust_factor` —— 因子表随查询时刻归一化，复权序列不可 PIT 复现

字段：`code, dividOperateDate, foreAdjustFactor, backAdjustFactor, adjustFactor`。

| 事实 | 证据（JSON 路径） |
| --- | --- |
| `adjustFactor` 与 `backAdjustFactor` 恒等 | `checks.adjust_factor.adjust_and_back_columns_identical=true`（按字段名取列，非硬编码下标） |
| **最新一条记录的 `foreAdjustFactor` 恒为 1.000000** | 6 个标的（沪/深/创业板，2000–2026 各期）全部成立；`checks.factor_normalization.all_latest_fore_unity=true` |
| 因子表与请求窗口无关 | 全窗口 vs 截断到 `2022-12-31`（`checks.adjust_factor.windows`）：共享 ex-date 的 `backAdjustFactor` 完全相同，`history_rewritten_when_window_changes=false` |
| 无 as-of 查询能力 | 接口只有 `code/start_date/end_date`（ex-date 区间），不接受「截至某时刻」的语义 |
| **窗口内无记录时静默返回空（与标的是否退市无关）** | `sz.000003` 用 `start_date=2000-01-01` 查得 0 行；但同一标的用 `start_date=1990-01-01` 查得 **15 行**（`1991-07-03`..`1996-07-29`），见 `checks.symbol_lifecycle.adjust_factor` |

> 上表最后一行是评审 Major 2 纠正的错误结论。原报告写「已退市标的静默返回空」，
> 实际是**窗口**造成的空；退市判定与因子表无关（§2.1）。

判定：**整张因子表以「查询时刻的最新公司行动」为基准归一化**（`fore` 最新值为 1.0、
`back` 单调累积到最新）。因此由它派生的任何复权序列都随查询时刻变化，
**不是 PIT 可复现的**；「窗口无关」只能排除一种重写方式，不能证明跨时间稳定。

复权价格映射规则（对 5 个标的、1 745 个标的日系统验证，0 失配
——`checks.factor_price_mapping.total_checked_days/ total_mismatches`）：

```text
hfq(D) = raw(D) * backAdjustFactor(最近一个 ex_date <= D)
qfq(D) = raw(D) * foreAdjustFactor(最近一个 ex_date <= D)
```

单点 golden 值（`checks.factor_price_mapping.sample_day`，测试锚定用）：

| 字段 | 值 |
| --- | --- |
| `code` / `day` | `sh.600000` / `2023-01-03` |
| `closes.raw` / `closes.hfq` / `closes.qfq` | `7.23` / `82.67093613` / `6.18469383` |
| 适用因子行 | `ex_date=2022-07-21`、`fore=0.855421`、`back=11.434431` |
| `hfq_over_raw` / `qfq_over_raw` | `11.434431` / `0.855421`（与规则精确一致） |

规则成立意味着：**复权序列完全由「原始价格 + 因子表」决定**。于是 PIT 复现的充分
条件就是把因子表在拉取时快照留痕（§30 append-only），而不是把 provider 的
`adjustflag=1/2` 序列直接写入 artifact。

> 影响 ADR-007：默认 forecast 输入取 `raw`；复权序列是派生视图，必须绑定因子表快照。

### 2.1 退市/生存偏差：价格序列里没有「退市」信号

`checks.symbol_lifecycle`（`sz.000003`，PT金田A）：

| 事实 | 证据（JSON 路径） |
| --- | --- |
| `query_stock_basic` 字段 | `code, code_name, ipoDate, outDate, type, status`（`checks.symbol_lifecycle.stock_basic.fields`） |
| 退市证据 | `sz.000003`：`outDate=2002-06-14`、`status=0`、名称 `PT金田A`（`stock_basic.rows`）；在册对照 `sh.600000`：`outDate=""`、`status=1`（`listed_reference.rows`） |
| 因子表**两种窗口**都在 artifact 里 | 从 `2000-01-01` 起 → `adjust_factor.from_2000.rows=0`（窗口效应）；从 `1990-01-01` 起 → `adjust_factor.from_1990.rows=15`，覆盖 `1991-07-03`..`1996-07-29` |
| 末日及之前的 bar | `history.table` 全量 10 行（`2002-06-03`..`2002-06-14`）：全部 `tradestatus=0`（`suspended_rows=10`）、flat `2.7100`（`flat_ohlc_rows=10`）、`volume=0`（`volume_zero_rows=9`）；末日 `2002-06-14` 的 `volume`/`amount` 为**空串**（`empty_numeric_field_rows`） |

判定：仅看 OHLC 无法区分「长期停牌」与「已退市」——`tradestatus=0` 的 flat bar 两者
都可能出现。退市必须由 `query_stock_basic` 的 `outDate`/`status` 判定（§6.5 生存偏差）。

---

## 3. 指数成分股 `query_hs300_stocks` / `query_zz500_stocks` —— 历史 PIT 可用，但两个指数深度不同

字段：`updateDate, code, code_name`（每行带该名单的生效/发布日期）。同一组日期对
**两个指数分别探测**（评审 Major 3：此前只对 HS300 做日期探测，才漏掉 ZZ500 的深度差异）。

| 事实 | 证据（`checks.index_constituents.<index>.by_probe_date`） |
| --- | --- |
| **HS300 深度约 2006 年初** | `2005-12-30` → 0 行；`2006-01-04` → 300 行、`updateDate=2006-01-02` |
| **ZZ500 深度约 2007 年初（比 HS300 晚一年）** | `2007-01-04` → 0 行；`2007-01-31` → 500 行、`updateDate=2007-01-29` |
| `date=` 返回「当时有效的名单」 | `2006-06-20` → `updateDate=2006-06-19`；`2006-06-30` → `2006-06-26`；`2020-06-30` → `2020-06-29`；`2008-06-30` → `2008-06-30` |
| 名单确实随日期变化（非「永远返回今天」） | 与最新名单的对称差在全部探针上随请求日期变早而单调不减：HS300 `2020-06-30` 差 208、`2008-06-30` 差 432…`2006-01-04` 差 490 |
| 规模正确 | 每个**非空**探针日恰好 HS300=300 行 / ZZ500=500 行；早于可用起点的探针为 0 行（`2005-06-30`、`2005-12-30` 对两个指数都为空，`2006-01-04`..`2007-01-04` 只有 ZZ500 为空） |
| **未来日期静默返回最新名单** | `date=2027-06-30` → HS300 300 行 / ZZ500 500 行、`updateDate=2026-09-21`（= 最新名单）、`error_code=0`、`same_set_as_latest=true` |
| **无法从响应识别 clamp**（评审补测） | `date_echo` 在**每个**探针（含 `2027-06-30`）都等于所请求的日期：服务端原样回显，不提示名单已被改写成最新版本 |

判定：历史成分股满足 PIT 要求，**不需要**引入 Tushare `index_weight` 作为补充数据源；
但两个指数的可用起点不同（HS300 ≈ 2006-01，ZZ500 ≈ 2007-01），**不得共用同一个起点**。
调用侧还必须自行 clamp 并校验 `updateDate <= knowledge_cutoff`：直接用未来日期查询会
拿到最新名单且没有任何错误提示（§29 泄漏风险），且 `data.date` 只是原样回显
（`checks.index_constituents.<index>.by_probe_date[*].date_echo` 与
`future_date_probe.date_echo`），无法据此识别 clamp。

> 影响 ADR-008：UniverseSnapshot 必须记录 `updateDate`，并把「查询日期 > 最新已发布
> 日期」显式失败；universe 的可用起点按**指数分别**声明（HS300 与 ZZ500 不共享）；
> 未来日期防护只能落在调用侧（不能用 `data.date` 反查）。

---

## 4. 停牌与 ST —— 停牌日仍有 bar，必须按 `tradestatus` 过滤

| 事实 | 证据（`checks.suspension`） |
| --- | --- |
| `tradestatus`：`1` 正常 / `0` 停牌 | 3 个标的 2015–2024 共 `total_suspended_rows=202` |
| 停牌行 O=H=L=C（延续前收） | `total_flat_ohlc=202`；`total_repeats_prev_close=201` |
| 停牌行 `volume=0`、`amount=0` | `total_volume_zero=201`（空串不计入）、`total_amount_zero=202`（空串按 0 计），空串仅 `total_empty_volume=1` |
| **存在空串数值字段** | `sh.600816` 1 行停牌 `volume=""`（`codes.sh.600816.suspended_empty_volume=1`）；退市末日 `sz.000003`(2002-06-14) `volume=""`+`amount=""`（§2.1） |
| `isST` 独立于 `tradestatus` | `codes.sh.600816.isST_rows=852`、`codes.sz.000010.isST_rows=300`：ST 可正常交易（`isST=1, tradestatus=1`），也可停牌（`isST=1, tradestatus=0`） |
| 停牌日是「有行但无效」，不是缺行 | 同上 |

判定：字段可用、语义清晰。「有效 bar」必须按 `tradestatus == 1` 且价格有效构造
（RX-KAI-004 已如此实现），不得用「该日无行」推断停牌，也不得假定数值字段非空
（`volume`/`amount` 可能是空串，RX-KAI-004 已解析为 `None`）。

---

## 5. 错误语义 —— 三类失败必须被区分

| 探针 | 预期 | `error_code` | 表现 |
| --- | --- | --- | --- |
| 未知代码 `sh.999999` | `silent_empty` | `0` | **0 行（静默空）** |
| 日期格式 `2020/01/01` | `client_none` | — | **客户端返回 `None`** |
| `start_date > end_date` | `server_error` | `10004009` | 服务端错误消息 |
| 未知字段名 | `server_error` | `10004012` | 服务端错误消息 |
| 未知 `adjustflag` | `server_error` | `10004012` | 服务端错误消息 |
| 全未来窗口 | `silent_empty` | `0` | **0 行（静默空）** |
| `query_adjust_factor` 未知代码 | `silent_empty` | `0` | **0 行（静默空）** |
| 单日节假日查询 | `ok_non_trading_row` | `0` | 1 行 `is_trading_day=0`（正常，非错误） |
| 单日中秋休市查询 | `ok_non_trading_row` | `0` | 1 行 `is_trading_day=0`（正常，非错误） |

9 个探针都带 `expectation` 与 `matches_expectation`，本次全部命中
（`checks.error_semantics.mismatches=[]`、`confirmed=true`）——「记录到错误」不等于
「符合预期」，探针行为与文档不一致时该检查即为 `error`（评审 M5）。

结论：**「静默空」与「返回 None」都不是错误信号**。Provider 层必须把二者显式转换为
领域错误（`ProviderError` / `DataQualityError` / `InsufficientHistoryError`），
否则「无数据」会被当作「无交易日」「无复权因子」继续传播（ADR-010）。

---

## 6. 未覆盖 / 未验证项（不得在实现中默认成立）

1. **盘后数据发布时刻未验证**：本次只能确认「已完成 session 的 bar 事后可取」
   ——`checks.publication_lag` 中三个标的的最后一条 bar 都停在 `2026-09-24`，
   而 `2026-09-25`/`09-26`/`09-27` 在日历上均为非 session（`is_trading_day=0`），
   即 `sessions_without_bars=[]`，**未观测到发布滞后**。但这只说明「与日历一致」，
   无法观测确切发布时刻；因此 `BAO_STOCK_PUBLISHED_AT = 18:00` 仍是**未核实的假设**，
   且方向是**乐观**而非保守：若真实发布晚于 18:00，同日晚间的 run 会用到尚未发布的
   bar。其语义由版本化的 `knowledge_cutoff_policy` 承载（`cutoff-policy-v1`），
   落地前须由 §30 快照与运行时刻校验兜住。
2. **跨时间因子重写**：只能通过「两次拉取 + 快照对比」观测，属 §30 快照机制要
   解决的问题，本次不作为结论；同理，交易日历的 point-in-time 稳定性也只有单次
   采样（§1 的判定已按此收紧）。
3. **成分股名单的发布节奏**：`updateDate` 是名单的生效/发布日期，本次未验证它相对
   `knowledge_cutoff` 的可见时刻（例如 `updateDate=2026-09-21` 的名单是否在当日开盘
   前即可见）。RX-KAI-007 必须显式声明其假设。
4. `query_dividend_data`（分红明细）未探测；默认 `raw` 输入模式下不需要。
5. 北交所（`bj.`）标的未探测。
6. API key / VIP 端点未使用（匿名会话可用，故未必需）。
7. **0.8.9 客户端的挂死行为不可在最终 artifact 中复现**：升级后无法再用旧客户端探测，
   故 §0 中「对端立即关闭 + `send_msg` 空转」只有 0.8.9 会话（本任务早期运行）的
   观察，最终 artifact 只落盘了**端点可达**这一事实（`legacy_endpoint_probe`）。

---

## 7. 对基线决策的直接影响

| 决策点 | 结论 |
| --- | --- |
| ADR-007 复权策略 | 默认 `raw`；复权为派生视图，绑定因子表快照；映射规则 + golden 值可复用 |
| ADR-008 PIT Universe | HS300 历史成分可用（约 2006-01 起），**ZZ500 约 2007-01 起**，两者起点不同；需 clamp + 校验 `updateDate`；退市判定靠 `query_stock_basic` |
| ADR-009 交易日历 | 覆盖 `1990-12-19`..当年度末；跨年与越界查询静默空 → 显式失败；`2026-09-25`（周五、中秋）休市，工作日规则不可替代日历 |
| RX-KAI-004 Provider | 已实现的 `tradestatus` 过滤、`None`/空结果显式失败、空串解析均与实测一致；取数改为 `next()`/`get_row_data()` 翻页并检测静默截断（评审 Major 1 / C1）；login 期间收紧 socket 超时上界（评审 M2） |
| 依赖版本 | `baostock` 从 0.8.9 升至 0.9.4（端点迁移，0.8.9 在本环境不可用） |
