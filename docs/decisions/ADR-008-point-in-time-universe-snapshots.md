# ADR-008: Point-in-Time Universe Snapshots

- 状态：Accepted
- 日期：2026-09-27
- 对应任务：RX-KAI-007（前置：RX-KAI-006 BaoStock 能力 spike）
- 对应基线：`Kronos-AI_v2_Greenfield_Rewrite_设计与落地方案.md` §6.0、§6.2、§6.5、§29、§30、§32、§36（ADR-008）

## 背景

§6.2 把 universe 定为 benchmark 的样本边界，并直接禁止「用今天的 CSI300 成分股回测
2020 年」。§6.5 进一步要求保留真实历史状态、不因今天退市而从历史样本中删除。
因此 universe 必须能按每个 Benchmark Date 复原，且「当时有效」这件事必须可举证。

RX-KAI-006 的 spike（§3，原始证据 `docs/spike/baostock-capability-raw.json`）给出四项
决定性事实：

1. **`date=` 返回当时有效的名单，而非「永远返回今天」**：与最新名单的对称差随请求
   日期变早而单调不减。HS300 全部 7 个非空探针：`2020-06-30` 差 208、`2008-06-30`
   差 432、`2007-01-31` 差 454、`2007-01-04` 差 464、`2006-06-30` 与 `2006-06-20` 各
   差 486、`2006-01-04` 差 490；ZZ500 三个非空探针 `2020-06-30` 差 642、`2008-06-30`
   差 892、`2007-01-31` 差 906，同样不减。且每行的 `updateDate` 给出该名单的修订
   日期（粒度：日）。
2. **两个指数的可用深度不同**：HS300 `2005-12-30` → 0 行、`2006-01-04` → 300 行；
   ZZ500 `2007-01-04` → 0 行、`2007-01-31` → 500 行。ZZ500 比 HS300 晚约一年，
   **不得共用同一个起点**（上一轮评审正是因为只探测 HS300 才漏掉这一差异）。
3. **未来日期被静默 clamp**：`date=2027-06-30` 返回最新名单（`updateDate=2026-09-21`）、
   `error_code=0`、`same_set_as_latest=true`；且 `result.date` 在**每个**探针（含未来
   日期）都原样回显所请求的日期（`by_probe_date[*].date_echo`、`future_date_probe.date_echo`），
   响应里没有任何可据以识别 clamp 的字段。这是 §29 级的泄漏风险。
4. **规模固定**：每个非空探针恰好 HS300=300 行 / ZZ500=500 行；早于可用起点的探针
   返回 0 行且 `error_code=0`——「空」是能力边界，不是「当时成分股较少」。

结论：历史成分股满足 PIT 要求，**不需要**引入 Tushare `index_weight`
（§6.0 的「数据源不满足时先引入补充数据源」条款不触发）。

## 决策

### 1. `UniverseSnapshot` 契约（§6.2 字段 + `update_date`）

```python
class UniverseSnapshot(BaseModel):  # frozen
    universe_id: str  # "hs300" / "zz500"
    effective_date: date  # 请求的交易日（研究对象时点）
    symbols: tuple[str, ...]  # 规范 6 位代码，去重 + 升序
    source: str  # "baostock:query_hs300_stocks"
    version: str  # 数据源数据集版本
    update_date: date  # 名单修订日（PIT 证据，必录）
```

相对基线 §6.2 的两处偏离，均为「更严」：

- `symbols` 用不可变 `tuple` 且强制**去重 + 升序**（规范顺序），使快照 hash 可复现
  （§30 append-only 快照与 §32 run metadata 都要求可复现）；
- 增加 `update_date`（spike §3 明确要求）：`effective_date` 是「问哪一天」，
  `update_date` 是「拿到的是哪一版」，两者不同（如 `2020-06-30` → `2020-06-29`）。
  不变式 **`update_date <= effective_date`**：晚于请求时点才修订的名单当日尚未生效，
  用它就是未来函数。

### 2. 按 Benchmark Date 逐次取快照，起点按指数分别声明

- 每个 Benchmark Date 单独加载，**禁止**跨日期复用同一份「最新」名单；
- HS300 与 ZZ500 的可用起点不共享，也不允许一方默认另一方；实现侧以
  `EXPECTED_MEMBER_COUNT`（300 / 500）逐指数校验规模；
- 名单规模不符（残缺名单）一律显式失败，不得当作「历史时点成分股较少」。

### 3. 加载器的 PIT 规则与错误分类

| 情形 | 处理 | 错误类 |
| --- | --- | --- |
| 未知 universe_id | 拒绝，列出受支持的 id | `ConfigurationError`（调用/配置错误） |
| `effective_date` 晚于 `knowledge_cutoff` 当日 | 拒绝，且**不发请求** | `ConfigurationError` |
| `effective_date` 早于可用起点（0 行） | 显式失败，**不得**用最新名单替代 | `UniverseError`（能力边界） |
| 规模 ≠ 300 / 500、重复代码、代码不合规、缺 `updateDate` 列、`updateDate` 混杂、`updateDate > effective_date` | 显式失败 | `DataQualityError` |
| 服务端错误码、客户端异常、字段元数据缺失、硬超时 | 显式失败（沿用 RX-KAI-004/006 语义） | `ProviderError` |

`UniverseError` 是本 ADR 新增的错误类：它既不是瞬时故障（`ProviderError`），也不是
数据缺陷（`DataQualityError`），而是**数据源能力边界**——语义上区别于 §6.4 的
`InsufficientHistoryError`（后者是标的自身历史不足的策略判定）。

### 4. 未来日期防护只能落在调用侧

因为响应既不报错、`data.date` 也只是回显（事实 3），防护规则为：

```text
update_date <= effective_date <= knowledge_cutoff.date()
```

其中 `effective_date <= knowledge_cutoff.date()` 在加载器内、发请求**之前**执行。
`knowledge_cutoff` 还必须满足与 `ResearchTime`（§5）相同的时区约束：**aware 且偏移为
+08:00**——否则 `.date()` 会漂移到另一天，未来 universe 又可绕过该检查。此处不做
「按 offset 换算日期」的宽容处理，非 +08:00 一律拒绝（`ConfigurationError`）。

规则本身带版本号 `UNIVERSE_PIT_POLICY_VERSION`（`src/kronos_ai/data/universe.py`），
随快照一并进入 run metadata（§32）：任何后续语义变更（如改为 `update_date <
knowledge_cutoff.date()`）都必须升版本，使历史 run 的复现状态可分辨，不得静默改变。

已知的未核实假设（与 `BAO_STOCK_PUBLISHED_AT` 同方向，记录备查）：名单修订的**日内
发布时刻未知**；同日修订（`update_date == effective_date`，如 `2008-06-30` 探针）被
当作在该日 18:00 的 cutoff 之前已知，方向是**乐观**的——这正是 v1 规则的含义，收紧
即为版本升级。

### 5. 退市与 ST 保留在历史名单中（§6.5）

快照就是**当时真实生效的名单**：不因今天退市而剔除、不因今天 ST 而剔除。
退市/ST 的可用性判定（`query_stock_basic` 的 `outDate`/`status`、`tradestatus`/`isST`）
属于标的级策略，由 §6.3/§6.5 与 ADR-009 处理，不在 universe 层过滤。

### 6. 实现边界

- 契约（`UniverseSnapshot` / `UniverseLoader` / `EXPECTED_MEMBER_COUNT`）在
  `src/kronos_ai/data/universe.py`；
- 适配器 `BaoStockUniverseLoader` 在 `src/kronos_ai/infrastructure/providers/baostock.py`，
  与价格 Provider **共享**同一进程级会话、引用计数、会话锁与硬超时策略
  （成分股查询同样走 `send_msg`，同样有 EOF 空转形态）；
- 快照的 append-only 持久化属 RX-KAI-015 artifact store，本任务不落盘。

## 后果

1. **跨指数 benchmark 的起点必须取二者更晚者**（ZZ500 → 约 2007-01-31 之后），
   由 dataset builder 显式声明，不允许静默用短历史样本充当长历史。
2. 每个快照自带 `update_date`，任何进入 artifact 的 universe 都能回答「哪一版名单、
   何时修订」，满足 §29/§30 的可复现要求。
3. **可核实的边界仍有限**：真实首个可用日期位于探测边界之间（HS300 在
   `2005-12-30`..`2006-01-04` 之间、ZZ500 在 `2007-01-04`..`2007-01-31` 之间），
   实现只做实报（0 行即失败），不猜测起点。
4. 不引入 Tushare：若将来需要**指数权重**（而非仅成分名单）用于加权 benchmark，
   属新需求，需另开 spike 与 ADR。
5. 成分股名单的修订粒度是「日」，与退市/停牌判定共用同一 TradingCalendar 与
   knowledge_cutoff 语义；三日历/停牌策略见 ADR-009。
