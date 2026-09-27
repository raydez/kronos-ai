"""Point-in-Time universe 契约（RX-KAI-007，基线文档 §6.2 / §6.5，ADR-008）。

禁止「用今天的成分股回测 2020 年」（§6.2）：每个 Benchmark Date 必须读取当时
有效的成分股名单，且快照必须能证明「该名单在 knowledge_cutoff 时已知」。

数据源事实（`docs/spike/baostock-capability.md` §3，artifact 可逐项复核）：

- `query_hs300_stocks` / `query_zz500_stocks` 的 `date=` 返回**当时生效**的名单
  （非「永远返回今天」），每行带 `updateDate`＝该名单的修订日期（粒度：日）；
- 可用起点按指数分别成立且**只知探测边界**：HS300 `2005-12-30` 为空、`2006-01-04`
  为 300 行；ZZ500 `2007-01-04` 为空、`2007-01-31` 为 500 行。真实起点在两者之间，
  未逐日探测，因此两个指数不得共用起点，也不得互相默认；
- 未来日期（2027-06-30）**静默返回最新名单**、`error_code=0`、`data.date` 原样回显
  所查询的日期，因此泄漏防护不能依赖服务端报错，必须由调用侧用 knowledge_cutoff
  与 `updateDate` 自行把关（§29）。

本模块只固化快照契约与加载器契约；成分股的持久化（append-only 快照）属
RX-KAI-015 artifact store。
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from kronos_ai.domain.symbols import validate_normalized_symbol

UNIVERSE_ID_HS300 = "hs300"
UNIVERSE_ID_ZZ500 = "zz500"

# 指数名单的成员数固定：数量不符说明拿到了残缺名单，必须显式失败而不是当作
# 「某个历史时点成分股较少（§6.2）」继续跑。
EXPECTED_MEMBER_COUNT: dict[str, int] = {
    UNIVERSE_ID_HS300: 300,
    UNIVERSE_ID_ZZ500: 500,
}

# PIT 判定规则版本：update_date <= effective_date <= knowledge_cutoff 当日。
#
# 其中「同日修订可用」是**未核实的乐观假设**：名单修订的日内发布时刻未知
# （spike §3 观察到 updateDate 与请求日同日的样本），同日修订被当作当日 cutoff
# 之前已知。收紧规则（改为严格早于 cutoff 当日，或引入发布时刻常量）属于语义
# 变更，必须升版本号并同步 ADR-008，不得静默改变历史 run 的复现状态。
UNIVERSE_PIT_POLICY_VERSION = "universe-pit-policy-v1"


class UniverseSnapshot(BaseModel):
    """某个 Benchmark Date 当时有效的成分股名单（§6.2 + ADR-008）。

    - ``effective_date``：请求的交易日（研究对象时点），不是名单修订日；
    - ``update_date``：数据源给出的名单修订日期（PIT 证据，ADR-008 要求必录）；
      不变式 ``update_date <= effective_date``——晚于请求时点的名单在当日尚未生效，
      用它就是未来函数；
    - ``symbols``：规范 6 位代码，去重且升序（规范化顺序，保证 hash 可复现）；
    - ``source`` / ``version``：provenance（§32 run metadata 需要能回答名单版本）。
    """

    model_config = ConfigDict(frozen=True)

    universe_id: str
    effective_date: date
    symbols: tuple[str, ...]
    source: str
    version: str
    update_date: date

    @field_validator("universe_id", "source", "version")
    @classmethod
    def _nonempty(cls, value: str, info: object) -> str:
        if not value.strip():
            field = getattr(info, "field_name", "value")
            raise ValueError(f"{field} must be non-empty")
        return value

    @field_validator("symbols")
    @classmethod
    def _symbols_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("symbols must not be empty")
        normalized = tuple(validate_normalized_symbol(symbol) for symbol in value)
        if len(set(normalized)) != len(normalized):
            raise ValueError("symbols must be unique")
        if list(normalized) != sorted(normalized):
            raise ValueError("symbols must be sorted ascending (canonical order)")
        return normalized

    @model_validator(mode="after")
    def _list_effective_at_request_date(self) -> UniverseSnapshot:
        if self.update_date > self.effective_date:
            raise ValueError(
                f"update_date {self.update_date} is after effective_date {self.effective_date}; "
                "a list revised later than the requested point in time was not yet in force"
            )
        return self

    @property
    def member_count(self) -> int:
        return len(self.symbols)


@runtime_checkable
class UniverseLoader(Protocol):
    """历史成分股加载器契约：按 universe 与时点取「当时有效」的名单快照。

    实现必须：

    - 拒绝早于数据源可用起点的时点，且**不**用当前名单替代（§6.2）；
    - 校验名单规模与该 index 的固定成员数一致；
    - 保证返回快照在 ``knowledge_cutoff`` 时点已知（``update_date`` 不得晚于
      ``effective_date``，``effective_date`` 不得晚于 cutoff 当日）；
    - 任何数据源失败显式抛错（ADR-010），不返回空快照。

    退市/ST 标的**不**从历史名单中剔除（§6.5）：返回的是当时真实生效的名单。
    """

    def load(
        self,
        universe_id: str,
        effective_date: date,
        knowledge_cutoff: datetime,
    ) -> UniverseSnapshot: ...
