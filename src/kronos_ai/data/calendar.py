"""TradingCalendar 与 A 股 session 语义（基线文档 §4 / §6.3 / §6.6 / §11，ADR-009）。

术语（本定义即版本化语义，修改须同步更新 ADR-009 并重跑依赖 hash 的基准）：

- market session：交易所开市日；Forecast 时间轴与 Label 时间轴的唯一来源
- stock session：个股在某个 market session 实际可交易的 session（停牌即缺席）
- valid bar：stock session 的日线 bar 且价格有效（停牌 / 非正价格行已在 Provider 层剔除）

时间轴唯一真源（§6.6）：

- ForecastPoint.timestamp = 市场未来第 N 个 market session，由 next_sessions 生成
- 个股停牌不改变 forecast 时间轴；label 侧以 SUSPENDED / INSUFFICIENT_FUTURE_BARS 显式标记
- Walk-forward 切点、embargo 边界、停牌标记必须经由同一 TradingCalendar 实例
- 禁止 pandas 工作日推导，禁止调用方各自推导

覆盖范围语义（错误消息前缀是调用方可依赖的约定，见 ADR-009）：

- 日历只对来源提供的区间负责；区间外（早于首个 / 晚于最后一个 session）抛
  CalendarError，消息以 ``outside calendar coverage`` 开头，不猜测、不静默补全
- coverage 不足以给出 count 个未来 session 时抛 CalendarError，消息以
  ``coverage ends at`` 开头
- market_date 本身必须是 market session：origin 落在非 session 上是调用方错误
"""

from __future__ import annotations

from bisect import bisect_left
from datetime import date, datetime
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, field_validator

from kronos_ai.errors import CalendarError, ConfigurationError

Exchange = Literal["SSE", "SZSE", "BSE"]


@runtime_checkable
class TradingCalendar(Protocol):
    """§6.6 固定签名；签名变更属于契约变更。

    实现须声明自己覆盖的 exchange 与 source（provenance，见 ADR-009）。
    """

    def is_session(self, day: date) -> bool:
        """day 是否为覆盖范围内的 market session；范围外抛 CalendarError。"""
        ...

    def next_sessions(self, market_date: date, count: int) -> list[date]:
        """market_date 之后（不含当日）的 count 个 market session，升序。

        market_date 非 session、或 coverage 不足以给出 count 个 session 时抛
        CalendarError；count < 1 抛 ConfigurationError。
        """
        ...


class StaticTradingCalendar(BaseModel):
    """显式 session 序列的日历：测试 fixture、离线回放、合成数据源。

    只如实反映调用方提供的官方 session 序列，不含任何规则推导
    （不按工作日 / 节假日规则生成日期）。生产日历的接入方式由
    RX-KAI-006 BaoStock 能力 spike（query_trade_dates 覆盖范围）结论决定。

    exchange / source 为 provenance：任何进入 artifact 的时间轴都必须能回答
    「这是哪个日历、来自哪里」，因此不提供默认值（§3.5）。
    """

    model_config = ConfigDict(frozen=True)

    exchange: Exchange
    source: str
    sessions: tuple[date, ...]

    @field_validator("source")
    @classmethod
    def _source_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("source must be non-empty")
        return value

    @field_validator("sessions")
    @classmethod
    def _sorted_unique_nonempty(cls, value: tuple[date, ...]) -> tuple[date, ...]:
        if not value:
            raise ValueError("sessions must not be empty")
        if list(value) != sorted(value):
            raise ValueError("sessions must be sorted ascending")
        if len(set(value)) != len(value):
            raise ValueError("sessions must be unique")
        return value

    @property
    def coverage(self) -> tuple[date, date]:
        if not self.sessions:
            # 仅 model_construct / model_copy 绕过校验时可达
            raise CalendarError("calendar has no sessions; construction bypassed validation")
        return (self.sessions[0], self.sessions[-1])

    def is_session(self, day: date) -> bool:
        _require_date(day, "day")
        self._require_covered(day)
        return self._locate(day) >= 0

    def next_sessions(self, market_date: date, count: int) -> list[date]:
        if count < 1:
            raise ConfigurationError(f"count must be >= 1, got {count}")
        _require_date(market_date, "market_date")
        self._require_covered(market_date)
        index = self._locate(market_date)
        if index < 0:
            raise CalendarError(f"{market_date} is not a market session")
        if index + count >= len(self.sessions):
            raise CalendarError(
                f"coverage ends at {self.sessions[-1]}; cannot produce {count} "
                f"sessions after {market_date}"
            )
        return list(self.sessions[index + 1 : index + 1 + count])

    def _locate(self, day: date) -> int:
        """day 在 sessions 中的下标；不存在返回 -1。"""
        index = bisect_left(self.sessions, day)
        if index < len(self.sessions) and self.sessions[index] == day:
            return index
        return -1

    def _require_covered(self, day: date) -> None:
        first, last = self.coverage
        if day < first or day > last:
            raise CalendarError(
                f"{day} outside calendar coverage [{first}, {last}] (exchange={self.exchange})"
            )


def _require_date(value: date, field: str) -> None:
    """datetime 是 date 的子类，误传会在比较处抛裸 TypeError，这里显式拒绝。"""
    if isinstance(value, datetime):
        raise ConfigurationError(f"{field} must be a date, not datetime: {value.isoformat()}")


if TYPE_CHECKING:
    # mypy 结构化校验：实现必须满足 §6.6 契约（runtime_checkable 的 isinstance
    # 只检查成员存在，不校验签名）
    _CONTRACT_ANCHOR: TradingCalendar = StaticTradingCalendar(
        exchange="SSE", source="type-check", sessions=(date(2020, 1, 2),)
    )
