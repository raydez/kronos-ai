"""Forecast Artifact Cache（基线文档 §15；ADR-011）。

职责：
- ``ForecastArtifactKey``：把一次 forecast 的全部身份信息（输入数据 / 研究时点 /
  模型 provenance / 采样参数 / **未来 session 时间轴** / **分布 spec**）折叠成确定性
  key，``digest`` 即 artifact_id（§15）。
- ``FileSystemForecastCache``：以 SHA-256 digest 分片落文件；写入走「临时文件 +
  原子 rename」，读缓存命中直接返回既有 ``ForecastResult``，命中时逐身份维校验。
- ``cached_forecast``：§15 的唯一执行点——「同一 key 不得重复推理，除非 force」。

边界（不在本任务内）：
- Parquet artifact 布局 + SQLite run registry（§30–§33）属 RX-KAI-015；
  本模块只定义一个可注入的 ``ForecastCache`` Protocol 与本地文件实现。
- ``--force`` 只绕过读缓存；写入路径仍必须原子（§15）。
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from kronos_ai.data.calendar import TradingCalendar
from kronos_ai.domain.forecast import (
    FORECAST_AGGREGATION_DEFINITION_VERSION,
    FORECAST_CONTRACT_VERSION,
    FORECAST_METRIC_REGISTRY_VERSION,
    ForecastRequest,
    ForecastResult,
)
from kronos_ai.domain.hashing import canonical_json, sha256_hex
from kronos_ai.domain.market import MarketHistory
from kronos_ai.domain.symbols import validate_normalized_symbol
from kronos_ai.domain.time import ensure_shanghai_aware
from kronos_ai.errors import ArtifactError, ConfigurationError
from kronos_ai.forecast.distribution import (
    DEFAULT_DISTRIBUTION_SPEC,
    DistributionSpec,
    distribution_spec_hash,
)

FORECAST_ARTIFACT_KEY_VERSION = "forecast-artifact-key-v1"

# §15 的模型维由 KronosRuntime.artifact_identity() 整体提供；config_hash 额外覆盖
# lookback_bars / max_context / clip / tokenizer 版本（§8：lookback_bars 必须进入
# config_hash），因此这里要求全部键都在、不接受子集——runtime docstring 明确禁止
# 缓存实现只挑部分字段。
REQUIRED_MODEL_IDENTITY_KEYS: tuple[str, ...] = (
    "model_id",
    "model_revision",
    "runtime_version",
    "device_class",
    "dtype",
    "config_hash",
)

_HEX256_LENGTH = 64
_HEX_DIGITS = set("0123456789abcdef")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _HEX256_LENGTH
        and set(value) <= _HEX_DIGITS
    )


def _require_nonempty(value: str, field: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field} must be non-empty")
    return stripped


def _reject_bool(value: object, field: str) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an int, not bool")


class ForecastArtifactKey(BaseModel):
    """一次 forecast 的确定性身份（§15）。

    字段构成属于契约：修改会让既有 artifact 全部失效，必须升级
    ``FORECAST_ARTIFACT_KEY_VERSION`` 并同步 golden 测试。
    ``hashing_payload`` 显式展开每个字段（不依赖字段声明顺序），canonical json 保证
    跨进程 / 跨语言的稳定字节。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    input_data_hash: str
    market_date: date
    knowledge_cutoff: datetime

    horizon: int = Field(ge=1)
    future_sessions: tuple[date, ...]

    model_id: str
    model_revision: str
    runtime_version: str
    device_class: str
    dtype: str
    config_hash: str
    distribution_spec_hash: str

    seed: int = Field(ge=0)
    sample_count: int = Field(ge=1)
    temperature: float = Field(gt=0)
    top_k: int = Field(ge=0)
    top_p: float = Field(gt=0, le=1)

    @field_validator("symbol")
    @classmethod
    def _symbol_normalized(cls, value: str) -> str:
        return validate_normalized_symbol(value)

    @field_validator("input_data_hash", "config_hash", "distribution_spec_hash")
    @classmethod
    def _hash_sha256(cls, value: str, info: ValidationInfo) -> str:
        if not _is_sha256(value):
            raise ValueError(f"{info.field_name} must be a 64-char lowercase sha256 hex digest")
        return value

    @field_validator(
        "model_id",
        "model_revision",
        "runtime_version",
        "device_class",
        "dtype",
        mode="after",
    )
    @classmethod
    def _identity_nonempty(cls, value: str, info: ValidationInfo) -> str:
        return _require_nonempty(value, str(info.field_name))

    @field_validator("knowledge_cutoff")
    @classmethod
    def _cutoff_shanghai_aware(cls, value: datetime) -> datetime:
        return ensure_shanghai_aware(value, "knowledge_cutoff")

    @field_validator("horizon", "seed", "sample_count", "top_k", mode="before")
    @classmethod
    def _ints_not_bool(cls, value: object, info: ValidationInfo) -> object:
        _reject_bool(value, str(info.field_name))
        return value

    @field_validator("temperature", "top_p")
    @classmethod
    def _floats_finite(cls, value: float, info: ValidationInfo) -> float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{info.field_name} must be finite")
        return value

    @model_validator(mode="after")
    def _future_sessions_consistent(self) -> ForecastArtifactKey:
        """未来时间轴属于身份：它既进入模型 stamp（影响 logits），也是 artifact 的 point 时间轴。"""
        if len(self.future_sessions) != self.horizon:
            raise ValueError(
                f"future_sessions has {len(self.future_sessions)} entries != horizon {self.horizon}"
            )
        if list(self.future_sessions) != sorted(self.future_sessions):
            raise ValueError("future_sessions must be sorted ascending")
        if len(set(self.future_sessions)) != len(self.future_sessions):
            raise ValueError("future_sessions must be unique")
        if any(day <= self.market_date for day in self.future_sessions):
            raise ValueError("future_sessions must all fall after market_date")
        return self

    @model_validator(mode="after")
    def _cutoff_on_market_date(self) -> ForecastArtifactKey:
        if self.knowledge_cutoff.date() != self.market_date:
            raise ValueError("knowledge_cutoff must fall on market_date (+08:00)")
        return self

    def hashing_payload(self) -> dict[str, Any]:
        """进入 digest 的完整 payload；kind/contract_version 使契约升级自动失效旧 artifact。"""
        return {
            "kind": "forecast_artifact_key",
            "contract_version": FORECAST_ARTIFACT_KEY_VERSION,
            "forecast_contract_version": FORECAST_CONTRACT_VERSION,
            "symbol": self.symbol,
            "input_data_hash": self.input_data_hash,
            "market_date": self.market_date.isoformat(),
            "knowledge_cutoff": self.knowledge_cutoff.isoformat(),
            "horizon": self.horizon,
            "future_sessions": [day.isoformat() for day in self.future_sessions],
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "runtime_version": self.runtime_version,
            "device_class": self.device_class,
            "dtype": self.dtype,
            "config_hash": self.config_hash,
            "distribution_spec_hash": self.distribution_spec_hash,
            "seed": self.seed,
            "sample_count": self.sample_count,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }

    @property
    def digest(self) -> str:
        """artifact_id：cache 路径与 ``ForecastResult.artifact_id`` 的唯一真源。"""
        return sha256_hex(self.hashing_payload())


def build_forecast_artifact_key(
    *,
    history: MarketHistory,
    request: ForecastRequest,
    model_identity: Mapping[str, str],
    calendar: TradingCalendar,
    distribution_spec: DistributionSpec = DEFAULT_DISTRIBUTION_SPEC,
) -> ForecastArtifactKey:
    """从「数据快照 + 请求 + 模型 identity + 日历 + 分布 spec」组装 §15 key。

    ``model_identity`` 必须是 ``KronosRuntime.artifact_identity()`` 的整体返回值：
    缺键显式失败（§3.2），缺 ``config_hash`` 会让 lookback_bars 变化不失效缓存。
    history 与 request 的 symbol / market_date / cutoff 必须一致，否则是调用方 bug。

    ``calendar`` 不是可省略的辅助输入：未来 session 时间轴由它生成，既作为模型 stamp
    参与推理，也是 artifact 中 ForecastPoint 的时间线。两个日历即使输入相同也会给出
    不同输出，因此时间轴必须进 key。用**与 sampler 同一**的 ``calendar.next_sessions``
    派生，保证 key 与推理实际使用的时间轴一致。
    """
    missing = [key for key in REQUIRED_MODEL_IDENTITY_KEYS if key not in model_identity]
    if missing:
        raise ConfigurationError(
            "model_identity is missing required keys "
            f"{missing}; expected the full KronosRuntime.artifact_identity() mapping"
        )
    if history.symbol != request.symbol:
        raise ConfigurationError(
            f"history symbol {history.symbol!r} != request symbol {request.symbol!r}"
        )
    if history.market_date != request.market_date:
        raise ConfigurationError(
            f"history market_date {history.market_date} != request market_date {request.market_date}"
        )
    if history.knowledge_cutoff != request.knowledge_cutoff:
        raise ConfigurationError(
            "history knowledge_cutoff "
            f"{history.knowledge_cutoff.isoformat()} != request knowledge_cutoff "
            f"{request.knowledge_cutoff.isoformat()}"
        )

    # calendar.next_sessions 在覆盖不足时抛 CalendarError（§6.6）；长度不对属于契约违背。
    future_sessions = tuple(calendar.next_sessions(request.market_date, request.horizon))
    if len(future_sessions) != request.horizon:
        raise ConfigurationError(
            f"calendar returned {len(future_sessions)} sessions for horizon "
            f"{request.horizon}; next_sessions must honour the requested count"
        )

    identity = {
        key: _require_identity_str(model_identity[key], key)
        for key in REQUIRED_MODEL_IDENTITY_KEYS
    }

    sampling = request.sampling
    return ForecastArtifactKey(
        symbol=request.symbol,
        input_data_hash=history.data_hash,
        market_date=request.market_date,
        knowledge_cutoff=request.knowledge_cutoff,
        horizon=request.horizon,
        future_sessions=future_sessions,
        distribution_spec_hash=distribution_spec_hash(distribution_spec),
        seed=sampling.seed,
        sample_count=sampling.sample_count,
        temperature=sampling.temperature,
        top_k=sampling.top_k,
        top_p=sampling.top_p,
        **identity,
    )


def _require_identity_str(value: object, key: str) -> str:
    """模型维必须是字符串；``str(None) == "None"`` 会静默造出一个合法但错误的身份。"""
    if not isinstance(value, str):
        raise ConfigurationError(
            f"model_identity[{key!r}] must be a string, got {type(value).__name__}"
        )
    return value


class ForecastCache(Protocol):
    """可注入的 forecast 缓存（§15）；实现必须保证写入原子性。"""

    def get(self, key: ForecastArtifactKey, *, force: bool = False) -> ForecastResult | None: ...

    def put(self, key: ForecastArtifactKey, result: ForecastResult) -> None: ...


def _artifact_path(root: Path, digest: str) -> Path:
    return root / digest[:2] / f"{digest}.json"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """临时文件 + fsync + 原子 rename；并发写同一 key 时读者只会看到完整文件。

    临时文件用 ``tempfile.mkstemp``（``O_EXCL`` + 随机后缀）创建：唯一性由内核保证，
    不依赖 32-bit 随机数；否则同进程多线程并发写同一 key 时名字可能碰撞，
    导致一个 writer 的 ``finally`` 删掉另一个 writer 正在写的临时文件。
    代价是文件权限为 mkstemp 的 0600（本地研究 artifact，比默认 0644 更严格）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        # os.replace 成功后 tmp 已不存在；异常路径下清理半成品。
        tmp.unlink(missing_ok=True)


class FileSystemForecastCache:
    """本地文件缓存（§15 的「临时文件 + 原子 rename」约定）。

    布局：``<root>/<digest 前两位>/<digest>.json``，内容为 canonical json 的
    ``ForecastResult``——相同 key 的重复写入字节完全一致，last-write-wins 无害。
    读到的 artifact 与 key 不一致（artifact_id / input_data_hash / 损坏）时显式失败，
    不静默重新推理（§3.2）。
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, key: ForecastArtifactKey) -> Path:
        return _artifact_path(self._root, key.digest)

    def get(self, key: ForecastArtifactKey, *, force: bool = False) -> ForecastResult | None:
        if force:
            return None
        path = self.path_for(key)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return None
        result = self._decode(path, raw)
        self._verify(key, path, result)
        return result

    def put(self, key: ForecastArtifactKey, result: ForecastResult) -> None:
        self._verify(key, None, result)
        payload = canonical_json(result.model_dump(mode="json")).encode("utf-8")
        _atomic_write_bytes(self.path_for(key), payload)

    @staticmethod
    def _decode(path: Path, raw: bytes) -> ForecastResult:
        try:
            return ForecastResult.model_validate_json(raw)
        except Exception as exc:  # ValueError / pydantic ValidationError 统一转 ArtifactError
            raise ArtifactError(f"cached forecast artifact at {path} is corrupt: {exc}") from exc

    @staticmethod
    def _verify(key: ForecastArtifactKey, path: Path | None, result: ForecastResult) -> None:
        """校验 artifact 的全部身份维与 key 一致（§14/§15）。

        不只比 artifact_id/input_data_hash：``compute`` 可能忘记回填 provenance，或
        用与 key 不符的模型/采样/分布 spec 产出结果。这里逐维比对，任一不符即
        :class:`ArtifactError`（§3.2），不静默接受「身份不明」的产物。
        """
        location = str(path) if path is not None else f"artifact for key {key.digest}"
        problems: list[str] = []

        def check(ok: bool, message: str) -> None:
            if not ok:
                problems.append(message)

        check(
            result.artifact_id == key.digest,
            f"artifact_id {result.artifact_id!r} != key digest {key.digest!r}",
        )
        check(result.symbol == key.symbol, f"symbol {result.symbol!r} != key {key.symbol!r}")
        check(
            result.market_date == key.market_date,
            f"market_date {result.market_date} != key {key.market_date}",
        )
        check(
            result.knowledge_cutoff == key.knowledge_cutoff,
            f"knowledge_cutoff {result.knowledge_cutoff.isoformat()} != key "
            f"{key.knowledge_cutoff.isoformat()}",
        )
        check(
            result.input_data_hash == key.input_data_hash,
            f"input_data_hash {result.input_data_hash!r} != key {key.input_data_hash!r}",
        )
        check(
            result.distribution.horizon == key.horizon,
            f"distribution.horizon {result.distribution.horizon} != key horizon {key.horizon}",
        )
        check(
            result.distribution.sample_count == key.sample_count,
            f"distribution.sample_count {result.distribution.sample_count} != key "
            f"sample_count {key.sample_count}",
        )
        check(
            result.distribution.distribution_spec_hash == key.distribution_spec_hash,
            f"distribution_spec_hash {result.distribution.distribution_spec_hash!r} != key "
            f"{key.distribution_spec_hash!r}",
        )
        # 直接用版本号再校一次：spec hash 已折叠两者，但显式比对能给出可读错误，
        # 并防止未来重构把版本号从 hash 中悄然移除而无人发现。
        check(
            result.distribution.metric_definition_version == FORECAST_METRIC_REGISTRY_VERSION,
            f"metric_definition_version {result.distribution.metric_definition_version!r} != "
            f"current {FORECAST_METRIC_REGISTRY_VERSION!r}",
        )
        check(
            result.distribution.aggregation_definition_version
            == FORECAST_AGGREGATION_DEFINITION_VERSION,
            f"aggregation_definition_version "
            f"{result.distribution.aggregation_definition_version!r} != "
            f"current {FORECAST_AGGREGATION_DEFINITION_VERSION!r}",
        )
        for field in ("seed", "sample_count", "temperature", "top_k", "top_p"):
            actual = getattr(result.sampling, field)
            expected = getattr(key, field)
            check(actual == expected, f"sampling.{field} {actual!r} != key {expected!r}")
        for key_field, model_field in (
            ("model_id", "model_id"),
            ("model_revision", "revision"),
            ("runtime_version", "runtime_version"),
            ("device_class", "device"),
            ("dtype", "dtype"),
            ("config_hash", "config_hash"),
        ):
            actual = getattr(result.model, model_field)
            expected = getattr(key, key_field)
            check(actual == expected, f"model.{model_field} {actual!r} != key {key_field} {expected!r}")
        for sample in result.samples:
            timeline = tuple(point.timestamp.date() for point in sample.points)
            check(
                timeline == key.future_sessions,
                f"sample {sample.sample_id} timeline {timeline} != key future_sessions "
                f"{key.future_sessions}",
            )

        if problems:
            raise ArtifactError(f"{location}: " + "; ".join(problems))


def cached_forecast(
    cache: ForecastCache,
    key: ForecastArtifactKey,
    compute: Callable[[], ForecastResult],
    *,
    force: bool = False,
) -> tuple[ForecastResult, bool]:
    """§15 的唯一执行点：命中则返回缓存，否则推理一次并原子写入。

    返回 ``(result, was_cached)``。``force=True`` 只绕过读缓存，写入路径不变。
    ``compute`` 产出的 ``ForecastResult.artifact_id`` 必须等于 ``key.digest``——由
    ``cache.put`` 校验；不一致说明推理层没有正确回填 provenance，属显式失败。
    """
    cached = cache.get(key, force=force)
    if cached is not None:
        return cached, True
    result = compute()
    cache.put(key, result)
    return result, False
