"""KronosRuntime：模型加载与生命周期（基线文档 §17/§32.1；RX-KAI-009）。

职责边界：

- 解析 device / dtype（显式失败，不静默回退）；加载 tokenizer + model 并置于 eval
- 持有模型超参（lookback_bars / max_context / clip）：lookback_bars 决定输入窗口
  与逐窗口归一化统计量，必须进入 config_hash（§8）
- 产出 ModelMetadata：backend / model_id / revision / runtime_version / device-class /
  dtype / config_hash，其中 runtime_version 编码 runtime 契约版本与 torch 版本（§9）

不承担：数据读取、采样循环、raw sample 截取、缓存（分别属于 backend / sampler / cache）。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import nn

from kronos_ai.domain.forecast import ModelMetadata
from kronos_ai.domain.hashing import sha256_hex
from kronos_ai.errors import ConfigurationError
from kronos_ai.forecast.backends.kronos.vendor import Kronos, KronosTokenizer

RUNTIME_CONTRACT_VERSION = "kronos-runtime-v1"

# 2026-09-27 核验的 HF revision（见 vendor/UPSTREAM.md）；默认固定 revision 以保证可复现
DEFAULT_MODEL_ID = "NeoQuasar/Kronos-small"
DEFAULT_MODEL_REVISION = "901c26c1332695a2a8f243eb2f37243a37bea320"
DEFAULT_TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
DEFAULT_TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"

DeviceClass = Literal["cpu", "mps", "cuda"]
DeviceChoice = Literal["auto", "cpu", "mps", "cuda"]
DtypeChoice = Literal["auto", "float32", "float64", "bfloat16", "float16"]

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

# 明确不被平台支持的组合（device-class -> 不可用 dtype）：构造期失败，不等到推理中途
_UNSUPPORTED_DTYPE: dict[DeviceClass, frozenset[str]] = {
    # MPS 不支持 float64 / bfloat16；CPU 上 float16 多数算子未实现
    "mps": frozenset({"float64", "bfloat16"}),
    "cpu": frozenset({"float16"}),
    "cuda": frozenset(),
}


def resolve_device_class(choice: DeviceChoice) -> DeviceClass:
    """device 选择 → device-class；显式指定不可用设备时失败（§3.2）。"""
    if choice == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise ConfigurationError("device='cuda' requested but CUDA is not available")
    if choice == "mps" and not torch.backends.mps.is_available():
        raise ConfigurationError("device='mps' requested but MPS is not available")
    return choice


def resolve_dtype(dtype_choice: DtypeChoice, device_class: DeviceClass) -> torch.dtype:
    """dtype 选择 → torch.dtype；auto = float32（跨设备一致，与 v1 强制 CPU/float32 语义一致）。"""
    resolved = "float32" if dtype_choice == "auto" else dtype_choice
    if resolved in _UNSUPPORTED_DTYPE[device_class]:
        raise ConfigurationError(f"dtype={resolved!r} is not supported on {device_class}")
    return _DTYPE_MAP[resolved]


class KronosRuntimeConfig(BaseModel):
    """Runtime 配置（§32.1 experiment config 的 forecast 段）。

    config_hash 覆盖 model/tokenizer 版本、resolved 前的 device/dtype 选择与全部
    模型超参（含 lookback_bars）；verbose 是纯 UX 开关，不参与 hash。
    """

    model_config = ConfigDict(frozen=True)

    model_id: str = DEFAULT_MODEL_ID
    model_revision: str = DEFAULT_MODEL_REVISION
    tokenizer_id: str = DEFAULT_TOKENIZER_ID
    tokenizer_revision: str = DEFAULT_TOKENIZER_REVISION

    device: DeviceChoice = "auto"
    dtype: DtypeChoice = "auto"

    lookback_bars: int = Field(default=256, ge=1)
    max_context: int = Field(default=512, ge=1)
    clip: float = Field(default=5.0, gt=0)

    verbose: bool = False

    @model_validator(mode="after")
    def _lookback_within_context(self) -> KronosRuntimeConfig:
        if self.lookback_bars > self.max_context:
            raise ValueError(
                "lookback_bars must be <= max_context "
                f"(lookback_bars={self.lookback_bars}, max_context={self.max_context})"
            )
        return self

    @property
    def device_class(self) -> DeviceClass:
        return resolve_device_class(self.device)

    @property
    def torch_dtype(self) -> torch.dtype:
        return resolve_dtype(self.dtype, self.device_class)

    @property
    def config_hash(self) -> str:
        return sha256_hex(self.hashing_payload())

    def hashing_payload(self) -> dict[str, Any]:
        return {
            "kind": "kronos_runtime_config",
            "contract_version": RUNTIME_CONTRACT_VERSION,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
            "device": self.device,
            "dtype": self.dtype,
            "lookback_bars": self.lookback_bars,
            "max_context": self.max_context,
            "clip": self.clip,
        }


def _local_path(source: str) -> Path | None:
    path = Path(source)
    return path if path.exists() else None


def _local_revision(path: Path) -> str:
    """本地权重的 revision 描述：以解析后路径的 hash 标识（可追溯、跨机器稳定）。"""
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()
    return f"local:{digest[:12]}"


class KronosRuntime:
    """已加载的 Kronos tokenizer + model，供 sampler 复用（同步，无全局单例）。"""

    def __init__(
        self,
        *,
        model: nn.Module,
        tokenizer: nn.Module,
        config: KronosRuntimeConfig,
        model_revision: str | None = None,
        tokenizer_revision: str | None = None,
    ) -> None:
        device = torch.device(config.device_class)
        dtype = config.torch_dtype
        self._config = config
        self._model = model.to(device=device, dtype=dtype).eval().requires_grad_(False)
        self._tokenizer = tokenizer.to(device=device, dtype=dtype).eval().requires_grad_(False)
        self._model_revision = model_revision or config.model_revision
        self._tokenizer_revision = tokenizer_revision or config.tokenizer_revision

    @classmethod
    def load(cls, config: KronosRuntimeConfig) -> KronosRuntime:
        """从 HF Hub（固定 revision）或本地目录加载模型；显式失败，无隐式回退。"""
        model_path = _local_path(config.model_id)
        tokenizer_path = _local_path(config.tokenizer_id)

        model = Kronos.from_pretrained(
            str(model_path) if model_path else config.model_id,
            revision=None if model_path else config.model_revision,
        )
        tokenizer = KronosTokenizer.from_pretrained(
            str(tokenizer_path) if tokenizer_path else config.tokenizer_id,
            revision=None if tokenizer_path else config.tokenizer_revision,
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            config=config,
            model_revision=_local_revision(model_path) if model_path else None,
            tokenizer_revision=_local_revision(tokenizer_path) if tokenizer_path else None,
        )

    @property
    def config(self) -> KronosRuntimeConfig:
        return self._config

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def tokenizer(self) -> nn.Module:
        return self._tokenizer

    @property
    def device_class(self) -> DeviceClass:
        return self._config.device_class

    @property
    def dtype_name(self) -> str:
        return str(self._config.torch_dtype).removeprefix("torch.")

    @property
    def lookback_bars(self) -> int:
        return self._config.lookback_bars

    @property
    def runtime_version(self) -> str:
        return f"{RUNTIME_CONTRACT_VERSION}/torch-{torch.__version__}"

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            backend="kronos",
            model_id=self._config.model_id,
            revision=self._model_revision,
            runtime_version=self.runtime_version,
            device=self.device_class,
            dtype=self.dtype_name,
            config_hash=self._config.config_hash,
        )
