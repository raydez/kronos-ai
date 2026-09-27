"""KronosRuntime：模型加载与生命周期（基线文档 §17/§32.1；RX-KAI-009）。

职责边界：

- 解析 device / dtype（显式失败，不静默回退）；加载 tokenizer + model 并置于 eval
- 解析模型来源与 revision（``hf:`` / ``local:`` 前缀；hf 必须 40 位 commit sha，
  local 由 checkpoint 内容 sha256 派生），加载后核验模块实际 device / dtype / 训练态
- 持有模型超参（lookback_bars / max_context / clip）：lookback_bars 决定输入窗口
  与逐窗口归一化统计量，必须进入 config_hash（§8）
- 产出 ModelMetadata：backend / model_id / revision / runtime_version / device-class /
  dtype / config_hash；runtime_version 编码 runtime 契约版本、vendored 上游 commit
  与 torch 版本（§9）。config_hash 覆盖 resolved 后的身份（不是配置意图），
  与 artifact_identity() 一起供 §15 ForecastArtifactKey 使用

不承担：数据读取、采样循环、raw sample 截取、缓存（分别属于 backend / sampler / cache）。

## 精度策略（实测依据：2026-09-27，torch 2.14.0）

只有 float32 可用。vendored 上游 ``KronosTokenizer.decode`` 的 ``indices_to_bits``
硬编码 ``x.float()``（vendor/kronos.py:134），随后 ``post_quant_embed``
（vendor/kronos.py:170）的 Linear 权重必须是 float32；float64 / bfloat16 / float16
实测在 Linear 处 dtype mismatch。MPS 上 bfloat16 / float16 更会在 MPSNDArray
矩阵乘中触发断言、直接 abort 进程（不可捕获）。vendor 代码不可修改（ADR-020），
故其余精度在配置期显式拒绝，不做静默降精度。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator
from torch import nn

from kronos_ai.domain.forecast import ModelMetadata
from kronos_ai.domain.hashing import sha256_hex
from kronos_ai.errors import ConfigurationError, ModelLoadError
from kronos_ai.forecast.backends.kronos.vendor import Kronos, KronosTokenizer

RUNTIME_CONTRACT_VERSION = "kronos-runtime-v1"

# vendored 上游代码的 commit（见 vendor/UPSTREAM.md）：vendor 升级必须同步此常量，
# 否则 runtime_version 不变会让旧 artifact 被误判为仍然有效（§15）
VENDOR_UPSTREAM_COMMIT = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"

# 2026-09-27 核验的 HF revision（见 vendor/UPSTREAM.md）；默认固定 revision 以保证可复现
DEFAULT_MODEL_ID = "NeoQuasar/Kronos-small"
DEFAULT_MODEL_REVISION = "901c26c1332695a2a8f243eb2f37243a37bea320"
DEFAULT_TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
DEFAULT_TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"

LOCAL_PREFIX = "local:"
HF_PREFIX = "hf:"

CONFIG_FILE_NAME = "config.json"
# 权重形态覆盖 huggingface_hub 可能加载的全部格式。实测锁定版本（huggingface_hub 2.0.0）
# 对本地目录只读 model.safetensors，bin 回退仅存在于 Hub 分支——加宽属防御性冗余：
# 加载器若改走 pytorch_model.bin，内容 revision 仍覆盖真实权重，不会指向旧内容。
WEIGHT_FILE_PATTERNS: tuple[str, ...] = ("*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt")

DeviceClass = Literal["cpu", "mps", "cuda"]
DeviceChoice = Literal["auto", "cpu", "mps", "cuda"]
DtypeChoice = Literal["auto", "float32", "float64", "bfloat16", "float16"]

_SHA40 = re.compile(r"[0-9a-f]{40}")

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


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


def resolve_dtype(dtype_choice: DtypeChoice) -> torch.dtype:
    """dtype 选择 → torch.dtype；auto = float32（唯一被 vendored pipeline 支持的精度）。

    不依赖 device：float64/bfloat16/float16 在任何设备上都被拒绝（原因见下方报错文本），
    因此没有「按设备放行」的分支。
    """
    resolved = "float32" if dtype_choice == "auto" else dtype_choice
    if resolved != "float32":
        raise ConfigurationError(
            f"dtype={resolved!r} cannot be used with the vendored Kronos pipeline: "
            "tokenizer.decode hardcodes float32 in indices_to_bits (vendor/kronos.py:134) "
            "before post_quant_embed (vendor/kronos.py:170), so float64/bfloat16/float16 "
            "hit a Linear dtype mismatch; on MPS bfloat16/float16 also abort the process "
            "inside MPSNDArray. Only float32 is supported."
        )
    return _DTYPE_MAP[resolved]


_CHECKPOINT_SUFFIXES = (".json", ".safetensors", ".bin", ".pt", ".pth", ".ckpt")


def _looks_like_path(source: str) -> bool:
    """裸 id 中显而易见的路径形态（显式报错比让 Hub 404 更早、更直白）。

    注意这只是启发式：``weights/kronos`` 既可能是相对目录也可能是合法 repo id，
    语法上无法区分——因此规则是「本地来源必须显式加 ``local:``」，而不是
    「任何裸路径都会被识破」。
    """
    if source.startswith((".", "~", "/", "\\")) or "\\" in source or ".." in source:
        return True
    if ":" in source:  # Windows 盘符（C:/weights）；HF repo id 字符集不含冒号
        return True
    if source.endswith(_CHECKPOINT_SUFFIXES):
        return True
    return source.count("/") > 1


def _validate_source_id(value: str, field: str) -> str:
    """校验 model/tokenizer 来源写法；裸路径显式拒绝，不做隐式 Hub 回退。"""
    source = value.strip()
    if not source:
        raise ValueError(f"{field} must be non-empty")
    if source.startswith(LOCAL_PREFIX):
        path = Path(source[len(LOCAL_PREFIX) :])
        if not path.is_dir():
            raise ValueError(f"{field} local path {path} is not an existing directory")
        return source
    if source.startswith(HF_PREFIX):
        repo_id = source[len(HF_PREFIX) :]
        if not repo_id or repo_id.startswith("/"):
            raise ValueError(
                f"{field} hf: source must be a repo id such as 'org/name'; got {value!r}"
            )
        return source
    if Path(source).exists() or _looks_like_path(source):
        raise ValueError(
            f"{field} {source!r} looks like a local path; use the '{LOCAL_PREFIX}' prefix "
            f"(or '{HF_PREFIX}' for a Hub repo id) so the source is explicit"
        )
    return source


def _local_content_revision(path: Path) -> str:
    """本地 checkpoint 的内容标识：同目录换权重必须改变 revision，否则 provenance 会串味。"""
    files: list[Path] = []
    config_file = path / CONFIG_FILE_NAME
    if config_file.is_file():
        files.append(config_file)
    weight_files = {
        file for pattern in WEIGHT_FILE_PATTERNS for file in path.glob(pattern) if file.is_file()
    }
    files.extend(sorted(weight_files))
    if not files:
        raise ConfigurationError(
            f"local source {path} contains neither {CONFIG_FILE_NAME} nor any of "
            f"{list(WEIGHT_FILE_PATTERNS)}"
        )
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode("utf-8"))
        digest.update(b"\x00")
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        digest.update(b"\x00")
    return f"local:{digest.hexdigest()[:12]}"


def _require_pinned_revision(revision: str, role: str) -> str:
    """revision 一律 40 位 commit sha；branch / tag 会移动，破坏可复现性（§54）。"""
    if not _SHA40.fullmatch(revision):
        raise ConfigurationError(
            f"{role} revision {revision!r} is not a full 40-char lowercase commit sha; "
            "branch/tag names are refused"
        )
    return revision


@dataclass(frozen=True)
class ModelSource:
    """解析后的模型来源；metadata 与 config_hash 只使用 resolved 值。"""

    role: str
    kind: Literal["hf", "local"]
    identifier: str
    revision: str
    local_path: Path | None = None

    @classmethod
    def declared(cls, declared_id: str, declared_revision: str | None, *, role: str) -> ModelSource:
        """注入已加载模块（不经过 Hub）时的来源描述。

        revision 必须由调用方显式给出：注入路径没有内容可校验，默认 pin 会让
        provenance 说谎。格式在此再次强制，不依赖调用方经过 pydantic 校验。
        """
        if declared_revision is None:
            raise ConfigurationError(
                f"{role} revision must be pinned when constructing KronosRuntime directly; "
                "use KronosRuntime.load for Hub default pins"
            )
        return cls(
            role=role,
            kind="local" if declared_id.startswith(LOCAL_PREFIX) else "hf",
            identifier=declared_id,
            revision=_require_pinned_revision(declared_revision, role),
        )


def resolve_source(
    declared_id: str,
    declared_revision: str | None,
    *,
    role: str,
    default_id: str,
    default_revision: str,
) -> ModelSource:
    """来源声明 → resolved 来源；revision 缺失或不可派生时显式失败。"""
    if declared_id.startswith(LOCAL_PREFIX):
        path = Path(declared_id[len(LOCAL_PREFIX) :])
        local_name = path.resolve().name or str(path.resolve())
        return ModelSource(
            role=role,
            kind="local",
            identifier=f"{LOCAL_PREFIX}{local_name}",
            revision=_local_content_revision(path),
            local_path=path,
        )
    repo_id = declared_id[len(HF_PREFIX) :] if declared_id.startswith(HF_PREFIX) else declared_id
    revision = declared_revision
    if revision is None:
        if repo_id != default_id:
            raise ConfigurationError(
                f"{role}_revision is required for {repo_id!r}: pin a 40-char commit sha "
                f"(only {default_id} ships a default pin)"
            )
        revision = default_revision
    return ModelSource(
        role=role,
        kind="hf",
        identifier=repo_id,
        revision=_require_pinned_revision(revision, role),
    )


def _load_pretrained(factory: Any, source: ModelSource) -> nn.Module:
    """加载 checkpoint；strict=True 保证 config 与权重逐 key 匹配，失败归一为 ModelLoadError。"""
    target = str(source.local_path) if source.kind == "local" else source.identifier
    try:
        return factory.from_pretrained(  # type: ignore[no-any-return]
            target,
            revision=source.revision if source.kind == "hf" else None,
            strict=True,
        )
    except Exception as exc:
        raise ModelLoadError(
            f"failed to load {source.role} from {target!r} @ {source.revision}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _verify_module(
    module: nn.Module, *, role: str, device_class: DeviceClass, dtype_name: str
) -> None:
    """核验模块实际状态：metadata 必须报告事实而不是配置意图（§3.2/§9）。"""
    tensors = [*module.parameters(), *module.buffers()]
    if not tensors:
        raise ModelLoadError(f"{role} module exposes neither parameters nor buffers")
    float_dtypes = {_dtype_name(t.dtype) for t in tensors if t.is_floating_point()}
    if float_dtypes != {dtype_name}:
        raise ModelLoadError(
            f"{role} module floating dtypes {sorted(float_dtypes)} != requested {dtype_name}"
        )
    devices = {t.device.type for t in tensors}
    if devices != {device_class}:
        raise ModelLoadError(
            f"{role} module is on devices {sorted(devices)} != requested {device_class}"
        )
    if module.training or any(p.requires_grad for p in module.parameters()):
        raise ModelLoadError(f"{role} module must be in eval mode with gradients disabled")


class KronosRuntimeConfig(BaseModel):
    """Runtime 配置（§32.1 experiment config 的 forecast 段）。

    model_id / tokenizer_id 支持三种写法：

    - ``NeoQuasar/Kronos-small``：裸 repo id，走 HuggingFace Hub
    - ``hf:org/name``：显式 Hub 来源
    - ``local:/path/to/dir``：本地 checkpoint 目录

    裸 id 中的明显路径形态被拒绝（``.`` / ``~`` / 分隔符开头、含 ``..``、多级
    分隔符、checkpoint 后缀，以及任何真实存在的路径），必须加 ``local:`` 前缀。
    这是启发式而非证明：``weights/kronos`` 既可能是相对目录也可能是合法 repo id，
    语法上不可区分——因此纪律是「本地来源必须显式」，未被识破的裸 id 一律按 Hub
    repo id 处理，不存在时由 Hub 显式失败（404 → ModelLoadError，§3.2）。

    revision 必须是 40 位 commit sha：branch / tag 会移动，破坏可复现性（§54）。
    默认 None 表示使用 DEFAULT_*_REVISION，且只对默认 model/tokenizer id 生效；
    ``local:`` 来源的 revision 由 checkpoint 内容派生，因此不允许显式给出。

    verbose 是纯 UX 开关（进度显示），不参与任何 hash。
    """

    model_config = ConfigDict(frozen=True)

    model_id: str = DEFAULT_MODEL_ID
    model_revision: str | None = None
    tokenizer_id: str = DEFAULT_TOKENIZER_ID
    tokenizer_revision: str | None = None

    device: DeviceChoice = "auto"
    dtype: DtypeChoice = "auto"

    lookback_bars: int = Field(default=256, ge=1)
    max_context: int = Field(default=512, ge=1)
    clip: float = Field(default=5.0, gt=0)

    verbose: bool = False

    @field_validator("model_id", "tokenizer_id")
    @classmethod
    def _source_wellformed(cls, value: str, info: ValidationInfo) -> str:
        return _validate_source_id(value, str(info.field_name))

    @field_validator("model_revision", "tokenizer_revision")
    @classmethod
    def _revision_pinned(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        if not _SHA40.fullmatch(value):
            raise ValueError(
                f"{info.field_name} must be a full 40-char lowercase commit sha "
                f"(branch/tag names are refused); got {value!r}"
            )
        return value

    @model_validator(mode="after")
    def _internally_consistent(self) -> KronosRuntimeConfig:
        if self.lookback_bars > self.max_context:
            raise ValueError(
                "lookback_bars must be <= max_context "
                f"(lookback_bars={self.lookback_bars}, max_context={self.max_context})"
            )
        for id_field, rev_field in (
            ("model_id", "model_revision"),
            ("tokenizer_id", "tokenizer_revision"),
        ):
            declared_id = getattr(self, id_field)
            revision = getattr(self, rev_field)
            if declared_id.startswith(LOCAL_PREFIX) and revision is not None:
                raise ValueError(
                    f"{rev_field} must not be set for a {LOCAL_PREFIX} source; "
                    "the revision is derived from checkpoint content"
                )
        return self

    @property
    def device_class(self) -> DeviceClass:
        return resolve_device_class(self.device)

    @property
    def torch_dtype(self) -> torch.dtype:
        return resolve_dtype(self.dtype)


class KronosRuntime:
    """已加载的 Kronos tokenizer + model，供 sampler 复用（同步，无全局单例）。"""

    def __init__(
        self,
        *,
        model: nn.Module,
        tokenizer: nn.Module,
        config: KronosRuntimeConfig,
        model_source: ModelSource | None = None,
        tokenizer_source: ModelSource | None = None,
    ) -> None:
        device_class = config.device_class
        dtype = config.torch_dtype
        dtype_name = _dtype_name(dtype)
        self._config = config
        self._model = model.to(device=torch.device(device_class), dtype=dtype).eval()
        self._tokenizer = tokenizer.to(device=torch.device(device_class), dtype=dtype).eval()
        for module in (self._model, self._tokenizer):
            module.requires_grad_(False)
        _verify_module(self._model, role="model", device_class=device_class, dtype_name=dtype_name)
        _verify_module(
            self._tokenizer, role="tokenizer", device_class=device_class, dtype_name=dtype_name
        )
        self._device_class = device_class
        self._dtype_name = dtype_name
        self._model_source = model_source or ModelSource.declared(
            config.model_id, config.model_revision, role="model"
        )
        self._tokenizer_source = tokenizer_source or ModelSource.declared(
            config.tokenizer_id, config.tokenizer_revision, role="tokenizer"
        )

    @classmethod
    def load(cls, config: KronosRuntimeConfig) -> KronosRuntime:
        """从 Hub（固定 revision）或本地目录加载；失败一律显式，无隐式回退。"""
        model_source = resolve_source(
            config.model_id,
            config.model_revision,
            role="model",
            default_id=DEFAULT_MODEL_ID,
            default_revision=DEFAULT_MODEL_REVISION,
        )
        tokenizer_source = resolve_source(
            config.tokenizer_id,
            config.tokenizer_revision,
            role="tokenizer",
            default_id=DEFAULT_TOKENIZER_ID,
            default_revision=DEFAULT_TOKENIZER_REVISION,
        )
        return cls(
            model=_load_pretrained(Kronos, model_source),
            tokenizer=_load_pretrained(KronosTokenizer, tokenizer_source),
            config=config,
            model_source=model_source,
            tokenizer_source=tokenizer_source,
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
        return self._device_class

    @property
    def dtype_name(self) -> str:
        return self._dtype_name

    @property
    def lookback_bars(self) -> int:
        return self._config.lookback_bars

    @property
    def runtime_version(self) -> str:
        return (
            f"{RUNTIME_CONTRACT_VERSION}/upstream-{VENDOR_UPSTREAM_COMMIT[:12]}"
            f"/torch-{torch.__version__}"
        )

    @property
    def config_hash(self) -> str:
        """resolved 配置的 hash：device-class / dtype 取实态，local 来源取内容 revision。

        刻意不含 ``runtime_version``（torch 版本 + vendored 上游 commit）：运行环境面由
        ``runtime_version`` 单独表达，二者一起进入 §15 的 ForecastArtifactKey。
        合并进 config_hash 会让「同一配置跑在不同 torch 上」与「配置变了」无法区分。
        """
        return sha256_hex(self._hashing_payload())

    def _hashing_payload(self) -> dict[str, Any]:
        return {
            "kind": "kronos_runtime",
            "contract_version": RUNTIME_CONTRACT_VERSION,
            "model_id": self._model_source.identifier,
            "model_revision": self._model_source.revision,
            "tokenizer_id": self._tokenizer_source.identifier,
            "tokenizer_revision": self._tokenizer_source.revision,
            "device": self._device_class,
            "dtype": self._dtype_name,
            "lookback_bars": self._config.lookback_bars,
            "max_context": self._config.max_context,
            "clip": self._config.clip,
        }

    def artifact_identity(self) -> dict[str, str]:
        """§15 ForecastArtifactKey 的模型维。

        缓存实现必须整体取用本方法的返回值，不得只挑子集（config_hash 已覆盖
        lookback_bars / max_context / clip 与 tokenizer 版本）。
        """
        return {
            "model_id": self._model_source.identifier,
            "model_revision": self._model_source.revision,
            "runtime_version": self.runtime_version,
            "device_class": self._device_class,
            "dtype": self._dtype_name,
            "config_hash": self.config_hash,
        }

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            backend="kronos",
            model_id=self._model_source.identifier,
            revision=self._model_source.revision,
            runtime_version=self.runtime_version,
            device=self._device_class,
            dtype=self._dtype_name,
            config_hash=self.config_hash,
        )
