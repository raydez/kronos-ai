"""KronosRuntime 单元测试（RX-KAI-009）。

覆盖三类契约：

- 配置期显式失败：来源写法（``local:`` / ``hf:`` / 裸路径）、revision 固定、精度仅 float32
- 构造期核验：metadata 报告实态（device / dtype / eval / resolved revision），
  config_hash 覆盖 resolved 身份并进入 §15 artifact identity
- 本地加载：revision 由 checkpoint 内容派生（换权重必须变），严格加载失败归一为
  ModelLoadError，成功路径逐 key 比对权重
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from pydantic import ValidationError
from torch import nn

from kronos_ai.domain.forecast import ModelMetadata
from kronos_ai.errors import ConfigurationError, ModelLoadError
from kronos_ai.forecast.backends.kronos.runtime import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    DEFAULT_TOKENIZER_ID,
    RUNTIME_CONTRACT_VERSION,
    VENDOR_UPSTREAM_COMMIT,
    KronosRuntime,
    KronosRuntimeConfig,
    ModelSource,
    resolve_device_class,
    resolve_dtype,
    resolve_source,
)
from kronos_ai.forecast.backends.kronos.vendor import Kronos, KronosTokenizer

PIN = "deadbeef" * 5


def config(**overrides: object) -> KronosRuntimeConfig:
    fields: dict[str, object] = {
        "model_revision": PIN,
        "tokenizer_revision": PIN,
        "device": "cpu",
    }
    fields.update(overrides)
    return KronosRuntimeConfig(**fields)  # type: ignore[arg-type]


def build_runtime(model: nn.Module, tokenizer: nn.Module, **overrides: object) -> KronosRuntime:
    return KronosRuntime(model=model, tokenizer=tokenizer, config=config(**overrides))


class TestKronosRuntimeConfig:
    def test_defaults(self) -> None:
        cfg = KronosRuntimeConfig()
        assert cfg.model_id == DEFAULT_MODEL_ID
        assert cfg.tokenizer_id == DEFAULT_TOKENIZER_ID
        assert cfg.model_revision is None
        assert cfg.tokenizer_revision is None
        assert cfg.lookback_bars == 256
        assert cfg.max_context == 512
        assert cfg.clip == 5.0
        assert cfg.verbose is False

    def test_lookback_must_fit_context(self) -> None:
        with pytest.raises(ValidationError, match="lookback_bars must be <= max_context"):
            config(lookback_bars=600, max_context=512)

    def test_lookback_positive(self) -> None:
        with pytest.raises(ValidationError):
            config(lookback_bars=0)

    def test_clip_positive(self) -> None:
        with pytest.raises(ValidationError):
            config(clip=0)

    def test_config_is_frozen(self) -> None:
        cfg = config()
        with pytest.raises(ValidationError):
            cfg.lookback_bars = 128  # type: ignore[misc]

    @pytest.mark.parametrize("revision", ["main", "v1.0", "deadbeef", "DEADBEEF" * 5, PIN + "0"])
    def test_revision_must_be_full_commit_sha(self, revision: str) -> None:
        # branch/tag 会移动；大小写不规范或长度不符也会让 provenance 不可比对
        with pytest.raises(ValidationError, match="40-char lowercase commit sha"):
            config(model_revision=revision)

    def test_revision_may_be_none_only_for_default_pin(self) -> None:
        assert KronosRuntimeConfig(model_id=DEFAULT_MODEL_ID).model_revision is None
        with pytest.raises(ConfigurationError, match="model_revision is required"):
            resolve_source(
                "someone/other-model",
                None,
                role="model",
                default_id=DEFAULT_MODEL_ID,
                default_revision=DEFAULT_MODEL_REVISION,
            )

    def test_local_source_rejects_explicit_revision(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="must not be set for a local: source"):
            config(model_id=f"local:{tmp_path}", model_revision=PIN)

    def test_local_source_requires_existing_directory(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope"
        with pytest.raises(ValidationError, match="is not an existing directory"):
            config(model_id=f"local:{missing}")

    def test_bare_local_path_is_refused(self, tmp_path: Path) -> None:
        # 裸路径若被当作 repo id 会静默走网络，必须显式拒绝（§3.2）
        with pytest.raises(ValidationError, match="use the 'local:' prefix"):
            config(model_id=str(tmp_path))

    @pytest.mark.parametrize(
        "shape",
        [
            "./weights/kronos",
            "../weights/kronos",
            "~/checkpoints/kronos",
            "/tmp/kronos",
            "weights\\kronos",
            "C:/weights/kronos",
            "C:/weights",
            "D:checkpoints",
            "weights/kronos/2026-09",
            "checkpoints/model.safetensors",
            "checkpoints/model.bin",
            "checkpoints/model.pt",
            "config/kronos.json",
        ],
    )
    def test_path_shaped_bare_ids_are_refused(self, shape: str) -> None:
        # 启发式只覆盖明显路径形态；weights/kronos 这类无法与 repo id 区分的裸 id
        # 仍按 Hub 处理（见 KronosRuntimeConfig docstring），此处不假装能识破
        with pytest.raises(ValidationError, match="looks like a local path"):
            config(model_id=shape)

    def test_repo_id_shaped_bare_id_is_accepted(self) -> None:
        assert config(model_id="org/name").model_id == "org/name"

    def test_hf_prefix_requires_repo_id(self) -> None:
        with pytest.raises(ValidationError, match="must be a repo id"):
            config(model_id="hf:")
        with pytest.raises(ValidationError):
            config(model_id="hf:/absolute/path")

    def test_empty_source_refused(self) -> None:
        with pytest.raises(ValidationError, match="must be non-empty"):
            config(tokenizer_id="   ")

    def test_hf_prefix_is_stripped_in_resolved_source(self) -> None:
        source = resolve_source(
            f"hf:{DEFAULT_MODEL_ID}",
            None,
            role="model",
            default_id=DEFAULT_MODEL_ID,
            default_revision=DEFAULT_MODEL_REVISION,
        )
        assert source.identifier == DEFAULT_MODEL_ID
        assert source.revision == DEFAULT_MODEL_REVISION

    def test_device_and_dtype_properties(self) -> None:
        cfg = config(device="cpu", dtype="float32")
        assert cfg.device_class == "cpu"
        assert cfg.torch_dtype is torch.float32


class TestDeviceDtypeResolution:
    def test_auto_resolves_to_available_device(self) -> None:
        resolved = resolve_device_class("auto")
        if torch.cuda.is_available():
            assert resolved == "cuda"
        elif torch.backends.mps.is_available():
            assert resolved == "mps"
        else:
            assert resolved == "cpu"

    def test_explicit_unavailable_device_fails(self) -> None:
        if not torch.cuda.is_available():
            with pytest.raises(ConfigurationError, match="CUDA is not available"):
                resolve_device_class("cuda")
        if not torch.backends.mps.is_available():
            with pytest.raises(ConfigurationError, match="MPS is not available"):
                resolve_device_class("mps")

    def test_auto_dtype_is_float32_everywhere(self) -> None:
        assert resolve_dtype("auto") is torch.float32

    def test_explicit_float32_supported(self) -> None:
        assert resolve_dtype("float32") is torch.float32

    @pytest.mark.parametrize("dtype", ["float64", "bfloat16", "float16"])
    def test_non_float32_dtypes_fail_explicitly(self, dtype: str) -> None:
        # vendored pipeline 的 indices_to_bits 硬编码 .float()（vendor/kronos.py:134），
        # 其余精度在 post_quant_embed（vendor/kronos.py:170）处 dtype mismatch
        with pytest.raises(ConfigurationError, match="Only float32 is supported"):
            resolve_dtype(dtype)  # type: ignore[arg-type]


class _UnconvertibleModule(nn.Module):
    """`_apply` 被掏空的模块：`.to()` 无法改写内部张量，模拟注入的非标准模块。"""

    def __init__(self, *, dtype: torch.dtype = torch.float32, device: str = "cpu") -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, dtype=dtype, device=device))

    def _apply(self, fn: object, recurse: bool = True) -> _UnconvertibleModule:
        return self


class _RefusesEvalModule(nn.Module):
    """`train()` 被掏空的模块：`eval()` 之后 training 仍为 True。"""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2))

    def train(self, mode: bool = True) -> _RefusesEvalModule:
        return self


class TestRuntimeConstruction:
    def test_metadata_fields(self, tiny_runtime: KronosRuntime) -> None:
        meta = tiny_runtime.metadata()
        assert isinstance(meta, ModelMetadata)
        assert meta.backend == "kronos"
        assert meta.revision == tiny_runtime.config.model_revision
        assert meta.device == "cpu"
        assert meta.dtype == "float32"
        assert meta.config_hash == tiny_runtime.config_hash
        assert meta.runtime_version.startswith(RUNTIME_CONTRACT_VERSION)
        assert f"upstream-{VENDOR_UPSTREAM_COMMIT[:12]}" in meta.runtime_version
        assert f"torch-{torch.__version__}" in meta.runtime_version

    def test_model_is_eval_and_frozen(self, tiny_runtime: KronosRuntime) -> None:
        assert not tiny_runtime.model.training
        assert not tiny_runtime.tokenizer.training
        assert all(not p.requires_grad for p in tiny_runtime.model.parameters())

    def test_lookback_bars_exposed(self, tiny_runtime: KronosRuntime) -> None:
        assert tiny_runtime.lookback_bars == 256

    def test_metadata_reports_resolved_device_not_intent(self, tiny_runtime_factory: Any) -> None:
        # device="auto" 时 metadata 必须报告实际解析结果，而不是 "auto"。
        # 必须用 factory 新建模块：KronosRuntime.__init__ 的 .to() 会原地改写模块设备，
        # 借用 session 级夹具试 auto 会把它们永久迁到 MPS/CUDA（同 d 类夹具污染）
        runtime = tiny_runtime_factory(device="auto")
        assert runtime.metadata().device == resolve_device_class("auto")

    def test_injected_modules_require_explicit_revision(
        self, tiny_model: Kronos, tiny_tokenizer: KronosTokenizer
    ) -> None:
        # 注入路径没有 checkpoint 内容可校验，沿用默认 pin 会让 provenance 说谎
        with pytest.raises(ConfigurationError, match="model revision must be pinned"):
            KronosRuntime(
                model=tiny_model,
                tokenizer=tiny_tokenizer,
                config=KronosRuntimeConfig(device="cpu"),
            )

    def test_unverifiable_module_is_refused(self, tiny_tokenizer: KronosTokenizer) -> None:
        # 核验必须真的执行：无参数模块无法确认 dtype/device，直接拒绝
        with pytest.raises(ModelLoadError, match="neither parameters nor buffers"):
            build_runtime(nn.Identity(), tiny_tokenizer)

    def test_module_that_resists_dtype_conversion_is_refused(
        self, tiny_tokenizer: KronosTokenizer
    ) -> None:
        with pytest.raises(ModelLoadError, match="floating dtypes"):
            build_runtime(_UnconvertibleModule(dtype=torch.float64), tiny_tokenizer)

    def test_module_that_resists_device_move_is_refused(
        self, tiny_tokenizer: KronosTokenizer
    ) -> None:
        with pytest.raises(ModelLoadError, match="devices \\['meta'\\]"):
            build_runtime(_UnconvertibleModule(device="meta"), tiny_tokenizer)

    def test_module_that_resists_eval_is_refused(self, tiny_tokenizer: KronosTokenizer) -> None:
        with pytest.raises(ModelLoadError, match="must be in eval mode"):
            build_runtime(_RefusesEvalModule(), tiny_tokenizer)

    def test_config_hash_stable_and_sensitive(
        self, tiny_model: Kronos, tiny_tokenizer: KronosTokenizer
    ) -> None:
        base = build_runtime(tiny_model, tiny_tokenizer).config_hash
        assert build_runtime(tiny_model, tiny_tokenizer).config_hash == base
        # 每个超参单独敏感：合并改动会被「至少一个字段参与哈希」蒙混过关
        assert build_runtime(tiny_model, tiny_tokenizer, lookback_bars=128).config_hash != base
        assert build_runtime(tiny_model, tiny_tokenizer, max_context=256).config_hash != base
        assert build_runtime(tiny_model, tiny_tokenizer, clip=4.0).config_hash != base
        assert (
            build_runtime(tiny_model, tiny_tokenizer, model_revision="cafe" * 10).config_hash
            != base
        )
        assert (
            build_runtime(tiny_model, tiny_tokenizer, tokenizer_revision="cafe" * 10).config_hash
            != base
        )

    def test_artifact_identity_matches_metadata(self, tiny_runtime: KronosRuntime) -> None:
        identity = tiny_runtime.artifact_identity()
        assert set(identity) == {
            "model_id",
            "model_revision",
            "runtime_version",
            "device_class",
            "dtype",
            "config_hash",
        }
        meta = tiny_runtime.metadata()
        assert identity["model_id"] == meta.model_id
        assert identity["model_revision"] == meta.revision
        assert identity["runtime_version"] == meta.runtime_version
        assert identity["device_class"] == meta.device
        assert identity["dtype"] == meta.dtype
        assert identity["config_hash"] == meta.config_hash


class TestLocalLoading:
    def _save(self, tmp_path: Path, tiny_model: Kronos, tiny_tokenizer: KronosTokenizer) -> None:
        tiny_model.save_pretrained(tmp_path / "model")
        tiny_tokenizer.save_pretrained(tmp_path / "tokenizer")

    def test_load_from_local_directory(
        self, tmp_path: Path, tiny_model: Kronos, tiny_tokenizer: KronosTokenizer
    ) -> None:
        self._save(tmp_path, tiny_model, tiny_tokenizer)
        runtime = KronosRuntime.load(
            KronosRuntimeConfig(
                model_id=f"local:{tmp_path / 'model'}",
                tokenizer_id=f"local:{tmp_path / 'tokenizer'}",
                device="cpu",
            )
        )
        meta = runtime.metadata()
        assert isinstance(runtime.model, Kronos)
        assert isinstance(runtime.tokenizer, KronosTokenizer)
        assert meta.revision.startswith("local:")
        assert meta.revision == runtime.metadata().revision
        assert runtime.model.training is False

    def test_loaded_weights_equal_saved_weights(
        self, tmp_path: Path, tiny_model: Kronos, tiny_tokenizer: KronosTokenizer
    ) -> None:
        self._save(tmp_path, tiny_model, tiny_tokenizer)
        runtime = KronosRuntime.load(
            KronosRuntimeConfig(
                model_id=f"local:{tmp_path / 'model'}",
                tokenizer_id=f"local:{tmp_path / 'tokenizer'}",
                device="cpu",
            )
        )
        for name, module in (("model", tiny_model), ("tokenizer", tiny_tokenizer)):
            saved = module.state_dict()
            loaded = (
                runtime.model.state_dict() if name == "model" else runtime.tokenizer.state_dict()
            )
            assert set(loaded) == set(saved)
            for key, value in saved.items():
                assert torch.equal(loaded[key], value), f"{name}.{key} differs after roundtrip"

    def test_local_revision_is_content_addressed(
        self, tmp_path: Path, tiny_model: Kronos, tiny_tokenizer: KronosTokenizer
    ) -> None:
        self._save(tmp_path, tiny_model, tiny_tokenizer)
        model_id = f"local:{tmp_path / 'model'}"
        first = resolve_source(
            model_id,
            None,
            role="model",
            default_id=DEFAULT_MODEL_ID,
            default_revision=DEFAULT_MODEL_REVISION,
        )
        again = resolve_source(
            model_id,
            None,
            role="model",
            default_id=DEFAULT_MODEL_ID,
            default_revision=DEFAULT_MODEL_REVISION,
        )
        assert first.revision == again.revision

        # 同目录换权重必须换 revision，否则 provenance 会指向错的 checkpoint。
        # 改动落在从磁盘重新加载的副本上：session 级夹具实例在别的用例里还要用，
        # 就地改权重会污染它们（此前版本的 bug）。
        copy = Kronos.from_pretrained(tmp_path / "model")
        with torch.no_grad():
            next(copy.parameters()).add_(1.0)
        copy.save_pretrained(tmp_path / "model")
        changed = resolve_source(
            model_id,
            None,
            role="model",
            default_id=DEFAULT_MODEL_ID,
            default_revision=DEFAULT_MODEL_REVISION,
        )
        assert changed.revision != first.revision

    def test_directory_without_checkpoint_files_fails(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ConfigurationError, match=r"contains neither config\.json"):
            resolve_source(
                f"local:{empty}",
                None,
                role="model",
                default_id=DEFAULT_MODEL_ID,
                default_revision=DEFAULT_MODEL_REVISION,
            )

    def test_local_revision_covers_non_safetensors_weights(
        self, tmp_path: Path, tiny_model: Kronos, tiny_tokenizer: KronosTokenizer
    ) -> None:
        """非 safetensors 权重也进入内容 revision（防御性：加载器若改用 bin，revision 不能漏）。"""
        self._save(tmp_path, tiny_model, tiny_tokenizer)
        model_dir = tmp_path / "model"

        def revision() -> str:
            return resolve_source(
                f"local:{model_dir}",
                None,
                role="model",
                default_id=DEFAULT_MODEL_ID,
                default_revision=DEFAULT_MODEL_REVISION,
            ).revision

        before = revision()
        (model_dir / "pytorch_model.bin").write_bytes(b"not-identical-to-the-safetensors-weights")
        assert revision() != before

    def test_mismatched_checkpoint_fails_explicitly(
        self, tmp_path: Path, tiny_model: Kronos, tiny_tokenizer: KronosTokenizer
    ) -> None:
        """shape 不一致即使非 strict 加载也会报错，这里断言的是失败语义的归一化。"""
        self._save(tmp_path, tiny_model, tiny_tokenizer)
        model_dir = tmp_path / "model"
        payload = json.loads((model_dir / "config.json").read_text())
        payload["d_model"] = payload["d_model"] * 2
        (model_dir / "config.json").write_text(json.dumps(payload))

        with pytest.raises(ModelLoadError, match="failed to load model"):
            KronosRuntime.load(
                KronosRuntimeConfig(
                    model_id=f"local:{model_dir}",
                    tokenizer_id=f"local:{tmp_path / 'tokenizer'}",
                    device="cpu",
                )
            )

    def test_missing_weight_key_fails_only_because_of_strict(
        self, tmp_path: Path, tiny_model: Kronos, tiny_tokenizer: KronosTokenizer
    ) -> None:
        """strict=True 的独立证据：删掉一个权重 key，缺 key 只能被 strict 检出。

        去掉 strict 后 huggingface_hub 只会 warning 并加载其余权重，本用例即失败
        （已实测：把 ``_load_pretrained`` 的 strict 改成 False，本用例变红）。
        """
        from safetensors.torch import load_file, save_file

        self._save(tmp_path, tiny_model, tiny_tokenizer)
        model_dir = tmp_path / "model"
        weight_file = next(model_dir.glob("*.safetensors"))
        tensors = load_file(weight_file)
        dropped = next(iter(tensors))
        tensors.pop(dropped)
        save_file(tensors, weight_file)

        with pytest.raises(ModelLoadError, match="failed to load model"):
            KronosRuntime.load(
                KronosRuntimeConfig(
                    model_id=f"local:{model_dir}",
                    tokenizer_id=f"local:{tmp_path / 'tokenizer'}",
                    device="cpu",
                )
            )

    def test_model_source_declared_requires_revision(self) -> None:
        with pytest.raises(ConfigurationError, match="revision must be pinned"):
            ModelSource.declared("some/repo", None, role="model")

    def test_model_source_declared_rejects_unpinned_revision(self) -> None:
        # 注入路径不经过 pydantic 校验：格式门禁必须在 runtime 侧同样成立
        with pytest.raises(ConfigurationError, match="40-char lowercase commit sha"):
            ModelSource.declared("some/repo", "main", role="model")

    def test_resolve_source_rejects_unpinned_revision(self) -> None:
        with pytest.raises(ConfigurationError, match="40-char lowercase commit sha"):
            resolve_source(
                "org/name",
                "v1.0",
                role="model",
                default_id=DEFAULT_MODEL_ID,
                default_revision=DEFAULT_MODEL_REVISION,
            )
