from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from kronos_ai.domain.forecast import ModelMetadata
from kronos_ai.errors import ConfigurationError
from kronos_ai.forecast.backends.kronos.runtime import (
    DEFAULT_MODEL_REVISION,
    RUNTIME_CONTRACT_VERSION,
    KronosRuntime,
    KronosRuntimeConfig,
    resolve_device_class,
    resolve_dtype,
)
from kronos_ai.forecast.backends.kronos.vendor import Kronos, KronosTokenizer


def config(**overrides: object) -> KronosRuntimeConfig:
    fields: dict[str, object] = {"model_id": "tiny", "tokenizer_id": "tiny"}
    fields.update(overrides)
    return KronosRuntimeConfig(**fields)  # type: ignore[arg-type]


class TestKronosRuntimeConfig:
    def test_defaults(self) -> None:
        cfg = config()
        assert cfg.lookback_bars == 256
        assert cfg.max_context == 512
        assert cfg.clip == 5.0
        assert cfg.verbose is False
        assert cfg.model_revision == DEFAULT_MODEL_REVISION

    def test_lookback_must_fit_context(self) -> None:
        with pytest.raises(ValidationError, match="lookback_bars must be <= max_context"):
            config(lookback_bars=600, max_context=512)

    def test_lookback_positive(self) -> None:
        with pytest.raises(ValidationError):
            config(lookback_bars=0)

    def test_clip_positive(self) -> None:
        with pytest.raises(ValidationError):
            config(clip=0)

    def test_hash_stable_and_sensitive(self) -> None:
        base = config().config_hash
        assert config().config_hash == base
        assert config(lookback_bars=128).config_hash != base
        assert config(model_revision="other").config_hash != base
        assert config(clip=4.0).config_hash != base
        assert config(max_context=256, lookback_bars=128).config_hash != base

    def test_verbose_not_hashed(self) -> None:
        # verbose 是纯 UX 开关，不改变数值结果，因此不进入 cache 键
        assert config(verbose=True).config_hash == config(verbose=False).config_hash

    def test_hash_payload_is_versioned(self) -> None:
        payload = config().hashing_payload()
        assert payload["contract_version"] == RUNTIME_CONTRACT_VERSION
        assert payload["lookback_bars"] == 256


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
        assert resolve_dtype("auto", "cpu") is torch.float32
        assert resolve_dtype("auto", "mps") is torch.float32
        assert resolve_dtype("auto", "cuda") is torch.float32

    def test_unsupported_dtype_combinations_fail(self) -> None:
        with pytest.raises(ConfigurationError, match="not supported on mps"):
            resolve_dtype("float64", "mps")
        with pytest.raises(ConfigurationError, match="not supported on mps"):
            resolve_dtype("bfloat16", "mps")
        with pytest.raises(ConfigurationError, match="not supported on cpu"):
            resolve_dtype("float16", "cpu")

    def test_cpu_float64_supported(self) -> None:
        assert resolve_dtype("float64", "cpu") is torch.float64


class TestKronosRuntime:
    def test_metadata_fields(self, tiny_runtime: KronosRuntime) -> None:
        meta = tiny_runtime.metadata()
        assert isinstance(meta, ModelMetadata)
        assert meta.backend == "kronos"
        assert meta.model_id == "tiny"
        assert meta.revision == "test"
        assert meta.device == tiny_runtime.device_class
        assert meta.dtype == "float32"
        assert meta.config_hash == tiny_runtime.config.config_hash
        assert meta.runtime_version.startswith(RUNTIME_CONTRACT_VERSION)
        assert f"torch-{torch.__version__}" in meta.runtime_version

    def test_model_is_eval_and_frozen(self, tiny_runtime: KronosRuntime) -> None:
        assert not tiny_runtime.model.training
        assert not tiny_runtime.tokenizer.training
        assert all(not p.requires_grad for p in tiny_runtime.model.parameters())

    def test_lookback_bars_exposed(self, tiny_runtime: KronosRuntime) -> None:
        assert tiny_runtime.lookback_bars == 256

    def test_load_from_local_directory(
        self, tmp_path: Path, tiny_model: Kronos, tiny_tokenizer: KronosTokenizer
    ) -> None:
        model_dir = tmp_path / "model"
        tokenizer_dir = tmp_path / "tokenizer"
        tiny_model.save_pretrained(model_dir)
        tiny_tokenizer.save_pretrained(tokenizer_dir)

        runtime = KronosRuntime.load(
            KronosRuntimeConfig(
                model_id=str(model_dir),
                tokenizer_id=str(tokenizer_dir),
                model_revision="unused",
                tokenizer_revision="unused",
            )
        )
        meta = runtime.metadata()
        assert meta.revision.startswith("local:")
        # 同一路径的 revision 描述稳定
        assert meta.revision == runtime.metadata().revision

    def test_load_from_local_directory_survives_roundtrip(
        self, tmp_path: Path, tiny_model: Kronos, tiny_tokenizer: KronosTokenizer
    ) -> None:
        model_dir = tmp_path / "model"
        tokenizer_dir = tmp_path / "tokenizer"
        tiny_model.save_pretrained(model_dir)
        tiny_tokenizer.save_pretrained(tokenizer_dir)

        runtime = KronosRuntime.load(
            KronosRuntimeConfig(model_id=str(model_dir), tokenizer_id=str(tokenizer_dir))
        )
        # 形状与参数可加载（不比对数值：safetensors 往返即可加载）
        assert isinstance(runtime.model, Kronos)
        assert isinstance(runtime.tokenizer, KronosTokenizer)
