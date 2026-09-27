"""共享测试夹具：tiny 随机初始化 Kronos 模型（不下载权重、不要求网络）。

tiny 模型参数被压缩到最小可运行规模；所有 dropout = 0 保证采样路径确定性。
"""

from __future__ import annotations

import pytest

from kronos_ai.forecast.backends.kronos.runtime import KronosRuntime, KronosRuntimeConfig
from kronos_ai.forecast.backends.kronos.vendor import Kronos, KronosTokenizer

TINY_S1_BITS = 4
TINY_S2_BITS = 4
TINY_D_MODEL = 16


def build_tiny_tokenizer() -> KronosTokenizer:
    return KronosTokenizer(
        d_in=6,
        d_model=TINY_D_MODEL,
        n_heads=2,
        ff_dim=32,
        n_enc_layers=2,
        n_dec_layers=2,
        ffn_dropout_p=0.0,
        attn_dropout_p=0.0,
        resid_dropout_p=0.0,
        s1_bits=TINY_S1_BITS,
        s2_bits=TINY_S2_BITS,
        beta=1.0,
        gamma0=0.1,
        gamma=0.1,
        zeta=0.1,
        group_size=4,
    )


def build_tiny_model() -> Kronos:
    return Kronos(
        s1_bits=TINY_S1_BITS,
        s2_bits=TINY_S2_BITS,
        n_layers=1,
        d_model=TINY_D_MODEL,
        n_heads=2,
        ff_dim=32,
        ffn_dropout_p=0.0,
        attn_dropout_p=0.0,
        resid_dropout_p=0.0,
        token_dropout_p=0.0,
        learn_te=True,
    )


@pytest.fixture(scope="session")
def tiny_tokenizer() -> KronosTokenizer:
    return build_tiny_tokenizer()


@pytest.fixture(scope="session")
def tiny_model() -> Kronos:
    return build_tiny_model()


@pytest.fixture
def tiny_runtime(tiny_model: Kronos, tiny_tokenizer: KronosTokenizer) -> KronosRuntime:
    return KronosRuntime(
        model=tiny_model,
        tokenizer=tiny_tokenizer,
        config=KronosRuntimeConfig(model_id="tiny", tokenizer_id="tiny", model_revision="test"),
    )
