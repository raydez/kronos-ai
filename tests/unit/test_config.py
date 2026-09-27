"""Experiment Config 单元测试（RX-KAI-019，基线文档 §32.1 / §34 / §27 / §28）。

覆盖：

1. **真实示例配置可加载**：``configs/benchmark-forecast.yaml`` 必须能被本 build 接受
   （示例配置漂移会让 CLI 文档失效）；
2. **config_hash 语义**：注释/键顺序变化不换 hash；任一字段变化换 hash；
3. **secret 拒绝**：YAML 里出现 secret 类键名即拒绝加载（§32.1）；
4. **单一真源**：embargo / horizon 与 LabelPolicy 版本化定义不一致时拒绝（§27/§28）；
5. **时间轴预算**：``required_sessions`` = lookback + Σsegments + embargo × gaps。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kronos_ai.config import ExperimentConfig, load_experiment_config
from kronos_ai.domain.hashing import is_sha256_hex
from kronos_ai.errors import ConfigurationError
from kronos_ai.evaluation.dataset import DEFAULT_EMBARGO_SESSIONS, DEFAULT_HORIZON_SESSIONS

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "configs" / "benchmark-forecast.yaml"


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


MINIMAL = """
dataset:
  symbols: ["600000"]
  start_session: 2026-01-05
  end_session: 2026-03-31
  segments:
    - {name: train, length_sessions: 10}
    - {name: test, length_sessions: 5}
benchmark:
  backends: [last_value]
"""


class TestExampleConfig:
    def test_loads(self) -> None:
        config, _raw = load_experiment_config(EXAMPLE_CONFIG)
        assert config.forecast.lookback_bars == 32
        assert config.benchmark.backends[-1] == "kronos"
        # 示例不截断（跑满全窗口）；pilot 旋钮留在注释里，避免「示例只跑 train 段」
        assert config.benchmark.origin_limit is None
        assert config.data.adjustment == "raw"

    def test_required_sessions_matches_plan(self) -> None:
        config, _ = load_experiment_config(EXAMPLE_CONFIG)
        policy = config.label_policy()
        plan_sessions = sum(segment.length_sessions for segment in config.dataset.segments)
        gaps = len(config.dataset.segments) - 1
        assert config.required_sessions() == (
            config.forecast.lookback_bars + plan_sessions + policy.embargo_sessions * gaps
        )

    def test_config_hash_is_sha256(self) -> None:
        config, _ = load_experiment_config(EXAMPLE_CONFIG)
        assert is_sha256_hex(config.config_hash)


class TestHashSemantics:
    def test_comments_and_key_order_do_not_change_hash(self, tmp_path: Path) -> None:
        first, _ = load_experiment_config(write(tmp_path, MINIMAL))
        reordered = (
            "# a comment that must not matter\n"
            "benchmark:\n  backends: [last_value]\n"
            "dataset:\n"
            "  segments:\n"
            "    - {name: train, length_sessions: 10}\n"
            "    - {name: test, length_sessions: 5}\n"
            '  symbols: ["600000"]\n'
            "  end_session: 2026-03-31\n"
            "  start_session: 2026-01-05\n"
        )
        second, _ = load_experiment_config(write(tmp_path, reordered))
        assert first.config_hash == second.config_hash

    def test_field_change_changes_hash(self, tmp_path: Path) -> None:
        base, _ = load_experiment_config(write(tmp_path, MINIMAL))
        changed, _ = load_experiment_config(
            write(tmp_path, MINIMAL.replace("backends: [last_value]", "backends: [drift]"))
        )
        assert base.config_hash != changed.config_hash

    def test_unknown_field_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError):
            load_experiment_config(write(tmp_path, MINIMAL + "\nunexpected: 1\n"))

    def test_symbol_order_does_not_change_hash(self, tmp_path: Path) -> None:
        """symbol 书写顺序是书写形式：dataset 会把它排序，config_hash 也必须与顺序无关。"""
        forward, _ = load_experiment_config(
            write(tmp_path, MINIMAL.replace('symbols: ["600000"]', 'symbols: ["600000", "000001"]'))
        )
        backward, _ = load_experiment_config(
            write(tmp_path, MINIMAL.replace('symbols: ["600000"]', 'symbols: ["000001", "600000"]'))
        )
        assert forward.dataset.symbols == backward.dataset.symbols
        assert forward.config_hash == backward.config_hash


class TestDataSection:
    def test_default_adjustment_is_raw(self, tmp_path: Path) -> None:
        config, _ = load_experiment_config(write(tmp_path, MINIMAL))
        assert config.data.adjustment == "raw"

    def test_adjustment_changes_config_hash(self, tmp_path: Path) -> None:
        """复权口径是数据身份：换口径必须换 config_hash（否则同一份 config 对应两个结论）。"""
        base, _ = load_experiment_config(write(tmp_path, MINIMAL))
        adjusted, _ = load_experiment_config(
            write(tmp_path, MINIMAL + "data:\n  adjustment: hfq\n")
        )
        assert adjusted.data.adjustment == "hfq"
        assert base.config_hash != adjusted.config_hash

    def test_unknown_adjustment_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError):
            load_experiment_config(write(tmp_path, MINIMAL + "data:\n  adjustment: split\n"))


class TestSecretRejection:
    @pytest.mark.parametrize("key", ["api_key", "API-KEY", "hf_token", "secret", "password"])
    def test_secret_like_keys_rejected(self, tmp_path: Path, key: str) -> None:
        payload = MINIMAL + f'\nforecast:\n  {key}: "leaked"\n'
        with pytest.raises(ConfigurationError, match="secret"):
            load_experiment_config(write(tmp_path, payload))

    def test_nested_secret_rejected(self, tmp_path: Path) -> None:
        payload = MINIMAL.replace(
            "  backends: [last_value]", "  backends:\n    - name: last_value\n      token: x"
        )
        with pytest.raises(ConfigurationError):
            load_experiment_config(write(tmp_path, payload))


class TestLabelPolicySingleSource:
    def test_embargo_mismatch_rejected(self, tmp_path: Path) -> None:
        payload = MINIMAL + f"evaluation:\n  embargo_sessions: {DEFAULT_EMBARGO_SESSIONS + 1}\n"
        with pytest.raises(ConfigurationError, match="embargo"):
            load_experiment_config(write(tmp_path, payload))

    def test_horizon_mismatch_rejected(self, tmp_path: Path) -> None:
        payload = MINIMAL + f"evaluation:\n  horizon_sessions: {DEFAULT_HORIZON_SESSIONS + 1}\n"
        with pytest.raises(ConfigurationError, match="horizon"):
            load_experiment_config(write(tmp_path, payload))

    def test_matching_values_accepted(self, tmp_path: Path) -> None:
        payload = (
            MINIMAL
            + "evaluation:\n"
            + f"  embargo_sessions: {DEFAULT_EMBARGO_SESSIONS}\n"
            + f"  horizon_sessions: {DEFAULT_HORIZON_SESSIONS}\n"
        )
        config, _ = load_experiment_config(write(tmp_path, payload))
        assert config.label_policy().embargo_sessions == DEFAULT_EMBARGO_SESSIONS


class TestValidation:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="not found"):
            load_experiment_config(tmp_path / "nope.yaml")

    def test_empty_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="empty"):
            load_experiment_config(write(tmp_path, ""))

    def test_non_mapping(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError):
            load_experiment_config(write(tmp_path, "- a\n- b\n"))

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="YAML"):
            load_experiment_config(write(tmp_path, "dataset: [unclosed\n"))

    def test_walk_forward_required(self, tmp_path: Path) -> None:
        payload = MINIMAL + "evaluation:\n  walk_forward: false\n"
        with pytest.raises(ConfigurationError):
            load_experiment_config(write(tmp_path, payload))

    def test_duplicate_backends_rejected(self, tmp_path: Path) -> None:
        payload = MINIMAL.replace("backends: [last_value]", "backends: [last_value, last_value]")
        with pytest.raises(ConfigurationError):
            load_experiment_config(write(tmp_path, payload))

    def test_session_window_ordered(self, tmp_path: Path) -> None:
        payload = MINIMAL.replace("end_session: 2026-03-31", "end_session: 2025-12-31")
        with pytest.raises(ConfigurationError):
            load_experiment_config(write(tmp_path, payload))

    def test_label_policy_and_sampling_accessors(self, tmp_path: Path) -> None:
        config, _ = load_experiment_config(write(tmp_path, MINIMAL))
        assert config.sampling.sample_count == 64
        assert config.benchmark_spec().backends == ("last_value",)
        assert isinstance(config, ExperimentConfig)
