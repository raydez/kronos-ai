"""Forecast Artifact Cache 契约（基线文档 §15；ADR-011）。"""

from __future__ import annotations

import threading
from datetime import date, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from kronos_ai.data.calendar import StaticTradingCalendar
from kronos_ai.domain.forecast import (
    ForecastDistribution as Distribution,
)
from kronos_ai.domain.forecast import (
    ForecastPoint,
    ForecastRequest,
    ForecastResult,
    ForecastSample,
    ModelMetadata,
    QuantileValue,
    SamplingConfig,
    SamplingMetadata,
    ThresholdProbability,
)
from kronos_ai.domain.hashing import canonical_json
from kronos_ai.domain.time import CN_TZ, MARKET_SESSION_CLOSE
from kronos_ai.errors import ArtifactError, CalendarError, ConfigurationError
from kronos_ai.forecast.cache import (
    FileSystemForecastCache,
    ForecastArtifactKey,
    build_forecast_artifact_key,
    cached_forecast,
)
from kronos_ai.forecast.distribution import (
    DEFAULT_DISTRIBUTION_SPEC,
    DistributionSpec,
    ThresholdSpec,
    distribution_spec_hash,
)

MD = date(2026, 9, 25)
CUTOFF = datetime(2026, 9, 25, 18, 0, tzinfo=CN_TZ)
SESSIONS = (date(2026, 9, 28), date(2026, 9, 29))
HASH = "a" * 64
CONFIG_HASH = "b" * 64
SPEC_HASH = distribution_spec_hash(DEFAULT_DISTRIBUTION_SPEC)

MODEL_IDENTITY: dict[str, str] = {
    "model_id": "NeoQuasar/Kronos-small",
    "model_revision": "901c26c1332695a2a8f243eb2f37243a37bea320",
    "runtime_version": "kronos-runtime-v1/upstream-67b630e67f6a/torch-2.14.0",
    "device_class": "cpu",
    "dtype": "float32",
    "config_hash": CONFIG_HASH,
}

KEY_FIELDS: dict[str, object] = {
    "symbol": "600000",
    "input_data_hash": HASH,
    "market_date": MD,
    "knowledge_cutoff": CUTOFF,
    "horizon": 2,
    "future_sessions": SESSIONS,
    "model_id": MODEL_IDENTITY["model_id"],
    "model_revision": MODEL_IDENTITY["model_revision"],
    "runtime_version": MODEL_IDENTITY["runtime_version"],
    "device_class": "cpu",
    "dtype": "float32",
    "config_hash": CONFIG_HASH,
    "distribution_spec_hash": SPEC_HASH,
    "seed": 7,
    "sample_count": 2,
    "temperature": 1.0,
    "top_k": 0,
    "top_p": 0.9,
}


def make_key(**overrides: object) -> ForecastArtifactKey:
    return ForecastArtifactKey(**{**KEY_FIELDS, **overrides})  # type: ignore[arg-type]


def make_points(close: float, sessions: tuple[date, ...] = SESSIONS) -> tuple[ForecastPoint, ...]:
    return tuple(
        ForecastPoint(
            timestamp=datetime.combine(day, MARKET_SESSION_CLOSE, tzinfo=CN_TZ),
            open=close,
            high=close,
            low=close,
            close=close,
            volume=1000.0,
            amount=10_000.0,
        )
        for day in sessions
    )


def make_result(
    *,
    artifact_id: str,
    symbol: str = "600000",
    input_data_hash: str = HASH,
    close: float = 10.0,
    sessions: tuple[date, ...] = SESSIONS,
    spec_hash: str = SPEC_HASH,
    market_date: date = MD,
    knowledge_cutoff: datetime = CUTOFF,
    seed: int = 7,
    config_hash: str = CONFIG_HASH,
) -> ForecastResult:
    points = make_points(close, sessions)
    samples = tuple(ForecastSample(sample_id=i, points=points) for i in range(2))
    distribution = Distribution(
        horizon=len(sessions),
        sample_count=2,
        expected_return=0.01,
        median_return=0.01,
        threshold_probabilities=(
            ThresholdProbability(
                metric="horizon_return", operator="gt", threshold=0.0, probability=1.0
            ),
        ),
        quantiles=(QuantileValue(metric="horizon_return", quantile=0.5, value=0.01),),
        forecast_dispersion=0.0,
        expected_max_drawdown=0.0,
        expected_path_volatility=0.0,
        distribution_spec_version="distribution-spec-v1",
        distribution_spec_hash=spec_hash,
        metric_definition_version="forecast-metrics-v1",
    )
    return ForecastResult(
        symbol=symbol,
        market_date=market_date,
        knowledge_cutoff=knowledge_cutoff,
        samples=samples,
        distribution=distribution,
        model=ModelMetadata(
            backend="kronos",
            model_id=MODEL_IDENTITY["model_id"],
            revision=MODEL_IDENTITY["model_revision"],
            runtime_version=MODEL_IDENTITY["runtime_version"],
            device="cpu",
            dtype="float32",
            config_hash=config_hash,
        ),
        sampling=SamplingMetadata.from_config(SamplingConfig(seed=seed, sample_count=2)),
        input_data_hash=input_data_hash,
        artifact_id=artifact_id,
    )


class TestForecastArtifactKey:
    def test_digest_is_deterministic(self) -> None:
        assert make_key().digest == make_key().digest

    def test_golden_digest(self) -> None:
        # golden：字段构成 / hashing payload / canonical json 任一改变都会让本断言变红。
        assert make_key().digest == "055aea41dfa8e95d09f345868980438f32ca75ecc8d5208dc83038f84b3779d7"

    def test_key_version_participates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """key 版本必须真的进入 payload：只改版本号，digest 就得变（而非同名同义反复）。"""
        base = make_key().digest
        monkeypatch.setattr(
            "kronos_ai.forecast.cache.FORECAST_ARTIFACT_KEY_VERSION",
            "forecast-artifact-key-v2",
        )
        assert make_key().digest != base

    def test_payload_covers_every_model_field(self) -> None:
        """穷尽性反射：新增字段但忘了加进 hashing_payload 时，本条测试显式失败。"""
        payload_keys = set(make_key().hashing_payload())
        meta = {"kind", "contract_version", "forecast_contract_version"}
        assert payload_keys == set(ForecastArtifactKey.model_fields) | meta

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("symbol", "000001"),
            ("input_data_hash", "c" * 64),
            ("future_sessions", (date(2026, 9, 29), date(2026, 9, 30))),
            ("distribution_spec_hash", "d" * 64),
            ("model_id", "other/model"),
            ("model_revision", "0" * 40),
            ("runtime_version", "kronos-runtime-v2"),
            ("device_class", "cuda"),
            ("dtype", "bfloat16"),
            ("config_hash", "d" * 64),
            ("seed", 8),
            ("sample_count", 3),
            ("temperature", 0.5),
            ("top_k", 5),
            ("top_p", 0.5),
        ],
    )
    def test_each_field_changes_digest(self, field: str, value: object) -> None:
        assert make_key(**{field: value}).digest != make_key().digest

    def test_market_date_and_cutoff_change_digest(self) -> None:
        shifted = make_key(
            market_date=date(2026, 9, 24),
            knowledge_cutoff=datetime(2026, 9, 24, 18, 0, tzinfo=CN_TZ),
        )
        assert shifted.digest != make_key().digest

    def test_horizon_extension_changes_digest(self) -> None:
        extended = make_key(horizon=3, future_sessions=(*SESSIONS, date(2026, 9, 30)))
        assert extended.digest != make_key().digest

    def test_bad_input_data_hash_rejected(self) -> None:
        with pytest.raises(ValidationError, match="input_data_hash"):
            make_key(input_data_hash="not-a-hash")

    def test_bad_distribution_spec_hash_rejected(self) -> None:
        with pytest.raises(ValidationError, match="distribution_spec_hash"):
            make_key(distribution_spec_hash="not-a-hash")

    def test_future_sessions_must_match_horizon(self) -> None:
        with pytest.raises(ValidationError, match="future_sessions"):
            make_key(future_sessions=(date(2026, 9, 28),))

    def test_future_sessions_must_be_sorted(self) -> None:
        with pytest.raises(ValidationError, match="sorted ascending"):
            make_key(future_sessions=(date(2026, 9, 29), date(2026, 9, 28)))

    def test_future_sessions_must_be_unique(self) -> None:
        with pytest.raises(ValidationError, match="unique"):
            make_key(future_sessions=(date(2026, 9, 28), date(2026, 9, 28)))

    def test_future_sessions_must_follow_market_date(self) -> None:
        with pytest.raises(ValidationError, match="after market_date"):
            make_key(future_sessions=(date(2026, 9, 24), date(2026, 9, 28)))

    def test_empty_identity_rejected(self) -> None:
        with pytest.raises(ValidationError, match="model_id"):
            make_key(model_id="  ")

    def test_cutoff_must_fall_on_market_date(self) -> None:
        with pytest.raises(ValidationError, match="knowledge_cutoff"):
            ForecastArtifactKey(**{**KEY_FIELDS, "market_date": date(2026, 9, 24)})  # type: ignore[arg-type]

    def test_bool_rejected_for_int_field(self) -> None:
        with pytest.raises(ValidationError, match="must be an int, not bool"):
            make_key(horizon=True)


class TestBuildForecastArtifactKey:
    def test_builds_from_inputs(self, make_history: object, session_calendar: object) -> None:
        history = make_history(symbol="600000")  # type: ignore[operator]
        request = ForecastRequest(
            symbol="600000",
            market_date=MD,
            knowledge_cutoff=CUTOFF,
            horizon=2,
            sampling=SamplingConfig(seed=7, sample_count=2),
        )
        key = build_forecast_artifact_key(
            history=history,  # type: ignore[arg-type]
            request=request,
            model_identity=MODEL_IDENTITY,
            calendar=session_calendar,  # type: ignore[arg-type]
        )
        assert key.input_data_hash == history.data_hash  # type: ignore[attr-defined]
        assert key.seed == 7
        assert key.config_hash == CONFIG_HASH
        assert key.future_sessions == tuple(
            session_calendar.next_sessions(MD, 2)  # type: ignore[attr-defined]
        )
        assert key.distribution_spec_hash == SPEC_HASH
        assert key.digest == make_key(input_data_hash=history.data_hash).digest  # type: ignore[attr-defined]

    def test_different_calendar_changes_digest(
        self, make_history: object, session_calendar: object
    ) -> None:
        """日历是身份维：未来 session 时间轴不同 → 不同 key（即使输入 bar 不变）。"""
        history = make_history(symbol="600000")  # type: ignore[operator]
        request = ForecastRequest(
            symbol="600000", market_date=MD, knowledge_cutoff=CUTOFF, horizon=2,
            sampling=SamplingConfig(seed=7, sample_count=2),
        )
        shifted = StaticTradingCalendar(
            exchange="SSE",
            source="test-fixture-shifted",
            sessions=(MD, date(2026, 9, 29), date(2026, 9, 30)),
        )
        base = build_forecast_artifact_key(
            history=history,  # type: ignore[arg-type]
            request=request,
            model_identity=MODEL_IDENTITY,
            calendar=session_calendar,  # type: ignore[arg-type]
        )
        other = build_forecast_artifact_key(
            history=history,  # type: ignore[arg-type]
            request=request,
            model_identity=MODEL_IDENTITY,
            calendar=shifted,
        )
        assert base.future_sessions != other.future_sessions
        assert base.digest != other.digest

    def test_different_distribution_spec_changes_digest(
        self, make_history: object, session_calendar: object
    ) -> None:
        """分布 spec 是身份维：阈值/分位不同 → 不同 key（即使样本完全相同）。"""
        history = make_history(symbol="600000")  # type: ignore[operator]
        request = ForecastRequest(
            symbol="600000", market_date=MD, knowledge_cutoff=CUTOFF, horizon=2,
            sampling=SamplingConfig(seed=7, sample_count=2),
        )
        spec = DistributionSpec(
            thresholds=(
                ThresholdSpec(metric="horizon_return", operator="gt", threshold=0.03),
            ),
            quantiles=DEFAULT_DISTRIBUTION_SPEC.quantiles,
        )
        base = build_forecast_artifact_key(
            history=history,  # type: ignore[arg-type]
            request=request,
            model_identity=MODEL_IDENTITY,
            calendar=session_calendar,  # type: ignore[arg-type]
        )
        other = build_forecast_artifact_key(
            history=history,  # type: ignore[arg-type]
            request=request,
            model_identity=MODEL_IDENTITY,
            calendar=session_calendar,  # type: ignore[arg-type]
            distribution_spec=spec,
        )
        assert other.distribution_spec_hash == distribution_spec_hash(spec)
        assert base.distribution_spec_hash != other.distribution_spec_hash
        assert base.digest != other.digest

    def test_missing_model_identity_key_explicit_failure(
        self, make_history: object, session_calendar: object
    ) -> None:
        identity = {k: v for k, v in MODEL_IDENTITY.items() if k != "config_hash"}
        request = ForecastRequest(
            symbol="600000", market_date=MD, knowledge_cutoff=CUTOFF, horizon=2,
            sampling=SamplingConfig(seed=7, sample_count=2),
        )
        with pytest.raises(ConfigurationError, match="config_hash"):
            build_forecast_artifact_key(
                history=make_history(symbol="600000"),  # type: ignore[arg-type,operator]
                request=request,
                model_identity=identity,
                calendar=session_calendar,  # type: ignore[arg-type]
            )

    def test_history_request_symbol_mismatch(
        self, make_history: object, session_calendar: object
    ) -> None:
        request = ForecastRequest(
            symbol="000001", market_date=MD, knowledge_cutoff=CUTOFF, horizon=2,
            sampling=SamplingConfig(seed=7, sample_count=2),
        )
        with pytest.raises(ConfigurationError, match="symbol"):
            build_forecast_artifact_key(
                history=make_history(symbol="600000"),  # type: ignore[arg-type,operator]
                request=request,
                model_identity=MODEL_IDENTITY,
                calendar=session_calendar,  # type: ignore[arg-type]
            )

    def test_market_date_mismatch_explicit_failure(
        self, make_history: object, session_calendar: object
    ) -> None:
        request = ForecastRequest(
            symbol="600000",
            market_date=date(2026, 9, 24),
            knowledge_cutoff=datetime(2026, 9, 24, 18, 0, tzinfo=CN_TZ),
            horizon=2,
            sampling=SamplingConfig(seed=7, sample_count=2),
        )
        with pytest.raises(ConfigurationError, match="market_date"):
            build_forecast_artifact_key(
                history=make_history(symbol="600000"),  # type: ignore[arg-type,operator]
                request=request,
                model_identity=MODEL_IDENTITY,
                calendar=session_calendar,  # type: ignore[arg-type]
            )

    def test_cutoff_mismatch_explicit_failure(
        self, make_history: object, session_calendar: object
    ) -> None:
        request = ForecastRequest(
            symbol="600000",
            market_date=MD,
            knowledge_cutoff=datetime(2026, 9, 25, 19, 0, tzinfo=CN_TZ),
            horizon=2,
            sampling=SamplingConfig(seed=7, sample_count=2),
        )
        with pytest.raises(ConfigurationError, match="knowledge_cutoff"):
            build_forecast_artifact_key(
                history=make_history(symbol="600000"),  # type: ignore[arg-type,operator]
                request=request,
                model_identity=MODEL_IDENTITY,
                calendar=session_calendar,  # type: ignore[arg-type]
            )

    def test_non_string_identity_value_explicit_failure(
        self, make_history: object, session_calendar: object
    ) -> None:
        request = ForecastRequest(
            symbol="600000", market_date=MD, knowledge_cutoff=CUTOFF, horizon=2,
            sampling=SamplingConfig(seed=7, sample_count=2),
        )
        identity: dict[str, object] = {**MODEL_IDENTITY, "model_id": None}
        with pytest.raises(ConfigurationError, match="model_id"):
            build_forecast_artifact_key(
                history=make_history(symbol="600000"),  # type: ignore[arg-type,operator]
                request=request,
                model_identity=identity,  # type: ignore[arg-type]
                calendar=session_calendar,  # type: ignore[arg-type]
            )

    def test_calendar_too_short_explicit_failure(
        self, make_history: object
    ) -> None:
        """日历覆盖不足时必须显式失败，而不是静默给出短时间轴。"""
        request = ForecastRequest(
            symbol="600000", market_date=MD, knowledge_cutoff=CUTOFF, horizon=2,
            sampling=SamplingConfig(seed=7, sample_count=2),
        )
        short = StaticTradingCalendar(
            exchange="SSE", source="test-fixture-short", sessions=(MD, date(2026, 9, 28))
        )
        with pytest.raises(CalendarError, match="coverage ends"):
            build_forecast_artifact_key(
                history=make_history(symbol="600000"),  # type: ignore[arg-type,operator]
                request=request,
                model_identity=MODEL_IDENTITY,
                calendar=short,
            )


class TestFileSystemForecastCache:
    def test_miss_returns_none(self, tmp_path: Path) -> None:
        cache = FileSystemForecastCache(tmp_path)
        assert cache.get(make_key()) is None

    def test_put_get_roundtrip(self, tmp_path: Path) -> None:
        cache = FileSystemForecastCache(tmp_path)
        key = make_key()
        result = make_result(artifact_id=key.digest)

        cache.put(key, result)
        loaded = cache.get(key)

        assert loaded == result

    def test_path_layout_and_no_temp_leftovers(self, tmp_path: Path) -> None:
        cache = FileSystemForecastCache(tmp_path)
        key = make_key()
        cache.put(key, make_result(artifact_id=key.digest))

        expected = tmp_path / key.digest[:2] / f"{key.digest}.json"
        assert expected.exists()
        assert cache.path_for(key) == expected
        assert [p.name for p in tmp_path.rglob("*.tmp")] == []

    def test_force_bypasses_read(self, tmp_path: Path) -> None:
        cache = FileSystemForecastCache(tmp_path)
        key = make_key()
        cache.put(key, make_result(artifact_id=key.digest))

        assert cache.get(key) is not None
        assert cache.get(key, force=True) is None

    def test_put_rejects_mismatched_artifact_id(self, tmp_path: Path) -> None:
        cache = FileSystemForecastCache(tmp_path)
        key = make_key()
        with pytest.raises(ArtifactError, match="artifact_id"):
            cache.put(key, make_result(artifact_id="0" * 64))

    def test_get_rejects_foreign_artifact(self, tmp_path: Path) -> None:
        cache = FileSystemForecastCache(tmp_path)
        key = make_key()
        # 直接落盘一个 input_data_hash 不匹配的合法 artifact（绕过 put 的校验）
        foreign = make_result(artifact_id=key.digest, input_data_hash="e" * 64)
        path = cache.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json(foreign.model_dump(mode="json")).encode("utf-8"))
        with pytest.raises(ArtifactError, match="input_data_hash"):
            cache.get(key)

    def test_put_overwrites_existing_file(self, tmp_path: Path) -> None:
        """单线程 overwrite：同 key 重写后读到的是新内容且无临时文件残留。"""
        cache = FileSystemForecastCache(tmp_path)
        key = make_key()
        cache.put(key, make_result(artifact_id=key.digest, close=10.0))
        cache.put(key, make_result(artifact_id=key.digest, close=12.0))

        loaded = cache.get(key)
        assert loaded is not None
        assert loaded.samples[0].points[0].close == 12.0
        assert [p.name for p in tmp_path.rglob("*.tmp")] == []

    def test_atomic_write_cleans_temp_on_replace_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = FileSystemForecastCache(tmp_path)
        key = make_key()

        def boom(*args: object, **kwargs: object) -> None:
            raise OSError("simulated replace failure")

        monkeypatch.setattr("kronos_ai.forecast.cache.os.replace", boom)
        with pytest.raises(OSError, match="simulated replace failure"):
            cache.put(key, make_result(artifact_id=key.digest))

        assert cache.path_for(key).exists() is False
        assert [p.name for p in tmp_path.rglob("*.tmp")] == []

    def test_corrupt_file_explicit_failure(self, tmp_path: Path) -> None:
        cache = FileSystemForecastCache(tmp_path)
        key = make_key()
        path = cache.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{not json")
        with pytest.raises(ArtifactError, match="corrupt"):
            cache.get(key)

    def test_concurrent_put_is_atomic(self, tmp_path: Path) -> None:
        cache = FileSystemForecastCache(tmp_path)
        key = make_key()
        variants = [make_result(artifact_id=key.digest, close=10.0 + i) for i in range(4)]
        errors: list[BaseException] = []
        barrier = threading.Barrier(5)

        def writer(result: ForecastResult) -> None:
            try:
                barrier.wait()
                for _ in range(20):
                    cache.put(key, result)
            except BaseException as exc:
                errors.append(exc)

        def reader() -> None:
            try:
                barrier.wait()
                for _ in range(60):
                    loaded = cache.get(key)
                    if loaded is not None:
                        assert loaded in variants
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(v,)) for v in variants]
        threads.append(threading.Thread(target=reader))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert cache.get(key) in variants


class TestProvenanceVerification:
    """读写两条路径都必须逐维校验 artifact 与 key 的一致性（§14/§15）。"""

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("input_data_hash", "e" * 64, "input_data_hash"),
            ("symbol", "000001", "symbol"),
            ("spec_hash", "d" * 64, "distribution_spec_hash"),
            ("config_hash", "d" * 64, "model.config_hash"),
            ("seed", 8, "sampling.seed"),
            (
                "market_date",
                date(2026, 9, 24),
                "market_date",
            ),
            ("sessions", (date(2026, 9, 29), date(2026, 9, 30)), "timeline"),
            (
                "sessions",
                (date(2026, 9, 28), date(2026, 9, 29), date(2026, 9, 30)),
                "distribution.horizon",
            ),
        ],
    )
    def test_put_rejects_mismatched_identity(
        self, tmp_path: Path, field: str, value: object, match: str
    ) -> None:
        cache = FileSystemForecastCache(tmp_path)
        key = make_key()
        overrides: dict[str, object] = {field: value}
        if field == "market_date":
            overrides["knowledge_cutoff"] = datetime(2026, 9, 24, 18, 0, tzinfo=CN_TZ)
        with pytest.raises(ArtifactError, match=match):
            cache.put(key, make_result(artifact_id=key.digest, **overrides))

    def test_market_date_mismatch_reports_both_dims(self, tmp_path: Path) -> None:
        cache = FileSystemForecastCache(tmp_path)
        key = make_key()
        with pytest.raises(ArtifactError, match="knowledge_cutoff"):
            cache.put(
                key,
                make_result(
                    artifact_id=key.digest,
                    market_date=date(2026, 9, 24),
                    knowledge_cutoff=datetime(2026, 9, 24, 18, 0, tzinfo=CN_TZ),
                ),
            )

    def test_get_rejects_artifact_computed_with_other_spec(self, tmp_path: Path) -> None:
        """用例：同 key 但 artifact 的分布由另一份 spec 算出——不得静默返回。"""
        cache = FileSystemForecastCache(tmp_path)
        key = make_key()
        foreign = make_result(artifact_id=key.digest, spec_hash="d" * 64)
        path = cache.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json(foreign.model_dump(mode="json")).encode("utf-8"))
        with pytest.raises(ArtifactError, match="distribution_spec_hash"):
            cache.get(key)

    def test_get_rejects_artifact_on_other_calendar_timeline(self, tmp_path: Path) -> None:
        """用例：同 key 但 artifact 的时间轴来自另一份日历——不得静默返回。"""
        cache = FileSystemForecastCache(tmp_path)
        key = make_key()
        foreign = make_result(
            artifact_id=key.digest,
            sessions=(date(2026, 9, 29), date(2026, 9, 30)),
        )
        path = cache.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json(foreign.model_dump(mode="json")).encode("utf-8"))
        with pytest.raises(ArtifactError, match="timeline"):
            cache.get(key)


class TestCachedForecast:
    def test_second_call_does_not_recompute(self, tmp_path: Path) -> None:
        cache = FileSystemForecastCache(tmp_path)
        key = make_key()
        calls = 0

        def compute() -> ForecastResult:
            nonlocal calls
            calls += 1
            return make_result(artifact_id=key.digest)

        first, first_cached = cached_forecast(cache, key, compute)
        second, second_cached = cached_forecast(cache, key, compute)

        assert calls == 1
        assert first_cached is False
        assert second_cached is True
        assert first == second

    def test_force_recomputes_and_still_writes(self, tmp_path: Path) -> None:
        cache = FileSystemForecastCache(tmp_path)
        key = make_key()
        calls = 0

        def compute() -> ForecastResult:
            nonlocal calls
            calls += 1
            return make_result(artifact_id=key.digest)

        cached_forecast(cache, key, compute)
        result, was_cached = cached_forecast(cache, key, compute, force=True)

        assert calls == 2
        assert was_cached is False
        assert cache.get(key) == result

    def test_put_validates_provenance(self, tmp_path: Path) -> None:
        cache = FileSystemForecastCache(tmp_path)
        key = make_key()
        with pytest.raises(ArtifactError, match="artifact_id"):
            cached_forecast(cache, key, lambda: make_result(artifact_id="0" * 64))

    def test_compute_with_wrong_distribution_spec_is_explicit_failure(
        self, tmp_path: Path
    ) -> None:
        cache = FileSystemForecastCache(tmp_path)
        key = make_key()
        with pytest.raises(ArtifactError, match="distribution_spec_hash"):
            cached_forecast(
                cache, key, lambda: make_result(artifact_id=key.digest, spec_hash="d" * 64)
            )
        assert cache.get(key) is None
