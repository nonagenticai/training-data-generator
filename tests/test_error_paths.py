"""Tests for uncovered error paths, edge cases, and utility functions.

Targets coverage gaps in:
- main.py: lifespan startup/shutdown, readiness when dc_ok fails
- logging_config.py: OTEL trace context injection, console renderer
- models/db_models.py: ORM helper methods (to_filter_reasons_dict, to_source_providers_dict)
- services/batch_service.py: _ensure_utc branches, _save_batch update path
- services/trigger_service.py: close() lifecycle, _get_quality_score edge cases
- settings.py: get_settings caching
- Collector returning empty/None responses
- Partial failures during MinIO upload
- Network timeouts
- Malformed sample data
"""

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest
from httpx import AsyncClient

from models.db_models import TrainingBatchORM
from models.domain import (
    BatchGenerateRequest,
    BatchStatus,
    InferenceSample,
    LivenessResponse,
    Message,
    ReadinessResponse,
    TrainingBatch,
)
from services.batch_service import BatchService, _domain_to_orm, _ensure_utc, _orm_to_domain
from services.collector_client import CollectorClient
from services.filter_service import FilterService
from services.trigger_service import TriggerService
from settings import Settings
from tests.conftest import PLACEHOLDER_DATABASE_URL, make_sample, make_samples

# ---------------------------------------------------------------------------
# _ensure_utc helper
# ---------------------------------------------------------------------------


class TestEnsureUtc:
    """Test the _ensure_utc helper in batch_service."""

    def test_none_returns_none(self) -> None:
        """_ensure_utc(None) should return None."""
        assert _ensure_utc(None) is None

    def test_naive_datetime_gets_utc(self) -> None:
        """_ensure_utc on a naive datetime should attach UTC."""
        naive = datetime(2026, 2, 26, 12, 0, 0)
        result = _ensure_utc(naive)
        assert result is not None
        assert result.tzinfo is UTC

    def test_aware_datetime_unchanged(self) -> None:
        """_ensure_utc on an already-aware datetime should return it unchanged."""
        aware = datetime(2026, 2, 26, 12, 0, 0, tzinfo=UTC)
        result = _ensure_utc(aware)
        assert result is aware


# ---------------------------------------------------------------------------
# _domain_to_orm / _orm_to_domain round-trip
# ---------------------------------------------------------------------------


class TestOrmConversions:
    """Test domain-to-ORM and ORM-to-domain conversion functions."""

    def test_domain_to_orm_produces_correct_dict(self) -> None:
        """_domain_to_orm should produce a dict with string id and JSON-encoded dicts."""
        batch = TrainingBatch(
            schema_id="test",
            filter_reasons={"duplicate": 3, "too_short": 1},
            source_providers={"Qwen3-4B-AWQ": 10, "deepseek": 2},
        )
        result = _domain_to_orm(batch)
        assert isinstance(result["id"], str)
        assert result["schema_id"] == "test"
        assert result["status"] == "generating"
        assert json.loads(result["filter_reasons"]) == {"duplicate": 3, "too_short": 1}
        assert json.loads(result["source_providers"]) == {"Qwen3-4B-AWQ": 10, "deepseek": 2}

    def test_orm_to_domain_with_naive_dates(self) -> None:
        """_orm_to_domain should handle naive datetimes from SQLite."""
        row = MagicMock()
        row.id = str(uuid4())
        row.schema_id = "test"
        row.mode = "per_schema"
        row.status = "completed"
        row.total_samples = 100
        row.train_samples = 90
        row.valid_samples = 10
        row.filtered_samples = 5
        row.filter_reasons = '{"duplicate": 5}'
        row.minio_path = "path/"
        row.train_file = "path/train.jsonl"
        row.valid_file = "path/valid.jsonl"
        row.source_providers = '{"Qwen3-4B-AWQ": 95}'
        row.source_schema_counts = "{}"
        row.teacher_samples = 3
        row.created_at = datetime(2026, 2, 26, 12, 0, 0)  # naive
        row.completed_at = datetime(2026, 2, 26, 13, 0, 0)  # naive
        row.error_message = None

        result = _orm_to_domain(row)
        assert result.schema_id == "test"
        assert result.status == BatchStatus.COMPLETED
        assert result.filter_reasons == {"duplicate": 5}
        assert result.source_providers == {"Qwen3-4B-AWQ": 95}
        # Naive dates should be converted to UTC
        assert result.created_at.tzinfo is UTC
        assert result.completed_at is not None
        assert result.completed_at.tzinfo is UTC

    def test_orm_to_domain_with_empty_json_fields(self) -> None:
        """_orm_to_domain should handle empty/None JSON fields gracefully."""
        row = MagicMock()
        row.id = str(uuid4())
        row.schema_id = "test"
        row.mode = "per_schema"
        row.status = "generating"
        row.total_samples = 0
        row.train_samples = 0
        row.valid_samples = 0
        row.filtered_samples = 0
        row.filter_reasons = ""  # empty string
        row.minio_path = None
        row.train_file = None
        row.valid_file = None
        row.source_providers = ""  # empty string
        row.source_schema_counts = ""
        row.teacher_samples = 0
        row.created_at = datetime(2026, 2, 26, 12, 0, 0, tzinfo=UTC)
        row.completed_at = None
        row.error_message = None

        result = _orm_to_domain(row)
        assert result.filter_reasons == {}
        assert result.source_providers == {}

    def test_orm_to_domain_with_string_uuid(self) -> None:
        """_orm_to_domain should parse string UUIDs correctly."""
        test_id = str(uuid4())
        row = MagicMock()
        row.id = test_id
        row.schema_id = "test"
        row.mode = "per_schema"
        row.status = "generating"
        row.total_samples = 0
        row.train_samples = 0
        row.valid_samples = 0
        row.filtered_samples = 0
        row.filter_reasons = "{}"
        row.minio_path = None
        row.train_file = None
        row.valid_file = None
        row.source_providers = "{}"
        row.source_schema_counts = "{}"
        row.teacher_samples = 0
        row.created_at = datetime.now(UTC)
        row.completed_at = None
        row.error_message = None

        result = _orm_to_domain(row)
        assert str(result.id) == test_id


# ---------------------------------------------------------------------------
# TrainingBatchORM helper methods
# ---------------------------------------------------------------------------


class TestTrainingBatchORMHelpers:
    """Test ORM model helper methods."""

    def test_to_filter_reasons_dict_with_data(self) -> None:
        """to_filter_reasons_dict should parse JSON string."""
        orm = TrainingBatchORM()
        orm.filter_reasons = '{"duplicate": 3, "too_short": 1}'
        result = orm.to_filter_reasons_dict()
        assert result == {"duplicate": 3, "too_short": 1}

    def test_to_filter_reasons_dict_empty_string(self) -> None:
        """to_filter_reasons_dict should return empty dict for empty string."""
        orm = TrainingBatchORM()
        orm.filter_reasons = ""
        result = orm.to_filter_reasons_dict()
        assert result == {}

    def test_to_source_providers_dict_with_data(self) -> None:
        """to_source_providers_dict should parse JSON string."""
        orm = TrainingBatchORM()
        orm.source_providers = '{"Qwen3-4B-AWQ": 10}'
        result = orm.to_source_providers_dict()
        assert result == {"Qwen3-4B-AWQ": 10}

    def test_to_source_providers_dict_empty_string(self) -> None:
        """to_source_providers_dict should return empty dict for empty string."""
        orm = TrainingBatchORM()
        orm.source_providers = ""
        result = orm.to_source_providers_dict()
        assert result == {}


# ---------------------------------------------------------------------------
# Batch service: _save_batch update path
# ---------------------------------------------------------------------------


class TestBatchServiceSaveBatch:
    """Test batch service save/update paths."""

    @pytest.mark.asyncio
    async def test_save_batch_updates_existing(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
    ) -> None:
        """_save_batch should update an existing batch record rather than insert a new one."""
        samples = make_samples(count=100)
        mock_collector_client.export_samples.return_value = samples

        request = BatchGenerateRequest(schema_id="test_schema", min_samples=10)
        batch = await batch_service.generate_batch(request)

        assert batch.status == BatchStatus.COMPLETED

        # Verify the batch was persisted
        retrieved = await batch_service.get_batch(str(batch.id))
        assert retrieved is not None
        assert retrieved.status == BatchStatus.COMPLETED

        # The generate_batch method calls _save_batch multiple times:
        # 1. Initial save (GENERATING)
        # 2. Final save (COMPLETED) - this is the update path
        # So we've already exercised the update path

    @pytest.mark.asyncio
    async def test_collector_export_raises(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
    ) -> None:
        """generate_batch should mark batch as FAILED when collector raises."""
        mock_collector_client.export_samples.side_effect = ConnectionError("Connection refused")

        request = BatchGenerateRequest(schema_id="test_schema", min_samples=10)
        batch = await batch_service.generate_batch(request)

        assert batch.status == BatchStatus.FAILED
        assert "Connection refused" in (batch.error_message or "")

    @pytest.mark.asyncio
    async def test_mark_used_raises(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
        mock_minio_client: AsyncMock,
    ) -> None:
        """generate_batch should mark batch as FAILED when mark_used raises."""
        samples = make_samples(count=100)
        mock_collector_client.export_samples.return_value = samples
        mock_collector_client.mark_used.side_effect = ConnectionError("mark failed")

        request = BatchGenerateRequest(schema_id="test_schema", min_samples=10)
        batch = await batch_service.generate_batch(request)

        assert batch.status == BatchStatus.FAILED
        assert "mark failed" in (batch.error_message or "")

    @pytest.mark.asyncio
    async def test_ensure_bucket_raises(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
        mock_minio_client: AsyncMock,
    ) -> None:
        """generate_batch should mark batch as FAILED when ensure_bucket raises."""
        samples = make_samples(count=100)
        mock_collector_client.export_samples.return_value = samples
        mock_minio_client.ensure_bucket.side_effect = ConnectionError("bucket error")

        request = BatchGenerateRequest(schema_id="test_schema", min_samples=10)
        batch = await batch_service.generate_batch(request)

        assert batch.status == BatchStatus.FAILED
        assert "bucket error" in (batch.error_message or "")


# ---------------------------------------------------------------------------
# Trigger service edge cases
# ---------------------------------------------------------------------------


class TestTriggerServiceEdgeCases:
    """Test trigger service edge cases and error paths."""

    @pytest.mark.asyncio
    async def test_close_owned_client(self, test_settings: Settings) -> None:
        """close() should close the HTTP client when TriggerService owns it."""
        collector_mock = AsyncMock()
        batch_mock = AsyncMock()
        # Create TriggerService without injecting http_client
        # We need to mock httpx.AsyncClient so it doesn't actually connect
        with patch("services.trigger_service.httpx.AsyncClient") as mock_cls:
            mock_instance = AsyncMock()
            mock_cls.return_value = mock_instance
            ts = TriggerService(
                settings=test_settings,
                collector_client=collector_mock,
                batch_service=batch_mock,
            )
            assert ts._owns_client is True
            await ts.close()
            mock_instance.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_injected_client(self, test_settings: Settings) -> None:
        """close() should not close the HTTP client when it was injected."""
        collector_mock = AsyncMock()
        batch_mock = AsyncMock()
        injected_client = AsyncMock()

        ts = TriggerService(
            settings=test_settings,
            collector_client=collector_mock,
            batch_service=batch_mock,
            http_client=injected_client,
        )
        assert ts._owns_client is False
        await ts.close()
        injected_client.aclose.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_quality_score_non_200(
        self,
        trigger_service: TriggerService,
        mock_http_client: AsyncMock,
    ) -> None:
        """_get_quality_score should return None for non-200 responses."""
        mock_http_client.get.return_value = MagicMock(
            status_code=500,
            json=lambda: {"error": "internal"},
        )

        result = await trigger_service._get_quality_score("test_schema")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_quality_score_missing_key(
        self,
        trigger_service: TriggerService,
        mock_http_client: AsyncMock,
    ) -> None:
        """_get_quality_score should return None when quality_score key is missing."""
        mock_http_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"other_field": "value"},
        )

        result = await trigger_service._get_quality_score("test_schema")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_quality_score_success(
        self,
        trigger_service: TriggerService,
        mock_http_client: AsyncMock,
    ) -> None:
        """_get_quality_score should return float for valid response."""
        mock_http_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"quality_score": 0.92},
        )

        result = await trigger_service._get_quality_score("test_schema")
        assert result == 0.92

    @pytest.mark.asyncio
    async def test_check_and_generate_no_triggers(
        self,
        trigger_service: TriggerService,
        mock_collector_client: AsyncMock,
        mock_http_client: AsyncMock,
    ) -> None:
        """check_and_generate with no schemas needing retrain returns empty list."""
        mock_collector_client.list_schema_ids.return_value = ["test_schema"]
        mock_collector_client.get_sample_count.return_value = 10  # Below threshold

        mock_http_client.get.return_value = MagicMock(
            status_code=404,
            json=lambda: {},
        )

        batches = await trigger_service.check_and_generate()
        assert batches == []

    @pytest.mark.asyncio
    async def test_check_triggers_empty_schemas(
        self,
        trigger_service: TriggerService,
        mock_collector_client: AsyncMock,
    ) -> None:
        """check_triggers with no schemas should return empty list."""
        mock_collector_client.list_schema_ids.return_value = []

        triggers = await trigger_service.check_triggers()
        assert triggers == []


# ---------------------------------------------------------------------------
# Logging config edge cases
# ---------------------------------------------------------------------------


class TestLoggingConfig:
    """Test logging configuration edge cases."""

    def test_add_trace_context_without_otel(self) -> None:
        """add_trace_context should pass through when OTEL is not available.

        Patches OTEL_AVAILABLE to False so we exercise the early-return branch
        rather than relying on the live module flag.
        """
        from logging_config import add_trace_context

        event_dict: dict[str, Any] = {"event": "test"}
        with patch("logging_config.OTEL_AVAILABLE", False):
            result = add_trace_context(None, "info", event_dict)
        assert result == event_dict
        assert "trace_id" not in result
        assert "span_id" not in result

    def test_add_service_context(self) -> None:
        """add_service_context should inject service and environment."""
        from logging_config import add_service_context

        event_dict: dict[str, Any] = {"event": "test"}
        result = add_service_context(None, "info", event_dict)
        assert "service" in result
        assert "environment" in result

    def test_get_logger_default_name(self) -> None:
        """get_logger with no name should use default."""
        from logging_config import get_logger

        logger = get_logger()
        assert logger is not None

    def test_get_logger_custom_name(self) -> None:
        """get_logger with a custom name should work."""
        from logging_config import get_logger

        logger = get_logger("custom_module")
        assert logger is not None

    def test_otel_import_failure_branch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """logging_config should set OTEL_AVAILABLE=False when opentelemetry import fails.

        Reloads the module with opentelemetry stripped from sys.modules and
        ImportError forced via a meta_path finder. Exercises lines 27-28
        (the except ImportError branch).
        """
        import importlib
        import sys

        # Remove cached opentelemetry submodules so the import re-runs
        otel_mods = [m for m in sys.modules if m.startswith("opentelemetry")]
        for mod in otel_mods:
            monkeypatch.delitem(sys.modules, mod, raising=False)

        # Block opentelemetry by inserting a finder that raises ImportError
        class _BlockOtelFinder:
            def find_spec(
                self,
                fullname: str,
                path: list[str] | None = None,
                target: object | None = None,
            ) -> None:
                if fullname == "opentelemetry" or fullname.startswith("opentelemetry."):
                    raise ImportError(f"blocked: {fullname}")
                return None

        finder = _BlockOtelFinder()
        monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
        # Drop the cached logging_config so the top-level import re-runs
        monkeypatch.delitem(sys.modules, "logging_config", raising=False)

        try:
            reloaded = importlib.import_module("logging_config")
            assert reloaded.OTEL_AVAILABLE is False
            assert reloaded.trace is None
        finally:
            # Force a re-import of the real module so subsequent tests see OTEL again
            sys.modules.pop("logging_config", None)
            importlib.import_module("logging_config")

    def test_console_renderer_branch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """configure_logging should pick the console renderer when log_format != json.

        Resets the _configured guard so the function actually re-executes its
        branch selection. Exercises line 109 (the else branch).
        """
        import logging_config
        from settings import Settings

        console_settings = Settings(
            service_name="test",
            log_format="console",
            log_level="DEBUG",
        )

        # Force re-configuration so the branch selection actually runs
        monkeypatch.setattr(logging_config, "_configured", False)
        with patch("logging_config.get_settings", return_value=console_settings):
            logging_config.configure_logging()

        # Sanity check: the function ran without crashing and reset the flag
        assert logging_config._configured is True

        # Restore json config for subsequent tests
        monkeypatch.setattr(logging_config, "_configured", False)
        logging_config.configure_logging()


# ---------------------------------------------------------------------------
# Domain model edge cases
# ---------------------------------------------------------------------------


class TestDomainModelEdgeCases:
    """Test domain model validation edge cases."""

    def test_liveness_response_default(self) -> None:
        """LivenessResponse defaults to status=alive."""
        resp = LivenessResponse()
        assert resp.status == "alive"

    def test_liveness_response_custom_status(self) -> None:
        """LivenessResponse can be created with a custom status."""
        resp = LivenessResponse(status="custom")
        assert resp.status == "custom"

    def test_readiness_response_defaults(self) -> None:
        """ReadinessResponse defaults to data_collector=False, minio=False."""
        resp = ReadinessResponse(ready=False)
        assert resp.data_collector is False
        assert resp.minio is False

    def test_inference_sample_defaults(self) -> None:
        """InferenceSample defaults for optional fields."""
        sample = InferenceSample(
            id=uuid4(),
            schema_id="test",
            messages=[Message(role="user", content="hi")],
            response_content="test",
        )
        assert sample.provider == "unknown"
        assert sample.fallback_used is False
        assert sample.validated_at is None
        assert sample.quality_score is None

    def test_training_batch_default_status(self) -> None:
        """TrainingBatch should default to GENERATING status."""
        batch = TrainingBatch(schema_id="test")
        assert batch.status == BatchStatus.GENERATING

    def test_batch_generate_request_defaults(self) -> None:
        """BatchGenerateRequest should have sensible defaults."""
        req = BatchGenerateRequest(schema_id="test")
        assert req.min_samples == 10
        assert req.max_samples is None
        assert req.train_split == 0.9

    def test_batch_generate_request_validation(self) -> None:
        """BatchGenerateRequest should validate train_split bounds."""
        with pytest.raises(Exception):
            BatchGenerateRequest(schema_id="test", train_split=0.1)  # below 0.5
        with pytest.raises(Exception):
            BatchGenerateRequest(schema_id="test", train_split=1.0)  # above 0.99

    def test_inference_sample_quality_score_bounds(self) -> None:
        """InferenceSample quality_score should be bounded [0.0, 1.0]."""
        with pytest.raises(Exception):
            InferenceSample(
                id=uuid4(),
                schema_id="test",
                messages=[Message(role="user", content="hi")],
                response_content="test",
                quality_score=1.5,
            )
        with pytest.raises(Exception):
            InferenceSample(
                id=uuid4(),
                schema_id="test",
                messages=[Message(role="user", content="hi")],
                response_content="test",
                quality_score=-0.1,
            )


# ---------------------------------------------------------------------------
# Lifespan coverage
# ---------------------------------------------------------------------------


class TestLifespan:
    """Test the app lifespan startup/shutdown sequence."""

    @pytest.mark.asyncio
    async def test_lifespan_starts_and_stops(self) -> None:
        """Lifespan should initialise DB and create services on startup, clean up on shutdown."""
        from main import app, lifespan

        test_settings = Settings(
            service_name="test",
            port=8152,
            database_url=PLACEHOLDER_DATABASE_URL,
            data_collector_url="http://localhost:8151",
            model_registry_url="http://localhost:8155",
            minio_endpoint="localhost:9000",
            minio_access_key="testkey",
            minio_secret_key="testsecret",  # pragma: allowlist secret
            minio_bucket="test-bucket",
            minio_use_ssl=False,
            environment="test",
            log_level="DEBUG",
            log_format="console",
        )

        with (
            patch("main.get_settings", return_value=test_settings),
            patch("main.init_db", new_callable=AsyncMock) as mock_init_db,
            patch("main.dispose_db", new_callable=AsyncMock) as mock_dispose_db,
            patch("main.setup_telemetry") as mock_telemetry,
            patch("main.CollectorClient") as mock_collector_cls,
            patch("main.MinioClient"),
            patch("main.FilterService"),
            patch("main.BatchService"),
            patch("main.TriggerService") as mock_trigger_cls,
        ):
            mock_collector_inst = AsyncMock()
            mock_collector_cls.return_value = mock_collector_inst
            mock_trigger_inst = AsyncMock()
            mock_trigger_cls.return_value = mock_trigger_inst

            async with lifespan(app):
                # Verify startup happened
                mock_init_db.assert_called_once_with(PLACEHOLDER_DATABASE_URL)
                mock_telemetry.assert_called_once()
                assert hasattr(app.state, "batch_service")
                assert hasattr(app.state, "trigger_service")

            # Verify shutdown happened
            mock_collector_inst.close.assert_called_once()
            mock_trigger_inst.close.assert_called_once()
            mock_dispose_db.assert_called_once()


# ---------------------------------------------------------------------------
# Settings edge cases
# ---------------------------------------------------------------------------


class TestSettings:
    """Test Settings configuration."""

    def test_settings_defaults(self) -> None:
        """Settings should have correct default values."""
        s = Settings()
        assert s.service_name == "training-data-generator"
        assert s.port == 8152
        assert s.version == "0.1.0"
        assert s.train_split == 0.9

    def test_get_settings_cached(self) -> None:
        """get_settings should return a cached singleton."""
        from settings import get_settings

        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_normalises_asyncpg_to_psycopg(self) -> None:
        """asyncpg URL should be rewritten to psycopg per the project standard."""
        s = Settings(database_url="postgresql+asyncpg://user:pw@host:5432/dbname")
        assert s.database_url == "postgresql+psycopg://user:pw@host:5432/dbname"

    def test_normalises_bare_postgresql_to_psycopg(self) -> None:
        """Bare ``postgresql://`` should be promoted to ``postgresql+psycopg://``."""
        s = Settings(database_url="postgresql://user:pw@host:5432/dbname")
        assert s.database_url == "postgresql+psycopg://user:pw@host:5432/dbname"

    def test_psycopg_url_passthrough_unchanged(self) -> None:
        """A URL that already specifies psycopg should be left untouched."""
        url = "postgresql+psycopg://user:pw@host:5432/dbname"
        s = Settings(database_url=url)
        assert s.database_url == url

    def test_non_postgres_url_passthrough(self) -> None:
        """Non-postgres URLs (e.g. sqlite) should not be touched by the normaliser."""
        url = "sqlite+aiosqlite:///./test.db"
        s = Settings(database_url=url)
        assert s.database_url == url

    def test_normalises_via_environment_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """asyncpg URL supplied via TRAINING_DATA_GEN_DATABASE_URL should be normalised."""
        monkeypatch.setenv(
            "TRAINING_DATA_GEN_DATABASE_URL",
            "postgresql+asyncpg://u:p@h:5432/d",
        )
        s = Settings()
        assert s.database_url == "postgresql+psycopg://u:p@h:5432/d"


# ---------------------------------------------------------------------------
# Collector returning empty/None responses
# ---------------------------------------------------------------------------


class TestCollectorEmptyResponses:
    """Test batch generation when collector returns empty or degenerate data."""

    @pytest.mark.asyncio
    async def test_collector_returns_none_response_text(self) -> None:
        """export_samples should return empty list when response text is just whitespace."""
        settings = Settings(
            service_name="test",
            port=8152,
            data_collector_url="http://localhost:8151",
            model_registry_url="http://localhost:8155",
            minio_endpoint="localhost:9000",
            minio_access_key="testkey",
            minio_secret_key="testsecret",  # pragma: allowlist secret
            minio_bucket="test-bucket",
            minio_use_ssl=False,
            environment="test",
            log_level="DEBUG",
            log_format="console",
        )
        mock_transport = httpx.MockTransport(
            lambda request: httpx.Response(200, text="\n\n  \n"),
        )
        async with httpx.AsyncClient(
            transport=mock_transport, base_url="http://localhost:8151"
        ) as http_client:
            client = CollectorClient(settings, http_client=http_client)
            result = await client.export_samples(schema_id="test")
        assert result == []

    @pytest.mark.asyncio
    async def test_batch_with_all_filtered_samples(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
    ) -> None:
        """Batch should fail when ALL samples are filtered out (too short)."""
        # All samples have response content that is too short (< 5 chars)
        bad_samples = [make_sample(response_content="hi") for _ in range(20)]
        mock_collector_client.export_samples.return_value = bad_samples

        request = BatchGenerateRequest(schema_id="test_schema", min_samples=5)
        batch = await batch_service.generate_batch(request)

        assert batch.status == BatchStatus.FAILED
        assert batch.total_samples == 20
        assert "0 samples passed filtering" in (batch.error_message or "")
        assert batch.filtered_samples == 20

    @pytest.mark.asyncio
    async def test_batch_with_all_invalid_json(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
    ) -> None:
        """Batch should fail when all samples have invalid JSON responses."""
        invalid_samples = [
            make_sample(response_content="This is not JSON at all, it is plain text"),
            make_sample(response_content="Another plain text response here too"),
            make_sample(response_content="Yet more non-JSON content in response"),
        ]
        mock_collector_client.export_samples.return_value = invalid_samples

        request = BatchGenerateRequest(schema_id="test_schema", min_samples=2)
        batch = await batch_service.generate_batch(request)

        assert batch.status == BatchStatus.FAILED
        assert batch.filtered_samples == 3
        assert batch.filter_reasons.get("invalid_json", 0) == 3

    @pytest.mark.asyncio
    async def test_batch_with_mixed_error_patterns(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
    ) -> None:
        """Batch should correctly categorize multiple filter reasons."""
        samples = [
            make_sample(response_content="hi"),  # too_short
            make_sample(response_content="I cannot help with that"),  # error_pattern
            make_sample(response_content="not json"),  # invalid_json
            make_sample(response_content=json.dumps({"ok": True})),  # passes
            make_sample(response_content=json.dumps({"ok": "second"})),  # passes
        ]
        mock_collector_client.export_samples.return_value = samples

        request = BatchGenerateRequest(schema_id="test_schema", min_samples=1)
        batch = await batch_service.generate_batch(request)

        assert batch.status == BatchStatus.COMPLETED
        assert batch.filter_reasons.get("too_short", 0) >= 1
        # The remaining two get filtered for error_pattern or invalid_json
        assert batch.filtered_samples == 3
        assert batch.train_samples + batch.valid_samples == 2


# ---------------------------------------------------------------------------
# Partial MinIO upload failures
# ---------------------------------------------------------------------------


class TestMinioPartialFailures:
    """Test batch generation when MinIO partially fails."""

    @pytest.mark.asyncio
    async def test_second_upload_fails(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
        mock_minio_client: AsyncMock,
    ) -> None:
        """Batch should fail if the valid.jsonl upload fails (second call)."""
        samples = make_samples(count=100)
        mock_collector_client.export_samples.return_value = samples

        # First upload (train.jsonl) succeeds, second (valid.jsonl) fails
        mock_minio_client.upload_jsonl.side_effect = [
            "test-bucket/train.jsonl",
            ConnectionError("MinIO connection reset on second upload"),
        ]

        request = BatchGenerateRequest(schema_id="test_schema", min_samples=10)
        batch = await batch_service.generate_batch(request)

        assert batch.status == BatchStatus.FAILED
        assert "MinIO connection reset" in (batch.error_message or "")
        # ensure_bucket was called before uploads
        mock_minio_client.ensure_bucket.assert_called_once()
        # First upload succeeded, second failed
        assert mock_minio_client.upload_jsonl.call_count == 2

    @pytest.mark.asyncio
    async def test_ensure_bucket_timeout(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
        mock_minio_client: AsyncMock,
    ) -> None:
        """Batch should fail if ensure_bucket times out."""
        samples = make_samples(count=50)
        mock_collector_client.export_samples.return_value = samples
        mock_minio_client.ensure_bucket.side_effect = TimeoutError("MinIO endpoint timed out")

        request = BatchGenerateRequest(schema_id="test_schema", min_samples=10)
        batch = await batch_service.generate_batch(request)

        assert batch.status == BatchStatus.FAILED
        assert "timed out" in (batch.error_message or "")
        # upload_jsonl should not have been called
        mock_minio_client.upload_jsonl.assert_not_called()

    @pytest.mark.asyncio
    async def test_upload_key_pattern(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
        mock_minio_client: AsyncMock,
    ) -> None:
        """MinIO upload keys should follow the training-data/{schema}/batch-{uuid}/ pattern."""
        samples = make_samples(count=20)
        mock_collector_client.export_samples.return_value = samples

        request = BatchGenerateRequest(schema_id="my-schema", min_samples=1)
        batch = await batch_service.generate_batch(request)

        assert batch.status == BatchStatus.COMPLETED
        # Check both uploaded keys follow the expected pattern
        upload_calls = mock_minio_client.upload_jsonl.call_args_list
        train_key = upload_calls[0].args[1]
        valid_key = upload_calls[1].args[1]

        expected_prefix = f"training-data/my-schema/batch-{batch.id}/"
        assert train_key == f"{expected_prefix}train.jsonl"
        assert valid_key == f"{expected_prefix}valid.jsonl"
        assert batch.minio_path == expected_prefix
        assert batch.train_file == train_key
        assert batch.valid_file == valid_key


# ---------------------------------------------------------------------------
# Network timeouts
# ---------------------------------------------------------------------------


class TestNetworkTimeouts:
    """Test batch generation under network timeout conditions."""

    @pytest.mark.asyncio
    async def test_collector_export_timeout(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
    ) -> None:
        """Batch should fail gracefully when collector export times out."""
        mock_collector_client.export_samples.side_effect = httpx.ReadTimeout("Read timed out")

        request = BatchGenerateRequest(schema_id="test_schema", min_samples=10)
        batch = await batch_service.generate_batch(request)

        assert batch.status == BatchStatus.FAILED
        assert "timed out" in (batch.error_message or "").lower() or "ReadTimeout" in (
            batch.error_message or ""
        )
        # Batch should still be persisted to DB
        retrieved = await batch_service.get_batch(str(batch.id))
        assert retrieved is not None
        assert retrieved.status == BatchStatus.FAILED

    @pytest.mark.asyncio
    async def test_collector_mark_used_timeout(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
        mock_minio_client: AsyncMock,
    ) -> None:
        """Batch should fail if mark_used times out after successful upload."""
        samples = make_samples(count=50)
        mock_collector_client.export_samples.return_value = samples
        mock_collector_client.mark_used.side_effect = httpx.ConnectTimeout("Connection timed out")

        request = BatchGenerateRequest(schema_id="test_schema", min_samples=10)
        batch = await batch_service.generate_batch(request)

        assert batch.status == BatchStatus.FAILED
        # Files were uploaded before mark_used failed
        assert mock_minio_client.upload_jsonl.call_count == 2

    @pytest.mark.asyncio
    async def test_collector_http_error(self) -> None:
        """CollectorClient should raise on non-2xx status from data-collector."""
        settings = Settings(
            service_name="test",
            port=8152,
            data_collector_url="http://localhost:8151",
            model_registry_url="http://localhost:8155",
            minio_endpoint="localhost:9000",
            minio_access_key="testkey",
            minio_secret_key="testsecret",  # pragma: allowlist secret
            minio_bucket="test-bucket",
            minio_use_ssl=False,
            environment="test",
            log_level="DEBUG",
            log_format="console",
        )
        mock_transport = httpx.MockTransport(
            lambda request: httpx.Response(500, text="Internal Server Error"),
        )
        async with httpx.AsyncClient(
            transport=mock_transport, base_url="http://localhost:8151"
        ) as http_client:
            client = CollectorClient(settings, http_client=http_client)
            with pytest.raises(httpx.HTTPStatusError):
                await client.export_samples(schema_id="test")

    @pytest.mark.asyncio
    async def test_mark_used_http_error(self) -> None:
        """CollectorClient.mark_used should raise on non-2xx status."""
        settings = Settings(
            service_name="test",
            port=8152,
            data_collector_url="http://localhost:8151",
            model_registry_url="http://localhost:8155",
            minio_endpoint="localhost:9000",
            minio_access_key="testkey",
            minio_secret_key="testsecret",  # pragma: allowlist secret
            minio_bucket="test-bucket",
            minio_use_ssl=False,
            environment="test",
            log_level="DEBUG",
            log_format="console",
        )
        mock_transport = httpx.MockTransport(
            lambda request: httpx.Response(503, text="Service Unavailable"),
        )
        async with httpx.AsyncClient(
            transport=mock_transport, base_url="http://localhost:8151"
        ) as http_client:
            client = CollectorClient(settings, http_client=http_client)
            with pytest.raises(httpx.HTTPStatusError):
                await client.mark_used(sample_ids=[uuid4()], batch_id="batch-1")


# ---------------------------------------------------------------------------
# Malformed sample data
# ---------------------------------------------------------------------------


class TestMalformedSampleData:
    """Test handling of malformed or edge-case sample data."""

    @pytest.mark.asyncio
    async def test_export_samples_malformed_jsonl_line(self) -> None:
        """export_samples should raise on malformed JSONL lines."""
        settings = Settings(
            service_name="test",
            port=8152,
            data_collector_url="http://localhost:8151",
            model_registry_url="http://localhost:8155",
            minio_endpoint="localhost:9000",
            minio_access_key="testkey",
            minio_secret_key="testsecret",  # pragma: allowlist secret
            minio_bucket="test-bucket",
            minio_use_ssl=False,
            environment="test",
            log_level="DEBUG",
            log_format="console",
        )
        # First line is valid, second is malformed
        sample = InferenceSample(
            id=uuid4(),
            schema_id="test",
            messages=[Message(role="user", content="hello")],
            response_content='{"answer": 1}',
            provider="Qwen3-4B-AWQ",
        )
        bad_jsonl = sample.model_dump_json() + "\n{this is not valid json}"

        mock_transport = httpx.MockTransport(
            lambda request: httpx.Response(200, text=bad_jsonl),
        )
        async with httpx.AsyncClient(
            transport=mock_transport, base_url="http://localhost:8151"
        ) as http_client:
            client = CollectorClient(settings, http_client=http_client)
            with pytest.raises(Exception):  # Pydantic validation or JSON parse error
                await client.export_samples(schema_id="test")

    @pytest.mark.asyncio
    async def test_export_samples_with_exclude_providers(self) -> None:
        """export_samples should filter out excluded providers client-side."""
        settings = Settings(
            service_name="test",
            port=8152,
            data_collector_url="http://localhost:8151",
            model_registry_url="http://localhost:8155",
            minio_endpoint="localhost:9000",
            minio_access_key="testkey",
            minio_secret_key="testsecret",  # pragma: allowlist secret
            minio_bucket="test-bucket",
            minio_use_ssl=False,
            environment="test",
            log_level="DEBUG",
            log_format="console",
        )
        sample_a = InferenceSample(
            id=uuid4(),
            schema_id="test",
            messages=[Message(role="user", content="hello")],
            response_content='{"answer": 1}',
            provider="Qwen3-4B-AWQ",
        )
        sample_b = InferenceSample(
            id=uuid4(),
            schema_id="test",
            messages=[Message(role="user", content="world")],
            response_content='{"answer": 2}',
            provider="deepseek",
        )
        sample_c = InferenceSample(
            id=uuid4(),
            schema_id="test",
            messages=[Message(role="user", content="test")],
            response_content='{"answer": 3}',
            provider="claude",
        )
        jsonl = "\n".join(s.model_dump_json() for s in [sample_a, sample_b, sample_c])

        mock_transport = httpx.MockTransport(
            lambda request: httpx.Response(200, text=jsonl),
        )
        async with httpx.AsyncClient(
            transport=mock_transport, base_url="http://localhost:8151"
        ) as http_client:
            client = CollectorClient(settings, http_client=http_client)
            result = await client.export_samples(
                schema_id="test",
                exclude_providers=["deepseek", "Claude"],  # case-insensitive
            )
        assert len(result) == 1
        assert result[0].provider == "Qwen3-4B-AWQ"

    @pytest.mark.asyncio
    async def test_batch_with_error_pattern_responses(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
    ) -> None:
        """Samples with LLM refusal patterns should be filtered out."""
        refusal_samples = [
            make_sample(response_content=json.dumps({"error": "I cannot help with that"})),
            make_sample(response_content=json.dumps({"data": "I'm sorry, but I can't"})),
            make_sample(response_content=json.dumps({"msg": "As an AI, I cannot"})),
        ]
        valid_samples = make_samples(count=5)
        mock_collector_client.export_samples.return_value = refusal_samples + valid_samples

        request = BatchGenerateRequest(schema_id="test_schema", min_samples=1)
        batch = await batch_service.generate_batch(request)

        assert batch.status == BatchStatus.COMPLETED
        assert batch.filter_reasons.get("error_pattern", 0) == 3
        assert batch.train_samples + batch.valid_samples == 5

    @pytest.mark.asyncio
    async def test_batch_persists_through_db_roundtrip(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
    ) -> None:
        """Verify batch data survives a full save/load DB roundtrip with real SQLite."""
        samples = make_samples(count=50, fallback_count=5)
        mock_collector_client.export_samples.return_value = samples

        request = BatchGenerateRequest(schema_id="roundtrip-test", min_samples=10)
        original = await batch_service.generate_batch(request)

        # Retrieve from DB and verify all fields match
        retrieved = await batch_service.get_batch(str(original.id))
        assert retrieved is not None
        assert retrieved.id == original.id
        assert retrieved.schema_id == "roundtrip-test"
        assert retrieved.status == original.status
        assert retrieved.total_samples == original.total_samples
        assert retrieved.train_samples == original.train_samples
        assert retrieved.valid_samples == original.valid_samples
        assert retrieved.filtered_samples == original.filtered_samples
        assert retrieved.filter_reasons == original.filter_reasons
        assert retrieved.minio_path == original.minio_path
        assert retrieved.train_file == original.train_file
        assert retrieved.valid_file == original.valid_file
        assert retrieved.source_providers == original.source_providers
        assert retrieved.teacher_samples == original.teacher_samples
        assert retrieved.error_message == original.error_message

    @pytest.mark.asyncio
    async def test_export_samples_sends_correct_query_params(self) -> None:
        """Verify export_samples sends the correct query params to data-collector."""
        settings = Settings(
            service_name="test",
            port=8152,
            data_collector_url="http://localhost:8151",
            model_registry_url="http://localhost:8155",
            minio_endpoint="localhost:9000",
            minio_access_key="testkey",
            minio_secret_key="testsecret",  # pragma: allowlist secret
            minio_bucket="test-bucket",
            minio_use_ssl=False,
            environment="test",
            log_level="DEBUG",
            log_format="console",
        )
        captured_request: httpx.Request | None = None

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal captured_request
            captured_request = request
            return httpx.Response(200, text="")

        mock_transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=mock_transport, base_url="http://localhost:8151"
        ) as http_client:
            client = CollectorClient(settings, http_client=http_client)
            await client.export_samples(
                schema_id="my-schema",
                only_validated=True,
                only_unused=False,
                limit=100,
                fallback_used=True,
            )

        assert captured_request is not None
        url_str = str(captured_request.url)
        assert "/api/samples/export" in url_str
        assert "schema_id=my-schema" in url_str
        assert "only_validated=true" in url_str.lower()
        assert "only_unused=false" in url_str.lower()
        assert "limit=100" in url_str
        assert "fallback_used=true" in url_str.lower()

    @pytest.mark.asyncio
    async def test_per_schema_max_length_override(self) -> None:
        """FilterService should apply per-schema max_length overrides."""
        service = FilterService(
            max_length=10_000,
            max_length_overrides={"strict-schema": 50},
        )

        # Under global limit but over per-schema limit
        response = json.dumps({"data": "x" * 60})
        sample = make_sample(response_content=response)

        # Without schema_id, passes global limit
        filtered_global, reasons_global = service.filter_samples([sample])
        assert len(filtered_global) == 1

        # With strict-schema, gets filtered by per-schema override
        filtered_schema, reasons_schema = service.filter_samples(
            [sample], schema_id="strict-schema"
        )
        assert len(filtered_schema) == 0
        assert reasons_schema.get("too_long", 0) == 1


# ---------------------------------------------------------------------------
# Lifespan JSON parse error coverage (main.py lines 60-61)
# ---------------------------------------------------------------------------


class TestLifespanJsonParseError:
    """Test the JSON parse error handler in lifespan for max_response_length_overrides."""

    @pytest.mark.asyncio
    async def test_invalid_json_override_falls_back_to_empty(self) -> None:
        """Lifespan should gracefully handle invalid JSON in max_response_length_overrides."""
        from main import app, lifespan

        test_settings = Settings(
            service_name="test",
            port=8152,
            database_url=PLACEHOLDER_DATABASE_URL,
            data_collector_url="http://localhost:8151",
            model_registry_url="http://localhost:8155",
            minio_endpoint="localhost:9000",
            minio_access_key="testkey",
            minio_secret_key="testsecret",  # pragma: allowlist secret
            minio_bucket="test-bucket",
            minio_use_ssl=False,
            environment="test",
            log_level="DEBUG",
            log_format="console",
            max_response_length_overrides="not valid json {{{",
        )

        with (
            patch("main.get_settings", return_value=test_settings),
            patch("main.init_db", new_callable=AsyncMock) as mock_init_db,
            patch("main.dispose_db", new_callable=AsyncMock),
            patch("main.setup_telemetry"),
            patch("main.CollectorClient") as mock_collector_cls,
            patch("main.MinioClient"),
            patch("main.RegistryClient") as mock_registry_cls,
            patch("main.BatchService"),
            patch("main.TriggerService") as mock_trigger_cls,
        ):
            mock_collector_inst = AsyncMock()
            mock_collector_cls.return_value = mock_collector_inst
            mock_registry_inst = AsyncMock()
            mock_registry_cls.return_value = mock_registry_inst
            mock_trigger_inst = AsyncMock()
            mock_trigger_cls.return_value = mock_trigger_inst

            async with lifespan(app):
                # If we get here, the lifespan didn't crash on bad JSON
                mock_init_db.assert_called_once()
                # FilterService should have been created with empty overrides
                assert hasattr(app.state, "filter_service")

            # Verify shutdown
            mock_collector_inst.close.assert_called_once()
            mock_registry_inst.close.assert_called_once()
            mock_trigger_inst.close.assert_called_once()


# ---------------------------------------------------------------------------
# Logging config: OTEL trace context injection with valid span
# ---------------------------------------------------------------------------


class TestTraceContextInjection:
    """Test add_trace_context with a valid recording span."""

    def test_trace_context_injected_with_valid_span(self) -> None:
        """add_trace_context should inject trace_id and span_id when span is valid."""
        from logging_config import add_trace_context

        mock_ctx = MagicMock()
        mock_ctx.is_valid = True
        mock_ctx.trace_id = 0x1234567890ABCDEF1234567890ABCDEF
        mock_ctx.span_id = 0x1234567890ABCDEF

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_span.get_span_context.return_value = mock_ctx

        with (
            patch("logging_config.OTEL_AVAILABLE", True),
            patch("logging_config.trace") as mock_trace,
        ):
            mock_trace.get_current_span.return_value = mock_span
            event_dict: dict[str, Any] = {"event": "test"}
            result = add_trace_context(None, "info", event_dict)

        assert "trace_id" in result
        assert "span_id" in result
        assert result["trace_id"] == format(mock_ctx.trace_id, "032x")
        assert result["span_id"] == format(mock_ctx.span_id, "016x")

    def test_trace_context_not_injected_when_span_not_recording(self) -> None:
        """add_trace_context should skip injection when span is not recording."""
        from logging_config import add_trace_context

        mock_span = MagicMock()
        mock_span.is_recording.return_value = False

        with (
            patch("logging_config.OTEL_AVAILABLE", True),
            patch("logging_config.trace") as mock_trace,
        ):
            mock_trace.get_current_span.return_value = mock_span
            event_dict: dict[str, Any] = {"event": "test"}
            result = add_trace_context(None, "info", event_dict)

        assert "trace_id" not in result
        assert "span_id" not in result

    def test_trace_context_not_injected_when_context_invalid(self) -> None:
        """add_trace_context should skip injection when span context is invalid."""
        from logging_config import add_trace_context

        mock_ctx = MagicMock()
        mock_ctx.is_valid = False

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_span.get_span_context.return_value = mock_ctx

        with (
            patch("logging_config.OTEL_AVAILABLE", True),
            patch("logging_config.trace") as mock_trace,
        ):
            mock_trace.get_current_span.return_value = mock_span
            event_dict: dict[str, Any] = {"event": "test"}
            result = add_trace_context(None, "info", event_dict)

        assert "trace_id" not in result
        assert "span_id" not in result


# ---------------------------------------------------------------------------
# Teacher sample supplementation edge case: no new samples
# ---------------------------------------------------------------------------


class TestTeacherSupplementNoNewSamples:
    """Test teacher supplementation when there are zero new samples."""

    @pytest.mark.asyncio
    async def test_no_new_samples_teacher_only(
        self,
        _init_test_db: None,
        test_settings: Settings,
        mock_minio_client: AsyncMock,
        mock_registry_client: AsyncMock,
        filter_service: FilterService,
    ) -> None:
        """When no new samples exist, teacher samples alone should fill the batch."""
        new_samples: list[InferenceSample] = []
        teacher_samples = make_samples(count=20, provider="deepseek", fallback_count=20)

        mock_collector = AsyncMock(spec=CollectorClient)
        mock_collector.close = AsyncMock()

        async def dispatch(*, schema_id: str, **kwargs: Any) -> list[InferenceSample]:
            if kwargs.get("fallback_used") is True:
                return teacher_samples
            return new_samples

        mock_collector.export_samples.side_effect = dispatch
        mock_collector.mark_used = AsyncMock()

        test_settings.reuse_teacher_samples = True
        test_settings.max_teacher_ratio = 0.7
        test_settings.min_samples = 1

        service = BatchService(
            collector_client=mock_collector,
            filter_service=filter_service,
            minio_client=mock_minio_client,
            registry_client=mock_registry_client,
            settings=test_settings,
        )

        request = BatchGenerateRequest(schema_id="test_schema", min_samples=1)
        batch = await service.generate_batch(request)

        assert batch.status == BatchStatus.COMPLETED
        # With 0 new samples, teachers should still fill the batch
        total = batch.train_samples + batch.valid_samples
        assert total > 0


# ---------------------------------------------------------------------------
# PEP 758 except branches: both exception types
# ---------------------------------------------------------------------------


class TestPEP758ExceptBranches:
    """Test that both exception types in PEP 758 multi-except clauses are caught.

    Python 3.14 PEP 758 syntax: ``except ValueError, json.JSONDecodeError:``
    is equivalent to ``except (ValueError, json.JSONDecodeError):``.

    These tests explicitly exercise each exception type to ensure full branch
    coverage.
    """

    @pytest.mark.asyncio
    async def test_lifespan_catches_value_error_on_json_override(self) -> None:
        """Lifespan should catch plain ValueError from json.loads (PEP 758 line 61).

        json.loads always raises JSONDecodeError (subclass of ValueError), so we
        patch json.loads to raise a bare ValueError to exercise the other branch.
        The lifespan does ``import json as _json`` locally, so patching the
        ``json`` module's ``loads`` function is sufficient.
        """
        from main import app, lifespan

        test_settings = Settings(
            service_name="test",
            port=8152,
            database_url=PLACEHOLDER_DATABASE_URL,
            data_collector_url="http://localhost:8151",
            model_registry_url="http://localhost:8155",
            minio_endpoint="localhost:9000",
            minio_access_key="testkey",
            minio_secret_key="testsecret",  # pragma: allowlist secret
            minio_bucket="test-bucket",
            minio_use_ssl=False,
            environment="test",
            log_level="DEBUG",
            log_format="console",
            max_response_length_overrides="{}",  # valid JSON, but we'll patch json.loads
        )

        with (
            patch("main.get_settings", return_value=test_settings),
            patch("main.init_db", new_callable=AsyncMock),
            patch("main.dispose_db", new_callable=AsyncMock),
            patch("main.setup_telemetry"),
            patch("main.CollectorClient") as mock_collector_cls,
            patch("main.MinioClient"),
            patch("main.RegistryClient") as mock_registry_cls,
            patch("main.FilterService") as mock_filter_cls,
            patch("main.BatchService"),
            patch("main.TriggerService") as mock_trigger_cls,
            patch("json.loads", side_effect=ValueError("totally broken")),
        ):
            mock_collector_inst = AsyncMock()
            mock_collector_cls.return_value = mock_collector_inst
            mock_registry_inst = AsyncMock()
            mock_registry_cls.return_value = mock_registry_inst
            mock_trigger_inst = AsyncMock()
            mock_trigger_cls.return_value = mock_trigger_inst

            async with lifespan(app):
                # Should not crash; FilterService gets empty overrides
                mock_filter_cls.assert_called_once()
                call_kwargs = mock_filter_cls.call_args.kwargs
                assert call_kwargs.get("max_length_overrides") == {}

            mock_collector_inst.close.assert_called_once()
            mock_registry_inst.close.assert_called_once()
            mock_trigger_inst.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_readiness_catches_oserror_from_collector(
        self,
        test_app: AsyncClient,
        mock_collector_client: AsyncMock,
        mock_minio_client: AsyncMock,
    ) -> None:
        """Readiness should catch OSError from collector (PEP 758 line 192).

        Exercises the OSError branch of ``except httpx.HTTPError, OSError:``.
        """
        mock_collector_client.list_schema_ids.side_effect = OSError("ECONNREFUSED")
        mock_minio_client.check_connectivity.return_value = True

        response = await test_app.get("/health/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["ready"] is False
        assert data["data_collector"] is False
        assert data["minio"] is True


# ---------------------------------------------------------------------------
# Narrowed _get_quality_score except branches
# ---------------------------------------------------------------------------


class TestGetQualityScoreNarrowExcept:
    """Test the narrowed (httpx.HTTPError, OSError) except in _get_quality_score."""

    @pytest.mark.asyncio
    async def test_httpx_error_returns_none(
        self,
        trigger_service: TriggerService,
        mock_http_client: AsyncMock,
    ) -> None:
        """_get_quality_score should return None on httpx.HTTPError."""
        mock_http_client.get.side_effect = httpx.ConnectError("connection refused")

        result = await trigger_service._get_quality_score("test_schema")
        assert result is None

    @pytest.mark.asyncio
    async def test_os_error_returns_none(
        self,
        trigger_service: TriggerService,
        mock_http_client: AsyncMock,
    ) -> None:
        """_get_quality_score should return None on OSError."""
        mock_http_client.get.side_effect = OSError("ECONNREFUSED")

        result = await trigger_service._get_quality_score("test_schema")
        assert result is None

    @pytest.mark.asyncio
    async def test_unexpected_error_propagates(
        self,
        trigger_service: TriggerService,
        mock_http_client: AsyncMock,
    ) -> None:
        """_get_quality_score should let unexpected errors (e.g. TypeError) propagate."""
        mock_http_client.get.side_effect = TypeError("programming bug")

        with pytest.raises(TypeError, match="programming bug"):
            await trigger_service._get_quality_score("test_schema")


# ---------------------------------------------------------------------------
# Narrowed _supplement_with_teachers except branches
# ---------------------------------------------------------------------------


class TestSupplementTeachersNarrowExcept:
    """Test the narrowed (httpx.HTTPError, OSError) except in _supplement_with_teachers."""

    @pytest.mark.asyncio
    async def test_httpx_error_returns_original_samples(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
        mock_minio_client: AsyncMock,
    ) -> None:
        """_supplement_with_teachers should return original samples on httpx error."""
        new_samples = make_samples(count=30)
        mock_collector_client.export_samples.side_effect = [
            new_samples,  # first call: fetch new samples
            httpx.ConnectError("teacher fetch failed"),  # second call: fetch teachers
        ]

        request = BatchGenerateRequest(schema_id="test_schema", min_samples=10)
        batch = await batch_service.generate_batch(request)

        # Should still complete with just new samples
        assert batch.status == BatchStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_os_error_returns_original_samples(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
        mock_minio_client: AsyncMock,
    ) -> None:
        """_supplement_with_teachers should return original samples on OSError."""
        new_samples = make_samples(count=30)
        mock_collector_client.export_samples.side_effect = [
            new_samples,  # first call: fetch new samples
            OSError("DNS resolution failed"),  # second call: fetch teachers
        ]

        request = BatchGenerateRequest(schema_id="test_schema", min_samples=10)
        batch = await batch_service.generate_batch(request)

        assert batch.status == BatchStatus.COMPLETED
