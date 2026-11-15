"""Tests for CollectorClient edge cases and db module.

Covers CollectorClient methods not exercised by the batch generation tests
(close, get_sample_count, list_schema_ids, mark_used with empty list,
export_samples with limit) and db.get_session error handling.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.pool import NullPool

import db as db_module
from db import check_db_connectivity, dispose_db, get_session, init_db
from models.domain import InferenceSample, Message
from services.collector_client import CollectorClient
from settings import Settings


@pytest.fixture
def collector_settings() -> Settings:
    """Minimal settings for CollectorClient tests."""
    return Settings(
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


class TestCollectorClientClose:
    """Test CollectorClient.close() lifecycle."""

    async def test_close_when_owns_client(self, collector_settings: Settings) -> None:
        """close() should close the HTTP client when CollectorClient created it."""
        client = CollectorClient(collector_settings)
        assert client._owns_client is True
        # Should not raise
        await client.close()

    async def test_close_when_injected_client(self, collector_settings: Settings) -> None:
        """close() should NOT close the HTTP client when it was injected."""
        http_client = httpx.AsyncClient(base_url="http://localhost:8151")
        client = CollectorClient(collector_settings, http_client=http_client)
        assert client._owns_client is False
        # close() should be a no-op for injected clients
        await client.close()
        # The injected client should still be usable (not closed)
        assert not http_client.is_closed
        await http_client.aclose()


class TestCollectorClientMarkUsedEmpty:
    """Test mark_used with empty sample_ids list."""

    async def test_mark_used_empty_list_returns_early(self, collector_settings: Settings) -> None:
        """mark_used should return immediately when sample_ids is empty."""

        # Use a transport that would fail if called
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("HTTP call should not be made for empty sample_ids")

        mock_transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=mock_transport, base_url="http://localhost:8151"
        ) as http_client:
            client = CollectorClient(collector_settings, http_client=http_client)
            # Should return without making any HTTP call
            await client.mark_used(sample_ids=[], batch_id="batch-1")


class TestCollectorClientGetSampleCount:
    """Test get_sample_count method."""

    async def test_get_sample_count(self, collector_settings: Settings) -> None:
        """get_sample_count should parse the count from the JSON response."""

        def handler(request: httpx.Request) -> httpx.Response:
            assert "/api/samples/count" in str(request.url)
            return httpx.Response(200, json={"count": 42})

        mock_transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=mock_transport, base_url="http://localhost:8151"
        ) as http_client:
            client = CollectorClient(collector_settings, http_client=http_client)
            count = await client.get_sample_count(schema_id="test_schema")

        assert count == 42

    async def test_get_sample_count_missing_key(self, collector_settings: Settings) -> None:
        """get_sample_count should return 0 when 'count' key is missing."""
        mock_transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={}),
        )
        async with httpx.AsyncClient(
            transport=mock_transport, base_url="http://localhost:8151"
        ) as http_client:
            client = CollectorClient(collector_settings, http_client=http_client)
            count = await client.get_sample_count(schema_id="test_schema")

        assert count == 0


class TestCollectorClientListSchemaIds:
    """Test list_schema_ids method."""

    async def test_list_schema_ids(self, collector_settings: Settings) -> None:
        """list_schema_ids should parse schema_ids from the JSON response."""

        def handler(request: httpx.Request) -> httpx.Response:
            assert "/api/schemas" in str(request.url)
            return httpx.Response(200, json={"schema_ids": ["schema_a", "schema_b", "schema_c"]})

        mock_transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=mock_transport, base_url="http://localhost:8151"
        ) as http_client:
            client = CollectorClient(collector_settings, http_client=http_client)
            result = await client.list_schema_ids()

        assert result == ["schema_a", "schema_b", "schema_c"]

    async def test_list_schema_ids_empty(self, collector_settings: Settings) -> None:
        """list_schema_ids should return empty list when no schemas exist."""
        mock_transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"schema_ids": []}),
        )
        async with httpx.AsyncClient(
            transport=mock_transport, base_url="http://localhost:8151"
        ) as http_client:
            client = CollectorClient(collector_settings, http_client=http_client)
            result = await client.list_schema_ids()

        assert result == []


class TestCollectorClientExportWithLimit:
    """Test export_samples with limit parameter."""

    async def test_export_samples_sends_limit_param(self, collector_settings: Settings) -> None:
        """export_samples should include 'limit' in query params when specified."""
        captured_request: httpx.Request | None = None

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal captured_request
            captured_request = request
            sample = InferenceSample(
                id=uuid4(),
                schema_id="test",
                messages=[Message(role="user", content="hello")],
                response_content='{"answer": 1}',
                provider="Qwen3-4B-AWQ",
            )
            return httpx.Response(200, text=sample.model_dump_json())

        mock_transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=mock_transport, base_url="http://localhost:8151"
        ) as http_client:
            client = CollectorClient(collector_settings, http_client=http_client)
            result = await client.export_samples(schema_id="test", limit=50)

        assert captured_request is not None
        assert "limit=50" in str(captured_request.url)
        assert len(result) == 1


class TestDatabaseGetSession:
    """Test db.get_session edge cases."""

    async def test_get_session_without_init_raises(self) -> None:
        """get_session should raise RuntimeError if database not initialised."""
        # Ensure database is not initialised by disposing first
        await dispose_db()

        with pytest.raises(RuntimeError, match="Database not initialised"):
            async with get_session():
                pass

    async def test_get_session_rollback_on_error(self, test_database_url: str) -> None:
        """get_session should rollback on exception and re-raise."""

        await init_db(test_database_url)
        try:
            with pytest.raises(ValueError, match="test error"):
                async with get_session() as session:
                    # Verify session was properly yielded before raising
                    assert session is not None
                    assert session.is_active
                    raise ValueError("test error")
        finally:
            await dispose_db()


class TestCheckDbConnectivity:
    """Test db.check_db_connectivity branches."""

    async def test_returns_false_when_engine_not_initialised(self) -> None:
        """check_db_connectivity should return False if init_db has not been called."""
        # Ensure engine is None
        await dispose_db()
        assert db_module._engine is None

        result = await check_db_connectivity()
        assert result is False

    async def test_returns_true_on_successful_query(self, test_database_url: str) -> None:
        """check_db_connectivity should return True when SELECT 1 succeeds."""

        await init_db(test_database_url)
        try:
            result = await check_db_connectivity()
            assert result is True
        finally:
            await dispose_db()

    async def test_returns_false_on_sqlalchemy_error(self) -> None:
        """check_db_connectivity should return False when SQLAlchemyError raised.

        Exercises the (SQLAlchemyError, OSError) narrow except branch by
        replacing the engine with a mock that raises OperationalError.
        """
        # Build a fake async engine whose connect() raises OperationalError
        fake_engine = MagicMock()

        class _FailingCM:
            async def __aenter__(self) -> None:
                raise OperationalError("SELECT 1", {}, Exception("connection refused"))

            async def __aexit__(self, *args: object) -> None:
                return None

        fake_engine.connect = MagicMock(return_value=_FailingCM())

        with patch.object(db_module, "_engine", fake_engine):
            result = await check_db_connectivity()
        assert result is False

    async def test_returns_false_on_os_error(self) -> None:
        """check_db_connectivity should return False when OSError raised.

        Covers the OSError side of the (SQLAlchemyError, OSError) tuple.
        """
        fake_engine = MagicMock()

        class _OSErrorCM:
            async def __aenter__(self) -> None:
                raise OSError("ECONNREFUSED")

            async def __aexit__(self, *args: object) -> None:
                return None

        fake_engine.connect = MagicMock(return_value=_OSErrorCM())

        with patch.object(db_module, "_engine", fake_engine):
            result = await check_db_connectivity()
        assert result is False

    async def test_unexpected_exception_propagates(self) -> None:
        """Unexpected exception types (e.g. TypeError) should NOT be swallowed.

        Verifies the narrow except is genuinely narrow — bugs that aren't
        connectivity-related should crash loudly instead of being hidden.
        """
        fake_engine = MagicMock()

        class _BugCM:
            async def __aenter__(self) -> None:
                raise TypeError("programming bug, not a connectivity issue")

            async def __aexit__(self, *args: object) -> None:
                return None

        fake_engine.connect = MagicMock(return_value=_BugCM())

        with patch.object(db_module, "_engine", fake_engine):
            with pytest.raises(TypeError, match="programming bug"):
                await check_db_connectivity()


class TestNotifyFlywheelHttpFailure:
    """Test trigger_service._notify_flywheel narrow except branch.

    The flywheel notification is fire-and-forget — any httpx error or OSError
    must be logged and swallowed so a flywheel outage cannot fail batch
    generation.
    """

    async def test_notify_flywheel_swallows_httpx_request_error(
        self,
        collector_settings: Settings,
    ) -> None:
        """_notify_flywheel should swallow httpx.RequestError without raising."""
        from models.domain import BatchStatus, TrainingBatch
        from services.batch_service import BatchService
        from services.collector_client import CollectorClient
        from services.filter_service import FilterService
        from services.minio_client import MinioClient
        from services.registry_client import RegistryClient
        from services.trigger_service import TriggerService

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("flywheel unreachable")

        mock_transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=mock_transport, base_url="http://localhost:8155"
        ) as http_client:
            ts = TriggerService(
                settings=collector_settings,
                collector_client=AsyncMock(spec=CollectorClient),
                batch_service=MagicMock(spec=BatchService),
                http_client=http_client,
            )
            batch = TrainingBatch(
                schema_id="test_schema",
                status=BatchStatus.COMPLETED,
                minio_path="bucket/path.jsonl",
            )
            # Should NOT raise — exception is logged and swallowed
            await ts._notify_flywheel("test_schema", batch)

        # Suppress unused-import lint by exercising types
        assert MinioClient is not None
        assert RegistryClient is not None
        assert FilterService is not None

    async def test_notify_flywheel_swallows_http_status_error(
        self,
        collector_settings: Settings,
    ) -> None:
        """_notify_flywheel should swallow 5xx errors raised by raise_for_status."""
        from models.domain import BatchStatus, TrainingBatch
        from services.batch_service import BatchService
        from services.collector_client import CollectorClient
        from services.trigger_service import TriggerService

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, json={"error": "service unavailable"})

        mock_transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=mock_transport, base_url="http://localhost:8155"
        ) as http_client:
            ts = TriggerService(
                settings=collector_settings,
                collector_client=AsyncMock(spec=CollectorClient),
                batch_service=MagicMock(spec=BatchService),
                http_client=http_client,
            )
            batch = TrainingBatch(
                schema_id="test_schema",
                status=BatchStatus.COMPLETED,
                minio_path="bucket/path.jsonl",
            )
            # Should NOT raise — HTTPStatusError is httpx.HTTPError
            await ts._notify_flywheel("test_schema", batch)


class TestInitDb:
    """init_db must not crashloop when startup create_all is denied DDL.

    The alembic migrate init-container is the schema source of truth. Under a
    Phase-D restricted (no-DDL) runtime role, create_all() raises
    permission-denied — init_db downgrades that to a warning so uvicorn startup
    survives (the table already exists, created by the init-container).
    """

    async def test_init_db_tolerates_create_all_permission_denied(self) -> None:
        """create_all raising a permission-denied ProgrammingError must not propagate."""
        await dispose_db()

        fake_engine = MagicMock()

        class _DenyingBegin:
            async def __aenter__(self) -> object:
                raise ProgrammingError(
                    "CREATE TABLE training_batches ...",
                    {},
                    Exception("permission denied for schema public"),
                )

            async def __aexit__(self, *args: object) -> None:
                return None

        fake_engine.begin = MagicMock(return_value=_DenyingBegin())
        fake_engine.dispose = AsyncMock()

        with (
            patch.object(db_module, "create_async_engine", return_value=fake_engine),
            patch.object(db_module, "async_sessionmaker", return_value=MagicMock()),
        ):
            # Must return normally despite create_all raising permission-denied.
            await init_db("postgresql+psycopg://x:y@localhost/z")

        # Engine + session factory are still wired — the app can serve; the
        # init-container owns the schema.
        assert db_module._engine is fake_engine
        assert db_module._session_factory is not None
        await dispose_db()

    async def test_init_db_does_not_swallow_unexpected_errors(self) -> None:
        """A non-DB error (e.g. TypeError) during create_all must crash loudly."""
        await dispose_db()

        fake_engine = MagicMock()

        class _BugBegin:
            async def __aenter__(self) -> object:
                raise TypeError("programming bug, not a DDL/permission issue")

            async def __aexit__(self, *args: object) -> None:
                return None

        fake_engine.begin = MagicMock(return_value=_BugBegin())
        fake_engine.dispose = AsyncMock()

        with (
            patch.object(db_module, "create_async_engine", return_value=fake_engine),
            patch.object(db_module, "async_sessionmaker", return_value=MagicMock()),
        ):
            with pytest.raises(TypeError, match="programming bug"):
                await init_db("postgresql+psycopg://x:y@localhost/z")
        db_module._engine = None
        db_module._session_factory = None


class TestAlembicBaseline:
    """The 0001 baseline migration applies cleanly and is idempotent.

    Also guards against ORM drift: if the schema and the baseline diverge, the
    upgrade/verify below fails in CI. Uses a synchronous psycopg engine against
    the test DB and restores a pristine state afterwards.
    """

    @staticmethod
    def _alembic_cfg(url: str):
        # Build the Config WITHOUT the .ini path so env.py skips fileConfig()
        # and does not reconfigure the test session's logging.
        from alembic.config import Config

        repo_root = Path(__file__).resolve().parents[1]
        cfg = Config()
        cfg.set_main_option("script_location", str(repo_root / "alembic"))
        cfg.set_main_option("sqlalchemy.url", url)
        return cfg

    @staticmethod
    def _reset(engine) -> None:
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS training_batches"))
            conn.execute(text("DROP TABLE IF EXISTS alembic_version"))

    def test_baseline_upgrade_creates_and_downgrade_drops(
        self, monkeypatch, test_database_url: str
    ) -> None:
        from alembic import command

        monkeypatch.setenv("DATABASE_URL", test_database_url)
        cfg = self._alembic_cfg(test_database_url)
        engine = create_engine(test_database_url, poolclass=NullPool)
        try:
            self._reset(engine)
            command.upgrade(cfg, "head")
            with engine.connect() as conn:
                assert inspect(conn).has_table("training_batches")
                ver = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
                assert ver == "0001_initial_schema"
            # Idempotent: a second upgrade is a no-op (already at head).
            command.upgrade(cfg, "head")
            command.downgrade(cfg, "base")
            with engine.connect() as conn:
                assert not inspect(conn).has_table("training_batches")
        finally:
            self._reset(engine)
            engine.dispose()

    def test_baseline_guard_skips_ddl_when_table_preexists(
        self, monkeypatch, test_database_url: str
    ) -> None:
        """Prod reconcile: table already created by create_all, no alembic_version.

        `alembic upgrade head` must NOT raise "already exists" — the has_table
        guard skips the DDL and just records the revision.
        """
        from alembic import command

        from models.db_models import Base

        monkeypatch.setenv("DATABASE_URL", test_database_url)
        cfg = self._alembic_cfg(test_database_url)
        engine = create_engine(test_database_url, poolclass=NullPool)
        try:
            self._reset(engine)
            # Simulate the legacy create_all-created schema (no alembic_version).
            Base.metadata.create_all(engine)
            command.upgrade(cfg, "head")  # must not raise
            with engine.connect() as conn:
                assert inspect(conn).has_table("training_batches")
                ver = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
                assert ver == "0001_initial_schema"
        finally:
            self._reset(engine)
            engine.dispose()


# ---------------------------------------------------------------------------
# Regression: data-collector emits null tool_calls / tool_call_id /
# reasoning_content on every non-tool message turn. InferenceSample.messages
# was typed list[dict[str, str]], so model_validate_json() rejected the very
# first sample of every export and 61/61 production batches failed.
# Payload below is copied verbatim from the live data-collector
# /api/samples/export response (schema 019f2489-199d-7189-87ac-2c7a5b556de2).
# ---------------------------------------------------------------------------

REAL_COLLECTOR_EXPORT_LINE = json.dumps(
    {
        "id": "019f2489-2db2-73ef-b0a5-ed19ac6aef16",
        "tenant_id": None,
        "schema_id": "019f2489-199d-7189-87ac-2c7a5b556de2",
        "job_id": "unknown",
        "inference_id": "99ed480e-3a3f-49bf-a648-b1de2f383ada",
        "messages": [
            {
                "role": "system",
                "content": "You challenge a single design Decision against a set of project principles.",
                "tool_calls": None,
                "tool_call_id": None,
                "reasoning_content": None,
            },
            {
                "role": "user",
                "content": "Decision: store credentials in the repo.",
                "tool_calls": None,
                "tool_call_id": None,
                "reasoning_content": None,
            },
        ],
        "response_content": '{"conflicts": []}',
        "provider": "pgx-local",
        "validation_passed": True,
        "fallback_used": False,
        "fallback_reason": None,
        "latency_ms": 5093,
        "failover_chain_id": "019f2489-1989-73c7-8b83-5c2e37257ffe",
        "used_in_training": False,
        "training_batch_id": None,
        "created_at": "2026-07-02T20:33:17.744288Z",
    }
)


class TestRealCollectorPayloadShape:
    """The export payload data-collector actually sends must validate."""

    async def test_export_samples_accepts_null_message_fields(
        self, collector_settings: Settings
    ) -> None:
        """Null tool_calls/tool_call_id/reasoning_content must not fail validation."""
        mock_transport = httpx.MockTransport(
            lambda request: httpx.Response(200, text=REAL_COLLECTOR_EXPORT_LINE),
        )
        http_client = httpx.AsyncClient(transport=mock_transport, base_url="http://test")
        client = CollectorClient(collector_settings, http_client=http_client)

        samples = await client.export_samples(schema_id="019f2489-199d-7189-87ac-2c7a5b556de2")

        assert len(samples) == 1
        assert len(samples[0].messages) == 2
        assert samples[0].messages[0].role == "system"
        assert samples[0].messages[0].tool_calls is None

    async def test_export_samples_tolerates_new_upstream_message_field(
        self, collector_settings: Settings
    ) -> None:
        """A field data-collector adds later must not fail the whole export."""
        payload = json.loads(REAL_COLLECTOR_EXPORT_LINE)
        payload["messages"][0]["some_future_field"] = {"nested": None}
        mock_transport = httpx.MockTransport(
            lambda request: httpx.Response(200, text=json.dumps(payload)),
        )
        http_client = httpx.AsyncClient(transport=mock_transport, base_url="http://test")
        client = CollectorClient(collector_settings, http_client=http_client)

        samples = await client.export_samples(schema_id="s")
        assert len(samples) == 1

    async def test_formatted_mlx_output_contains_no_null_valued_keys(
        self, collector_settings: Settings
    ) -> None:
        """train.jsonl must not carry tool_calls:null into the MLX training file."""
        from services.batch_service import BatchService
        from services.filter_service import FilterService

        mock_transport = httpx.MockTransport(
            lambda request: httpx.Response(200, text=REAL_COLLECTOR_EXPORT_LINE),
        )
        http_client = httpx.AsyncClient(transport=mock_transport, base_url="http://test")
        client = CollectorClient(collector_settings, http_client=http_client)
        samples = await client.export_samples(schema_id="s")

        service = BatchService(
            collector_client=client,
            filter_service=FilterService(),
            minio_client=MagicMock(),
            settings=collector_settings,
        )
        formatted = service._format_samples(samples, system_prompt=None)

        assert len(formatted) == 1
        for message in formatted[0]["messages"]:
            null_keys = [k for k, v in message.items() if v is None]
            assert null_keys == [], f"null-valued keys leaked into MLX output: {null_keys}"
