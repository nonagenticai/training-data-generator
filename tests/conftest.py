"""Shared test fixtures for the training data generator service.

Provides mock clients, sample data factories, and a configured test
application for integration testing with httpx.AsyncClient.

DB-backed tests require PostgreSQL via TEST_DATABASE_URL (or DATABASE_URL; CI exports
DATABASE_URL). There is NO host:5432 default -- it silently hits whatever runs on that
port on a dev box. If neither is set, an ephemeral docker postgres:18 is started for the
session (tests/ephemeral_postgres.py); without docker the session stops (fail closed).
"""

import json
import os

# Ensure test runs don't trigger production validation when modules import
# settings at import time (logging_config calls get_settings()).
os.environ.setdefault("TRAINING_DATA_GEN_ENVIRONMENT", "test")
# Telemetry tests assume an endpoint is configured; without it telemetry is
# disabled and the tracer-provider/exporter mocks aren't exercised.
os.environ.setdefault("TRAINING_DATA_GEN_OTLP_TRACES_ENDPOINT", "http://test-otlp:4318/v1/traces")

from collections.abc import AsyncIterator  # noqa: E402
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlparse
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from db import dispose_db, init_db
from models.domain import InferenceSample, Message
from services.batch_service import BatchService
from services.collector_client import CollectorClient
from services.filter_service import FilterService
from services.minio_client import MinioClient
from services.registry_client import RegistryClient
from services.trigger_service import TriggerService
from settings import Settings
from tests.ephemeral_postgres import (
    resolve_test_database_url,
    start_session_database,
    stop_session_database,
)

# PostgreSQL URL for tests -- the ``test_database_url`` fixture. Resolution order:
#   1. TEST_DATABASE_URL (explicit local override)
#   2. DATABASE_URL (the python-service archetype exports this in CI --
#      an ephemeral per-job docker postgres on 127.0.0.1:<mapped>)
#   3. an ephemeral docker postgres:18 for this session (never a host:5432
#      default); without docker the session stops with a one-line message.
# The teardown fixtures call Base.metadata.drop_all, so the guard below
# refuses any URL that isn't obviously a CI/test DB (2026-04-18 secrets
# incident -- see auto-memory/secrets-db-wipe-2026-04-18.md).


# For tests that only need a well-formed DSN string (init_db mocked, nothing
# connects). ``.invalid`` never resolves, so an accidental connect fails fast
# instead of reaching whatever runs on a dev box.
PLACEHOLDER_DATABASE_URL = (
    "postgresql+psycopg://unused:unused@db.invalid/unused_test"  # pragma: allowlist secret
)


def _assert_test_db_url_is_safe(url: str) -> None:
    """Refuse to run tests against any DB that could be production."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    dbname = (parsed.path or "").lstrip("/")
    if host in ("localhost", "127.0.0.1", "::1"):
        return
    if dbname.endswith("_ci") or dbname.endswith("_test"):
        return
    pytest.exit(
        f"Refusing to run tests against TEST_DATABASE_URL={url!r}: dbname "
        f"{dbname!r} on host {host!r} is not a known CI/test database. Tests "
        f"call Base.metadata.drop_all and would destroy live data. Set "
        f"TEST_DATABASE_URL to a *_ci/*_test database or localhost Postgres.",
        returncode=2,
    )


def pytest_configure(config: pytest.Config) -> None:
    """Start the ONE session database in the xdist controller (no-op in workers), before
    workers spawn, so they inherit the exported DSN instead of starting their own."""
    start_session_database(
        env_vars=("TEST_DATABASE_URL", "DATABASE_URL"),
        export_var="TEST_DATABASE_URL",
        dbname="training_data_generator_test",
        config=config,
    )


def pytest_unconfigure(config: pytest.Config) -> None:
    """Controller only: remove the session container by its own ID."""
    stop_session_database(config)


@pytest.fixture(scope="session")
def test_database_url(request: pytest.FixtureRequest) -> str:
    """The DSN DB-backed tests use (resolved once per session, then safety-checked)."""
    url = resolve_test_database_url(
        env_vars=("TEST_DATABASE_URL", "DATABASE_URL"),
        export_var="TEST_DATABASE_URL",
        dbname="training_data_generator_test",
        request=request,
    )
    _assert_test_db_url_is_safe(url)
    return url


@pytest.fixture
def test_settings(test_database_url: str) -> Settings:
    """Create test settings with sensible defaults."""
    return Settings(
        service_name="training-data-generator-test",
        port=8152,
        database_url=test_database_url,
        data_collector_url="http://localhost:8151",
        model_registry_url="http://localhost:8155",
        model_trainer_url="http://localhost:8153",
        global_base_model="Qwen/Qwen3-4B-AWQ",
        minio_endpoint="localhost:9000",
        minio_access_key="testkey",
        minio_secret_key="testsecret",  # pragma: allowlist secret
        minio_bucket="test-training-data",
        minio_use_ssl=False,
        initial_sample_threshold=200,
        subsequent_sample_threshold=100,
        quality_threshold=0.80,
        retrain_interval_days=14,
        train_split=0.9,
        environment="test",
        log_level="DEBUG",
        log_format="console",
    )


@pytest.fixture
def filter_service() -> FilterService:
    """Create a FilterService instance."""
    return FilterService()


@pytest.fixture
def mock_collector_client() -> AsyncMock:
    """Create a mock CollectorClient."""
    client = AsyncMock(spec=CollectorClient)
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_minio_client() -> AsyncMock:
    """Create a mock MinioClient."""
    client = AsyncMock(spec=MinioClient)
    client.ensure_bucket = AsyncMock()
    client.upload_jsonl = AsyncMock(return_value="test-bucket/path")
    client.check_connectivity = AsyncMock(return_value=True)
    return client


@pytest.fixture
def mock_registry_client() -> AsyncMock:
    """Create a mock RegistryClient."""
    client = AsyncMock(spec=RegistryClient)
    client.close = AsyncMock()
    client.get_routing_rule = AsyncMock(return_value={})
    return client


@pytest_asyncio.fixture
async def _init_test_db(test_database_url: str) -> AsyncIterator[None]:
    """Initialise the test database for each test and tear it down after."""
    await init_db(test_database_url)
    # Drop and recreate to ensure clean state (PostgreSQL persists between runs)
    from db import _engine
    from models.db_models import Base

    if _engine is None:
        raise RuntimeError("init_db() did not create an engine")

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
    await dispose_db()


@pytest_asyncio.fixture
async def batch_service(
    _init_test_db: None,
    test_settings: Settings,
    mock_collector_client: AsyncMock,
    mock_minio_client: AsyncMock,
    mock_registry_client: AsyncMock,
    filter_service: FilterService,
) -> BatchService:
    """Create a BatchService with mocked dependencies and a test database."""
    return BatchService(
        collector_client=mock_collector_client,
        filter_service=filter_service,
        minio_client=mock_minio_client,
        registry_client=mock_registry_client,
        settings=test_settings,
    )


@pytest.fixture
def mock_http_client() -> AsyncMock:
    """Create a mock httpx.AsyncClient for trigger service tests."""
    mock_http: AsyncMock = AsyncMock()
    mock_http.get = AsyncMock()
    # Default post response with sync raise_for_status (matches httpx.Response)
    default_post_response = MagicMock()
    default_post_response.raise_for_status = MagicMock()
    mock_http.post = AsyncMock(return_value=default_post_response)
    mock_http.aclose = AsyncMock()
    return mock_http


@pytest_asyncio.fixture
async def trigger_service(
    test_settings: Settings,
    mock_collector_client: AsyncMock,
    batch_service: BatchService,
    mock_http_client: AsyncMock,
) -> TriggerService:
    """Create a TriggerService with mocked dependencies."""
    return TriggerService(
        settings=test_settings,
        collector_client=mock_collector_client,
        batch_service=batch_service,
        http_client=mock_http_client,
    )


def make_sample(
    schema_id: str = "test_schema",
    messages: list[Message] | None = None,
    response_content: str | None = None,
    provider: str = "Qwen3-4B-AWQ",
    fallback_used: bool = False,
    quality_score: float = 0.95,
) -> InferenceSample:
    """Factory for creating test InferenceSample instances.

    Args:
        schema_id: Schema identifier.
        messages: Conversation messages. Defaults to a simple system+user pair.
        response_content: Assistant response. Defaults to valid JSON.
        provider: Inference provider name.
        fallback_used: Whether fallback was used (teacher model pattern).
        quality_score: Quality score for the sample.

    Returns:
        An InferenceSample instance.
    """
    if messages is None:
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content=f"Test query {uuid4().hex[:8]}"),
        ]
    if response_content is None:
        response_content = json.dumps({"result": "test response", "confidence": 0.95})

    return InferenceSample(
        id=uuid4(),
        schema_id=schema_id,
        messages=messages,
        response_content=response_content,
        provider=provider,
        fallback_used=fallback_used,
        validated_at=datetime.now(UTC),
        quality_score=quality_score,
    )


def make_samples(
    count: int = 10,
    schema_id: str = "test_schema",
    provider: str = "Qwen3-4B-AWQ",
    fallback_count: int = 0,
) -> list[InferenceSample]:
    """Create a list of unique test samples.

    Args:
        count: Number of samples to generate.
        schema_id: Schema identifier for all samples.
        provider: Default provider name.
        fallback_count: Number of samples to mark as fallback (teacher pattern).

    Returns:
        List of InferenceSample instances with unique messages.
    """
    samples = []
    for i in range(count):
        is_fallback = i < fallback_count
        sample = make_sample(
            schema_id=schema_id,
            messages=[
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content=f"Unique test query number {i}"),
            ],
            response_content=json.dumps(
                {
                    "result": f"response_{i}",
                    "data": {"index": i, "value": f"value_{i}"},
                }
            ),
            provider="deepseek" if is_fallback else provider,
            fallback_used=is_fallback,
        )
        samples.append(sample)
    return samples


@pytest_asyncio.fixture
async def test_app(
    test_settings: Settings,
    mock_collector_client: AsyncMock,
    mock_minio_client: AsyncMock,
    batch_service: BatchService,
    trigger_service: TriggerService,
    filter_service: FilterService,
) -> AsyncIterator[AsyncClient]:
    """Create a test FastAPI app with mocked dependencies.

    Patches the lifespan to inject test mocks instead of real clients.
    The in-memory database is already initialised via the batch_service fixture.
    """
    with patch("main.get_settings", return_value=test_settings):
        from main import app

        # Inject test dependencies directly onto app state
        app.state.collector_client = mock_collector_client
        app.state.minio_client = mock_minio_client
        app.state.filter_service = filter_service
        app.state.batch_service = batch_service
        app.state.trigger_service = trigger_service

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
