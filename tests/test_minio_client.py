"""Tests for the MinIO client in the training data generator service.

Covers all public methods of MinioClient: endpoint URL construction,
bucket creation, JSONL upload, object listing, and connectivity checks.

Uses unittest.mock.AsyncMock and patch to mock aioboto3/S3 calls.
"""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from services.minio_client import MinioClient
from settings import Settings


def _client_error(code: str, message: str, op: str = "HeadBucket") -> ClientError:
    """Build a botocore ClientError with the given error code/message."""
    return ClientError({"Error": {"Code": code, "Message": message}}, op)


@pytest.fixture
def settings() -> Settings:
    """Test settings for MinIO client."""
    return Settings(
        database_url="postgresql+psycopg://postgres:postgres@localhost:5440/tdg_test",
        service_name="training-data-generator-test",
        port=8152,
        data_collector_url="http://localhost:8151",
        model_registry_url="http://localhost:8155",
        minio_endpoint="localhost:9000",
        minio_access_key="test-key",
        minio_secret_key="test-secret",  # pragma: allowlist secret
        minio_bucket="test-training-data",
        minio_use_ssl=False,
        environment="test",
        log_level="DEBUG",
        log_format="console",
    )


@pytest.fixture
def minio_client(settings: Settings) -> MinioClient:
    """MinIO client with test settings."""
    return MinioClient(settings)


def _make_mock_s3_context(mock_s3: AsyncMock) -> AsyncMock:
    """Create a mock async context manager that yields mock_s3.

    Args:
        mock_s3: The mock S3 client to be returned on __aenter__.

    Returns:
        A mock context manager suitable for patching session.client().
    """
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_s3)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    return mock_ctx


# ---------------------------------------------------------------------------
# _get_endpoint_url
# ---------------------------------------------------------------------------


class TestGetEndpointUrl:
    """Tests for _get_endpoint_url and _get_client_kwargs."""

    def test_http_when_ssl_disabled(self, settings: Settings) -> None:
        """Endpoint URL uses http:// when minio_use_ssl is False."""
        settings.minio_use_ssl = False
        client = MinioClient(settings)
        assert client._get_endpoint_url() == "http://localhost:9000"

    def test_https_when_ssl_enabled(self, settings: Settings) -> None:
        """Endpoint URL uses https:// when minio_use_ssl is True."""
        settings.minio_use_ssl = True
        client = MinioClient(settings)
        assert client._get_endpoint_url() == "https://localhost:9000"

    def test_endpoint_already_has_http_prefix(self, settings: Settings) -> None:
        """Endpoint starting with http:// is returned unchanged."""
        settings.minio_endpoint = "http://custom-minio:9000"
        settings.minio_use_ssl = True  # SSL flag should be ignored
        client = MinioClient(settings)
        assert client._get_endpoint_url() == "http://custom-minio:9000"

    def test_endpoint_already_has_https_prefix(self, settings: Settings) -> None:
        """Endpoint starting with https:// is returned unchanged."""
        settings.minio_endpoint = "https://secure-minio:9000"
        settings.minio_use_ssl = False  # SSL flag should be ignored
        client = MinioClient(settings)
        assert client._get_endpoint_url() == "https://secure-minio:9000"

    def test_endpoint_with_path(self, settings: Settings) -> None:
        """Endpoint with a path component is handled correctly."""
        settings.minio_endpoint = "minio.example.com:9000/path"
        settings.minio_use_ssl = False
        client = MinioClient(settings)
        assert client._get_endpoint_url() == "http://minio.example.com:9000/path"

    def test_endpoint_hostname_only(self, settings: Settings) -> None:
        """Endpoint with hostname only (no port) works."""
        settings.minio_endpoint = "minio.cluster.local"
        settings.minio_use_ssl = True
        client = MinioClient(settings)
        assert client._get_endpoint_url() == "https://minio.cluster.local"

    def test_client_kwargs_structure(self, minio_client: MinioClient) -> None:
        """_get_client_kwargs returns correct structure and values."""
        kwargs = minio_client._get_client_kwargs()
        assert kwargs["service_name"] == "s3"
        assert kwargs["endpoint_url"] == "http://localhost:9000"
        assert kwargs["aws_access_key_id"] == "test-key"
        assert kwargs["aws_secret_access_key"] == "test-secret"  # pragma: allowlist secret
        assert kwargs["region_name"] == "us-east-1"

    def test_client_kwargs_with_ssl(self, settings: Settings) -> None:
        """_get_client_kwargs reflects SSL setting in endpoint_url."""
        settings.minio_use_ssl = True
        client = MinioClient(settings)
        kwargs = client._get_client_kwargs()
        assert kwargs["endpoint_url"] == "https://localhost:9000"


# ---------------------------------------------------------------------------
# ensure_bucket
# ---------------------------------------------------------------------------


class TestEnsureBucket:
    """Tests for ensure_bucket."""

    @pytest.mark.asyncio
    async def test_bucket_already_exists(self, minio_client: MinioClient) -> None:
        """No bucket creation when head_bucket succeeds."""
        mock_s3 = AsyncMock()
        mock_s3.head_bucket = AsyncMock(return_value={})
        mock_ctx = _make_mock_s3_context(mock_s3)

        with patch.object(minio_client._session, "client", return_value=mock_ctx):
            await minio_client.ensure_bucket()

        mock_s3.head_bucket.assert_called_once_with(Bucket="test-training-data")
        mock_s3.create_bucket.assert_not_called()

    @pytest.mark.asyncio
    async def test_bucket_does_not_exist_creates_it(self, minio_client: MinioClient) -> None:
        """Bucket is created when head_bucket raises an exception."""
        mock_s3 = AsyncMock()
        mock_s3.head_bucket = AsyncMock(side_effect=_client_error("404", "Not Found"))
        mock_s3.create_bucket = AsyncMock(return_value={})
        mock_ctx = _make_mock_s3_context(mock_s3)

        with patch.object(minio_client._session, "client", return_value=mock_ctx):
            await minio_client.ensure_bucket()

        mock_s3.head_bucket.assert_called_once_with(Bucket="test-training-data")
        mock_s3.create_bucket.assert_called_once_with(Bucket="test-training-data")

    @pytest.mark.asyncio
    async def test_bucket_creation_error_propagates(self, minio_client: MinioClient) -> None:
        """Exception from create_bucket propagates to caller."""
        mock_s3 = AsyncMock()
        mock_s3.head_bucket = AsyncMock(side_effect=_client_error("404", "Not Found"))
        mock_s3.create_bucket = AsyncMock(
            side_effect=_client_error("AccessDenied", "Access Denied", op="CreateBucket")
        )
        mock_ctx = _make_mock_s3_context(mock_s3)

        with (
            patch.object(minio_client._session, "client", return_value=mock_ctx),
            pytest.raises(ClientError, match="Access Denied"),
        ):
            await minio_client.ensure_bucket()

    @pytest.mark.asyncio
    async def test_ensure_bucket_uses_correct_bucket_name(self, settings: Settings) -> None:
        """ensure_bucket uses the bucket name from settings."""
        settings.minio_bucket = "custom-bucket-name"
        client = MinioClient(settings)

        mock_s3 = AsyncMock()
        mock_s3.head_bucket = AsyncMock(return_value={})
        mock_ctx = _make_mock_s3_context(mock_s3)

        with patch.object(client._session, "client", return_value=mock_ctx):
            await client.ensure_bucket()

        mock_s3.head_bucket.assert_called_once_with(Bucket="custom-bucket-name")


# ---------------------------------------------------------------------------
# upload_jsonl
# ---------------------------------------------------------------------------


class TestUploadJsonl:
    """Tests for upload_jsonl."""

    @pytest.mark.asyncio
    async def test_upload_success_returns_full_path(self, minio_client: MinioClient) -> None:
        """Successful upload returns bucket/key string."""
        mock_s3 = AsyncMock()
        mock_s3.upload_fileobj = AsyncMock()
        mock_ctx = _make_mock_s3_context(mock_s3)

        records: list[dict[str, Any]] = [
            {"messages": [{"role": "user", "content": "hello"}]},
            {"messages": [{"role": "user", "content": "world"}]},
        ]

        with patch.object(minio_client._session, "client", return_value=mock_ctx):
            result = await minio_client.upload_jsonl(records, "data/train.jsonl")

        assert result == "test-training-data/data/train.jsonl"
        mock_s3.upload_fileobj.assert_called_once()

        # Verify the call arguments
        call_args = mock_s3.upload_fileobj.call_args
        assert call_args[0][1] == "test-training-data"  # bucket
        assert call_args[0][2] == "data/train.jsonl"  # key
        assert call_args[1]["ExtraArgs"] == {"ContentType": "application/jsonl"}

    @pytest.mark.asyncio
    async def test_upload_jsonl_serializes_records_correctly(
        self, minio_client: MinioClient
    ) -> None:
        """Records are serialized as newline-delimited JSON."""
        mock_s3 = AsyncMock()
        mock_s3.upload_fileobj = AsyncMock()
        mock_ctx = _make_mock_s3_context(mock_s3)

        records: list[dict[str, Any]] = [
            {"key": "value1"},
            {"key": "value2"},
            {"key": "value3"},
        ]

        with patch.object(minio_client._session, "client", return_value=mock_ctx):
            await minio_client.upload_jsonl(records, "test.jsonl")

        # Extract the BytesIO argument and read its content
        call_args = mock_s3.upload_fileobj.call_args
        fileobj = call_args[0][0]
        content = fileobj.read().decode("utf-8")

        lines = content.strip().split("\n")
        assert len(lines) == 3
        assert lines[0] == '{"key": "value1"}'
        assert lines[1] == '{"key": "value2"}'
        assert lines[2] == '{"key": "value3"}'

    @pytest.mark.asyncio
    async def test_upload_jsonl_empty_records(self, minio_client: MinioClient) -> None:
        """Empty record list uploads an empty body."""
        mock_s3 = AsyncMock()
        mock_s3.upload_fileobj = AsyncMock()
        mock_ctx = _make_mock_s3_context(mock_s3)

        with patch.object(minio_client._session, "client", return_value=mock_ctx):
            result = await minio_client.upload_jsonl([], "empty.jsonl")

        assert result == "test-training-data/empty.jsonl"
        mock_s3.upload_fileobj.assert_called_once()

        # Verify the uploaded data is empty
        call_args = mock_s3.upload_fileobj.call_args
        fileobj = call_args[0][0]
        content = fileobj.read()
        assert content == b""

    @pytest.mark.asyncio
    async def test_upload_jsonl_single_record(self, minio_client: MinioClient) -> None:
        """Single record is uploaded with trailing newline."""
        mock_s3 = AsyncMock()
        mock_s3.upload_fileobj = AsyncMock()
        mock_ctx = _make_mock_s3_context(mock_s3)

        records: list[dict[str, Any]] = [{"single": "record"}]

        with patch.object(minio_client._session, "client", return_value=mock_ctx):
            await minio_client.upload_jsonl(records, "single.jsonl")

        call_args = mock_s3.upload_fileobj.call_args
        fileobj = call_args[0][0]
        content = fileobj.read().decode("utf-8")
        assert content == '{"single": "record"}\n'

    @pytest.mark.asyncio
    async def test_upload_jsonl_unicode_records(self, minio_client: MinioClient) -> None:
        """Unicode content is correctly serialized with ensure_ascii=False."""
        mock_s3 = AsyncMock()
        mock_s3.upload_fileobj = AsyncMock()
        mock_ctx = _make_mock_s3_context(mock_s3)

        records: list[dict[str, Any]] = [{"text": "Hello, world!"}]

        with patch.object(minio_client._session, "client", return_value=mock_ctx):
            await minio_client.upload_jsonl(records, "unicode.jsonl")

        call_args = mock_s3.upload_fileobj.call_args
        fileobj = call_args[0][0]
        content = fileobj.read().decode("utf-8")
        # ensure_ascii=False preserves unicode characters directly
        assert "Hello, world!" in content

    @pytest.mark.asyncio
    async def test_upload_jsonl_failure_propagates(self, minio_client: MinioClient) -> None:
        """S3 upload failure propagates the exception."""
        mock_s3 = AsyncMock()
        mock_s3.upload_fileobj = AsyncMock(side_effect=Exception("Upload failed: connection reset"))
        mock_ctx = _make_mock_s3_context(mock_s3)

        records: list[dict[str, Any]] = [{"data": "test"}]

        with (
            patch.object(minio_client._session, "client", return_value=mock_ctx),
            pytest.raises(Exception, match="Upload failed: connection reset"),
        ):
            await minio_client.upload_jsonl(records, "fail.jsonl")

    @pytest.mark.asyncio
    async def test_upload_jsonl_nested_records(self, minio_client: MinioClient) -> None:
        """Nested/complex records are serialized correctly."""
        mock_s3 = AsyncMock()
        mock_s3.upload_fileobj = AsyncMock()
        mock_ctx = _make_mock_s3_context(mock_s3)

        records: list[dict[str, Any]] = [
            {
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4"},
                ],
                "metadata": {"score": 0.95, "tags": ["math", "simple"]},
            }
        ]

        with patch.object(minio_client._session, "client", return_value=mock_ctx):
            result = await minio_client.upload_jsonl(records, "nested.jsonl")

        assert result == "test-training-data/nested.jsonl"

        import json

        call_args = mock_s3.upload_fileobj.call_args
        fileobj = call_args[0][0]
        content = fileobj.read().decode("utf-8")
        parsed = json.loads(content.strip())
        assert parsed["messages"][0]["role"] == "system"
        assert parsed["metadata"]["score"] == 0.95

    @pytest.mark.asyncio
    async def test_upload_jsonl_uses_correct_key(self, minio_client: MinioClient) -> None:
        """The S3 key argument is passed through correctly."""
        mock_s3 = AsyncMock()
        mock_s3.upload_fileobj = AsyncMock()
        mock_ctx = _make_mock_s3_context(mock_s3)

        key = "schemas/my_schema/2026-02-26/train.jsonl"
        with patch.object(minio_client._session, "client", return_value=mock_ctx):
            await minio_client.upload_jsonl([{"x": 1}], key)

        call_args = mock_s3.upload_fileobj.call_args
        assert call_args[0][2] == key


# ---------------------------------------------------------------------------
# list_objects
# ---------------------------------------------------------------------------


class TestListObjects:
    """Tests for list_objects."""

    @pytest.mark.asyncio
    async def test_list_objects_single_page(self, minio_client: MinioClient) -> None:
        """Returns objects from a single page response."""
        now = datetime(2026, 2, 26, 12, 0, 0, tzinfo=UTC)
        mock_s3 = AsyncMock()
        mock_paginator = MagicMock()

        async def mock_paginate(**kwargs: Any) -> Any:
            yield {
                "Contents": [
                    {"Key": "data/train.jsonl", "Size": 1024, "LastModified": now},
                    {"Key": "data/valid.jsonl", "Size": 512, "LastModified": now},
                ]
            }

        mock_paginator.paginate = mock_paginate
        mock_s3.get_paginator = MagicMock(return_value=mock_paginator)
        mock_ctx = _make_mock_s3_context(mock_s3)

        with patch.object(minio_client._session, "client", return_value=mock_ctx):
            result = await minio_client.list_objects("data/")

        assert len(result) == 2
        assert result[0]["key"] == "data/train.jsonl"
        assert result[0]["size"] == 1024
        assert result[0]["last_modified"] == now.isoformat()
        assert result[1]["key"] == "data/valid.jsonl"
        assert result[1]["size"] == 512

    @pytest.mark.asyncio
    async def test_list_objects_multiple_pages(self, minio_client: MinioClient) -> None:
        """Returns objects aggregated from multiple pages."""
        now = datetime(2026, 2, 26, 12, 0, 0, tzinfo=UTC)
        mock_s3 = AsyncMock()
        mock_paginator = MagicMock()

        async def mock_paginate(**kwargs: Any) -> Any:
            yield {
                "Contents": [
                    {"Key": "data/file1.jsonl", "Size": 100, "LastModified": now},
                ]
            }
            yield {
                "Contents": [
                    {"Key": "data/file2.jsonl", "Size": 200, "LastModified": now},
                ]
            }

        mock_paginator.paginate = mock_paginate
        mock_s3.get_paginator = MagicMock(return_value=mock_paginator)
        mock_ctx = _make_mock_s3_context(mock_s3)

        with patch.object(minio_client._session, "client", return_value=mock_ctx):
            result = await minio_client.list_objects("data/")

        assert len(result) == 2
        assert result[0]["key"] == "data/file1.jsonl"
        assert result[1]["key"] == "data/file2.jsonl"

    @pytest.mark.asyncio
    async def test_list_objects_empty_prefix_no_contents(self, minio_client: MinioClient) -> None:
        """Returns empty list when no objects match the prefix."""
        mock_s3 = AsyncMock()
        mock_paginator = MagicMock()

        async def mock_paginate(**kwargs: Any) -> Any:
            yield {}  # No "Contents" key

        mock_paginator.paginate = mock_paginate
        mock_s3.get_paginator = MagicMock(return_value=mock_paginator)
        mock_ctx = _make_mock_s3_context(mock_s3)

        with patch.object(minio_client._session, "client", return_value=mock_ctx):
            result = await minio_client.list_objects("nonexistent/")

        assert result == []

    @pytest.mark.asyncio
    async def test_list_objects_empty_contents_list(self, minio_client: MinioClient) -> None:
        """Returns empty list when Contents is an empty list."""
        mock_s3 = AsyncMock()
        mock_paginator = MagicMock()

        async def mock_paginate(**kwargs: Any) -> Any:
            yield {"Contents": []}

        mock_paginator.paginate = mock_paginate
        mock_s3.get_paginator = MagicMock(return_value=mock_paginator)
        mock_ctx = _make_mock_s3_context(mock_s3)

        with patch.object(minio_client._session, "client", return_value=mock_ctx):
            result = await minio_client.list_objects("empty/")

        assert result == []

    @pytest.mark.asyncio
    async def test_list_objects_handles_missing_last_modified(
        self, minio_client: MinioClient
    ) -> None:
        """Objects without LastModified get None for last_modified."""
        mock_s3 = AsyncMock()
        mock_paginator = MagicMock()

        async def mock_paginate(**kwargs: Any) -> Any:
            yield {
                "Contents": [
                    {"Key": "data/file.jsonl", "Size": 256},  # No LastModified
                ]
            }

        mock_paginator.paginate = mock_paginate
        mock_s3.get_paginator = MagicMock(return_value=mock_paginator)
        mock_ctx = _make_mock_s3_context(mock_s3)

        with patch.object(minio_client._session, "client", return_value=mock_ctx):
            result = await minio_client.list_objects("data/")

        assert len(result) == 1
        assert result[0]["key"] == "data/file.jsonl"
        assert result[0]["size"] == 256
        assert result[0]["last_modified"] is None

    @pytest.mark.asyncio
    async def test_list_objects_error_propagates(self, minio_client: MinioClient) -> None:
        """S3 errors in list_objects propagate to the caller."""
        mock_s3 = AsyncMock()
        mock_paginator = MagicMock()

        async def mock_paginate(**kwargs: Any) -> Any:
            raise Exception("Access Denied")
            yield  # Make this an async generator  # noqa: RET503

        mock_paginator.paginate = mock_paginate
        mock_s3.get_paginator = MagicMock(return_value=mock_paginator)
        mock_ctx = _make_mock_s3_context(mock_s3)

        with (
            patch.object(minio_client._session, "client", return_value=mock_ctx),
            pytest.raises(Exception, match="Access Denied"),
        ):
            await minio_client.list_objects("data/")

    @pytest.mark.asyncio
    async def test_list_objects_uses_correct_bucket_and_prefix(
        self, minio_client: MinioClient
    ) -> None:
        """Paginator is called with the correct bucket and prefix."""
        mock_s3 = AsyncMock()
        mock_paginator = MagicMock()

        captured_kwargs: dict[str, Any] = {}

        async def mock_paginate(**kwargs: Any) -> Any:
            captured_kwargs.update(kwargs)
            yield {"Contents": []}

        mock_paginator.paginate = mock_paginate
        mock_s3.get_paginator = MagicMock(return_value=mock_paginator)
        mock_ctx = _make_mock_s3_context(mock_s3)

        with patch.object(minio_client._session, "client", return_value=mock_ctx):
            await minio_client.list_objects("schemas/v1/")

        mock_s3.get_paginator.assert_called_once_with("list_objects_v2")
        assert captured_kwargs["Bucket"] == "test-training-data"
        assert captured_kwargs["Prefix"] == "schemas/v1/"

    @pytest.mark.asyncio
    async def test_list_objects_empty_prefix(self, minio_client: MinioClient) -> None:
        """Empty prefix string lists all objects."""
        mock_s3 = AsyncMock()
        mock_paginator = MagicMock()

        captured_kwargs: dict[str, Any] = {}

        async def mock_paginate(**kwargs: Any) -> Any:
            captured_kwargs.update(kwargs)
            yield {"Contents": []}

        mock_paginator.paginate = mock_paginate
        mock_s3.get_paginator = MagicMock(return_value=mock_paginator)
        mock_ctx = _make_mock_s3_context(mock_s3)

        with patch.object(minio_client._session, "client", return_value=mock_ctx):
            await minio_client.list_objects("")

        assert captured_kwargs["Prefix"] == ""


# ---------------------------------------------------------------------------
# check_connectivity
# ---------------------------------------------------------------------------


class TestCheckConnectivity:
    """Tests for check_connectivity."""

    @pytest.mark.asyncio
    async def test_connectivity_success(self, minio_client: MinioClient) -> None:
        """Returns True when list_buckets succeeds."""
        mock_s3 = AsyncMock()
        mock_s3.list_buckets = AsyncMock(return_value={"Buckets": []})
        mock_ctx = _make_mock_s3_context(mock_s3)

        with patch.object(minio_client._session, "client", return_value=mock_ctx):
            result = await minio_client.check_connectivity()

        assert result is True
        mock_s3.list_buckets.assert_called_once()

    @pytest.mark.asyncio
    async def test_connectivity_connection_failure(self, minio_client: MinioClient) -> None:
        """Returns False when connection to MinIO fails."""
        mock_s3 = AsyncMock()
        mock_s3.list_buckets = AsyncMock(side_effect=ConnectionError("Connection refused"))
        mock_ctx = _make_mock_s3_context(mock_s3)

        with patch.object(minio_client._session, "client", return_value=mock_ctx):
            result = await minio_client.check_connectivity()

        assert result is False

    @pytest.mark.asyncio
    async def test_connectivity_auth_failure(self, minio_client: MinioClient) -> None:
        """Returns False when authentication fails."""
        mock_s3 = AsyncMock()
        mock_s3.list_buckets = AsyncMock(
            side_effect=_client_error(
                "InvalidAccessKeyId",
                "The access key does not exist",
                op="ListBuckets",
            )
        )
        mock_ctx = _make_mock_s3_context(mock_s3)

        with patch.object(minio_client._session, "client", return_value=mock_ctx):
            result = await minio_client.check_connectivity()

        assert result is False

    @pytest.mark.asyncio
    async def test_connectivity_timeout(self, minio_client: MinioClient) -> None:
        """Returns False when connection times out."""
        mock_s3 = AsyncMock()
        mock_s3.list_buckets = AsyncMock(side_effect=TimeoutError("Connection timed out"))
        mock_ctx = _make_mock_s3_context(mock_s3)

        with patch.object(minio_client._session, "client", return_value=mock_ctx):
            result = await minio_client.check_connectivity()

        assert result is False

    @pytest.mark.asyncio
    async def test_connectivity_endpoint_failure(self, minio_client: MinioClient) -> None:
        """Returns False when the endpoint is unreachable (BotoCoreError)."""
        mock_s3 = AsyncMock()
        mock_s3.list_buckets = AsyncMock(
            side_effect=EndpointConnectionError(endpoint_url="http://minio.invalid")
        )
        mock_ctx = _make_mock_s3_context(mock_s3)

        with patch.object(minio_client._session, "client", return_value=mock_ctx):
            result = await minio_client.check_connectivity()

        assert result is False

    @pytest.mark.asyncio
    async def test_connectivity_context_manager_failure(self, minio_client: MinioClient) -> None:
        """Returns False when the S3 client context manager itself fails."""
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(side_effect=OSError("Cannot create client"))
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch.object(minio_client._session, "client", return_value=mock_ctx):
            result = await minio_client.check_connectivity()

        assert result is False


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestMinioClientInit:
    """Tests for MinioClient initialization."""

    def test_init_stores_settings(self, settings: Settings) -> None:
        """Client stores settings values correctly."""
        client = MinioClient(settings)
        assert client._endpoint == "localhost:9000"
        assert client._access_key == "test-key"
        assert client._secret_key == "test-secret"  # pragma: allowlist secret
        assert client._bucket == "test-training-data"
        assert client._use_ssl is False
        assert client._session is not None

    def test_init_with_ssl(self, settings: Settings) -> None:
        """Client stores SSL setting correctly when enabled."""
        settings.minio_use_ssl = True
        client = MinioClient(settings)
        assert client._use_ssl is True

    def test_init_creates_aioboto3_session(self, settings: Settings) -> None:
        """Client creates an aioboto3 Session on init."""
        client = MinioClient(settings)
        import aioboto3

        assert isinstance(client._session, aioboto3.Session)
