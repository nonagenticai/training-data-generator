"""Tests for batch generation service and API.

Covers the full batch generation pipeline including sample fetching,
filtering, MLX formatting, train/valid split, and MinIO upload.
Also covers CollectorClient HTTP contract correctness.
"""

import json
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from httpx import AsyncClient

from models.domain import BatchGenerateRequest, BatchStatus, InferenceSample, Message
from services.batch_service import BatchService, _content_hash
from services.collector_client import CollectorClient
from services.filter_service import FilterService
from settings import Settings
from tests.conftest import make_sample, make_samples


def _get_all_uploaded_records(mock_minio_client: AsyncMock) -> list[dict]:
    """Extract all records passed to upload_jsonl across both train and valid calls.

    Uses named keyword access (.args) instead of brittle [0][0][0] indexing.
    Returns the concatenated list of all records from all upload_jsonl calls.
    """
    all_records: list[dict] = []
    for c in mock_minio_client.upload_jsonl.call_args_list:
        records = c.args[0]  # first positional arg = records list
        all_records.extend(records)
    return all_records


def _get_upload_keys(mock_minio_client: AsyncMock) -> list[str]:
    """Extract all S3 keys passed to upload_jsonl calls.

    Returns list of key strings in call order.
    """
    return [c.args[1] for c in mock_minio_client.upload_jsonl.call_args_list]


class TestBatchService:
    """Test the BatchService orchestration logic."""

    @pytest.mark.asyncio
    async def test_successful_batch_generation(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
        mock_minio_client: AsyncMock,
    ) -> None:
        """A successful batch generation should produce a completed batch with correct metadata."""
        samples = make_samples(count=100, fallback_count=10)
        mock_collector_client.export_samples.return_value = samples

        request = BatchGenerateRequest(
            schema_id="test_schema",
            min_samples=50,
            train_split=0.9,
        )

        batch = await batch_service.generate_batch(request)

        assert batch.status == BatchStatus.COMPLETED
        assert batch.schema_id == "test_schema"
        assert batch.total_samples == 100
        assert batch.train_samples + batch.valid_samples == 100
        assert batch.train_samples == 90
        assert batch.valid_samples == 10
        assert batch.teacher_samples == 10
        assert batch.minio_path is not None
        assert batch.train_file is not None
        assert batch.valid_file is not None
        assert batch.completed_at is not None
        assert batch.error_message is None

        # Verify MinIO was called
        assert mock_minio_client.ensure_bucket.called
        assert mock_minio_client.upload_jsonl.call_count == 2

        # Verify samples were marked as used with correct args
        mock_collector_client.mark_used.assert_called_once()
        mark_args = mock_collector_client.mark_used.call_args
        marked_ids = mark_args.args[0]
        marked_batch_id = mark_args.args[1]
        assert len(marked_ids) == 100
        assert marked_batch_id == str(batch.id)

    @pytest.mark.asyncio
    async def test_no_samples_available(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
    ) -> None:
        """Batch should fail when no samples are available."""
        mock_collector_client.export_samples.return_value = []

        request = BatchGenerateRequest(schema_id="empty_schema", min_samples=50)
        batch = await batch_service.generate_batch(request)

        assert batch.status == BatchStatus.FAILED
        assert "0 samples passed filtering" in (batch.error_message or "")
        assert "need at least" in (batch.error_message or "")
        # Failed batch should still be persisted
        retrieved = await batch_service.get_batch(str(batch.id))
        assert retrieved is not None
        assert retrieved.status == BatchStatus.FAILED

    @pytest.mark.asyncio
    async def test_insufficient_samples_after_filtering(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
    ) -> None:
        """Batch should fail when too few samples pass filtering."""
        # Create samples that will mostly be filtered (too short)
        short_samples = [make_sample(response_content="hi") for _ in range(20)]
        # Add a few valid ones but below min threshold
        valid_samples = make_samples(count=5)
        mock_collector_client.export_samples.return_value = short_samples + valid_samples

        request = BatchGenerateRequest(schema_id="test_schema", min_samples=50)
        batch = await batch_service.generate_batch(request)

        assert batch.status == BatchStatus.FAILED
        assert "Only 5 samples passed filtering" in (batch.error_message or "")
        assert "need at least" in (batch.error_message or "")
        # Should still record total_samples and filtered counts
        assert batch.total_samples == 25
        assert batch.filtered_samples >= 20

    @pytest.mark.asyncio
    async def test_custom_train_split(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
    ) -> None:
        """Custom train_split should be respected."""
        samples = make_samples(count=100)
        mock_collector_client.export_samples.return_value = samples

        request = BatchGenerateRequest(
            schema_id="test_schema",
            min_samples=10,
            train_split=0.8,
        )
        batch = await batch_service.generate_batch(request)

        assert batch.status == BatchStatus.COMPLETED
        assert batch.train_samples == 80
        assert batch.valid_samples == 20

    @pytest.mark.asyncio
    async def test_minio_upload_failure(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
        mock_minio_client: AsyncMock,
    ) -> None:
        """Batch should fail if MinIO upload raises an exception."""
        samples = make_samples(count=100)
        mock_collector_client.export_samples.return_value = samples
        mock_minio_client.upload_jsonl.side_effect = ConnectionError("MinIO unreachable")

        request = BatchGenerateRequest(schema_id="test_schema", min_samples=50)
        batch = await batch_service.generate_batch(request)

        assert batch.status == BatchStatus.FAILED
        assert "MinIO unreachable" in (batch.error_message or "")

    @pytest.mark.asyncio
    async def test_source_provider_counts(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
    ) -> None:
        """Source provider counts should accurately reflect sample provenance."""
        samples = [
            make_sample(provider="Qwen3-4B-AWQ"),
            make_sample(provider="Qwen3-4B-AWQ"),
            make_sample(provider="deepseek"),
            make_sample(provider="claude"),
        ]
        mock_collector_client.export_samples.return_value = samples

        request = BatchGenerateRequest(schema_id="test_schema", min_samples=1)
        batch = await batch_service.generate_batch(request)

        assert batch.status == BatchStatus.COMPLETED
        assert batch.source_providers["Qwen3-4B-AWQ"] == 2
        assert batch.source_providers["deepseek"] == 1
        assert batch.source_providers["claude"] == 1

    @pytest.mark.asyncio
    async def test_filter_reasons_tracked(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
    ) -> None:
        """Filter reasons should be accurately tracked in the batch."""
        samples = make_samples(count=10)
        # Add some that will be filtered
        samples.extend(
            [
                make_sample(response_content="hi"),  # too_short
                make_sample(response_content="not json"),  # invalid_json
            ]
        )
        mock_collector_client.export_samples.return_value = samples

        request = BatchGenerateRequest(schema_id="test_schema", min_samples=1)
        batch = await batch_service.generate_batch(request)

        assert batch.status == BatchStatus.COMPLETED
        assert batch.filtered_samples >= 2
        assert batch.total_samples == 12
        # Verify specific filter reasons are recorded
        assert "too_short" in batch.filter_reasons
        assert batch.filter_reasons["too_short"] >= 1

    @pytest.mark.asyncio
    async def test_min_samples_uses_request_value_not_hardcoded(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
    ) -> None:
        """min_samples should come from the request, not be hardcoded to 50 or any other value.

        This was a known bug (min_samples was hardcoded to 50). Verify it uses the
        request's min_samples correctly by testing boundary conditions.
        """
        # 8 valid samples
        samples = make_samples(count=8)
        mock_collector_client.export_samples.return_value = samples

        # With min_samples=10, 8 samples should fail (need ceil(10/0.9)=12 total)
        request_fail = BatchGenerateRequest(schema_id="test_schema", min_samples=10)
        batch_fail = await batch_service.generate_batch(request_fail)
        assert batch_fail.status == BatchStatus.FAILED
        assert "need at least" in (batch_fail.error_message or "")

        # With min_samples=5, same 8 samples should succeed
        mock_collector_client.export_samples.return_value = make_samples(count=8)
        request_pass = BatchGenerateRequest(schema_id="test_schema", min_samples=5)
        batch_pass = await batch_service.generate_batch(request_pass)
        assert batch_pass.status == BatchStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_uploaded_records_are_valid_mlx_format(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
        mock_minio_client: AsyncMock,
    ) -> None:
        """Every record uploaded to MinIO should be a valid MLX chat dict with messages key."""
        samples = make_samples(count=20)
        mock_collector_client.export_samples.return_value = samples

        request = BatchGenerateRequest(schema_id="test_schema", min_samples=1, train_split=0.9)
        await batch_service.generate_batch(request)

        all_records = _get_all_uploaded_records(mock_minio_client)
        assert len(all_records) == 20
        for record in all_records:
            assert isinstance(record, dict)
            assert "messages" in record
            messages = record["messages"]
            assert isinstance(messages, list)
            assert len(messages) >= 2  # At least user + assistant
            # Last message should always be assistant
            assert messages[-1]["role"] == "assistant"
            assert len(messages[-1]["content"]) > 0

    @pytest.mark.asyncio
    async def test_upload_keys_contain_schema_and_batch_id(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
        mock_minio_client: AsyncMock,
    ) -> None:
        """Upload keys should follow the pattern training-data/{schema}/batch-{uuid}/."""
        samples = make_samples(count=20)
        mock_collector_client.export_samples.return_value = samples

        request = BatchGenerateRequest(schema_id="my_schema", min_samples=1)
        batch = await batch_service.generate_batch(request)

        keys = _get_upload_keys(mock_minio_client)
        assert len(keys) == 2
        assert keys[0].startswith("training-data/my_schema/batch-")
        assert keys[0].endswith("/train.jsonl")
        assert keys[1].startswith("training-data/my_schema/batch-")
        assert keys[1].endswith("/valid.jsonl")
        # Both keys should reference the same batch
        assert str(batch.id) in keys[0]
        assert str(batch.id) in keys[1]


class TestMLXFormatting:
    """Test MLX chat JSONL format conversion."""

    @pytest.mark.asyncio
    async def test_mlx_format_structure(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
        mock_minio_client: AsyncMock,
    ) -> None:
        """Formatted samples should have the MLX chat messages structure."""
        sample = make_sample(
            messages=[
                Message(role="system", content="You are a helper."),
                Message(role="user", content="What is 2+2?"),
            ],
            response_content=json.dumps({"answer": 4}),
        )
        sample2 = make_sample()
        mock_collector_client.export_samples.return_value = [sample, sample2]

        request = BatchGenerateRequest(
            schema_id="test_schema",
            min_samples=1,
            train_split=0.9,
        )
        await batch_service.generate_batch(request)

        # Extract all uploaded records using helper (avoids brittle [0][0][0] indexing)
        all_records = _get_all_uploaded_records(mock_minio_client)
        assert len(all_records) == 2

        # Find the record with our specific sample (the one with "What is 2+2?")
        target = [
            r for r in all_records if any(m.get("content") == "What is 2+2?" for m in r["messages"])
        ]
        assert len(target) == 1
        record = target[0]
        assert "messages" in record
        messages = record["messages"]
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == json.dumps({"answer": 4})

    @pytest.mark.asyncio
    async def test_mlx_format_preserves_system_prompt(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
        mock_minio_client: AsyncMock,
    ) -> None:
        """System prompt should be preserved in the formatted output."""
        system_content = "You are an entity extraction assistant for the KarmaService schema."
        sample = make_sample(
            messages=[
                Message(role="system", content=system_content),
                Message(role="user", content="Extract entities from: John works at Google"),
            ],
            response_content=json.dumps(
                {
                    "entities": [
                        {"name": "John", "type": "PERSON"},
                        {"name": "Google", "type": "ORG"},
                    ],
                }
            ),
        )
        sample2 = make_sample()
        mock_collector_client.export_samples.return_value = [sample, sample2]

        request = BatchGenerateRequest(
            schema_id="test_schema",
            min_samples=1,
            train_split=0.9,
        )
        await batch_service.generate_batch(request)

        all_records = _get_all_uploaded_records(mock_minio_client)
        assert len(all_records) == 2
        # Find the record with our entity extraction sample
        target = [r for r in all_records if r["messages"][0]["content"] == system_content]
        assert len(target) == 1
        assert target[0]["messages"][0]["content"] == system_content

    @pytest.mark.asyncio
    async def test_system_prompt_injected_from_routing_rule(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
        mock_minio_client: AsyncMock,
        mock_registry_client: AsyncMock,
    ) -> None:
        """System prompt from routing rule should be prepended to training samples."""
        system_prompt = "You are a structured data extraction assistant. Respond with JSON."
        mock_registry_client.get_routing_rule.return_value = {
            "schema_id": "test_schema",
            "system_prompt": system_prompt,
        }

        # Sample WITHOUT a system message (typical data-collector export)
        sample = make_sample(
            messages=[
                Message(role="user", content="Extract entities from: Paris is in France"),
            ],
            response_content=json.dumps({"entities": [{"name": "Paris"}]}),
        )
        sample2 = make_sample(
            messages=[
                Message(role="user", content="Extract entities from: London is in England"),
            ],
            response_content=json.dumps({"entities": [{"name": "London"}]}),
        )
        mock_collector_client.export_samples.return_value = [sample, sample2]

        request = BatchGenerateRequest(
            schema_id="test_schema",
            min_samples=1,
            train_split=0.9,
        )
        await batch_service.generate_batch(request)

        all_records = _get_all_uploaded_records(mock_minio_client)
        assert len(all_records) == 2
        # Find the Paris sample
        target = [
            r for r in all_records if any("Paris" in m.get("content", "") for m in r["messages"])
        ]
        assert len(target) == 1
        messages = target[0]["messages"]
        # System prompt should be first, then user, then assistant
        assert messages[0] == {"role": "system", "content": system_prompt}
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_system_prompt_not_duplicated_if_already_present(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
        mock_minio_client: AsyncMock,
        mock_registry_client: AsyncMock,
    ) -> None:
        """If sample already has a system message, routing rule prompt should not be prepended."""
        mock_registry_client.get_routing_rule.return_value = {
            "system_prompt": "New system prompt from routing rule",
        }

        # Samples WITH an existing system message
        sample = make_sample(
            messages=[
                Message(role="system", content="Original system prompt"),
                Message(role="user", content="Test query"),
            ],
            response_content=json.dumps({"result": "ok"}),
        )
        sample2 = make_sample(
            messages=[
                Message(role="system", content="Original system prompt"),
                Message(role="user", content="Another query"),
            ],
            response_content=json.dumps({"result": "ok2"}),
        )
        mock_collector_client.export_samples.return_value = [sample, sample2]

        request = BatchGenerateRequest(
            schema_id="test_schema",
            min_samples=1,
            train_split=0.9,
        )
        await batch_service.generate_batch(request)

        all_records = _get_all_uploaded_records(mock_minio_client)
        messages = all_records[0]["messages"]
        # Should keep original system prompt, not prepend new one
        assert messages[0] == {"role": "system", "content": "Original system prompt"}
        assert len(messages) == 3  # system + user + assistant

    @pytest.mark.asyncio
    async def test_no_system_prompt_when_routing_rule_empty(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
        mock_minio_client: AsyncMock,
        mock_registry_client: AsyncMock,
    ) -> None:
        """When routing rule has no system_prompt, samples should not get one prepended."""
        mock_registry_client.get_routing_rule.return_value = {
            "schema_id": "test_schema",
        }

        sample = make_sample(
            messages=[
                Message(role="user", content="Test query"),
            ],
            response_content=json.dumps({"result": "ok"}),
        )
        sample2 = make_sample(
            messages=[
                Message(role="user", content="Another test query"),
            ],
            response_content=json.dumps({"result": "ok2"}),
        )
        mock_collector_client.export_samples.return_value = [sample, sample2]

        request = BatchGenerateRequest(
            schema_id="test_schema",
            min_samples=1,
            train_split=0.9,
        )
        await batch_service.generate_batch(request)

        all_records = _get_all_uploaded_records(mock_minio_client)
        # Find the record with "Test query" (not "Another test query")
        target = [
            r for r in all_records if any(m.get("content") == "Test query" for m in r["messages"])
        ]
        assert len(target) == 1
        messages = target[0]["messages"]
        # Only user + assistant, no system
        assert messages[0]["role"] == "user"
        assert len(messages) == 2


class TestBatchLookup:
    """Test batch retrieval and listing."""

    @pytest.mark.asyncio
    async def test_get_batch_by_id(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
    ) -> None:
        """Generated batches should be retrievable by ID."""
        samples = make_samples(count=100)
        mock_collector_client.export_samples.return_value = samples

        request = BatchGenerateRequest(schema_id="test_schema", min_samples=10)
        batch = await batch_service.generate_batch(request)

        retrieved = await batch_service.get_batch(str(batch.id))
        assert retrieved is not None
        assert retrieved.id == batch.id
        assert retrieved.schema_id == batch.schema_id
        assert retrieved.status == batch.status
        assert retrieved.train_samples == batch.train_samples
        assert retrieved.valid_samples == batch.valid_samples

    @pytest.mark.asyncio
    async def test_get_nonexistent_batch(self, batch_service: BatchService) -> None:
        """Getting a nonexistent batch should return None."""
        result = await batch_service.get_batch("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_batches(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
    ) -> None:
        """Listing batches should return all generated batches."""
        samples = make_samples(count=100)
        mock_collector_client.export_samples.return_value = samples

        for _ in range(3):
            await batch_service.generate_batch(
                BatchGenerateRequest(schema_id="test_schema", min_samples=10)
            )

        batches = await batch_service.list_batches()
        assert len(batches) == 3

    @pytest.mark.asyncio
    async def test_list_batches_by_schema(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
    ) -> None:
        """Listing with schema_id filter should only return matching batches."""
        samples = make_samples(count=100)
        mock_collector_client.export_samples.return_value = samples

        await batch_service.generate_batch(
            BatchGenerateRequest(schema_id="schema_a", min_samples=10)
        )
        await batch_service.generate_batch(
            BatchGenerateRequest(schema_id="schema_b", min_samples=10)
        )
        await batch_service.generate_batch(
            BatchGenerateRequest(schema_id="schema_a", min_samples=10)
        )

        a_batches = await batch_service.list_batches(schema_id="schema_a")
        b_batches = await batch_service.list_batches(schema_id="schema_b")

        assert len(a_batches) == 2
        assert len(b_batches) == 1


class TestBatchAPI:
    """Test batch API endpoints via httpx."""

    @pytest.mark.asyncio
    async def test_generate_batch_endpoint(
        self,
        test_app: AsyncClient,
        mock_collector_client: AsyncMock,
    ) -> None:
        """POST /api/batches/generate should return a completed batch."""
        samples = make_samples(count=100)
        mock_collector_client.export_samples.return_value = samples

        response = await test_app.post(
            "/api/batches/generate",
            json={
                "schema_id": "test_schema",
                "min_samples": 10,
                "train_split": 0.9,
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "completed"
        assert data["schema_id"] == "test_schema"
        assert data["train_samples"] + data["valid_samples"] == 100
        assert data["train_samples"] == 90
        assert data["valid_samples"] == 10
        assert data["total_samples"] == 100
        assert data["minio_path"] is not None
        assert data["train_file"] is not None
        assert data["valid_file"] is not None
        assert data["completed_at"] is not None
        assert data["error_message"] is None
        assert "id" in data
        assert len(data["id"]) > 0

    @pytest.mark.asyncio
    async def test_generate_batch_failure(
        self,
        test_app: AsyncClient,
        mock_collector_client: AsyncMock,
    ) -> None:
        """POST /api/batches/generate should return 422 when generation fails."""
        mock_collector_client.export_samples.return_value = []

        response = await test_app.post(
            "/api/batches/generate",
            json={"schema_id": "empty", "min_samples": 50},
        )

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert "0 samples" in data["detail"]

    @pytest.mark.asyncio
    async def test_list_batches_endpoint(
        self,
        test_app: AsyncClient,
        mock_collector_client: AsyncMock,
    ) -> None:
        """GET /api/batches should return all batches."""
        samples = make_samples(count=100)
        mock_collector_client.export_samples.return_value = samples

        # Generate a batch first
        await test_app.post(
            "/api/batches/generate",
            json={"schema_id": "test", "min_samples": 10},
        )

        response = await test_app.get("/api/batches")
        assert response.status_code == 200
        data = response.json()
        assert "batches" in data
        assert "total" in data
        assert data["total"] >= 1
        assert len(data["batches"]) == data["total"]
        # Verify each batch in list has required fields
        for batch in data["batches"]:
            assert "id" in batch
            assert "schema_id" in batch
            assert "status" in batch

    @pytest.mark.asyncio
    async def test_get_batch_endpoint(
        self,
        test_app: AsyncClient,
        mock_collector_client: AsyncMock,
    ) -> None:
        """GET /api/batches/{id} should return batch details."""
        samples = make_samples(count=100)
        mock_collector_client.export_samples.return_value = samples

        create_response = await test_app.post(
            "/api/batches/generate",
            json={"schema_id": "test", "min_samples": 10},
        )
        batch_id = create_response.json()["id"]

        response = await test_app.get(f"/api/batches/{batch_id}")
        assert response.status_code == 200
        assert response.json()["id"] == batch_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_batch_endpoint(
        self,
        test_app: AsyncClient,
    ) -> None:
        """GET /api/batches/{id} should return 404 for unknown batch."""
        response = await test_app.get("/api/batches/nonexistent")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_health_endpoint(self, test_app: AsyncClient) -> None:
        """GET /health should return service health."""
        response = await test_app.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "training-data-generator"


class TestCollectorClientContract:
    """Test that CollectorClient correctly implements the data-collector API contract.

    These tests mock at the HTTP transport level to verify JSONL parsing,
    correct HTTP methods, and correct field names in payloads.
    """

    @pytest.fixture
    def settings(self) -> Settings:
        """Minimal settings for CollectorClient."""
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

    @pytest.mark.asyncio
    async def test_export_samples_parses_jsonl(self, settings: Settings) -> None:
        """export_samples should parse JSONL (one JSON object per line), not a JSON array."""
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
        # Build JSONL body: one JSON object per line
        jsonl_body = sample_a.model_dump_json() + "\n" + sample_b.model_dump_json()

        mock_transport = httpx.MockTransport(
            lambda request: httpx.Response(200, text=jsonl_body),
        )
        async with httpx.AsyncClient(
            transport=mock_transport, base_url="http://localhost:8151"
        ) as http_client:
            client = CollectorClient(settings, http_client=http_client)
            result = await client.export_samples(schema_id="test")

        assert len(result) == 2
        assert result[0].id == sample_a.id
        assert result[1].id == sample_b.id

    @pytest.mark.asyncio
    async def test_export_samples_handles_empty_response(self, settings: Settings) -> None:
        """export_samples should return an empty list for empty JSONL response."""
        mock_transport = httpx.MockTransport(
            lambda request: httpx.Response(200, text=""),
        )
        async with httpx.AsyncClient(
            transport=mock_transport, base_url="http://localhost:8151"
        ) as http_client:
            client = CollectorClient(settings, http_client=http_client)
            result = await client.export_samples(schema_id="test")

        assert result == []

    @pytest.mark.asyncio
    async def test_mark_used_sends_patch(self, settings: Settings) -> None:
        """mark_used should use PATCH, not POST."""
        captured_request: httpx.Request | None = None

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal captured_request
            captured_request = request
            return httpx.Response(200, json={"updated_count": 1, "training_batch_id": "b1"})

        mock_transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=mock_transport, base_url="http://localhost:8151"
        ) as http_client:
            client = CollectorClient(settings, http_client=http_client)
            await client.mark_used(sample_ids=[uuid4()], batch_id="b1")

        assert captured_request is not None
        assert captured_request.method == "PATCH"

    @pytest.mark.asyncio
    async def test_mark_used_sends_training_batch_id(self, settings: Settings) -> None:
        """mark_used should send 'training_batch_id' field, not 'batch_id'."""
        captured_body: dict | None = None

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal captured_body
            captured_body = json.loads(request.content)
            return httpx.Response(200, json={"updated_count": 1, "training_batch_id": "batch-123"})

        mock_transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=mock_transport, base_url="http://localhost:8151"
        ) as http_client:
            client = CollectorClient(settings, http_client=http_client)
            await client.mark_used(sample_ids=[uuid4()], batch_id="batch-123")

        assert captured_body is not None
        assert "training_batch_id" in captured_body
        assert "batch_id" not in captured_body
        assert captured_body["training_batch_id"] == "batch-123"


def _dispatch_export(
    new_samples: list,
    teacher_samples: list | Exception | None = None,
):
    """Create a side_effect function for export_samples that dispatches by kwargs.

    Instead of relying on call order (brittle), this checks kwargs to decide
    which samples to return:
    - First call (only_unused=True, no fallback_used) -> new_samples
    - Second call (fallback_used=True) -> teacher_samples
    """

    async def _side_effect(*, schema_id: str, **kwargs) -> list:
        if kwargs.get("fallback_used") is True:
            if isinstance(teacher_samples, Exception):
                raise teacher_samples
            return teacher_samples if teacher_samples is not None else []
        return new_samples

    return _side_effect


class TestTeacherSampleReuse:
    """Test teacher sample supplementation in batch generation."""

    @pytest.mark.asyncio
    async def test_teacher_samples_supplement_batch(
        self,
        _init_test_db: None,
        test_settings: Settings,
        mock_minio_client: AsyncMock,
        mock_registry_client: AsyncMock,
        filter_service: FilterService,
    ) -> None:
        """Teacher samples should supplement new samples when enabled."""
        new_samples = make_samples(count=5)
        teacher_samples = make_samples(count=10, provider="deepseek", fallback_count=10)

        mock_collector = AsyncMock(spec=CollectorClient)
        mock_collector.close = AsyncMock()
        mock_collector.export_samples.side_effect = _dispatch_export(new_samples, teacher_samples)
        mock_collector.mark_used = AsyncMock()

        test_settings.reuse_teacher_samples = True
        test_settings.max_teacher_ratio = 0.7

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
        # Should have more than just the 5 new samples
        total = batch.train_samples + batch.valid_samples
        assert total > 5
        # Verify both calls were made (new + teacher)
        assert mock_collector.export_samples.call_count == 2

    @pytest.mark.asyncio
    async def test_teacher_fetch_failure_graceful(
        self,
        _init_test_db: None,
        test_settings: Settings,
        mock_minio_client: AsyncMock,
        mock_registry_client: AsyncMock,
        filter_service: FilterService,
    ) -> None:
        """If teacher sample fetch fails, batch still proceeds with new samples only."""
        new_samples = make_samples(count=15)

        mock_collector = AsyncMock(spec=CollectorClient)
        mock_collector.close = AsyncMock()
        mock_collector.export_samples.side_effect = _dispatch_export(
            new_samples, httpx.ConnectError("connection refused")
        )
        mock_collector.mark_used = AsyncMock()

        test_settings.reuse_teacher_samples = True

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
        total = batch.train_samples + batch.valid_samples
        assert total == 15

    @pytest.mark.asyncio
    async def test_teacher_reuse_disabled(
        self,
        _init_test_db: None,
        test_settings: Settings,
        mock_minio_client: AsyncMock,
        mock_registry_client: AsyncMock,
        filter_service: FilterService,
    ) -> None:
        """When reuse_teacher_samples is False, no teacher fetch should happen."""
        new_samples = make_samples(count=15)

        mock_collector = AsyncMock(spec=CollectorClient)
        mock_collector.close = AsyncMock()
        mock_collector.export_samples.return_value = new_samples
        mock_collector.mark_used = AsyncMock()

        test_settings.reuse_teacher_samples = False

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
        # export_samples called only once (no teacher fetch)
        assert mock_collector.export_samples.call_count == 1

    @pytest.mark.asyncio
    async def test_teacher_empty_no_supplement(
        self,
        _init_test_db: None,
        test_settings: Settings,
        mock_minio_client: AsyncMock,
        mock_registry_client: AsyncMock,
        filter_service: FilterService,
    ) -> None:
        """When teacher fetch returns empty, batch uses only new samples."""
        new_samples = make_samples(count=15)

        mock_collector = AsyncMock(spec=CollectorClient)
        mock_collector.close = AsyncMock()
        mock_collector.export_samples.side_effect = _dispatch_export(new_samples, [])
        mock_collector.mark_used = AsyncMock()

        test_settings.reuse_teacher_samples = True

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
        total = batch.train_samples + batch.valid_samples
        assert total == 15


class TestRegistryClient:
    """Test RegistryClient HTTP interactions."""

    @pytest.fixture
    def settings(self) -> Settings:
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

    @pytest.mark.asyncio
    async def test_get_routing_rule_success(self, settings: Settings) -> None:
        """Should return routing rule dict on 200 response."""
        from services.registry_client import RegistryClient

        rule_data = {"system_prompt": "You are helpful.", "provider_chain": ["Qwen3-4B-AWQ"]}
        mock_transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json=rule_data),
        )
        async with httpx.AsyncClient(
            transport=mock_transport, base_url="http://localhost:8155"
        ) as http_client:
            client = RegistryClient(settings, http_client=http_client)
            result = await client.get_routing_rule("test-schema")

        assert result == rule_data
        assert result["system_prompt"] == "You are helpful."

    @pytest.mark.asyncio
    async def test_get_routing_rule_not_found(self, settings: Settings) -> None:
        """Should return empty dict on 404."""
        from services.registry_client import RegistryClient

        mock_transport = httpx.MockTransport(
            lambda request: httpx.Response(404),
        )
        async with httpx.AsyncClient(
            transport=mock_transport, base_url="http://localhost:8155"
        ) as http_client:
            client = RegistryClient(settings, http_client=http_client)
            result = await client.get_routing_rule("nonexistent")

        assert result == {}

    @pytest.mark.asyncio
    async def test_get_routing_rule_http_error(self, settings: Settings) -> None:
        """Should return empty dict on HTTP error."""
        from services.registry_client import RegistryClient

        mock_transport = httpx.MockTransport(
            lambda request: httpx.Response(500),
        )
        async with httpx.AsyncClient(
            transport=mock_transport, base_url="http://localhost:8155"
        ) as http_client:
            client = RegistryClient(settings, http_client=http_client)
            result = await client.get_routing_rule("failing-schema")

        assert result == {}

    @pytest.mark.asyncio
    async def test_close_owned_client(self, settings: Settings) -> None:
        """Should close client when it owns it."""
        from services.registry_client import RegistryClient

        client = RegistryClient(settings)
        await client.close()
        # Should not raise

    @pytest.mark.asyncio
    async def test_close_injected_client(self, settings: Settings) -> None:
        """Should not close client when injected."""
        from services.registry_client import RegistryClient

        mock_transport = httpx.MockTransport(
            lambda request: httpx.Response(200),
        )
        async with httpx.AsyncClient(
            transport=mock_transport, base_url="http://localhost:8155"
        ) as http_client:
            client = RegistryClient(settings, http_client=http_client)
            await client.close()
            # http_client should still be usable
            assert not http_client.is_closed


class TestContentHash:
    """Test the _content_hash dedup function used for teacher sample dedup."""

    def test_identical_content_same_hash(self) -> None:
        """Identical messages+response should produce the same hash."""
        messages = [Message(role="user", content="hello")]
        response = '{"result": "world"}'
        h1 = _content_hash(messages, response)
        h2 = _content_hash(messages, response)
        assert h1 == h2

    def test_different_response_different_hash(self) -> None:
        """Different responses should produce different hashes."""
        messages = [Message(role="user", content="hello")]
        h1 = _content_hash(messages, '{"result": "world"}')
        h2 = _content_hash(messages, '{"result": "different"}')
        assert h1 != h2

    def test_different_messages_different_hash(self) -> None:
        """Different messages should produce different hashes."""
        response = '{"result": "same"}'
        h1 = _content_hash([Message(role="user", content="hello")], response)
        h2 = _content_hash([Message(role="user", content="goodbye")], response)
        assert h1 != h2

    def test_hash_is_sha256_hex(self) -> None:
        """Hash should be a 64-char hex string (SHA-256)."""
        h = _content_hash([Message(role="user", content="test")], "response")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


class TestTeacherDedup:
    """Test teacher sample deduplication against new samples."""

    @pytest.mark.asyncio
    async def test_teacher_dedup_removes_duplicates(
        self,
        _init_test_db: None,
        test_settings: Settings,
        mock_minio_client: AsyncMock,
        mock_registry_client: AsyncMock,
        filter_service: FilterService,
    ) -> None:
        """Teacher samples that duplicate new samples should be excluded."""
        shared_messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Identical query"),
        ]
        shared_response = json.dumps({"result": "identical_response"})

        # Need 2 new samples to meet min_total=ceil(1/0.9)=2
        new_samples = [
            make_sample(messages=shared_messages, response_content=shared_response),
            make_sample(),
        ]
        teacher_samples = [
            make_sample(
                messages=shared_messages,
                response_content=shared_response,
                provider="deepseek",
                fallback_used=True,
            ),
        ]

        mock_collector = AsyncMock(spec=CollectorClient)
        mock_collector.close = AsyncMock()
        mock_collector.export_samples.side_effect = _dispatch_export(new_samples, teacher_samples)
        mock_collector.mark_used = AsyncMock()

        test_settings.reuse_teacher_samples = True

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
        total = batch.train_samples + batch.valid_samples
        # Should have 2 new samples, not the duplicate teacher
        assert total == 2

    @pytest.mark.asyncio
    async def test_only_new_samples_marked_used(
        self,
        _init_test_db: None,
        test_settings: Settings,
        mock_minio_client: AsyncMock,
        mock_registry_client: AsyncMock,
        filter_service: FilterService,
    ) -> None:
        """Only NEW samples (not reused teacher samples) should be marked as used."""
        new_samples = make_samples(count=5)
        teacher_samples = [make_sample(provider="deepseek", fallback_used=True) for _ in range(3)]
        # Mark teacher samples as already used
        for s in teacher_samples:
            s.used_in_training = True

        mock_collector = AsyncMock(spec=CollectorClient)
        mock_collector.close = AsyncMock()
        mock_collector.export_samples.side_effect = _dispatch_export(new_samples, teacher_samples)
        mock_collector.mark_used = AsyncMock()

        test_settings.reuse_teacher_samples = True

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
        # mark_used should only be called with the 5 new sample IDs
        mock_collector.mark_used.assert_called_once()
        marked_ids = mock_collector.mark_used.call_args.args[0]
        assert len(marked_ids) == 5


class TestCheckAndGenerateMinSamplesFromSettings:
    """Verify check_and_generate uses settings.min_samples, not a hardcoded value."""

    @pytest.mark.asyncio
    async def test_check_and_generate_uses_settings_min_samples(
        self,
        test_settings: Settings,
        mock_collector_client: AsyncMock,
        mock_minio_client: AsyncMock,
        mock_http_client: AsyncMock,
        batch_service: BatchService,
    ) -> None:
        """TriggerService.check_and_generate should pass settings.min_samples to BatchGenerateRequest.

        This verifies the fix for the bug where min_samples was hardcoded to 50.
        """
        from unittest.mock import MagicMock

        from services.trigger_service import TriggerService

        # Set min_samples to 3 in settings
        test_settings.min_samples = 3

        trigger_service = TriggerService(
            settings=test_settings,
            collector_client=mock_collector_client,
            batch_service=batch_service,
            http_client=mock_http_client,
        )

        mock_collector_client.list_schema_ids.return_value = ["test_schema"]
        mock_collector_client.get_sample_count.return_value = 250
        # Provide exactly 4 samples (passes min_samples=3, would fail if hardcoded to 50)
        mock_collector_client.export_samples.return_value = make_samples(count=4)
        mock_http_client.get.return_value = MagicMock(
            status_code=404,
            json=lambda: {},
        )

        batches = await trigger_service.check_and_generate()

        assert len(batches) == 1
        assert batches[0].status == BatchStatus.COMPLETED
        # With only 4 samples and min_samples=3, this should succeed
        total = batches[0].train_samples + batches[0].valid_samples
        assert total == 4


# ---------------------------------------------------------------------------
# Phase 3: mode='global' aggregated batch (single-LoRA rollout)
# ---------------------------------------------------------------------------


class TestGlobalBatchBuilder:
    """Tests for the mode='global' aggregated batch path.

    Per single-adapter-rollout-plan.md Phase 3:
      - aggregates samples across all schemas (no schema filter)
      - applies existing quality filters
      - applies a per-schema balance cap (default 30%) to prevent skew
      - uploads to training-data/global/<batch_id>.jsonl
      - marks consumed samples in data-collector
      - reports source_schema_counts in the result
    """

    @pytest.mark.asyncio
    async def test_global_batch_aggregates_all_schemas(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
        mock_minio_client: AsyncMock,
    ) -> None:
        """All represented schemas should appear in source_schema_counts."""
        # 3 schemas, balanced enough to NOT trip the 30% cap (33% each is the
        # natural floor with 3 schemas, but we use 30/30/30 with a 90-sample
        # pool so each is 33% — the cap will trim each to 27 (=0.30*90)).
        # To get a clean "all included, no cap effect" test we use 4 schemas
        # at 25% each, well under the 30% cap.
        samples = (
            make_samples(count=25, schema_id="schema_a")
            + make_samples(count=25, schema_id="schema_b")
            + make_samples(count=25, schema_id="schema_c")
            + make_samples(count=25, schema_id="schema_d")
        )
        mock_collector_client.list_validated_samples.return_value = samples

        request = BatchGenerateRequest(
            mode="global",
            min_samples=50,
            train_split=0.9,
        )
        batch = await batch_service.generate_batch(request)

        assert batch.status == BatchStatus.COMPLETED
        assert batch.mode == "global"
        # schema_id is empty for global batches (no single owning schema).
        assert batch.schema_id == ""
        assert batch.total_samples == 100
        assert set(batch.source_schema_counts.keys()) == {
            "schema_a",
            "schema_b",
            "schema_c",
            "schema_d",
        }
        # No cap was triggered (each schema is 25% of pool, < 30%).
        assert all(c == 25 for c in batch.source_schema_counts.values())
        # collector.list_validated_samples must have been called with
        # schema_id=None (i.e. no schema filter).
        call_kwargs = mock_collector_client.list_validated_samples.call_args.kwargs
        assert call_kwargs.get("schema_id") is None
        # Upload key must live under the global/ prefix.
        keys = _get_upload_keys(mock_minio_client)
        assert all("training-data/global/" in k for k in keys), keys

    @pytest.mark.asyncio
    async def test_global_batch_applies_balance_cap(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
        mock_minio_client: AsyncMock,
        test_settings: Settings,
    ) -> None:
        """One dominant schema should be downsampled to <=cap_fraction of pool."""
        # 90% of pool from one schema → must be downsampled to <=30% (cap default).
        dominant = make_samples(count=180, schema_id="dominant")
        minor_a = make_samples(count=10, schema_id="minor_a")
        minor_b = make_samples(count=10, schema_id="minor_b")
        mock_collector_client.list_validated_samples.return_value = dominant + minor_a + minor_b

        request = BatchGenerateRequest(
            mode="global",
            min_samples=20,
            train_split=0.9,
        )
        batch = await batch_service.generate_batch(request)

        assert batch.status == BatchStatus.COMPLETED
        assert batch.mode == "global"
        cap = test_settings.batch_max_schema_fraction
        # Compute final pool size from the recorded counts.
        final_total = sum(batch.source_schema_counts.values())
        dominant_count = batch.source_schema_counts.get("dominant", 0)
        # The dominant schema must not exceed cap of the final pool.
        assert dominant_count / final_total <= cap + 1e-9, (
            f"dominant={dominant_count}/{final_total} > cap={cap}"
        )
        # The two minor schemas should be untouched (they were each well under cap).
        assert batch.source_schema_counts["minor_a"] == 10
        assert batch.source_schema_counts["minor_b"] == 10
        # Sample dropped from the dominant schema must NOT have been marked
        # consumed (only kept samples should be marked). Confirm mark_used was
        # called with at most final_total IDs.
        mock_collector_client.mark_used.assert_called_once()
        marked_ids = mock_collector_client.mark_used.call_args.args[0]
        assert len(marked_ids) <= final_total

    @pytest.mark.asyncio
    async def test_global_batch_balance_cap_is_reproducible(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
        mock_minio_client: AsyncMock,
    ) -> None:
        """Same input + same batch_id seed must produce same per-schema counts.

        We exercise the static helper directly because TrainingBatch.id is
        UUIDv7 (time-keyed), so two real batches can't share an id; the seed
        is derived from the id so two real runs naturally differ. This test
        pins the cap algorithm itself.
        """
        samples = make_samples(count=200, schema_id="dominant") + make_samples(
            count=20, schema_id="minor"
        )
        capped_a, counts_a = BatchService._apply_schema_balance_cap(
            samples, cap_fraction=0.30, seed=42
        )
        capped_b, counts_b = BatchService._apply_schema_balance_cap(
            samples, cap_fraction=0.30, seed=42
        )
        assert counts_a == counts_b
        assert [s.id for s in capped_a] == [s.id for s in capped_b]

    @pytest.mark.asyncio
    async def test_per_schema_batch_unchanged(
        self,
        batch_service: BatchService,
        mock_collector_client: AsyncMock,
        mock_minio_client: AsyncMock,
    ) -> None:
        """Default mode='per_schema' path must keep its old behavior + key prefix."""
        samples = make_samples(count=100, fallback_count=10)
        mock_collector_client.export_samples.return_value = samples

        request = BatchGenerateRequest(
            schema_id="test_schema",
            min_samples=50,
            train_split=0.9,
        )
        batch = await batch_service.generate_batch(request)

        assert batch.status == BatchStatus.COMPLETED
        assert batch.mode == "per_schema"
        assert batch.schema_id == "test_schema"
        # source_schema_counts must be empty for per_schema mode.
        assert batch.source_schema_counts == {}
        # Per-schema upload prefix must NOT be the global one.
        keys = _get_upload_keys(mock_minio_client)
        assert all("training-data/test_schema/" in k for k in keys), keys
        assert not any("training-data/global/" in k for k in keys), keys
        # Collector must have been called WITH a schema_id (string).
        call_kwargs = mock_collector_client.export_samples.call_args.kwargs
        assert call_kwargs.get("schema_id") == "test_schema"

    @pytest.mark.asyncio
    async def test_global_batch_endpoint_no_schema_id(
        self,
        test_app: AsyncClient,
        mock_collector_client: AsyncMock,
    ) -> None:
        """POST /api/batches/generate with mode=global and no schema_id succeeds."""
        samples = (
            make_samples(count=40, schema_id="alpha")
            + make_samples(count=40, schema_id="beta")
            + make_samples(count=40, schema_id="gamma")
        )
        mock_collector_client.list_validated_samples.return_value = samples

        response = await test_app.post(
            "/api/batches/generate",
            json={"mode": "global", "min_samples": 20, "train_split": 0.9},
        )
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["mode"] == "global"
        assert body["schema_id"] == ""
        assert body["status"] == "completed"
        assert isinstance(body["source_schema_counts"], dict)
        # All three source schemas must appear.
        assert set(body["source_schema_counts"].keys()) == {"alpha", "beta", "gamma"}

    def test_per_schema_request_requires_schema_id(self) -> None:
        """BatchGenerateRequest validation: per_schema without schema_id rejected."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="schema_id is required"):
            BatchGenerateRequest(mode="per_schema", min_samples=10)

    def test_global_request_does_not_require_schema_id(self) -> None:
        """mode='global' lets schema_id stay None."""
        req = BatchGenerateRequest(mode="global", min_samples=10)
        assert req.schema_id is None
        assert req.mode == "global"
