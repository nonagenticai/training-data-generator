"""Tests for retraining trigger evaluation.

Covers all four trigger conditions:
1. Initial threshold (>= 200 samples, no prior training)
2. New samples (>= 100 samples since last training)
3. Quality drop (quality below 80%)
4. Time elapsed (>= 14 days since last training)
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import AsyncClient

from models.domain import BatchStatus, TrainingBatch
from services.batch_service import BatchService
from services.trigger_service import TriggerService
from tests.conftest import make_samples


class TestInitialThreshold:
    """Test initial training trigger (>= 200 samples, never trained)."""

    @pytest.mark.asyncio
    async def test_triggers_at_threshold(
        self,
        trigger_service: TriggerService,
        mock_collector_client: AsyncMock,
        mock_http_client: AsyncMock,
    ) -> None:
        """Should trigger when sample count reaches initial threshold."""
        mock_collector_client.list_schema_ids.return_value = ["test_schema"]
        mock_collector_client.get_sample_count.return_value = 200

        # Mock quality endpoint - return None (no model yet)
        mock_http_client.get.return_value = MagicMock(
            status_code=404,
            json=lambda: {},
        )

        triggers = await trigger_service.check_triggers()

        assert len(triggers) == 1
        assert triggers[0].should_retrain is True
        assert triggers[0].trigger_reason == "initial_threshold"
        assert triggers[0].available_samples == 200

    @pytest.mark.asyncio
    async def test_does_not_trigger_below_threshold(
        self,
        trigger_service: TriggerService,
        mock_collector_client: AsyncMock,
        mock_http_client: AsyncMock,
    ) -> None:
        """Should not trigger when below initial threshold."""
        mock_collector_client.list_schema_ids.return_value = ["test_schema"]
        mock_collector_client.get_sample_count.return_value = 150

        mock_http_client.get.return_value = MagicMock(
            status_code=404,
            json=lambda: {},
        )

        triggers = await trigger_service.check_triggers()

        assert len(triggers) == 1
        assert triggers[0].should_retrain is False
        assert triggers[0].trigger_reason == "none"


class TestNewSamples:
    """Test new samples trigger (>= 100 new since last training)."""

    @pytest.mark.asyncio
    async def test_triggers_on_new_samples(
        self,
        trigger_service: TriggerService,
        mock_collector_client: AsyncMock,
        mock_http_client: AsyncMock,
        batch_service: BatchService,
    ) -> None:
        """Should trigger when enough new samples are available after prior training."""
        mock_collector_client.list_schema_ids.return_value = ["test_schema"]
        mock_collector_client.get_sample_count.return_value = 100

        # Create a prior completed batch to simulate previous training
        prior_batch = TrainingBatch(
            schema_id="test_schema",
            status=BatchStatus.COMPLETED,
            completed_at=datetime.now(UTC) - timedelta(days=1),
        )
        await batch_service._save_batch(prior_batch)

        mock_http_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"quality_score": 0.90},
        )

        triggers = await trigger_service.check_triggers()

        assert len(triggers) == 1
        assert triggers[0].should_retrain is True
        assert triggers[0].trigger_reason == "new_samples"

    @pytest.mark.asyncio
    async def test_does_not_trigger_insufficient_new_samples(
        self,
        trigger_service: TriggerService,
        mock_collector_client: AsyncMock,
        mock_http_client: AsyncMock,
        batch_service: BatchService,
    ) -> None:
        """Should not trigger when too few new samples are available."""
        mock_collector_client.list_schema_ids.return_value = ["test_schema"]
        mock_collector_client.get_sample_count.return_value = 50

        prior_batch = TrainingBatch(
            schema_id="test_schema",
            status=BatchStatus.COMPLETED,
            completed_at=datetime.now(UTC) - timedelta(days=1),
        )
        await batch_service._save_batch(prior_batch)

        mock_http_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"quality_score": 0.90},
        )

        triggers = await trigger_service.check_triggers()

        assert len(triggers) == 1
        assert triggers[0].should_retrain is False


class TestQualityDrop:
    """Test quality drop trigger (quality < 80%)."""

    @pytest.mark.asyncio
    async def test_triggers_on_quality_drop(
        self,
        trigger_service: TriggerService,
        mock_collector_client: AsyncMock,
        mock_http_client: AsyncMock,
        batch_service: BatchService,
    ) -> None:
        """Should trigger when model quality drops below threshold."""
        mock_collector_client.list_schema_ids.return_value = ["test_schema"]
        mock_collector_client.get_sample_count.return_value = 30  # Below new_samples threshold

        prior_batch = TrainingBatch(
            schema_id="test_schema",
            status=BatchStatus.COMPLETED,
            completed_at=datetime.now(UTC) - timedelta(days=3),
        )
        await batch_service._save_batch(prior_batch)

        # Quality below threshold
        mock_http_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"quality_score": 0.65},
        )

        triggers = await trigger_service.check_triggers()

        assert len(triggers) == 1
        assert triggers[0].should_retrain is True
        assert triggers[0].trigger_reason == "quality_drop"
        assert triggers[0].current_quality == 0.65

    @pytest.mark.asyncio
    async def test_does_not_trigger_quality_above_threshold(
        self,
        trigger_service: TriggerService,
        mock_collector_client: AsyncMock,
        mock_http_client: AsyncMock,
        batch_service: BatchService,
    ) -> None:
        """Should not trigger on quality alone when above threshold."""
        mock_collector_client.list_schema_ids.return_value = ["test_schema"]
        mock_collector_client.get_sample_count.return_value = 30

        prior_batch = TrainingBatch(
            schema_id="test_schema",
            status=BatchStatus.COMPLETED,
            completed_at=datetime.now(UTC) - timedelta(days=3),
        )
        await batch_service._save_batch(prior_batch)

        mock_http_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"quality_score": 0.85},
        )

        triggers = await trigger_service.check_triggers()

        assert len(triggers) == 1
        assert triggers[0].should_retrain is False


class TestTimeElapsed:
    """Test time elapsed trigger (>= 14 days since last training)."""

    @pytest.mark.asyncio
    async def test_triggers_after_14_days(
        self,
        trigger_service: TriggerService,
        mock_collector_client: AsyncMock,
        mock_http_client: AsyncMock,
        batch_service: BatchService,
    ) -> None:
        """Should trigger when 14+ days have passed since last training."""
        mock_collector_client.list_schema_ids.return_value = ["test_schema"]
        mock_collector_client.get_sample_count.return_value = 30  # Below new_samples threshold

        prior_batch = TrainingBatch(
            schema_id="test_schema",
            status=BatchStatus.COMPLETED,
            completed_at=datetime.now(UTC) - timedelta(days=15),
        )
        await batch_service._save_batch(prior_batch)

        # Quality above threshold
        mock_http_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"quality_score": 0.90},
        )

        triggers = await trigger_service.check_triggers()

        assert len(triggers) == 1
        assert triggers[0].should_retrain is True
        assert triggers[0].trigger_reason == "time_elapsed"

    @pytest.mark.asyncio
    async def test_does_not_trigger_before_14_days(
        self,
        trigger_service: TriggerService,
        mock_collector_client: AsyncMock,
        mock_http_client: AsyncMock,
        batch_service: BatchService,
    ) -> None:
        """Should not trigger when less than 14 days have passed."""
        mock_collector_client.list_schema_ids.return_value = ["test_schema"]
        mock_collector_client.get_sample_count.return_value = 30

        prior_batch = TrainingBatch(
            schema_id="test_schema",
            status=BatchStatus.COMPLETED,
            completed_at=datetime.now(UTC) - timedelta(days=10),
        )
        await batch_service._save_batch(prior_batch)

        mock_http_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"quality_score": 0.90},
        )

        triggers = await trigger_service.check_triggers()

        assert len(triggers) == 1
        assert triggers[0].should_retrain is False


class TestCheckAndGenerate:
    """Test the combined check-and-generate workflow."""

    @pytest.mark.asyncio
    async def test_generates_batches_for_triggered_schemas(
        self,
        trigger_service: TriggerService,
        mock_collector_client: AsyncMock,
        mock_minio_client: AsyncMock,
        mock_http_client: AsyncMock,
    ) -> None:
        """Should generate batches only for schemas that need retraining."""
        mock_collector_client.list_schema_ids.return_value = ["needs_train", "not_needed"]
        mock_collector_client.get_sample_count.side_effect = [
            250,  # needs_train - above initial threshold
            50,  # not_needed - below threshold
        ]
        mock_collector_client.export_samples.return_value = make_samples(count=250)

        mock_http_client.get.return_value = MagicMock(
            status_code=404,
            json=lambda: {},
        )

        batches = await trigger_service.check_and_generate()

        assert len(batches) == 1
        assert batches[0].schema_id == "needs_train"


class TestTriggerAPI:
    """Test trigger API endpoints via httpx."""

    @pytest.mark.asyncio
    async def test_check_triggers_endpoint(
        self,
        test_app: AsyncClient,
        mock_collector_client: AsyncMock,
        trigger_service: TriggerService,
        mock_http_client: AsyncMock,
    ) -> None:
        """GET /api/triggers/check should return trigger evaluations."""
        mock_collector_client.list_schema_ids.return_value = ["schema_a"]
        mock_collector_client.get_sample_count.return_value = 300

        mock_http_client.get.return_value = MagicMock(
            status_code=404,
            json=lambda: {},
        )

        response = await test_app.get("/api/triggers/check")

        assert response.status_code == 200
        data = response.json()
        assert "triggers" in data
        assert "schemas_needing_retrain" in data
        assert data["schemas_needing_retrain"] >= 1
        # Verify trigger structure
        trigger = data["triggers"][0]
        assert "schema_id" in trigger
        assert "trigger_reason" in trigger
        assert "available_samples" in trigger
        assert "should_retrain" in trigger
        assert trigger["schema_id"] == "schema_a"
        assert trigger["should_retrain"] is True
        assert trigger["trigger_reason"] == "initial_threshold"

    @pytest.mark.asyncio
    async def test_check_and_generate_endpoint(
        self,
        test_app: AsyncClient,
        mock_collector_client: AsyncMock,
        mock_minio_client: AsyncMock,
        trigger_service: TriggerService,
        mock_http_client: AsyncMock,
    ) -> None:
        """POST /api/triggers/check-and-generate should generate batches."""
        mock_collector_client.list_schema_ids.return_value = ["needs_train"]
        mock_collector_client.get_sample_count.return_value = 250
        mock_collector_client.export_samples.return_value = make_samples(count=250)

        mock_http_client.get.return_value = MagicMock(
            status_code=404,
            json=lambda: {},
        )

        response = await test_app.post("/api/triggers/check-and-generate")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        # Verify batch result structure (this is what flywheel consumes)
        batch = data[0]
        assert "id" in batch
        assert "schema_id" in batch
        assert batch["schema_id"] == "needs_train"
        assert "status" in batch
        assert "train_samples" in batch
        assert "valid_samples" in batch
        assert "train_file" in batch
        assert "valid_file" in batch
        assert "minio_path" in batch
        assert "source_providers" in batch


class TestGracefulDegradation:
    """Test graceful degradation when model-registry is unavailable."""

    @pytest.mark.asyncio
    async def test_quality_check_handles_registry_failure(
        self,
        trigger_service: TriggerService,
        mock_collector_client: AsyncMock,
        mock_http_client: AsyncMock,
    ) -> None:
        """Trigger check should still work when model-registry is down."""
        mock_collector_client.list_schema_ids.return_value = ["test_schema"]
        mock_collector_client.get_sample_count.return_value = 250

        # Simulate registry failure
        mock_http_client.get.side_effect = ConnectionError("Registry down")

        triggers = await trigger_service.check_triggers()

        assert len(triggers) == 1
        assert triggers[0].current_quality is None
        # Should still trigger on initial threshold
        assert triggers[0].should_retrain is True
        assert triggers[0].trigger_reason == "initial_threshold"

    @pytest.mark.asyncio
    async def test_multiple_schemas(
        self,
        trigger_service: TriggerService,
        mock_collector_client: AsyncMock,
        mock_http_client: AsyncMock,
    ) -> None:
        """Should evaluate triggers for all schemas independently."""
        mock_collector_client.list_schema_ids.return_value = [
            "schema_a",
            "schema_b",
            "schema_c",
        ]
        mock_collector_client.get_sample_count.side_effect = [250, 50, 300]

        mock_http_client.get.return_value = MagicMock(
            status_code=404,
            json=lambda: {},
        )

        triggers = await trigger_service.check_triggers()

        assert len(triggers) == 3
        needing = [t for t in triggers if t.should_retrain]
        assert len(needing) == 2  # schema_a and schema_c


class TestViabilityGate:
    """Test the viability gate that blocks retraining when samples are insufficient for a batch.

    With test_settings: min_samples=10, train_split=0.9,
    min_viable = ceil(10/0.9) = 12.  Any trigger that fires with
    available < 12 should be suppressed.
    """

    @pytest.mark.asyncio
    async def test_quality_drop_blocked_by_viability_gate(
        self,
        trigger_service: TriggerService,
        mock_collector_client: AsyncMock,
        mock_http_client: AsyncMock,
        batch_service: BatchService,
    ) -> None:
        """Quality drop trigger should be blocked when available < min_viable."""
        mock_collector_client.list_schema_ids.return_value = ["test_schema"]
        # 5 samples is below min_viable (12) but quality_drop trigger fires anyway
        mock_collector_client.get_sample_count.return_value = 5

        prior_batch = TrainingBatch(
            schema_id="test_schema",
            status=BatchStatus.COMPLETED,
            completed_at=datetime.now(UTC) - timedelta(days=3),
        )
        await batch_service._save_batch(prior_batch)

        mock_http_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"quality_score": 0.50},
        )

        triggers = await trigger_service.check_triggers()

        assert len(triggers) == 1
        assert triggers[0].should_retrain is False
        assert triggers[0].trigger_reason == "none"
        assert triggers[0].available_samples == 5

    @pytest.mark.asyncio
    async def test_time_elapsed_blocked_by_viability_gate(
        self,
        trigger_service: TriggerService,
        mock_collector_client: AsyncMock,
        mock_http_client: AsyncMock,
        batch_service: BatchService,
    ) -> None:
        """Time elapsed trigger should be blocked when available < min_viable."""
        mock_collector_client.list_schema_ids.return_value = ["test_schema"]
        mock_collector_client.get_sample_count.return_value = 8  # Below 12

        prior_batch = TrainingBatch(
            schema_id="test_schema",
            status=BatchStatus.COMPLETED,
            completed_at=datetime.now(UTC) - timedelta(days=30),
        )
        await batch_service._save_batch(prior_batch)

        mock_http_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"quality_score": 0.90},
        )

        triggers = await trigger_service.check_triggers()

        assert len(triggers) == 1
        assert triggers[0].should_retrain is False
        assert triggers[0].trigger_reason == "none"

    @pytest.mark.asyncio
    async def test_quality_drop_passes_viability_gate(
        self,
        trigger_service: TriggerService,
        mock_collector_client: AsyncMock,
        mock_http_client: AsyncMock,
        batch_service: BatchService,
    ) -> None:
        """Quality drop trigger should pass when available >= min_viable."""
        mock_collector_client.list_schema_ids.return_value = ["test_schema"]
        # 12 == min_viable, so should pass
        mock_collector_client.get_sample_count.return_value = 12

        prior_batch = TrainingBatch(
            schema_id="test_schema",
            status=BatchStatus.COMPLETED,
            completed_at=datetime.now(UTC) - timedelta(days=3),
        )
        await batch_service._save_batch(prior_batch)

        mock_http_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"quality_score": 0.50},
        )

        triggers = await trigger_service.check_triggers()

        assert len(triggers) == 1
        assert triggers[0].should_retrain is True
        assert triggers[0].trigger_reason == "quality_drop"

    @pytest.mark.asyncio
    async def test_viability_gate_at_exact_boundary(
        self,
        trigger_service: TriggerService,
        mock_collector_client: AsyncMock,
        mock_http_client: AsyncMock,
        batch_service: BatchService,
    ) -> None:
        """At exactly min_viable-1, should be blocked; at min_viable, should pass."""
        mock_collector_client.list_schema_ids.return_value = ["test_schema"]
        # 11 < 12, should be blocked
        mock_collector_client.get_sample_count.return_value = 11

        prior_batch = TrainingBatch(
            schema_id="test_schema",
            status=BatchStatus.COMPLETED,
            completed_at=datetime.now(UTC) - timedelta(days=30),
        )
        await batch_service._save_batch(prior_batch)

        mock_http_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"quality_score": 0.90},
        )

        triggers = await trigger_service.check_triggers()

        assert len(triggers) == 1
        assert triggers[0].should_retrain is False
        assert triggers[0].trigger_reason == "none"


class TestGenerateAndTrain:
    """Test the generate_and_train method called by the fire endpoint."""

    @pytest.mark.asyncio
    async def test_generate_and_train_calls_batch_service(
        self,
        trigger_service: TriggerService,
        mock_collector_client: AsyncMock,
        mock_minio_client: AsyncMock,
        mock_http_client: AsyncMock,
    ) -> None:
        """generate_and_train should call batch_service.generate_batch and return result."""
        mock_collector_client.export_samples.return_value = make_samples(count=50)

        batch = await trigger_service.generate_and_train("test_schema")

        assert batch.schema_id == "test_schema"
        assert batch.status == BatchStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_generate_and_train_notifies_flywheel_on_success(
        self,
        trigger_service: TriggerService,
        mock_collector_client: AsyncMock,
        mock_minio_client: AsyncMock,
        mock_http_client: AsyncMock,
    ) -> None:
        """Should POST to flywheel when batch completes successfully."""
        mock_collector_client.export_samples.return_value = make_samples(count=50)
        post_response = MagicMock()
        post_response.raise_for_status = MagicMock()
        mock_http_client.post.return_value = post_response

        batch = await trigger_service.generate_and_train("test_schema")

        assert batch.status == BatchStatus.COMPLETED
        assert batch.minio_path is not None
        mock_http_client.post.assert_called_once()
        call_kwargs = mock_http_client.post.call_args
        assert "/api/flywheel/run" in call_kwargs.args[0]
        payload = call_kwargs.kwargs["json"]
        assert payload["schema_id"] == "test_schema"
        assert payload["batch_id"] == str(batch.id)
        assert payload["minio_path"] == batch.minio_path

    @pytest.mark.asyncio
    async def test_generate_and_train_does_not_notify_on_failure(
        self,
        trigger_service: TriggerService,
        mock_collector_client: AsyncMock,
        mock_minio_client: AsyncMock,
        mock_http_client: AsyncMock,
    ) -> None:
        """Should NOT call flywheel when batch fails (insufficient samples)."""
        # Return too few samples so batch fails
        mock_collector_client.export_samples.return_value = make_samples(count=2)

        batch = await trigger_service.generate_and_train("test_schema")

        assert batch.status == BatchStatus.FAILED
        mock_http_client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_notify_flywheel_is_fire_and_forget(
        self,
        trigger_service: TriggerService,
        mock_collector_client: AsyncMock,
        mock_minio_client: AsyncMock,
        mock_http_client: AsyncMock,
    ) -> None:
        """Flywheel notification failure should not crash generate_and_train."""
        mock_collector_client.export_samples.return_value = make_samples(count=50)
        mock_http_client.post.side_effect = ConnectionError("Flywheel is down")

        # Should not raise
        batch = await trigger_service.generate_and_train("test_schema")

        assert batch.status == BatchStatus.COMPLETED
        assert batch.minio_path is not None


class TestFireTriggerAPI:
    """Test the POST /api/triggers/fire endpoint."""

    @pytest.mark.asyncio
    async def test_fire_endpoint_returns_accepted(
        self,
        test_app: AsyncClient,
    ) -> None:
        """POST /api/triggers/fire should return 202 accepted immediately."""
        response = await test_app.post(
            "/api/triggers/fire",
            json={"schema_id": "test_schema", "available_samples": 50},
        )

        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "accepted"
        assert data["schema_id"] == "test_schema"

    @pytest.mark.asyncio
    async def test_fire_endpoint_validates_request(
        self,
        test_app: AsyncClient,
    ) -> None:
        """POST /api/triggers/fire with missing fields should return 422."""
        response = await test_app.post(
            "/api/triggers/fire",
            json={"schema_id": "test_schema"},  # missing available_samples
        )

        assert response.status_code == 422


class TestGenerateAndTrainGlobal:
    """Test :meth:`TriggerService.generate_and_train_global` for the system-wide
    single-LoRA event-based path."""

    @pytest.mark.asyncio
    async def test_global_submits_job_to_model_trainer_on_success(
        self,
        trigger_service: TriggerService,
        mock_collector_client: AsyncMock,
        mock_minio_client: AsyncMock,
        mock_http_client: AsyncMock,
    ) -> None:
        """On a completed mode=global batch, POST a mode=global job to model-trainer."""
        # mode=global uses list_validated_samples (NOT export_samples like per-schema).
        mock_collector_client.list_validated_samples.return_value = make_samples(count=50)
        mock_collector_client.export_samples.return_value = make_samples(count=50)
        post_response = MagicMock()
        post_response.raise_for_status = MagicMock()
        post_response.json = MagicMock(return_value={"id": "fake-job-uuid", "status": "queued"})
        mock_http_client.post.return_value = post_response

        batch = await trigger_service.generate_and_train_global()

        assert batch.status == BatchStatus.COMPLETED
        mock_http_client.post.assert_called_once()
        url_arg = mock_http_client.post.call_args.args[0]
        assert url_arg.endswith("/api/jobs")
        payload = mock_http_client.post.call_args.kwargs["json"]
        assert payload["mode"] == "global"
        assert payload["schema_id"] == "__global__"
        assert payload["training_batch_id"] == str(batch.id)
        assert payload["minio_path"] == batch.minio_path
        assert payload["config"]["base_model"] == "Qwen/Qwen3-4B-AWQ"

    @pytest.mark.asyncio
    async def test_global_does_not_submit_when_batch_fails(
        self,
        trigger_service: TriggerService,
        mock_collector_client: AsyncMock,
        mock_minio_client: AsyncMock,
        mock_http_client: AsyncMock,
    ) -> None:
        """If batch generation fails, no model-trainer POST is made."""
        mock_collector_client.list_validated_samples.return_value = make_samples(count=2)
        mock_collector_client.export_samples.return_value = make_samples(count=2)

        batch = await trigger_service.generate_and_train_global()

        assert batch.status == BatchStatus.FAILED
        mock_http_client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_global_submission_is_fire_and_forget(
        self,
        trigger_service: TriggerService,
        mock_collector_client: AsyncMock,
        mock_minio_client: AsyncMock,
        mock_http_client: AsyncMock,
    ) -> None:
        """A model-trainer POST failure must not crash generate_and_train_global."""
        mock_collector_client.list_validated_samples.return_value = make_samples(count=50)
        mock_collector_client.export_samples.return_value = make_samples(count=50)
        mock_http_client.post.side_effect = ConnectionError("model-trainer is down")

        batch = await trigger_service.generate_and_train_global()

        assert batch.status == BatchStatus.COMPLETED


class TestFireGlobalTriggerAPI:
    """Test the POST /api/triggers/fire-global endpoint."""

    @pytest.mark.asyncio
    async def test_fire_global_returns_accepted(self, test_app: AsyncClient) -> None:
        response = await test_app.post(
            "/api/triggers/fire-global",
            json={"available_samples": 1000},
        )
        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "accepted"
        assert data["scope"] == "global"

    @pytest.mark.asyncio
    async def test_fire_global_validates_request(self, test_app: AsyncClient) -> None:
        response = await test_app.post(
            "/api/triggers/fire-global",
            json={},  # missing available_samples
        )
        assert response.status_code == 422


class TestRunGenerateAndTrainBackgroundTask:
    """Test the _run_generate_and_train background task helper directly.

    The background task wrapper must catch any Exception raised by
    generate_and_train so the worker process is never killed by a transient
    failure. Tests call the helper as a plain coroutine for deterministic
    coverage of the success and failure paths.
    """

    @pytest.mark.asyncio
    async def test_logs_completion_on_success(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """On success, the helper should log 'Fire trigger complete' and return."""
        import logging

        from models.domain import BatchStatus, TrainingBatch
        from routers.triggers import _run_generate_and_train

        completed_batch = TrainingBatch(
            schema_id="test_schema",
            status=BatchStatus.COMPLETED,
            minio_path="bucket/path.jsonl",
        )

        mock_service = MagicMock()
        mock_service.generate_and_train = AsyncMock(return_value=completed_batch)

        caplog.set_level(logging.INFO, logger="routers.triggers")
        await _run_generate_and_train(mock_service, "test_schema")

        mock_service.generate_and_train.assert_awaited_once_with("test_schema")
        # Verify info log was emitted (structlog routes through stdlib logging)
        assert any("Fire trigger complete" in rec.getMessage() for rec in caplog.records) or any(
            "Fire trigger complete" in str(rec.msg) for rec in caplog.records
        )

    @pytest.mark.asyncio
    async def test_swallows_exception_and_logs_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """On failure, the helper should swallow the exception and log a warning.

        Exercises the broad-but-deliberate Exception handler in the background
        task wrapper.
        """
        import logging

        from routers.triggers import _run_generate_and_train

        mock_service = MagicMock()
        mock_service.generate_and_train = AsyncMock(
            side_effect=RuntimeError("batch generation imploded")
        )

        caplog.set_level(logging.WARNING, logger="routers.triggers")
        # Should NOT raise
        await _run_generate_and_train(mock_service, "test_schema")

        mock_service.generate_and_train.assert_awaited_once_with("test_schema")
        # Verify warning was emitted with the schema_id context
        assert any("Fire trigger failed" in str(rec.msg) for rec in caplog.records) or any(
            "Fire trigger failed" in rec.getMessage() for rec in caplog.records
        )

    @pytest.mark.asyncio
    async def test_swallows_value_error(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A different exception type (ValueError) should also be swallowed."""
        import logging

        from routers.triggers import _run_generate_and_train

        mock_service = MagicMock()
        mock_service.generate_and_train = AsyncMock(side_effect=ValueError("bad input"))

        caplog.set_level(logging.WARNING, logger="routers.triggers")
        await _run_generate_and_train(mock_service, "another_schema")

        mock_service.generate_and_train.assert_awaited_once_with("another_schema")
