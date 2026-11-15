"""Retraining trigger evaluation service.

Checks whether schemas need retraining based on four trigger conditions:
1. Initial threshold: >= 200 quality samples for first-time training
2. New samples: >= 100 new samples since last training batch
3. Quality drop: Model quality falls below 80%
4. Time elapsed: 14+ days since last training batch
"""

import math
from datetime import UTC, datetime, timedelta

import httpx

from logging_config import get_logger
from models.domain import BatchGenerateRequest, RetrainingTrigger, TrainingBatch
from services.batch_service import BatchService
from services.collector_client import CollectorClient
from settings import Settings

logger = get_logger("trigger_service")


class TriggerService:
    """Evaluates retraining triggers for all known schemas."""

    def __init__(
        self,
        settings: Settings,
        collector_client: CollectorClient,
        batch_service: BatchService,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._collector = collector_client
        self._batch_service = batch_service
        self._registry_url = settings.model_registry_url.rstrip("/")
        self._http_client = http_client or httpx.AsyncClient(
            base_url=self._registry_url,
            timeout=httpx.Timeout(15.0, connect=5.0),
        )
        self._owns_client = http_client is None

    async def close(self) -> None:
        """Close the underlying HTTP client if owned."""
        if self._owns_client:
            await self._http_client.aclose()

    async def check_triggers(self) -> list[RetrainingTrigger]:
        """Check retraining triggers for all known schemas.

        Evaluates each schema against the four trigger conditions and returns
        a list of trigger results.

        Returns:
            List of RetrainingTrigger objects for each schema.
        """
        schema_ids = await self._collector.list_schema_ids()

        logger.info("Checking retraining triggers", schema_count=len(schema_ids))

        triggers: list[RetrainingTrigger] = []
        for schema_id in schema_ids:
            trigger = await self._check_schema(schema_id)
            triggers.append(trigger)

        needing_retrain = sum(1 for t in triggers if t.should_retrain)
        logger.info(
            "Trigger check complete",
            total_schemas=len(triggers),
            needing_retrain=needing_retrain,
        )

        return triggers

    async def generate_and_train(self, schema_id: str) -> TrainingBatch:
        """Generate a training batch and notify the flywheel for a single schema.

        Called by the fire endpoint when data-collector reports a schema
        has crossed the training sample threshold.

        Args:
            schema_id: The schema to generate training data for.

        Returns:
            The generated TrainingBatch.
        """
        request = BatchGenerateRequest(
            schema_id=schema_id,
            min_samples=self._settings.min_samples,
            train_split=self._settings.train_split,
        )
        batch = await self._batch_service.generate_batch(request)
        if batch.status == "completed" and batch.minio_path:
            await self._notify_flywheel(schema_id, batch)
        return batch

    async def generate_and_train_global(self) -> TrainingBatch:
        """Build the aggregated mode=global batch and submit a mode=global training job.

        Called by the fire-global endpoint when data-collector reports the
        AGGREGATE validated/unused sample count has crossed
        ``DATA_COLLECTOR_GLOBAL_TRAINING_THRESHOLD``. Mirrors
        :meth:`generate_and_train` but for the system-wide single-LoRA path —
        no schema_id, no per-schema flywheel notification.

        Returns:
            The generated TrainingBatch (mode=global).
        """
        request = BatchGenerateRequest(
            mode="global",
            min_samples=self._settings.min_samples,
            train_split=self._settings.train_split,
        )
        batch = await self._batch_service.generate_batch(request)
        if batch.status == "completed" and batch.minio_path:
            await self._submit_global_training_job(batch)
        return batch

    async def _submit_global_training_job(self, batch: TrainingBatch) -> None:
        """Submit a mode=global job to model-trainer. Fire-and-forget.

        Args:
            batch: The completed mode=global TrainingBatch.
        """
        trainer_url = self._settings.model_trainer_url.rstrip("/")
        if not trainer_url:
            logger.warning(
                "model_trainer_url empty — skipping global job submission",
                batch_id=str(batch.id),
            )
            return
        base_model = self._settings.global_base_model
        if not base_model:
            logger.warning(
                "global_base_model empty — skipping global job submission",
                batch_id=str(batch.id),
            )
            return
        # schema_id is the sentinel "__global__" — model-trainer's
        # _register_as_global_adapter derives the real schema list from per-record
        # schema_id fields in the aggregated JSONL, so the job-level schema_id is
        # only a label.
        try:
            resp = await self._http_client.post(
                f"{trainer_url}/api/jobs",
                json={
                    "schema_id": "__global__",
                    "training_batch_id": str(batch.id),
                    "minio_path": batch.minio_path,
                    "mode": "global",
                    "config": {"base_model": base_model},
                },
                timeout=15.0,
            )
            resp.raise_for_status()
            logger.info(
                "global training job submitted",
                batch_id=str(batch.id),
                job=resp.json(),
            )
        except (httpx.HTTPError, OSError) as exc:
            logger.warning(
                "Failed to submit global training job (non-fatal)",
                batch_id=str(batch.id),
                error=str(exc),
            )

    async def _notify_flywheel(self, schema_id: str, batch: TrainingBatch) -> None:
        """Fire-and-forget notification to model-registry flywheel.

        Args:
            schema_id: Schema that was trained.
            batch: The completed training batch.
        """
        flywheel_url = f"{self._registry_url}/api/flywheel/run"
        try:
            resp = await self._http_client.post(
                flywheel_url,
                json={
                    "schema_id": schema_id,
                    "batch_id": str(batch.id),
                    "minio_path": batch.minio_path,
                },
                timeout=10.0,
            )
            resp.raise_for_status()
            logger.info(
                "Flywheel triggered",
                schema_id=schema_id,
                batch_id=str(batch.id),
            )
        except (httpx.HTTPError, OSError) as exc:
            # httpx.HTTPError covers all transport, timeout, and HTTP status
            # errors raised by raise_for_status. OSError catches raw socket
            # errors that may escape mock transports in tests.
            logger.warning(
                "Failed to notify flywheel (non-fatal)",
                schema_id=schema_id,
                error=str(exc),
            )

    async def check_and_generate(self) -> list[TrainingBatch]:
        """Check triggers and auto-generate batches for schemas that need retraining.

        Returns:
            List of generated TrainingBatch objects.
        """
        triggers = await self.check_triggers()
        batches: list[TrainingBatch] = []

        for trigger in triggers:
            if not trigger.should_retrain:
                continue

            logger.info(
                "Auto-generating batch",
                schema_id=trigger.schema_id,
                trigger_reason=trigger.trigger_reason,
                available_samples=trigger.available_samples,
            )

            request = BatchGenerateRequest(
                schema_id=trigger.schema_id,
                min_samples=self._settings.min_samples,
                train_split=self._settings.train_split,
            )

            batch = await self._batch_service.generate_batch(request)
            batches.append(batch)

        return batches

    async def _check_schema(self, schema_id: str) -> RetrainingTrigger:
        """Evaluate retraining triggers for a single schema.

        Args:
            schema_id: The schema to evaluate.

        Returns:
            RetrainingTrigger with the evaluation result.
        """
        # Get available sample count
        available = await self._collector.get_sample_count(
            schema_id=schema_id,
            only_validated=True,
            only_unused=True,
        )

        # Get last training time from batch history
        existing_batches = await self._batch_service.list_batches(schema_id=schema_id)
        completed_times = [b.completed_at for b in existing_batches if b.completed_at is not None]
        last_training_at: datetime | None = None
        if completed_times:
            last_training_at = max(completed_times)

        # Get current quality from model-registry (graceful degradation)
        current_quality = await self._get_quality_score(schema_id)

        # Evaluate triggers in priority order
        trigger_reason = "none"
        should_retrain = False

        if last_training_at is None:
            # Initial training check
            if available >= self._settings.initial_sample_threshold:
                trigger_reason = "initial_threshold"
                should_retrain = True
        else:
            # Subsequent training checks
            if available >= self._settings.subsequent_sample_threshold:
                trigger_reason = "new_samples"
                should_retrain = True
            elif current_quality is not None and current_quality < self._settings.quality_threshold:
                trigger_reason = "quality_drop"
                should_retrain = True
            elif datetime.now(UTC) - last_training_at > timedelta(
                days=self._settings.retrain_interval_days
            ):
                trigger_reason = "time_elapsed"
                should_retrain = True

        # Final viability gate: even if a trigger fires, don't recommend
        # retraining unless we have enough samples to produce a viable batch
        # after the train/validation split.
        if should_retrain:
            min_viable = math.ceil(self._settings.min_samples / self._settings.train_split)
            if available < min_viable:
                logger.info(
                    "Trigger fired but insufficient samples for viable batch",
                    schema_id=schema_id,
                    trigger_reason=trigger_reason,
                    available=available,
                    min_viable=min_viable,
                )
                trigger_reason = "none"
                should_retrain = False

        return RetrainingTrigger(
            schema_id=schema_id,
            trigger_reason=trigger_reason,
            available_samples=available,
            last_training_at=last_training_at,
            current_quality=current_quality,
            should_retrain=should_retrain,
        )

    async def _get_quality_score(self, schema_id: str) -> float | None:
        """Fetch the current quality score from model-registry.

        Falls back gracefully if the registry is unreachable.

        Args:
            schema_id: Schema identifier.

        Returns:
            Quality score float, or None if unavailable.
        """
        try:
            response = await self._http_client.get(
                f"/api/models/{schema_id}/quality",
            )
            if response.status_code == 200:
                data = response.json()
                score = data.get("quality_score")
                if score is not None:
                    return float(score)
            return None
        except (httpx.HTTPError, OSError) as exc:
            # httpx.HTTPError covers transport errors, timeouts, and HTTP status
            # errors. OSError covers raw socket failures that escape before httpx
            # wraps them (e.g. ECONNREFUSED on DNS resolution failure).
            logger.warning(
                "Failed to fetch quality score from model-registry",
                schema_id=schema_id,
                error=str(exc),
                exc_info=True,
            )
            return None
