"""Batch generation service.

Orchestrates the full training data pipeline:
1. Fetch validated samples from data-collector
2. Apply quality filters
3. Optionally supplement with reused teacher samples
4. Format to MLX chat JSONL
5. Split into train/valid sets
6. Upload to MinIO
7. Mark samples as used

Batches are persisted to PostgreSQL via SQLAlchemy async sessions.
"""

import hashlib
import json
import math
import random
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import httpx
from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy.exc import SQLAlchemyError

from logging_config import get_logger
from models.domain import (
    BatchGenerateRequest,
    BatchStatus,
    InferenceSample,
    Message,
    TrainingBatch,
)
from services.collector_client import CollectorClient
from services.filter_service import FilterService
from services.minio_client import MinioClient
from services.registry_client import RegistryClient
from settings import Settings, get_settings

logger = get_logger("batch_service")


def _ensure_utc(dt: datetime | None) -> datetime | None:
    """Ensure a datetime is timezone-aware (UTC).

    Guard against naive datetimes that may appear in legacy data.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def _domain_to_orm(batch: TrainingBatch) -> dict[str, Any]:
    """Convert a Pydantic TrainingBatch to a dict of ORM-compatible column values."""
    return {
        "id": str(batch.id),
        "schema_id": batch.schema_id,
        "mode": batch.mode,
        "status": batch.status.value,
        "total_samples": batch.total_samples,
        "train_samples": batch.train_samples,
        "valid_samples": batch.valid_samples,
        "filtered_samples": batch.filtered_samples,
        "filter_reasons": json.dumps(batch.filter_reasons),
        "minio_path": batch.minio_path,
        "train_file": batch.train_file,
        "valid_file": batch.valid_file,
        "source_providers": json.dumps(batch.source_providers),
        "source_schema_counts": json.dumps(batch.source_schema_counts),
        "teacher_samples": batch.teacher_samples,
        "created_at": batch.created_at,
        "completed_at": batch.completed_at,
        "error_message": batch.error_message,
    }


def _orm_to_domain(row: Any) -> TrainingBatch:
    """Convert a SQLAlchemy ORM row to a Pydantic TrainingBatch."""
    # `mode` and `source_schema_counts` were added with the global-batch path;
    # legacy rows written before that migration have them attribute-missing
    # rather than NULL on freshly-created tables, so getattr-with-default is
    # the safe read.
    return TrainingBatch.model_validate(
        {
            "id": UUID(row.id) if isinstance(row.id, str) else row.id,
            "schema_id": row.schema_id,
            "mode": getattr(row, "mode", None) or "per_schema",
            "status": row.status,
            "total_samples": row.total_samples,
            "train_samples": row.train_samples,
            "valid_samples": row.valid_samples,
            "filtered_samples": row.filtered_samples,
            "filter_reasons": json.loads(row.filter_reasons) if row.filter_reasons else {},
            "minio_path": row.minio_path,
            "train_file": row.train_file,
            "valid_file": row.valid_file,
            "source_providers": json.loads(row.source_providers) if row.source_providers else {},
            "source_schema_counts": (json.loads(getattr(row, "source_schema_counts", "") or "{}")),
            "teacher_samples": row.teacher_samples,
            "created_at": _ensure_utc(row.created_at),
            "completed_at": _ensure_utc(row.completed_at),
            "error_message": row.error_message,
        }
    )


def _content_hash(messages: list[Message], response: str) -> str:
    """Compute content hash for dedup across new + teacher samples."""
    payload = [m.model_dump(exclude_none=True) for m in messages]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\0" + response
    return hashlib.sha256(canonical.encode()).hexdigest()


class BatchService:
    """Orchestrates training batch generation with PostgreSQL persistence."""

    def __init__(
        self,
        collector_client: CollectorClient,
        filter_service: FilterService,
        minio_client: MinioClient,
        registry_client: RegistryClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._collector = collector_client
        self._filter = filter_service
        self._minio = minio_client
        self._registry = registry_client
        self._settings = settings or get_settings()

    async def _save_batch(self, batch: TrainingBatch) -> None:
        """Insert or update a batch in the database."""
        from db import get_session
        from models.db_models import TrainingBatchORM

        async with get_session() as session:
            existing = await session.get(TrainingBatchORM, str(batch.id))
            if existing is None:
                orm_obj = TrainingBatchORM(**_domain_to_orm(batch))
                session.add(orm_obj)
            else:
                data = _domain_to_orm(batch)
                for key, value in data.items():
                    setattr(existing, key, value)

    async def generate_batch(self, request: BatchGenerateRequest) -> TrainingBatch:
        """Generate a complete training batch.

        Dispatches on ``request.mode``:
        - ``per_schema`` (default): single-schema batch keyed by ``schema_id``.
        - ``global``: aggregated batch across all schemas with a per-schema
          balance cap, used to feed the single-LoRA training pipeline.

        Args:
            request: Batch generation parameters.

        Returns:
            The completed TrainingBatch with all metadata.

        Raises:
            ValueError: If not enough samples pass filtering.
        """
        if request.mode == "global":
            return await self.build_global_batch(request)

        # mode='per_schema' — schema_id is required (validated by Pydantic).
        assert request.schema_id is not None
        batch = TrainingBatch(schema_id=request.schema_id, mode="per_schema")
        await self._save_batch(batch)

        logger.info(
            "Starting batch generation",
            batch_id=str(batch.id),
            schema_id=request.schema_id,
            min_samples=request.min_samples,
        )

        try:
            # 1. Fetch new unused samples from data-collector
            samples = await self._collector.export_samples(
                schema_id=request.schema_id,
                only_validated=True,
                only_unused=True,
                limit=request.max_samples,
                exclude_providers=request.exclude_providers or None,
            )
            batch.total_samples = len(samples)

            logger.info(
                "Fetched new samples",
                batch_id=str(batch.id),
                total_samples=batch.total_samples,
            )

            # 2. Filter new samples
            filtered, reasons = self._filter.filter_samples(samples, schema_id=request.schema_id)
            batch.filtered_samples = sum(reasons.values())
            batch.filter_reasons = reasons

            logger.info(
                "Filtering complete",
                batch_id=str(batch.id),
                passed=len(filtered),
                filtered_out=batch.filtered_samples,
                reasons=reasons,
            )

            # 3. Supplement with reused teacher samples if enabled
            teacher_reused_count = 0
            effective_max = request.max_samples or 500
            if self._settings.reuse_teacher_samples and len(filtered) < effective_max:
                filtered, teacher_reused_count = await self._supplement_with_teachers(
                    request, filtered
                )
                logger.info(
                    "Teacher sample supplementation",
                    batch_id=str(batch.id),
                    teacher_reused=teacher_reused_count,
                    total_after_supplement=len(filtered),
                )

            # Check that post-split training count meets min_samples.
            # With train_split=0.9 and min_samples=20, we need ceil(20/0.9)=23 total.
            min_total = math.ceil(request.min_samples / request.train_split)
            if len(filtered) < min_total:
                batch.status = BatchStatus.FAILED
                batch.error_message = (
                    f"Only {len(filtered)} samples passed filtering "
                    f"(+{teacher_reused_count} teacher reused), "
                    f"need at least {min_total} total to produce "
                    f"{request.min_samples} training samples after "
                    f"{request.train_split:.0%}/{1 - request.train_split:.0%} split"
                )
                await self._save_batch(batch)
                return batch

            # 4. Count teacher samples (fallback_used=True means teacher model)
            batch.teacher_samples = sum(1 for s in filtered if s.fallback_used)

            # 5. Aggregate source provider counts
            batch.source_providers = self._count_providers(filtered)

            # 6. Fetch system prompt from routing rule so training data
            # includes the same system message the model sees at inference.
            system_prompt = None
            if self._registry:
                routing = await self._registry.get_routing_rule(request.schema_id)
                system_prompt = routing.get("system_prompt")
                if system_prompt:
                    logger.info(
                        "Injecting system prompt from routing rule",
                        schema_id=request.schema_id,
                    )
                else:
                    logger.warning(
                        "No system_prompt in routing rule, training data will lack system message",
                        schema_id=request.schema_id,
                    )

            # 7. Format to MLX chat format
            training_samples = self._format_samples(filtered, system_prompt=system_prompt)

            # 8. Shuffle and split train/valid
            random.shuffle(training_samples)
            split_idx = int(len(training_samples) * request.train_split)
            train = training_samples[:split_idx]
            valid = training_samples[split_idx:]

            batch.train_samples = len(train)
            batch.valid_samples = len(valid)

            # 9. Upload to MinIO
            batch.status = BatchStatus.UPLOADING
            batch_path = f"training-data/{request.schema_id}/batch-{batch.id}/"
            batch.minio_path = batch_path

            await self._minio.ensure_bucket()
            train_key = f"{batch_path}train.jsonl"
            valid_key = f"{batch_path}valid.jsonl"

            await self._minio.upload_jsonl(train, train_key)
            await self._minio.upload_jsonl(valid, valid_key)

            batch.train_file = train_key
            batch.valid_file = valid_key

            logger.info(
                "Uploaded training files",
                batch_id=str(batch.id),
                train_key=train_key,
                valid_key=valid_key,
                train_count=len(train),
                valid_count=len(valid),
            )

            # 10. Mark NEW samples as used in data-collector
            # Only mark unused samples (not reused teacher samples which are already marked)
            new_sample_ids = [s.id for s in filtered if not s.used_in_training]
            if new_sample_ids:
                await self._collector.mark_used(new_sample_ids, str(batch.id))

            # 11. Mark batch as completed
            batch.status = BatchStatus.COMPLETED
            batch.completed_at = datetime.now(UTC)

            logger.info(
                "Batch generation complete",
                batch_id=str(batch.id),
                schema_id=request.schema_id,
                train_samples=batch.train_samples,
                valid_samples=batch.valid_samples,
                teacher_samples=batch.teacher_samples,
                teacher_reused=teacher_reused_count,
            )

        except (
            httpx.HTTPError,
            BotoCoreError,
            ClientError,
            SQLAlchemyError,
            OSError,
            ValueError,
            RuntimeError,
            KeyError,
        ) as exc:
            batch.status = BatchStatus.FAILED
            batch.error_message = str(exc)
            logger.error(
                "Batch generation failed",
                batch_id=str(batch.id),
                error=str(exc),
                exc_info=True,
            )

        await self._save_batch(batch)
        return batch

    async def _supplement_with_teachers(
        self,
        request: BatchGenerateRequest,
        new_samples: list[InferenceSample],
    ) -> tuple[list[InferenceSample], int]:
        """Fetch previously-used teacher samples to supplement the batch.

        Teacher samples (fallback_used=True) are high-quality responses from
        deepseek/claude. With 98-99% fallback rate, these ARE the training data.

        Args:
            request: Batch generation parameters.
            new_samples: Already-filtered new samples.

        Returns:
            Tuple of (merged samples, count of teacher samples reused).
        """
        try:
            teacher_samples = await self._collector.export_samples(
                schema_id=request.schema_id,
                only_validated=True,
                only_unused=False,
                fallback_used=True,
                limit=request.max_samples,
                exclude_providers=request.exclude_providers or None,
            )
        except (httpx.HTTPError, OSError) as exc:
            # httpx.HTTPError covers all transport, timeout, and HTTP status
            # errors from the collector HTTP call. OSError covers raw socket
            # failures that may escape before httpx wraps them.
            logger.warning(
                "Failed to fetch teacher samples, proceeding with new samples only",
                error=str(exc),
                exc_info=True,
            )
            return new_samples, 0

        if not teacher_samples:
            return new_samples, 0

        # Filter teacher samples with same quality filters
        teacher_filtered, teacher_reasons = self._filter.filter_samples(
            teacher_samples, schema_id=request.schema_id
        )

        if teacher_reasons:
            logger.info(
                "Teacher sample filtering",
                passed=len(teacher_filtered),
                reasons=teacher_reasons,
            )

        if not teacher_filtered:
            return new_samples, 0

        # Dedup across new + teacher by content hash
        seen_hashes: set[str] = set()
        for s in new_samples:
            seen_hashes.add(_content_hash(s.messages, s.response_content))

        unique_teachers: list[InferenceSample] = []
        for s in teacher_filtered:
            h = _content_hash(s.messages, s.response_content)
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_teachers.append(s)

        if not unique_teachers:
            return new_samples, 0

        # Cap teacher reuse at max_teacher_ratio of total batch
        max_total = request.max_samples or 500
        new_count = len(new_samples)
        max_teachers = int(max_total * self._settings.max_teacher_ratio)
        if new_count > 0:
            # Ensure teachers don't exceed the ratio of the combined batch
            max_by_ratio = int(new_count / (1 - self._settings.max_teacher_ratio)) - new_count
            teacher_cap = min(max_teachers, max_by_ratio, len(unique_teachers))
        else:
            # No new samples — let teachers fill up to the absolute cap
            teacher_cap = min(max_teachers, len(unique_teachers))
        teacher_cap = max(teacher_cap, 0)

        selected_teachers = unique_teachers[:teacher_cap]

        merged = new_samples + selected_teachers
        return merged, len(selected_teachers)

    @staticmethod
    def _apply_schema_balance_cap(
        samples: list[InferenceSample],
        cap_fraction: float,
        seed: int,
    ) -> tuple[list[InferenceSample], dict[str, int]]:
        """Downsample any schema that exceeds ``cap_fraction`` of the pool.

        For schema ``X`` with count ``c_x`` and other-schemas total ``S``,
        the constraint ``c_x_new / (c_x_new + S) <= cap_fraction`` gives
        ``c_x_new <= cap_fraction * S / (1 - cap_fraction)``. We compute
        that ceiling per-schema in a single pass (only schemas exceeding it
        get downsampled; others are kept whole). This is stable: a single
        pass produces a final pool where every schema satisfies the cap
        relative to the FINAL totals, no iteration required.

        Edge case: if ``cap_fraction >= 1.0``, no cap is applied. If
        ``cap_fraction <= 0.0``, ditto (empty cap is meaningless). A fixed
        seed makes the sampling reproducible across reruns of the same
        input.

        Args:
            samples: Pool of validated samples across all schemas.
            cap_fraction: Maximum fraction (0..1) one schema may contribute.
            seed: PRNG seed for reproducible random sampling.

        Returns:
            Tuple of (capped samples, per-schema counts after capping).
        """
        if not samples or cap_fraction <= 0.0 or cap_fraction >= 1.0:
            counts: dict[str, int] = {}
            for s in samples:
                counts[s.schema_id] = counts.get(s.schema_id, 0) + 1
            return list(samples), counts

        rng = random.Random(seed)
        # Group by schema_id (preserve insertion order for determinism).
        by_schema: dict[str, list[InferenceSample]] = {}
        for s in samples:
            by_schema.setdefault(s.schema_id, []).append(s)

        # Single-pass: for each schema, max-allowed = cap * S_other / (1-cap).
        # S_other uses the ORIGINAL counts of other schemas, which is the
        # stable point — capping schema X cannot make any OTHER schema
        # over-represented because every other schema's count is unchanged
        # and S_other is computed against unchanged peers.
        # NB: schemas already <= their cap are kept whole, which means
        # total_kept can be lower than original total (expected — we are
        # downsampling the dominant ones).
        original_counts = {sid: len(g) for sid, g in by_schema.items()}
        total_original = sum(original_counts.values())
        for schema_id, group in by_schema.items():
            others = total_original - original_counts[schema_id]
            # cap_fraction < 1 by guard above, so denom > 0.
            max_allowed = int(cap_fraction * others / (1.0 - cap_fraction))
            # Floor of zero would erase a schema with 100% representation
            # (i.e. only one schema in the pool); clamp to original count
            # when it's the sole schema (others == 0).
            if others == 0:
                continue
            if len(group) > max_allowed:
                by_schema[schema_id] = rng.sample(group, max_allowed)

        capped: list[InferenceSample] = []
        final_counts: dict[str, int] = {}
        for schema_id, group in by_schema.items():
            capped.extend(group)
            final_counts[schema_id] = len(group)
        # Shuffle so train/valid split doesn't get same-schema clumps.
        rng.shuffle(capped)
        return capped, final_counts

    async def build_global_batch(self, request: BatchGenerateRequest) -> TrainingBatch:
        """Generate an aggregated training batch across all schemas.

        Unlike :py:meth:`generate_batch` (per_schema), this:
        - Pulls validated unused samples from data-collector with NO schema filter.
        - Applies the existing quality filters per-schema (so per-schema
          ``max_response_length`` overrides still apply).
        - Applies a per-schema balance cap (default 30%, configurable via
          ``BATCH_MAX_SCHEMA_FRACTION``) to prevent a single dominant schema
          from skewing the global LoRA's training signal.
        - Uploads the resulting JSONL to MinIO under
          ``training-data/global/<batch_id>.jsonl``.
        - Marks consumed samples in data-collector.

        ``schema_id`` is ignored. ``schema_id`` is stored as the empty string
        on the persisted ``TrainingBatch`` row to denote a global batch.

        Args:
            request: Batch generation parameters (``mode='global'``).

        Returns:
            The completed TrainingBatch with ``mode='global'`` and
            ``source_schema_counts`` populated.
        """
        batch = TrainingBatch(schema_id="", mode="global")
        await self._save_batch(batch)

        logger.info(
            "Starting global batch generation",
            batch_id=str(batch.id),
            min_samples=request.min_samples,
        )

        try:
            # 1. Fetch new unused validated samples across ALL schemas.
            samples = await self._collector.list_validated_samples(
                schema_id=None,
                only_unused=True,
                limit=request.max_samples,
                exclude_providers=request.exclude_providers or None,
            )
            batch.total_samples = len(samples)

            logger.info(
                "Fetched aggregated samples",
                batch_id=str(batch.id),
                total_samples=batch.total_samples,
                source_schemas=len({s.schema_id for s in samples}),
            )

            # 2. Filter per-schema (so per-schema length overrides still apply).
            filtered: list[InferenceSample] = []
            agg_reasons: dict[str, int] = {}
            grouped: dict[str, list[InferenceSample]] = {}
            for s in samples:
                grouped.setdefault(s.schema_id, []).append(s)
            for schema_id, group in grouped.items():
                kept, reasons = self._filter.filter_samples(group, schema_id=schema_id)
                filtered.extend(kept)
                for reason, n in reasons.items():
                    agg_reasons[reason] = agg_reasons.get(reason, 0) + n
            batch.filtered_samples = sum(agg_reasons.values())
            batch.filter_reasons = agg_reasons

            logger.info(
                "Aggregated filtering complete",
                batch_id=str(batch.id),
                passed=len(filtered),
                filtered_out=batch.filtered_samples,
                reasons=agg_reasons,
            )

            # 3. Apply per-schema balance cap.
            cap = self._settings.batch_max_schema_fraction
            # Use the batch UUID's int form as seed for reproducibility within
            # one batch_id while still varying across runs.
            capped, schema_counts = self._apply_schema_balance_cap(
                filtered,
                cap_fraction=cap,
                seed=batch.id.int & 0xFFFFFFFF,
            )
            batch.source_schema_counts = schema_counts

            logger.info(
                "Applied per-schema balance cap",
                batch_id=str(batch.id),
                cap_fraction=cap,
                samples_before=len(filtered),
                samples_after=len(capped),
                schema_counts=schema_counts,
            )

            # 4. Min-samples check (post-cap, post-split).
            min_total = math.ceil(request.min_samples / request.train_split)
            if len(capped) < min_total:
                batch.status = BatchStatus.FAILED
                batch.error_message = (
                    f"Only {len(capped)} samples after global aggregation + "
                    f"per-schema cap (was {len(filtered)} pre-cap), need at "
                    f"least {min_total} total to produce {request.min_samples} "
                    f"training samples after "
                    f"{request.train_split:.0%}/{1 - request.train_split:.0%} split"
                )
                await self._save_batch(batch)
                return batch

            batch.teacher_samples = sum(1 for s in capped if s.fallback_used)
            batch.source_providers = self._count_providers(capped)

            # 5. Format. No system_prompt injection in global mode — the
            # global adapter is schema-agnostic; per-schema system prompts
            # would re-introduce schema coupling.
            training_samples = self._format_samples(capped, system_prompt=None)

            random.shuffle(training_samples)
            split_idx = int(len(training_samples) * request.train_split)
            train = training_samples[:split_idx]
            valid = training_samples[split_idx:]

            batch.train_samples = len(train)
            batch.valid_samples = len(valid)

            # 6. Upload to MinIO under training-data/global/<batch_id>.jsonl.
            # Spec wires a single .jsonl path, but model-trainer needs both
            # train + valid splits, so we keep the same train.jsonl/valid.jsonl
            # convention used by the per-schema path under the global prefix.
            batch.status = BatchStatus.UPLOADING
            batch_path = f"training-data/global/batch-{batch.id}/"
            batch.minio_path = batch_path

            await self._minio.ensure_bucket()
            train_key = f"{batch_path}train.jsonl"
            valid_key = f"{batch_path}valid.jsonl"

            await self._minio.upload_jsonl(train, train_key)
            await self._minio.upload_jsonl(valid, valid_key)

            batch.train_file = train_key
            batch.valid_file = valid_key

            logger.info(
                "Uploaded global training files",
                batch_id=str(batch.id),
                train_key=train_key,
                valid_key=valid_key,
                train_count=len(train),
                valid_count=len(valid),
            )

            # 7. Mark consumed samples (no schema filter; consumed regardless
            # of which schema they came from).
            new_sample_ids = [s.id for s in capped if not s.used_in_training]
            if new_sample_ids:
                await self._collector.mark_used(new_sample_ids, str(batch.id))

            batch.status = BatchStatus.COMPLETED
            batch.completed_at = datetime.now(UTC)

            logger.info(
                "Global batch generation complete",
                batch_id=str(batch.id),
                train_samples=batch.train_samples,
                valid_samples=batch.valid_samples,
                source_schemas=len(schema_counts),
            )

        except (
            httpx.HTTPError,
            BotoCoreError,
            ClientError,
            SQLAlchemyError,
            OSError,
            ValueError,
            RuntimeError,
            KeyError,
        ) as exc:
            batch.status = BatchStatus.FAILED
            batch.error_message = str(exc)
            logger.error(
                "Global batch generation failed",
                batch_id=str(batch.id),
                error=str(exc),
                exc_info=True,
            )

        await self._save_batch(batch)
        return batch

    async def get_batch(self, batch_id: str) -> TrainingBatch | None:
        """Retrieve a batch by ID."""
        from db import get_session
        from models.db_models import TrainingBatchORM

        async with get_session() as session:
            row = await session.get(TrainingBatchORM, batch_id)
            if row is None:
                return None
            return _orm_to_domain(row)

    async def list_batches(self, schema_id: str | None = None) -> list[TrainingBatch]:
        """List all generated batches, optionally filtered by schema_id."""
        from sqlalchemy import select

        from db import get_session
        from models.db_models import TrainingBatchORM

        async with get_session() as session:
            stmt = select(TrainingBatchORM)
            if schema_id:
                stmt = stmt.where(TrainingBatchORM.schema_id == schema_id)
            stmt = stmt.order_by(TrainingBatchORM.created_at.desc())
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [_orm_to_domain(row) for row in rows]

    def _format_samples(
        self,
        samples: list[InferenceSample],
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """Convert inference samples to MLX chat JSONL format."""
        system_msg = {"role": "system", "content": system_prompt} if system_prompt else None

        formatted: list[dict[str, Any]] = []
        for sample in samples:
            messages: list[dict[str, Any]] = []
            # Prepend system message if not already present in the sample
            if system_msg and (not sample.messages or sample.messages[0].role != "system"):
                messages.append(system_msg)
            # exclude_none drops data-collector's null tool_calls/tool_call_id/
            # reasoning_content so they never reach the MLX training file.
            messages.extend(m.model_dump(exclude_none=True) for m in sample.messages)
            messages.append({"role": "assistant", "content": sample.response_content})
            formatted.append({"messages": messages})
        return formatted

    def _count_providers(self, samples: list[InferenceSample]) -> dict[str, int]:
        """Count samples by provider."""
        counts: dict[str, int] = {}
        for sample in samples:
            counts[sample.provider] = counts.get(sample.provider, 0) + 1
        return counts
