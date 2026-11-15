"""Domain models for training data generation.

All models use Pydantic v2 BaseModel for strict type safety and validation.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID, uuid7

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BatchStatus(StrEnum):
    """Status of a training batch generation."""

    GENERATING = "generating"
    UPLOADING = "uploading"
    COMPLETED = "completed"
    FAILED = "failed"


class Message(BaseModel):
    """A single conversation turn as emitted by data-collector.

    data-collector emits OpenAI-shaped messages in which ``tool_calls``,
    ``tool_call_id`` and ``reasoning_content`` are always present and are
    ``null`` on every non-tool turn. The previous ``list[dict[str, str]]``
    annotation rejected those nulls, so ``model_validate_json`` raised on the
    first sample of every export.

    ``extra="allow"`` is deliberate: this model mirrors a payload owned by
    another service, and a field added upstream must not fail a whole export
    again. Unknown fields are carried through to the MLX output as-is.
    """

    model_config = ConfigDict(extra="allow")

    role: str
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    reasoning_content: str | None = None


class InferenceSample(BaseModel):
    """A validated inference sample from data-collector.

    Represents a single inference interaction that has been validated
    and is eligible for inclusion in training data.
    """

    id: UUID
    schema_id: str
    messages: list[Message] = Field(
        description="Conversation messages (system + user turns)",
    )
    response_content: str = Field(
        description="The assistant response to use as training label",
    )
    provider: str = Field(
        default="unknown",
        description="Which inference provider generated this response",
    )
    fallback_used: bool = Field(
        default=False,
        description="Whether this sample came from a fallback provider (teacher model pattern)",
    )
    used_in_training: bool = Field(
        default=False,
        description="Whether this sample has already been used in a training batch",
    )
    validated_at: datetime | None = None
    quality_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Quality score assigned during validation",
    )


class TrainingBatch(BaseModel):
    """A generated training data batch.

    Tracks the full lifecycle of batch generation from sample collection
    through filtering, formatting, and MinIO upload.
    """

    id: UUID = Field(default_factory=uuid7)
    schema_id: str
    status: BatchStatus = BatchStatus.GENERATING
    total_samples: int = Field(
        default=0,
        description="Total samples fetched from data-collector before filtering",
    )
    train_samples: int = Field(
        default=0,
        description="Number of samples in the training split",
    )
    valid_samples: int = Field(
        default=0,
        description="Number of samples in the validation split",
    )
    filtered_samples: int = Field(
        default=0,
        description="Number of samples that were filtered out",
    )
    filter_reasons: dict[str, int] = Field(
        default_factory=dict,
        description="Mapping of filter reason to count of removed samples",
    )
    minio_path: str | None = Field(
        default=None,
        description="MinIO object path prefix, e.g. training-data/schema/batch-uuid/",
    )
    train_file: str | None = Field(
        default=None,
        description="MinIO key for the train.jsonl file",
    )
    valid_file: str | None = Field(
        default=None,
        description="MinIO key for the valid.jsonl file",
    )
    source_providers: dict[str, int] = Field(
        default_factory=dict,
        description="Mapping of provider name to sample count",
    )
    source_schema_counts: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "For mode=global batches, mapping of schema_id to its final "
            "sample contribution count after the per-schema balance cap."
        ),
    )
    mode: Literal["per_schema", "global"] = Field(
        default="per_schema",
        description="Batch generation mode: per_schema (default) or global (aggregated single-LoRA)",
    )
    teacher_samples: int = Field(
        default=0,
        description="Number of samples from fallback (teacher model pattern)",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    error_message: str | None = None


class BatchGenerateRequest(BaseModel):
    """Request to generate a training batch."""

    mode: Literal["per_schema", "global"] = Field(
        default="per_schema",
        description=(
            "Batch generation mode. 'per_schema' (default) generates a "
            "single-schema batch keyed by schema_id. 'global' aggregates "
            "validated samples across all schemas with a per-schema balance "
            "cap to feed the single-LoRA training pipeline; schema_id is "
            "ignored when mode='global'."
        ),
    )
    schema_id: str | None = Field(
        default=None,
        description=(
            "Schema identifier to generate training data for. Required when "
            "mode='per_schema'; ignored when mode='global'."
        ),
    )
    min_samples: int = Field(
        default=10,
        ge=1,
        description="Minimum number of samples required to generate a batch",
    )
    max_samples: int | None = Field(
        default=None,
        description="Maximum number of samples to include (None = unlimited)",
    )
    train_split: float = Field(
        default=0.9,
        ge=0.5,
        le=0.99,
        description="Fraction of samples used for training",
    )
    exclude_providers: list[str] = Field(
        default=[],
        description="Provider names to exclude from training data (prevents model collapse)",
    )

    @model_validator(mode="after")
    def _check_schema_id_required_for_per_schema(self) -> BatchGenerateRequest:
        """schema_id is required when mode='per_schema'."""
        if self.mode == "per_schema" and not self.schema_id:
            raise ValueError("schema_id is required when mode='per_schema'")
        return self


class RetrainingTrigger(BaseModel):
    """Result of checking whether a schema needs retraining."""

    schema_id: str
    trigger_reason: str = Field(
        description="Why retraining was triggered: initial_threshold, new_samples, quality_drop, time_elapsed",
    )
    available_samples: int = Field(
        description="Number of unused validated samples available",
    )
    last_training_at: datetime | None = Field(
        default=None,
        description="When the last training batch was generated",
    )
    current_quality: float | None = Field(
        default=None,
        description="Current model quality score from model-registry eval",
    )
    should_retrain: bool = Field(
        description="Whether retraining is recommended",
    )


class TrainingSample(BaseModel):
    """MLX-compatible training format.

    Each sample is a chat conversation in the format expected by mlx_lm
    fine-tuning scripts.
    """

    messages: list[dict[str, str]] = Field(
        description="Chat messages in [{role, content}] format",
    )


class FireTriggerRequest(BaseModel):
    """Request from data-collector when a schema crosses the training threshold."""

    schema_id: str = Field(description="Schema that crossed the threshold")
    available_samples: int = Field(
        description="Number of validated unused samples available",
    )


class FireGlobalTriggerRequest(BaseModel):
    """Request from data-collector when the AGGREGATE validated/unused count
    crosses the system-wide single-LoRA training threshold."""

    available_samples: int = Field(
        description="Aggregate validated unused samples across all schemas",
    )


class BatchListResponse(BaseModel):
    """Response containing a list of training batches."""

    batches: list[TrainingBatch]
    total: int


class TriggerCheckResponse(BaseModel):
    """Response containing retraining trigger check results."""

    triggers: list[RetrainingTrigger]
    schemas_needing_retrain: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    service: str = "training-data-generator"
    version: str = "0.1.0"


class LivenessResponse(BaseModel):
    """Liveness probe response."""

    status: str = "alive"


class ReadinessResponse(BaseModel):
    """Readiness check response."""

    ready: bool
    database: bool = False
    data_collector: bool = False
    minio: bool = False
