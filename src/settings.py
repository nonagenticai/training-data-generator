"""Application settings using Pydantic BaseSettings with env_prefix.

All settings can be overridden via environment variables prefixed with
``TRAINING_DATA_GEN_``.  For example, ``TRAINING_DATA_GEN_PORT=8152``.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Training data generator service configuration.

    All settings can be overridden via environment variables prefixed with
    ``TRAINING_DATA_GEN_``.
    """

    model_config = SettingsConfigDict(
        env_prefix="TRAINING_DATA_GEN_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Database — DSN comes from env (TRAINING_DATA_GEN_DATABASE_URL).
    database_url: str = Field(
        default="",
        description="SQLAlchemy async database URL for batch persistence",
    )

    # Service metadata
    service_name: str = Field(
        default="training-data-generator",
        description="Service name for telemetry",
    )
    port: int = Field(default=8152, description="HTTP port to listen on")
    version: str = Field(default="0.1.0", description="Service version")

    # Upstream services — placeholders; real values come from the deployment env
    data_collector_url: str = Field(
        default="",
        description="URL for data-collector service",
    )
    model_registry_url: str = Field(
        default="",
        description="URL for model-registry service",
    )
    model_trainer_url: str = Field(
        default="",
        description=(
            "URL for model-trainer service. Used by the mode=global event-based "
            "trigger to POST a `mode=global` training job after building the "
            "aggregated batch. Empty disables the global trigger flow."
        ),
    )
    global_base_model: str = Field(
        default="",
        description=(
            "Base model identifier passed in the model-trainer JobCreateRequest "
            "config when firing a `mode=global` training job. Must match the "
            "model vLLM is currently serving on PGX."
        ),
    )

    # MinIO / S3-compatible object storage
    minio_endpoint: str = Field(
        default="",
        description="MinIO endpoint for training data storage",
    )
    minio_access_key: str = Field(default="", description="MinIO access key")
    minio_secret_key: str = Field(default="", description="MinIO secret key")
    minio_bucket: str = Field(
        default="training-data",
        description="MinIO bucket for training data uploads",
    )
    minio_use_ssl: bool = Field(
        default=False,
        description="Whether to use SSL for MinIO connections",
    )

    # Retraining thresholds
    initial_sample_threshold: int = Field(
        default=50,
        description="Minimum quality samples for initial training of a schema",
    )
    subsequent_sample_threshold: int = Field(
        default=30,
        description="New samples needed since last train to trigger retraining",
    )
    quality_threshold: float = Field(
        default=0.80,
        description="Quality score below which retraining is triggered",
    )
    retrain_interval_days: int = Field(
        default=14,
        description="Maximum days between retraining runs",
    )

    # Batch generation
    min_samples: int = Field(
        default=10,
        ge=1,
        description="Minimum samples required to generate a training batch",
    )
    max_response_length: int = Field(
        default=10_000,
        description="Default maximum response length for filtering",
    )
    max_response_length_overrides: str = Field(
        default="{}",
        description="JSON dict of schema_id -> max_length for per-schema overrides",
    )

    # Global-mode (single-LoRA) batch balance cap
    batch_max_schema_fraction: float = Field(
        default=0.30,
        gt=0.0,
        le=1.0,
        description=(
            "Maximum fraction of a mode=global aggregated batch that any "
            "single schema_id may contribute. Schemas above this cap are "
            "downsampled with a fixed seed for reproducibility. Default 0.30 "
            "(30%) prevents one chatty schema from dominating the global "
            "LoRA's training signal."
        ),
    )

    # Teacher sample reuse
    reuse_teacher_samples: bool = Field(
        default=True,
        description="Whether to reuse previously-used teacher samples to supplement batches",
    )
    max_teacher_ratio: float = Field(
        default=0.7,
        ge=0.0,
        lt=1.0,
        description="Max fraction of batch that can be reused teacher samples",
    )

    # Training configuration
    train_split: float = Field(
        default=0.9,
        description="Fraction of samples used for training (rest is validation)",
        ge=0.5,
        le=0.99,
    )

    # Observability
    otlp_traces_endpoint: str = Field(
        default="",
        description="OTLP HTTP endpoint for trace export",
    )
    environment: str = Field(
        default="production",
        description="Deployment environment tag",
    )
    log_level: str = Field(default="INFO", description="Root log level")
    log_format: str = Field(
        default="json",
        description="Log format: 'json' for production, 'console' for local dev",
    )

    _REQUIRED_IN_PRODUCTION: tuple[str, ...] = (
        "database_url",
        "data_collector_url",
        "model_registry_url",
        "minio_endpoint",
        "otlp_traces_endpoint",
    )

    def model_post_init(self, __context: object) -> None:
        """Normalize DB URL and validate required fields in production."""
        if self.database_url.startswith("postgresql+asyncpg://"):
            self.database_url = self.database_url.replace(
                "postgresql+asyncpg://", "postgresql+psycopg://", 1
            )
        elif self.database_url.startswith("postgresql://"):
            self.database_url = self.database_url.replace(
                "postgresql://", "postgresql+psycopg://", 1
            )
        if self.environment == "production":
            missing = [n for n in self._REQUIRED_IN_PRODUCTION if not getattr(self, n)]
            if missing:
                raise RuntimeError(
                    f"training-data-generator settings missing required fields in production: {missing}. "
                    "Set them via env vars on the Deployment (see gitops-production)."
                )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance.

    Returns:
        The application Settings parsed from environment variables.
    """
    return Settings()
