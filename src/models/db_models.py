"""SQLAlchemy ORM models for training data generator persistence.

Uses SQLAlchemy 2.0 DeclarativeBase with Mapped type annotations.
"""

import json
from datetime import UTC, datetime

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


class TrainingBatchORM(Base):
    """SQLAlchemy model for the training_batches table.

    Mirrors the fields of the ``TrainingBatch`` Pydantic domain model,
    storing JSON-serialisable dicts as TEXT columns.
    """

    __tablename__ = "training_batches"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    # schema_id is empty for mode='global' batches (no single owning schema).
    schema_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    mode: Mapped[str] = mapped_column(String(20), nullable=False, default="per_schema")
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="generating")
    total_samples: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    train_samples: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    valid_samples: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    filtered_samples: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    filter_reasons: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    minio_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    train_file: Mapped[str | None] = mapped_column(String(512), nullable=True)
    valid_file: Mapped[str | None] = mapped_column(String(512), nullable=True)
    source_providers: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    source_schema_counts: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    teacher_samples: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    def to_filter_reasons_dict(self) -> dict[str, int]:
        """Deserialise the filter_reasons JSON column."""
        return json.loads(self.filter_reasons) if self.filter_reasons else {}

    def to_source_providers_dict(self) -> dict[str, int]:
        """Deserialise the source_providers JSON column."""
        return json.loads(self.source_providers) if self.source_providers else {}

    def to_source_schema_counts_dict(self) -> dict[str, int]:
        """Deserialise the source_schema_counts JSON column (global mode only)."""
        return json.loads(self.source_schema_counts) if self.source_schema_counts else {}
