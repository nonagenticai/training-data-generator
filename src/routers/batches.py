"""Batch generation API endpoints.

Provides endpoints for generating training data batches and querying
their status and details.
"""

from fastapi import APIRouter, HTTPException, Request, status

from logging_config import get_logger
from models.domain import BatchGenerateRequest, BatchListResponse, TrainingBatch

logger = get_logger("routers.batches")

router = APIRouter(prefix="/api/batches", tags=["batches"])


@router.post(
    "/generate",
    response_model=TrainingBatch,
    status_code=status.HTTP_201_CREATED,
    summary="Generate a training batch",
    description="Generate a training batch for a given schema_id by fetching validated samples, "
    "filtering, formatting to MLX JSONL, and uploading to MinIO.",
)
async def generate_batch(
    request: BatchGenerateRequest,
    http_request: Request,
) -> TrainingBatch:
    """Generate a new training batch for the specified schema.

    Args:
        request: Batch generation parameters.
        http_request: The incoming HTTP request (for accessing app state).

    Returns:
        The generated TrainingBatch with full metadata.
    """
    batch_service = http_request.app.state.batch_service

    logger.info(
        "Batch generation requested",
        mode=request.mode,
        schema_id=request.schema_id,
        min_samples=request.min_samples,
        max_samples=request.max_samples,
        train_split=request.train_split,
    )

    batch = await batch_service.generate_batch(request)

    if batch.status == "failed":
        logger.warning(
            "Batch generation failed",
            batch_id=str(batch.id),
            error=batch.error_message,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=batch.error_message or "Batch generation failed",
        )

    return batch


@router.get(
    "",
    response_model=BatchListResponse,
    summary="List generated batches",
    description="List all generated training batches, optionally filtered by schema_id.",
)
async def list_batches(
    http_request: Request,
    schema_id: str | None = None,
) -> BatchListResponse:
    """List all generated batches.

    Args:
        http_request: The incoming HTTP request (for accessing app state).
        schema_id: Optional filter by schema identifier.

    Returns:
        List of TrainingBatch objects.
    """
    batch_service = http_request.app.state.batch_service
    batches = await batch_service.list_batches(schema_id=schema_id)

    return BatchListResponse(
        batches=batches,
        total=len(batches),
    )


@router.get(
    "/{batch_id}",
    response_model=TrainingBatch,
    summary="Get batch details",
    description="Retrieve details for a specific training batch by its ID.",
)
async def get_batch(
    batch_id: str,
    http_request: Request,
) -> TrainingBatch:
    """Get details for a specific batch.

    Args:
        batch_id: The UUID of the batch to retrieve.
        http_request: The incoming HTTP request (for accessing app state).

    Returns:
        The TrainingBatch if found.

    Raises:
        HTTPException: 404 if batch not found.
    """
    batch_service = http_request.app.state.batch_service
    batch = await batch_service.get_batch(batch_id)

    if batch is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch {batch_id} not found",
        )

    return batch
