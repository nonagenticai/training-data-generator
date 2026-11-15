"""Retraining trigger API endpoints.

Provides endpoints for checking which schemas need retraining and
optionally auto-generating batches for them.
"""

import httpx
from fastapi import APIRouter, BackgroundTasks, Request
from sqlalchemy.exc import SQLAlchemyError

from logging_config import get_logger
from models.domain import (
    FireGlobalTriggerRequest,
    FireTriggerRequest,
    TrainingBatch,
    TriggerCheckResponse,
)
from services.trigger_service import TriggerService

logger = get_logger("routers.triggers")

router = APIRouter(prefix="/api/triggers", tags=["triggers"])


@router.get(
    "/check",
    response_model=TriggerCheckResponse,
    summary="Check retraining triggers",
    description="Evaluate all known schemas against retraining trigger conditions: "
    "initial threshold (>=200 samples), new samples (>=100), quality drop (<80%), "
    "or time elapsed (>14 days).",
)
async def check_triggers(
    http_request: Request,
) -> TriggerCheckResponse:
    """Check which schemas need retraining.

    Args:
        http_request: The incoming HTTP request (for accessing app state).

    Returns:
        TriggerCheckResponse with all trigger evaluations.
    """
    trigger_service = http_request.app.state.trigger_service

    logger.info("Checking retraining triggers")
    triggers = await trigger_service.check_triggers()

    needing_retrain = sum(1 for t in triggers if t.should_retrain)

    logger.info(
        "Trigger check complete",
        total_schemas=len(triggers),
        needing_retrain=needing_retrain,
    )

    return TriggerCheckResponse(
        triggers=triggers,
        schemas_needing_retrain=needing_retrain,
    )


@router.post(
    "/check-and-generate",
    response_model=list[TrainingBatch],
    summary="Check triggers and auto-generate batches",
    description="Check retraining triggers for all schemas and automatically generate "
    "training batches for those that need retraining.",
)
async def check_and_generate(
    http_request: Request,
) -> list[TrainingBatch]:
    """Check triggers and auto-generate batches for schemas needing retraining.

    Args:
        http_request: The incoming HTTP request (for accessing app state).

    Returns:
        List of generated TrainingBatch objects.
    """
    trigger_service = http_request.app.state.trigger_service

    logger.info("Running check-and-generate")
    batches = await trigger_service.check_and_generate()

    logger.info(
        "Check-and-generate complete",
        batches_generated=len(batches),
    )

    return batches


@router.post(
    "/fire",
    status_code=202,
    summary="Fire training trigger for a specific schema",
    description="Accept a training trigger from data-collector when a schema "
    "crosses the training sample threshold. Returns 202 immediately and runs "
    "batch generation + flywheel notification in the background.",
)
async def fire_trigger(
    request: FireTriggerRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    """Accept a training trigger from data-collector.

    Returns 202 immediately and runs batch generation + flywheel
    notification in the background. This prevents the caller's
    HTTP timeout from interfering with long-running batch generation.

    Args:
        request: The fire trigger request with schema_id and available_samples.
        http_request: The incoming HTTP request (for accessing app state).
        background_tasks: FastAPI background task queue.

    Returns:
        Acceptance confirmation with schema_id.
    """
    trigger_service = http_request.app.state.trigger_service

    logger.info(
        "Fire trigger accepted",
        schema_id=request.schema_id,
        available_samples=request.available_samples,
    )

    background_tasks.add_task(_run_generate_and_train, trigger_service, request.schema_id)

    return {"status": "accepted", "schema_id": request.schema_id}


@router.post(
    "/fire-global",
    status_code=202,
    summary="Fire system-wide single-LoRA training trigger",
    description=(
        "Accept a global training trigger from data-collector when the "
        "aggregate validated/unused sample count crosses the global "
        "threshold. Returns 202 immediately and runs mode=global batch "
        "generation + model-trainer job submission in the background. "
        "Idempotent at the data level — model-trainer dedupes/queues; "
        "duplicate fires while a global training is in-flight produce a "
        "second queued job that the worker handles serially."
    ),
)
async def fire_global_trigger(
    request: FireGlobalTriggerRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    """Accept a global-training trigger from data-collector."""
    trigger_service = http_request.app.state.trigger_service

    logger.info(
        "Fire global trigger accepted",
        available_samples=request.available_samples,
    )

    background_tasks.add_task(_run_generate_and_train_global, trigger_service)

    return {"status": "accepted", "scope": "global"}


async def _run_generate_and_train_global(trigger_service: TriggerService) -> None:
    """Background task: build mode=global batch + submit mode=global training job.

    Mirrors :func:`_run_generate_and_train` for the per-schema path.
    """
    try:
        batch = await trigger_service.generate_and_train_global()
        logger.info(
            "Fire global trigger complete",
            batch_id=str(batch.id),
            status=batch.status,
        )
    except (
        httpx.HTTPError,
        SQLAlchemyError,
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
    ):
        logger.warning("Fire global trigger failed", exc_info=True)


async def _run_generate_and_train(trigger_service: TriggerService, schema_id: str) -> None:
    """Background task: generate batch and trigger flywheel.

    This runs detached from the request so any unhandled exception would
    otherwise crash silently in BackgroundTasks. We deliberately catch
    ``Exception`` (the broadest concrete exception class) so a transient
    failure in batch generation, MinIO upload, registry lookup, or flywheel
    notification cannot kill the worker — it gets logged with full traceback
    instead. ``BaseException`` (KeyboardInterrupt, SystemExit, asyncio
    cancellation) is intentionally NOT caught.
    """
    try:
        batch = await trigger_service.generate_and_train(schema_id)
        logger.info(
            "Fire trigger complete",
            schema_id=schema_id,
            batch_id=str(batch.id),
            status=batch.status,
        )
    except (
        httpx.HTTPError,
        SQLAlchemyError,
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
    ):
        logger.warning("Fire trigger failed", schema_id=schema_id, exc_info=True)
