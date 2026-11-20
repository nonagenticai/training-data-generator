"""Training Data Generator service entrypoint.

FastAPI application with async lifespan for managing service dependencies
(HTTP clients, MinIO client). Follows karma_service patterns.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from db import dispose_db, init_db
from logging_config import configure_logging, get_logger, redact_dsn
from models.domain import HealthResponse, LivenessResponse, ReadinessResponse
from routers.batches import router as batches_router
from routers.triggers import router as triggers_router
from services.batch_service import BatchService
from services.collector_client import CollectorClient
from services.filter_service import FilterService
from services.minio_client import MinioClient
from services.registry_client import RegistryClient
from services.trigger_service import TriggerService
from settings import get_settings
from telemetry import setup_telemetry

configure_logging()
logger = get_logger("main")

_settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Manage async service dependencies lifecycle.

    Creates and tears down HTTP clients and service objects.
    """
    settings = get_settings()
    logger.info(
        "Starting training-data-generator",
        port=settings.port,
        environment=settings.environment,
    )

    # Initialize database
    await init_db(settings.database_url)
    logger.info("Database initialised", database_url=redact_dsn(settings.database_url))

    # Initialize clients
    collector_client = CollectorClient(settings)
    minio_client = MinioClient(settings)
    registry_client = RegistryClient(settings)
    # Parse per-schema max_length overrides from settings
    import json as _json

    try:
        max_length_overrides = _json.loads(settings.max_response_length_overrides)
    except ValueError, _json.JSONDecodeError:
        max_length_overrides = {}

    filter_service = FilterService(
        max_length=settings.max_response_length,
        max_length_overrides=max_length_overrides,
    )

    # Initialize services
    batch_service = BatchService(
        collector_client=collector_client,
        filter_service=filter_service,
        minio_client=minio_client,
        registry_client=registry_client,
        settings=settings,
    )
    trigger_service = TriggerService(
        settings=settings,
        collector_client=collector_client,
        batch_service=batch_service,
    )

    # Store on app state for access in route handlers
    app.state.collector_client = collector_client
    app.state.minio_client = minio_client
    app.state.filter_service = filter_service
    app.state.batch_service = batch_service
    app.state.trigger_service = trigger_service

    # Setup telemetry (graceful degradation)
    setup_telemetry(app, service_name=settings.service_name)

    logger.info("Service started successfully")
    yield

    # Shutdown
    logger.info("Shutting down training-data-generator")
    await collector_client.close()
    await registry_client.close()
    await trigger_service.close()
    await dispose_db()
    logger.info("Shutdown complete")


app = FastAPI(
    title="Training Data Generator",
    description=(
        "Generates MLX-compatible training JSONL files from validated inference "
        "samples and uploads them to MinIO for model fine-tuning."
    ),
    version=_settings.version,
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(batches_router)
app.include_router(triggers_router)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Health check",
)
async def health() -> HealthResponse:
    """Basic health check endpoint.

    Returns:
        Health status indicating the service is running.
    """
    return HealthResponse()


@app.get(
    "/health/live",
    response_model=LivenessResponse,
    tags=["health"],
    summary="Liveness probe",
)
async def liveness() -> LivenessResponse:
    """Kubernetes liveness probe. Always returns 200 if the process is alive.

    Returns:
        Liveness status.
    """
    return LivenessResponse()


@app.get(
    "/health/ready",
    response_model=ReadinessResponse,
    tags=["health"],
    summary="Readiness check",
    responses={503: {"model": ReadinessResponse, "description": "Service not ready"}},
)
async def readiness(request: Request) -> ReadinessResponse | JSONResponse:
    """Readiness check that verifies upstream dependencies.

    Tests connectivity to data-collector and MinIO.

    Args:
        request: The incoming HTTP request (for accessing app state).

    Returns:
        Readiness status with per-dependency checks. HTTP 503 if not ready.
    """
    from db import check_db_connectivity

    minio_client: MinioClient = request.app.state.minio_client
    collector_client: CollectorClient = request.app.state.collector_client

    db_ok = await check_db_connectivity()
    minio_ok = await minio_client.check_connectivity()

    # Quick data-collector check. CollectorClient uses httpx, so any
    # transport/HTTP error surfaces as httpx.HTTPError. We also catch
    # OSError defensively (e.g. DNS failures that escape httpx wrapping).
    dc_ok = False
    try:
        await collector_client.list_schema_ids()
        dc_ok = True
    except httpx.HTTPError, OSError:
        logger.debug("health_check_failed", exc_info=True)
        dc_ok = False

    ready = db_ok and minio_ok and dc_ok

    response = ReadinessResponse(
        ready=ready,
        database=db_ok,
        data_collector=dc_ok,
        minio=minio_ok,
    )

    if not ready:
        return JSONResponse(
            status_code=503,
            content=response.model_dump(mode="json"),
        )

    return response


@app.get("/", tags=["health"])
async def root() -> dict[str, str]:
    """Root endpoint with service metadata.

    Returns:
        Service name, version, and useful paths.
    """
    return {
        "service": _settings.service_name,
        "version": _settings.version,
        "docs": "/docs",
        "health": "/health",
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unhandled exceptions.

    Args:
        request: The incoming request.
        exc: The unhandled exception.

    Returns:
        A JSON error response with 500 status.
    """
    from fastapi import HTTPException

    if isinstance(exc, HTTPException):
        raise exc

    logger.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "server_error",
                "code": 500,
            }
        },
    )
