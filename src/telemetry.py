"""OpenTelemetry setup with graceful degradation.

Tests connection to OTLP endpoint before enabling, falls back gracefully
if Grafana agent is unavailable, and respects OTEL_SDK_DISABLED env var.
Follows karma_service telemetry patterns.
"""

import logging
import os
import socket
from urllib.parse import urlparse

from fastapi import FastAPI

logger = logging.getLogger(__name__)

_initialized: bool = False


def _test_connection(host: str, port: int, timeout: float = 2.0) -> bool:
    """Test if OTLP endpoint is reachable via TCP.

    Args:
        host: Hostname to connect to.
        port: Port number to connect to.
        timeout: Connection timeout in seconds.

    Returns:
        True if connection succeeded, False otherwise.
    """
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return True
    except (TimeoutError, ConnectionRefusedError, OSError):  # fmt: skip
        return False


def setup_telemetry(app: FastAPI, service_name: str = "training-data-generator") -> bool:
    """Initialize OpenTelemetry with graceful degradation.

    Attempts to configure OTEL tracing with an OTLP HTTP exporter. If the
    endpoint is unreachable or packages are missing, tracing is silently
    disabled.

    Args:
        app: FastAPI application instance to instrument.
        service_name: Name to identify this service in traces.

    Returns:
        True if OTEL was successfully initialized, False otherwise.
    """
    global _initialized

    if _initialized:
        return True

    if os.getenv("OTEL_SDK_DISABLED", "false").lower() == "true":
        logger.info("OTEL SDK disabled via environment variable")
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as e:
        logger.warning("OTEL packages not available: %s", e)
        return False

    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)

    otlp_endpoint = os.getenv("TRAINING_DATA_GEN_OTLP_TRACES_ENDPOINT", "")
    if not otlp_endpoint:
        logger.info("TRAINING_DATA_GEN_OTLP_TRACES_ENDPOINT not set; tracing disabled")
        return False

    try:
        parsed = urlparse(otlp_endpoint.replace("/v1/traces", ""))
        host = parsed.hostname or "alloy"
        port = parsed.port or 4318

        if _test_connection(host, port):
            exporter = OTLPSpanExporter(endpoint=otlp_endpoint, timeout=5)
            processor = BatchSpanProcessor(
                exporter,
                max_queue_size=2048,
                schedule_delay_millis=5000,
                max_export_batch_size=512,
                export_timeout_millis=30000,
            )
            provider.add_span_processor(processor)
            logger.info("OTEL exporter configured: %s", otlp_endpoint)
        else:
            logger.warning(
                "OTLP endpoint not reachable: %s:%s, tracing disabled",
                host,
                port,
            )
    except (ValueError, OSError) as e:
        # urlparse / DNS / TLS handshake / exporter constructor errors.
        logger.warning("Failed to configure OTEL exporter: %s", e)

    trace.set_tracer_provider(provider)

    try:
        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI instrumentation enabled")
    except (ImportError, RuntimeError) as e:
        # Version incompatibility, double-instrumentation, or missing deps.
        logger.warning("FastAPI instrumentation failed: %s", e)

    try:
        HTTPXClientInstrumentor().instrument()
        logger.info("HTTPX instrumentation enabled")
    except (ImportError, RuntimeError) as e:
        # Version incompatibility, double-instrumentation, or missing deps.
        logger.warning("HTTPX instrumentation failed: %s", e)

    _initialized = True
    return True
