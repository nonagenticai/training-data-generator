"""Structured logging with OpenTelemetry trace correlation.

Provides JSON-formatted structured logs in production and human-readable
console output in development, with optional trace/span ID injection
from OpenTelemetry.
"""

import logging
import re
import sys
import types
from typing import TYPE_CHECKING, Any

import structlog
from structlog.typing import EventDict

from settings import get_settings

# Check if OTEL is available at import time
OTEL_AVAILABLE = False
if TYPE_CHECKING:
    trace: types.ModuleType | None = None
else:
    try:
        from opentelemetry import trace

        OTEL_AVAILABLE = True
    except ImportError:
        trace = None


def add_trace_context(
    logger: Any,
    _method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Inject OpenTelemetry trace context into structured logs.

    Args:
        logger: The wrapped logger object.
        _method_name: The name of the log method called.
        event_dict: The event dictionary to enrich.

    Returns:
        The enriched event dictionary with trace_id and span_id if available.
    """
    if OTEL_AVAILABLE and trace:
        span = trace.get_current_span()
        if span and span.is_recording():
            ctx = span.get_span_context()
            if ctx.is_valid:
                event_dict["trace_id"] = format(ctx.trace_id, "032x")
                event_dict["span_id"] = format(ctx.span_id, "016x")
    return event_dict


def add_service_context(
    logger: Any,
    _method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Add service metadata to every log entry.

    Args:
        logger: The wrapped logger object.
        _method_name: The name of the log method called.
        event_dict: The event dictionary to enrich.

    Returns:
        The enriched event dictionary with service and environment fields.
    """
    settings = get_settings()
    event_dict["service"] = settings.service_name
    event_dict["environment"] = settings.environment
    return event_dict


_configured = False


def configure_logging() -> None:
    """Configure structured JSON logging with trace context.

    Reads LOG_FORMAT and LOG_LEVEL from settings to determine output style.
    Idempotent: calling multiple times has no effect after the first call.
    """
    global _configured

    if _configured:
        return

    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    if settings.log_format == "json":
        processors: list[structlog.typing.Processor] = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            add_service_context,
            add_trace_context,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            add_service_context,
            add_trace_context,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=log_level)

    # Suppress noisy loggers
    for logger_name in (
        "httpx",
        "httpcore",
        "uvicorn.access",
        "urllib3",
        "botocore",
        "aiobotocore",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a module-level structured logger.

    Args:
        name: Logger name. Defaults to 'training_data_generator'.

    Returns:
        A configured structlog BoundLogger instance.
    """
    configure_logging()
    logger: Any = structlog.get_logger(name or "training_data_generator")
    return logger


def redact_dsn(dsn: str) -> str:
    """Strip the password out of a database URL before it is logged.

    ``settings.database_url`` is a full libpq DSN whose netloc carries the
    credential as ``user`` then ``pass``, separated by a colon, ahead of the
    ``@``. Logging it verbatim put the live password into every pod's stdout —
    and therefore into Loki — on every single startup. Measured on the running
    strato pod 2026-08-18.

    The example is spelled out in prose rather than shown as a literal URL on
    purpose: a literal reads as ``Basic Auth Credentials`` to detect-secrets and
    fails the repo's secret gate. Suppressing that with an allowlist pragma
    would be the wrong trade — the scanner is right about the SHAPE, so the
    honest fix is not to write the shape here.

    Only the credential is removed; user, host, port and database are kept,
    because those are the fields you actually need when diagnosing a connection
    failure. A DSN with no credential is returned unchanged, so this is safe to
    apply unconditionally.
    """
    return re.sub(r"(?<=://)([^:/@]*):([^@]*)(?=@)", r"\1:***", dsn)
