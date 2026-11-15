"""Tests for the telemetry module.

Covers:
- _test_connection: socket-level connectivity checks
- setup_telemetry: OTEL_SDK_DISABLED, idempotent initialization,
  import errors, connection success/failure, endpoint env var,
  FastAPI instrumentation failure, HTTPX instrumentation failure,
  exporter configuration exception
"""

import os
import socket
from unittest.mock import MagicMock, patch

import telemetry


class TestTestConnection:
    """Tests for _test_connection helper."""

    def test_connection_success(self) -> None:
        """Should return True when socket connects successfully."""
        mock_sock = MagicMock()
        with patch.object(socket, "create_connection", return_value=mock_sock) as mock_create:
            result = telemetry._test_connection("localhost", 4318)
            assert result is True
            mock_create.assert_called_once_with(("localhost", 4318), timeout=2.0)
            mock_sock.close.assert_called_once()

    def test_connection_timeout(self) -> None:
        """Should return False on TimeoutError."""
        with patch.object(socket, "create_connection", side_effect=TimeoutError):
            result = telemetry._test_connection("localhost", 4318)
            assert result is False

    def test_connection_refused(self) -> None:
        """Should return False on ConnectionRefusedError."""
        with patch.object(socket, "create_connection", side_effect=ConnectionRefusedError):
            result = telemetry._test_connection("localhost", 4318)
            assert result is False

    def test_connection_os_error(self) -> None:
        """Should return False on generic OSError."""
        with patch.object(socket, "create_connection", side_effect=OSError("network error")):
            result = telemetry._test_connection("localhost", 4318)
            assert result is False

    def test_connection_custom_timeout(self) -> None:
        """Should pass custom timeout to create_connection."""
        mock_sock = MagicMock()
        with patch.object(socket, "create_connection", return_value=mock_sock) as mock_create:
            telemetry._test_connection("myhost", 9090, timeout=5.0)
            mock_create.assert_called_once_with(("myhost", 9090), timeout=5.0)


class TestSetupTelemetry:
    """Tests for setup_telemetry."""

    def setup_method(self) -> None:
        """Reset module state before each test."""
        telemetry._initialized = False

    def teardown_method(self) -> None:
        """Reset module state after each test."""
        telemetry._initialized = False

    def test_otel_disabled(self) -> None:
        """Should return False when OTEL_SDK_DISABLED is 'true'."""
        app = MagicMock()
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}):
            result = telemetry.setup_telemetry(app)
            assert result is False
            assert telemetry._initialized is False

    def test_otel_disabled_case_insensitive(self) -> None:
        """Should handle OTEL_SDK_DISABLED='TRUE' (mixed case)."""
        app = MagicMock()
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "TRUE"}):
            result = telemetry.setup_telemetry(app)
            assert result is False

    def test_otel_disabled_mixed_case(self) -> None:
        """Should handle OTEL_SDK_DISABLED='True' (title case)."""
        app = MagicMock()
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "True"}):
            result = telemetry.setup_telemetry(app)
            assert result is False

    def test_already_initialized(self) -> None:
        """Should return True immediately when already initialized."""
        telemetry._initialized = True
        app = MagicMock()
        result = telemetry.setup_telemetry(app)
        assert result is True

    def test_import_error(self) -> None:
        """Should return False when opentelemetry packages are unavailable."""
        app = MagicMock()
        original_import = (
            __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        )

        def mock_import(name, *args, **kwargs):
            if name.startswith("opentelemetry"):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = telemetry.setup_telemetry(app)
            assert result is False
            assert telemetry._initialized is False

    def _make_otel_modules(
        self,
        *,
        mock_trace: MagicMock | None = None,
        mock_exporter_cls: MagicMock | None = None,
        mock_fastapi_instr: MagicMock | None = None,
        mock_httpx_instr: MagicMock | None = None,
        mock_resource_cls: MagicMock | None = None,
        mock_provider_cls: MagicMock | None = None,
        mock_processor_cls: MagicMock | None = None,
    ) -> dict[str, MagicMock]:
        """Build a sys.modules dict for patching OTEL imports.

        Returns a dict suitable for use with patch.dict("sys.modules", ...).
        Any mock not provided will be auto-created.
        """
        if mock_trace is None:
            mock_trace = MagicMock()
        if mock_exporter_cls is None:
            mock_exporter_cls = MagicMock()
        if mock_fastapi_instr is None:
            mock_fastapi_instr = MagicMock()
        if mock_httpx_instr is None:
            mock_httpx_instr = MagicMock()
        if mock_resource_cls is None:
            mock_resource_cls = MagicMock()
        if mock_provider_cls is None:
            mock_provider_cls = MagicMock()
        if mock_processor_cls is None:
            mock_processor_cls = MagicMock()

        return {
            "opentelemetry": MagicMock(trace=mock_trace),
            "opentelemetry.trace": mock_trace,
            "opentelemetry.exporter.otlp.proto.http.trace_exporter": MagicMock(
                OTLPSpanExporter=mock_exporter_cls,
            ),
            "opentelemetry.instrumentation.fastapi": MagicMock(
                FastAPIInstrumentor=mock_fastapi_instr,
            ),
            "opentelemetry.instrumentation.httpx": MagicMock(
                HTTPXClientInstrumentor=mock_httpx_instr,
            ),
            "opentelemetry.sdk.resources": MagicMock(
                SERVICE_NAME="service.name",
                Resource=mock_resource_cls,
            ),
            "opentelemetry.sdk.trace": MagicMock(
                TracerProvider=mock_provider_cls,
            ),
            "opentelemetry.sdk.trace.export": MagicMock(
                BatchSpanProcessor=mock_processor_cls,
            ),
        }

    def test_connection_success_full_init(self) -> None:
        """Should return True and set _initialized when endpoint is reachable."""
        app = MagicMock()
        mock_trace = MagicMock()
        mock_exporter_cls = MagicMock()
        mock_provider = MagicMock()
        mock_provider_cls = MagicMock(return_value=mock_provider)
        mock_processor = MagicMock()
        mock_processor_cls = MagicMock(return_value=mock_processor)
        mock_resource_cls = MagicMock()

        modules = self._make_otel_modules(
            mock_trace=mock_trace,
            mock_exporter_cls=mock_exporter_cls,
            mock_provider_cls=mock_provider_cls,
            mock_processor_cls=mock_processor_cls,
            mock_resource_cls=mock_resource_cls,
        )

        with (
            patch.dict(os.environ, {"OTEL_SDK_DISABLED": "false"}, clear=False),
            patch.dict("sys.modules", modules),
            patch("telemetry._test_connection", return_value=True),
        ):
            result = telemetry.setup_telemetry(app, service_name="test-service")

        assert result is True
        assert telemetry._initialized is True
        mock_trace.set_tracer_provider.assert_called_once()
        mock_exporter_cls.assert_called_once()
        mock_processor_cls.assert_called_once()
        mock_provider.add_span_processor.assert_called_once_with(mock_processor)

    def test_connection_failure_still_returns_true(self) -> None:
        """Should still return True when endpoint is unreachable (tracing disabled)."""
        app = MagicMock()
        mock_trace = MagicMock()
        mock_provider = MagicMock()
        mock_provider_cls = MagicMock(return_value=mock_provider)

        modules = self._make_otel_modules(
            mock_trace=mock_trace,
            mock_provider_cls=mock_provider_cls,
        )

        with (
            patch.dict(os.environ, {"OTEL_SDK_DISABLED": "false"}, clear=False),
            patch.dict("sys.modules", modules),
            patch("telemetry._test_connection", return_value=False),
        ):
            result = telemetry.setup_telemetry(app)

        assert result is True
        assert telemetry._initialized is True
        # Tracer provider is still set even without exporter
        mock_trace.set_tracer_provider.assert_called_once()
        # add_span_processor should NOT have been called since endpoint is unreachable
        mock_provider.add_span_processor.assert_not_called()

    def test_default_service_name(self) -> None:
        """Should use 'training-data-generator' as default service name."""
        app = MagicMock()
        mock_resource_cls = MagicMock()

        modules = self._make_otel_modules(mock_resource_cls=mock_resource_cls)

        with (
            patch.dict(os.environ, {"OTEL_SDK_DISABLED": "false"}, clear=False),
            patch.dict("sys.modules", modules),
            patch("telemetry._test_connection", return_value=False),
        ):
            telemetry.setup_telemetry(app)

        mock_resource_cls.create.assert_called_once_with(
            {"service.name": "training-data-generator"}
        )

    def test_custom_service_name(self) -> None:
        """Should use a custom service name when provided."""
        app = MagicMock()
        mock_resource_cls = MagicMock()

        modules = self._make_otel_modules(mock_resource_cls=mock_resource_cls)

        with (
            patch.dict(os.environ, {"OTEL_SDK_DISABLED": "false"}, clear=False),
            patch.dict("sys.modules", modules),
            patch("telemetry._test_connection", return_value=False),
        ):
            telemetry.setup_telemetry(app, service_name="my-custom-service")

        mock_resource_cls.create.assert_called_once_with({"service.name": "my-custom-service"})

    def test_custom_endpoint_env_var(self) -> None:
        """Should use TRAINING_DATA_GEN_OTLP_TRACES_ENDPOINT when set."""
        app = MagicMock()
        mock_exporter_cls = MagicMock()

        modules = self._make_otel_modules(mock_exporter_cls=mock_exporter_cls)

        custom_endpoint = "http://custom-host:4318/v1/traces"

        with (
            patch.dict(
                os.environ,
                {
                    "OTEL_SDK_DISABLED": "false",
                    "TRAINING_DATA_GEN_OTLP_TRACES_ENDPOINT": custom_endpoint,
                },
                clear=False,
            ),
            patch.dict("sys.modules", modules),
            patch("telemetry._test_connection", return_value=True),
        ):
            telemetry.setup_telemetry(app)

        mock_exporter_cls.assert_called_once_with(endpoint=custom_endpoint, timeout=5)

    def test_unset_endpoint_disables_tracing(self) -> None:
        """Should disable tracing (return False, no exporter) when env var is unset."""
        app = MagicMock()
        mock_exporter_cls = MagicMock()

        modules = self._make_otel_modules(mock_exporter_cls=mock_exporter_cls)

        env = {"OTEL_SDK_DISABLED": "false"}
        with (
            patch.dict(os.environ, env, clear=False),
            patch.dict("sys.modules", modules),
            patch("telemetry._test_connection", return_value=True),
        ):
            os.environ.pop("TRAINING_DATA_GEN_OTLP_TRACES_ENDPOINT", None)
            result = telemetry.setup_telemetry(app)

        assert result is False
        mock_exporter_cls.assert_not_called()

    def test_fastapi_instrumentation_failure(self) -> None:
        """Should still return True when FastAPI instrumentation fails."""
        app = MagicMock()
        mock_fastapi_instr = MagicMock()
        mock_fastapi_instr.instrument_app = MagicMock(
            side_effect=RuntimeError("instrumentation failed")
        )

        modules = self._make_otel_modules(mock_fastapi_instr=mock_fastapi_instr)

        with (
            patch.dict(os.environ, {"OTEL_SDK_DISABLED": "false"}, clear=False),
            patch.dict("sys.modules", modules),
            patch("telemetry._test_connection", return_value=False),
        ):
            result = telemetry.setup_telemetry(app)

        assert result is True
        assert telemetry._initialized is True

    def test_httpx_instrumentation_failure(self) -> None:
        """Should still return True when HTTPX instrumentation fails."""
        app = MagicMock()
        mock_httpx_instance = MagicMock()
        mock_httpx_instance.instrument = MagicMock(
            side_effect=RuntimeError("httpx instrument failed")
        )
        mock_httpx_instr = MagicMock(return_value=mock_httpx_instance)

        modules = self._make_otel_modules(mock_httpx_instr=mock_httpx_instr)

        with (
            patch.dict(os.environ, {"OTEL_SDK_DISABLED": "false"}, clear=False),
            patch.dict("sys.modules", modules),
            patch("telemetry._test_connection", return_value=False),
        ):
            result = telemetry.setup_telemetry(app)

        assert result is True
        assert telemetry._initialized is True

    def test_exporter_config_exception(self) -> None:
        """Should still return True when exporter config raises a generic exception."""
        app = MagicMock()
        mock_trace = MagicMock()
        mock_provider = MagicMock()
        mock_provider_cls = MagicMock(return_value=mock_provider)

        modules = self._make_otel_modules(
            mock_trace=mock_trace,
            mock_provider_cls=mock_provider_cls,
        )

        with (
            patch.dict(
                os.environ,
                {
                    "OTEL_SDK_DISABLED": "false",
                    "TRAINING_DATA_GEN_OTLP_TRACES_ENDPOINT": "://malformed-url",
                },
                clear=False,
            ),
            patch.dict("sys.modules", modules),
            patch("telemetry._test_connection", side_effect=OSError("parse error")),
        ):
            result = telemetry.setup_telemetry(app)

        assert result is True
        assert telemetry._initialized is True
        # Tracer provider should still be set
        mock_trace.set_tracer_provider.assert_called_once()

    def test_batch_processor_parameters(self) -> None:
        """Should configure BatchSpanProcessor with correct parameters."""
        app = MagicMock()
        mock_exporter_cls = MagicMock()
        mock_exporter = MagicMock()
        mock_exporter_cls.return_value = mock_exporter
        mock_processor_cls = MagicMock()

        modules = self._make_otel_modules(
            mock_exporter_cls=mock_exporter_cls,
            mock_processor_cls=mock_processor_cls,
        )

        with (
            patch.dict(os.environ, {"OTEL_SDK_DISABLED": "false"}, clear=False),
            patch.dict("sys.modules", modules),
            patch("telemetry._test_connection", return_value=True),
        ):
            telemetry.setup_telemetry(app)

        mock_processor_cls.assert_called_once_with(
            mock_exporter,
            max_queue_size=2048,
            schedule_delay_millis=5000,
            max_export_batch_size=512,
            export_timeout_millis=30000,
        )

    def test_idempotent_after_success(self) -> None:
        """Calling setup_telemetry twice should skip on the second call."""
        app = MagicMock()
        mock_trace = MagicMock()

        modules = self._make_otel_modules(mock_trace=mock_trace)

        with (
            patch.dict(os.environ, {"OTEL_SDK_DISABLED": "false"}, clear=False),
            patch.dict("sys.modules", modules),
            patch("telemetry._test_connection", return_value=False),
        ):
            first = telemetry.setup_telemetry(app)
            assert first is True
            assert telemetry._initialized is True

        # Second call should return True immediately without re-initializing
        second = telemetry.setup_telemetry(app)
        assert second is True
        # set_tracer_provider should only have been called once (during first init)
        mock_trace.set_tracer_provider.assert_called_once()

    def test_otel_not_disabled_by_default(self) -> None:
        """Should proceed with initialization when OTEL_SDK_DISABLED is not set."""
        app = MagicMock()
        mock_trace = MagicMock()

        modules = self._make_otel_modules(mock_trace=mock_trace)

        with (
            patch.dict(os.environ, {}, clear=False),
            patch.dict("sys.modules", modules),
            patch("telemetry._test_connection", return_value=False),
        ):
            os.environ.pop("OTEL_SDK_DISABLED", None)
            result = telemetry.setup_telemetry(app)

        assert result is True
        assert telemetry._initialized is True

    def test_otel_sdk_disabled_false_proceeds(self) -> None:
        """Should proceed with initialization when OTEL_SDK_DISABLED is 'false'."""
        app = MagicMock()
        mock_trace = MagicMock()

        modules = self._make_otel_modules(mock_trace=mock_trace)

        with (
            patch.dict(os.environ, {"OTEL_SDK_DISABLED": "false"}, clear=False),
            patch.dict("sys.modules", modules),
            patch("telemetry._test_connection", return_value=False),
        ):
            result = telemetry.setup_telemetry(app)

        assert result is True
        assert telemetry._initialized is True
