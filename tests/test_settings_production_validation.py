"""Settings refuse to start in production with empty URL placeholders."""

import pytest

from settings import Settings


def _mk(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "database_url": "postgresql+psycopg://u:p@h:6432/tdg",  # pragma: allowlist secret
        "data_collector_url": "http://dc:8151",
        "model_registry_url": "http://mr:8155",
        "minio_endpoint": "minio:9000",
        "otlp_traces_endpoint": "http://alloy:4318/v1/traces",
        "environment": "production",
    }
    base.update(overrides)
    return Settings.model_validate(base)


@pytest.mark.parametrize(
    "empty_field",
    [
        "database_url",
        "data_collector_url",
        "model_registry_url",
        "minio_endpoint",
        "otlp_traces_endpoint",
    ],
)
def test_empty_required_raises(empty_field: str) -> None:
    with pytest.raises(RuntimeError, match=empty_field):
        _mk(**{empty_field: ""})


def test_complete_config_allows_init() -> None:
    _mk()


def test_test_environment_allows_empty() -> None:
    Settings.model_validate({"environment": "test"})
