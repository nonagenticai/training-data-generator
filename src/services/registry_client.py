"""HTTP client for the model-registry service.

Fetches routing rules (including JSON schemas) so that training data
can include the same system message that inference-router injects at
inference time.
"""

from __future__ import annotations

from typing import Any

import httpx

from logging_config import get_logger
from settings import Settings

logger = get_logger("registry_client")


class RegistryClient:
    """Async client for the model-registry service."""

    def __init__(self, settings: Settings, http_client: httpx.AsyncClient | None = None) -> None:
        self._base_url = settings.model_registry_url.rstrip("/")
        self._client = http_client or httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(30.0, connect=5.0),
        )
        self._owns_client = http_client is None

    async def close(self) -> None:
        """Close the underlying HTTP client if we own it."""
        if self._owns_client:
            await self._client.aclose()

    async def get_routing_rule(self, schema_id: str) -> dict[str, Any]:
        """Fetch the routing rule for a schema, including its JSON schema.

        Args:
            schema_id: Schema identifier (e.g. "ks-entity-spotter").

        Returns:
            Routing rule dict with keys like ``json_schema``, ``provider_chain``, etc.
            Returns an empty dict if the routing rule is not found.
        """
        try:
            response = await self._client.get(f"/api/routing/{schema_id}")
            if response.status_code == 404:
                logger.warning("routing_rule_not_found", schema_id=schema_id)
                return {}
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError:
            logger.warning("routing_rule_fetch_failed", schema_id=schema_id, exc_info=True)
            return {}
