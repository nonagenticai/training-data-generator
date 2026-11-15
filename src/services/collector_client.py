"""HTTP client for the data-collector service.

Provides methods to fetch validated inference samples and mark them as used
after batch generation.
"""

from uuid import UUID

import httpx

from logging_config import get_logger
from models.domain import InferenceSample
from settings import Settings

logger = get_logger("collector_client")


class CollectorClient:
    """Async client for the data-collector service API."""

    def __init__(self, settings: Settings, http_client: httpx.AsyncClient | None = None) -> None:
        self._base_url = settings.data_collector_url.rstrip("/")
        self._client = http_client or httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(30.0, connect=5.0),
        )
        self._owns_client = http_client is None

    async def close(self) -> None:
        """Close the underlying HTTP client if we own it."""
        if self._owns_client:
            await self._client.aclose()

    async def export_samples(
        self,
        schema_id: str | None,
        only_validated: bool = True,
        only_unused: bool = True,
        limit: int | None = None,
        exclude_providers: list[str] | None = None,
        fallback_used: bool | None = None,
    ) -> list[InferenceSample]:
        """Fetch validated samples from data-collector for training.

        Args:
            schema_id: Schema identifier to fetch samples for. If ``None``,
                no schema filter is sent and the collector returns validated
                samples from every schema (used by mode='global' aggregated
                batches).
            only_validated: Only include validated samples.
            only_unused: Only include samples not yet used in training.
            limit: Maximum number of samples to return.
            exclude_providers: Provider names to exclude (filtered client-side).
            fallback_used: If set, filter by fallback_used flag (True = teacher samples).

        Returns:
            List of validated inference samples.

        Raises:
            httpx.HTTPStatusError: If the data-collector returns an error.
        """
        params: dict[str, str | int | bool] = {
            "only_validated": only_validated,
            "only_unused": only_unused,
        }
        if schema_id is not None:
            params["schema_id"] = schema_id
        if limit is not None:
            params["limit"] = limit
        if fallback_used is not None:
            params["fallback_used"] = fallback_used

        logger.info(
            "Fetching samples from data-collector",
            schema_id=schema_id,
            only_validated=only_validated,
            only_unused=only_unused,
            exclude_providers=exclude_providers,
        )

        response = await self._client.get("/api/samples/export", params=params)
        response.raise_for_status()

        samples = [
            InferenceSample.model_validate_json(line)
            for line in response.text.strip().split("\n")
            if line.strip()
        ]

        # Filter out excluded providers client-side (data-collector only supports
        # single provider filter, not exclude lists)
        if exclude_providers:
            excluded = set(p.lower() for p in exclude_providers)
            before_count = len(samples)
            samples = [s for s in samples if s.provider.lower() not in excluded]
            if before_count != len(samples):
                logger.info(
                    "Excluded provider samples",
                    excluded_providers=exclude_providers,
                    removed=before_count - len(samples),
                    remaining=len(samples),
                )

        logger.info(
            "Fetched samples from data-collector",
            schema_id=schema_id,
            count=len(samples),
        )
        return samples

    async def list_validated_samples(
        self,
        schema_id: str | None = None,
        only_unused: bool = True,
        limit: int | None = None,
        exclude_providers: list[str] | None = None,
    ) -> list[InferenceSample]:
        """Fetch validated samples; ``schema_id=None`` means no schema filter.

        Thin wrapper around :py:meth:`export_samples` for the global-mode
        aggregation path. Always sets ``only_validated=True``.
        """
        return await self.export_samples(
            schema_id=schema_id,
            only_validated=True,
            only_unused=only_unused,
            limit=limit,
            exclude_providers=exclude_providers,
        )

    async def mark_used(self, sample_ids: list[UUID], batch_id: str) -> None:
        """Mark samples as used in a training batch.

        Args:
            sample_ids: List of sample UUIDs to mark as used.
            batch_id: The training batch ID that consumed these samples.

        Raises:
            httpx.HTTPStatusError: If the data-collector returns an error.
        """
        if not sample_ids:
            return

        payload = {
            "sample_ids": [str(sid) for sid in sample_ids],
            "training_batch_id": batch_id,
        }

        logger.info(
            "Marking samples as used",
            count=len(sample_ids),
            batch_id=batch_id,
        )

        response = await self._client.patch("/api/samples/mark-used", json=payload)
        response.raise_for_status()

        logger.info(
            "Marked samples as used",
            count=len(sample_ids),
            batch_id=batch_id,
        )

    async def get_sample_count(
        self,
        schema_id: str,
        only_validated: bool = True,
        only_unused: bool = True,
    ) -> int:
        """Get the count of available samples for a schema.

        Args:
            schema_id: Schema identifier.
            only_validated: Only count validated samples.
            only_unused: Only count unused samples.

        Returns:
            Number of available samples.
        """
        params: dict[str, str | bool] = {
            "schema_id": schema_id,
            "only_validated": only_validated,
            "only_unused": only_unused,
        }

        response = await self._client.get("/api/samples/count", params=params)
        response.raise_for_status()

        data = response.json()
        return int(data.get("count", 0))

    async def list_schema_ids(self) -> list[str]:
        """List all known schema IDs from data-collector.

        Returns:
            List of schema identifier strings.
        """
        response = await self._client.get("/api/schemas")
        response.raise_for_status()

        data = response.json()
        return list(data.get("schema_ids", []))
