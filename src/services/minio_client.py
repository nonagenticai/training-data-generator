"""MinIO / S3-compatible object storage client.

Handles uploading training JSONL files to MinIO using aioboto3 for
async S3 operations.
"""

import io
import json
from typing import Any

import aioboto3
from botocore.exceptions import BotoCoreError, ClientError

from logging_config import get_logger
from settings import Settings

logger = get_logger("minio_client")


class MinioClient:
    """Async MinIO client using aioboto3 for S3-compatible operations."""

    def __init__(self, settings: Settings) -> None:
        self._endpoint = settings.minio_endpoint
        self._access_key = settings.minio_access_key
        self._secret_key = settings.minio_secret_key
        self._bucket = settings.minio_bucket
        self._use_ssl = settings.minio_use_ssl
        self._session = aioboto3.Session()

    def _get_endpoint_url(self) -> str:
        """Build the full endpoint URL.

        Returns:
            The endpoint URL with protocol prefix.
        """
        protocol = "https" if self._use_ssl else "http"
        if self._endpoint.startswith("http://") or self._endpoint.startswith("https://"):
            return self._endpoint
        return f"{protocol}://{self._endpoint}"

    def _get_client_kwargs(self) -> dict[str, Any]:
        """Build kwargs for the S3 client context manager.

        Returns:
            Dictionary of keyword arguments for aioboto3 client creation.
        """
        return {
            "service_name": "s3",
            "endpoint_url": self._get_endpoint_url(),
            "aws_access_key_id": self._access_key,
            "aws_secret_access_key": self._secret_key,
            "region_name": "us-east-1",
        }

    async def ensure_bucket(self) -> None:
        """Ensure the target bucket exists, creating it if necessary."""
        async with self._session.client(**self._get_client_kwargs()) as s3:
            try:
                await s3.head_bucket(Bucket=self._bucket)
                logger.debug("Bucket exists", bucket=self._bucket)
            except BotoCoreError, ClientError, OSError:
                logger.info("Creating bucket", bucket=self._bucket)
                await s3.create_bucket(Bucket=self._bucket)

    async def upload_jsonl(self, records: list[dict[str, Any]], key: str) -> str:
        """Upload a list of records as a JSONL file to MinIO.

        Each record is serialized as a single JSON line. The resulting file
        is uploaded to the configured bucket at the specified key.

        Args:
            records: List of dictionaries to serialize as JSONL.
            key: S3 object key (path within the bucket).

        Returns:
            The full S3 path (bucket/key).

        Raises:
            Exception: If the upload fails.
        """
        lines = [json.dumps(record, ensure_ascii=False) for record in records]
        body = "\n".join(lines) + "\n" if lines else ""
        data = body.encode("utf-8")

        logger.info(
            "Uploading JSONL to MinIO",
            key=key,
            records=len(records),
            size_bytes=len(data),
        )

        async with self._session.client(**self._get_client_kwargs()) as s3:
            await s3.upload_fileobj(
                io.BytesIO(data),
                self._bucket,
                key,
                ExtraArgs={"ContentType": "application/jsonl"},
            )

        full_path = f"{self._bucket}/{key}"
        logger.info("Upload complete", path=full_path)
        return full_path

    async def list_objects(self, prefix: str) -> list[dict[str, Any]]:
        """List objects under a prefix in the bucket.

        Args:
            prefix: S3 key prefix to list.

        Returns:
            List of object metadata dictionaries.
        """
        objects: list[dict[str, Any]] = []
        async with self._session.client(**self._get_client_kwargs()) as s3:
            paginator = s3.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    objects.append(
                        {
                            "key": obj["Key"],
                            "size": obj["Size"],
                            "last_modified": obj["LastModified"].isoformat()
                            if obj.get("LastModified")
                            else None,
                        }
                    )
        return objects

    async def check_connectivity(self) -> bool:
        """Test connectivity to MinIO.

        Returns:
            True if the connection succeeds, False otherwise.
        """
        try:
            async with self._session.client(**self._get_client_kwargs()) as s3:
                await s3.list_buckets()
            return True
        except (BotoCoreError, ClientError, OSError) as exc:
            logger.warning("MinIO connectivity check failed", error=str(exc))
            return False
