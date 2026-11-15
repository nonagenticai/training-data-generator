"""Filtering service for inference samples.

Applies multiple quality filters to validated samples before they are
included in training data:
- JSON extraction from markdown code fences
- Response length bounds (with per-schema overrides)
- Error pattern detection
- JSON validity check

Deduplication is handled by data-collector at ingest time via content_hash
unique constraint, so it is NOT repeated here.
"""

import json
import re

from logging_config import get_logger
from models.domain import InferenceSample

logger = get_logger("filter_service")

# Common LLM error/refusal patterns that indicate low-quality training data
_ERROR_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bI cannot\b", re.IGNORECASE),
    re.compile(r"\bI'm sorry\b", re.IGNORECASE),
    re.compile(r"\bAs an AI\b", re.IGNORECASE),
    re.compile(r"\bI'm unable to\b", re.IGNORECASE),
    re.compile(r"\bI am unable to\b", re.IGNORECASE),
    re.compile(r"\bI apologize\b", re.IGNORECASE),
    re.compile(r"\bcannot assist\b", re.IGNORECASE),
    re.compile(r"\bTraceback \(most recent call last\)", re.IGNORECASE),
    re.compile(r"\bInternal Server Error\b", re.IGNORECASE),
]

# Markdown code fence pattern: ```json ... ``` or ``` ... ```
_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*\n(.*?)\n\s*```\s*$",
    re.DOTALL,
)

# Response length bounds
MIN_RESPONSE_LENGTH = 5
MAX_RESPONSE_LENGTH = 10_000


class FilterService:
    """Applies quality filters to inference samples."""

    def __init__(
        self,
        min_length: int = MIN_RESPONSE_LENGTH,
        max_length: int = MAX_RESPONSE_LENGTH,
        error_patterns: list[re.Pattern[str]] | None = None,
        max_length_overrides: dict[str, int] | None = None,
    ) -> None:
        self._min_length = min_length
        self._max_length = max_length
        self._error_patterns = error_patterns if error_patterns is not None else _ERROR_PATTERNS
        self._max_length_overrides = max_length_overrides or {}

    def filter_samples(
        self,
        samples: list[InferenceSample],
        schema_id: str | None = None,
    ) -> tuple[list[InferenceSample], dict[str, int]]:
        """Filter samples for training data quality.

        Applies the following filters in order:
        0. Extract JSON from markdown code fences (cleans response_content)
        1. Response length bounds (too short / too long), with per-schema override
        2. Error pattern detection (LLM refusals, stack traces)
        3. JSON validity check

        Args:
            samples: Raw validated inference samples.
            schema_id: Optional schema identifier for per-schema length overrides.

        Returns:
            Tuple of (passing_samples, filter_reasons) where filter_reasons
            maps reason strings to the count of samples removed.
        """
        reasons: dict[str, int] = {}
        filtered: list[InferenceSample] = []
        cleaned_count = 0

        # Use per-schema max_length if available
        effective_max_length = self._max_length
        if schema_id and schema_id in self._max_length_overrides:
            effective_max_length = self._max_length_overrides[schema_id]

        for sample in samples:
            # 0. Extract JSON from markdown code fences if present
            cleaned = self._extract_json(sample.response_content)
            if cleaned != sample.response_content:
                sample.response_content = cleaned
                cleaned_count += 1

            # 1. Length bounds
            response_len = len(sample.response_content)
            if response_len < self._min_length:
                reasons["too_short"] = reasons.get("too_short", 0) + 1
                continue
            if response_len > effective_max_length:
                reasons["too_long"] = reasons.get("too_long", 0) + 1
                continue

            # 2. Error pattern detection
            if self._has_error_pattern(sample.response_content):
                reasons["error_pattern"] = reasons.get("error_pattern", 0) + 1
                continue

            # 3. JSON validity check
            if not self._is_valid_json(sample.response_content):
                reasons["invalid_json"] = reasons.get("invalid_json", 0) + 1
                continue

            filtered.append(sample)

        logger.info(
            "Filtering complete",
            input_count=len(samples),
            output_count=len(filtered),
            cleaned_code_fences=cleaned_count,
            filter_reasons=reasons,
            effective_max_length=effective_max_length,
        )

        return filtered, reasons

    def _extract_json(self, content: str) -> str:
        """Extract JSON from markdown code fences if present."""
        match = _CODE_FENCE_RE.match(content)
        if match:
            return match.group(1).strip()
        return content.strip()

    def _has_error_pattern(self, content: str) -> bool:
        """Check if response content matches known error/refusal patterns."""
        for pattern in self._error_patterns:
            if pattern.search(content):
                return True
        return False

    def _is_valid_json(self, content: str) -> bool:
        """Check if response content is valid JSON."""
        try:
            json.loads(content)
        except json.JSONDecodeError:
            return False
        return True
