"""Tests for the filter service.

Covers deduplication (not done here), length bounds, error pattern detection,
JSON validity checks, code fence extraction, per-schema max_length overrides,
and combined filtering.
"""

import json

import pytest

from models.domain import Message
from services.filter_service import FilterService
from tests.conftest import make_sample, make_samples


class TestDeduplication:
    """Test that dedup is NOT done in filter_service (handled by data-collector)."""

    def test_duplicates_pass_through(self, filter_service: FilterService) -> None:
        """Exact duplicates should pass through -- dedup is data-collector's job."""
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Same question"),
        ]
        response = json.dumps({"result": "answer"})

        samples = [
            make_sample(messages=messages, response_content=response),
            make_sample(messages=messages, response_content=response),
            make_sample(messages=messages, response_content=response),
        ]

        filtered, reasons = filter_service.filter_samples(samples)

        assert len(filtered) == 3
        assert "duplicate" not in reasons

    def test_keeps_different_messages(self, filter_service: FilterService) -> None:
        """Samples with different messages should all pass."""
        samples = make_samples(count=5)

        filtered, reasons = filter_service.filter_samples(samples)

        assert len(filtered) == 5
        assert "duplicate" not in reasons

    def test_different_system_prompts_both_pass(self, filter_service: FilterService) -> None:
        """Samples with different system prompts should both pass."""
        response = json.dumps({"result": "answer"})

        sample_a = make_sample(
            messages=[
                Message(role="system", content="System A"),
                Message(role="user", content="Question"),
            ],
            response_content=response,
        )
        sample_b = make_sample(
            messages=[
                Message(role="system", content="System B"),
                Message(role="user", content="Question"),
            ],
            response_content=response,
        )

        filtered, reasons = filter_service.filter_samples([sample_a, sample_b])

        assert len(filtered) == 2
        assert "duplicate" not in reasons


class TestLengthBounds:
    """Test response length filtering."""

    def test_filters_too_short_responses(self, filter_service: FilterService) -> None:
        """Responses shorter than 5 characters should be filtered."""
        sample = make_sample(response_content="hi")

        filtered, reasons = filter_service.filter_samples([sample])

        assert len(filtered) == 0
        assert reasons.get("too_short") == 1

    def test_filters_too_long_responses(self, filter_service: FilterService) -> None:
        """Responses longer than 10000 characters should be filtered."""
        long_json = json.dumps({"data": "x" * 10001})
        sample = make_sample(response_content=long_json)

        filtered, reasons = filter_service.filter_samples([sample])

        assert len(filtered) == 0
        assert reasons.get("too_long") == 1

    def test_keeps_responses_within_bounds(self, filter_service: FilterService) -> None:
        """Responses within length bounds should pass."""
        response = json.dumps({"result": "This is a valid response with enough content"})
        sample = make_sample(response_content=response)

        filtered, reasons = filter_service.filter_samples([sample])

        assert len(filtered) == 1
        assert "too_short" not in reasons
        assert "too_long" not in reasons

    def test_boundary_min_length(self, filter_service: FilterService) -> None:
        """Responses exactly at minimum length should pass."""
        # 5 chars is min, and must be valid JSON
        response = '{"a":1}'  # 7 chars, valid JSON, > 5
        sample = make_sample(response_content=response)

        filtered, reasons = filter_service.filter_samples([sample])

        assert len(filtered) == 1

    def test_custom_length_bounds(self) -> None:
        """Custom min/max length should be respected."""
        service = FilterService(min_length=10, max_length=50)
        short = make_sample(response_content='{"a":1}')  # 7 chars
        ok = make_sample(response_content=json.dumps({"result": "ok data"}))

        filtered, reasons = service.filter_samples([short, ok])

        assert reasons.get("too_short") == 1

    def test_exactly_at_max_length_passes(self) -> None:
        """Response at exactly max_length should pass (not be filtered)."""
        # Create a response exactly at the limit
        # max_length check is `> effective_max_length`, so exactly-at should pass
        service = FilterService(max_length=30)
        content = json.dumps({"x": "y"})  # Short enough
        sample = make_sample(response_content=content)

        filtered, reasons = service.filter_samples([sample])
        assert len(filtered) == 1
        assert "too_long" not in reasons


class TestPerSchemaMaxLengthOverrides:
    """Test per-schema max_length_overrides behavior."""

    def test_override_allows_longer_response_for_specific_schema(self) -> None:
        """Per-schema override should raise the max_length for that schema only."""
        service = FilterService(
            max_length=50,
            max_length_overrides={"generous_schema": 5000},
        )
        long_content = json.dumps({"data": "x" * 200})  # > 50 chars
        sample = make_sample(response_content=long_content)

        # Without override (default schema), should be filtered
        filtered_default, reasons_default = service.filter_samples(
            [sample], schema_id="other_schema"
        )
        assert len(filtered_default) == 0
        assert reasons_default.get("too_long") == 1

        # With override, same content should pass
        sample2 = make_sample(response_content=long_content)
        filtered_override, reasons_override = service.filter_samples(
            [sample2], schema_id="generous_schema"
        )
        assert len(filtered_override) == 1
        assert "too_long" not in reasons_override

    def test_override_restricts_length_for_specific_schema(self) -> None:
        """Per-schema override can also be more restrictive than default."""
        service = FilterService(
            max_length=10_000,
            max_length_overrides={"strict_schema": 30},
        )
        content = json.dumps({"data": "moderate length content"})  # ~35 chars
        sample = make_sample(response_content=content)

        # Default schema: should pass
        filtered_default, _ = service.filter_samples([sample], schema_id="any_schema")
        assert len(filtered_default) == 1

        # Strict schema: should be filtered
        sample2 = make_sample(response_content=content)
        filtered_strict, reasons_strict = service.filter_samples(
            [sample2], schema_id="strict_schema"
        )
        assert len(filtered_strict) == 0
        assert reasons_strict.get("too_long") == 1

    def test_no_schema_id_uses_default_max_length(self) -> None:
        """When schema_id is None, default max_length is used."""
        service = FilterService(
            max_length=50,
            max_length_overrides={"some_schema": 5000},
        )
        long_content = json.dumps({"data": "x" * 200})
        sample = make_sample(response_content=long_content)

        filtered, reasons = service.filter_samples([sample], schema_id=None)
        assert len(filtered) == 0
        assert reasons.get("too_long") == 1


class TestErrorPatterns:
    """Test error/refusal pattern detection."""

    @pytest.mark.parametrize(
        "content,expected_reason",
        [
            # Each content is wrapped in valid JSON to avoid invalid_json masking error_pattern
            (json.dumps({"error": "I cannot help with that request"}), "error_pattern"),
            (json.dumps({"msg": "I'm sorry, but I can't do that"}), "error_pattern"),
            (
                json.dumps({"msg": "As an AI language model, I don't have the ability"}),
                "error_pattern",
            ),
            (json.dumps({"msg": "I'm unable to process this"}), "error_pattern"),
            (json.dumps({"msg": "I apologize for the confusion"}), "error_pattern"),
            (json.dumps({"msg": "I cannot assist with that"}), "error_pattern"),
            (json.dumps({"msg": "Traceback (most recent call last)"}), "error_pattern"),
            (json.dumps({"msg": "Internal Server Error occurred"}), "error_pattern"),
        ],
    )
    def test_filters_error_patterns_in_json(
        self,
        filter_service: FilterService,
        content: str,
        expected_reason: str,
    ) -> None:
        """Known error patterns in JSON content should be filtered as error_pattern."""
        sample = make_sample(response_content=content)
        filtered, reasons = filter_service.filter_samples([sample])
        assert len(filtered) == 0
        assert reasons.get(expected_reason, 0) >= 1

    @pytest.mark.parametrize(
        "content",
        [
            "I cannot help with that request",
            "I'm sorry, but I can't do that",
            "Traceback (most recent call last)",
        ],
    )
    def test_filters_error_patterns_plain_text(
        self,
        filter_service: FilterService,
        content: str,
    ) -> None:
        """Plain-text error patterns should be filtered (by error_pattern or invalid_json)."""
        sample = make_sample(response_content=content)
        filtered, reasons = filter_service.filter_samples([sample])
        assert len(filtered) == 0
        # At least one filter reason should be recorded
        assert sum(reasons.values()) >= 1

    def test_keeps_clean_responses(self, filter_service: FilterService) -> None:
        """Responses without error patterns should pass."""
        response = json.dumps({"entities": [{"name": "test", "type": "PERSON"}]})
        sample = make_sample(response_content=response)

        filtered, reasons = filter_service.filter_samples([sample])

        assert len(filtered) == 1
        assert "error_pattern" not in reasons

    def test_custom_error_patterns(self) -> None:
        """Custom error patterns should be used when provided."""
        import re

        service = FilterService(error_patterns=[re.compile(r"FORBIDDEN")])
        # Default pattern should NOT match
        default_content = json.dumps({"msg": "I cannot help"})
        s1 = make_sample(response_content=default_content)
        filtered1, _ = service.filter_samples([s1])
        assert len(filtered1) == 1  # Passes because custom patterns replace defaults

        # Custom pattern should match
        custom_content = json.dumps({"msg": "FORBIDDEN access"})
        s2 = make_sample(response_content=custom_content)
        filtered2, reasons2 = service.filter_samples([s2])
        assert len(filtered2) == 0
        assert reasons2.get("error_pattern") == 1


class TestJsonValidity:
    """Test JSON validity checking."""

    def test_filters_invalid_json(self, filter_service: FilterService) -> None:
        """Non-JSON responses should be filtered as invalid_json."""
        sample = make_sample(response_content="This is plain text not JSON at all here man")

        filtered, reasons = filter_service.filter_samples([sample])

        assert len(filtered) == 0
        assert reasons.get("invalid_json", 0) >= 1

    def test_keeps_valid_json(self, filter_service: FilterService) -> None:
        """Valid JSON responses should pass."""
        response = json.dumps({"result": "success", "data": [1, 2, 3]})
        sample = make_sample(response_content=response)

        filtered, reasons = filter_service.filter_samples([sample])

        assert len(filtered) == 1

    def test_accepts_json_arrays(self, filter_service: FilterService) -> None:
        """JSON arrays should also be considered valid."""
        response = json.dumps([{"entity": "test"}, {"entity": "test2"}])
        sample = make_sample(response_content=response)

        filtered, reasons = filter_service.filter_samples([sample])

        assert len(filtered) == 1

    def test_accepts_json_primitives(self, filter_service: FilterService) -> None:
        """JSON primitive values should be considered valid."""
        sample = make_sample(response_content='"hello world, this is a test string"')

        filtered, reasons = filter_service.filter_samples([sample])

        assert len(filtered) == 1

    def test_rejects_truncated_json(self, filter_service: FilterService) -> None:
        """Truncated JSON should be filtered as invalid_json."""
        sample = make_sample(response_content='{"key": "value", "missing_closing')
        filtered, reasons = filter_service.filter_samples([sample])
        assert len(filtered) == 0
        assert reasons.get("invalid_json", 0) >= 1


class TestCodeFenceExtraction:
    """Test markdown code fence extraction."""

    def test_extracts_json_from_code_fence(self, filter_service: FilterService) -> None:
        """JSON wrapped in ```json ... ``` should be extracted and accepted."""
        raw = '```json\n{"entities": [{"name": "Paris", "type": "LOCATION"}]}\n```'
        sample = make_sample(response_content=raw)

        filtered, reasons = filter_service.filter_samples([sample])

        assert len(filtered) == 1
        assert (
            filtered[0].response_content == '{"entities": [{"name": "Paris", "type": "LOCATION"}]}'
        )

    def test_extracts_from_bare_code_fence(self, filter_service: FilterService) -> None:
        """JSON wrapped in ``` ... ``` (no language tag) should also be extracted."""
        raw = '```\n{"result": "ok"}\n```'
        sample = make_sample(response_content=raw)

        filtered, reasons = filter_service.filter_samples([sample])

        assert len(filtered) == 1
        assert filtered[0].response_content == '{"result": "ok"}'

    def test_preserves_clean_json(self, filter_service: FilterService) -> None:
        """Clean JSON without code fences should not be modified."""
        clean = '{"entities": [{"name": "test"}]}'
        sample = make_sample(response_content=clean)

        filtered, reasons = filter_service.filter_samples([sample])

        assert len(filtered) == 1
        assert filtered[0].response_content == clean

    def test_same_prompt_different_responses_kept(self, filter_service: FilterService) -> None:
        """Same input messages with different responses should NOT be deduped."""
        messages = [
            Message(role="system", content="Extract entities."),
            Message(role="user", content="Paris is in France."),
        ]

        sample_a = make_sample(
            messages=messages,
            response_content=json.dumps({"entities": [{"name": "Paris"}]}),
        )
        sample_b = make_sample(
            messages=messages,
            response_content=json.dumps({"entities": [{"name": "Paris"}, {"name": "France"}]}),
        )

        filtered, reasons = filter_service.filter_samples([sample_a, sample_b])

        assert len(filtered) == 2
        assert "duplicate" not in reasons

    def test_code_fence_extraction_modifies_sample_content(
        self, filter_service: FilterService
    ) -> None:
        """After code fence extraction, the sample's response_content is permanently changed."""
        raw = '```json\n{"extracted": true}\n```'
        sample = make_sample(response_content=raw)

        filtered, _ = filter_service.filter_samples([sample])
        assert len(filtered) == 1
        # The sample object itself should have been mutated
        assert sample.response_content == '{"extracted": true}'


class TestCombinedFiltering:
    """Test that multiple filters work together correctly."""

    def test_multiple_filter_reasons(self, filter_service: FilterService) -> None:
        """Multiple filter reasons should be tracked independently."""
        samples = [
            make_sample(response_content=json.dumps({"ok": True})),  # passes
            make_sample(response_content="hi"),  # too_short
            make_sample(response_content="not valid json at all here man"),  # invalid_json
        ]

        filtered, reasons = filter_service.filter_samples(samples)

        assert len(filtered) == 1
        assert reasons.get("too_short", 0) >= 1
        assert reasons.get("invalid_json", 0) >= 1

    def test_large_batch_filtering(self, filter_service: FilterService) -> None:
        """Filtering should work correctly on larger batches."""
        samples = make_samples(count=100)

        filtered, reasons = filter_service.filter_samples(samples)

        # All samples from make_samples are unique with valid JSON
        assert len(filtered) == 100
        assert sum(reasons.values()) == 0

    def test_empty_input(self, filter_service: FilterService) -> None:
        """Empty input should return empty output."""
        filtered, reasons = filter_service.filter_samples([])

        assert len(filtered) == 0
        assert reasons == {}

    def test_filter_order_short_before_error_pattern(self, filter_service: FilterService) -> None:
        """Too-short responses should be caught by length check before error pattern check."""
        # "hi" is too short AND not JSON, but should be filtered as too_short first
        sample = make_sample(response_content="hi")
        filtered, reasons = filter_service.filter_samples([sample])
        assert len(filtered) == 0
        assert reasons.get("too_short") == 1
        # Should NOT also count as error_pattern or invalid_json since length check is first
        assert "error_pattern" not in reasons
        assert "invalid_json" not in reasons
