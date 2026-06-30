"""Tests for token usage normalization."""

import pytest

from fancy_llm_router.schemas.requests import CompletionResponse
from fancy_llm_router.utils.usage import normalize_usage


class TestNormalizeUsage:
    def test_empty_usage(self):
        assert normalize_usage(None) == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        assert normalize_usage({}) == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def test_simple_openai_usage(self):
        raw = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        assert normalize_usage(raw) == raw

    def test_nebius_style_nested_details(self):
        raw = {
            "prompt_tokens": 12,
            "completion_tokens": 8,
            "total_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0, "audio_tokens": 0},
        }
        assert normalize_usage(raw) == {
            "prompt_tokens": 12,
            "completion_tokens": 8,
            "total_tokens": 20,
        }

    def test_computes_total_when_missing(self):
        assert normalize_usage({"prompt_tokens": 3, "completion_tokens": 4}) == {
            "prompt_tokens": 3,
            "completion_tokens": 4,
            "total_tokens": 7,
        }

    def test_completion_response_accepts_normalized_usage(self):
        usage = normalize_usage(
            {
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 0},
            }
        )
        response = CompletionResponse(
            id="cmpl-test",
            created=1,
            model="test-model",
            choices=[],
            usage=usage,
        )
        assert response.usage == {
            "prompt_tokens": 12,
            "completion_tokens": 8,
            "total_tokens": 20,
        }

    def test_rejects_unnormalized_usage_in_schema(self):
        with pytest.raises(Exception):
            CompletionResponse(
                id="cmpl-test",
                created=1,
                model="test-model",
                choices=[],
                usage={
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            )
