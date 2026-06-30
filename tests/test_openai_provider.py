"""Tests for the OpenAI-compatible provider."""

import pytest

from fancy_llm_router.models.openai import OpenAIProvider
from fancy_llm_router.schemas.requests import (
    ChatRequest,
    ChatMessage,
    CompletionRequest,
    MessageRole,
)
from tests.conftest import NEBIUS_CHAT_PAYLOAD, NEBIUS_COMPLETION_PAYLOAD


@pytest.fixture
def provider():
    return OpenAIProvider(
        model_id="nvidia/nemotron-3-super-120b-a12b",
        api_key="test-key",
        base_url="https://api.tokenfactory.nebius.com/v1",
    )


class TestOpenAIProvider:
    @pytest.mark.asyncio
    async def test_completion_normalizes_nebius_usage(self, provider, monkeypatch):
        async def fake_request(method, endpoint, data, **kwargs):
            assert method == "POST"
            assert endpoint == "/completions"
            return NEBIUS_COMPLETION_PAYLOAD

        monkeypatch.setattr(provider, "_make_request", fake_request)

        response = await provider.completion(
            CompletionRequest(prompt="What is the capital of France?", max_tokens=64)
        )

        assert response.choices[0].text == "Paris is the capital of France."
        assert response.usage == {
            "prompt_tokens": 12,
            "completion_tokens": 8,
            "total_tokens": 20,
        }
        assert "prompt_tokens_details" not in response.usage

    @pytest.mark.asyncio
    async def test_chat_normalizes_nebius_usage(self, provider, monkeypatch):
        async def fake_request(method, endpoint, data, **kwargs):
            assert endpoint == "/chat/completions"
            return NEBIUS_CHAT_PAYLOAD

        monkeypatch.setattr(provider, "_make_request", fake_request)

        response = await provider.chat(
            ChatRequest(
                messages=[ChatMessage(role=MessageRole.USER, content="Hi")],
                max_tokens=32,
            )
        )

        assert response.choices[0].message.content == "Bonjour"
        assert response.usage["total_tokens"] == 7

    @pytest.mark.asyncio
    async def test_completion_uses_request_model_id_in_api_call(self, provider, monkeypatch):
        captured = {}

        async def fake_request(method, endpoint, data, **kwargs):
            captured.update(data)
            return {
                **NEBIUS_COMPLETION_PAYLOAD,
                "choices": [{"text": "ok", "index": 0, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

        monkeypatch.setattr(provider, "_make_request", fake_request)

        await provider.completion(CompletionRequest(prompt="hi", max_tokens=10))

        assert captured["model"] == "nvidia/nemotron-3-super-120b-a12b"
        assert captured["prompt"] == "hi"
