"""Shared pytest fixtures."""

import pytest

from fancy_llm_router.core.config_loader import build_router_from_config


MOCK_CONFIG = {
    "router": {"default_strategy": "balanced"},
    "models": {
        "mock-fast@mock": {
            "model": "mock-fast",
            "source": "mock",
            "provider": "mock",
            "model_id": "mock-fast",
            "context_window": 8192,
            "input_token_price": 0.0,
            "output_token_price": 0.0,
            "tokens_per_second": 1000,
            "enabled": True,
        },
        "qwen@nebius": {
            "model": "Qwen/Qwen3-32B",
            "source": "nebius",
            "provider": "mock",
            "model_id": "Qwen/Qwen3-32B",
            "input_token_price": 1e-7,
            "output_token_price": 3e-7,
            "context_window": 41000,
            "tokens_per_second": 23.0,
            "enabled": True,
        },
        "qwen@ollama": {
            "model": "Qwen/Qwen3-32B",
            "source": "ollama",
            "provider": "mock",
            "model_id": "qwen3:32b",
            "input_token_price": 0.0,
            "output_token_price": 0.0,
            "context_window": 41000,
            "tokens_per_second": 12.0,
            "enabled": True,
        },
    },
}


@pytest.fixture
def mock_router():
    """Router with mock deployments including multi-source Qwen."""
    return build_router_from_config(MOCK_CONFIG)


NEBIUS_COMPLETION_PAYLOAD = {
    "id": "cmpl-nebius-test",
    "object": "text_completion",
    "created": 1710000000,
    "model": "nvidia/nemotron-3-super-120b-a12b",
    "choices": [
        {
            "text": "Paris is the capital of France.",
            "index": 0,
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 12,
        "completion_tokens": 8,
        "total_tokens": 20,
        "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0, "audio_tokens": 0},
    },
}

NEBIUS_CHAT_PAYLOAD = {
    "id": "chatcmpl-nebius-test",
    "object": "chat.completion",
    "created": 1710000000,
    "model": "nvidia/nemotron-3-super-120b-a12b",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Bonjour"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 5,
        "completion_tokens": 2,
        "total_tokens": 7,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    },
}
