"""Mock model provider.

Returns deterministic, offline responses with estimated token usage. Useful for
testing the routing/metrics pipeline and for load/"token burning" without
calling a real LLM API.
"""

import time
from typing import Optional

from fancy_llm_router.models.base import BaseModelProvider
from fancy_llm_router.schemas.models import ModelCapabilities, ModelPricing, ModelProvider
from fancy_llm_router.schemas.requests import (
    ChatChoice,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    MessageRole,
)


class MockProvider(BaseModelProvider):
    """A provider that fabricates responses without any network calls."""

    def __init__(self, model_id: str = "mock-model", **kwargs):
        super().__init__(provider=ModelProvider.MOCK, model_id=model_id, **kwargs)

    def _get_default_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            max_tokens=4096,
            max_input_tokens=8192,
            context_window=8192,
            supports_streaming=True,
            supports_chat=True,
            supports_completions=True,
            supports_embeddings=False,
        )

    def _get_default_pricing(self) -> ModelPricing:
        return ModelPricing(input_token_price=0.0, output_token_price=0.0)

    @staticmethod
    def _usage(prompt: str, completion: str) -> dict:
        prompt_tokens = max(1, len(prompt.split()))
        completion_tokens = max(1, len(completion.split()))
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    async def completion(
        self,
        request: CompletionRequest,
        git_commit: Optional[str] = None,
        **kwargs,
    ) -> CompletionResponse:
        start = time.time()
        prompt_lower = request.prompt.lower()
        if "capital of france" in prompt_lower:
            text = "The capital of France is Paris."
        elif "capital of ukraine" in prompt_lower:
            text = "The capital of Ukraine is Kyiv."
        else:
            text = f"[mock:{self.model_id}] Response to: {request.prompt[:120]}"
        return CompletionResponse(
            id=f"mock-{int(start)}",
            object="text_completion",
            created=int(start),
            model=self.model_id,
            choices=[CompletionChoice(text=text, index=0, finish_reason="stop")],
            usage=self._usage(request.prompt, text),
            request_id=request.request_id,
            latency_ms=(time.time() - start) * 1000,
        )

    async def chat(
        self,
        request: ChatRequest,
        git_commit: Optional[str] = None,
        **kwargs,
    ) -> ChatResponse:
        start = time.time()
        last_user = next(
            (m.content for m in reversed(request.messages) if m.role == MessageRole.USER),
            "",
        )
        text = f"[mock:{self.model_id}] Reply to: {last_user[:120]}"
        prompt_text = " ".join(m.content for m in request.messages)
        return ChatResponse(
            id=f"mock-{int(start)}",
            object="chat.completion",
            created=int(start),
            model=self.model_id,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role=MessageRole.ASSISTANT, content=text),
                    finish_reason="stop",
                )
            ],
            usage=self._usage(prompt_text, text),
            request_id=request.request_id,
            latency_ms=(time.time() - start) * 1000,
        )

    async def health_check(self) -> bool:
        return True
