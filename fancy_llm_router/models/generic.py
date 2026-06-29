"""Generic model provider for unsupported providers."""

from typing import Optional, Dict, Any, Union, AsyncIterator
import asyncio

from fancy_llm_router.models.base import BaseModelProvider, ModelError
from fancy_llm_router.schemas.models import ModelProvider
from fancy_llm_router.schemas.requests import (
    CompletionRequest,
    CompletionResponse,
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ChatMessage,
    CompletionChoice,
    ChatChoice,
)


class GenericProvider(BaseModelProvider):
    """A generic provider that can be customized for any API."""
    
    def __init__(
        self,
        provider: Union[ModelProvider, str],
        model_id: str,
        completion_endpoint: Optional[str] = None,
        chat_endpoint: Optional[str] = None,
        embedding_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        provider_enum = provider if isinstance(provider, ModelProvider) else ModelProvider(provider)
        super().__init__(
            provider=provider_enum,
            model_id=model_id,
            api_key=api_key,
            **kwargs
        )
        self.completion_endpoint = completion_endpoint
        self.chat_endpoint = chat_endpoint
        self.embedding_endpoint = embedding_endpoint
    
    async def completion(
        self,
        request: CompletionRequest,
        git_commit: Optional[str] = None,
        **kwargs
    ) -> CompletionResponse:
        """Generate a text completion."""
        raise NotImplementedError(
            f"Completion not implemented for {self.full_id}. "
            "Please implement the completion method or use a specific provider."
        )
    
    async def chat(
        self,
        request: ChatRequest,
        git_commit: Optional[str] = None,
        **kwargs
    ) -> ChatResponse:
        """Generate a chat completion."""
        raise NotImplementedError(
            f"Chat not implemented for {self.full_id}. "
            "Please implement the chat method or use a specific provider."
        )
    
    async def embedding(
        self,
        request: EmbeddingRequest,
        git_commit: Optional[str] = None,
        **kwargs
    ) -> EmbeddingResponse:
        """Generate embeddings."""
        raise NotImplementedError(
            f"Embeddings not implemented for {self.full_id}. "
            "Please implement the embedding method or use a specific provider."
        )
    
    async def stream_completion(
        self,
        request: CompletionRequest,
        git_commit: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream a text completion."""
        # Default implementation: call completion and yield the result
        response = await self.completion(request, git_commit, **kwargs)
        if response.choices:
            for choice in response.choices:
                if choice.text:
                    yield choice.text
    
    async def stream_chat(
        self,
        request: ChatRequest,
        git_commit: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[ChatMessage]:
        """Stream a chat completion."""
        # Default implementation: call chat and yield messages
        response = await self.chat(request, git_commit, **kwargs)
        if response.choices:
            for choice in response.choices:
                if choice.message:
                    yield choice.message
