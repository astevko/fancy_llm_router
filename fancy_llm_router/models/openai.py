"""OpenAI model provider."""

import asyncio
import time
from typing import Optional, Dict, Any, List, AsyncIterator
import httpx

from fancy_llm_router.models.base import BaseModelProvider, ModelError, ModelTimeoutError, ModelAuthenticationError
from fancy_llm_router.schemas.models import ModelProvider, ModelInfo, ModelCapabilities, ModelPricing
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
    EmbeddingData,
)


# Known OpenAI models and their capabilities
OPENAI_MODELS = {
    "gpt-4": {
        "name": "GPT-4",
        "context_window": 8192,
        "max_tokens": 4096,
        "input_price": 0.03,
        "output_price": 0.06,
        "supports_chat": True,
        "supports_completions": True,
        "supports_embeddings": False,
        "supports_function_calls": True,
    },
    "gpt-3.5-turbo": {
        "name": "GPT-3.5 Turbo",
        "context_window": 4096,
        "max_tokens": 4096,
        "input_price": 0.0015,
        "output_price": 0.002,
        "supports_chat": True,
        "supports_completions": False,
        "supports_embeddings": False,
        "supports_function_calls": True,
    },
}


class OpenAIProvider(BaseModelProvider):
    """OpenAI API provider."""
    
    def __init__(
        self,
        model_id: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ):
        super().__init__(
            provider=ModelProvider.OPENAI,
            model_id=model_id,
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1",
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            **kwargs
        )
        self.organization = organization
        self.project = project
        self._client: Optional[httpx.AsyncClient] = None
    
    def _get_default_capabilities(self) -> ModelCapabilities:
        model_config = OPENAI_MODELS.get(self.model_id, {})
        return ModelCapabilities(
            max_tokens=model_config.get("max_tokens", 4096),
            max_input_tokens=model_config.get("context_window", 4096),
            context_window=model_config.get("context_window", 4096),
            supports_streaming=True,
            supports_chat=model_config.get("supports_chat", True),
            supports_completions=model_config.get("supports_completions", True),
            supports_embeddings=model_config.get("supports_embeddings", False),
            supports_function_calls=model_config.get("supports_function_calls", False),
        )
    
    def _get_default_pricing(self) -> ModelPricing:
        model_config = OPENAI_MODELS.get(self.model_id, {})
        return ModelPricing(
            input_token_price=model_config.get("input_price", 0.0),
            output_token_price=model_config.get("output_price", 0.0),
        )
    
    def _create_model_info(self) -> ModelInfo:
        model_config = OPENAI_MODELS.get(self.model_id, {})
        return ModelInfo(
            provider=self.provider,
            model_id=self.model_id,
            name=model_config.get("name", self.model_id),
            version=None,
            description=model_config.get("description"),
            capabilities=self._get_default_capabilities(),
            pricing=self._get_default_pricing(),
            parameters=None,
            metadata={"model_config": model_config},
        )
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers=self._get_headers(),
            )
        return self._client
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        if self.project:
            headers["OpenAI-Project"] = self.project
        return headers
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Make a non-streaming API request with retries."""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client.request(method, endpoint, json=data, **kwargs)
                response.raise_for_status()
                return response.json()
            except httpx.TimeoutException as e:
                last_error = ModelTimeoutError(f"Request timed out: {e}")
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    # Rate limit
                    retry_after = e.response.headers.get("Retry-After", self.retry_delay)
                    await asyncio.sleep(float(retry_after))
                    continue
                elif e.response.status_code == 401:
                    last_error = ModelAuthenticationError("Invalid API key")
                    break
                else:
                    last_error = ModelError(f"API error: {e.response.status_code} - {e.response.text}")
            except Exception as e:
                last_error = ModelError(f"Request failed: {e}")
            
            if attempt < self.max_retries:
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        if last_error:
            raise last_error
    
    async def _stream_request(
        self,
        method: str,
        endpoint: str,
        data: Dict[str, Any],
        **kwargs
    ) -> AsyncIterator[str]:
        """Make a streaming API request."""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                async with self.client.stream(method, endpoint, json=data, **kwargs) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            yield line
                return  # Success, exit the function
            except httpx.TimeoutException as e:
                last_error = ModelTimeoutError(f"Request timed out: {e}")
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    # Rate limit
                    retry_after = e.response.headers.get("Retry-After", self.retry_delay)
                    await asyncio.sleep(float(retry_after))
                    continue
                elif e.response.status_code == 401:
                    last_error = ModelAuthenticationError("Invalid API key")
                    break
                else:
                    last_error = ModelError(f"API error: {e.response.status_code} - {e.response.text}")
            except Exception as e:
                last_error = ModelError(f"Request failed: {e}")
            
            if attempt < self.max_retries:
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        if last_error:
            raise last_error
    
    async def completion(
        self,
        request: CompletionRequest,
        git_commit: Optional[str] = None,
        **kwargs
    ) -> CompletionResponse:
        """Generate a text completion."""
        start_time = time.time()
        
        try:
            data = {
                "model": self.model_id,
                "prompt": request.prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "n": request.n,
                "stream": False,
            }
            if request.stop:
                data["stop"] = request.stop
            if request.echo:
                data["echo"] = request.echo
            data.update(request.extra)
            
            response_data = await self._make_request("POST", "/completions", data)
            
            choices = []
            for i, choice_data in enumerate(response_data.get("choices", [])):
                choices.append(CompletionChoice(
                    text=choice_data.get("text", ""),
                    index=choice_data.get("index", i),
                    finish_reason=choice_data.get("finish_reason"),
                    logprobs=choice_data.get("logprobs"),
                ))
            
            return CompletionResponse(
                id=response_data.get("id", ""),
                object="text_completion",
                created=response_data.get("created", int(time.time())),
                model=self.model_id,
                choices=choices,
                usage=response_data.get("usage", {}),
                request_id=request.request_id,
                latency_ms=(time.time() - start_time) * 1000,
            )
            
        except Exception as e:
            raise ModelError(f"Completion failed: {e}")
    
    async def chat(
        self,
        request: ChatRequest,
        git_commit: Optional[str] = None,
        **kwargs
    ) -> ChatResponse:
        """Generate a chat completion."""
        start_time = time.time()
        
        try:
            messages = []
            for msg in request.messages:
                message = {
                    "role": msg.role.value,
                    "content": msg.content,
                }
                if msg.name:
                    message["name"] = msg.name
                messages.append(message)
            
            data = {
                "model": self.model_id,
                "messages": messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "n": request.n,
                "stream": False,
            }
            if request.stop:
                data["stop"] = request.stop
            if request.tools:
                data["tools"] = request.tools
            if request.tool_choice:
                data["tool_choice"] = request.tool_choice
            data.update(request.extra)
            
            response_data = await self._make_request("POST", "/chat/completions", data)
            
            choices = []
            for i, choice_data in enumerate(response_data.get("choices", [])):
                message_data = choice_data.get("message", {})
                message = ChatMessage(
                    role=message_data.get("role", "assistant"),
                    content=message_data.get("content", ""),
                    name=message_data.get("name"),
                    tool_call_id=message_data.get("tool_call_id"),
                    tool_calls=message_data.get("tool_calls"),
                )
                choices.append(ChatChoice(
                    index=choice_data.get("index", i),
                    message=message,
                    finish_reason=choice_data.get("finish_reason"),
                ))
            
            return ChatResponse(
                id=response_data.get("id", ""),
                object="chat.completion",
                created=response_data.get("created", int(time.time())),
                model=self.model_id,
                choices=choices,
                usage=response_data.get("usage", {}),
                request_id=request.request_id,
                latency_ms=(time.time() - start_time) * 1000,
            )
            
        except Exception as e:
            raise ModelError(f"Chat failed: {e}")
    
    async def embedding(
        self,
        request: EmbeddingRequest,
        git_commit: Optional[str] = None,
        **kwargs
    ) -> EmbeddingResponse:
        """Generate embeddings."""
        start_time = time.time()
        
        try:
            data = {
                "model": self.model_id,
                "input": request.input,
            }
            if request.user:
                data["user"] = request.user
            
            response_data = await self._make_request("POST", "/embeddings", data)
            
            embeddings = []
            for i, embedding_data in enumerate(response_data.get("data", [])):
                embeddings.append(EmbeddingData(
                    object="embedding",
                    embedding=embedding_data.get("embedding", []),
                    index=embedding_data.get("index", i),
                ))
            
            return EmbeddingResponse(
                object="list",
                data=embeddings,
                model=self.model_id,
                usage=response_data.get("usage", {}),
                request_id=request.request_id,
                latency_ms=(time.time() - start_time) * 1000,
            )
            
        except Exception as e:
            raise ModelError(f"Embedding failed: {e}")
    
    async def stream_completion(
        self,
        request: CompletionRequest,
        git_commit: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream a text completion."""
        data = {
            "model": self.model_id,
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "n": request.n,
            "stream": True,
        }
        if request.stop:
            data["stop"] = request.stop
        if request.echo:
            data["echo"] = request.echo
        data.update(request.extra)
        
        async for line in self._stream_request("POST", "/completions", data):
            if line.strip() and line.strip() != "data: [DONE]":
                import json
                try:
                    chunk = json.loads(line.strip()[5:])  # Remove "data: " prefix
                    for choice in chunk.get("choices", []):
                        if choice.get("text"):
                            yield choice["text"]
                except json.JSONDecodeError:
                    continue
    
    async def stream_chat(
        self,
        request: ChatRequest,
        git_commit: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[ChatMessage]:
        """Stream a chat completion."""
        messages = []
        for msg in request.messages:
            message = {
                "role": msg.role.value,
                "content": msg.content,
            }
            if msg.name:
                message["name"] = msg.name
            messages.append(message)
        
        data = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "n": request.n,
            "stream": True,
        }
        if request.stop:
            data["stop"] = request.stop
        if request.tools:
            data["tools"] = request.tools
        if request.tool_choice:
            data["tool_choice"] = request.tool_choice
        data.update(request.extra)
        
        async for line in self._stream_request("POST", "/chat/completions", data):
            if line.strip() and line.strip() != "data: [DONE]":
                import json
                try:
                    chunk = json.loads(line.strip()[5:])  # Remove "data: " prefix
                    for choice in chunk.get("choices", []):
                        if choice.get("delta"):
                            delta = choice["delta"]
                            message = ChatMessage(
                                role=delta.get("role", "assistant"),
                                content=delta.get("content", ""),
                                name=delta.get("name"),
                                tool_call_id=delta.get("tool_call_id"),
                                tool_calls=delta.get("tool_calls"),
                            )
                            yield message
                except json.JSONDecodeError:
                    continue
    
    async def count_tokens(self, text: str) -> int:
        """Count tokens using OpenAI's tokenization."""
        try:
            import tiktoken
            encoder = tiktoken.encoding_for_model(self.model_id)
            return len(encoder.encode(text))
        except ImportError:
            return super().count_tokens(text)
    
    async def health_check(self) -> bool:
        """Check if the OpenAI API is available."""
        try:
            await self._make_request("GET", "/models", {})
            return True
        except Exception:
            return False
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
