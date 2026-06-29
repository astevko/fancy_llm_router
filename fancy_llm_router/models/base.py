"""Base class for all model providers."""

import asyncio
import hashlib
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union, AsyncIterator
from datetime import datetime

from fancy_llm_router.schemas.models import ModelInfo, ModelProvider
from fancy_llm_router.schemas.requests import (
    CompletionRequest,
    CompletionResponse,
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ChatMessage,
)
from fancy_llm_router.schemas.metrics import RequestMetrics, TokenUsage, CostMetrics, LatencyMetrics


class ModelError(Exception):
    """Base exception for model errors."""
    pass


class ModelTimeoutError(ModelError):
    """Timeout error for model calls."""
    pass


class ModelRateLimitError(ModelError):
    """Rate limit error for model calls."""
    pass


class ModelAuthenticationError(ModelError):
    """Authentication error for model calls."""
    pass


class BaseModelProvider(ABC):
    """Abstract base class for all model providers."""
    
    def __init__(
        self,
        provider: ModelProvider,
        model_id: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ):
        self.provider = provider
        self.model_id = model_id
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.extra_config = kwargs
        
        # Initialize model info
        self._model_info: Optional[ModelInfo] = None
        
        # Request tracking
        self._request_counter = 0
    
    @property
    def model_info(self) -> ModelInfo:
        """Get model information."""
        if self._model_info is None:
            self._model_info = self._create_model_info()
        return self._model_info
    
    @property
    def full_id(self) -> str:
        """Get the full model identifier."""
        return f"{self.provider.value}:{self.model_id}"
    
    def _create_model_info(self) -> ModelInfo:
        """Create model info. Override in subclasses for specific models."""
        return ModelInfo(
            provider=self.provider,
            model_id=self.model_id,
            name=self.model_id,
            capabilities=self._get_default_capabilities(),
            pricing=self._get_default_pricing(),
        )
    
    def _get_default_capabilities(self) -> Any:
        """Get default capabilities. Override in subclasses."""
        from fancy_llm_router.schemas.models import ModelCapabilities
        return ModelCapabilities(
            max_tokens=4096,
            max_input_tokens=4096,
            context_window=4096,
            supports_streaming=True,
            supports_chat=True,
            supports_completions=True,
            supports_embeddings=False,
        )
    
    def _get_default_pricing(self) -> Any:
        """Get default pricing. Override in subclasses."""
        from fancy_llm_router.schemas.models import ModelPricing
        return ModelPricing(
            input_token_price=0.0,
            output_token_price=0.0,
        )
    
    def _generate_request_id(self) -> str:
        """Generate a unique request ID."""
        self._request_counter += 1
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        return f"{self.full_id}-{timestamp}-{self._request_counter}"
    
    def _hash_prompt(self, prompt: str) -> str:
        """Hash a prompt for deduplication."""
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]
    
    def _calculate_cost(
        self, 
        input_tokens: int, 
        output_tokens: int,
        input_price: Optional[float] = None,
        output_price: Optional[float] = None
    ) -> CostMetrics:
        """Calculate cost metrics."""
        from fancy_llm_router.schemas.metrics import CostMetrics
        
        pricing = self.model_info.pricing
        input_price = input_price or pricing.input_token_price
        output_price = output_price or pricing.output_token_price
        
        return CostMetrics(
            input_token_cost=input_tokens * input_price,
            output_token_cost=output_tokens * output_price,
            total_cost=(input_tokens * input_price) + (output_tokens * output_price),
            input_token_price=input_price,
            output_token_price=output_price,
        )
    
    def _create_token_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int
    ) -> TokenUsage:
        """Create token usage metrics."""
        from fancy_llm_router.schemas.metrics import TokenUsage
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
    
    def _create_latency_metrics(
        self,
        start_time: float,
        first_token_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> LatencyMetrics:
        """Create latency metrics."""
        from fancy_llm_router.schemas.metrics import LatencyMetrics
        
        end_time = end_time or time.time()
        duration = (end_time - start_time) * 1000  # Convert to ms
        
        time_to_first_token = None
        if first_token_time:
            time_to_first_token = (first_token_time - start_time) * 1000
        
        return LatencyMetrics(
            time_to_first_token_ms=time_to_first_token,
            time_to_complete_ms=duration,
            tokens_per_second=None,  # Can be calculated later
        )
    
    def _create_request_metrics(
        self,
        request: Union[CompletionRequest, ChatRequest, EmbeddingRequest],
        response: Union[CompletionResponse, ChatResponse, EmbeddingResponse],
        start_time: float,
        end_time: float,
        first_token_time: Optional[float] = None,
        git_commit: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> RequestMetrics:
        """Create complete request metrics."""
        from fancy_llm_router.schemas.metrics import RequestMetrics, ModelMetrics
        
        # Extract token usage
        if hasattr(response, 'usage'):
            usage = response.usage
            token_usage = self._create_token_usage(
                prompt_tokens=usage.get('prompt_tokens', 0),
                completion_tokens=usage.get('completion_tokens', 0)
            )
        else:
            token_usage = self._create_token_usage(0, 0)
        
        # Calculate cost
        cost = self._calculate_cost(
            input_tokens=token_usage.prompt_tokens,
            output_tokens=token_usage.completion_tokens
        )
        
        # Calculate latency
        latency = self._create_latency_metrics(
            start_time=start_time,
            first_token_time=first_token_time,
            end_time=end_time
        )
        
        # Model metrics
        model_metrics = ModelMetrics(
            model_id=self.model_id,
            model_provider=self.provider.value,
            model_version=self.model_info.version,
            model_parameters=self.model_info.parameters,
            context_window=self.model_info.capabilities.context_window,
        )
        
        # Create request metrics
        metrics = RequestMetrics(
            request_id=request.request_id or self._generate_request_id(),
            session_id=request.session_id,
            prompt_hash=request.prompt_hash or self._hash_prompt(
                request.prompt if isinstance(request, CompletionRequest) 
                else str(request.messages) if isinstance(request, ChatRequest)
                else str(request.input)
            ),
            created_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            request_type=type(request).__name__.replace('Request', '').lower(),
            model_info=model_metrics,
            token_usage=token_usage,
            cost=cost,
            latency=latency,
            git_commit=git_commit,
            metadata=extra_metadata or {},
        )
        
        return metrics
    
    @abstractmethod
    async def completion(
        self,
        request: CompletionRequest,
        git_commit: Optional[str] = None,
        **kwargs
    ) -> CompletionResponse:
        """Generate a text completion."""
        pass
    
    @abstractmethod
    async def chat(
        self,
        request: ChatRequest,
        git_commit: Optional[str] = None,
        **kwargs
    ) -> ChatResponse:
        """Generate a chat completion."""
        pass
    
    async def embedding(
        self,
        request: EmbeddingRequest,
        git_commit: Optional[str] = None,
        **kwargs
    ) -> EmbeddingResponse:
        """Generate embeddings. Override if supported."""
        raise NotImplementedError(f"Embeddings not supported for {self.full_id}")
    
    async def stream_completion(
        self,
        request: CompletionRequest,
        git_commit: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream a text completion. Override if supported."""
        raise NotImplementedError(f"Streaming not supported for {self.full_id}")
    
    async def stream_chat(
        self,
        request: ChatRequest,
        git_commit: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[ChatMessage]:
        """Stream a chat completion. Override if supported."""
        raise NotImplementedError(f"Streaming not supported for {self.full_id}")
    
    async def count_tokens(self, text: str) -> int:
        """Count tokens in text. Override for provider-specific tokenization."""
        # Default: approximate with character count / 4
        return len(text) // 4
    
    async def count_message_tokens(self, messages: List[ChatMessage]) -> int:
        """Count tokens in a list of messages."""
        total = 0
        for message in messages:
            total += await self.count_tokens(message.content)
        return total
    
    async def health_check(self) -> bool:
        """Check if the model is available and healthy."""
        return True
    
    async def close(self):
        """Clean up resources."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.full_id})"


class ModelProviderFactory:
    """Factory for creating model providers."""
    
    _providers: Dict[str, Dict[str, BaseModelProvider]] = {}
    
    @classmethod
    def register(
        cls,
        provider: ModelProvider,
        model_id: str,
        provider_instance: BaseModelProvider
    ):
        """Register a provider instance."""
        if provider.value not in cls._providers:
            cls._providers[provider.value] = {}
        cls._providers[provider.value][model_id] = provider_instance
    
    @classmethod
    def get(
        cls,
        provider: Union[ModelProvider, str],
        model_id: str
    ) -> Optional[BaseModelProvider]:
        """Get a registered provider."""
        provider_str = provider.value if isinstance(provider, ModelProvider) else provider
        if provider_str in cls._providers:
            return cls._providers[provider_str].get(model_id)
        return None
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """List all registered providers."""
        return list(cls._providers.keys())
    
    @classmethod
    def list_models(cls, provider: Optional[str] = None) -> List[str]:
        """List all registered models for a provider."""
        if provider:
            if provider in cls._providers:
                return list(cls._providers[provider].keys())
            return []
        
        models = []
        for provider_models in cls._providers.values():
            models.extend(provider_models.keys())
        return models
    
    @classmethod
    def create(
        cls,
        provider: Union[ModelProvider, str],
        model_id: str,
        **kwargs
    ) -> BaseModelProvider:
        """Create a new provider instance."""
        provider_str = provider.value if isinstance(provider, ModelProvider) else provider
        
        # Map provider strings to classes
        from fancy_llm_router.models.openai import OpenAIProvider
        from fancy_llm_router.models.anthropic import AnthropicProvider
        from fancy_llm_router.models.local import LocalProvider
        provider_classes = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "local": LocalProvider,
        }
        
        if provider_str in provider_classes:
            provider_class = provider_classes[provider_str]
            instance = provider_class(model_id=model_id, **kwargs)
            cls.register(ModelProvider(provider_str), model_id, instance)
            return instance
        
        # Default to base implementation
        from fancy_llm_router.models.generic import GenericProvider
        instance = GenericProvider(
            provider=ModelProvider(provider_str),
            model_id=model_id,
            **kwargs
        )
        cls.register(ModelProvider(provider_str), model_id, instance)
        return instance
    
    @classmethod
    async def close_all(cls):
        """Close all registered providers."""
        for provider_models in cls._providers.values():
            for provider in provider_models.values():
                await provider.close()
        cls._providers.clear()


# Import concrete providers (will be defined in separate files)
from fancy_llm_router.models.generic import GenericProvider
