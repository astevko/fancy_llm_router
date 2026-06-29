"""Local model provider for running models locally."""

from typing import Optional, Dict, Any

from fancy_llm_router.models.base import BaseModelProvider
from fancy_llm_router.schemas.models import ModelProvider, ModelInfo, ModelCapabilities, ModelPricing


class LocalProvider(BaseModelProvider):
    """Local model provider for running LLMs locally (Ollama, vLLM, etc.)."""
    
    def __init__(
        self,
        model_id: str = "llama-2-7b",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            provider=ModelProvider.LOCAL,
            model_id=model_id,
            api_key=api_key,
            base_url=base_url or "http://localhost:11434",
            **kwargs
        )
    
    def _get_default_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            max_tokens=2048,
            max_input_tokens=2048,
            context_window=2048,
            supports_streaming=True,
            supports_chat=True,
            supports_completions=True,
            supports_embeddings=False,
            supports_function_calls=False,
        )
    
    def _get_default_pricing(self) -> ModelPricing:
        return ModelPricing(
            input_token_price=0.0,
            output_token_price=0.0,
        )
    
    def _create_model_info(self) -> ModelInfo:
        return ModelInfo(
            provider=self.provider,
            model_id=self.model_id,
            name=self.model_id,
            capabilities=self._get_default_capabilities(),
            pricing=self._get_default_pricing(),
        )
