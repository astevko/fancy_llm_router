"""Anthropic model provider."""

from typing import Optional, Dict, Any

from fancy_llm_router.models.base import BaseModelProvider
from fancy_llm_router.schemas.models import ModelProvider, ModelInfo, ModelCapabilities, ModelPricing


class AnthropicProvider(BaseModelProvider):
    """Anthropic API provider."""
    
    def __init__(
        self,
        model_id: str = "claude-3-sonnet-20240229",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            provider=ModelProvider.ANTHROPIC,
            model_id=model_id,
            api_key=api_key,
            base_url=base_url or "https://api.anthropic.com/v1",
            **kwargs
        )
    
    def _get_default_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            max_tokens=4096,
            max_input_tokens=200000,
            context_window=200000,
            supports_streaming=True,
            supports_chat=True,
            supports_completions=False,
            supports_embeddings=False,
            supports_function_calls=True,
        )
    
    def _get_default_pricing(self) -> ModelPricing:
        return ModelPricing(
            input_token_price=0.003,
            output_token_price=0.015,
        )
    
    def _create_model_info(self) -> ModelInfo:
        return ModelInfo(
            provider=self.provider,
            model_id=self.model_id,
            name=self.model_id,
            capabilities=self._get_default_capabilities(),
            pricing=self._get_default_pricing(),
        )
