"""Schemas for LLM model definitions and capabilities."""

from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, HttpUrl


class ModelProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    MISTRAL = "mistral"
    LOCAL = "local"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    VLLM = "vllm"
    CUSTOM = "custom"


class ModelCapabilities(BaseModel):
    """Capabilities and limits of a model."""
    max_tokens: int = Field(..., description="Maximum tokens in a single response")
    max_input_tokens: int = Field(default=4096, description="Maximum tokens in the input/prompt")
    context_window: int = Field(..., description="Total context window size")
    supports_streaming: bool = True
    supports_chat: bool = True
    supports_completions: bool = True
    supports_embeddings: bool = False
    supports_function_calls: bool = False
    supports_vision: bool = False
    supports_audio: bool = False
    
    # Performance characteristics (can be estimated)
    tokens_per_second: Optional[float] = None
    time_to_first_token_ms: Optional[float] = None


class ModelPricing(BaseModel):
    """Pricing information for a model."""
    input_token_price: float = Field(..., description="Price per input token in USD")
    output_token_price: float = Field(..., description="Price per output token in USD")
    currency: str = "USD"
    
    # For models with different pricing tiers
    pricing_tiers: Optional[Dict[str, Dict[str, float]]] = None


class ModelInfo(BaseModel):
    """Complete information about a model."""
    provider: ModelProvider
    model_id: str = Field(..., description="Unique identifier for the model")
    name: str = Field(..., description="Human-readable name")
    version: Optional[str] = None
    description: Optional[str] = None
    
    capabilities: ModelCapabilities
    pricing: ModelPricing
    
    # Metadata
    parameters: Optional[int] = Field(None, description="Number of parameters")
    release_date: Optional[str] = None
    deprecated: bool = False
    
    # Custom metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def full_id(self) -> str:
        """Get the full model identifier including provider."""
        return f"{self.provider.value}:{self.model_id}"
    
    def __hash__(self):
        return hash(self.full_id)
    
    def __eq__(self, other):
        if isinstance(other, ModelInfo):
            return self.full_id == other.full_id
        return False
