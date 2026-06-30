"""Schemas for routing decisions and strategies."""

from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class RoutingStrategy(str, Enum):
    """Available routing strategies."""
    COST_OPTIMIZED = "cost_optimized"  # Cheapest model that meets quality threshold
    LATENCY_OPTIMIZED = "latency_optimized"  # Fastest model that meets quality threshold
    QUALITY_OPTIMIZED = "quality_optimized"  # Best quality within budget
    BALANCED = "balanced"  # Weighted combination of cost, latency, quality
    FALLBACK = "fallback"  # Primary with automatic fallback
    RANDOM = "random"  # Random selection (for testing)
    ROUND_ROBIN = "round_robin"  # Distribute evenly across models
    CUSTOM = "custom"  # Custom routing logic


class RoutingCriteria(BaseModel):
    """Criteria for routing decisions."""
    # Quality thresholds (0-1)
    min_relevance_score: float = 0.7
    min_accuracy_score: float = 0.7
    min_coherence_score: float = 0.7
    
    # Cost constraints
    max_cost_usd: Optional[float] = None
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    
    # Latency constraints (milliseconds)
    max_latency_ms: Optional[float] = None
    max_time_to_first_token_ms: Optional[float] = None
    
    # Model constraints
    allowed_providers: Optional[List[str]] = None
    allowed_models: Optional[List[str]] = None
    blocked_providers: Optional[List[str]] = None
    blocked_models: Optional[List[str]] = None
    
    # Required capabilities
    requires_chat: bool = False
    requires_streaming: bool = False
    requires_function_calls: bool = False
    requires_embeddings: bool = False
    
    # Context constraints
    min_context_window: Optional[int] = None
    
    # Strategy weights (for BALANCED strategy)
    cost_weight: float = 0.4
    latency_weight: float = 0.3
    quality_weight: float = 0.3
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FallbackConfig(BaseModel):
    """Configuration for fallback routing."""
    primary_model: str
    fallback_models: List[str] = Field(default_factory=list)
    
    # Conditions for fallback
    fallback_on_error: bool = True
    fallback_on_timeout: bool = True
    fallback_on_quality_failure: bool = True
    fallback_on_cost_exceeded: bool = False
    
    # Timeout settings
    primary_timeout_seconds: float = 30.0
    fallback_timeout_seconds: float = 60.0
    
    # Quality threshold for fallback
    min_quality_score: float = 0.5
    
    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RoutingDecision(BaseModel):
    """The result of a routing decision."""
    request_id: str
    session_id: Optional[str] = None
    
    # Selected model
    selected_model: str  # logical model name the caller asked for
    selected_provider: str
    selected_deployment: Optional[str] = None  # unique deployment that served it
    
    # Strategy used
    strategy: RoutingStrategy
    criteria: RoutingCriteria
    
    # Reasoning
    reasoning: str
    confidence: float  # 0-1 confidence in the decision
    
    # Candidates considered
    candidates: List[str] = Field(default_factory=list)
    candidate_scores: Dict[str, float] = Field(default_factory=dict)
    
    # Fallback info
    is_fallback: bool = False
    fallback_reason: Optional[str] = None
    original_model: Optional[str] = None
    
    # Timing
    decision_time_ms: float
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def full_model_id(self) -> str:
        """Unique deployment id that served the request (registry key)."""
        if self.selected_deployment:
            return self.selected_deployment
        return f"{self.selected_provider}:{self.selected_model}"
