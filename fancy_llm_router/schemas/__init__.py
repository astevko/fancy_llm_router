"""Pydantic schemas for the LLM Router system."""

from fancy_llm_router.schemas.config import (
    ModelConfig,
    RouterConfig,
    ToolConfig,
    MetricsConfig,
    StorageConfig,
    AppConfig,
)
from fancy_llm_router.schemas.metrics import (
    TokenUsage,
    CostMetrics,
    LatencyMetrics,
    QualityMetrics,
    ModelMetrics,
    RequestMetrics,
    SessionMetrics,
    BenchmarkMetrics,
)
from fancy_llm_router.schemas.models import (
    ModelProvider,
    ModelCapabilities,
    ModelPricing,
    ModelInfo,
)
from fancy_llm_router.schemas.requests import (
    CompletionRequest,
    CompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ChatMessage,
    ChatRequest,
    ChatResponse,
)
from fancy_llm_router.schemas.routing import (
    RoutingDecision,
    RoutingCriteria,
    RoutingStrategy,
    FallbackConfig,
)
from fancy_llm_router.schemas.sessions import (
    SessionConfig,
    SessionState,
    PromptChain,
    ChainStep,
)
from fancy_llm_router.schemas.tools import ToolDefinition, ToolCall, ToolResult

__all__ = [
    # Config
    "ModelConfig",
    "RouterConfig",
    "ToolConfig",
    "MetricsConfig",
    "StorageConfig",
    "AppConfig",
    # Metrics
    "TokenUsage",
    "CostMetrics",
    "LatencyMetrics",
    "QualityMetrics",
    "ModelMetrics",
    "RequestMetrics",
    "SessionMetrics",
    "BenchmarkMetrics",
    # Models
    "ModelProvider",
    "ModelCapabilities",
    "ModelPricing",
    "ModelInfo",
    # Requests
    "CompletionRequest",
    "CompletionResponse",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    # Routing
    "RoutingDecision",
    "RoutingCriteria",
    "RoutingStrategy",
    "FallbackConfig",
    # Sessions
    "SessionConfig",
    "SessionState",
    "PromptChain",
    "ChainStep",
    # Tools
    "ToolDefinition",
    "ToolCall",
    "ToolResult",
]
