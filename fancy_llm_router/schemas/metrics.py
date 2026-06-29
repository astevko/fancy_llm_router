"""Schemas for metrics collection and tracking."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class TokenUsage(BaseModel):
    """Token usage metrics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    @property
    def input_tokens(self) -> int:
        return self.prompt_tokens
    
    @property
    def output_tokens(self) -> int:
        return self.completion_tokens


class CostMetrics(BaseModel):
    """Cost-related metrics."""
    input_token_cost: float = 0.0
    output_token_cost: float = 0.0
    total_cost: float = 0.0
    currency: str = "USD"
    
    # Pricing used for calculation
    input_token_price: Optional[float] = None
    output_token_price: Optional[float] = None


class LatencyMetrics(BaseModel):
    """Latency and performance metrics."""
    time_to_first_token_ms: Optional[float] = None
    time_to_complete_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None
    
    # Additional timing breakdown
    request_queue_time_ms: Optional[float] = None
    model_inference_time_ms: Optional[float] = None
    network_time_ms: Optional[float] = None


class QualityMetrics(BaseModel):
    """Quality and accuracy metrics."""
    # Automated evaluation scores (0-1)
    relevance_score: Optional[float] = None
    accuracy_score: Optional[float] = None
    coherence_score: Optional[float] = None
    helpfulness_score: Optional[float] = None
    
    # Human evaluation (if available)
    human_rating: Optional[float] = None
    human_feedback: Optional[str] = None
    
    # Benchmark comparison
    benchmark_score: Optional[float] = None
    benchmark_name: Optional[str] = None
    
    # Confidence metrics
    model_confidence: Optional[float] = None
    evaluator_confidence: Optional[float] = None
    
    # Error tracking
    is_error: bool = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None


class ModelMetrics(BaseModel):
    """Metrics specific to the model used."""
    model_id: str
    model_provider: str
    model_version: Optional[str] = None
    model_parameters: Optional[int] = None
    context_window: Optional[int] = None
    
    # Drift detection
    performance_drift: Optional[float] = None  # Change from baseline
    price_drift: Optional[float] = None  # Change in pricing
    version_drift: Optional[str] = None  # Version change detected


class RequestMetrics(BaseModel):
    """Complete metrics for a single LLM request."""
    # Identifiers
    request_id: str = Field(..., description="Unique request identifier")
    session_id: Optional[str] = None
    prompt_hash: str = Field(..., description="Hash of the prompt for deduplication")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # Request details
    request_type: str  # "completion", "chat", "embedding"
    model_info: ModelMetrics
    
    # Usage and cost
    token_usage: TokenUsage
    cost: CostMetrics
    latency: LatencyMetrics
    quality: QualityMetrics
    
    # Context
    git_commit: Optional[str] = None
    environment: Optional[str] = None
    user_id: Optional[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    
    @property
    def total_cost(self) -> float:
        return self.cost.total_cost
    
    @property
    def total_tokens(self) -> int:
        return self.token_usage.total_tokens
    
    @property
    def total_latency_ms(self) -> Optional[float]:
        return self.latency.time_to_complete_ms


class SessionMetrics(BaseModel):
    """Aggregated metrics for a session (chain of prompts)."""
    session_id: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    
    # Aggregated metrics
    total_requests: int = 0
    total_token_usage: TokenUsage = Field(default_factory=TokenUsage)
    total_cost: CostMetrics = Field(default_factory=CostMetrics)
    total_latency: LatencyMetrics = Field(default_factory=LatencyMetrics)
    
    # Average metrics
    avg_quality: QualityMetrics = Field(default_factory=QualityMetrics)
    
    # Individual request metrics
    request_metrics: List[RequestMetrics] = Field(default_factory=list)
    
    # Session-specific info
    chain_length: int = 0
    is_complete: bool = False
    final_output: Optional[str] = None
    
    # Metadata
    git_commit: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkMetrics(BaseModel):
    """Metrics for benchmark evaluations."""
    benchmark_id: str
    benchmark_name: str
    model_info: ModelMetrics
    
    # Aggregated across multiple runs
    avg_token_usage: TokenUsage
    avg_cost: CostMetrics
    avg_latency: LatencyMetrics
    avg_quality: QualityMetrics
    
    # Statistical measures
    std_cost: Optional[float] = None
    std_latency: Optional[float] = None
    std_quality: Optional[float] = None
    
    # Number of runs
    num_runs: int = 1
    
    # Comparison to baseline
    baseline_model_id: Optional[str] = None
    cost_savings_pct: Optional[float] = None
    quality_improvement_pct: Optional[float] = None
    latency_improvement_pct: Optional[float] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
