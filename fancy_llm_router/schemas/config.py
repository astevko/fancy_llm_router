"""Schemas for configuration files."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from pathlib import Path


class ModelConfig(BaseModel):
    """Configuration for a single model."""
    provider: str
    model_id: str
    name: Optional[str] = None
    
    # API configuration
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None
    api_version: Optional[str] = None
    
    # Model settings
    default_temperature: float = 0.7
    default_max_tokens: int = 256
    default_top_p: float = 1.0
    
    # Pricing (can be auto-detected)
    input_token_price: Optional[float] = None
    output_token_price: Optional[float] = None
    
    # Capabilities (can be auto-detected)
    max_tokens: Optional[int] = None
    context_window: Optional[int] = None
    supports_chat: bool = True
    supports_completions: bool = True
    supports_streaming: bool = True
    supports_embeddings: bool = False

    # Performance and serving characteristics
    tokens_per_second: Optional[float] = None
    quantization: Optional[str] = None
    
    # Timeouts
    timeout_seconds: float = 60.0
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Priority (for routing)
    priority: int = 0
    
    # Enabled/disabled
    enabled: bool = True
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ToolConfig(BaseModel):
    """Configuration for tools."""
    # Tool module path
    module: str
    
    # Or direct configuration
    tool_id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    tool_type: Optional[str] = None
    
    # API configuration (for API tools)
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    
    # Settings
    timeout_seconds: float = 30.0
    max_retries: int = 3
    
    # Enabled/disabled
    enabled: bool = True
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MetricsConfig(BaseModel):
    """Configuration for metrics collection."""
    # Storage
    enabled: bool = True
    
    # What to track
    track_token_usage: bool = True
    track_cost: bool = True
    track_latency: bool = True
    track_quality: bool = True
    track_model_drift: bool = True
    
    # Quality evaluation
    evaluate_relevance: bool = True
    evaluate_accuracy: bool = True
    evaluate_coherence: bool = True
    
    # Drift detection
    drift_detection_window_days: int = 30
    drift_threshold: float = 0.1  # 10% change triggers alert
    
    # Benchmarking
    benchmark_interval_hours: int = 24
    benchmark_models: List[str] = Field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StorageConfig(BaseModel):
    """Configuration for storage backends."""
    # Primary storage
    backend: str = "sqlite"  # sqlite, postgres, mysql, etc.
    
    # SQLite configuration
    sqlite_path: str = "data/metrics.db"
    
    # Postgres configuration
    postgres_host: Optional[str] = None
    postgres_port: int = 5432
    postgres_db: Optional[str] = None
    postgres_user: Optional[str] = None
    postgres_password: Optional[str] = None
    
    # Connection pooling
    pool_size: int = 5
    max_overflow: int = 10
    
    # Retention
    retention_days: Optional[int] = None  # None = keep forever
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RouterConfig(BaseModel):
    """Configuration for the router."""
    # Default routing strategy
    default_strategy: str = "balanced"
    
    # Fallback configuration
    use_fallback: bool = True
    fallback_models: List[str] = Field(default_factory=list)
    
    # Routing criteria
    default_criteria: Dict[str, Any] = Field(default_factory=dict)
    
    # Model selection
    default_model: Optional[str] = None
    
    # Load balancing
    load_balance: bool = False
    
    # Caching
    cache_responses: bool = True
    cache_ttl_seconds: int = 3600
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AppConfig(BaseSettings):
    """Main application configuration."""
    # Application
    app_name: str = "fancy_llm_router"
    app_version: str = "0.1.0"
    environment: str = "development"  # development, staging, production
    debug: bool = False
    log_level: str = "INFO"
    
    # Paths
    config_path: Path = Path("configs")
    data_path: Path = Path("data")
    logs_path: Path = Path("logs")
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Models
    models: Dict[str, ModelConfig] = Field(default_factory=dict)
    
    # Tools
    tools: Dict[str, ToolConfig] = Field(default_factory=dict)
    
    # Metrics
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    
    # Storage
    storage: StorageConfig = Field(default_factory=StorageConfig)
    
    # Router
    router: RouterConfig = Field(default_factory=RouterConfig)
    
    # API keys (can be overridden by environment variables)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None
    mistral_api_key: Optional[str] = None
    
    # Security
    api_auth_token: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
