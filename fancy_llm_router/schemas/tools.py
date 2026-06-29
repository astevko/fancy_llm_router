"""Schemas for tools and external resources."""

from enum import Enum
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field
from datetime import datetime


class ToolType(str, Enum):
    """Types of tools."""
    FUNCTION = "function"  # Python function
    API = "api"  # External API call
    DATABASE = "database"  # Database query
    FILE = "file"  # File operations
    WEB = "web"  # Web scraping/search
    CUSTOM = "custom"  # Custom implementation


class ToolDefinition(BaseModel):
    """Definition of a tool that can be used by LLMs."""
    tool_id: str = Field(..., description="Unique identifier for the tool")
    name: str = Field(..., description="Human-readable name")
    tool_type: ToolType
    description: str = Field(..., description="Description of what the tool does")
    
    # For function tools
    function_name: Optional[str] = None
    module_path: Optional[str] = None
    
    # For API tools
    api_endpoint: Optional[str] = None
    api_method: Optional[str] = None
    api_headers: Optional[Dict[str, str]] = None
    
    # Parameters
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema for tool parameters"
    )
    required_parameters: List[str] = Field(default_factory=list)
    
    # Return schema
    return_schema: Optional[Dict[str, Any]] = None
    
    # Execution
    timeout_seconds: float = 30.0
    max_retries: int = 3
    
    # Security
    is_safe: bool = True
    requires_auth: bool = False
    allowed_in_sandbox: bool = True
    
    # Cost tracking
    cost_per_call: Optional[float] = None
    cost_currency: str = "USD"
    
    # Metadata
    version: str = "1.0"
    author: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Availability
    is_available: bool = True
    last_checked: Optional[datetime] = None


class ToolCall(BaseModel):
    """A call to a tool from an LLM."""
    call_id: str = Field(..., description="Unique call identifier")
    tool_id: str
    tool_name: str
    
    # Arguments
    arguments: Dict[str, Any] = Field(default_factory=dict)
    
    # Context
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    model_id: Optional[str] = None
    
    # Timing
    called_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result from a tool call."""
    call_id: str
    tool_id: str
    tool_name: str
    
    # Result
    output: Any
    is_error: bool = False
    error_message: Optional[str] = None
    
    # Timing
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    
    # Cost
    cost: Optional[float] = None
    cost_currency: str = "USD"
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        return not self.is_error
