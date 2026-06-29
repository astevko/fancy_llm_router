"""Schemas for LLM requests and responses."""

from enum import Enum
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime


class MessageRole(str, Enum):
    """Role of a message in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""
    role: MessageRole
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    
    # For tool/function messages
    tool_calls: Optional[List[Dict[str, Any]]] = None
    
    # Metadata
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CompletionRequest(BaseModel):
    """Request for a text completion."""
    model: str
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    echo: bool = False
    
    # Metadata for tracking
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    prompt_hash: Optional[str] = None
    git_commit: Optional[str] = None
    
    # Extra provider-specific parameters
    extra: Dict[str, Any] = Field(default_factory=dict)


class CompletionChoice(BaseModel):
    """A single completion choice."""
    text: str
    index: int = 0
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None


class CompletionResponse(BaseModel):
    """Response from a text completion."""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Dict[str, int]
    
    # Our metadata
    request_id: Optional[str] = None
    latency_ms: Optional[float] = None
    

class ChatRequest(BaseModel):
    """Request for a chat completion."""
    model: str
    messages: List[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    
    # Tool/function calling
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[str] = None
    
    # Metadata for tracking
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    prompt_hash: Optional[str] = None
    git_commit: Optional[str] = None
    
    # Extra provider-specific parameters
    extra: Dict[str, Any] = Field(default_factory=dict)


class ChatChoice(BaseModel):
    """A single chat completion choice."""
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatResponse(BaseModel):
    """Response from a chat completion."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Dict[str, int]
    
    # Our metadata
    request_id: Optional[str] = None
    latency_ms: Optional[float] = None


class EmbeddingRequest(BaseModel):
    """Request for embeddings."""
    model: str
    input: Union[str, List[str]]
    user: Optional[str] = None
    
    # Metadata for tracking
    request_id: Optional[str] = None
    session_id: Optional[str] = None


class EmbeddingData(BaseModel):
    """A single embedding result."""
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    """Response from an embedding request."""
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: Dict[str, int]
    
    # Our metadata
    request_id: Optional[str] = None
    latency_ms: Optional[float] = None
