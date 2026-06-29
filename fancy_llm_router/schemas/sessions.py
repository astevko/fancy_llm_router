"""Schemas for session management and prompt chaining."""

from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime


class SessionStatus(str, Enum):
    """Status of a session."""
    CREATED = "created"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class ChainStepStatus(str, Enum):
    """Status of a step in a prompt chain."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ChainStep(BaseModel):
    """A single step in a prompt chain."""
    step_id: str
    step_name: str
    
    # Prompt configuration
    prompt_template: str
    prompt_variables: Dict[str, Any] = Field(default_factory=dict)
    
    # Model selection (can be overridden by router)
    model: Optional[str] = None
    provider: Optional[str] = None
    
    # Routing criteria for this step
    routing_criteria: Optional[Dict[str, Any]] = None
    
    # Execution
    status: ChainStepStatus = ChainStepStatus.PENDING
    request_id: Optional[str] = None
    response: Optional[str] = None
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Dependencies
    depends_on: List[str] = Field(default_factory=list)
    
    # Conditional execution
    condition: Optional[str] = None  # Python expression to evaluate
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PromptChain(BaseModel):
    """A chain of prompts that form a workflow."""
    chain_id: str
    chain_name: str
    description: Optional[str] = None
    
    # Steps in the chain
    steps: List[ChainStep] = Field(default_factory=list)
    
    # Entry and exit points
    entry_step_id: Optional[str] = None
    exit_step_id: Optional[str] = None
    
    # Execution settings
    max_concurrent_steps: int = 1
    continue_on_failure: bool = False
    
    # Input/output schema
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def get_entry_step(self) -> Optional[ChainStep]:
        """Get the entry step."""
        if self.entry_step_id:
            for step in self.steps:
                if step.step_id == self.entry_step_id:
                    return step
        return None
    
    def get_exit_step(self) -> Optional[ChainStep]:
        """Get the exit step."""
        if self.exit_step_id:
            for step in self.steps:
                if step.step_id == self.exit_step_id:
                    return step
        return None


class SessionConfig(BaseModel):
    """Configuration for a session."""
    session_id: Optional[str] = None  # Generated if not provided
    
    # Chain to execute
    chain: Optional[PromptChain] = None
    chain_id: Optional[str] = None
    
    # Initial input
    input_data: Dict[str, Any] = Field(default_factory=dict)
    
    # Model selection
    default_model: Optional[str] = None
    default_provider: Optional[str] = None
    
    # Routing
    routing_strategy: Optional[str] = None
    routing_criteria: Optional[Dict[str, Any]] = None
    
    # Constraints
    max_total_cost: Optional[float] = None
    max_total_tokens: Optional[int] = None
    max_total_time: Optional[float] = None  # seconds
    
    # Metadata
    user_id: Optional[str] = None
    git_commit: Optional[str] = None
    environment: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SessionState(BaseModel):
    """Runtime state of a session."""
    session_id: str
    config: SessionConfig
    
    # Status
    status: SessionStatus = SessionStatus.CREATED
    
    # Current state
    current_step_id: Optional[str] = None
    completed_steps: List[str] = Field(default_factory=list)
    failed_steps: List[str] = Field(default_factory=list)
    
    # Data
    variables: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    errors: Dict[str, str] = Field(default_factory=dict)
    
    # Metrics
    total_cost: float = 0.0
    total_tokens: int = 0
    total_time_seconds: float = 0.0
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Request tracking
    request_ids: List[str] = Field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def get_current_step(self) -> Optional[ChainStep]:
        """Get the current step being executed."""
        if self.current_step_id and self.config.chain:
            for step in self.config.chain.steps:
                if step.step_id == self.current_step_id:
                    return step
        return None
    
    def is_complete(self) -> bool:
        """Check if the session is complete."""
        return self.status in [SessionStatus.COMPLETED, SessionStatus.FAILED, SessionStatus.TIMEOUT]
