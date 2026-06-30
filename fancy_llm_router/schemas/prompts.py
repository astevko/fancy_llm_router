"""Schemas for prompt lineage and baseline measurement."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PromptRoot(BaseModel):
    """A generic root prompt from the client catalog."""

    root_id: str
    generic_text: str
    generic_hash: str
    category: Optional[str] = None
    expected_answer: Optional[str] = None
    source: str = "burner"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PromptVariant(BaseModel):
    """A deployment-specific specialized prompt derived from a root."""

    variant_id: str
    root_id: str
    deployment_id: str
    revision: int = 1
    parent_variant_id: Optional[str] = None
    prompt_text: str
    prompt_hash: str
    mutation_reason: Optional[str] = None
    judge_passed: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaselineRun(BaseModel):
    """A batch of baseline measurements."""

    run_id: str
    run_type: str = "smoke"
    prompt_scope: str = "single"
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    config_snapshot: Dict[str, Any] = Field(default_factory=dict)


class JudgeResult(BaseModel):
    """Outcome from evaluating a response."""

    pass_: bool = Field(alias="pass")
    accuracy_score: float = 0.0
    relevance_score: Optional[float] = None
    rationale: str = ""
    warnings: List[str] = Field(default_factory=list)

    class Config:
        populate_by_name = True


class BaselineResult(BaseModel):
    """One measured response for a root/variant on a deployment."""

    result_id: str
    run_id: str
    root_id: str
    deployment_id: str
    variant_id: Optional[str] = None
    generic_prompt: str
    prompt_used: str
    response_text: str
    response_hash: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    latency_ms: float = 0.0
    judge: JudgeResult
    is_canonical: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ResolvedPrompt(BaseModel):
    """Prompt text chosen for inference."""

    root_id: str
    deployment_id: str
    generic_text: str
    prompt_text: str
    variant_id: Optional[str] = None


class MeasurementEnvelope(BaseModel):
    """Returned to clients after a measure request."""

    benchmark_run_id: str
    root_id: Optional[str] = None
    deployment_id: str
    variant_id: Optional[str] = None
    prompt_used: str
    generic_prompt: Optional[str] = None
    judge: JudgeResult
    baseline_result_id: str
    optimized: bool = False
    revision: int = 0


class BenchmarkRunRequest(BaseModel):
    """Start a benchmark run from the client."""

    run_type: str = "smoke"
    prompt_scope: str = "single"
    optimize: bool = True
    max_revisions: int = 3


class BenchmarkMeasureRequest(BaseModel):
    """Measure one prompt against one deployment."""

    root_id: str
    prompt: str
    expected_answer: Optional[str] = None
    deployment_id: str
    benchmark_run_id: Optional[str] = None
    category: Optional[str] = None
    optimize: bool = True
    max_revisions: int = 3
    max_tokens: int = 256
    temperature: float = 0.0


class DeploymentInfo(BaseModel):
    """Summary returned by GET /deployments."""

    deployment_id: str
    logical_model: str
    source: Optional[str] = None
    provider: str
