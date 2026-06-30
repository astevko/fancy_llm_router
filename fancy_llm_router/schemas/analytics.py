"""Schemas for baseline analytics summaries and dashboards."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DeploymentBreakdown(BaseModel):
    """One measured deployment for a parent prompt."""

    result_id: str
    deployment_id: str
    variant_id: Optional[str] = None
    generic_prompt: str
    prompt_used: str
    response_text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    latency_ms: float = 0.0
    judge_pass: bool = False
    judge_accuracy: float = 0.0
    judge_rationale: str = ""
    judge_warnings: List[str] = Field(default_factory=list)
    is_canonical: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MetricStats(BaseModel):
    """Min / max / avg / median / total for a numeric metric."""

    total: float = 0.0
    avg: float = 0.0
    median: float = 0.0
    min: float = 0.0
    max: float = 0.0


class RootPromptSummary(BaseModel):
    """Aggregated analytics for one parent (root) prompt."""

    root_id: str
    generic_prompt: str
    category: Optional[str] = None
    expected_answer: Optional[str] = None
    deployment_count: int = 0
    pass_count: int = 0
    fail_count: int = 0
    cost: MetricStats = Field(default_factory=MetricStats)
    latency_ms: MetricStats = Field(default_factory=MetricStats)
    prompt_tokens: MetricStats = Field(default_factory=MetricStats)
    completion_tokens: MetricStats = Field(default_factory=MetricStats)
    total_tokens: MetricStats = Field(default_factory=MetricStats)
    by_deployment: List[DeploymentBreakdown] = Field(default_factory=list)


class BaselineRunSummary(BaseModel):
    """Full analytics payload for a benchmark run."""

    run_id: str
    run_type: str = "smoke"
    prompt_scope: str = "single"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    root_count: int = 0
    result_count: int = 0
    pass_count: int = 0
    fail_count: int = 0
    cost: MetricStats = Field(default_factory=MetricStats)
    latency_ms: MetricStats = Field(default_factory=MetricStats)
    total_tokens: MetricStats = Field(default_factory=MetricStats)
    roots: List[RootPromptSummary] = Field(default_factory=list)


class BaselineRunInfo(BaseModel):
    """Lightweight run listing entry."""

    run_id: str
    run_type: str
    prompt_scope: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result_count: int = 0
    root_count: int = 0


class RootPromptInfo(BaseModel):
    """Parent prompt with cross-run measurement counts."""

    root_id: str
    generic_prompt: str
    category: Optional[str] = None
    expected_answer: Optional[str] = None
    run_count: int = 0
    result_count: int = 0
    last_measured_at: Optional[datetime] = None


class RunPromptBreakdown(BaseModel):
    """One benchmark run's deployments for a single parent prompt."""

    run_id: str
    run_type: str = "smoke"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    deployment_count: int = 0
    pass_count: int = 0
    fail_count: int = 0
    cost: MetricStats = Field(default_factory=MetricStats)
    latency_ms: MetricStats = Field(default_factory=MetricStats)
    total_tokens: MetricStats = Field(default_factory=MetricStats)
    by_deployment: List[DeploymentBreakdown] = Field(default_factory=list)


class DeploymentOption(BaseModel):
    """A deployment measured for a parent prompt."""

    deployment_id: str
    run_count: int = 0
    pass_count: int = 0
    fail_count: int = 0
    last_measured_at: Optional[datetime] = None
    state: str = "unset"


class RunMeasurement(BaseModel):
    """One benchmark run's result for a prompt + deployment pair."""

    run_id: str
    run_type: str = "smoke"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: DeploymentBreakdown


class PromptDeploymentHistory(BaseModel):
    """All benchmark runs for one parent prompt on one deployment."""

    root_id: str
    generic_prompt: str
    category: Optional[str] = None
    expected_answer: Optional[str] = None
    deployment_id: str
    state: str = "unset"
    run_count: int = 0
    pass_count: int = 0
    fail_count: int = 0
    cost: MetricStats = Field(default_factory=MetricStats)
    latency_ms: MetricStats = Field(default_factory=MetricStats)
    total_tokens: MetricStats = Field(default_factory=MetricStats)
    by_run: List[RunMeasurement] = Field(default_factory=list)
