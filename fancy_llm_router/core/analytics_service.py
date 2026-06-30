"""Aggregate baseline measurement results for analytics dashboards."""

from collections import defaultdict
from typing import Iterable, List, Optional

from fancy_llm_router.core.prompt_registry import PromptRegistry
from fancy_llm_router.schemas.analytics import (
    BaselineRunInfo,
    BaselineRunSummary,
    DeploymentBreakdown,
    MetricStats,
    RootPromptSummary,
)
from fancy_llm_router.schemas.prompts import BaselineResult, BaselineRun


def _metric_stats(values: Iterable[float]) -> MetricStats:
    nums = list(values)
    if not nums:
        return MetricStats()
    return MetricStats(
        total=sum(nums),
        avg=sum(nums) / len(nums),
        min=min(nums),
        max=max(nums),
    )


def _deployment_from_result(result: BaselineResult) -> DeploymentBreakdown:
    return DeploymentBreakdown(
        result_id=result.result_id,
        deployment_id=result.deployment_id,
        variant_id=result.variant_id,
        generic_prompt=result.generic_prompt,
        prompt_used=result.prompt_used,
        response_text=result.response_text,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.total_tokens,
        total_cost=result.total_cost,
        latency_ms=result.latency_ms,
        judge_pass=result.judge.pass_,
        judge_accuracy=result.judge.accuracy_score,
        judge_rationale=result.judge.rationale,
        judge_warnings=result.judge.warnings,
        is_canonical=result.is_canonical,
        created_at=result.created_at,
    )


def build_run_summary(
    run: BaselineRun,
    results: List[BaselineResult],
    registry: PromptRegistry,
) -> BaselineRunSummary:
    """Group baseline results by parent prompt with per-deployment telemetry."""
    by_root: dict[str, List[BaselineResult]] = defaultdict(list)
    for result in results:
        by_root[result.root_id].append(result)

    roots: List[RootPromptSummary] = []
    for root_id in sorted(by_root):
        root_results = by_root[root_id]
        root_meta = registry.get_root(root_id)
        generic_prompt = (
            root_meta.generic_text
            if root_meta
            else (root_results[0].generic_prompt if root_results else "")
        )
        deployments = [_deployment_from_result(r) for r in root_results]
        pass_count = sum(1 for d in deployments if d.judge_pass)
        roots.append(
            RootPromptSummary(
                root_id=root_id,
                generic_prompt=generic_prompt,
                category=root_meta.category if root_meta else None,
                expected_answer=root_meta.expected_answer if root_meta else None,
                deployment_count=len(deployments),
                pass_count=pass_count,
                fail_count=len(deployments) - pass_count,
                cost=_metric_stats(d.total_cost for d in deployments),
                latency_ms=_metric_stats(d.latency_ms for d in deployments),
                prompt_tokens=_metric_stats(float(d.prompt_tokens) for d in deployments),
                completion_tokens=_metric_stats(float(d.completion_tokens) for d in deployments),
                total_tokens=_metric_stats(float(d.total_tokens) for d in deployments),
                by_deployment=deployments,
            )
        )

    all_deployments = [d for root in roots for d in root.by_deployment]
    pass_count = sum(1 for d in all_deployments if d.judge_pass)

    return BaselineRunSummary(
        run_id=run.run_id,
        run_type=run.run_type,
        prompt_scope=run.prompt_scope,
        started_at=run.started_at,
        completed_at=run.completed_at,
        root_count=len(roots),
        result_count=len(all_deployments),
        pass_count=pass_count,
        fail_count=len(all_deployments) - pass_count,
        cost=_metric_stats(d.total_cost for d in all_deployments),
        latency_ms=_metric_stats(d.latency_ms for d in all_deployments),
        total_tokens=_metric_stats(float(d.total_tokens) for d in all_deployments),
        roots=roots,
    )


class AnalyticsService:
    """Read-only analytics over the prompt registry."""

    def __init__(self, registry: PromptRegistry):
        self.registry = registry

    def list_runs(self, limit: int = 50) -> List[BaselineRunInfo]:
        runs = self.registry.list_runs(limit=limit)
        infos: List[BaselineRunInfo] = []
        for run in runs:
            results = self.registry.list_results(run.run_id)
            root_ids = {r.root_id for r in results}
            infos.append(
                BaselineRunInfo(
                    run_id=run.run_id,
                    run_type=run.run_type,
                    prompt_scope=run.prompt_scope,
                    started_at=run.started_at,
                    completed_at=run.completed_at,
                    result_count=len(results),
                    root_count=len(root_ids),
                )
            )
        return infos

    def get_run_summary(self, run_id: str) -> Optional[BaselineRunSummary]:
        run = self.registry.get_run(run_id)
        if run is None:
            return None
        results = self.registry.list_results(run_id)
        if not results:
            return None
        return build_run_summary(run, results, self.registry)
