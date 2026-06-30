"""Benchmark measurement, judging, and prompt optimization."""

import copy
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from fancy_llm_router.core.evaluator import LLEvaluator
from fancy_llm_router.core.optimizer import PromptOptimizer
from fancy_llm_router.core.router import LLMRouter
from fancy_llm_router.metrics.collector import MetricsCollector
from fancy_llm_router.schemas.metrics import CostMetrics, ModelMetrics, QualityMetrics, TokenUsage
from fancy_llm_router.schemas.prompts import (
    BaselineResult,
    BenchmarkMeasureRequest,
    DeploymentInfo,
    JudgeResult,
    MeasurementEnvelope,
    ResolvedPrompt,
)
from fancy_llm_router.schemas.requests import CompletionRequest, CompletionResponse
from fancy_llm_router.schemas.routing import RoutingCriteria
from fancy_llm_router.core.prompt_registry import PromptRegistry
from fancy_llm_router.utils.hash_utils import prompt_hash

logger = logging.getLogger(__name__)


class BenchmarkService:
    """Drive baseline measurement and per-deployment prompt specialization."""

    def __init__(
        self,
        router: LLMRouter,
        registry: PromptRegistry,
        metrics: Optional[MetricsCollector] = None,
        evaluator: Optional[LLEvaluator] = None,
        optimizer: Optional[PromptOptimizer] = None,
    ):
        self.router = router
        self.registry = registry
        self.metrics = metrics
        self.evaluator = evaluator or LLEvaluator()
        self.optimizer = optimizer or PromptOptimizer()

    def list_deployments(self) -> List[DeploymentInfo]:
        deployments: List[DeploymentInfo] = []
        for dep_id in self.router.list_models():
            info = self.router.get_model_info(dep_id)
            if info is None:
                continue
            deployments.append(
                DeploymentInfo(
                    deployment_id=dep_id,
                    logical_model=info.logical_model,
                    source=info.source,
                    provider=info.provider.value,
                )
            )
        return deployments

    async def handle_completion(
        self,
        request: CompletionRequest,
    ) -> Dict[str, Any]:
        """Route infer vs measure and return API payload."""
        if request.intent == "measure":
            envelope, response, decision = await self.measure_request(request)
            return {
                "response": response.dict(),
                "model": response.model,
                "deployment": decision.selected_deployment if decision else request.model,
                "measurement": envelope.dict(by_alias=True),
            }

        exec_request = copy.deepcopy(request)
        response, decision, _metrics = await self._execute_with_metrics(exec_request)
        payload: Dict[str, Any] = {
            "response": response.dict(),
            "model": response.model,
            "deployment": decision.selected_deployment if decision else None,
        }
        if request.root_id:
            payload["root_id"] = request.root_id
            if exec_request.extra.get("variant_id"):
                payload["variant_id"] = exec_request.extra["variant_id"]
                payload["generic_prompt"] = exec_request.extra.get("generic_prompt", request.prompt)
                payload["prompt_used"] = exec_request.prompt
        return payload

    async def measure_request(
        self, request: CompletionRequest
    ) -> tuple[MeasurementEnvelope, CompletionResponse, Any]:
        if not request.model or request.model == "auto":
            raise ValueError("intent=measure requires a pinned deployment id in model")

        deployment_id = request.model
        if deployment_id not in self.router.list_models():
            raise ValueError(f"Unknown deployment: {deployment_id}")

        if not request.root_id:
            request.root_id = f"adhoc-{prompt_hash(request.prompt, 8)}"

        root = self.registry.ensure_root(
            root_id=request.root_id,
            generic_text=request.prompt,
            expected_answer=request.expected_answer,
            category=request.category,
        )

        run_id = request.benchmark_run_id or str(uuid.uuid4())
        self.registry.ensure_run(
            run_id=run_id,
            run_type="client",
            prompt_scope="deployments",
            config_snapshot={
                "root_id": request.root_id,
                "deployment_id": deployment_id,
            },
        )

        expected = request.expected_answer or root.expected_answer
        model_info = self.router.get_model_info(deployment_id)
        max_revisions = max(1, request.max_revisions)
        optimize = request.optimize_on_fail

        response: Optional[CompletionResponse] = None
        decision = None
        variant_id: Optional[str] = None
        revision = 0
        optimized = False
        judge = JudgeResult(pass_=False, accuracy_score=0.0, rationale="not run")
        response_text = ""
        result_id = str(uuid.uuid4())
        usage: Dict[str, int] = {}
        latency_ms = 0.0
        prompt_used = root.generic_text

        for attempt in range(max_revisions):
            if attempt == 0:
                # Canonical baseline always starts from the generic parent prompt.
                prompt_used = root.generic_text
            elif variant_id:
                variant = self.registry.get_variant(variant_id)
                prompt_used = variant.prompt_text if variant else root.generic_text
            else:
                resolved = self.registry.resolve_prompt(
                    request.root_id, deployment_id, root.generic_text
                )
                prompt_used = resolved.prompt_text
                if resolved.variant_id:
                    variant_id = resolved.variant_id
                    revision = (
                        self.registry.get_variant(variant_id).revision
                        if variant_id
                        else revision
                    )

            exec_request = CompletionRequest(
                model=deployment_id,
                prompt=prompt_used,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                intent="infer",
                root_id=None,
                request_id=request.request_id or str(uuid.uuid4()),
                session_id=request.session_id,
                prompt_hash=prompt_hash(prompt_used),
            )

            response, decision, req_metrics = await self._execute_with_metrics(exec_request)
            response_text = self._extract_completion_text(response)
            usage = dict(response.usage or {})
            latency_ms = response.latency_ms or (
                req_metrics.latency.time_to_complete_ms if req_metrics else 0.0
            )

            judge = self.evaluator.evaluate(root.generic_text, response_text, expected)

            result_id = str(uuid.uuid4())
            baseline = BaselineResult(
                result_id=result_id,
                run_id=run_id,
                root_id=request.root_id,
                deployment_id=deployment_id,
                variant_id=variant_id,
                generic_prompt=root.generic_text,
                prompt_used=prompt_used,
                response_text=response_text,
                response_hash=prompt_hash(response_text),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                total_cost=req_metrics.cost.total_cost if req_metrics else 0.0,
                latency_ms=latency_ms or 0.0,
                judge=judge,
                is_canonical=judge.pass_,
            )
            self.registry.store_result(baseline)

            if judge.pass_:
                if variant_id:
                    self.registry.mark_variant_passed(variant_id)
                break

            if not optimize or attempt >= max_revisions - 1:
                break

            new_prompt = self.optimizer.refactor(
                generic_prompt=root.generic_text,
                deployment_id=deployment_id,
                model_info=model_info,
                response=response_text,
                judge=judge,
                expected_answer=expected,
                revision=revision + attempt + 1,
            )
            variant = self.registry.save_variant(
                root_id=request.root_id,
                deployment_id=deployment_id,
                prompt_text=new_prompt,
                parent_variant_id=variant_id,
                mutation_reason=judge.rationale,
            )
            variant_id = variant.variant_id
            revision = variant.revision
            optimized = True

        if response is None:
            raise RuntimeError("measurement produced no response")

        envelope = MeasurementEnvelope(
            benchmark_run_id=run_id,
            root_id=request.root_id,
            deployment_id=deployment_id,
            variant_id=variant_id,
            prompt_used=prompt_used,
            generic_prompt=root.generic_text,
            judge=judge,
            baseline_result_id=result_id,
            optimized=optimized,
            revision=revision,
        )
        return envelope, response, decision

    async def measure_deployment(self, req: BenchmarkMeasureRequest) -> MeasurementEnvelope:
        request = CompletionRequest(
            intent="measure",
            model=req.deployment_id,
            prompt=req.prompt,
            root_id=req.root_id,
            expected_answer=req.expected_answer,
            benchmark_run_id=req.benchmark_run_id,
            category=req.category,
            optimize_on_fail=req.optimize,
            max_revisions=req.max_revisions,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
        envelope, _response, _decision = await self.measure_request(request)
        return envelope

    async def measure_all_deployments(
        self,
        root_id: str,
        prompt: str,
        expected_answer: Optional[str] = None,
        deployments: Optional[List[str]] = None,
        optimize: bool = True,
        max_revisions: int = 3,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        run_id = run_id or str(uuid.uuid4())
        self.registry.create_run(
            run_type="smoke",
            prompt_scope="all" if not deployments else "subset",
            run_id=run_id,
            config_snapshot={"deployments": deployments or "all"},
        )
        dep_ids = deployments or self.router.list_models()
        results: List[Dict[str, Any]] = []
        for dep_id in dep_ids:
            envelope = await self.measure_deployment(
                BenchmarkMeasureRequest(
                    root_id=root_id,
                    prompt=prompt,
                    expected_answer=expected_answer,
                    deployment_id=dep_id,
                    benchmark_run_id=run_id,
                    optimize=optimize,
                    max_revisions=max_revisions,
                )
            )
            results.append(envelope.dict(by_alias=True))
        self.registry.complete_run(run_id)
        return {"benchmark_run_id": run_id, "results": results}

    async def _execute_with_metrics(
        self,
        request: CompletionRequest,
    ):
        if (
            request.root_id
            and request.model
            and request.model in self.router.list_models()
            and request.intent != "measure"
        ):
            resolved = self.registry.resolve_prompt(
                request.root_id, request.model, request.prompt
            )
            if resolved.variant_id:
                request.extra["generic_prompt"] = request.prompt
                request.extra["variant_id"] = resolved.variant_id
                request.prompt = resolved.prompt_text

        metrics_obj = None
        if self.metrics:
            async with self.metrics.track_request(request, None) as metrics_obj:
                response = await self.router.execute(request, track_metrics=False)
                decision = self.router._last_routing_decision
                provider = (
                    self.router._providers.get(decision.selected_deployment)
                    if decision
                    else None
                )
                self._fill_metrics(metrics_obj, response, decision, provider)
        else:
            response = await self.router.execute(request, track_metrics=False)
            decision = self.router._last_routing_decision

        return response, decision, metrics_obj

    def _fill_metrics(self, metrics, response, decision, provider) -> None:
        if provider and provider.model_info:
            metrics.model_info = ModelMetrics(
                model_id=provider.model_info.full_id,
                model_provider=provider.model_info.provider.value,
                model_version=provider.model_info.version,
                model_parameters=provider.model_info.parameters,
                context_window=provider.model_info.capabilities.context_window,
            )
        usage = response.usage or {}
        metrics.token_usage = TokenUsage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )
        if provider and provider.model_info:
            pricing = provider.model_info.pricing
            in_cost = metrics.token_usage.prompt_tokens * pricing.input_token_price
            out_cost = metrics.token_usage.completion_tokens * pricing.output_token_price
            metrics.cost = CostMetrics(
                input_token_cost=in_cost,
                output_token_cost=out_cost,
                total_cost=in_cost + out_cost,
                input_token_price=pricing.input_token_price,
                output_token_price=pricing.output_token_price,
            )
        metrics.metadata["selected_deployment"] = decision.selected_deployment
        if response.latency_ms:
            metrics.latency.time_to_complete_ms = response.latency_ms

    @staticmethod
    def _extract_completion_text(response: CompletionResponse) -> str:
        if response.choices:
            return response.choices[0].text or ""
        return ""
