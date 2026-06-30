"""Metrics collector for tracking LLM usage and performance."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Union
from datetime import datetime
from contextlib import asynccontextmanager

from fancy_llm_router.schemas.metrics import (
    RequestMetrics,
    SessionMetrics,
    BenchmarkMetrics,
    TokenUsage,
    CostMetrics,
    LatencyMetrics,
    QualityMetrics,
    ModelMetrics,
)
from fancy_llm_router.schemas.requests import (
    CompletionRequest,
    ChatRequest,
    EmbeddingRequest,
)
from fancy_llm_router.schemas.routing import RoutingDecision
from fancy_llm_router.models.base import BaseModelProvider

logger = logging.getLogger(__name__)


@dataclass
class MetricsConfig:
    """Configuration for metrics collection."""
    enabled: bool = True
    track_token_usage: bool = True
    track_cost: bool = True
    track_latency: bool = True
    track_quality: bool = True
    track_model_drift: bool = True
    
    # Quality evaluation
    evaluate_relevance: bool = True
    evaluate_accuracy: bool = True
    evaluate_coherence: bool = True
    
    # Storage
    storage_backend: str = "sqlite"
    storage_config: Dict[str, Any] = field(default_factory=dict)
    
    # Callbacks
    on_metrics_collected: Optional[Callable[[RequestMetrics], None]] = None
    on_session_completed: Optional[Callable[[SessionMetrics], None]] = None
    on_benchmark_completed: Optional[Callable[[BenchmarkMetrics], None]] = None


class MetricsCollector:
    """
    Collects and manages metrics for LLM requests.
    
    Features:
    - Automatic metrics collection for all LLM requests
    - Session-level aggregation
    - Benchmark tracking
    - Model drift detection
    - Custom callbacks for metrics events
    """
    
    def __init__(
        self,
        config: Optional[MetricsConfig] = None,
        storage: Optional[Any] = None,
        **kwargs
    ):
        self.config = config or MetricsConfig()
        self.storage = storage
        self.extra_config = kwargs
        
        # Metrics storage
        self._request_metrics: Dict[str, RequestMetrics] = {}
        self._session_metrics: Dict[str, SessionMetrics] = {}
        self._benchmark_metrics: Dict[str, BenchmarkMetrics] = {}
        
        # Aggregated metrics
        self._model_metrics: Dict[str, Dict[str, Any]] = {}
        self._daily_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Drift detection
        self._baseline_metrics: Dict[str, Dict[str, float]] = {}
        self._drift_alerts: Dict[str, List[Dict[str, Any]]] = {}
        
        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "on_metrics_collected": [],
            "on_session_completed": [],
            "on_benchmark_completed": [],
            "on_drift_detected": [],
        }
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
    
    def register_callback(
        self,
        event: str,
        callback: Callable
    ):
        """Register a callback for metrics events."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def _trigger_callbacks(
        self,
        event: str,
        *args,
        **kwargs
    ):
        """Trigger all callbacks for an event."""
        for callback in self._callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(*args, **kwargs))
                else:
                    callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in callback {callback}: {e}")
    
    @asynccontextmanager
    async def track_request(
        self,
        request: Union[CompletionRequest, ChatRequest, EmbeddingRequest],
        provider: BaseModelProvider,
        git_commit: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs
    ):
        """
        Context manager for tracking a single request.
        
        Usage:
            async with metrics_collector.track_request(request, provider) as metrics:
                response = await provider.chat(request)
                # metrics will be automatically populated
        """
        start_time = time.time()
        first_token_time = None
        request_id = request.request_id or f"req-{int(time.time())}"
        
        # Resolve model info from the provider when available, otherwise fall
        # back to the request (the API tracks before a provider is selected).
        if provider is not None:
            model_metrics = ModelMetrics(
                model_id=provider.model_id,
                model_provider=provider.provider.value,
                model_version=provider.model_info.version,
                model_parameters=provider.model_info.parameters,
                context_window=provider.model_info.capabilities.context_window,
            )
        else:
            model_metrics = ModelMetrics(
                model_id=getattr(request, "model", None) or "unknown",
                model_provider="unknown",
                context_window=0,
            )

        # Create initial metrics
        metrics = RequestMetrics(
            request_id=request_id,
            session_id=session_id,
            prompt_hash=self._hash_request(request),
            created_at=datetime.utcnow(),
            request_type=type(request).__name__.replace('Request', '').lower(),
            model_info=model_metrics,
            token_usage=TokenUsage(),
            cost=CostMetrics(),
            latency=LatencyMetrics(),
            quality=QualityMetrics(),
            git_commit=git_commit,
            metadata=kwargs,
        )
        
        try:
            yield metrics
            
            # Finalize metrics
            end_time = time.time()
            metrics.completed_at = datetime.utcnow()
            
            # Calculate latency
            metrics.latency.time_to_complete_ms = (end_time - start_time) * 1000
            if first_token_time:
                metrics.latency.time_to_first_token_ms = (first_token_time - start_time) * 1000
            
            # Store metrics
            await self._store_request_metrics(metrics)
            
            # Trigger callbacks
            if self.config.on_metrics_collected:
                self.config.on_metrics_collected(metrics)
            self._trigger_callbacks("on_metrics_collected", metrics)
            
        except Exception as e:
            metrics.quality.is_error = True
            metrics.quality.error_type = type(e).__name__
            metrics.quality.error_message = str(e)
            metrics.completed_at = datetime.utcnow()
            
            # Store error metrics
            await self._store_request_metrics(metrics)
            
            # Trigger callbacks
            if self.config.on_metrics_collected:
                self.config.on_metrics_collected(metrics)
            self._trigger_callbacks("on_metrics_collected", metrics)
            
            raise
    
    @asynccontextmanager
    async def track_session(
        self,
        session_id: str,
        git_commit: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Context manager for tracking a session.
        
        Usage:
            async with metrics_collector.track_session("session-1") as session_metrics:
                # Execute multiple requests
                async with metrics_collector.track_request(req1, provider) as req_metrics:
                    response1 = await provider.chat(req1)
                async with metrics_collector.track_request(req2, provider) as req_metrics:
                    response2 = await provider.chat(req2)
                # Session metrics will be automatically aggregated
        """
        start_time = time.time()
        
        session_metrics = SessionMetrics(
            session_id=session_id,
            created_at=datetime.utcnow(),
            git_commit=git_commit,
            metadata=metadata or {},
        )
        
        # Store session reference
        self._session_metrics[session_id] = session_metrics
        
        try:
            yield session_metrics
            
            # Finalize session metrics
            end_time = time.time()
            session_metrics.completed_at = datetime.utcnow()
            session_metrics.total_latency.time_to_complete_ms = (end_time - start_time) * 1000
            session_metrics.is_complete = True
            
            # Aggregate request metrics for this session
            request_metrics = [
                m for m in self._request_metrics.values()
                if m.session_id == session_id
            ]
            session_metrics.request_metrics = request_metrics
            session_metrics.total_requests = len(request_metrics)
            
            # Aggregate token usage
            for req_metrics in request_metrics:
                session_metrics.total_token_usage.prompt_tokens += req_metrics.token_usage.prompt_tokens
                session_metrics.total_token_usage.completion_tokens += req_metrics.token_usage.completion_tokens
                session_metrics.total_token_usage.total_tokens += req_metrics.token_usage.total_tokens
                
                session_metrics.total_cost.input_token_cost += req_metrics.cost.input_token_cost
                session_metrics.total_cost.output_token_cost += req_metrics.cost.output_token_cost
                session_metrics.total_cost.total_cost += req_metrics.cost.total_cost
            
            # Store session metrics
            await self._store_session_metrics(session_metrics)
            
            # Trigger callbacks
            if self.config.on_session_completed:
                self.config.on_session_completed(session_metrics)
            self._trigger_callbacks("on_session_completed", session_metrics)
            
        except Exception as e:
            session_metrics.is_complete = False
            session_metrics.completed_at = datetime.utcnow()
            
            # Store incomplete session
            await self._store_session_metrics(session_metrics)
            
            raise
    
    async def track_benchmark(
        self,
        benchmark_id: str,
        benchmark_name: str,
        model_info: ModelMetrics,
        num_runs: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Context manager for tracking benchmark runs.
        
        Usage:
            async with metrics_collector.track_benchmark("bench-1", "My Benchmark", model_info) as bench_metrics:
                for i in range(10):
                    async with metrics_collector.track_request(req, provider) as req_metrics:
                        response = await provider.chat(req)
                    bench_metrics.request_metrics.append(req_metrics)
        """
        start_time = time.time()
        
        benchmark_metrics = BenchmarkMetrics(
            benchmark_id=benchmark_id,
            benchmark_name=benchmark_name,
            model_info=model_info,
            num_runs=num_runs,
            created_at=datetime.utcnow(),
            metadata=metadata or {},
        )
        
        # Store benchmark reference
        self._benchmark_metrics[benchmark_id] = benchmark_metrics
        
        try:
            yield benchmark_metrics
            
            # Finalize benchmark metrics
            end_time = time.time()
            benchmark_metrics.updated_at = datetime.utcnow()
            
            # Calculate averages
            if benchmark_metrics.request_metrics:
                self._calculate_benchmark_averages(benchmark_metrics)
            
            # Store benchmark metrics
            await self._store_benchmark_metrics(benchmark_metrics)
            
            # Trigger callbacks
            if self.config.on_benchmark_completed:
                self.config.on_benchmark_completed(benchmark_metrics)
            self._trigger_callbacks("on_benchmark_completed", benchmark_metrics)
            
        except Exception as e:
            benchmark_metrics.updated_at = datetime.utcnow()
            
            # Store incomplete benchmark
            await self._store_benchmark_metrics(benchmark_metrics)
            
            raise
    
    def _calculate_benchmark_averages(self, benchmark_metrics: BenchmarkMetrics):
        """Calculate average metrics for a benchmark."""
        from fancy_llm_router.schemas.metrics import TokenUsage, CostMetrics, LatencyMetrics, QualityMetrics
        
        num_requests = len(benchmark_metrics.request_metrics)
        if num_requests == 0:
            return
        
        # Initialize sums
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_input_cost = 0.0
        total_output_cost = 0.0
        total_latency = 0.0
        total_ttft = 0.0
        
        total_relevance = 0.0
        total_accuracy = 0.0
        total_coherence = 0.0
        
        for req_metrics in benchmark_metrics.request_metrics:
            total_prompt_tokens += req_metrics.token_usage.prompt_tokens
            total_completion_tokens += req_metrics.token_usage.completion_tokens
            total_input_cost += req_metrics.cost.input_token_cost
            total_output_cost += req_metrics.cost.output_token_cost
            
            if req_metrics.latency.time_to_complete_ms:
                total_latency += req_metrics.latency.time_to_complete_ms
            if req_metrics.latency.time_to_first_token_ms:
                total_ttft += req_metrics.latency.time_to_first_token_ms
            
            if req_metrics.quality.relevance_score:
                total_relevance += req_metrics.quality.relevance_score
            if req_metrics.quality.accuracy_score:
                total_accuracy += req_metrics.quality.accuracy_score
            if req_metrics.quality.coherence_score:
                total_coherence += req_metrics.quality.coherence_score
        
        # Calculate averages
        benchmark_metrics.avg_token_usage = TokenUsage(
            prompt_tokens=total_prompt_tokens / num_requests,
            completion_tokens=total_completion_tokens / num_requests,
            total_tokens=(total_prompt_tokens + total_completion_tokens) / num_requests,
        )
        
        benchmark_metrics.avg_cost = CostMetrics(
            input_token_cost=total_input_cost / num_requests,
            output_token_cost=total_output_cost / num_requests,
            total_cost=(total_input_cost + total_output_cost) / num_requests,
        )
        
        benchmark_metrics.avg_latency = LatencyMetrics(
            time_to_first_token_ms=total_ttft / num_requests if num_requests > 0 else None,
            time_to_complete_ms=total_latency / num_requests if num_requests > 0 else None,
        )
        
        avg_relevance = total_relevance / num_requests if num_requests > 0 else None
        avg_accuracy = total_accuracy / num_requests if num_requests > 0 else None
        avg_coherence = total_coherence / num_requests if num_requests > 0 else None
        
        benchmark_metrics.avg_quality = QualityMetrics(
            relevance_score=avg_relevance,
            accuracy_score=avg_accuracy,
            coherence_score=avg_coherence,
        )
    
    async def _store_request_metrics(self, metrics: RequestMetrics):
        """Store request metrics."""
        async with self._lock:
            self._request_metrics[metrics.request_id] = metrics
            
            # Update model metrics
            model_id = metrics.model_info.model_id
            if model_id not in self._model_metrics:
                self._model_metrics[model_id] = {
                    "total_requests": 0,
                    "total_tokens": 0,
                    "total_cost": 0.0,
                    "total_latency": 0.0,
                    "avg_quality": 0.0,
                    "last_request": None,
                }
            
            model_metrics = self._model_metrics[model_id]
            model_metrics["total_requests"] += 1
            model_metrics["total_tokens"] += metrics.token_usage.total_tokens
            model_metrics["total_cost"] += metrics.cost.total_cost
            model_metrics["total_latency"] += metrics.latency.time_to_complete_ms or 0
            model_metrics["last_request"] = metrics.created_at
            
            # Update average quality (simple moving average)
            if metrics.quality.relevance_score:
                current_avg = model_metrics.get("avg_quality", 0.0)
                new_avg = (current_avg * (model_metrics["total_requests"] - 1) + metrics.quality.relevance_score) / model_metrics["total_requests"]
                model_metrics["avg_quality"] = new_avg
            
            # Store in storage backend if configured
            if self.storage:
                await self.storage.store_request_metrics(metrics)
            
            # Check for drift
            await self._check_drift(metrics)
    
    async def _store_session_metrics(self, metrics: SessionMetrics):
        """Store session metrics."""
        async with self._lock:
            self._session_metrics[metrics.session_id] = metrics
            
            if self.storage:
                await self.storage.store_session_metrics(metrics)
    
    async def _store_benchmark_metrics(self, metrics: BenchmarkMetrics):
        """Store benchmark metrics."""
        async with self._lock:
            self._benchmark_metrics[metrics.benchmark_id] = metrics
            
            if self.storage:
                await self.storage.store_benchmark_metrics(metrics)
    
    async def _check_drift(self, metrics: RequestMetrics):
        """Check for model drift based on new metrics."""
        model_id = metrics.model_info.model_id
        
        # Initialize baseline if not exists
        if model_id not in self._baseline_metrics:
            self._baseline_metrics[model_id] = {
                "cost": metrics.cost.total_cost,
                "latency": metrics.latency.time_to_complete_ms or 0,
                "quality": metrics.quality.relevance_score or 0.8,
                "created_at": metrics.created_at,
            }
            return
        
        baseline = self._baseline_metrics[model_id]
        
        # Calculate drift percentages
        cost_drift = 0.0
        latency_drift = 0.0
        quality_drift = 0.0
        
        if baseline["cost"] > 0:
            cost_drift = abs(metrics.cost.total_cost - baseline["cost"]) / baseline["cost"]
        if baseline["latency"] > 0:
            latency_drift = abs((metrics.latency.time_to_complete_ms or 0) - baseline["latency"]) / baseline["latency"]
        if baseline["quality"] > 0 and metrics.quality.relevance_score:
            quality_drift = abs(metrics.quality.relevance_score - baseline["quality"]) / baseline["quality"]
        
        # Check if drift exceeds threshold
        drift_threshold = self.config.drift_threshold if hasattr(self.config, 'drift_threshold') else 0.1
        
        alerts = []
        if cost_drift > drift_threshold:
            alerts.append({
                "type": "cost_drift",
                "value": cost_drift,
                "threshold": drift_threshold,
                "baseline": baseline["cost"],
                "current": metrics.cost.total_cost,
            })
        
        if latency_drift > drift_threshold:
            alerts.append({
                "type": "latency_drift",
                "value": latency_drift,
                "threshold": drift_threshold,
                "baseline": baseline["latency"],
                "current": metrics.latency.time_to_complete_ms or 0,
            })
        
        if quality_drift > drift_threshold:
            alerts.append({
                "type": "quality_drift",
                "value": quality_drift,
                "threshold": drift_threshold,
                "baseline": baseline["quality"],
                "current": metrics.quality.relevance_score or 0,
            })
        
        if alerts:
            self._drift_alerts.setdefault(model_id, []).extend(alerts)
            logger.warning(f"Model drift detected for {model_id}: {alerts}")
            self._trigger_callbacks("on_drift_detected", model_id, alerts)
    
    def _hash_request(self, request: Union[CompletionRequest, ChatRequest, EmbeddingRequest]) -> str:
        """Hash a request for deduplication."""
        import hashlib
        
        if isinstance(request, CompletionRequest):
            content = request.prompt
        elif isinstance(request, ChatRequest):
            content = str([(m.role.value, m.content) for m in request.messages])
        elif isinstance(request, EmbeddingRequest):
            content = str(request.input)
        else:
            content = str(request)
        
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def get_request_metrics(self, request_id: str) -> Optional[RequestMetrics]:
        """Get metrics for a specific request."""
        return self._request_metrics.get(request_id)
    
    def get_session_metrics(self, session_id: str) -> Optional[SessionMetrics]:
        """Get metrics for a specific session."""
        return self._session_metrics.get(session_id)
    
    def get_benchmark_metrics(self, benchmark_id: str) -> Optional[BenchmarkMetrics]:
        """Get metrics for a specific benchmark."""
        return self._benchmark_metrics.get(benchmark_id)
    
    def get_model_metrics(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get aggregated metrics for a model."""
        return self._model_metrics.get(model_id)
    
    def get_drift_alerts(self, model_id: str) -> List[Dict[str, Any]]:
        """Get drift alerts for a model."""
        return self._drift_alerts.get(model_id, [])
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all collected metrics."""
        return {
            "total_requests": len(self._request_metrics),
            "total_sessions": len(self._session_metrics),
            "total_benchmarks": len(self._benchmark_metrics),
            "models": self._model_metrics,
            "drift_alerts": self._drift_alerts,
        }
    
    async def close(self):
        """Clean up resources."""
        if self.storage:
            await self.storage.close()
