"""LLM Router - Dynamic model selection based on criteria."""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Union
from datetime import datetime

from fancy_llm_router.schemas.routing import (
    RoutingStrategy,
    RoutingCriteria,
    RoutingDecision,
    FallbackConfig,
)
from fancy_llm_router.schemas.models import ModelInfo, ModelProvider
from fancy_llm_router.schemas.requests import (
    CompletionRequest,
    ChatRequest,
    EmbeddingRequest,
)
from fancy_llm_router.schemas.metrics import RequestMetrics
from fancy_llm_router.models.base import BaseModelProvider, ModelProviderFactory

logger = logging.getLogger(__name__)


@dataclass
class ModelCandidate:
    """A candidate model for routing."""
    model_info: ModelInfo
    provider: BaseModelProvider
    score: float = 0.0
    reasoning: str = ""
    
    # Performance estimates
    estimated_cost: float = 0.0
    estimated_latency: float = 0.0
    estimated_quality: float = 0.0
    
    # Availability
    is_available: bool = True
    last_checked: Optional[datetime] = None


class RouterError(Exception):
    """Error in routing."""
    pass


class NoAvailableModelsError(RouterError):
    """No models available that meet the criteria."""
    pass


class LLMRouter:
    """
    Dynamic LLM Router that selects the best model based on criteria.
    
    Features:
    - Multiple routing strategies (cost, latency, quality, balanced)
    - Fallback support
    - Model drift detection
    - Performance-based routing
    - Custom routing logic
    """
    
    def __init__(
        self,
        default_strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        default_criteria: Optional[RoutingCriteria] = None,
        fallback_config: Optional[FallbackConfig] = None,
        models: Optional[List[ModelInfo]] = None,
        **kwargs
    ):
        self.default_strategy = default_strategy
        self.default_criteria = default_criteria or RoutingCriteria()
        self.fallback_config = fallback_config
        self.extra_config = kwargs
        
        # Model registry
        self._models: Dict[str, ModelCandidate] = {}
        self._providers: Dict[str, BaseModelProvider] = {}
        
        # Performance tracking
        self._model_performance: Dict[str, Dict[str, float]] = {}
        self._model_availability: Dict[str, bool] = {}
        
        # Request tracking
        self._request_counter = 0
        self._last_routing_decision: Optional[RoutingDecision] = None
        
        # Initialize with provided models
        if models:
            for model_info in models:
                self.register_model(model_info)
        
        # Load default models
        self._load_default_models()
    
    def _load_default_models(self):
        """Load default model configurations."""
        # This would be populated from config files in a real implementation
        pass
    
    def register_model(
        self,
        model_info: ModelInfo,
        provider: Optional[BaseModelProvider] = None
    ):
        """Register a model for routing."""
        model_id = model_info.full_id
        
        if model_id in self._models:
            logger.warning(f"Model {model_id} already registered, updating")
        
        # Create or use provided provider
        if provider is None:
            provider = ModelProviderFactory.get(
                model_info.provider,
                model_info.model_id
            )
            if provider is None:
                # Create a new provider instance
                provider = ModelProviderFactory.create(
                    model_info.provider,
                    model_info.model_id
                )
        
        self._models[model_id] = ModelCandidate(
            model_info=model_info,
            provider=provider,
            is_available=True,
            last_checked=datetime.utcnow(),
        )
        self._providers[model_id] = provider
        
        logger.info(f"Registered model: {model_id}")
    
    def unregister_model(self, model_id: str):
        """Unregister a model."""
        if model_id in self._models:
            del self._models[model_id]
        if model_id in self._providers:
            del self._providers[model_id]
        logger.info(f"Unregistered model: {model_id}")
    
    async def check_model_availability(self, model_id: str) -> bool:
        """Check if a model is available."""
        if model_id not in self._models:
            return False
        
        candidate = self._models[model_id]
        try:
            is_available = await candidate.provider.health_check()
            candidate.is_available = is_available
            candidate.last_checked = datetime.utcnow()
            self._model_availability[model_id] = is_available
            return is_available
        except Exception as e:
            logger.error(f"Error checking availability for {model_id}: {e}")
            candidate.is_available = False
            candidate.last_checked = datetime.utcnow()
            self._model_availability[model_id] = False
            return False
    
    async def check_all_models(self):
        """Check availability of all registered models."""
        tasks = []
        for model_id in self._models:
            tasks.append(self.check_model_availability(model_id))
        
        await asyncio.gather(*tasks)
    
    def get_available_models(
        self,
        criteria: Optional[RoutingCriteria] = None
    ) -> List[ModelCandidate]:
        """Get list of available models that meet the criteria."""
        criteria = criteria or self.default_criteria
        available = []
        
        for model_id, candidate in self._models.items():
            # Check basic availability
            if not candidate.is_available:
                continue
            
            # Check provider constraints
            if criteria.allowed_providers:
                if candidate.model_info.provider.value not in criteria.allowed_providers:
                    continue
            
            if criteria.blocked_providers:
                if candidate.model_info.provider.value in criteria.blocked_providers:
                    continue
            
            # Check model constraints
            if criteria.allowed_models:
                if model_id not in criteria.allowed_models:
                    continue
            
            if criteria.blocked_models:
                if model_id in criteria.blocked_models:
                    continue
            # Check capability constraints
            if criteria.requires_chat and not candidate.model_info.capabilities.supports_chat:
                continue
            if criteria.requires_embeddings and not candidate.model_info.capabilities.supports_embeddings:
                continue
            if criteria.requires_streaming and not candidate.model_info.capabilities.supports_streaming:
                continue
            if criteria.requires_function_calls and not candidate.model_info.capabilities.supports_function_calls:
                continue
            if criteria.requires_embeddings and not candidate.model_info.capabilities.supports_embeddings:
                continue
            if criteria.requires_streaming and not candidate.model_info.capabilities.supports_streaming:
                continue
            if criteria.requires_function_calls and not candidate.model_info.capabilities.supports_function_calls:
                continue
            
            # Check context window
            if criteria.min_context_window:
                if candidate.model_info.capabilities.context_window < criteria.min_context_window:
                    continue
            
            available.append(candidate)
        
        return available
    
    def _calculate_model_score(
        self,
        candidate: ModelCandidate,
        criteria: RoutingCriteria,
        request: Union[CompletionRequest, ChatRequest, EmbeddingRequest],
    ) -> float:
        """Calculate a score for a model candidate based on criteria."""
        strategy = criteria.metadata.get("strategy", self.default_strategy)
        
        # Get performance metrics (would come from historical data)
        model_id = candidate.model_info.full_id
        performance = self._model_performance.get(model_id, {})
        
        # Estimate cost
        input_tokens = 100  # Would be estimated from request
        output_tokens = 50  # Would be estimated from request
        estimated_cost = (
            input_tokens * candidate.model_info.pricing.input_token_price +
            output_tokens * candidate.model_info.pricing.output_token_price
        )
        
        # Estimate latency (from historical data or model capabilities)
        estimated_latency = performance.get("avg_latency_ms", 1000.0)
        
        # Estimate quality (from historical data)
        estimated_quality = performance.get("avg_quality", 0.8)
        
        # Normalize scores (0-1, lower is better for cost/latency, higher is better for quality)
        max_cost = 10.0  # $10 max for normalization
        max_latency = 10000.0  # 10 seconds max for normalization
        
        cost_score = 1.0 - min(estimated_cost / max_cost, 1.0)
        latency_score = 1.0 - min(estimated_latency / max_latency, 1.0)
        quality_score = estimated_quality
        
        # Apply strategy weights
        if strategy == RoutingStrategy.COST_OPTIMIZED:
            score = cost_score
        elif strategy == RoutingStrategy.LATENCY_OPTIMIZED:
            score = latency_score
        elif strategy == RoutingStrategy.QUALITY_OPTIMIZED:
            score = quality_score
        elif strategy == RoutingStrategy.BALANCED:
            cost_weight = criteria.cost_weight
            latency_weight = criteria.latency_weight
            quality_weight = criteria.quality_weight
            score = (
                cost_score * cost_weight +
                latency_score * latency_weight +
                quality_score * quality_weight
            )
        else:
            # Default balanced
            score = (cost_score + latency_score + quality_score) / 3.0
        
        return score
    
    def _select_model(
        self,
        criteria: RoutingCriteria,
        request: Union[CompletionRequest, ChatRequest, EmbeddingRequest],
        candidates: List[ModelCandidate],
    ) -> ModelCandidate:
        """Select the best model from candidates."""
        if not candidates:
            raise NoAvailableModelsError("No models available that meet the criteria")
        
        # Score all candidates
        scored_candidates = []
        for candidate in candidates:
            score = self._calculate_model_score(candidate, criteria, request)
            candidate.score = score
            scored_candidates.append(candidate)
        
        # Sort by score (descending)
        scored_candidates.sort(key=lambda c: c.score, reverse=True)
        
        # Return the best candidate
        return scored_candidates[0]
    
    async def route(
        self,
        request: Union[CompletionRequest, ChatRequest, EmbeddingRequest],
        strategy: Optional[RoutingStrategy] = None,
        criteria: Optional[RoutingCriteria] = None,
        git_commit: Optional[str] = None,
        **kwargs
    ) -> RoutingDecision:
        """
        Route a request to the best model based on criteria.
        
        Args:
            request: The LLM request to route
            strategy: Routing strategy to use (defaults to router's default)
            criteria: Routing criteria (defaults to router's default)
            git_commit: Git commit hash for tracking
            **kwargs: Additional arguments
            
        Returns:
            RoutingDecision with the selected model and reasoning
        """
        start_time = time.time()
        strategy = strategy or self.default_strategy
        criteria = criteria or self.default_criteria
        
        # Generate request ID if not provided
        if not request.request_id:
            request.request_id = self._generate_request_id()
        
        # Generate prompt hash if not provided
        if not request.prompt_hash:
            request.prompt_hash = self._hash_request(request)
        
        # Get available models
        candidates = self.get_available_models(criteria)
        
        if not candidates:
            raise NoAvailableModelsError(
                f"No models available that meet criteria: {criteria}"
            )
        
        # Select the best model
        selected = self._select_model(criteria, request, candidates)
        
        # Create routing decision
        decision = RoutingDecision(
            request_id=request.request_id,
            session_id=request.session_id,
            selected_model=selected.model_info.model_id,
            selected_provider=selected.model_info.provider.value,
            strategy=strategy,
            criteria=criteria,
            reasoning=f"Selected {selected.model_info.full_id} with score {selected.score:.3f}",
            confidence=selected.score,
            candidates=[c.model_info.full_id for c in candidates],
            candidate_scores={c.model_info.full_id: c.score for c in candidates},
            decision_time_ms=(time.time() - start_time) * 1000,
            metadata={
                "git_commit": git_commit,
                **kwargs,
            },
        )
        
        self._last_routing_decision = decision
        logger.info(f"Routed request {request.request_id} to {decision.full_model_id}")
        
        return decision
    
    async def execute(
        self,
        request: Union[CompletionRequest, ChatRequest, EmbeddingRequest],
        strategy: Optional[RoutingStrategy] = None,
        criteria: Optional[RoutingCriteria] = None,
        git_commit: Optional[str] = None,
        track_metrics: bool = True,
        **kwargs
    ) -> Any:
        """
        Route and execute a request.
        
        Args:
            request: The LLM request to execute
            strategy: Routing strategy to use
            criteria: Routing criteria
            git_commit: Git commit hash for tracking
            track_metrics: Whether to track metrics
            **kwargs: Additional arguments
            
        Returns:
            The response from the selected model
        """
        # Route the request
        decision = await self.route(
            request,
            strategy=strategy,
            criteria=criteria,
            git_commit=git_commit,
            **kwargs
        )
        
        # Get the provider
        model_id = decision.full_model_id
        if model_id not in self._providers:
            raise RouterError(f"Model {model_id} not available")
        
        provider = self._providers[model_id]
        
        # Execute the request based on type
        start_time = time.time()
        first_token_time = None
        
        try:
            if isinstance(request, CompletionRequest):
                response = await provider.completion(request, git_commit)
            elif isinstance(request, ChatRequest):
                response = await provider.chat(request, git_commit)
            elif isinstance(request, EmbeddingRequest):
                response = await provider.embedding(request, git_commit)
            else:
                raise RouterError(f"Unsupported request type: {type(request)}")
            
            end_time = time.time()
            
            # Create metrics if tracking is enabled
            if track_metrics:
                metrics = await self._create_metrics(
                    request, response, decision, start_time, end_time, git_commit
                )
                # Store metrics (would be handled by MetricsCollector in real implementation)
                self._store_metrics(metrics)
            
            return response
            
        except Exception as e:
            # Handle fallback if configured
            if self.fallback_config and self.fallback_config.fallback_on_error:
                logger.warning(f"Request failed with {model_id}, trying fallback: {e}")
                return await self._handle_fallback(
                    request, decision, e, git_commit, track_metrics, **kwargs
                )
            raise
    
    async def _handle_fallback(
        self,
        request: Union[CompletionRequest, ChatRequest, EmbeddingRequest],
        original_decision: RoutingDecision,
        error: Exception,
        git_commit: Optional[str],
        track_metrics: bool,
        **kwargs
    ) -> Any:
        """Handle fallback routing when a request fails."""
        if not self.fallback_config:
            raise error
        
        # Try fallback models
        fallback_models = self.fallback_config.fallback_models
        if not fallback_models:
            raise error
        
        last_error = error
        for fallback_model in fallback_models:
            try:
                # Update request with fallback model
                if isinstance(request, CompletionRequest):
                    request.model = fallback_model
                elif isinstance(request, ChatRequest):
                    request.model = fallback_model
                elif isinstance(request, EmbeddingRequest):
                    request.model = fallback_model
                
                # Route with fallback criteria
                fallback_criteria = RoutingCriteria(
                    allowed_models=[fallback_model],
                    fallback_on_error=False,  # Don't recurse
                )
                
                decision = await self.route(
                    request,
                    strategy=RoutingStrategy.FALLBACK,
                    criteria=fallback_criteria,
                    git_commit=git_commit,
                    **kwargs
                )
                
                # Get provider
                model_id = decision.full_model_id
                if model_id not in self._providers:
                    continue
                
                provider = self._providers[model_id]
                
                # Execute
                start_time = time.time()
                if isinstance(request, CompletionRequest):
                    response = await provider.completion(request, git_commit)
                elif isinstance(request, ChatRequest):
                    response = await provider.chat(request, git_commit)
                elif isinstance(request, EmbeddingRequest):
                    response = await provider.embedding(request, git_commit)
                else:
                    continue
                
                end_time = time.time()
                
                # Update decision with fallback info
                decision.is_fallback = True
                decision.fallback_reason = str(error)
                decision.original_model = original_decision.selected_model
                
                # Create metrics
                if track_metrics:
                    metrics = await self._create_metrics(
                        request, response, decision, start_time, end_time, git_commit
                    )
                    self._store_metrics(metrics)
                
                return response
                
            except Exception as e:
                last_error = e
                logger.warning(f"Fallback to {fallback_model} failed: {e}")
        
        # All fallbacks failed
        raise RouterError(f"All fallback models failed: {last_error}")
    
    async def _create_metrics(
        self,
        request: Union[CompletionRequest, ChatRequest, EmbeddingRequest],
        response: Any,
        decision: RoutingDecision,
        start_time: float,
        end_time: float,
        git_commit: Optional[str],
    ) -> RequestMetrics:
        """Create metrics for a request."""
        from fancy_llm_router.schemas.metrics import RequestMetrics, ModelMetrics
        
        # Get token usage
        if hasattr(response, 'usage'):
            usage = response.usage
            token_usage = {
                "prompt_tokens": usage.get('prompt_tokens', 0),
                "completion_tokens": usage.get('completion_tokens', 0),
                "total_tokens": usage.get('total_tokens', 0),
            }
        else:
            token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        # Get model info
        model_info = decision.model_info
        
        # Create metrics
        metrics = RequestMetrics(
            request_id=request.request_id,
            session_id=request.session_id,
            prompt_hash=request.prompt_hash,
            created_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            request_type=type(request).__name__.replace('Request', '').lower(),
            model_info=ModelMetrics(
                model_id=model_info.model_id,
                model_provider=model_info.provider.value,
                model_version=model_info.version,
                model_parameters=model_info.parameters,
                context_window=model_info.capabilities.context_window,
            ),
            token_usage=token_usage,
            cost={},  # Would be calculated properly
            latency={},  # Would be calculated properly
            git_commit=git_commit,
            metadata={
                "routing_decision": decision.dict(),
                "is_fallback": decision.is_fallback,
            },
        )
        
        return metrics
    
    def _store_metrics(self, metrics: RequestMetrics):
        """Store metrics. In a real implementation, this would use MetricsCollector."""
        # For now, just log
        logger.debug(f"Stored metrics for request {metrics.request_id}")
    
    def _generate_request_id(self) -> str:
        """Generate a unique request ID."""
        self._request_counter += 1
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        return f"req-{timestamp}-{self._request_counter}"
    
    def _hash_request(self, request: Union[CompletionRequest, ChatRequest, EmbeddingRequest]) -> str:
        """Hash a request for deduplication."""
        if isinstance(request, CompletionRequest):
            content = request.prompt
        elif isinstance(request, ChatRequest):
            content = str([(m.role.value, m.content) for m in request.messages])
        elif isinstance(request, EmbeddingRequest):
            content = str(request.input)
        else:
            content = str(request)
        
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self._models.keys())
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get model information."""
        if model_id in self._models:
            return self._models[model_id].model_info
        return None
    
    async def close(self):
        """Clean up resources."""
        for provider in self._providers.values():
            await provider.close()
        self._providers.clear()
        self._models.clear()
