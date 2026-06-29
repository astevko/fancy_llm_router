"""API routes for the LLM Router."""

import logging
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from fancy_llm_router.schemas.requests import (
    CompletionRequest,
    ChatRequest,
    EmbeddingRequest,
)
from fancy_llm_router.schemas.routing import RoutingDecision, RoutingStrategy
from fancy_llm_router.schemas.metrics import RequestMetrics, SessionMetrics
from fancy_llm_router.schemas.sessions import SessionConfig, SessionState
from fancy_llm_router.core.router import LLMRouter
from fancy_llm_router.metrics.collector import MetricsCollector
from fancy_llm_router.tools.base import ToolRegistry

logger = logging.getLogger(__name__)

router = APIRouter()


def get_router(request: Request) -> LLMRouter:
    """Get the router instance from request state."""
    if not hasattr(request.app.state, 'router'):
        raise HTTPException(status_code=500, detail="Router not initialized")
    return request.app.state.router


def get_metrics(request: Request) -> MetricsCollector:
    """Get the metrics collector instance from request state."""
    if not hasattr(request.app.state, 'metrics'):
        raise HTTPException(status_code=500, detail="Metrics collector not initialized")
    return request.app.state.metrics


def get_tools(request: Request) -> ToolRegistry:
    """Get the tool registry instance from request state."""
    if not hasattr(request.app.state, 'tools'):
        raise HTTPException(status_code=500, detail="Tool registry not initialized")
    return request.app.state.tools


# Router endpoints
@router.get("/models", summary="List available models")
async def list_models(
    router: LLMRouter = Depends(get_router)
) -> List[str]:
    """List all available models."""
    return router.list_models()


@router.get("/models/{model_id}", summary="Get model information")
async def get_model(
    model_id: str,
    router: LLMRouter = Depends(get_router)
) -> dict:
    """Get information about a specific model."""
    model_info = router.get_model_info(model_id)
    if model_info is None:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    return model_info.dict()


@router.post("/route", summary="Route a request")
async def route_request(
    request: Union[CompletionRequest, ChatRequest, EmbeddingRequest],
    strategy: Optional[RoutingStrategy] = None,
    router: LLMRouter = Depends(get_router)
) -> RoutingDecision:
    """Route a request to the best model."""
    try:
        decision = await router.route(request, strategy=strategy)
        return decision
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/complete", summary="Generate a completion")
async def generate_completion(
    request: CompletionRequest,
    router: LLMRouter = Depends(get_router),
    metrics: MetricsCollector = Depends(get_metrics)
) -> dict:
    """Generate a text completion using the best model."""
    try:
        # Execute with metrics tracking
        async with metrics.track_request(request, None):
            response = await router.execute(request, track_metrics=False)
        
        return {
            "response": response.dict(),
            "model": response.model,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/chat", summary="Generate a chat completion")
async def generate_chat(
    request: ChatRequest,
    router: LLMRouter = Depends(get_router),
    metrics: MetricsCollector = Depends(get_metrics)
) -> dict:
    """Generate a chat completion using the best model."""
    try:
        async with metrics.track_request(request, None):
            response = await router.execute(request, track_metrics=False)
        
        return {
            "response": response.dict(),
            "model": response.model,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/embed", summary="Generate embeddings")
async def generate_embedding(
    request: EmbeddingRequest,
    router: LLMRouter = Depends(get_router),
    metrics: MetricsCollector = Depends(get_metrics)
) -> dict:
    """Generate embeddings using the best model."""
    try:
        async with metrics.track_request(request, None):
            response = await router.execute(request, track_metrics=False)
        
        return {
            "response": response.dict(),
            "model": response.model,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Metrics endpoints
@router.get("/metrics/requests", summary="List request metrics")
async def list_request_metrics(
    limit: int = 100,
    offset: int = 0,
    metrics: MetricsCollector = Depends(get_metrics)
) -> List[dict]:
    """List recent request metrics."""
    all_metrics = list(metrics._request_metrics.values())
    return [m.dict() for m in all_metrics[offset:offset + limit]]


@router.get("/metrics/requests/{request_id}", summary="Get request metrics")
async def get_request_metrics(
    request_id: str,
    metrics: MetricsCollector = Depends(get_metrics)
) -> dict:
    """Get metrics for a specific request."""
    request_metrics = metrics.get_request_metrics(request_id)
    if request_metrics is None:
        raise HTTPException(status_code=404, detail=f"Request {request_id} not found")
    return request_metrics.dict()


@router.get("/metrics/summary", summary="Get metrics summary")
async def get_metrics_summary(
    metrics: MetricsCollector = Depends(get_metrics)
) -> dict:
    """Get a summary of all collected metrics."""
    return metrics.get_summary()


@router.get("/metrics/models", summary="Get model metrics")
async def get_model_metrics(
    metrics: MetricsCollector = Depends(get_metrics)
) -> dict:
    """Get aggregated metrics for all models."""
    return {
        model_id: metrics.get_model_metrics(model_id)
        for model_id in metrics._model_metrics
    }


@router.get("/metrics/drift", summary="Get drift alerts")
async def get_drift_alerts(
    metrics: MetricsCollector = Depends(get_metrics)
) -> dict:
    """Get model drift alerts."""
    return {
        model_id: alerts
        for model_id, alerts in metrics._drift_alerts.items()
    }


# Session endpoints
@router.post("/sessions", summary="Create a new session")
async def create_session(
    config: SessionConfig,
    router: LLMRouter = Depends(get_router),
    metrics: MetricsCollector = Depends(get_metrics)
) -> dict:
    """Create a new session."""
    from fancy_llm_router.core.session import SessionManager
    
    # Create session manager (in a real implementation, this would be persistent)
    session_manager = SessionManager(router=router, metrics_collector=metrics)
    
    session = await session_manager.create_session(config)
    
    return {
        "session_id": session.session_id,
        "status": session.state.status.value,
        "created_at": session.state.created_at.isoformat(),
    }


@router.get("/sessions/{session_id}", summary="Get session state")
async def get_session(
    session_id: str,
    router: LLMRouter = Depends(get_router),
    metrics: MetricsCollector = Depends(get_metrics)
) -> dict:
    """Get the state of a session."""
    from fancy_llm_router.core.session import SessionManager
    
    session_manager = SessionManager(router=router, metrics_collector=metrics)
    session = await session_manager.get_session(session_id)
    
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    return session.state.dict()


# Tool endpoints
@router.get("/tools", summary="List available tools")
async def list_tools(
    tools: ToolRegistry = Depends(get_tools)
) -> List[dict]:
    """List all available tools."""
    return [tool.definition.dict() for tool in tools.list_tools()]


@router.get("/tools/{tool_id}", summary="Get tool information")
async def get_tool(
    tool_id: str,
    tools: ToolRegistry = Depends(get_tools)
) -> dict:
    """Get information about a specific tool."""
    tool = tools.get(tool_id)
    if tool is None:
        raise HTTPException(status_code=404, detail=f"Tool {tool_id} not found")
    return tool.definition.dict()


@router.post("/tools/{tool_id}/call", summary="Call a tool")
async def call_tool(
    tool_id: str,
    parameters: dict,
    tools: ToolRegistry = Depends(get_tools)
) -> dict:
    """Call a tool with parameters."""
    tool = tools.get(tool_id)
    if tool is None:
        raise HTTPException(status_code=404, detail=f"Tool {tool_id} not found")
    
    try:
        result = await tool.call(parameters)
        return result.dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/tools/health", summary="Check tool health")
async def check_tool_health(
    tools: ToolRegistry = Depends(get_tools)
) -> dict:
    """Check health of all tools."""
    health = await tools.check_health()
    return {"health": health}


# System endpoints
@router.get("/system/status", summary="Get system status")
async def get_system_status(
    router: LLMRouter = Depends(get_router),
    metrics: MetricsCollector = Depends(get_metrics),
    tools: ToolRegistry = Depends(get_tools)
) -> dict:
    """Get overall system status."""
    return {
        "router": {
            "models": len(router.list_models()),
        },
        "metrics": metrics.get_summary(),
        "tools": {
            "count": len(tools.list_tools()),
        },
    }
