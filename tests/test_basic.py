"""Basic tests for the Fancy LLM Router."""

import pytest
import asyncio
from datetime import datetime

from fancy_llm_router.core.router import LLMRouter, RoutingStrategy
from fancy_llm_router.schemas.requests import CompletionRequest, ChatRequest
from fancy_llm_router.schemas.models import ModelInfo, ModelProvider, ModelCapabilities, ModelPricing
from fancy_llm_router.schemas.routing import RoutingCriteria
from fancy_llm_router.metrics.collector import MetricsCollector
from fancy_llm_router.tools.base import BaseTool, ToolRegistry
from fancy_llm_router.tools.mock import MockTool


@pytest.fixture
def router():
    """Create a router with test models."""
    router = LLMRouter()
    
    # Add test models
    models = [
        ModelInfo(
            provider=ModelProvider.OPENAI,
            model_id="gpt-4",
            name="GPT-4",
            capabilities=ModelCapabilities(
                max_tokens=8192,
                max_input_tokens=8192,
                context_window=8192,
                supports_chat=True,
                supports_completions=True,
            ),
            pricing=ModelPricing(
                input_token_price=0.03,
                output_token_price=0.06,
            ),
            parameters=1000000000,
        ),
        ModelInfo(
            provider=ModelProvider.OPENAI,
            model_id="gpt-3.5-turbo",
            name="GPT-3.5 Turbo",
            capabilities=ModelCapabilities(
                max_tokens=4096,
                max_input_tokens=4096,
                context_window=4096,
                supports_chat=True,
                supports_completions=False,
            ),
            pricing=ModelPricing(
                input_token_price=0.0015,
                output_token_price=0.002,
            ),
            parameters=175000000,
        ),
    ]
    
    for model in models:
        router.register_model(model)
    
    return router


@pytest.fixture
def metrics_collector():
    """Create a metrics collector."""
    return MetricsCollector()


@pytest.fixture
def tool_registry():
    """Create a tool registry with test tools."""
    registry = ToolRegistry()
    
    # Add test tools
    echo_tool = MockTool(
        tool_id="echo",
        name="Echo Tool",
        description="Echoes input back",
    )
    registry.register(echo_tool)
    
    return registry


class TestRouter:
    """Tests for the LLMRouter."""
    
    @pytest.mark.asyncio
    async def test_list_models(self, router):
        """Test listing models."""
        models = router.list_models()
        assert len(models) == 2
        assert "openai:gpt-4" in models
        assert "openai:gpt-3.5-turbo" in models
    
    @pytest.mark.asyncio
    async def test_get_model_info(self, router):
        """Test getting model info."""
        model_info = router.get_model_info("openai:gpt-4")
        assert model_info is not None
        assert model_info.name == "GPT-4"
        assert model_info.provider == ModelProvider.OPENAI
    
    @pytest.mark.asyncio
    async def test_route_completion(self, router):
        """Test routing a completion request."""
        request = CompletionRequest(
            prompt="Test prompt",
            max_tokens=100,
        )
        
        decision = await router.route(request, strategy=RoutingStrategy.COST_OPTIMIZED)
        
        assert decision is not None
        assert decision.request_id is not None
        assert decision.selected_model in ["gpt-4", "gpt-3.5-turbo"]
        assert decision.selected_provider == "openai"
        assert decision.strategy == RoutingStrategy.COST_OPTIMIZED
    
    @pytest.mark.asyncio
    async def test_route_chat(self, router):
        """Test routing a chat request."""
        from fancy_llm_router.schemas.requests import ChatMessage, MessageRole
        
        request = ChatRequest(
            messages=[
                ChatMessage(role=MessageRole.USER, content="Hello")
            ],
            max_tokens=100,
        )
        
        decision = await router.route(request, strategy=RoutingStrategy.LATENCY_OPTIMIZED)
        
        assert decision is not None
        assert decision.selected_model in ["gpt-4", "gpt-3.5-turbo"]
    
    @pytest.mark.asyncio
    async def test_routing_criteria(self, router):
        """Test routing with criteria."""
        request = CompletionRequest(
            prompt="Test prompt",
            max_tokens=100,
        )
        
        criteria = RoutingCriteria(
            allowed_models=["gpt-3.5-turbo"],
            max_cost_usd=0.1,
        )
        
        decision = await router.route(request, criteria=criteria)
        
        assert decision is not None
        assert decision.selected_model == "gpt-3.5-turbo"


class TestMetricsCollector:
    """Tests for the MetricsCollector."""
    
    @pytest.mark.asyncio
    async def test_track_request(self, metrics_collector):
        """Test tracking a request."""
        from fancy_llm_router.models.generic import GenericProvider
        from fancy_llm_router.schemas.models import ModelProvider
        
        # Create a mock provider
        provider = GenericProvider(
            provider=ModelProvider.OPENAI,
            model_id="test-model",
        )
        
        request = CompletionRequest(
            prompt="Test prompt",
            max_tokens=100,
        )
        
        async with metrics_collector.track_request(request, provider) as metrics:
            # Simulate some work
            await asyncio.sleep(0.01)
            
            # Update metrics
            metrics.token_usage.prompt_tokens = 10
            metrics.token_usage.completion_tokens = 20
            metrics.token_usage.total_tokens = 30
        
        # Check that metrics were stored
        stored = metrics_collector.get_request_metrics(metrics.request_id)
        assert stored is not None
        assert stored.token_usage.total_tokens == 30
    
    @pytest.mark.asyncio
    async def test_track_session(self, metrics_collector):
        """Test tracking a session."""
        async with metrics_collector.track_session("test-session") as session_metrics:
            # Simulate some work
            await asyncio.sleep(0.01)
        
        # Check that session metrics were stored
        stored = metrics_collector.get_session_metrics("test-session")
        assert stored is not None
        assert stored.session_id == "test-session"
    
    def test_get_summary(self, metrics_collector):
        """Test getting metrics summary."""
        summary = metrics_collector.get_summary()
        assert "total_requests" in summary
        assert "total_sessions" in summary
        assert "models" in summary


class TestTools:
    """Tests for tools."""
    
    @pytest.mark.asyncio
    async def test_tool_execution(self, tool_registry):
        """Test tool execution."""
        tool = tool_registry.get("echo")
        assert tool is not None
        
        result = await tool.call({"input": "Hello"})
        
        assert result is not None
        assert result.success
        assert result.output is not None
    
    @pytest.mark.asyncio
    async def test_tool_registry(self, tool_registry):
        """Test tool registry."""
        tools = tool_registry.list_tools()
        assert len(tools) == 1
        assert tools[0].tool_id == "echo"
    
    @pytest.mark.asyncio
    async def test_tool_health_check(self, tool_registry):
        """Test tool health check."""
        health = await tool_registry.check_health()
        assert "echo" in health
        assert health["echo"] is True


class TestSchemas:
    """Tests for Pydantic schemas."""
    
    def test_completion_request(self):
        """Test CompletionRequest schema."""
        request = CompletionRequest(
            model="gpt-4",
            prompt="Test prompt",
            max_tokens=100,
            temperature=0.7,
        )
        
        assert request.model == "gpt-4"
        assert request.prompt == "Test prompt"
        assert request.max_tokens == 100
        assert request.temperature == 0.7
    
    def test_chat_request(self):
        """Test ChatRequest schema."""
        from fancy_llm_router.schemas.requests import ChatMessage, MessageRole
        
        request = ChatRequest(
            model="gpt-4",
            messages=[
                ChatMessage(role=MessageRole.USER, content="Hello"),
                ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!"),
            ],
            max_tokens=100,
        )
        
        assert request.model == "gpt-4"
        assert len(request.messages) == 2
    
    def test_model_info(self):
        """Test ModelInfo schema."""
        model_info = ModelInfo(
            provider=ModelProvider.OPENAI,
            model_id="gpt-4",
            name="GPT-4",
            capabilities=ModelCapabilities(
                max_tokens=8192,
                context_window=8192,
            ),
            pricing=ModelPricing(
                input_token_price=0.03,
                output_token_price=0.06,
            ),
        )
        
        assert model_info.full_id == "openai:gpt-4"
        assert model_info.capabilities.context_window == 8192
        assert model_info.pricing.input_token_price == 0.03
