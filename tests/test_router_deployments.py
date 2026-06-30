"""Tests for multi-source deployment routing."""

import pytest

from fancy_llm_router.core.router import RoutingStrategy
from fancy_llm_router.schemas.requests import CompletionRequest


class TestDeploymentRouting:
    @pytest.mark.asyncio
    async def test_logical_model_narrows_candidates(self, mock_router):
        request = CompletionRequest(prompt="hi", model="Qwen/Qwen3-32B")
        decision = await mock_router.route(
            request, strategy=RoutingStrategy.COST_OPTIMIZED
        )

        assert set(decision.candidates) == {"qwen@nebius", "qwen@ollama"}
        assert decision.selected_deployment == "qwen@ollama"
        assert decision.selected_model == "Qwen/Qwen3-32B"

    @pytest.mark.asyncio
    async def test_latency_strategy_picks_faster_source(self, mock_router):
        request = CompletionRequest(prompt="hi", model="Qwen/Qwen3-32B")
        decision = await mock_router.route(
            request, strategy=RoutingStrategy.LATENCY_OPTIMIZED
        )
        assert decision.selected_deployment == "qwen@nebius"

    @pytest.mark.asyncio
    async def test_pin_deployment_by_id(self, mock_router):
        request = CompletionRequest(prompt="hi", model="qwen@nebius")
        decision = await mock_router.route(request)
        assert decision.selected_deployment == "qwen@nebius"
        assert len(decision.candidates) == 1

    @pytest.mark.asyncio
    async def test_execute_end_to_end_with_mock(self, mock_router):
        response = await mock_router.execute(
            CompletionRequest(prompt="burn tokens", max_tokens=32),
            strategy=RoutingStrategy.COST_OPTIMIZED,
        )
        assert response.choices
        assert response.usage["total_tokens"] > 0

    @pytest.mark.asyncio
    async def test_lookup_by_logical_model(self, mock_router):
        info = mock_router.get_model_info("Qwen/Qwen3-32B")
        assert info is not None
        assert info.logical_model == "Qwen/Qwen3-32B"
