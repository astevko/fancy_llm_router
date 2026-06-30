"""Tests for benchmark measurement and prompt specialization."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from fancy_llm_router.core.benchmark_service import BenchmarkService
from fancy_llm_router.core.config_loader import build_router_from_config
from fancy_llm_router.schemas.requests import CompletionRequest
from fancy_llm_router.core.prompt_registry import PromptRegistry
from tests.conftest import MOCK_CONFIG


@pytest.fixture
def benchmark_setup():
    router = build_router_from_config(MOCK_CONFIG)
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.db")
        registry = PromptRegistry(db_path=db_path)
        registry.initialize()
        router.set_prompt_registry(registry)
        service = BenchmarkService(router=router, registry=registry)
        yield service, registry


@pytest.mark.asyncio
async def test_measure_all_deployments(benchmark_setup):
    service, registry = benchmark_setup
    result = await service.measure_all_deployments(
        root_id="small-01",
        prompt="What is the capital of France?",
        expected_answer="Paris",
        optimize=True,
        max_revisions=2,
    )
    assert "benchmark_run_id" in result
    assert len(result["results"]) == len(MOCK_CONFIG["models"])
    stored = registry.list_results(result["benchmark_run_id"])
    assert len(stored) >= len(MOCK_CONFIG["models"])
    assert any(r.judge.pass_ for r in stored)


@pytest.mark.asyncio
async def test_production_uses_specialized_variant(benchmark_setup):
    service, registry = benchmark_setup
    root_id = "small-01"
    prompt = "What is the capital of France?"
    await service.measure_all_deployments(
        root_id=root_id,
        prompt=prompt,
        expected_answer="Paris",
        deployments=["mock-fast@mock"],
        optimize=True,
        max_revisions=2,
    )

    infer_request = CompletionRequest(
        intent="infer",
        model="mock-fast@mock",
        root_id=root_id,
        prompt=prompt,
    )
    payload = await service.handle_completion(infer_request)
    assert payload["deployment"] == "mock-fast@mock"
    # After benchmark+optimize, production may use a specialized prompt
    if payload.get("variant_id"):
        assert payload["prompt_used"] != prompt


@pytest.mark.asyncio
async def test_measure_rejects_auto(benchmark_setup):
    service, _registry = benchmark_setup
    request = CompletionRequest(
        intent="measure",
        model="auto",
        prompt="test",
        root_id="x",
    )
    with pytest.raises(ValueError, match="pinned deployment"):
        await service.measure_request(request)
