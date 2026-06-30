"""Tests for prompt/deployment operator states."""

import tempfile
from pathlib import Path

import pytest

from fancy_llm_router.core.benchmark_service import BenchmarkService
from fancy_llm_router.core.prompt_registry import PromptRegistry
from fancy_llm_router.schemas.prompts import (
    BaselineResult,
    DeploymentPairState,
    JudgeResult,
)
from tests.conftest import MOCK_CONFIG


@pytest.fixture
def benchmark_setup():
    from fancy_llm_router.core.config_loader import build_router_from_config

    router = build_router_from_config(MOCK_CONFIG)
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.db")
        registry = PromptRegistry(db_path=db_path)
        registry.initialize()
        router.set_prompt_registry(registry)
        service = BenchmarkService(router=router, registry=registry)
        yield service, registry


@pytest.fixture
def registry():
    with tempfile.TemporaryDirectory() as tmp:
        reg = PromptRegistry(db_path=str(Path(tmp) / "states.db"))
        reg.initialize()
        reg.ensure_root("small-01", "What is the capital of France?", expected_answer="Paris")
        yield reg


def test_set_and_get_pair_state(registry):
    registry.set_pair_state("small-01", "mock@mock", DeploymentPairState.BLOCKED)
    state = registry.get_pair_state("small-01", "mock@mock")
    assert state.state == DeploymentPairState.BLOCKED

    registry.set_pair_state("small-01", "mock@mock", DeploymentPairState.UNSET)
    assert registry.get_pair_state("small-01", "mock@mock").state == DeploymentPairState.UNSET


def test_preferred_clears_other_preferred(registry):
    registry.set_pair_state("small-01", "a@mock", DeploymentPairState.PREFERRED)
    registry.set_pair_state("small-01", "b@mock", DeploymentPairState.PREFERRED)
    assert registry.get_pair_state("small-01", "a@mock").state == DeploymentPairState.UNSET
    assert registry.get_pair_state("small-01", "b@mock").state == DeploymentPairState.PREFERRED


def test_resolve_prompt_respects_blocked(registry):
    variant = registry.save_variant(
        root_id="small-01",
        deployment_id="mock@mock",
        prompt_text="Answer: Paris only.",
    )
    registry.mark_variant_passed(variant.variant_id)

    resolved = registry.resolve_prompt("small-01", "mock@mock", "generic")
    assert resolved.prompt_text == "Answer: Paris only."

    registry.set_pair_state("small-01", "mock@mock", DeploymentPairState.BLOCKED)
    blocked = registry.resolve_prompt("small-01", "mock@mock", "generic")
    assert blocked.prompt_text == "What is the capital of France?"
    assert blocked.variant_id is None


@pytest.mark.asyncio
async def test_measure_all_skips_blocked(benchmark_setup):
    service, registry = benchmark_setup
    registry.ensure_root("small-01", "What is the capital of France?", expected_answer="Paris")
    registry.set_pair_state("small-01", "mock-large@mock", DeploymentPairState.BLOCKED)
    result = await service.measure_all_deployments(
        root_id="small-01",
        prompt="What is the capital of France?",
        expected_answer="Paris",
        deployments=["mock-fast@mock", "mock-large@mock"],
        optimize=False,
        max_revisions=1,
    )
    assert len(result["results"]) == 1
    assert result["skipped_blocked"] == ["mock-large@mock"]


@pytest.mark.asyncio
async def test_improve_promotes_to_preferred_on_pass(benchmark_setup):
    service, registry = benchmark_setup
    dep = "mock-fast@mock"
    registry.ensure_root("small-01", "What is the capital of France?", expected_answer="Paris")
    registry.store_result(
        BaselineResult(
            result_id="fail-1",
            run_id="run-1",
            root_id="small-01",
            deployment_id=dep,
            generic_prompt="What is the capital of France?",
            prompt_used="What is the capital of France?",
            response_text="Lyon",
            response_hash="x",
            judge=JudgeResult(pass_=False, accuracy_score=0.0, rationale="wrong"),
        )
    )

    envelope = await service.improve_deployment("small-01", dep)
    state = registry.get_pair_state("small-01", dep)
    assert envelope.judge.pass_
    assert state.state == DeploymentPairState.PREFERRED
