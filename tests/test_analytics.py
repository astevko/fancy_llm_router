"""Tests for baseline analytics aggregation."""

import tempfile
from pathlib import Path

import pytest

from fancy_llm_router.core.analytics_service import AnalyticsService, build_run_summary
from fancy_llm_router.core.prompt_registry import PromptRegistry
from fancy_llm_router.schemas.prompts import BaselineResult, BaselineRun, JudgeResult


@pytest.fixture
def registry_with_data():
    with tempfile.TemporaryDirectory() as tmp:
        reg = PromptRegistry(db_path=str(Path(tmp) / "analytics.db"))
        reg.initialize()
        reg.ensure_root(
            "small-01",
            "What is the capital of France?",
            expected_answer="Paris",
            category="small",
        )
        run = reg.create_run(run_type="smoke")
        results = [
            BaselineResult(
                result_id="r1",
                run_id=run.run_id,
                root_id="small-01",
                deployment_id="mock-fast@mock",
                generic_prompt="What is the capital of France?",
                prompt_used="Answer with one word: capital of France?",
                response_text="Paris",
                response_hash="abc",
                prompt_tokens=10,
                completion_tokens=2,
                total_tokens=12,
                total_cost=0.00001,
                latency_ms=45.0,
                judge=JudgeResult(pass_=True, accuracy_score=1.0, rationale="match"),
            ),
            BaselineResult(
                result_id="r2",
                run_id=run.run_id,
                root_id="small-01",
                deployment_id="mock-large@mock",
                generic_prompt="What is the capital of France?",
                prompt_used="What is the capital of France?",
                response_text="Lyon",
                response_hash="def",
                prompt_tokens=12,
                completion_tokens=3,
                total_tokens=15,
                total_cost=0.00002,
                latency_ms=120.0,
                judge=JudgeResult(pass_=False, accuracy_score=0.0, rationale="wrong"),
            ),
        ]
        for r in results:
            reg.store_result(r)
        yield reg, run, results


def test_build_run_summary_groups_by_root(registry_with_data):
    reg, run, results = registry_with_data
    summary = build_run_summary(run, results, reg)
    assert summary.root_count == 1
    assert summary.result_count == 2
    assert summary.pass_count == 1
    assert summary.fail_count == 1
    assert summary.cost.total == pytest.approx(0.00003)
    root = summary.roots[0]
    assert root.root_id == "small-01"
    assert root.deployment_count == 2
    assert root.cost.max == pytest.approx(0.00002)
    assert root.latency_ms.max == pytest.approx(120.0)
    assert len(root.by_deployment) == 2
    tuned = next(d for d in root.by_deployment if d.deployment_id == "mock-fast@mock")
    assert "one word" in tuned.prompt_used
    assert tuned.response_text == "Paris"


def test_analytics_service_list_and_summary(registry_with_data):
    reg, run, _results = registry_with_data
    service = AnalyticsService(reg)
    runs = service.list_runs()
    assert len(runs) == 1
    assert runs[0].run_id == run.run_id
    assert runs[0].root_count == 1
    assert runs[0].result_count == 2
    summary = service.get_run_summary(run.run_id)
    assert summary is not None
    assert summary.roots[0].expected_answer == "Paris"


def test_analytics_service_missing_run(registry_with_data):
    reg, _, _ = registry_with_data
    service = AnalyticsService(reg)
    assert service.get_run_summary("missing") is None


def test_list_runs_from_orphan_results():
    """Runs appear in analytics when only baseline_results exist (burner client id)."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        reg = PromptRegistry(db_path=str(Path(tmp) / "orphan.db"))
        reg.initialize()
        run_id = "client-run-no-row"
        reg.store_result(
            BaselineResult(
                result_id="orphan-1",
                run_id=run_id,
                root_id="small-01",
                deployment_id="mock-fast@mock",
                generic_prompt="What is the capital of France?",
                prompt_used="What is the capital of France?",
                response_text="Paris",
                response_hash="abc",
                judge=JudgeResult(pass_=True, accuracy_score=1.0, rationale="ok"),
            )
        )
        assert reg.backfill_runs_from_results() == 1
        runs = reg.list_runs()
        assert len(runs) == 1
        assert runs[0].run_id == run_id
        assert runs[0].config_snapshot.get("result_count") == 1
        summary = AnalyticsService(reg).get_run_summary(run_id)
        assert summary is not None
        assert summary.result_count == 1
