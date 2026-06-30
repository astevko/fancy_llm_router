"""Tests for prompt specialization."""

from fancy_llm_router.core.evaluator import LLEvaluator
from fancy_llm_router.core.optimizer import PromptOptimizer
from fancy_llm_router.schemas.prompts import JudgeResult


def test_optimizer_does_not_embed_failure_response():
    noisy = "- 'What is the capital of France?'\n" * 20
    judge = JudgeResult(
        pass_=False,
        accuracy_score=0.0,
        rationale=f"Response did not contain any of: paris. Got: {noisy}",
    )
    result = PromptOptimizer().refactor(
        generic_prompt="What is the capital of France?",
        deployment_id="gpt-oss-120b@nebius",
        model_info=None,
        response=noisy,
        judge=judge,
        expected_answer="Paris",
        revision=1,
    )
    assert noisy not in result
    assert "Got:" not in result
    assert "Question: What is the capital of France?" in result


def test_evaluator_failure_rationale_omits_response_body():
    noisy = "x" * 500
    judge = LLEvaluator().evaluate(
        "What is the capital of France?",
        noisy,
        "Paris",
    )
    assert not judge.pass_
    assert "Got:" not in judge.rationale
    assert noisy not in judge.rationale
