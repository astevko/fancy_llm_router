"""Tests for benchmark prompt envelope."""

from fancy_llm_router.core.prompt_envelope import (
    measure_max_tokens,
    measure_system_prompt,
    measure_user_message,
)


def test_small_measure_system_requires_one_sentence():
    system = measure_system_prompt("small")
    assert "one short sentence" in system
    assert "bullet" in system.lower()


def test_measure_user_message_is_unchanged_question():
    q = "What is the capital of France?"
    assert measure_user_message(q) == q


def test_measure_max_tokens_caps_small():
    assert measure_max_tokens("small", 256) == 80
    assert measure_max_tokens("large", 256) == 256
