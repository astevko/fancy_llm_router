"""Evaluate LLM responses against expected answers."""

import re
from typing import Optional

from fancy_llm_router.schemas.prompts import JudgeResult


class LLEvaluator:
    """Score responses for benchmark and drift analytics."""

    def evaluate(
        self,
        prompt: str,
        response: str,
        expected_answer: Optional[str] = None,
    ) -> JudgeResult:
        if not expected_answer or not expected_answer.strip():
            return JudgeResult(
                pass_=True,
                accuracy_score=1.0,
                relevance_score=1.0,
                rationale="No expected answer configured; marked as pass.",
            )

        response_norm = self._normalize(response)
        alternatives = [
            self._normalize(part)
            for part in re.split(r"[|;/]", expected_answer)
            if part.strip()
        ]

        for alt in alternatives:
            if alt and alt in response_norm:
                return JudgeResult(
                    pass_=True,
                    accuracy_score=1.0,
                    relevance_score=1.0,
                    rationale=f"Response contains expected value '{alt}'.",
                )

        return JudgeResult(
            pass_=False,
            accuracy_score=0.0,
            relevance_score=0.2,
            rationale=(
                f"Response did not contain any of: {', '.join(alternatives)}. "
                f"Got: {response[:200]}"
            ),
        )

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())
