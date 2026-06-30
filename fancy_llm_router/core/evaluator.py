"""Evaluate LLM responses against expected answers."""

import re
from collections import Counter
from typing import List, Optional

from fancy_llm_router.schemas.prompts import JudgeResult

# Max completion tokens before flagging verbosity by catalog category.
_OUTPUT_TOKEN_THRESHOLDS = {
    "small": 40,
    "medium": 150,
    "large": 400,
    "xl": 800,
}


class LLEvaluator:
    """Score responses for benchmark and drift analytics."""

    def evaluate(
        self,
        prompt: str,
        response: str,
        expected_answer: Optional[str] = None,
        *,
        finish_reason: Optional[str] = None,
        completion_tokens: int = 0,
        category: Optional[str] = None,
    ) -> JudgeResult:
        warnings = self._rambler_warnings(
            response=response,
            finish_reason=finish_reason,
            completion_tokens=completion_tokens,
            category=category,
            expected_answer=expected_answer,
        )

        if not expected_answer or not expected_answer.strip():
            return JudgeResult(
                pass_=True,
                accuracy_score=1.0,
                relevance_score=1.0,
                rationale="No expected answer configured; marked as pass.",
                warnings=warnings,
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
                    warnings=warnings,
                )

        return JudgeResult(
            pass_=False,
            accuracy_score=0.0,
            relevance_score=0.2,
            rationale=(
                f"Response did not contain any of: {', '.join(alternatives)}."
            ),
            warnings=warnings,
        )

    def _rambler_warnings(
        self,
        response: str,
        finish_reason: Optional[str],
        completion_tokens: int,
        category: Optional[str],
        expected_answer: Optional[str],
    ) -> List[str]:
        """Heuristics for noisy / rambling outputs (independent of pass/fail)."""
        warnings: List[str] = []
        cat = (category or "small").lower()

        if finish_reason == "length":
            warnings.append("hit_max_tokens")

        threshold = _OUTPUT_TOKEN_THRESHOLDS.get(cat, 150)
        tokens = completion_tokens or max(1, len(response.split()))
        if tokens > threshold:
            warnings.append(f"high_output_tokens:{tokens}")

        lines = [line.strip() for line in response.splitlines() if line.strip()]
        if lines:
            _line, count = Counter(lines).most_common(1)[0]
            if count >= 3:
                warnings.append("repetitive_lines")

        bullet_lines = sum(
            1 for line in lines if line.startswith(("- ", "* ", "• "))
        )
        if bullet_lines >= 3:
            warnings.append("bullet_list_spiral")

        if cat == "small" and response.count("?") >= 3:
            warnings.append("question_echo")

        if expected_answer and len(response.strip()) > max(
            200, len(expected_answer.strip()) * 25
        ):
            warnings.append("verbose_despite_pass")

        if "```" in response:
            warnings.append("contains_code_fence")

        return warnings

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())
