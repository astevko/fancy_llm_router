"""Refactor generic prompts into deployment-specific variants."""

from typing import Optional

from fancy_llm_router.schemas.models import ModelInfo
from fancy_llm_router.schemas.prompts import JudgeResult


class PromptOptimizer:
    """Specialize a root prompt for a deployment after a failed judge."""

    def refactor(
        self,
        generic_prompt: str,
        deployment_id: str,
        model_info: Optional[ModelInfo],
        response: str,
        judge: JudgeResult,
        expected_answer: Optional[str] = None,
        revision: int = 1,
    ) -> str:
        source = model_info.source if model_info else "unknown"
        logical = model_info.logical_model if model_info else deployment_id
        quantization = None
        if model_info and model_info.capabilities:
            quantization = model_info.capabilities.quantization

        hints = [
            f"Answer concisely and directly.",
            f"Target deployment: {deployment_id} ({logical} via {source}).",
        ]
        if expected_answer:
            hints.append(f"The correct answer must include: {expected_answer}.")
        if quantization:
            hints.append(f"Model quantization: {quantization}.")
        if judge.rationale:
            hints.append(f"Previous attempt issue: {judge.rationale}")

        specialization = (
            f"[Specialized for {deployment_id}, revision {revision}]\n"
            + "\n".join(f"- {h}" for h in hints)
            + f"\n\nTask:\n{generic_prompt.strip()}"
        )
        return specialization
