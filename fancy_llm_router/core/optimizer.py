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
            "Answer in one short sentence. No lists, no repetition, no code.",
            f"Deployment: {deployment_id} ({logical} via {source}).",
        ]
        if expected_answer:
            hints.append(f"Include this answer: {expected_answer}.")
        if quantization:
            hints.append(f"Quantization: {quantization}.")
        if judge.rationale:
            short = judge.rationale.strip().replace("\n", " ")
            if ". Got:" in short:
                short = short.split(". Got:", 1)[0] + "."
            if len(short) > 120:
                short = short[:117] + "..."
            hints.append(f"Prior issue: {short}")

        task = generic_prompt.strip()
        return (
            f"Answer the following question concisely.\n"
            + "\n".join(f"- {h}" for h in hints)
            + f"\n\nQuestion: {task}"
        )
