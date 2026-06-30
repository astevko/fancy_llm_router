"""Wrap benchmark prompts so models answer in one concise sentence."""

from typing import List, Optional

# System instructions by catalog category.
MEASURE_SYSTEM = {
    "small": (
        "You are a precise Q&A assistant. Answer in exactly one short sentence. "
        "Give only the direct answer to the question. "
        "Do not use lists, bullet points, numbering, code blocks, or repetition."
    ),
    "medium": (
        "You are a precise assistant. Answer in 2-4 concise sentences. "
        "Stay on topic. Do not use bullet lists unless the question explicitly asks for them."
    ),
    "large": (
        "You are a precise assistant. Answer the question directly and stay on topic. "
        "Use short paragraphs or a brief numbered list only when the question requires structure."
    ),
    "xl": (
        "You are a precise assistant. Follow the question structure but avoid filler, "
        "repetition, and unrelated tangents."
    ),
}

# Hard caps on completion tokens during measure (in addition to client budget).
MEASURE_TOKEN_CAP = {
    "small": 80,
    "medium": 200,
    "large": 512,
    "xl": 1024,
}

MEASURE_STOPS: List[str] = ["\n\n", "\n- ", "\n* ", "\n1.", "```"]


def normalize_category(category: Optional[str]) -> str:
    cat = (category or "small").lower()
    return cat if cat in MEASURE_SYSTEM else "medium"


def measure_system_prompt(category: Optional[str]) -> str:
    return MEASURE_SYSTEM[normalize_category(category)]


def measure_user_message(question: str) -> str:
    """User turn: the catalog question only (constraints live in the system prompt)."""
    return question.strip()


def measure_stop_sequences() -> List[str]:
    return list(MEASURE_STOPS)


def measure_max_tokens(category: Optional[str], requested: int) -> int:
    cap = MEASURE_TOKEN_CAP.get(normalize_category(category), requested)
    return min(requested, cap)


def format_wire_prompt(system: str, user: str) -> str:
    """Human-readable prompt as sent via chat API (for analytics storage)."""
    return f"[system]\n{system}\n\n[user]\n{user}"
