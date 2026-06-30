"""Normalize token usage payloads from LLM provider APIs."""

from typing import Any, Dict, Optional


def normalize_usage(raw: Optional[Dict[str, Any]]) -> Dict[str, int]:
    """Extract integer token counts from a provider ``usage`` object.

    OpenAI-compatible APIs (including Nebius Token Factory) often return nested
    detail objects such as ``prompt_tokens_details`` and
    ``completion_tokens_details``. Our response schemas only keep the three
    standard integer counters.
    """
    if not raw:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _as_int(key: str, default: int = 0) -> int:
        value = raw.get(key, default)
        try:
            return int(value) if value is not None else default
        except (TypeError, ValueError):
            return default

    prompt_tokens = _as_int("prompt_tokens")
    completion_tokens = _as_int("completion_tokens")
    total_tokens = _as_int("total_tokens", prompt_tokens + completion_tokens)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
