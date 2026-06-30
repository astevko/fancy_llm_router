"""Shared hashing helpers."""

import hashlib


def prompt_hash(text: str, length: int = 16) -> str:
    """Return a SHA256 hex digest, truncated for deduplication keys."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]
