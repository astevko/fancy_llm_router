"""
Fancy LLM Router - A production-grade LLM routing and evaluation system.
"""

__version__ = "0.1.0"

from fancy_llm_router.core.router import LLMRouter
from fancy_llm_router.core.session import Session, SessionManager
from fancy_llm_router.metrics.collector import MetricsCollector
from fancy_llm_router.models.base import BaseModelProvider
from fancy_llm_router.tools.base import BaseTool

__all__ = [
    "__version__",
    "LLMRouter",
    "Session",
    "SessionManager",
    "MetricsCollector",
    "BaseModelProvider",
    "BaseTool",
]
