"""API endpoints for the LLM Router."""

from fancy_llm_router.api.server import create_app
from fancy_llm_router.api.routes import router as api_router

__all__ = [
    "create_app",
    "api_router",
]
