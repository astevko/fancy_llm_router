"""FastAPI server for the LLM Router."""

import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from fancy_llm_router.schemas.config import AppConfig
from fancy_llm_router.core.router import LLMRouter
from fancy_llm_router.core.config_loader import (
    create_router,
    load_config,
    find_default_config,
    create_storage_from_config,
    get_storage_db_path,
)
from fancy_llm_router.core.benchmark_service import BenchmarkService
from fancy_llm_router.metrics.collector import MetricsCollector
from fancy_llm_router.core.prompt_registry import PromptRegistry
from fancy_llm_router.tools.base import ToolRegistry
from fancy_llm_router.api.routes import router as api_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting LLM Router API...")

    config_path = getattr(app.state, "config_path", None)
    config_dict: Dict[str, Any] = {}
    if config_path:
        try:
            config_dict = load_config(config_path)
        except FileNotFoundError:
            logger.warning("Config file not found: %s", config_path)
    elif find_default_config():
        config_dict = load_config(find_default_config())

    storage = await create_storage_from_config(config_dict)
    app.state.metrics = MetricsCollector(storage=storage)
    app.state.tools = ToolRegistry()
    app.state.config = AppConfig()

    db_path = get_storage_db_path(config_dict)
    app.state.prompt_registry = PromptRegistry(db_path=db_path)
    app.state.prompt_registry.initialize()

    try:
        app.state.router = create_router(config_path)
        app.state.router.set_prompt_registry(app.state.prompt_registry)
        logger.info(
            "Registered %d model(s) from config",
            len(app.state.router.list_models()),
        )
    except Exception as e:
        logger.error(f"Failed to load model configuration: {e}")
        app.state.router = LLMRouter()
        app.state.router.set_prompt_registry(app.state.prompt_registry)

    app.state.benchmark = BenchmarkService(
        router=app.state.router,
        registry=app.state.prompt_registry,
        metrics=app.state.metrics,
    )

    await _initialize_tools(app)

    logger.info("LLM Router API started successfully")

    yield

    logger.info("Shutting down LLM Router API...")

    if hasattr(app.state, "router"):
        await app.state.router.close()
    if hasattr(app.state, "metrics"):
        await app.state.metrics.close()
    if hasattr(app.state, "tools"):
        await app.state.tools.close_all()

    logger.info("LLM Router API shutdown complete")


async def _initialize_tools(app: FastAPI):
    """Initialize tools from configuration."""
    config = app.state.config
    tools = app.state.tools

    for tool_id, tool_config in config.tools.items():
        try:
            from fancy_llm_router.tools.mock import MockTool

            tool = MockTool(
                tool_id=tool_id,
                name=tool_config.name or tool_id,
                description=tool_config.description or "",
            )

            tools.register(tool)
            logger.info(f"Initialized tool: {tool_id}")

        except Exception as e:
            logger.error(f"Failed to initialize tool {tool_id}: {e}")


def create_app(
    config: Optional[AppConfig] = None,
    config_path: Optional[str] = None,
) -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="Fancy LLM Router",
        description="A production-grade LLM routing and evaluation system",
        version="0.1.0",
        lifespan=lifespan,
    )

    if config:
        app.state.config = config

    app.state.config_path = config_path

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)},
        )

    app.include_router(api_router, prefix="/api/v1")

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    @app.get("/")
    async def root():
        return {
            "name": "Fancy LLM Router",
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/health",
        }

    return app


if __name__ == "__main__":
    import uvicorn

    app = create_app()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
