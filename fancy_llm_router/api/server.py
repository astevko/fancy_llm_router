"""FastAPI server for the LLM Router."""

import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from fancy_llm_router.schemas.config import AppConfig
from fancy_llm_router.core.router import LLMRouter
from fancy_llm_router.metrics.collector import MetricsCollector
from fancy_llm_router.tools.base import ToolRegistry
from fancy_llm_router.api.routes import router as api_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting LLM Router API...")
    
    # Initialize components
    app.state.router = LLMRouter()
    app.state.metrics = MetricsCollector()
    app.state.tools = ToolRegistry()
    
    # Load configuration
    app.state.config = AppConfig()
    
    # Initialize models
    await _initialize_models(app)
    
    # Initialize tools
    await _initialize_tools(app)
    
    logger.info("LLM Router API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down LLM Router API...")
    
    # Clean up
    if hasattr(app.state, 'router'):
        await app.state.router.close()
    if hasattr(app.state, 'metrics'):
        await app.state.metrics.close()
    if hasattr(app.state, 'tools'):
        await app.state.tools.close_all()
    
    logger.info("LLM Router API shutdown complete")


async def _initialize_models(app: FastAPI):
    """Initialize model providers from configuration."""
    config = app.state.config
    router = app.state.router
    
    # Register models from configuration
    for model_id, model_config in config.models.items():
        try:
            from fancy_llm_router.models.base import ModelProviderFactory
            from fancy_llm_router.schemas.models import ModelInfo, ModelProvider
            
            # Create model info
            model_info = ModelInfo(
                provider=ModelProvider(model_config.provider),
                model_id=model_id,
                name=model_config.name or model_id,
                capabilities=model_config.capabilities,
                pricing=model_config.pricing,
            )
            
            # Register with router
            router.register_model(model_info)
            
            logger.info(f"Initialized model: {model_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model {model_id}: {e}")


async def _initialize_tools(app: FastAPI):
    """Initialize tools from configuration."""
    config = app.state.config
    tools = app.state.tools
    
    # Register tools from configuration
    for tool_id, tool_config in config.tools.items():
        try:
            # In a real implementation, this would create tool instances
            # based on the configuration
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


def create_app(config: Optional[AppConfig] = None) -> FastAPI:
    """
    Create the FastAPI application.
    
    Args:
        config: Optional application configuration
        
    Returns:
        FastAPI application instance
    """
    app = FastAPI(
        title="Fancy LLM Router",
        description="A production-grade LLM routing and evaluation system",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    # Store config
    if config:
        app.state.config = config
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure properly in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add exception handler
    @app.exception_handler(Exception)
    async def exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)},
        )
    
    # Include API routes
    app.include_router(api_router, prefix="/api/v1")
    
    # Health check endpoint
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
