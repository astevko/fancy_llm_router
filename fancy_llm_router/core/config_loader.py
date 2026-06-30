"""Load YAML configuration files and build a router with registered models."""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from fancy_llm_router.core.router import LLMRouter
from fancy_llm_router.models.base import ModelProviderFactory
from fancy_llm_router.schemas.models import (
    ModelCapabilities,
    ModelInfo,
    ModelPricing,
    ModelProvider,
)
from fancy_llm_router.schemas.routing import RoutingStrategy

logger = logging.getLogger(__name__)

# Default locations searched when no explicit path is given (first match wins).
DEFAULT_CONFIG_PATHS = (
    "configs/local.yaml",
    "configs/local.yml",
)

_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _ensure_dotenv_loaded() -> None:
    """Load a ``.env`` file from the working directory if present.

    ``run.sh`` sources ``.env`` for shell users; this mirrors that behavior when
    invoking ``uv run fancy-llm`` directly so ``${NEBIUS_API_KEY}`` etc. resolve.
    Existing environment variables are not overwritten.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


def _expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` references using environment variables."""
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def find_default_config() -> Optional[Path]:
    """Return the first existing default config path, if any."""
    for candidate in DEFAULT_CONFIG_PATHS:
        path = Path(candidate)
        if path.exists():
            return path
    return None


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load and env-expand a YAML configuration file."""
    _ensure_dotenv_loaded()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return _expand_env(raw)


def _model_info_from_entry(key: str, entry: Dict[str, Any]) -> ModelInfo:
    """Build a ``ModelInfo`` from a single ``models`` config entry.

    The YAML key is the unique ``deployment_id``. ``model`` is the logical model
    name callers ask for (defaults to the wire ``model_id``), so the same model
    can be served by several deployments/sources at once.
    """
    # Logical model name and the wire id sent to the host's API.
    logical_model = entry.get("model") or entry.get("model_id") or key
    model_id = entry.get("model_id") or logical_model
    context_window = int(entry.get("context_window") or 4096)
    max_tokens = int(entry.get("max_tokens") or entry.get("default_max_tokens") or context_window)

    capabilities = ModelCapabilities(
        max_tokens=max_tokens,
        max_input_tokens=context_window,
        context_window=context_window,
        supports_chat=entry.get("supports_chat", True),
        supports_completions=entry.get("supports_completions", True),
        supports_streaming=entry.get("supports_streaming", True),
        supports_embeddings=entry.get("supports_embeddings", False),
        tokens_per_second=entry.get("tokens_per_second"),
        quantization=entry.get("quantization"),
    )

    pricing = ModelPricing(
        input_token_price=float(entry.get("input_token_price") or 0.0),
        output_token_price=float(entry.get("output_token_price") or 0.0),
    )

    return ModelInfo(
        provider=ModelProvider(entry.get("provider", "custom")),
        model_id=model_id,
        deployment_id=key,
        model=logical_model,
        source=entry.get("source"),
        name=entry.get("name") or logical_model,
        capabilities=capabilities,
        pricing=pricing,
        metadata=entry.get("metadata", {}) or {},
    )


def build_router_from_config(config: Dict[str, Any]) -> LLMRouter:
    """Construct an ``LLMRouter`` and register all enabled models from config."""
    router_cfg = config.get("router", {}) or {}
    try:
        default_strategy = RoutingStrategy(router_cfg.get("default_strategy", "balanced"))
    except ValueError:
        default_strategy = RoutingStrategy.BALANCED

    router = LLMRouter(default_strategy=default_strategy)

    models: Dict[str, Any] = config.get("models", {}) or {}
    for key, entry in models.items():
        if not isinstance(entry, dict):
            continue
        if not entry.get("enabled", True):
            continue
        try:
            model_info = _model_info_from_entry(key, entry)
            api_key = entry.get("api_key") or None
            if api_key is not None and not str(api_key).strip():
                logger.warning(
                    "Deployment %s: api_key is empty — add it to .env or export the "
                    "variable referenced in the config (e.g. NEBIUS_API_KEY)",
                    key,
                )
                api_key = None
            # Build the provider with the configured connection details so the
            # router can actually reach the backend (base URL + API key).
            provider = ModelProviderFactory.create(
                entry.get("provider", "custom"),
                model_info.model_id,
                api_key=api_key,
                base_url=entry.get("api_base_url") or None,
            )
            router.register_model(model_info, provider=provider)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to register model %s: %s", key, exc)

    return router


def create_router(config_path: Optional[Union[str, Path]] = None) -> LLMRouter:
    """Create a router from an explicit path, the default path, or empty.

    Args:
        config_path: Explicit config file. If ``None``, the default locations
            are searched; if none exist, an empty router is returned.
    """
    if config_path is None:
        found = find_default_config()
        if found is None:
            return LLMRouter()
        config_path = found

    config = load_config(config_path)
    return build_router_from_config(config)


async def create_storage_from_config(
    config: Dict[str, Any],
) -> Optional["SQLiteStorage"]:
    """Create and initialize SQLite storage from a loaded config dict."""
    from fancy_llm_router.storage.sqlite import SQLiteStorage

    storage_cfg = config.get("storage") or {}
    backend = storage_cfg.get("backend", "sqlite")
    if backend != "sqlite":
        return None
    db_path = (
        storage_cfg.get("sqlite_path")
        or storage_cfg.get("db_path")
        or "data/metrics.db"
    )
    storage = SQLiteStorage(db_path=db_path)
    await storage.initialize()
    return storage


def get_storage_db_path(config: Dict[str, Any]) -> str:
    """Resolve the SQLite path used by metrics and prompt registry."""
    storage_cfg = config.get("storage") or {}
    return (
        storage_cfg.get("sqlite_path")
        or storage_cfg.get("db_path")
        or "data/metrics.db"
    )


def build_app_config(config_dict: Optional[Dict[str, Any]] = None) -> "AppConfig":
    """Merge YAML app settings and environment into ``AppConfig``."""
    from fancy_llm_router.schemas.config import AppConfig

    _ensure_dotenv_loaded()
    cfg = AppConfig()
    token = cfg.api_auth_token

    if config_dict:
        app_section = config_dict.get("app") or {}
        yaml_token = app_section.get("api_auth_token") or config_dict.get("api_auth_token")
        if yaml_token and str(yaml_token).strip():
            token = str(yaml_token).strip()

    if not token:
        env_token = os.environ.get("ROUTER_API_KEY")
        if env_token and env_token.strip():
            token = env_token.strip()

    cfg.api_auth_token = token if token and str(token).strip() else None
    return cfg
