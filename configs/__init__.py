"""Configuration files for the LLM Router."""

import os
from pathlib import Path
from typing import Optional, Dict, Any

import yaml


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file. If None, tries:
            - configs/local.yaml
            - configs/default.yaml
            - configs/example.yaml
    
    Returns:
        Dictionary with configuration
    """
    if config_path is None:
        # Try default paths
        base_dir = Path(__file__).parent
        for path in ["local.yaml", "default.yaml", "example.yaml"]:
            full_path = base_dir / path
            if full_path.exists():
                config_path = str(full_path)
                break
    
    if config_path is None:
        return {}
    
    # Expand environment variables
    config_path = os.path.expandvars(config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Expand environment variables in values
    if config:
        config = _expand_env_vars(config)
    
    return config or {}


def _expand_env_vars(data: Any) -> Any:
    """Recursively expand environment variables in a data structure."""
    import os
    
    if isinstance(data, str):
        # Check if the string is in the format ${VAR_NAME}
        if data.startswith("${$") and data.endswith("}"):
            var_name = data[2:-1]
            return os.environ.get(var_name, data)
        return data
    
    elif isinstance(data, dict):
        return {k: _expand_env_vars(v) for k, v in data.items()}
    
    elif isinstance(data, list):
        return [_expand_env_vars(item) for item in data]
    
    return data


def get_config_path() -> Optional[str]:
    """Get the path to the configuration file."""
    # Check environment variable
    config_path = os.environ.get("FANCY_LLM_CONFIG")
    if config_path and os.path.exists(config_path):
        return config_path
    
    # Check default paths
    base_dir = Path(__file__).parent
    for path in ["local.yaml", "default.yaml", "example.yaml"]:
        full_path = base_dir / path
        if full_path.exists():
            return str(full_path)
    
    return None
