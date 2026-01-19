"""
Configuration loading utilities.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        config_path: Path to the YAML config file

    Returns:
        Dictionary containing configuration
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_config_path(config_name: str) -> Path:
    """Get the full path to a config file."""
    return get_project_root() / "configs" / config_name


class Config:
    """Configuration container with dot notation access."""

    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value with optional default."""
        return getattr(self, key, default)


def load_model_config() -> Config:
    """Load model configuration."""
    config_dict = load_config(get_config_path("model_config.yaml"))
    return Config(config_dict)


def load_training_config() -> Config:
    """Load training configuration."""
    config_dict = load_config(get_config_path("training_config.yaml"))
    return Config(config_dict)


def load_inference_config() -> Config:
    """Load inference configuration."""
    config_dict = load_config(get_config_path("inference_config.yaml"))
    return Config(config_dict)


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    Later configs override earlier ones.
    """
    result = {}
    for config in configs:
        _deep_merge(result, config)
    return result


def _deep_merge(base: Dict, override: Dict) -> None:
    """Deep merge override into base."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
