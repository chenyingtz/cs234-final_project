"""
Model configuration constants.
Centralized location for base model name that can be easily modified.
"""

import json
from pathlib import Path
from typing import Optional

# Default base model (can be overridden by config file)
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Path to default config file
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "models_config.json"


def get_base_model(config_path: Optional[str] = None) -> str:
    """
    Get the base model name from config file or return default.
    
    Args:
        config_path: Path to config file. If None, uses default config.
    
    Returns:
        Base model name string.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    config_file = Path(config_path)
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
                return config.get("base_model", DEFAULT_BASE_MODEL)
        except Exception as e:
            print(f"Warning: Could not load base_model from {config_path}: {e}")
            print(f"Using default: {DEFAULT_BASE_MODEL}")
    
    return DEFAULT_BASE_MODEL


# Export the default for direct import
BASE_MODEL = DEFAULT_BASE_MODEL
