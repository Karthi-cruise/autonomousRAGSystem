"""Configuration loader for the RAG system."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        base = Path(__file__).parent.parent.parent
        config_path = base / "configs" / "rag.yaml"
    
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent
