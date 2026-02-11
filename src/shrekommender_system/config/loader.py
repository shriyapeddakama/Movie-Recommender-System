"""File loading helpers for configuration files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


class ConfigLoaderError(RuntimeError):
    pass


def load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise ConfigLoaderError(
            "PyYAML is required to parse YAML configuration files. Install 'PyYAML' first."
        )
    if not path.exists():
        raise ConfigLoaderError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)  # type: ignore[attr-defined]
    if not isinstance(data, dict):
        raise ConfigLoaderError(f"Configuration {path} must contain a mapping")
    return data


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigLoaderError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ConfigLoaderError(f"Configuration {path} must contain a mapping")
    return data
