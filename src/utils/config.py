from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded or {}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    root = _load_yaml(config_path)
    includes = root.pop("includes", [])
    merged: dict[str, Any] = {}
    for include in includes:
        include_path = (config_path.parent / include).resolve()
        merged = _deep_merge(merged, _load_yaml(include_path))
    return _deep_merge(merged, root)
