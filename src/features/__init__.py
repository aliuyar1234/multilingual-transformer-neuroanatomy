"""Feature extraction and cache utilities for the sequel project."""

from __future__ import annotations

import importlib
from typing import Any

from src.features.build_shared_specific import build_output_feature_manifest, build_shared_specific_caches

__all__ = [
    "EncodedStateBundle",
    "ModelStateAdapter",
    "build_model_state_adapter",
    "build_output_feature_manifest",
    "build_shared_specific_caches",
    "extract_internal_states",
    "run_model_forward_with_state_capture",
]


def __getattr__(name: str) -> Any:
    if name in {
        "EncodedStateBundle",
        "ModelStateAdapter",
        "build_model_state_adapter",
        "extract_internal_states",
        "run_model_forward_with_state_capture",
    }:
        extract_internal_states_module = importlib.import_module("src.features.extract_internal_states")
        return getattr(extract_internal_states_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
