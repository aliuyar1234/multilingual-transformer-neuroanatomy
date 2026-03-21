"""Attribution-stage helpers and CLI entry points."""

from __future__ import annotations

from typing import Any

__all__ = ["compute_word_attributions", "run_deletion_validation", "select_representative_states"]


def __getattr__(name: str) -> Any:
    if name == "compute_word_attributions":
        from src.attribution.compute_word_attributions import compute_word_attributions

        return compute_word_attributions
    if name == "run_deletion_validation":
        from src.attribution.run_deletion_validation import run_deletion_validation

        return run_deletion_validation
    if name == "select_representative_states":
        from src.attribution.select_representative_states import select_representative_states

        return select_representative_states
    raise AttributeError(name)
