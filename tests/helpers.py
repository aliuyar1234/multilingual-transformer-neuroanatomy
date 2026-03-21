from __future__ import annotations

from pathlib import Path
import uuid


def test_scratch_root() -> Path:
    root = Path.cwd() / "outputs" / ".test_scratch"
    root.mkdir(parents=True, exist_ok=True)
    return root


def make_temp_test_dir(label: str) -> Path:
    path = test_scratch_root() / f"{label}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path
