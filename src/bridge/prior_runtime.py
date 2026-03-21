from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any


def prior_repo_root(config: dict[str, Any]) -> Path:
    mirror_root = Path(config["paths"]["prior_mirror_root"]).resolve()
    if mirror_root.exists():
        return mirror_root
    return Path(config["paths"]["prior_repo_root"]).resolve()


def prior_python(prior_repo_root: str | Path, config: dict[str, Any]) -> Path:
    root = Path(prior_repo_root).resolve()
    candidate = root / ".venv" / "Scripts" / "python.exe"
    if candidate.exists():
        return candidate
    configured = config["paths"].get("prior_python")
    if configured:
        configured_path = Path(configured).resolve()
        if configured_path.exists():
            return configured_path
    env_path = os.environ.get("PRIOR_REPO_PYTHON")
    if env_path:
        env_candidate = Path(env_path).resolve()
        if env_candidate.exists():
            return env_candidate
    return Path(os.environ.get("PYTHON", "python"))


def prior_repo_python_env(prior_repo_root: str | Path) -> dict[str, str]:
    root = Path(prior_repo_root).resolve()
    env = os.environ.copy()
    src_path = str((root / "src").resolve())
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_path if not existing else os.pathsep.join((src_path, existing))
    return env


def run_prior_repo_python(
    code: str,
    *,
    prior_repo_root: str | Path,
    python_exe: str | Path | None = None,
    timeout_ms: int | None = None,
) -> subprocess.CompletedProcess[str]:
    root = Path(prior_repo_root).resolve()
    return subprocess.run(
        [str(python_exe or "python"), "-c", code],
        cwd=root,
        env=prior_repo_python_env(root),
        text=True,
        capture_output=True,
        check=True,
        timeout=None if timeout_ms is None else timeout_ms / 1000,
    )
