from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.bridge.prior_runtime import prior_repo_root
from src.encoding.run_refresh_baseline import _default_mismatch_shuffles, run_refresh_baseline
from src.utils.config import load_config


def _subject_output_path(config: dict[str, Any], *, model: str, output_tag: str) -> Path:
    stem = "nllb" if model == "nllb" else model
    return prior_repo_root(config) / "outputs" / "stats" / f"{stem}_subject_level_roi_results__{output_tag}.parquet"


def _count_subjects(path: Path) -> int:
    if not path.exists():
        return 0
    df = pd.read_parquet(path, columns=["subject_id"])
    return int(df["subject_id"].nunique())


def _subject_metadata_path(config: dict[str, Any], *, model: str, output_tag: str) -> Path:
    return _subject_output_path(config, model=model, output_tag=output_tag).with_name(
        f"{_subject_output_path(config, model=model, output_tag=output_tag).stem}__metadata.json"
    )


def _clear_stale_resume_metadata(subject_path: Path, metadata_path: Path) -> str | None:
    if subject_path.exists() or not metadata_path.exists():
        return None
    metadata_path.unlink()
    return f"Removed stale resume metadata without subject results: {metadata_path}"


def _progress_log_path(config: dict[str, Any], *, model: str, output_tag: str) -> Path:
    root = Path(config["paths"]["local_outputs_root"]).resolve() / "provenance" / "refresh_pass_logs"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{model}__{output_tag}.jsonl"


def _append_log(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def advance_refresh_checkpoint(
    config: dict[str, Any],
    *,
    model: str,
    language: str,
    layers: tuple[int, ...],
    mismatch_shuffles: int,
    output_tag: str,
    pass_timeout_ms: int,
    passes: int,
    mismatch_only: bool = False,
) -> dict[str, Any]:
    subject_path = _subject_output_path(config, model=model, output_tag=output_tag)
    metadata_path = _subject_metadata_path(config, model=model, output_tag=output_tag)
    log_path = _progress_log_path(config, model=model, output_tag=output_tag)

    summary: dict[str, Any] = {
        "model": model,
        "language": language,
        "output_tag": output_tag,
        "passes_requested": passes,
        "pass_timeout_ms": pass_timeout_ms,
        "subject_path": str(subject_path),
        "metadata_path": str(metadata_path),
        "log_path": str(log_path),
        "passes": [],
    }

    for pass_index in range(1, passes + 1):
        subjects_before = _count_subjects(subject_path)
        status = "completed"
        error_text = None
        cleanup_note = _clear_stale_resume_metadata(subject_path, metadata_path)
        try:
            run_refresh_baseline(
                config,
                models=(model,),
                languages=(language,),
                layer_indices=layers,
                mismatch_shuffles=mismatch_shuffles,
                output_tag=output_tag,
                max_subjects=None,
                subject_only=True,
                resume=True,
                mismatch_only=mismatch_only,
                timeout_ms=pass_timeout_ms,
            )
        except subprocess.TimeoutExpired as exc:
            status = "timeout"
            error_text = str(exc)
        except subprocess.CalledProcessError as exc:
            status = "failed"
            error_text = (exc.stderr or exc.stdout or str(exc)).strip()
        subjects_after = _count_subjects(subject_path)
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "pass_index": pass_index,
            "status": status,
            "subjects_before": subjects_before,
            "subjects_after": subjects_after,
            "subjects_added": subjects_after - subjects_before,
            "model": model,
            "language": language,
            "output_tag": output_tag,
            "layers": list(layers),
            "mismatch_shuffles": mismatch_shuffles,
            "mismatch_only": mismatch_only,
        }
        if cleanup_note is not None:
            payload["cleanup"] = cleanup_note
        if error_text is not None:
            payload["error"] = error_text
        _append_log(log_path, payload)
        summary["passes"].append(payload)

        if status == "failed" or subjects_after == subjects_before:
            break

    summary["subjects_total"] = _count_subjects(subject_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advance a resumable refresh checkpoint in timed passes.")
    parser.add_argument("--config", default="conf/base.yaml", help="Path to sequel config.")
    parser.add_argument("--model", required=True, choices=["xlmr", "nllb"])
    parser.add_argument("--language", required=True, choices=["en", "fr", "zh"])
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--mismatch-shuffles", type=int, default=None)
    parser.add_argument("--output-tag", required=True)
    parser.add_argument("--pass-timeout-ms", type=int, default=900000)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--mismatch-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    mismatch_shuffles = (
        int(args.mismatch_shuffles)
        if args.mismatch_shuffles is not None
        else _default_mismatch_shuffles(config, mismatch_only=args.mismatch_only, layer_indices=tuple(args.layers))
    )
    summary = advance_refresh_checkpoint(
        config,
        model=args.model,
        language=args.language,
        layers=tuple(args.layers),
        mismatch_shuffles=mismatch_shuffles,
        output_tag=args.output_tag,
        pass_timeout_ms=args.pass_timeout_ms,
        passes=args.passes,
        mismatch_only=args.mismatch_only,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
