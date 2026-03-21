from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.encoding.run_state_sweep import _group_subject_results, _group_results_path, _provenance_path, _subject_results_path
from src.utils.config import load_config


def _pattern(path: Path) -> str:
    suffix = "".join(path.suffixes)
    stem = path.name[: -len(suffix)] if suffix else path.name
    return f"{stem}__*{suffix}"


def _glob_tagged_paths(path: Path) -> list[Path]:
    return sorted(candidate for candidate in path.parent.glob(_pattern(path)) if candidate.is_file())


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("shards"), list) or not data["shards"]:
        raise RuntimeError(f"Invalid state-sweep manifest: {manifest_path}")
    return data


def _manifest_subject_paths(manifest: dict[str, Any]) -> list[Path]:
    paths = [Path(str(entry["subject_results_path"])) for entry in manifest["shards"]]
    missing = [path for path in paths if not path.exists()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        suffix = "" if len(missing) <= 3 else f" ... (+{len(missing) - 3} more)"
        raise FileNotFoundError(f"Missing expected state-sweep shard outputs: {preview}{suffix}")
    return paths


def merge_state_sweep_shards(
    config: dict[str, Any],
    *,
    manifest_path: str | Path | None = None,
) -> dict[str, Path]:
    manifest: dict[str, Any] | None = None
    if manifest_path is not None:
        manifest = _load_manifest(Path(manifest_path))
        subject_shards = _manifest_subject_paths(manifest)
        expected_languages = {str(entry["language"]).upper() for entry in manifest["shards"]}
    else:
        subject_shards = _glob_tagged_paths(_subject_results_path(config))
        expected_languages = set()
    if not subject_shards:
        raise FileNotFoundError("No tagged state-sweep subject-result shards were found to merge.")

    subject_df = pd.concat([pd.read_parquet(path).copy() for path in subject_shards], ignore_index=True)
    duplicate_mask = subject_df.duplicated(
        subset=["language", "model", "subject_id", "roi", "block", "state", "condition"],
        keep=False,
    )
    if duplicate_mask.any():
        duplicate_rows = subject_df.loc[
            duplicate_mask,
            ["language", "model", "subject_id", "roi", "block", "state", "condition"],
        ].drop_duplicates()
        preview = duplicate_rows.head(3).to_dict(orient="records")
        suffix = "" if len(duplicate_rows) <= 3 else f" ... (+{len(duplicate_rows) - 3} more)"
        raise RuntimeError(f"Duplicate subject rows detected across sweep shards: {preview}{suffix}")
    subject_df = subject_df.sort_values(
        ["language", "model", "subject_id", "roi", "block", "state", "condition"]
    ).reset_index(drop=True)
    if expected_languages:
        observed_languages = set(subject_df["language"].astype(str).str.upper())
        if observed_languages != expected_languages:
            raise RuntimeError(
                "Merged state-sweep shards do not match expected languages: "
                f"expected {sorted(expected_languages)}, got {sorted(observed_languages)}."
            )
    group_df = _group_subject_results(subject_df)

    subject_path = _subject_results_path(config)
    group_path = _group_results_path(config)
    subject_path.parent.mkdir(parents=True, exist_ok=True)
    group_path.parent.mkdir(parents=True, exist_ok=True)
    subject_df.to_parquet(subject_path, index=False)
    group_df.to_parquet(group_path, index=False)

    provenance_path = _provenance_path(config)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        "\n".join(
            [
                "# State Sweep Merge",
                "",
                f"- shard_subject_results: `{len(subject_shards)}`",
                f"- manifest_path: `{manifest_path or 'glob discovery'}`",
                f"- run_id: `{manifest.get('run_id', 'unknown') if manifest is not None else 'unknown'}`",
                f"- subject_results: `{subject_path}`",
                f"- group_results: `{group_path}`",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "subject_results": subject_path,
        "group_results": group_path,
        "provenance": provenance_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge tagged state-sweep language shards into canonical outputs.")
    parser.add_argument("--config", default="conf/base.yaml", help="Path to sequel config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    outputs = merge_state_sweep_shards(config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
