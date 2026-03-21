from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import load_config


def _local_outputs_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["local_outputs_root"]).resolve()


def _now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _tagged_path(path: Path, artifact_tag: str | None) -> Path:
    if not artifact_tag:
        return path
    suffix = "".join(path.suffixes)
    stem = path.name[: -len(suffix)] if suffix else path.name
    tagged_name = f"{stem}__{artifact_tag}{suffix}"
    return path.with_name(tagged_name)


def _triplets_path(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["local_manifests_root"]).resolve() / "triplets.parquet"


def _state_cache_manifest_path(config: dict[str, Any], artifact_tag: str | None = None) -> Path:
    path = _local_outputs_root(config) / "caches" / "features" / "state_cache_manifest.parquet"
    return _tagged_path(path, artifact_tag)


def _output_state_manifest_path(config: dict[str, Any], artifact_tag: str | None = None) -> Path:
    path = _local_outputs_root(config) / "caches" / "features" / "output_state_manifest.parquet"
    return _tagged_path(path, artifact_tag)


def _token_metadata_path(config: dict[str, Any], artifact_tag: str | None = None) -> Path:
    path = _local_outputs_root(config) / "caches" / "features" / "tokenization_metadata.parquet"
    return _tagged_path(path, artifact_tag)


def _triplet_ids_path(config: dict[str, Any], artifact_tag: str | None = None) -> Path:
    path = _local_outputs_root(config) / "caches" / "features" / "triplet_ids.parquet"
    return _tagged_path(path, artifact_tag)


def _state_extraction_qc_path(config: dict[str, Any], artifact_tag: str | None = None) -> Path:
    path = _local_outputs_root(config) / "qc" / "state_extraction_equivalence.parquet"
    return _tagged_path(path, artifact_tag)


def _state_extraction_note_path(config: dict[str, Any], artifact_tag: str | None = None) -> Path:
    path = _local_outputs_root(config) / "provenance" / "state_extraction.md"
    return _tagged_path(path, artifact_tag)


def _state_adapter_note_path(config: dict[str, Any], artifact_tag: str | None = None) -> Path:
    path = _local_outputs_root(config) / "provenance" / "state_adapter_notes.md"
    return _tagged_path(path, artifact_tag)


def _write_state_adapter_notes(config: dict[str, Any]) -> Path:
    note_path = _state_adapter_note_path(config)
    note_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# State Adapter Notes",
        "",
        f"- generated_utc: `{_now_utc()}`",
        "",
        "## XLM-R / Roberta",
        "",
        "- `INPUT`: encoder-layer forward pre-hook",
        "- `ATTN`: attention-side delta before residual normalization",
        "- `POST_ATTN`: post-attention residual state",
        "- `FFN`: FFN-side delta before residual normalization",
        "- `OUTPUT`: encoder-layer forward output",
        "",
        "## NLLB / M2M100 encoder",
        "",
        "- `INPUT`: encoder-layer forward pre-hook",
        "- `ATTN`: attention branch update in eval mode",
        "- `POST_ATTN`: post-attention residual state",
        "- `FFN`: FFN branch update in eval mode",
        "- `OUTPUT`: encoder-layer forward output",
        "",
    ]
    note_path.write_text("\n".join(lines), encoding="utf-8")
    return note_path


def _pattern(path: Path) -> str:
    suffix = "".join(path.suffixes)
    stem = path.name[: -len(suffix)] if suffix else path.name
    return f"{stem}__*{suffix}"


def _glob_tagged_paths(path: Path) -> list[Path]:
    return sorted(candidate for candidate in path.parent.glob(_pattern(path)) if candidate.is_file())


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("shards"), list) or not data["shards"]:
        raise RuntimeError(f"Invalid extraction manifest: {manifest_path}")
    return data


def _manifest_paths(manifest: dict[str, Any], *, key: str) -> list[Path]:
    paths = [Path(str(entry[key])) for entry in manifest["shards"]]
    missing = [path for path in paths if not path.exists()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        suffix = "" if len(missing) <= 3 else f" ... (+{len(missing) - 3} more)"
        raise FileNotFoundError(f"Missing expected extraction shard outputs: {preview}{suffix}")
    return paths


def _concat_frames(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(path).copy() for path in paths if path.exists()]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _write_sorted(frame: pd.DataFrame, path: Path, *, sort_columns: tuple[str, ...]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frame.empty:
        frame = frame.sort_values(list(sort_columns)).reset_index(drop=True)
    frame.to_parquet(path, index=False)
    return path


def _assert_no_duplicates(frame: pd.DataFrame, *, subset: list[str], label: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    duplicate_mask = frame.duplicated(subset=subset, keep=False)
    if duplicate_mask.any():
        duplicate_rows = frame.loc[duplicate_mask, subset].drop_duplicates().to_dict(orient="records")
        preview = duplicate_rows[:3]
        suffix = "" if len(duplicate_rows) <= 3 else f" ... (+{len(duplicate_rows) - 3} more)"
        raise RuntimeError(f"Duplicate {label} rows detected during consolidation: {preview}{suffix}")
    return frame


def _deduplicate_rows(frame: pd.DataFrame, *, subset: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.drop_duplicates(subset=subset).reset_index(drop=True)


def consolidate_internal_state_shards(
    config: dict[str, Any],
    *,
    manifest_path: str | Path | None = None,
) -> dict[str, Path]:
    manifest: dict[str, Any] | None = None
    if manifest_path is not None:
        manifest = _load_manifest(Path(manifest_path))
        state_shards = _manifest_paths(manifest, key="state_manifest_path")
        output_shards = _manifest_paths(manifest, key="output_manifest_path")
        token_shards = _manifest_paths(manifest, key="token_metadata_path")
        triplet_shards = _manifest_paths(manifest, key="triplet_ids_path")
        qc_shards = _manifest_paths(manifest, key="qc_path")
    else:
        state_shards = _glob_tagged_paths(_state_cache_manifest_path(config))
        if not state_shards:
            raise FileNotFoundError("No tagged state-cache shard manifests were found to consolidate.")
        output_shards = _glob_tagged_paths(_output_state_manifest_path(config))
        token_shards = _glob_tagged_paths(_token_metadata_path(config))
        triplet_shards = _glob_tagged_paths(_triplet_ids_path(config))
        qc_shards = _glob_tagged_paths(_state_extraction_qc_path(config))

    state_df = _assert_no_duplicates(
        _concat_frames(state_shards),
        subset=["model", "language", "state", "block"],
        label="state-manifest",
    )
    output_df = _assert_no_duplicates(
        _concat_frames(output_shards),
        subset=["model", "language", "block"],
        label="output-manifest",
    )
    token_df = _assert_no_duplicates(
        _concat_frames(token_shards),
        subset=["model", "language", "triplet_id"],
        label="token-metadata",
    )
    triplet_df = _deduplicate_rows(
        _concat_frames(triplet_shards),
        subset=["triplet_id"],
    )
    qc_df = _assert_no_duplicates(
        _concat_frames(qc_shards),
        subset=["model", "language", "block", "state"],
        label="state-qc",
    )

    state_manifest_path = _write_sorted(
        state_df,
        _state_cache_manifest_path(config),
        sort_columns=("model", "language", "state", "block"),
    )
    output_manifest_path = _write_sorted(
        output_df,
        _output_state_manifest_path(config),
        sort_columns=("model", "language", "block"),
    )
    token_metadata_path = _write_sorted(
        token_df,
        _token_metadata_path(config),
        sort_columns=("model", "language", "triplet_id"),
    )
    if triplet_df.empty:
        triplets = pd.read_parquet(_triplets_path(config)).copy()
        triplet_df = triplets.loc[:, ["triplet_id"]].copy()
    triplet_ids_path = _write_sorted(
        triplet_df,
        _triplet_ids_path(config),
        sort_columns=("triplet_id",),
    )
    qc_path = _write_sorted(
        qc_df,
        _state_extraction_qc_path(config),
        sort_columns=("model", "language", "block", "state"),
    )

    adapter_note_path = _write_state_adapter_notes(config)
    provenance_path = _state_extraction_note_path(config)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        "\n".join(
            [
                "# State Extraction Consolidation",
                "",
                f"- generated_utc: `{_now_utc()}`",
                f"- shard_state_manifests: `{len(state_shards)}`",
                f"- shard_output_manifests: `{len(output_shards)}`",
                f"- shard_token_metadata_tables: `{len(token_shards)}`",
                f"- shard_qc_tables: `{len(qc_shards)}`",
                f"- manifest_path: `{manifest_path or 'glob discovery'}`",
                f"- run_id: `{manifest.get('run_id', 'unknown') if manifest is not None else 'unknown'}`",
                f"- state_cache_manifest: `{state_manifest_path}`",
                f"- output_state_manifest: `{output_manifest_path}`",
                f"- token_metadata: `{token_metadata_path}`",
                f"- triplet_ids: `{triplet_ids_path}`",
                f"- output_equivalence_qc: `{qc_path}`",
                f"- adapter_notes: `{adapter_note_path}`",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "state_manifest": state_manifest_path,
        "output_manifest": output_manifest_path,
        "token_metadata": token_metadata_path,
        "triplet_ids": triplet_ids_path,
        "qc": qc_path,
        "provenance": provenance_path,
        "adapter_notes": adapter_note_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidate tagged internal-state extraction shards into canonical manifests.")
    parser.add_argument("--config", default="conf/base.yaml", help="Path to sequel config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    outputs = consolidate_internal_state_shards(config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
