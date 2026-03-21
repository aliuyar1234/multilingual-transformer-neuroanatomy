from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import load_config


LANGUAGE_INDEX = {"en": 0, "fr": 1, "zh": 2}


def _mirror_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["prior_mirror_root"]).resolve()


def _feature_manifest_path(mirror_root: Path) -> Path:
    return mirror_root / "data" / "processed" / "features" / "feature_manifest.parquet"


def _triplets_path(mirror_root: Path) -> Path:
    return mirror_root / "data" / "processed" / "alignment_triplets.parquet"


def _permutation_manifest_path(mirror_root: Path, model_name: str) -> Path:
    return mirror_root / "data" / "processed" / "features" / model_name / "mismatched_shared_permutations.parquet"


def _feature_array_path(
    mirror_root: Path,
    *,
    model_name: str,
    language: str,
    layer_index: int,
    shuffle_index: int,
) -> Path:
    return (
        mirror_root
        / "data"
        / "processed"
        / "features"
        / model_name
        / language
        / "mismatched_shared"
        / f"shuffle_{shuffle_index:02d}"
        / f"layer_{layer_index:02d}.npy"
    )


def _derangement_for_indices(indices: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    if len(indices) < 2:
        raise RuntimeError("MISMATCHED_SHARED requires at least two triplets within each canonical run/section.")
    for _ in range(128):
        perm = rng.permutation(indices)
        if not np.any(perm == indices):
            return perm
    return np.roll(indices, 1)


def _build_run_local_permutations(
    triplets: pd.DataFrame,
    *,
    model_name: str,
    language: str,
    n_shuffles: int,
    random_seed: int,
) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    metadata_rows: list[dict[str, Any]] = []
    permutations: dict[int, np.ndarray] = {}
    base_index = np.arange(len(triplets), dtype=np.int64)
    sections = triplets["section_index"].astype(int).to_numpy()
    triplet_ids = triplets["triplet_id"].astype(int).to_numpy()
    language_index = LANGUAGE_INDEX[language]

    for shuffle_index in range(n_shuffles):
        rng = np.random.default_rng(random_seed + 10_000 * language_index + shuffle_index)
        permuted_index = base_index.copy()
        for section_index in sorted(np.unique(sections).tolist()):
            mask = np.flatnonzero(sections == section_index)
            permuted_index[mask] = _derangement_for_indices(mask, rng=rng)
        permutations[shuffle_index] = permuted_index
        metadata_rows.extend(
            {
                "model": model_name,
                "language": language,
                "shuffle_index": shuffle_index,
                "section_index": int(section_index),
                "triplet_id": int(triplet_ids[row_index]),
                "permuted_triplet_id": int(triplet_ids[permuted_index[row_index]]),
            }
            for row_index, section_index in enumerate(sections)
        )

    return permutations, pd.DataFrame(metadata_rows)


def refresh_mismatched_shared_in_mirror(
    config: dict[str, Any],
    *,
    n_shuffles: int = 20,
) -> tuple[Path, list[Path]]:
    mirror_root = _mirror_root(config)
    feature_manifest_path = _feature_manifest_path(mirror_root)
    triplets = pd.read_parquet(_triplets_path(mirror_root)).copy()
    feature_manifest = pd.read_parquet(feature_manifest_path).copy()

    seed = int(config["project"]["seed"]) if "project" in config and "seed" in config["project"] else 1337
    # Keep the prior paper's original random-seed surface for permutation families.
    prior_style_seed = 20260314 if seed == 1337 else seed

    non_mismatch = feature_manifest.loc[feature_manifest["condition"] != "mismatched_shared"].copy()
    new_rows: list[dict[str, Any]] = []
    permutation_paths: list[Path] = []

    for model_name in sorted(non_mismatch["model"].astype(str).unique().tolist()):
        model_shared = non_mismatch.loc[non_mismatch["model"] == model_name].copy()
        permutation_frames: list[pd.DataFrame] = []
        for language in sorted(model_shared["language"].astype(str).unique().tolist()):
            lang_shared = model_shared.loc[
                (model_shared["language"] == language) & (model_shared["condition"] == "shared")
            ].copy()
            permutations, metadata_df = _build_run_local_permutations(
                triplets,
                model_name=model_name,
                language=language,
                n_shuffles=n_shuffles,
                random_seed=prior_style_seed,
            )
            permutation_frames.append(metadata_df)
            for row in lang_shared.itertuples(index=False):
                shared = np.load(row.filepath).astype(np.float32, copy=False)
                layer_index = int(row.layer_index)
                for shuffle_index, permuted_index in permutations.items():
                    mismatched_shared = shared[permuted_index]
                    array_path = _feature_array_path(
                        mirror_root,
                        model_name=model_name,
                        language=language,
                        layer_index=layer_index,
                        shuffle_index=shuffle_index,
                    )
                    array_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(array_path, mismatched_shared.astype(np.float32, copy=False))
                    new_rows.append(
                        {
                            "model": model_name,
                            "language": language,
                            "layer_index": layer_index,
                            "condition": "mismatched_shared",
                            "shuffle_index": shuffle_index,
                            "filepath": str(array_path),
                            "n_rows": int(mismatched_shared.shape[0]),
                            "feature_dim": int(mismatched_shared.shape[1]),
                        }
                    )
        permutation_path = _permutation_manifest_path(mirror_root, model_name)
        permutation_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(permutation_frames, ignore_index=True).to_parquet(permutation_path, index=False)
        permutation_paths.append(permutation_path)

    refreshed_manifest = pd.concat([non_mismatch, pd.DataFrame(new_rows)], ignore_index=True)
    refreshed_manifest = refreshed_manifest.sort_values(
        ["model", "language", "condition", "shuffle_index", "layer_index"],
        na_position="first",
    ).reset_index(drop=True)
    refreshed_manifest.to_parquet(feature_manifest_path, index=False)
    return feature_manifest_path, permutation_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh local mirror mismatched_shared assets to the sequel shuffle count.")
    parser.add_argument("--config", default="conf/base.yaml", help="Path to sequel config.")
    parser.add_argument("--n-shuffles", type=int, default=20, help="Number of run-local mismatch shuffles to materialize.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    manifest_path, permutation_paths = refresh_mismatched_shared_in_mirror(config, n_shuffles=args.n_shuffles)
    print(f"Refreshed mirror feature manifest at {manifest_path}")
    for path in permutation_paths:
        print(f"Wrote permutation manifest {path}")


if __name__ == "__main__":
    main()
