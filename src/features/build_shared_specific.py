from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src.utils.config import load_config


MODEL_MAP = {"xlmr": "xlmr", "nllb_encoder": "nllb"}
LANGUAGE_MAP = {"en": "EN", "fr": "FR", "zh": "ZH"}
CONDITION_MAP = {
    "raw": "RAW",
    "shared": "SHARED",
    "specific": "SPECIFIC",
    "full": "FULL",
    "mismatched_shared": "MISMATCHED_SHARED",
}
VALID_STATES = ("INPUT", "ATTN", "POST_ATTN", "FFN", "OUTPUT")
TARGET_LANGUAGES = ("EN", "FR", "ZH")


def _local_outputs_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["local_outputs_root"]).resolve()


def _state_cache_manifest_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "caches" / "features" / "state_cache_manifest.parquet"


def _output_state_feature_manifest_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "caches" / "derived_features" / "output_state_feature_manifest.parquet"


def _state_feature_manifest_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "caches" / "derived_features" / "state_feature_manifest.parquet"


def _shared_specific_qc_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "qc" / "shared_specific_orthogonality.parquet"


def _legacy_output_qc_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "qc" / "output_shared_specific_orthogonality.parquet"


def _derived_array_path(
    config: dict[str, Any],
    *,
    model: str,
    target_language: str,
    state: str,
    condition: str,
    block: int,
) -> Path:
    return (
        _local_outputs_root(config)
        / "caches"
        / "derived_features"
        / model
        / target_language
        / state
        / condition
        / f"block_{int(block):02d}.npy"
    )


def _specific_residual(raw: np.ndarray, shared: np.ndarray, *, eps: float) -> np.ndarray:
    residual = raw - shared
    denom = np.sum(shared * shared, axis=1, keepdims=True) + eps
    projection_scale = np.sum(shared * residual, axis=1, keepdims=True) / denom
    return (residual - projection_scale * shared).astype(np.float32, copy=False)


def _orthogonality_row(
    *,
    model: str,
    target_language: str,
    block: int,
    state: str,
    shared: np.ndarray,
    specific: np.ndarray,
) -> dict[str, object]:
    dot = np.sum(shared * specific, axis=1)
    return {
        "model": model,
        "target_language": target_language,
        "block": int(block),
        "state": state,
        "row_count": int(shared.shape[0]),
        "shared_specific_dot_mean": float(dot.mean()),
        "shared_specific_dot_abs_max": float(np.abs(dot).max()),
        "shared_norm_mean": float(np.linalg.norm(shared, axis=1).mean()),
        "specific_norm_mean": float(np.linalg.norm(specific, axis=1).mean()),
    }


def _write_feature_array(
    config: dict[str, Any],
    *,
    model: str,
    target_language: str,
    state: str,
    condition: str,
    block: int,
    block_depth_norm: float,
    array: np.ndarray,
    source_artifact: str,
) -> dict[str, object]:
    output_path = _derived_array_path(
        config,
        model=model,
        target_language=target_language,
        state=state,
        condition=condition,
        block=block,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, array.astype(np.float32, copy=False))
    return {
        "model": model,
        "target_language": target_language,
        "block": int(block),
        "block_depth_norm": float(block_depth_norm),
        "state": state,
        "condition": condition,
        "shuffle_index": pd.NA,
        "feature_dim": int(array.shape[1]),
        "n_rows": int(array.shape[0]),
        "source_array_path": str(output_path),
        "source_artifact": source_artifact,
    }


def _merge_on_scope(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    *,
    scope_columns: Sequence[str],
) -> pd.DataFrame:
    if existing.empty:
        return incoming.copy()
    scope_rows = incoming.loc[:, list(scope_columns)].drop_duplicates().copy()
    merged = existing.merge(scope_rows.assign(_drop=True), on=list(scope_columns), how="left")
    merged = merged.loc[merged["_drop"].isna()].drop(columns=["_drop"])
    return pd.concat([merged, incoming], ignore_index=True)


def _write_with_merge(
    path: Path,
    incoming: pd.DataFrame,
    *,
    scope_columns: Sequence[str],
    sort_columns: Sequence[str],
) -> pd.DataFrame:
    if path.exists():
        existing = pd.read_parquet(path).copy()
        merged = _merge_on_scope(existing, incoming, scope_columns=scope_columns)
    else:
        merged = incoming.copy()
    merged = merged.sort_values(list(sort_columns), na_position="first").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, index=False)
    return merged


def _bridge_output_orthogonality_rows(feature_df: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    shared_df = feature_df.loc[feature_df["condition"] == "SHARED"].copy()
    for shared_row in shared_df.itertuples(index=False):
        specific_row = feature_df.loc[
            (feature_df["model"] == shared_row.model)
            & (feature_df["target_language"] == shared_row.target_language)
            & (feature_df["block"] == shared_row.block)
            & (feature_df["condition"] == "SPECIFIC")
            & (feature_df["shuffle_index"].isna())
        ]
        if specific_row.empty:
            continue
        shared = np.load(shared_row.source_array_path).astype(np.float32, copy=False)
        specific = np.load(specific_row.iloc[0]["source_array_path"]).astype(np.float32, copy=False)
        rows.append(
            _orthogonality_row(
                model=str(shared_row.model),
                target_language=str(shared_row.target_language),
                block=int(shared_row.block),
                state="OUTPUT",
                shared=shared,
                specific=specific,
            )
        )
    return rows


def build_output_feature_manifest(config: dict[str, Any]) -> tuple[Path, Path]:
    prior_root = Path(config["paths"]["prior_data_root"]).resolve()
    source_path = prior_root / "processed" / "features" / "feature_manifest.parquet"
    target_path = _output_state_feature_manifest_path(config)
    qc_path = _legacy_output_qc_path(config)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    qc_path.parent.mkdir(parents=True, exist_ok=True)

    source_df = pd.read_parquet(source_path).copy()
    max_block = max(1, int(source_df["layer_index"].astype(int).max()))
    manifest_df = pd.DataFrame(
        {
            "model": source_df["model"].map(MODEL_MAP),
            "target_language": source_df["language"].map(LANGUAGE_MAP),
            "block": source_df["layer_index"].astype(int),
            "block_depth_norm": source_df["layer_index"].astype(int) / max_block,
            "state": "OUTPUT",
            "condition": source_df["condition"].map(CONDITION_MAP),
            "shuffle_index": source_df["shuffle_index"],
            "feature_dim": source_df["feature_dim"].astype(int),
            "n_rows": source_df["n_rows"].astype(int),
            "source_array_path": source_df["filepath"].astype(str),
            "source_artifact": str(source_path),
        }
    ).sort_values(["model", "target_language", "block", "condition", "shuffle_index"], na_position="first").reset_index(drop=True)
    manifest_df.to_parquet(target_path, index=False)

    qc_df = pd.DataFrame(_bridge_output_orthogonality_rows(manifest_df))
    qc_df.to_parquet(qc_path, index=False)
    return target_path, qc_path


def _selected_states(
    config: dict[str, Any],
    *,
    states: Iterable[str] | None,
    all_states: bool,
    state_manifest_df: pd.DataFrame,
) -> tuple[str, ...]:
    if all_states:
        return tuple(sorted(state_manifest_df["state"].astype(str).unique().tolist()))
    if states is not None:
        return tuple(dict.fromkeys(str(state).upper() for state in states))
    configured_states = config.get("features", {}).get("state_names")
    if configured_states:
        configured = [str(state).upper() for state in configured_states]
        return tuple(dict.fromkeys(configured))
    return tuple(sorted(state_manifest_df["state"].astype(str).unique().tolist()))


def build_shared_specific_caches(
    config: dict[str, Any],
    *,
    states: Iterable[str] | None = None,
    all_states: bool = False,
    include_full: bool = False,
    include_raw_output: bool = True,
    include_raw_all_states: bool = False,
) -> dict[str, Path]:
    state_manifest_path = _state_cache_manifest_path(config)
    if not state_manifest_path.exists():
        raise FileNotFoundError(
            f"State cache manifest missing: {state_manifest_path}. Run src.features.extract_internal_states first."
        )

    state_manifest_df = pd.read_parquet(state_manifest_path).copy()
    selected_states = _selected_states(
        config,
        states=states,
        all_states=all_states,
        state_manifest_df=state_manifest_df,
    )
    state_manifest_df = state_manifest_df.loc[state_manifest_df["state"].astype(str).isin(selected_states)].copy()
    if state_manifest_df.empty:
        raise RuntimeError(f"No state cache rows found for states={selected_states}")

    eps = float(config.get("features", {}).get("eps", 1.0e-8))
    manifest_rows: list[dict[str, object]] = []
    orthogonality_rows: list[dict[str, object]] = []

    grouped = state_manifest_df.groupby(["model", "block", "state"], as_index=False)
    for _, group_df in grouped:
        model = str(group_df["model"].iloc[0])
        block = int(group_df["block"].iloc[0])
        state = str(group_df["state"].iloc[0])
        block_depth_norm = float(group_df["block_depth_norm"].iloc[0])

        arrays_by_language = {
            str(row.language): np.load(row.source_array_path).astype(np.float32, copy=False)
            for row in group_df.itertuples(index=False)
        }
        missing_languages = sorted(set(TARGET_LANGUAGES).difference(arrays_by_language))
        if missing_languages:
            raise RuntimeError(
                f"State cache rows missing languages {missing_languages} for model={model}, block={block}, state={state}"
            )

        expected_rows = {array.shape[0] for array in arrays_by_language.values()}
        expected_dims = {array.shape[1] for array in arrays_by_language.values()}
        if len(expected_rows) != 1 or len(expected_dims) != 1:
            raise RuntimeError(
                f"Inconsistent cache shapes for model={model}, block={block}, state={state}: "
                f"{ {language: array.shape for language, array in arrays_by_language.items()} }"
            )

        for target_language in TARGET_LANGUAGES:
            raw = arrays_by_language[target_language]
            other_languages = [language for language in TARGET_LANGUAGES if language != target_language]
            shared = (
                arrays_by_language[other_languages[0]] + arrays_by_language[other_languages[1]]
            ) / 2.0
            specific = _specific_residual(raw, shared, eps=eps)

            if include_raw_all_states or (include_raw_output and state == "OUTPUT"):
                manifest_rows.append(
                    _write_feature_array(
                        config,
                        model=model,
                        target_language=target_language,
                        state=state,
                        condition="RAW",
                        block=block,
                        block_depth_norm=block_depth_norm,
                        array=raw,
                        source_artifact=str(state_manifest_path),
                    )
                )
            manifest_rows.append(
                _write_feature_array(
                    config,
                    model=model,
                    target_language=target_language,
                    state=state,
                    condition="SHARED",
                    block=block,
                    block_depth_norm=block_depth_norm,
                    array=shared.astype(np.float32, copy=False),
                    source_artifact=str(state_manifest_path),
                )
            )
            manifest_rows.append(
                _write_feature_array(
                    config,
                    model=model,
                    target_language=target_language,
                    state=state,
                    condition="SPECIFIC",
                    block=block,
                    block_depth_norm=block_depth_norm,
                    array=specific,
                    source_artifact=str(state_manifest_path),
                )
            )
            if include_full:
                manifest_rows.append(
                    _write_feature_array(
                        config,
                        model=model,
                        target_language=target_language,
                        state=state,
                        condition="FULL",
                        block=block,
                        block_depth_norm=block_depth_norm,
                        array=np.concatenate([shared, specific], axis=1).astype(np.float32, copy=False),
                        source_artifact=str(state_manifest_path),
                    )
                )

            orthogonality_rows.append(
                _orthogonality_row(
                    model=model,
                    target_language=target_language,
                    block=block,
                    state=state,
                    shared=shared.astype(np.float32, copy=False),
                    specific=specific,
                )
            )

    feature_manifest_df = pd.DataFrame(manifest_rows)
    feature_manifest_path = _state_feature_manifest_path(config)
    merged_feature_manifest = _write_with_merge(
        feature_manifest_path,
        feature_manifest_df,
        scope_columns=("model", "target_language", "block", "state", "condition"),
        sort_columns=("model", "target_language", "state", "block", "condition"),
    )

    orthogonality_df = pd.DataFrame(orthogonality_rows)
    qc_path = _shared_specific_qc_path(config)
    merged_qc_df = _write_with_merge(
        qc_path,
        orthogonality_df,
        scope_columns=("model", "target_language", "block", "state"),
        sort_columns=("model", "target_language", "state", "block"),
    )

    output_subset = merged_feature_manifest.loc[merged_feature_manifest["state"] == "OUTPUT"].copy()
    output_feature_manifest_path = _output_state_feature_manifest_path(config)
    output_feature_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_subset.to_parquet(output_feature_manifest_path, index=False)

    legacy_output_qc = merged_qc_df.loc[merged_qc_df["state"] == "OUTPUT"].copy()
    legacy_output_qc_path = _legacy_output_qc_path(config)
    legacy_output_qc_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_output_qc.to_parquet(legacy_output_qc_path, index=False)

    return {
        "manifest_path": feature_manifest_path,
        "qc_path": qc_path,
        "output_manifest_path": output_feature_manifest_path,
        "output_qc_path": legacy_output_qc_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SHARED/SPECIFIC feature manifests for sequel caches.")
    parser.add_argument("--config", default="conf/base.yaml", help="Path to sequel config.")
    parser.add_argument("--states", nargs="+", default=None, choices=list(VALID_STATES))
    parser.add_argument("--all-states", action="store_true", help="Use every state present in the state cache manifest.")
    parser.add_argument("--include-full", action="store_true", help="Also materialize FULL = [SHARED | SPECIFIC].")
    parser.add_argument(
        "--raw-all-states",
        action="store_true",
        help="Materialize RAW rows for every selected state instead of OUTPUT only.",
    )
    parser.add_argument(
        "--bridge-output-only",
        action="store_true",
        help="Use the prior OUTPUT-state feature manifest instead of sequel-native state caches.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.bridge_output_only or not _state_cache_manifest_path(config).exists():
        manifest_path, qc_path = build_output_feature_manifest(config)
        print(f"manifest_path: {manifest_path}")
        print(f"qc_path: {qc_path}")
        return

    outputs = build_shared_specific_caches(
        config,
        states=None if args.states is None else tuple(args.states),
        all_states=args.all_states,
        include_full=args.include_full,
        include_raw_all_states=args.raw_all_states,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
