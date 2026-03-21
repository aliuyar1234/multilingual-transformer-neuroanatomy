from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.helpers import make_temp_test_dir


def _build_expected_shared_specific(raw: np.ndarray, other_a: np.ndarray, other_b: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shared = (other_a + other_b) / 2.0
    residual = raw - shared
    denom = np.sum(shared * shared, axis=1, keepdims=True) + eps
    projection = np.sum(shared * residual, axis=1, keepdims=True) / denom
    specific = residual - projection * shared
    full = np.concatenate([shared, specific], axis=1)
    return shared.astype(np.float32, copy=False), specific.astype(np.float32, copy=False), full.astype(np.float32, copy=False)


def _write_state_cache_manifest(base_path: Path) -> tuple[Path, dict[tuple[str, str, int, str], np.ndarray]]:
    outputs_root = base_path / "outputs"
    arrays_root = outputs_root / "caches" / "features"
    arrays_root.mkdir(parents=True, exist_ok=True)
    (base_path / "data" / "manifests").mkdir(parents=True, exist_ok=True)
    (outputs_root / "qc").mkdir(parents=True, exist_ok=True)

    templates = {
        "OUTPUT": {
            "EN": np.asarray([[2.0, 0.0, 1.0], [1.0, 2.0, 0.0]], dtype=np.float32),
            "FR": np.asarray([[0.0, 2.0, 1.0], [2.0, 0.0, 1.0]], dtype=np.float32),
            "ZH": np.asarray([[1.0, 1.0, 2.0], [0.0, 1.0, 1.0]], dtype=np.float32),
        },
        "FFN": {
            "EN": np.asarray([[3.0, 1.0, 0.0], [1.0, 0.0, 2.0]], dtype=np.float32),
            "FR": np.asarray([[1.0, 3.0, 0.0], [0.0, 2.0, 1.0]], dtype=np.float32),
            "ZH": np.asarray([[2.0, 1.0, 1.0], [1.0, 1.0, 2.0]], dtype=np.float32),
        },
    }

    manifest_rows: list[dict[str, object]] = []
    arrays_by_key: dict[tuple[str, str, int, str], np.ndarray] = {}
    for state, state_templates in templates.items():
        for language, array in state_templates.items():
            state_dir = arrays_root / "xlmr" / language / state
            state_dir.mkdir(parents=True, exist_ok=True)
            source_path = state_dir / "block_00.npy"
            np.save(source_path, array)
            arrays_by_key[("xlmr", language, 0, state)] = array
            manifest_rows.append(
                {
                    "model": "xlmr",
                    "language": language,
                    "block": 0,
                    "block_depth_norm": 0.0,
                    "state": state,
                    "n_rows": int(array.shape[0]),
                    "hidden_size": int(array.shape[1]),
                    "dtype": "float32",
                    "source_array_path": source_path.as_posix(),
                    "source_artifact": "synthetic_state_cache_manifest",
                }
            )

    manifest_path = arrays_root / "state_cache_manifest.parquet"
    pd.DataFrame(manifest_rows).to_parquet(manifest_path, index=False)
    pd.DataFrame(manifest_rows).to_parquet(base_path / "data" / "manifests" / "state_cache_manifest.parquet", index=False)
    return manifest_path, arrays_by_key


def _load_builder_module():
    return importlib.import_module("src.features.build_shared_specific")


def _call_generalized_builder(module, config):
    builder = getattr(module, "build_shared_specific_caches", None)
    if builder is None:
        pytest.skip("build_shared_specific_caches is not implemented yet")
    return builder(config)


def _resolve_output_paths(result, *, manifest_candidates: list[Path], qc_candidates: list[Path]) -> tuple[Path, Path]:
    if isinstance(result, tuple) and len(result) >= 2:
        return Path(result[0]), Path(result[1])
    if isinstance(result, dict):
        manifest = result.get("manifest_path") or result.get("feature_manifest_path")
        qc = result.get("qc_path") or result.get("orthogonality_path")
        if manifest is not None and qc is not None:
            return Path(manifest), Path(qc)
    manifest_path = next((candidate for candidate in manifest_candidates if candidate.exists()), None)
    qc_path = next((candidate for candidate in qc_candidates if candidate.exists()), None)
    if manifest_path is None or qc_path is None:
        raise AssertionError(
            "None of the expected outputs exist: "
            f"manifest={manifest_candidates}, qc={qc_candidates}"
        )
    return manifest_path, qc_path


def test_generalized_shared_specific_builder_emits_expected_manifest_and_qc() -> None:
    base_path = make_temp_test_dir("shared_specific")
    base_path.mkdir(parents=True, exist_ok=True)
    _, arrays_by_key = _write_state_cache_manifest(base_path)
    module = _load_builder_module()
    config = {
        "paths": {
            "local_outputs_root": (base_path / "outputs").as_posix(),
            "local_manifests_root": (base_path / "data" / "manifests").as_posix(),
        },
        "features": {
            "eps": 1.0e-8,
        },
    }

    result = _call_generalized_builder(module, config)
    expected_manifest_candidates = [
        base_path / "outputs" / "caches" / "derived_features" / "state_feature_manifest.parquet",
        base_path / "outputs" / "caches" / "derived_features" / "output_state_feature_manifest.parquet",
    ]
    expected_qc_candidates = [
        base_path / "outputs" / "qc" / "shared_specific_orthogonality.parquet",
        base_path / "outputs" / "qc" / "state_shared_specific_orthogonality.parquet",
        base_path / "outputs" / "qc" / "output_shared_specific_orthogonality.parquet",
    ]

    if isinstance(result, tuple) and len(result) == 2:
        manifest_path = Path(result[0])
        qc_path = Path(result[1])
    else:
        manifest_path, qc_path = _resolve_output_paths(
            result,
            manifest_candidates=expected_manifest_candidates,
            qc_candidates=expected_qc_candidates,
        )

    if not manifest_path.exists() or not qc_path.exists():
        manifest_path, qc_path = _resolve_output_paths(
            result,
            manifest_candidates=expected_manifest_candidates,
            qc_candidates=expected_qc_candidates,
        )

    manifest_df = pd.read_parquet(manifest_path).copy()
    qc_df = pd.read_parquet(qc_path).copy()

    assert set(manifest_df["model"].unique()) == {"xlmr"}
    assert set(manifest_df["target_language"].unique()) == {"EN", "FR", "ZH"}
    assert set(manifest_df["state"].unique()) == {"OUTPUT", "FFN"}
    assert {"RAW", "SHARED", "SPECIFIC"}.issubset(set(manifest_df["condition"].unique()))

    base_manifest = manifest_df.loc[manifest_df["condition"].isin(["RAW", "SHARED", "SPECIFIC"])].copy()
    assert len(base_manifest) == 15
    assert set(
        base_manifest.loc[base_manifest["condition"] == "RAW", "state"].unique()
    ) == {"OUTPUT"}

    eps = 1.0e-8
    for state in ("OUTPUT", "FFN"):
        for target_language in ("EN", "FR", "ZH"):
            row_subset = manifest_df.loc[
                (manifest_df["state"] == state)
                & (manifest_df["target_language"] == target_language)
            ].copy()
            assert not row_subset.empty

            shared_row = row_subset.loc[row_subset["condition"] == "SHARED"].iloc[0]
            specific_row = row_subset.loc[row_subset["condition"] == "SPECIFIC"].iloc[0]

            raw_rows = row_subset.loc[row_subset["condition"] == "RAW"]
            raw = (
                np.load(raw_rows.iloc[0]["source_array_path"]).astype(np.float32, copy=False)
                if not raw_rows.empty
                else arrays_by_key[("xlmr", target_language, 0, state)]
            )
            shared = np.load(shared_row["source_array_path"]).astype(np.float32, copy=False)
            specific = np.load(specific_row["source_array_path"]).astype(np.float32, copy=False)
            full_rows = row_subset.loc[row_subset["condition"] == "FULL"]
            if not full_rows.empty:
                full = np.load(full_rows.iloc[0]["source_array_path"]).astype(np.float32, copy=False)
            else:
                full = None

            other_languages = [language for language in ("EN", "FR", "ZH") if language != target_language]
            other_a = arrays_by_key[("xlmr", other_languages[0], 0, state)]
            other_b = arrays_by_key[("xlmr", other_languages[1], 0, state)]
            expected_shared, expected_specific, expected_full = _build_expected_shared_specific(
                raw=arrays_by_key[("xlmr", target_language, 0, state)],
                other_a=other_a,
                other_b=other_b,
                eps=eps,
            )

            np.testing.assert_allclose(raw, arrays_by_key[("xlmr", target_language, 0, state)], atol=1e-6, rtol=1e-6)
            np.testing.assert_allclose(shared, expected_shared, atol=1e-6, rtol=1e-6)
            np.testing.assert_allclose(specific, expected_specific, atol=1e-6, rtol=1e-6)
            np.testing.assert_allclose(np.sum(shared * specific, axis=1), np.zeros(shared.shape[0]), atol=1e-6, rtol=1e-6)
            if full is not None:
                np.testing.assert_allclose(full, expected_full, atol=1e-6, rtol=1e-6)

    assert set(qc_df["target_language"].unique()) == {"EN", "FR", "ZH"}
    assert set(qc_df["state"].unique()) == {"OUTPUT", "FFN"}
    assert len(qc_df) == 6
    assert qc_df["shared_specific_dot_abs_max"].max() < 1.0e-6
    assert qc_df["shared_specific_dot_mean"].abs().max() < 1.0e-6
