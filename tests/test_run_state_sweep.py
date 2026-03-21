from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tests.helpers import make_temp_test_dir


class _FakeOperator:
    def transform_feature_matrix(self, feature_matrix: np.ndarray) -> dict[int, np.ndarray]:
        feature_matrix = np.asarray(feature_matrix, dtype=np.float32)
        return {
            1: feature_matrix[:2, :],
            2: feature_matrix[2:4, :],
        }


def _load_module():
    return importlib.import_module("src.encoding.run_state_sweep")


def _feature_manifest(base_path: Path) -> Path:
    arrays_root = base_path / "outputs" / "caches" / "derived_features"
    arrays_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for language_index, language in enumerate(("EN", "FR"), start=1):
        for state_index, state in enumerate(("ATTN", "OUTPUT"), start=1):
            for condition_index, condition in enumerate(("SHARED", "SPECIFIC"), start=1):
                array = np.full(
                    (4, 2),
                    fill_value=float(language_index + state_index + condition_index),
                    dtype=np.float32,
                )
                path = arrays_root / "xlmr" / language / state / condition / "block_00.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, array)
                rows.append(
                    {
                        "model": "xlmr",
                        "target_language": language,
                        "block": 0,
                        "block_depth_norm": 0.0,
                        "state": state,
                        "condition": condition,
                        "feature_dim": int(array.shape[1]),
                        "n_rows": int(array.shape[0]),
                        "source_array_path": path.as_posix(),
                        "source_artifact": "synthetic_state_feature_manifest",
                    }
                )

    manifest_path = arrays_root / "state_feature_manifest.parquet"
    pd.DataFrame(rows).to_parquet(manifest_path, index=False)
    return manifest_path


def _prepared_language(module, language: str):
    target_metadata = pd.DataFrame(
        [
            {"subject_id": "sub-01", "roi": "AG", "roi_family": "SEMANTIC"},
            {"subject_id": "sub-01", "roi": "pSTG", "roi_family": "AUDITORY"},
            {"subject_id": "sub-02", "roi": "AG", "roi_family": "SEMANTIC"},
            {"subject_id": "sub-02", "roi": "pSTG", "roi_family": "AUDITORY"},
        ]
    )
    return module.PreparedLanguageSweep(
        language=language,
        run_order=(1, 2),
        operator=_FakeOperator(),
        run_ids=np.asarray([1, 1, 2, 2], dtype=np.int64),
        target_matrix=np.asarray(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.2, 0.1, 0.4, 0.3],
                [0.3, 0.4, 0.1, 0.2],
                [0.4, 0.3, 0.2, 0.1],
            ],
            dtype=np.float32,
        ),
        nuisance_matrix=np.ones((4, 1), dtype=np.float32),
        target_metadata=target_metadata,
        n_timepoints_total=4,
    )


def test_run_state_sweep_writes_subject_and_group_outputs(monkeypatch) -> None:
    module = _load_module()
    base_path = make_temp_test_dir("state_sweep")
    manifest_path = _feature_manifest(base_path)
    outputs_root = base_path / "outputs"

    prepared_by_language = {
        language: _prepared_language(module, language)
        for language in ("EN", "FR")
    }

    def fake_prepare_language_sweep(config, *, language: str, max_subjects=None):
        _ = (config, max_subjects)
        return prepared_by_language[language]

    def fake_run_nested_cv_multi_target(**kwargs):
        n_targets = int(kwargs["Y"].shape[1])
        mean_signal = float(np.asarray(kwargs["X"], dtype=np.float32).mean())
        z_mean = np.asarray([mean_signal + (index * 0.01) for index in range(n_targets)], dtype=np.float64)
        return SimpleNamespace(
            outer_runs=(1, 2),
            r_mean=(z_mean / 10.0).astype(np.float64, copy=False),
            z_mean=z_mean,
            r2_concat=np.full(n_targets, 0.25 + (mean_signal / 100.0), dtype=np.float64),
            alpha_best_mode=np.full(n_targets, 100.0, dtype=np.float64),
            alpha_best_mean_log10=np.full(n_targets, 2.0, dtype=np.float64),
            n_pcs_mean=np.full(n_targets, 3.0, dtype=np.float64),
        )

    monkeypatch.setattr(module, "_prepare_language_sweep", fake_prepare_language_sweep)
    monkeypatch.setattr(module, "run_nested_cv_multi_target", fake_run_nested_cv_multi_target)

    config = {
        "paths": {
            "local_outputs_root": outputs_root.as_posix(),
            "local_manifests_root": (base_path / "data" / "manifests").as_posix(),
        },
        "project": {
            "seed": 1337,
        },
        "features": {
            "fine_hz": 10.0,
        },
        "dataset": {
            "tr_seconds": 2.0,
        },
        "encoding": {
            "max_pcs": 32,
            "variance_threshold": 0.95,
            "target_chunk_size": 8,
        },
    }

    assert manifest_path.exists()
    outputs = module.run_state_sweep(
        config,
        models=("xlmr",),
        languages=("EN", "FR"),
        states=("ATTN", "OUTPUT"),
        conditions=("SHARED", "SPECIFIC"),
    )

    subject_df = pd.read_parquet(outputs["subject_results"]).copy()
    group_df = pd.read_parquet(outputs["group_results"]).copy()
    provenance_text = Path(outputs["provenance"]).read_text(encoding="utf-8")

    assert len(subject_df) == 32
    assert len(group_df) == 16
    assert set(subject_df["language"].unique()) == {"EN", "FR"}
    assert set(subject_df["state"].unique()) == {"ATTN", "OUTPUT"}
    assert set(subject_df["condition"].unique()) == {"SHARED", "SPECIFIC"}
    assert set(subject_df["roi_family"].unique()) == {"SEMANTIC", "AUDITORY"}
    assert (subject_df["n_outer_folds"] == 2).all()
    assert (group_df["n_subjects"] == 2).all()
    assert "states: `ATTN, OUTPUT`" in provenance_text
    assert "conditions: `SHARED, SPECIFIC`" in provenance_text


def test_align_run_frame_columns_pads_missing_highpass_terms() -> None:
    module = _load_module()
    frames = {
        1: pd.DataFrame(
            {
                "intercept": [1.0, 1.0],
                "linear_trend": [-1.0, 1.0],
                "highpass_00": [0.2, -0.2],
            }
        ),
        2: pd.DataFrame(
            {
                "intercept": [1.0, 1.0],
                "linear_trend": [-1.0, 1.0],
                "highpass_00": [0.1, -0.1],
                "highpass_01": [0.3, -0.3],
            }
        ),
    }

    aligned = module._align_run_frame_columns(frames)

    assert list(aligned[1].columns) == ["highpass_00", "highpass_01", "intercept", "linear_trend"]
    assert list(aligned[2].columns) == ["highpass_00", "highpass_01", "intercept", "linear_trend"]
    assert aligned[1]["highpass_01"].tolist() == [0.0, 0.0]
    assert aligned[2]["highpass_01"].tolist() == [0.3, -0.3]
