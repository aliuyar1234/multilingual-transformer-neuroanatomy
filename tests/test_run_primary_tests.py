from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd

from tests.helpers import make_temp_test_dir


ATTN_STATES = ("ATTN", "POST_ATTN")
FFN_STATES = ("FFN", "OUTPUT")


def _load_module():
    return importlib.import_module("src.stats.run_primary_tests")


def _subject_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    semantic_rois = ("AG", "IFGtri")
    auditory_rois = ("pSTG", "Heschl")
    blocks = (0, 1)
    subjects = ("sub-01", "sub-02", "sub-03", "sub-04")

    for model in ("xlmr", "nllb"):
        for language in ("EN", "FR", "ZH"):
            for subject_index, subject_id in enumerate(subjects):
                subject_boost = 0.05 * (subject_index + 1)
                for block in blocks:
                    block_depth_norm = float(block)
                    for state in (*ATTN_STATES, *FFN_STATES):
                        for roi in semantic_rois:
                            shared_delta = (
                                0.85 + subject_boost if state in FFN_STATES else 0.20 + (subject_boost / 2.0)
                            )
                            for condition, z_mean in (("SHARED", shared_delta), ("SPECIFIC", 0.0)):
                                rows.append(
                                    {
                                        "subject_id": subject_id,
                                        "language": language,
                                        "model": model,
                                        "block": block,
                                        "block_depth_norm": block_depth_norm,
                                        "state": state,
                                        "condition": condition,
                                        "roi": roi,
                                        "roi_family": "SEMANTIC",
                                        "z_mean": float(z_mean),
                                        "r_mean": float(np.tanh(z_mean / 4.0)),
                                        "r2_concat": float(z_mean / 10.0),
                                        "alpha_best_mode": 100.0,
                                        "alpha_best_mean_log10": 2.0,
                                        "n_pcs_mean": 5.0,
                                        "n_outer_folds": 9,
                                        "n_timepoints_total": 90,
                                        "seed": 1337,
                                    }
                                )
                        for roi in auditory_rois:
                            shared_delta = (
                                0.80 + subject_boost if state in ATTN_STATES else 0.15 + (subject_boost / 2.0)
                            )
                            for condition, z_mean in (("SHARED", shared_delta), ("SPECIFIC", 0.0)):
                                rows.append(
                                    {
                                        "subject_id": subject_id,
                                        "language": language,
                                        "model": model,
                                        "block": block,
                                        "block_depth_norm": block_depth_norm,
                                        "state": state,
                                        "condition": condition,
                                        "roi": roi,
                                        "roi_family": "AUDITORY",
                                        "z_mean": float(z_mean),
                                        "r_mean": float(np.tanh(z_mean / 4.0)),
                                        "r2_concat": float(z_mean / 10.0),
                                        "alpha_best_mode": 100.0,
                                        "alpha_best_mean_log10": 2.0,
                                        "n_pcs_mean": 5.0,
                                        "n_outer_folds": 9,
                                        "n_timepoints_total": 90,
                                        "seed": 1337,
                                    }
                                )
    return rows


def test_run_primary_tests_writes_expected_12_confirmatory_rows() -> None:
    module = _load_module()
    base_path = make_temp_test_dir("primary_stats")
    outputs_root = base_path / "outputs"
    subject_results_path = outputs_root / "subject_results" / "state_sweep_subject_results.parquet"
    subject_results_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(_subject_rows()).to_parquet(subject_results_path, index=False)
    config = {
        "paths": {
            "local_outputs_root": outputs_root.as_posix(),
        },
        "project": {
            "seed": 1337,
        },
        "stats": {
            "permutations": 999,
            "bootstraps": 500,
        },
    }

    outputs = module.run_primary_tests(config)

    summary_df = pd.read_parquet(outputs["primary_effect_tables"]).copy()
    subject_effects_df = pd.read_parquet(outputs["primary_subject_effects"]).copy()
    permutation_df = pd.read_parquet(outputs["primary_permutation_summary"]).copy()
    bootstrap_df = pd.read_parquet(outputs["primary_bootstrap_summary"]).copy()
    table03_df = pd.read_csv(outputs["table03"]).copy()

    assert len(summary_df) == 12
    assert len(table03_df) == 12
    assert set(summary_df["model"].unique()) == {"xlmr", "nllb"}
    assert set(summary_df["language"].unique()) == {"EN", "FR", "ZH"}
    assert set(summary_df["hypothesis"].unique()) == {
        "H1_semantic_ffn_preference",
        "H2_auditory_attention_preference",
    }
    assert summary_df["p_holm"].notna().all()
    assert (summary_df["effect_mean"] > 0.0).all()
    assert (summary_df["n_subjects"] == 4).all()

    assert len(subject_effects_df) == 48
    assert len(permutation_df) == 12
    assert len(bootstrap_df) == 12
    assert set(table03_df.columns) == {
        "model",
        "language",
        "hypothesis",
        "effect_mean",
        "ci_low",
        "ci_high",
        "p_raw",
        "p_holm",
        "n_subjects",
        "permutation_count",
        "bootstrap_count",
    }


def test_run_primary_tests_fails_when_locked_language_coverage_is_incomplete() -> None:
    module = _load_module()
    base_path = make_temp_test_dir("primary_stats_missing_language")
    outputs_root = base_path / "outputs"
    subject_results_path = outputs_root / "subject_results" / "state_sweep_subject_results.parquet"
    subject_results_path.parent.mkdir(parents=True, exist_ok=True)

    incomplete_rows = [row for row in _subject_rows() if row["language"] != "ZH"]
    pd.DataFrame(incomplete_rows).to_parquet(subject_results_path, index=False)
    config = {
        "paths": {
            "local_outputs_root": outputs_root.as_posix(),
        },
        "project": {
            "seed": 1337,
        },
        "stats": {
            "permutations": 999,
            "bootstraps": 500,
        },
    }

    try:
        module.run_primary_tests(config)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected run_primary_tests to fail when one locked language is missing.")

    assert "language coverage mismatch" in message
