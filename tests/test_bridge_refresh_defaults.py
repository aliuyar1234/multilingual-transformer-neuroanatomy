from __future__ import annotations

import importlib

import pandas as pd


def test_build_refresh_job_manifest_uses_missing_layer_and_representative_defaults() -> None:
    module = importlib.import_module("src.encoding.build_refresh_job_manifest")
    coverage = pd.DataFrame([{"language": "EN"}, {"language": "FR"}, {"language": "ZH"}])
    config = {
        "bridge": {
            "prior_refresh": {
                "missing_layer_mismatch_shuffles_default": 1,
                "representative_mismatch_shuffles_default": 5,
            }
        }
    }

    jobs = module._build_refresh_jobs(config, coverage)

    missing_layers = jobs.loc[
        jobs["job_id"].astype(str) == "nllb_en_missing_layers_refresh",
        "mismatch_shuffles",
    ].iloc[0]
    representative = jobs.loc[
        jobs["job_id"].astype(str) == "xlmr_en_semantic_representative_mismatch_refresh",
        "mismatch_shuffles",
    ].iloc[0]

    assert int(missing_layers) == 1
    assert int(representative) == 5


def test_run_refresh_baseline_default_mismatch_policy_switches_by_job_type() -> None:
    module = importlib.import_module("src.encoding.run_refresh_baseline")
    config = {
        "bridge": {
            "prior_refresh": {
                "dense_refresh_mismatch_shuffles_default": 20,
                "missing_layer_mismatch_shuffles_default": 1,
                "representative_mismatch_shuffles_default": 5,
            }
        }
    }

    assert module._default_mismatch_shuffles(config, mismatch_only=True, layer_indices=None) == 5
    assert module._default_mismatch_shuffles(config, mismatch_only=False, layer_indices=(0, 1, 2, 3, 5)) == 1
    assert module._default_mismatch_shuffles(config, mismatch_only=False, layer_indices=None) == 20
