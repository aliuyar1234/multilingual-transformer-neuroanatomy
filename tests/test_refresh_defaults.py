from __future__ import annotations

from pathlib import Path
import importlib

import pandas as pd

from tests.helpers import make_temp_test_dir


def test_refresh_defaults_and_manifest_encode_k5_representative_grid() -> None:
    refresh_module = importlib.import_module("src.encoding.run_refresh_baseline")
    manifest_module = importlib.import_module("src.encoding.build_refresh_job_manifest")

    config = {
        "paths": {"local_outputs_root": ""},
        "bridge": {
            "prior_refresh": {
                "dense_refresh_mismatch_shuffles_default": 20,
                "missing_layer_mismatch_shuffles_default": 1,
                "representative_mismatch_shuffles_default": 5,
                "representative_roi_families": ["SEMANTIC", "AUDITORY"],
            }
        },
    }
    assert refresh_module._default_mismatch_shuffles(config, mismatch_only=True, layer_indices=(10, 11)) == 5
    assert refresh_module._default_mismatch_shuffles(config, mismatch_only=False, layer_indices=(0, 1, 2)) == 1
    assert refresh_module._default_mismatch_shuffles(config, mismatch_only=False, layer_indices=(4, 6, 8)) == 20

    base_path = make_temp_test_dir("refresh_defaults")
    outputs_root = base_path / "outputs"
    bridge_root = outputs_root / "bridge" / "prior_release"
    bridge_root.mkdir(parents=True, exist_ok=True)
    coverage_path = bridge_root / "prior_release_coverage.parquet"
    pd.DataFrame({"language": ["EN", "FR", "ZH"]}).to_parquet(coverage_path, index=False)

    try:
        manifest_outputs = manifest_module.build_refresh_job_manifest(
            {
                "paths": {"local_outputs_root": outputs_root.as_posix()},
                "bridge": config["bridge"],
            }
        )
        manifest_df = pd.read_parquet(manifest_outputs["manifest"]).copy()

        representative_rows = manifest_df.loc[
            manifest_df["refresh_mode"].astype(str) == "mismatch_only_representative_layers"
        ].copy()
        assert len(representative_rows) == 12
        assert set(representative_rows["roi_family"].astype(str)) == {"SEMANTIC", "AUDITORY"}
        assert set(representative_rows["mismatch_shuffles"].astype(int)) == {5}
        assert set(representative_rows["expected_representative_rows"].astype(int)) == {1}

        selection_rows = manifest_df.loc[
            manifest_df["refresh_mode"].astype(str) == "representative_output_layer"
        ].copy()
        assert len(selection_rows) == 3
        assert set(selection_rows["expected_representative_rows"].astype(int)) == {2}

        dense_rows = manifest_df.loc[
            manifest_df["job_id"].astype(str).str.endswith("_missing_layers_refresh")
        ].copy()
        assert set(dense_rows["mismatch_shuffles"].astype(int)) == {1}
    finally:
        for path in sorted(base_path.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()
