from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd

from tests.helpers import make_temp_test_dir


def _load_module():
    return importlib.import_module("src.encoding.select_output_representative_layers")


def _build_state_sweep_group_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    family_blocks = {
        "SEMANTIC": {0: (0.35, 0.20), 1: (0.85, 0.15)},
        "AUDITORY": {0: (0.80, 0.20), 1: (0.45, 0.25)},
        "CONTROL": {0: (0.95, 0.10), 1: (0.20, 0.10)},
    }
    for model in ("xlmr", "nllb"):
        for language in ("EN", "FR", "ZH"):
            for roi_family, block_map in family_blocks.items():
                for block, (shared_mean, specific_mean) in block_map.items():
                    for condition, mean_z in (("SHARED", shared_mean), ("SPECIFIC", specific_mean)):
                        rows.append(
                            {
                                "model": model,
                                "language": language,
                                "roi_family": roi_family,
                                "block": block,
                                "block_depth_norm": block / 12.0,
                                "state": "OUTPUT",
                                "condition": condition,
                                "roi": f"{roi_family}_ROI",
                                "mean_z": mean_z,
                            }
                        )
    return pd.DataFrame(rows)


def test_select_output_representative_layers_prefers_fixed_semantic_auditory_grid() -> None:
    module = _load_module()
    base_path = make_temp_test_dir("select_output_representative_layers")
    outputs_root = base_path / "outputs"
    group_root = outputs_root / "group_results"
    group_root.mkdir(parents=True, exist_ok=True)

    state_sweep_path = group_root / "state_sweep_group_results.parquet"
    _build_state_sweep_group_rows().to_parquet(state_sweep_path, index=False)
    config = {
        "paths": {"local_outputs_root": outputs_root.as_posix()},
        "dataset": {"languages": ["EN", "FR", "ZH"]},
        "models": {"xlmr": {}, "nllb": {}},
        "bridge": {"prior_refresh": {"representative_roi_families": ["SEMANTIC", "AUDITORY"]}},
    }

    try:
        outputs = module.select_output_representative_layers(config, group_paths=[state_sweep_path])
        representative_df = pd.read_parquet(outputs["representative_layers"]).copy()

        assert len(representative_df) == 12
        assert set(representative_df["roi_family"].astype(str)) == {"SEMANTIC", "AUDITORY"}
        assert set(representative_df["state"].astype(str)) == {"OUTPUT"}
        assert set(representative_df.loc[representative_df["roi_family"].astype(str) == "SEMANTIC", "block"].astype(int)) == {1}
        assert set(representative_df.loc[representative_df["roi_family"].astype(str) == "AUDITORY", "block"].astype(int)) == {0}
    finally:
        for path in sorted(base_path.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()
