from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.stats.generate_refresh_tables import generate_refresh_tables
from tests.helpers import make_temp_test_dir


def test_generate_refresh_tables_builds_table01_and_table02() -> None:
    base_path = make_temp_test_dir("generate_refresh_tables")
    outputs_root = base_path / "outputs"
    manifests_root = base_path / "data" / "manifests"
    cache_root = outputs_root / "caches" / "features"
    qc_root = outputs_root / "qc"
    cache_root.mkdir(parents=True, exist_ok=True)
    qc_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {"language": "EN", "subject_id": "sub-EN001", "n_canonical_runs": 9, "included": True, "exclusion_reason": ""},
            {"language": "EN", "subject_id": "sub-EN002", "n_canonical_runs": 9, "included": False, "exclusion_reason": "motion"},
            {"language": "FR", "subject_id": "sub-FR001", "n_canonical_runs": 8, "included": True, "exclusion_reason": ""},
        ]
    ).to_parquet(manifests_root / "sample_manifest.parquet", index=False)

    pd.DataFrame(
        [
            {"subject_id": "sub-EN001", "language": "EN", "canonical_run": 1, "n_scans": 300, "n_rois": 18},
            {"subject_id": "sub-EN001", "language": "EN", "canonical_run": 2, "n_scans": 330, "n_rois": 18},
            {"subject_id": "sub-FR001", "language": "FR", "canonical_run": 1, "n_scans": 280, "n_rois": 18},
        ]
    ).to_parquet(manifests_root / "roi_manifest.parquet", index=False)

    cache_array = np.zeros((2, 4), dtype=np.float32)
    cache_array_path = cache_root / "xlmr_EN_OUTPUT_block00.npy"
    np.save(cache_array_path, cache_array)
    pd.DataFrame(
        [
            {
                "model": "xlmr",
                "language": "EN",
                "block": 0,
                "state": "OUTPUT",
                "source_array_path": cache_array_path.as_posix(),
            },
            {
                "model": "nllb",
                "language": "FR",
                "block": 0,
                "state": "ATTN",
                "source_array_path": cache_array_path.as_posix(),
            },
        ]
    ).to_parquet(cache_root / "state_cache_manifest.parquet", index=False)

    pd.DataFrame(
        [
            {"model": "xlmr", "language": "EN", "block": 0, "state": "OUTPUT", "max_abs_diff": 1.0e-7},
            {"model": "nllb", "language": "FR", "block": 0, "state": "OUTPUT", "max_abs_diff": 2.0e-4},
        ]
    ).to_parquet(qc_root / "state_extraction_equivalence.parquet", index=False)

    config = {
        "paths": {
            "local_outputs_root": outputs_root.as_posix(),
            "local_manifests_root": manifests_root.as_posix(),
        },
        "dataset": {"tr_seconds": 2.0},
        "models": {
            "xlmr": {"hidden_size": 768, "state_names": ["INPUT", "ATTN", "POST_ATTN", "FFN", "OUTPUT"]},
            "nllb": {"hidden_size": 1024, "state_names": ["INPUT", "ATTN", "POST_ATTN", "FFN", "OUTPUT"]},
        },
    }

    try:
        outputs = generate_refresh_tables(config)
        assert outputs["table01"].exists()
        assert outputs["table02"].exists()

        table01 = pd.read_csv(outputs["table01"])
        table02 = pd.read_csv(outputs["table02"])

        assert {"language", "n_subjects", "n_runs", "mean_duration", "TR", "any_exclusions"}.issubset(table01.columns)
        assert {"model", "hidden_size", "n_blocks", "states_extracted", "output_equivalence", "cache_size_summary"}.issubset(table02.columns)
        assert set(table01["language"].astype(str)) == {"EN", "FR"}
        assert "motion" in set(table01["any_exclusions"].astype(str))
        assert "PASS" in set(table02["output_equivalence"].astype(str))
        assert "FAIL" in set(table02["output_equivalence"].astype(str))
    finally:
        for path in sorted(base_path.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()
