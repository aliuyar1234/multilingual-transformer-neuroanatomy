from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd

from tests.helpers import make_temp_test_dir


def test_write_draft_uses_current_tables_and_writes_number_sources(monkeypatch) -> None:
    write_draft_module = importlib.import_module("src.manuscript.write_draft")
    base_path = make_temp_test_dir("write_draft")
    outputs_root = base_path / "outputs"
    tables_root = outputs_root / "tables"
    group_root = outputs_root / "group_results"
    tables_root.mkdir(parents=True, exist_ok=True)
    group_root.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "model": "xlmr",
                "language": "EN",
                "hypothesis": "H1_semantic_ffn_preference",
                "effect_mean": 0.4,
                "ci_low": 0.2,
                "ci_high": 0.6,
                "p_raw": 0.01,
                "p_holm": 0.02,
                "n_subjects": 4,
            }
        ]
    ).to_csv(tables_root / "table03_primary_confirmatory_stats.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "xlmr",
                "language": "EN",
                "roi_family": "SEMANTIC",
                "representative_block": 1,
                "representative_state": "FFN",
                "roi": "AG",
                "shared_mean_z": 0.6,
                "specific_mean_z": 0.2,
                "delta_z_mean": 0.4,
            }
        ]
    ).to_csv(tables_root / "table04_representative_roi_summaries.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "xlmr",
                "language": "EN",
                "roi_family": "SEMANTIC",
                "condition": "SHARED",
                "class": "CONTENT",
                "positive_mass": 0.7,
                "content_bias": 0.5,
            }
        ]
    ).to_csv(tables_root / "table05_token_class_attribution.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "xlmr",
                "language": "EN",
                "roi_family": "SEMANTIC",
                "k": 1,
                "effect_topk": 0.4,
                "effect_random": 0.2,
                "diff": 0.2,
                "ci_low": 0.1,
                "ci_high": 0.3,
            }
        ]
    ).to_csv(tables_root / "table06_deletion_validation.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "xlmr",
                "language": "EN",
                "hypothesis": "H1_semantic_ffn_preference",
                "effect_mean": 0.4,
                "ci_low": 0.2,
                "ci_high": 0.6,
                "p_raw": 0.01,
                "p_holm": 0.02,
                "n_subjects": 4,
            }
        ]
    ).to_parquet(group_root / "primary_effect_tables.parquet", index=False)

    config = {"paths": {"local_outputs_root": outputs_root.as_posix()}}
    draft_target = base_path / "manuscript" / "paper" / "main.md"
    monkeypatch.setattr(write_draft_module, "_paper_path", lambda: draft_target)

    try:
        outputs = write_draft_module.write_draft(config)
        draft_path = outputs["draft"]
        number_sources_path = outputs["number_sources"]

        assert draft_path.exists()
        assert number_sources_path.exists()

        draft_text = draft_path.read_text(encoding="utf-8")
        number_sources_text = number_sources_path.read_text(encoding="utf-8")

        assert "Results Snapshot" in draft_text
        assert "H1 is Holm-significant in 1 rows" in draft_text
        assert "Current attribution summary" in draft_text
        assert "deletion table contains 1 rows" in draft_text
        assert "table03_primary_confirmatory_stats.csv" in number_sources_text
        assert "table06_deletion_validation.csv" in number_sources_text
    finally:
        for path in sorted(base_path.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()
