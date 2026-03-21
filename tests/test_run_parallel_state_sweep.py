from __future__ import annotations

import importlib
import json
from pathlib import Path

import pandas as pd

from tests.helpers import make_temp_test_dir


def test_run_parallel_state_sweep_runs_language_shards_and_merges(monkeypatch) -> None:
    module = importlib.import_module("src.encoding.run_parallel_state_sweep")
    base_path = make_temp_test_dir("run_parallel_state_sweep")
    outputs_root = base_path / "outputs"

    calls: list[tuple[str, str]] = []

    def fake_run_shard(*, config_path, shard, models, states, conditions, max_subjects):
        _ = (config_path, models, states, conditions, max_subjects)
        calls.append((shard.language, shard.output_tag))
        return {"language": shard.language, "output_tag": shard.output_tag}

    def fake_merge(config, *, manifest_path):
        _ = config
        assert Path(manifest_path).exists()
        return {"subject_results": outputs_root / "subject_results" / "state_sweep_subject_results.parquet"}

    monkeypatch.setattr(module, "_run_shard", fake_run_shard)
    monkeypatch.setattr(module, "merge_state_sweep_shards", fake_merge)

    config = {
        "paths": {
            "local_outputs_root": outputs_root.as_posix(),
        },
    }

    outputs = module.run_parallel_state_sweep(
        config,
        config_path="conf/base.yaml",
        models=("xlmr", "nllb"),
        languages=("EN", "FR", "ZH"),
        states=("ATTN", "OUTPUT"),
        conditions=("SHARED", "SPECIFIC"),
        max_parallel=3,
    )

    provenance_text = Path(outputs["parallel_provenance"]).read_text(encoding="utf-8")

    assert sorted(calls) == [("EN", "en"), ("FR", "fr"), ("ZH", "zh")]
    assert outputs["subject_results"] == outputs_root / "subject_results" / "state_sweep_subject_results.parquet"
    assert Path(outputs["run_manifest"]).exists()
    assert "shards: `3`" in provenance_text
    assert "max_parallel: `3`" in provenance_text
    assert "conditions: `SHARED, SPECIFIC`" in provenance_text


def test_merge_state_sweep_shards_uses_only_manifest_listed_files() -> None:
    module = importlib.import_module("src.encoding.merge_state_sweep_shards")
    base_path = make_temp_test_dir("merge_state_sweep_shards")
    outputs_root = base_path / "outputs"
    config = {
        "paths": {
            "local_outputs_root": outputs_root.as_posix(),
        },
    }

    def _write_subject_rows(path: Path, *, language: str, subject_id: str, value: float) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "language": language,
                    "model": "xlmr",
                    "subject_id": subject_id,
                    "roi": "AG",
                    "roi_family": "SEMANTIC",
                    "block": 0,
                    "block_depth_norm": 0.0,
                    "state": "ATTN",
                    "condition": "SHARED",
                    "z_mean": value,
                    "r_mean": value / 10.0,
                    "r2_concat": 0.1,
                    "alpha_best_mode": 100.0,
                    "alpha_best_mean_log10": 2.0,
                    "n_pcs_mean": 3.0,
                    "n_outer_folds": 9,
                    "n_timepoints_total": 90,
                    "seed": 1337,
                }
            ]
        ).to_parquet(path, index=False)
        return str(path)

    en_path = Path(module._subject_results_path(config, "en"))
    fr_path = Path(module._subject_results_path(config, "fr"))
    stale_path = Path(module._subject_results_path(config, "stale_old"))
    _write_subject_rows(en_path, language="EN", subject_id="sub-en", value=0.5)
    _write_subject_rows(fr_path, language="FR", subject_id="sub-fr", value=0.6)
    _write_subject_rows(stale_path, language="ZZ", subject_id="sub-stale", value=9.9)

    manifest_path = outputs_root / "provenance" / "parallel_state_sweep_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "test-run",
                "shards": [
                    {"language": "EN", "subject_results_path": str(en_path)},
                    {"language": "FR", "subject_results_path": str(fr_path)},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    outputs = module.merge_state_sweep_shards(config, manifest_path=manifest_path)
    subject_df = pd.read_parquet(outputs["subject_results"]).copy()

    assert set(subject_df["language"]) == {"EN", "FR"}
