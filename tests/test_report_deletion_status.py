from __future__ import annotations

import importlib
import json
from pathlib import Path

import pandas as pd

from tests.helpers import make_temp_test_dir


def test_report_deletion_status_summarizes_partial_progress_and_pending_settings() -> None:
    module = importlib.import_module("src.attribution.report_deletion_status")
    base_path = make_temp_test_dir("report_deletion_status")
    outputs_root = base_path / "outputs"
    manifests_root = base_path / "data" / "manifests"
    group_results_root = outputs_root / "group_results"
    shards_root = outputs_root / "subject_results" / "deletion_validation_shards"
    progress_root = outputs_root / "provenance" / "deletion_progress_logs"
    group_results_root.mkdir(parents=True, exist_ok=True)
    shards_root.mkdir(parents=True, exist_ok=True)
    progress_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {"model": "nllb", "language": "EN", "roi_family": "SEMANTIC", "block": 8, "state": "ATTN"},
            {"model": "xlmr", "language": "FR", "roi_family": "SEMANTIC", "block": 8, "state": "ATTN"},
        ]
    ).to_parquet(group_results_root / "representative_states_token_scope.parquet", index=False)

    pd.DataFrame(
        [
            {"language": "EN", "subject_id": "sub-EN001", "included": True},
            {"language": "EN", "subject_id": "sub-EN002", "included": True},
            {"language": "FR", "subject_id": "sub-FR001", "included": True},
        ]
    ).to_parquet(manifests_root / "sample_manifest.parquet", index=False)

    shard_path = shards_root / "deletion_validation__nllb__en__semantic__b08__attn.parquet"
    pd.DataFrame(
        [
            {
                "subject_id": "sub-EN001",
                "model": "nllb",
                "language": "EN",
                "roi_family": "SEMANTIC",
                "representative_block": 8,
                "representative_state": "ATTN",
                "k": 1,
                "base_z": 0.4,
                "topk_z": 0.3,
                "random_z_mean": 0.35,
                "delta_topk": 0.1,
                "delta_random": 0.05,
                "delta_diff": 0.05,
                "n_random_draws": 100,
                "matching_policy": "exact",
            },
            {
                "subject_id": "sub-EN001",
                "model": "nllb",
                "language": "EN",
                "roi_family": "SEMANTIC",
                "representative_block": 8,
                "representative_state": "ATTN",
                "k": 2,
                "base_z": 0.4,
                "topk_z": 0.28,
                "random_z_mean": 0.34,
                "delta_topk": 0.12,
                "delta_random": 0.06,
                "delta_diff": 0.06,
                "n_random_draws": 100,
                "matching_policy": "exact",
            },
        ]
    ).to_parquet(shard_path, index=False)

    metadata_path = shards_root / "deletion_validation__nllb__en__semantic__b08__attn__metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "status": "running",
                "model": "nllb",
                "target_language": "EN",
                "roi_family": "SEMANTIC",
                "representative_block": 8,
                "representative_state": "ATTN",
                "started_utc": "2026-03-20T10:00:00Z",
                "updated_utc": "2026-03-20T10:10:00Z",
                "resume_count": 1,
                "subject_ids_total": 2,
                "subject_ids": ["sub-EN001", "sub-EN002"],
                "subjects_completed": 1,
                "completed_subject_ids": ["sub-EN001"],
                "current_subject_id": "sub-EN002",
                "current_subject_index": 2,
                "rows_written": 2,
                "deletion_ks": [1, 2],
                "subject_progress": {
                    "sub-EN001": {"status": "completed", "completed_ks": [1, 2], "updated_utc": "2026-03-20T10:09:00Z"},
                    "sub-EN002": {"status": "running", "completed_ks": [], "updated_utc": "2026-03-20T10:10:00Z"},
                },
                "last_error": None,
            },
            ensure_ascii=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (progress_root / "deletion_validation__nllb__en__semantic__b08__attn.jsonl").write_text(
        '{"timestamp_utc":"2026-03-20T10:10:00Z","event":"subject_started","subject_id":"sub-EN002"}\n',
        encoding="utf-8",
    )

    config = {
        "paths": {
            "local_outputs_root": outputs_root.as_posix(),
            "local_manifests_root": manifests_root.as_posix(),
        },
        "attribution": {
            "deletion_ks": [1, 2],
            "adaptive_k3": {"enabled": True},
        },
    }

    try:
        summary = module.build_deletion_status_summary(config, roi_families=["SEMANTIC"])
        outputs = module.write_deletion_status_artifacts(
            config,
            summary,
            models=None,
            target_languages=None,
            roi_families=["SEMANTIC"],
        )
        markdown = module.render_deletion_status_markdown(summary)

        assert summary["settings_total"] == 2
        assert summary["settings_by_status"]["running"] == 1
        assert summary["settings_by_status"]["not_started"] == 1
        assert summary["rows_written_total"] == 2
        assert summary["expected_k_slots_total"] == 6
        assert summary["completed_subjects_total"] == 1
        assert summary["subject_ids_total"] == 3
        assert summary["active_settings"][0]["setting_label"] == "nllb/EN/SEMANTIC/b08/ATTN"
        assert summary["active_settings"][0]["current_subject_id"] == "sub-EN002"
        assert outputs["json"].exists()
        assert outputs["markdown"].exists()
        assert "nllb/EN/SEMANTIC/b08/ATTN" in markdown
        assert "xlmr/FR/SEMANTIC/b08/ATTN" in markdown
    finally:
        for path in sorted(base_path.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()
