from __future__ import annotations

import importlib
from pathlib import Path

from tests.helpers import make_temp_test_dir


def test_run_full_paper_pipeline_runs_explicit_stage_contract(monkeypatch) -> None:
    module = importlib.import_module("src.encoding.run_full_paper_pipeline")
    base_path = make_temp_test_dir("run_full_paper_pipeline")
    outputs_root = base_path / "outputs"

    call_order: list[str] = []

    def fake_refresh_manifest(config):
        _ = config
        call_order.append("refresh_manifest")
        return {"manifest": outputs_root / "provenance" / "refresh_output_job_manifest.parquet"}

    def fake_fastpath(config, *, config_path, models, languages, states, max_subjects, include_downstream):
        _ = (config, config_path)
        call_order.append(
            f"fastpath:{','.join(models)}:{','.join(languages)}:{','.join(states)}:{max_subjects}:{include_downstream}"
        )
        return {"subject_results": outputs_root / "subject_results" / "state_sweep_subject_results.parquet"}

    def fake_refresh_tables(config):
        _ = config
        call_order.append("refresh_tables")
        return {"table01": outputs_root / "tables" / "table01_sample_summary.csv"}

    def fake_representatives(config):
        _ = config
        call_order.append("representatives")
        return {"representative_states": outputs_root / "group_results" / "representative_states.parquet"}

    def fake_attribution(config):
        _ = config
        call_order.append("attribution")
        return {"word_attributions": outputs_root / "subject_results" / "word_attributions.parquet"}

    def fake_deletion(config):
        _ = config
        call_order.append("deletion")
        return {"deletion_validation": outputs_root / "subject_results" / "deletion_validation.parquet"}

    def fake_tables(config):
        _ = config
        call_order.append("tables")
        return {"table03": outputs_root / "tables" / "table03_primary_confirmatory_stats.csv"}

    def fake_figures(config):
        _ = config
        call_order.append("figures")
        return {"fig01_png": outputs_root / "figures" / "fig01_pipeline_bridge.png"}

    def fake_claim_map(config):
        _ = config
        call_order.append("claim_map")
        return outputs_root / "group_results" / "claim_evidence_map_final.yaml"

    def fake_draft(config):
        _ = config
        call_order.append("draft")
        return {"draft": outputs_root / "manuscript" / "paper" / "main.md"}

    def fake_artifact_manifest(config):
        _ = config
        call_order.append("artifact_manifest")
        return {"yaml": outputs_root / "provenance" / "paper_artifact_manifest.yaml"}

    monkeypatch.setattr(module, "_run_refresh_job_manifest_stage", fake_refresh_manifest)
    monkeypatch.setattr(module, "run_sequel_fastpath", fake_fastpath)
    monkeypatch.setattr(module, "_run_refresh_table_stage", fake_refresh_tables)
    monkeypatch.setattr(module, "_run_representatives_stage", fake_representatives)
    monkeypatch.setattr(module, "_run_attribution_stage", fake_attribution)
    monkeypatch.setattr(module, "_run_deletion_stage", fake_deletion)
    monkeypatch.setattr(module, "_run_tables_stage", fake_tables)
    monkeypatch.setattr(module, "_run_figures_stage", fake_figures)
    monkeypatch.setattr(module, "_run_claim_map_stage", fake_claim_map)
    monkeypatch.setattr(module, "_run_draft_stage", fake_draft)
    monkeypatch.setattr(module, "_run_artifact_manifest_stage", fake_artifact_manifest)

    config = {
        "paths": {
            "local_outputs_root": outputs_root.as_posix(),
        },
    }

    outputs = module.run_full_paper_pipeline(
        config,
        models=("xlmr", "nllb"),
        languages=("en", "fr"),
        states=("ATTN", "OUTPUT"),
        max_subjects=3,
        require_complete_artifacts=False,
    )

    provenance_text = Path(outputs["provenance"]).read_text(encoding="utf-8")

    assert call_order == [
        "refresh_manifest",
        "fastpath:xlmr,nllb:EN,FR:ATTN,OUTPUT:3:False",
        "refresh_tables",
        "representatives",
        "attribution",
        "deletion",
        "tables",
        "figures",
        "claim_map",
        "draft",
        "artifact_manifest",
    ]
    assert outputs["fastpath_subject_results"] == outputs_root / "subject_results" / "state_sweep_subject_results.parquet"
    assert outputs["artifact_manifest_yaml"] == outputs_root / "provenance" / "paper_artifact_manifest.yaml"
    assert "Stage order: refresh job manifest -> sequel fastpath core -> refresh tables -> representative states -> attribution -> deletion -> tables -> figures -> claim map -> draft -> paper artifact manifest" in provenance_text
