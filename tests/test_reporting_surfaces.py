from __future__ import annotations

from pathlib import Path
import importlib

import pandas as pd
import yaml

from src.stats.generate_all_tables import generate_all_tables
from tests.helpers import make_temp_test_dir


def test_generate_all_tables_builds_table04_from_representative_states() -> None:
    base_path = make_temp_test_dir("generate_all_tables")
    outputs_root = base_path / "outputs"
    group_root = outputs_root / "group_results"
    tables_root = outputs_root / "tables"
    group_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)

    representative_df = pd.DataFrame(
        [
            {
                "model": "xlmr",
                "language": "EN",
                "roi_family": "SEMANTIC",
                "block": 0,
                "state": "FFN",
                "selection_metric": "max_mean_subject_shared_minus_specific",
                "delta_z_mean": 0.8,
                "tie_break_notes": "test",
            },
            {
                "model": "xlmr",
                "language": "EN",
                "roi_family": "AUDITORY",
                "block": 0,
                "state": "POST_ATTN",
                "selection_metric": "max_mean_subject_shared_minus_specific",
                "delta_z_mean": 0.7,
                "tie_break_notes": "test",
            },
        ]
    )
    representative_df.to_parquet(group_root / "representative_states.parquet", index=False)

    group_df = pd.DataFrame(
        [
            {"model": "xlmr", "language": "EN", "roi_family": "SEMANTIC", "block": 0, "state": "FFN", "condition": "SHARED", "roi": "L_AG", "mean_z": 0.9},
            {"model": "xlmr", "language": "EN", "roi_family": "SEMANTIC", "block": 0, "state": "FFN", "condition": "SPECIFIC", "roi": "L_AG", "mean_z": 0.1},
            {"model": "xlmr", "language": "EN", "roi_family": "SEMANTIC", "block": 0, "state": "FFN", "condition": "SHARED", "roi": "R_IFGtri", "mean_z": 0.8},
            {"model": "xlmr", "language": "EN", "roi_family": "SEMANTIC", "block": 0, "state": "FFN", "condition": "SPECIFIC", "roi": "R_IFGtri", "mean_z": 0.2},
            {"model": "xlmr", "language": "EN", "roi_family": "AUDITORY", "block": 0, "state": "POST_ATTN", "condition": "SHARED", "roi": "L_pSTG", "mean_z": 0.85},
            {"model": "xlmr", "language": "EN", "roi_family": "AUDITORY", "block": 0, "state": "POST_ATTN", "condition": "SPECIFIC", "roi": "L_pSTG", "mean_z": 0.15},
            {"model": "xlmr", "language": "EN", "roi_family": "AUDITORY", "block": 0, "state": "POST_ATTN", "condition": "SHARED", "roi": "R_Heschl", "mean_z": 0.75},
            {"model": "xlmr", "language": "EN", "roi_family": "AUDITORY", "block": 0, "state": "POST_ATTN", "condition": "SPECIFIC", "roi": "R_Heschl", "mean_z": 0.25},
        ]
    )
    group_df.to_parquet(group_root / "state_sweep_group_results.parquet", index=False)

    config = {"paths": {"local_outputs_root": outputs_root.as_posix()}}

    try:
        outputs = generate_all_tables(config)
        table04_path = outputs["table04"]
        assert table04_path.exists()
        table04_df = pd.read_csv(table04_path)

        assert len(table04_df) == 4
        assert set(table04_df["roi"]) == {"AG", "IFGtri", "pSTG", "Heschl"}
        assert set(table04_df["representative_state"]) == {"FFN", "POST_ATTN"}
        assert set(table04_df["representative_block"]) == {0}
    finally:
        for path in sorted(base_path.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()


def test_build_claim_evidence_map_marks_existing_and_missing_artifacts(monkeypatch) -> None:
    claim_map_module = importlib.import_module("src.manuscript.build_claim_evidence_map")
    base_path = make_temp_test_dir("claim_map")
    contracts_root = base_path / "contracts"
    outputs_root = base_path / "outputs"
    (contracts_root).mkdir(parents=True, exist_ok=True)
    (outputs_root / "figures").mkdir(parents=True, exist_ok=True)
    (outputs_root / "tables").mkdir(parents=True, exist_ok=True)
    (outputs_root / "group_results").mkdir(parents=True, exist_ok=True)
    (base_path / "src" / "stats").mkdir(parents=True, exist_ok=True)

    claim_contract = {
        "project_title": "Test Project",
        "claims": {
            "C_demo": {
                "text": "Demo claim",
                "status": "required",
                "acceptance_rule": "Exists",
                "evidence": {
                    "figures": ["fig_demo"],
                    "tables": ["table_demo"],
                    "result_files": ["outputs/group_results/demo.parquet"],
                    "scripts": ["src/stats/demo.py"],
                },
            }
        },
        "narrative_modes": {},
    }
    figure_manifest = {
        "figures": [
            {"id": "fig_demo", "filename_png": "outputs/figures/fig_demo.png", "filename_pdf": "outputs/figures/fig_demo.pdf"}
        ]
    }

    with (contracts_root / "claim_evidence_map.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(claim_contract, handle, sort_keys=False)
    with (contracts_root / "figure_manifest.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(figure_manifest, handle, sort_keys=False)

    (outputs_root / "figures" / "fig_demo.png").write_bytes(b"png")
    (outputs_root / "tables" / "table_demo.csv").write_text("x\n1\n", encoding="utf-8")
    (outputs_root / "group_results" / "demo.parquet").write_text("demo", encoding="utf-8")
    (base_path / "src" / "stats" / "demo.py").write_text("print('demo')\n", encoding="utf-8")

    config = {"paths": {"local_outputs_root": outputs_root.as_posix()}}
    monkeypatch.setattr(claim_map_module, "_repo_root", lambda: base_path)

    try:
        output_path = claim_map_module.build_claim_evidence_map(config)
        assert output_path.exists()
        with output_path.open("r", encoding="utf-8") as handle:
            final_map = yaml.safe_load(handle)

        claim_status = final_map["claims"]["C_demo"]
        assert claim_status["satisfied_on_disk"] is True
        assert claim_status["evidence_status"]["figures"]["fig_demo"]["exists"] is True
        assert claim_status["evidence_status"]["tables"]["table_demo"]["exists"] is True
        assert claim_status["evidence_status"]["result_files"]["outputs/group_results/demo.parquet"]["exists"] is True
        assert claim_status["evidence_status"]["scripts"]["src/stats/demo.py"]["exists"] is True
    finally:
        for path in sorted(base_path.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()


def test_build_claim_evidence_map_uses_local_outputs_root_for_outputs_contracts(monkeypatch) -> None:
    claim_map_module = importlib.import_module("src.manuscript.build_claim_evidence_map")
    base_path = make_temp_test_dir("claim_map_local_outputs")
    contracts_root = base_path / "contracts"
    repo_outputs_root = base_path / "outputs"
    local_outputs_root = base_path / "custom_outputs"
    (contracts_root).mkdir(parents=True, exist_ok=True)
    (repo_outputs_root / "figures").mkdir(parents=True, exist_ok=True)
    (local_outputs_root / "figures").mkdir(parents=True, exist_ok=True)
    (local_outputs_root / "tables").mkdir(parents=True, exist_ok=True)
    (local_outputs_root / "group_results").mkdir(parents=True, exist_ok=True)
    (base_path / "src" / "stats").mkdir(parents=True, exist_ok=True)

    claim_contract = {
        "project_title": "Test Project",
        "claims": {
            "C_demo": {
                "text": "Demo claim",
                "status": "required",
                "acceptance_rule": "Exists",
                "evidence": {
                    "figures": ["fig_demo"],
                    "tables": ["table_demo"],
                    "result_files": ["outputs/group_results/demo.parquet"],
                    "scripts": ["src/stats/demo.py"],
                },
            }
        },
        "narrative_modes": {},
    }
    figure_manifest = {
        "figures": [
            {"id": "fig_demo", "filename_png": "outputs/figures/fig_demo.png", "filename_pdf": "outputs/figures/fig_demo.pdf"}
        ]
    }

    with (contracts_root / "claim_evidence_map.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(claim_contract, handle, sort_keys=False)
    with (contracts_root / "figure_manifest.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(figure_manifest, handle, sort_keys=False)

    (local_outputs_root / "figures" / "fig_demo.png").write_bytes(b"png")
    (local_outputs_root / "tables" / "table_demo.csv").write_text("x\n1\n", encoding="utf-8")
    (local_outputs_root / "group_results" / "demo.parquet").write_text("demo", encoding="utf-8")
    (base_path / "src" / "stats" / "demo.py").write_text("print('demo')\n", encoding="utf-8")

    config = {"paths": {"local_outputs_root": local_outputs_root.as_posix()}}
    monkeypatch.setattr(claim_map_module, "_repo_root", lambda: base_path)

    try:
        output_path = claim_map_module.build_claim_evidence_map(config)
        with output_path.open("r", encoding="utf-8") as handle:
            final_map = yaml.safe_load(handle)

        claim_status = final_map["claims"]["C_demo"]
        assert claim_status["satisfied_on_disk"] is True
        assert claim_status["evidence_status"]["figures"]["fig_demo"]["path"].endswith("custom_outputs\\figures\\fig_demo.png") or claim_status["evidence_status"]["figures"]["fig_demo"]["path"].endswith("custom_outputs/figures/fig_demo.png")
        assert claim_status["evidence_status"]["result_files"]["outputs/group_results/demo.parquet"]["exists"] is True
    finally:
        for path in sorted(base_path.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()


def test_build_paper_artifact_manifest_summarizes_required_gaps(monkeypatch) -> None:
    artifact_module = importlib.import_module("src.manuscript.build_paper_artifact_manifest")
    base_path = make_temp_test_dir("paper_artifact_manifest")
    contracts_root = base_path / "contracts"
    outputs_root = base_path / "outputs"
    (contracts_root).mkdir(parents=True, exist_ok=True)
    (outputs_root / "figures").mkdir(parents=True, exist_ok=True)
    (outputs_root / "tables").mkdir(parents=True, exist_ok=True)
    (outputs_root / "group_results").mkdir(parents=True, exist_ok=True)
    (base_path / "src" / "stats").mkdir(parents=True, exist_ok=True)

    claim_contract = {
        "project_title": "Artifact Test",
        "claims": {
            "C_required": {
                "text": "Required claim",
                "status": "required",
                "acceptance_rule": "Exists",
                "evidence": {
                    "figures": ["fig_required"],
                    "tables": ["table_required"],
                    "result_files": ["outputs/group_results/required.parquet"],
                    "scripts": ["src/stats/required.py"],
                },
            },
            "C_optional": {
                "text": "Optional claim",
                "status": "secondary",
                "acceptance_rule": "Optional",
                "evidence": {
                    "figures": ["fig_optional"],
                    "tables": [],
                    "result_files": [],
                    "scripts": [],
                },
            },
        },
        "narrative_modes": {
            "strong": {"required_claims": ["C_required"], "optional_claims": ["C_optional"]},
        },
    }
    figure_manifest = {
        "figures": [
            {"id": "fig_required", "filename_png": "outputs/figures/fig_required.png", "filename_pdf": "outputs/figures/fig_required.pdf", "required": True},
            {"id": "fig_optional", "filename_png": "outputs/figures/fig_optional.png", "filename_pdf": "outputs/figures/fig_optional.pdf", "required": False},
        ]
    }

    with (contracts_root / "claim_evidence_map.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(claim_contract, handle, sort_keys=False)
    with (contracts_root / "figure_manifest.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(figure_manifest, handle, sort_keys=False)

    (outputs_root / "figures" / "fig_required.png").write_bytes(b"png")
    (outputs_root / "tables" / "table_required.csv").write_text("x\n1\n", encoding="utf-8")
    (outputs_root / "group_results" / "required.parquet").write_text("demo", encoding="utf-8")
    (base_path / "src" / "stats" / "required.py").write_text("print('required')\n", encoding="utf-8")

    config = {"paths": {"local_outputs_root": outputs_root.as_posix()}}
    monkeypatch.setattr(artifact_module, "_repo_root", lambda: base_path)
    monkeypatch.setattr(importlib.import_module("src.manuscript.build_claim_evidence_map"), "_repo_root", lambda: base_path)

    try:
        outputs = artifact_module.build_paper_artifact_manifest(config)
        with outputs["yaml"].open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)

        assert payload["paper_ready"] is False
        assert payload["required_figures_missing"] == ["fig_required"]
        assert payload["required_claims_missing"] == ["C_required"]
        assert payload["narrative_modes"]["strong"]["ready"] is False
    finally:
        for path in sorted(base_path.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()
