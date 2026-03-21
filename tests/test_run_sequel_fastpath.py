from __future__ import annotations

import importlib
from pathlib import Path

from tests.helpers import make_temp_test_dir


def test_run_sequel_fastpath_runs_stages_in_order(monkeypatch) -> None:
    module = importlib.import_module("src.encoding.run_sequel_fastpath")
    base_path = make_temp_test_dir("run_sequel_fastpath")
    outputs_root = base_path / "outputs"

    call_order: list[str] = []

    def fake_extract(config, *, config_path, models, languages, states):
        _ = (config, config_path)
        call_order.append(f"extract:{','.join(models)}:{','.join(languages)}:{','.join(states)}")
        return {"manifest": outputs_root / "caches" / "features" / "state_cache_manifest.parquet"}

    def fake_build(config, *, states):
        _ = config
        call_order.append(f"build:{','.join(states)}")
        return {"manifest": outputs_root / "caches" / "derived_features" / "state_feature_manifest.parquet"}

    def fake_sweep(config, *, config_path, models, languages, states, max_subjects):
        _ = (config, config_path)
        call_order.append(f"sweep:{','.join(models)}:{','.join(languages)}:{','.join(states)}:{max_subjects}")
        return {"subject_results": outputs_root / "subject_results" / "state_sweep_subject_results.parquet"}

    def fake_primary(config, *, models, languages):
        _ = config
        call_order.append(f"primary:{','.join(models)}:{','.join(languages)}")
        return {"table03": outputs_root / "tables" / "table03_primary_confirmatory_stats.csv"}

    monkeypatch.setattr(module, "_run_extract_stage", fake_extract)
    monkeypatch.setattr(module, "_run_build_shared_specific_stage", fake_build)
    monkeypatch.setattr(module, "_run_state_sweep_stage", fake_sweep)
    monkeypatch.setattr(module, "_run_primary_tests_stage", fake_primary)

    config = {
        "paths": {
            "local_outputs_root": outputs_root.as_posix(),
        },
    }

    outputs = module.run_sequel_fastpath(
        config,
        models=("xlmr", "nllb"),
        languages=("en", "fr"),
        states=("ATTN", "OUTPUT"),
        max_subjects=3,
    )

    provenance_text = Path(outputs["provenance"]).read_text(encoding="utf-8")

    assert call_order == [
        "extract:xlmr,nllb:EN,FR:ATTN,OUTPUT",
        "build:ATTN,OUTPUT",
        "sweep:xlmr,nllb:EN,FR:ATTN,OUTPUT:3",
        "primary:xlmr,nllb:EN,FR",
    ]
    assert outputs["extract_manifest"] == outputs_root / "caches" / "features" / "state_cache_manifest.parquet"
    assert outputs["features_manifest"] == outputs_root / "caches" / "derived_features" / "state_feature_manifest.parquet"
    assert outputs["sweep_subject_results"] == outputs_root / "subject_results" / "state_sweep_subject_results.parquet"
    assert outputs["primary_table03"] == outputs_root / "tables" / "table03_primary_confirmatory_stats.csv"
    assert "Stage order: extract pooled states -> build SHARED/SPECIFIC caches -> run state sweep -> run primary tests" in provenance_text
    assert "models: `xlmr, nllb`" in provenance_text
    assert "languages: `EN, FR`" in provenance_text
