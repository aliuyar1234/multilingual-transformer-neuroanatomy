from __future__ import annotations

import importlib
import json
from pathlib import Path

import pandas as pd

from tests.helpers import make_temp_test_dir


def test_run_parallel_internal_state_extraction_runs_all_shards(monkeypatch) -> None:
    module = importlib.import_module("src.features.run_parallel_internal_state_extraction")
    base_path = make_temp_test_dir("run_parallel_internal_state_extraction")
    outputs_root = base_path / "outputs"

    calls: list[tuple[str, tuple[str, ...], str]] = []

    def fake_run_shard(*, config_path, shard, states, max_tokens_per_batch, cache_dtype):
        _ = (config_path, states, max_tokens_per_batch, cache_dtype)
        calls.append((shard.model, shard.languages, shard.artifact_tag))
        return {"model": shard.model, "languages": shard.languages, "artifact_tag": shard.artifact_tag}

    def fake_consolidate(config, *, manifest_path):
        _ = config
        assert Path(manifest_path).exists()
        return {"state_manifest": outputs_root / "caches" / "features" / "state_cache_manifest.parquet"}

    monkeypatch.setattr(module, "_run_shard", fake_run_shard)
    monkeypatch.setattr(module, "consolidate_internal_state_shards", fake_consolidate)

    config = {
        "paths": {
            "local_outputs_root": outputs_root.as_posix(),
        },
    }

    outputs = module.run_parallel_internal_state_extraction(
        config,
        config_path="conf/base.yaml",
        models=("xlmr", "nllb"),
        languages=("EN", "FR"),
        states=("ATTN", "OUTPUT"),
        max_parallel=2,
    )

    provenance_text = Path(outputs["parallel_provenance"]).read_text(encoding="utf-8")

    assert sorted(calls) == [
        ("nllb", ("EN", "FR"), "nllb"),
        ("xlmr", ("EN", "FR"), "xlmr"),
    ]
    assert outputs["state_manifest"] == outputs_root / "caches" / "features" / "state_cache_manifest.parquet"
    assert Path(outputs["run_manifest"]).exists()
    assert "shards: `2`" in provenance_text
    assert "max_parallel: `2`" in provenance_text


def test_consolidate_internal_state_shards_uses_only_manifest_listed_files() -> None:
    module = importlib.import_module("src.features.consolidate_internal_state_shards")
    base_path = make_temp_test_dir("consolidate_internal_state_shards")
    outputs_root = base_path / "outputs"
    manifests_root = base_path / "data" / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)
    config = {
        "paths": {
            "local_outputs_root": outputs_root.as_posix(),
            "local_manifests_root": manifests_root.as_posix(),
        },
    }

    def _write_parquet(path: Path, rows: list[dict[str, object]]) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)
        return str(path)

    manifest_shards: list[dict[str, object]] = []
    for model, language, artifact_tag in (
        ("xlmr", "EN", "xlmr_en"),
        ("nllb", "FR", "nllb_fr"),
    ):
        manifest_shards.append(
            {
                "model": model,
                "language": language,
                "artifact_tag": artifact_tag,
                "state_manifest_path": _write_parquet(
                    module._state_cache_manifest_path(config, artifact_tag),
                    [
                        {
                            "model": model,
                            "language": language,
                            "state": "ATTN",
                            "block": 0,
                            "source_array_path": f"/arrays/{artifact_tag}/attn.npy",
                        }
                    ],
                ),
                "output_manifest_path": _write_parquet(
                    module._output_state_manifest_path(config, artifact_tag),
                    [{"model": model, "language": language, "block": 0, "source_array_path": f"/arrays/{artifact_tag}/output.npy"}],
                ),
                "token_metadata_path": _write_parquet(
                    module._token_metadata_path(config, artifact_tag),
                    [{"model": model, "language": language, "triplet_id": f"{artifact_tag}_triplet"}],
                ),
                "triplet_ids_path": _write_parquet(
                    module._triplet_ids_path(config, artifact_tag),
                    [{"triplet_id": f"{artifact_tag}_triplet"}],
                ),
                "qc_path": _write_parquet(
                    module._state_extraction_qc_path(config, artifact_tag),
                    [{"model": model, "language": language, "block": 0, "state": "ATTN", "max_abs_diff": 0.0}],
                ),
                "provenance_path": str(module._state_extraction_note_path(config, artifact_tag)),
                "adapter_notes_path": str(module._state_adapter_note_path(config, artifact_tag)),
            }
        )

    _write_parquet(
        module._state_cache_manifest_path(config, "stale_old"),
        [{"model": "stale", "language": "ZZ", "state": "ATTN", "block": 9, "source_array_path": "/arrays/stale.npy"}],
    )
    _write_parquet(
        module._output_state_manifest_path(config, "stale_old"),
        [{"model": "stale", "language": "ZZ", "block": 9, "source_array_path": "/arrays/stale_output.npy"}],
    )
    _write_parquet(
        module._token_metadata_path(config, "stale_old"),
        [{"model": "stale", "language": "ZZ", "triplet_id": "stale_triplet"}],
    )
    _write_parquet(
        module._triplet_ids_path(config, "stale_old"),
        [{"triplet_id": "stale_triplet"}],
    )
    _write_parquet(
        module._state_extraction_qc_path(config, "stale_old"),
        [{"model": "stale", "language": "ZZ", "block": 9, "state": "ATTN", "max_abs_diff": 9.0}],
    )

    manifest_path = outputs_root / "provenance" / "parallel_state_extraction_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "test-run",
                "shards": manifest_shards,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    outputs = module.consolidate_internal_state_shards(config, manifest_path=manifest_path)

    state_df = pd.read_parquet(outputs["state_manifest"]).copy()
    output_df = pd.read_parquet(outputs["output_manifest"]).copy()
    token_df = pd.read_parquet(outputs["token_metadata"]).copy()

    assert set(state_df["language"]) == {"EN", "FR"}
    assert set(output_df["language"]) == {"EN", "FR"}
    assert set(token_df["language"]) == {"EN", "FR"}


def test_consolidate_internal_state_shards_deduplicates_identical_triplet_id_shards() -> None:
    module = importlib.import_module("src.features.consolidate_internal_state_shards")
    base_path = make_temp_test_dir("consolidate_internal_state_shards_triplets")
    outputs_root = base_path / "outputs"
    manifests_root = base_path / "data" / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)
    config = {
        "paths": {
            "local_outputs_root": outputs_root.as_posix(),
            "local_manifests_root": manifests_root.as_posix(),
        },
    }

    def _write_parquet(path: Path, rows: list[dict[str, object]]) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)
        return str(path)

    shard_rows = []
    for model_tag, language in (("xlmr", "EN"), ("nllb", "FR")):
        artifact_tag = f"{model_tag}_lane"
        shard_rows.append(
            {
                "model": model_tag,
                "language": language,
                "artifact_tag": artifact_tag,
                "state_manifest_path": _write_parquet(
                    module._state_cache_manifest_path(config, artifact_tag),
                    [{"model": model_tag, "language": language, "state": "ATTN", "block": 0, "source_array_path": f"/arrays/{artifact_tag}.npy"}],
                ),
                "output_manifest_path": _write_parquet(
                    module._output_state_manifest_path(config, artifact_tag),
                    [{"model": model_tag, "language": language, "block": 0, "source_array_path": f"/arrays/{artifact_tag}_out.npy"}],
                ),
                "token_metadata_path": _write_parquet(
                    module._token_metadata_path(config, artifact_tag),
                    [{"model": model_tag, "language": language, "triplet_id": "shared_triplet"}],
                ),
                "triplet_ids_path": _write_parquet(
                    module._triplet_ids_path(config, artifact_tag),
                    [{"triplet_id": 1}, {"triplet_id": 2}],
                ),
                "qc_path": _write_parquet(
                    module._state_extraction_qc_path(config, artifact_tag),
                    [{"model": model_tag, "language": language, "block": 0, "state": "ATTN", "max_abs_diff": 0.0}],
                ),
                "provenance_path": str(module._state_extraction_note_path(config, artifact_tag)),
                "adapter_notes_path": str(module._state_adapter_note_path(config, artifact_tag)),
            }
        )

    manifest_path = outputs_root / "provenance" / "parallel_state_extraction_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"run_id": "test-run", "shards": shard_rows}, indent=2), encoding="utf-8")

    outputs = module.consolidate_internal_state_shards(config, manifest_path=manifest_path)
    triplet_df = pd.read_parquet(outputs["triplet_ids"]).copy()

    assert triplet_df["triplet_id"].tolist() == [1, 2]
