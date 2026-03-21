from __future__ import annotations

import importlib
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.helpers import make_temp_test_dir


DELETION_VALIDATION_REQUIRED_COLUMNS = {
    "subject_id",
    "model",
    "language",
    "roi_family",
    "representative_block",
    "representative_state",
    "k",
    "base_z",
    "topk_z",
    "random_z_mean",
    "delta_topk",
    "delta_random",
    "delta_diff",
    "n_random_draws",
    "matching_policy",
}


def _load_module():
    spec = importlib.util.find_spec("src.attribution.run_deletion_validation")
    if spec is None:
        pytest.skip("src.attribution.run_deletion_validation has not landed yet")
    return importlib.import_module("src.attribution.run_deletion_validation")


def _build_synthetic_word_attributions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject_id": "S01",
                "model": "xlmr",
                "target_language": "EN",
                "language": "EN",
                "roi_family": "SEMANTIC",
                "triplet_id": "t001",
                "representative_block": 11,
                "representative_state": "FFN",
                "condition": "SHARED",
                "sentence_language": "EN",
                "word_index": 0,
                "word_text": "alpha",
                "word_char_start": 0,
                "word_char_end": 5,
                "upos": "NOUN",
                "token_class": "CONTENT",
                "attribution_score": 0.90,
                "mapping_status": "OK",
                "n_subwords_deleted": 1,
            },
            {
                "subject_id": "S01",
                "model": "xlmr",
                "target_language": "EN",
                "language": "EN",
                "roi_family": "SEMANTIC",
                "triplet_id": "t001",
                "representative_block": 11,
                "representative_state": "FFN",
                "condition": "SHARED",
                "sentence_language": "FR",
                "word_index": 0,
                "word_text": "beta",
                "word_char_start": 0,
                "word_char_end": 4,
                "upos": "NOUN",
                "token_class": "CONTENT",
                "attribution_score": 0.80,
                "mapping_status": "OK",
                "n_subwords_deleted": 1,
            },
            {
                "subject_id": "S01",
                "model": "xlmr",
                "target_language": "EN",
                "language": "EN",
                "roi_family": "SEMANTIC",
                "triplet_id": "t001",
                "representative_block": 11,
                "representative_state": "FFN",
                "condition": "SHARED",
                "sentence_language": "EN",
                "word_index": 1,
                "word_text": "gamma",
                "word_char_start": 6,
                "word_char_end": 11,
                "upos": "VERB",
                "token_class": "FUNCTION",
                "attribution_score": 0.15,
                "mapping_status": "OK",
                "n_subwords_deleted": 1,
            },
        ]
    )


def _build_synthetic_representative_states() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "xlmr",
                "language": "EN",
                "roi_family": "SEMANTIC",
                "block": 11,
                "state": "FFN",
                "selection_metric": "max_mean_subject_shared_minus_specific",
                "delta_z_mean": 0.8,
                "tie_break_notes": "synthetic",
            }
        ]
    )


def _build_synthetic_deletion_rows() -> pd.DataFrame:
    rows = []
    for k in (1, 2, 3):
        base_z = 0.50
        topk_z = 0.50 - (0.05 * k)
        random_z_mean = 0.50 - (0.02 * k)
        delta_topk = base_z - topk_z
        delta_random = base_z - random_z_mean
        rows.append(
            {
                "subject_id": "S01",
                "model": "xlmr",
                "language": "EN",
                "roi_family": "SEMANTIC",
                "representative_block": 11,
                "representative_state": "FFN",
                "k": k,
                "base_z": base_z,
                "topk_z": topk_z,
                "random_z_mean": random_z_mean,
                "delta_topk": delta_topk,
                "delta_random": delta_random,
                "delta_diff": delta_topk - delta_random,
                "n_random_draws": 10,
                "matching_policy": "sentence_and_token_class_multiset",
            }
        )
    return pd.DataFrame(rows)


def test_run_deletion_validation_writes_expected_outputs(monkeypatch) -> None:
    module = _load_module()
    base_path = make_temp_test_dir("run_deletion_validation")
    outputs_root = base_path / "outputs"
    manifests_root = base_path / "data" / "manifests"
    subject_results_root = outputs_root / "subject_results"
    group_results_root = outputs_root / "group_results"
    subject_results_root.mkdir(parents=True, exist_ok=True)
    group_results_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)

    word_attributions_path = subject_results_root / "word_attributions.parquet"
    representative_states_path = group_results_root / "representative_states.parquet"
    _build_synthetic_word_attributions().to_parquet(word_attributions_path, index=False)
    _build_synthetic_representative_states().to_parquet(representative_states_path, index=False)
    pd.DataFrame(
        [
            {
                "triplet_id": "t001",
                "triplet_row_index": 0,
                "canonical_run": 1,
                "en_text": "alpha gamma",
                "fr_text": "beta delta",
                "zh_text": "jia yi",
            }
        ]
    ).to_parquet(manifests_root / "triplets.parquet", index=False)

    config = {
        "paths": {
            "local_outputs_root": outputs_root.as_posix(),
            "local_manifests_root": manifests_root.as_posix(),
        },
        "project": {"seed": 1337},
        "attribution": {"deletion_ks": [1, 2, 3], "matched_random_draws": 10},
    }

    synthetic_deletion_rows = _build_synthetic_deletion_rows()

    def _fake_evaluate_subject_setting(*_args, **_kwargs):
        row_callback = _kwargs.get("row_callback")
        if row_callback is not None:
            for row in synthetic_deletion_rows.to_dict(orient="records"):
                row_callback(dict(row))
        return synthetic_deletion_rows.copy(), {"exact_language_token_class_no_overlap": 3}

    monkeypatch.setattr(
        module,
        "_load_state_array_lookup",
        lambda *_args, **_kwargs: {
            "EN": np.zeros((1, 4), dtype=np.float32),
            "FR": np.zeros((1, 4), dtype=np.float32),
            "ZH": np.zeros((1, 4), dtype=np.float32),
        },
        raising=False,
    )
    monkeypatch.setattr(module, "_evaluate_subject_setting", _fake_evaluate_subject_setting, raising=False)
    monkeypatch.setattr(module, "_warm_start_mutation_encoding", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(
        module,
        "_build_subject_mutation_plan",
        lambda *args, **kwargs: ({}, {}),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "_prepare_subject_family",
        lambda *_args, **kwargs: SimpleNamespace(subject_id=str(kwargs["subject_id"])),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "_write_parquet",
        lambda frame, path: (Path(path).parent.mkdir(parents=True, exist_ok=True), pd.DataFrame(frame).to_parquet(path, index=False))[-1],
        raising=False,
    )

    try:
        outputs = module.run_deletion_validation(config)

        assert isinstance(outputs, dict)
        deletion_path = next(path for path in outputs.values() if path.name == "deletion_validation.parquet")
        provenance_path = next(path for path in outputs.values() if path.suffix == ".md")
        metadata_path = module._deletion_shard_metadata_path(
            config,
            model="xlmr",
            target_language="EN",
            roi_family="SEMANTIC",
            block=11,
            state="FFN",
        )
        progress_log_path = module._deletion_progress_log_path(
            config,
            model="xlmr",
            target_language="EN",
            roi_family="SEMANTIC",
            block=11,
            state="FFN",
        )

        assert deletion_path.exists()
        assert provenance_path.exists()
        assert metadata_path.exists()
        assert progress_log_path.exists()

        deletion_df = pd.read_parquet(deletion_path).copy()
        provenance_text = provenance_path.read_text(encoding="utf-8")
        metadata = module._read_json(metadata_path)
        progress_lines = progress_log_path.read_text(encoding="utf-8").strip().splitlines()

        assert len(deletion_df) == 3
        assert set(deletion_df["k"].astype(int).tolist()) == {1, 2, 3}
        assert DELETION_VALIDATION_REQUIRED_COLUMNS.issubset(set(deletion_df.columns))
        assert (deletion_df["n_random_draws"].astype(int) == 10).all()
        assert deletion_df["matching_policy"].astype(str).nunique() == 1
        assert "deletion" in provenance_text.lower()
        assert deletion_path.name in provenance_text
        assert metadata is not None
        assert metadata["status"] == "completed"
        assert metadata["subjects_completed"] == 1
        assert len(progress_lines) >= 4
        assert any('"event": "k_completed"' in line for line in progress_lines)
    finally:
        for path in sorted(base_path.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()


def test_adaptive_k3_helper_requires_incremental_gain() -> None:
    module = _load_module()

    should_run, reason = module._should_run_k(
        k=3,
        prior_rows_by_k={
            1: {"delta_diff": 0.12},
            2: {"delta_diff": 0.12},
        },
        adaptive_cfg={"enabled": True, "min_delta_diff_gain": 0.0},
    )
    assert should_run is False
    assert reason == "adaptive_k3_skipped_no_incremental_gain"

    should_run, reason = module._should_run_k(
        k=3,
        prior_rows_by_k={
            1: {"delta_diff": 0.05},
            2: {"delta_diff": 0.11},
        },
        adaptive_cfg={"enabled": True, "min_delta_diff_gain": 0.0},
    )
    assert should_run is True
    assert reason == "adaptive_k3_triggered"


def test_run_deletion_validation_reuses_existing_setting_shards(monkeypatch) -> None:
    module = _load_module()
    base_path = make_temp_test_dir("run_deletion_validation_resume")
    outputs_root = base_path / "outputs"
    manifests_root = base_path / "data" / "manifests"
    subject_results_root = outputs_root / "subject_results"
    group_results_root = outputs_root / "group_results"
    subject_results_root.mkdir(parents=True, exist_ok=True)
    group_results_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)

    word_attributions_path = subject_results_root / "word_attributions.parquet"
    representative_states_path = group_results_root / "representative_states.parquet"
    _build_synthetic_word_attributions().to_parquet(word_attributions_path, index=False)
    _build_synthetic_representative_states().to_parquet(representative_states_path, index=False)
    pd.DataFrame(
        [
            {
                "triplet_id": "t001",
                "triplet_row_index": 0,
                "canonical_run": 1,
                "en_text": "alpha gamma",
                "fr_text": "beta delta",
                "zh_text": "jia yi",
            }
        ]
    ).to_parquet(manifests_root / "triplets.parquet", index=False)

    config = {
        "paths": {
            "local_outputs_root": outputs_root.as_posix(),
            "local_manifests_root": manifests_root.as_posix(),
        },
        "project": {"seed": 1337},
        "attribution": {"deletion_ks": [1, 2], "matched_random_draws": 10},
    }

    shard_path = module._deletion_shard_path(
        config,
        model="xlmr",
        target_language="EN",
        roi_family="SEMANTIC",
        block=11,
        state="FFN",
    )
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    _build_synthetic_deletion_rows().iloc[:2].to_parquet(shard_path, index=False)

    def _unexpected_evaluate(*_args, **_kwargs):
        raise AssertionError("existing shard should have been reused instead of recomputed")

    monkeypatch.setattr(module, "_evaluate_subject_setting", _unexpected_evaluate, raising=False)
    monkeypatch.setattr(module, "_warm_start_mutation_encoding", lambda *args, **kwargs: None, raising=False)

    try:
        outputs = module.run_deletion_validation(config)
        deletion_df = pd.read_parquet(outputs["deletion_validation"]).copy()

        assert len(deletion_df) == 2
        assert set(deletion_df["k"].astype(int)) == {1, 2}
        assert outputs["deletion_validation"].name == "deletion_validation.parquet"
    finally:
        for path in sorted(base_path.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()


def test_run_deletion_validation_resumes_partial_setting_checkpoint(monkeypatch) -> None:
    module = _load_module()
    base_path = make_temp_test_dir("run_deletion_validation_partial_resume")
    outputs_root = base_path / "outputs"
    manifests_root = base_path / "data" / "manifests"
    subject_results_root = outputs_root / "subject_results"
    group_results_root = outputs_root / "group_results"
    subject_results_root.mkdir(parents=True, exist_ok=True)
    group_results_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)

    word_attributions_path = subject_results_root / "word_attributions.parquet"
    representative_states_path = group_results_root / "representative_states.parquet"
    _build_synthetic_word_attributions().to_parquet(word_attributions_path, index=False)
    _build_synthetic_representative_states().to_parquet(representative_states_path, index=False)
    pd.DataFrame(
        [
            {
                "triplet_id": "t001",
                "triplet_row_index": 0,
                "canonical_run": 1,
                "en_text": "alpha gamma",
                "fr_text": "beta delta",
                "zh_text": "jia yi",
            }
        ]
    ).to_parquet(manifests_root / "triplets.parquet", index=False)

    config = {
        "paths": {
            "local_outputs_root": outputs_root.as_posix(),
            "local_manifests_root": manifests_root.as_posix(),
        },
        "project": {"seed": 1337},
        "attribution": {"deletion_ks": [1, 2], "matched_random_draws": 10},
    }

    partial_rows = _build_synthetic_deletion_rows().iloc[:1].copy()
    remaining_rows = _build_synthetic_deletion_rows().iloc[1:2].copy()
    shard_path = module._deletion_shard_path(
        config,
        model="xlmr",
        target_language="EN",
        roi_family="SEMANTIC",
        block=11,
        state="FFN",
    )
    metadata_path = module._deletion_shard_metadata_path(
        config,
        model="xlmr",
        target_language="EN",
        roi_family="SEMANTIC",
        block=11,
        state="FFN",
    )
    progress_log_path = module._deletion_progress_log_path(
        config,
        model="xlmr",
        target_language="EN",
        roi_family="SEMANTIC",
        block=11,
        state="FFN",
    )
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    partial_rows.to_parquet(shard_path, index=False)
    metadata_path.write_text(
        """
{
  "status": "running",
  "model": "xlmr",
  "target_language": "EN",
  "roi_family": "SEMANTIC",
  "representative_block": 11,
  "representative_state": "FFN",
  "started_utc": "2026-03-20T10:00:00Z",
  "updated_utc": "2026-03-20T10:05:00Z",
  "resume_count": 0,
  "subject_ids_total": 1,
  "subject_ids": ["S01"],
  "subjects_completed": 0,
  "completed_subject_ids": [],
  "current_subject_id": "S01",
  "current_subject_index": 1,
  "rows_written": 1,
  "deletion_ks": [1, 2],
  "subject_progress": {
    "S01": {
      "status": "running",
      "completed_ks": [1],
      "updated_utc": "2026-03-20T10:05:00Z"
    }
  },
  "last_error": null
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    def _resume_evaluate(*_args, **_kwargs):
        existing_rows_by_k = _kwargs["existing_rows_by_k"]
        assert set(existing_rows_by_k) == {1}
        row_callback = _kwargs.get("row_callback")
        if row_callback is not None:
            for row in remaining_rows.to_dict(orient="records"):
                row_callback(dict(row))
        return remaining_rows.copy(), {"exact_language_token_class_no_overlap": 1}

    monkeypatch.setattr(
        module,
        "_load_state_array_lookup",
        lambda *_args, **_kwargs: {
            "EN": np.zeros((1, 4), dtype=np.float32),
            "FR": np.zeros((1, 4), dtype=np.float32),
            "ZH": np.zeros((1, 4), dtype=np.float32),
        },
        raising=False,
    )
    monkeypatch.setattr(module, "_evaluate_subject_setting", _resume_evaluate, raising=False)
    monkeypatch.setattr(module, "_warm_start_mutation_encoding", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(
        module,
        "_prepare_subject_family",
        lambda *_args, **kwargs: SimpleNamespace(subject_id=str(kwargs["subject_id"])),
        raising=False,
    )

    try:
        outputs = module.run_deletion_validation(config)
        deletion_df = pd.read_parquet(outputs["deletion_validation"]).copy()
        metadata = module._read_json(metadata_path)
        progress_lines = progress_log_path.read_text(encoding="utf-8").strip().splitlines()

        assert len(deletion_df) == 2
        assert set(deletion_df["k"].astype(int).tolist()) == {1, 2}
        assert metadata is not None
        assert metadata["status"] == "completed"
        assert metadata["subjects_completed"] == 1
        assert metadata["completed_subject_ids"] == ["S01"]
        assert any('"event": "setting_resumed"' in line for line in progress_lines)
        assert any('"event": "k_completed"' in line for line in progress_lines)
    finally:
        for path in sorted(base_path.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()


def test_setting_mutation_cache_primes_unique_texts_once(monkeypatch) -> None:
    module = _load_module()
    calls: list[tuple[str, list[str]]] = []

    def _fake_run_mutation_encoder_batched(*_args, **kwargs):
        texts = [str(text) for text in kwargs["texts"]]
        calls.append((str(kwargs["language"]), texts))
        return np.arange(len(texts) * 4, dtype=np.float32).reshape(len(texts), 4)

    monkeypatch.setattr(module, "_run_mutation_encoder_batched", _fake_run_mutation_encoder_batched, raising=False)

    cache = module.SettingMutationCache(
        config={"attribution": {}},
        model="xlmr",
        state="FFN",
        block=11,
        batch_size=256,
        vectors_by_language={},
    )
    cache.prime("FR", ["texte supprime", "texte supprime", "autre"])
    cache.prime("ZH", ["甲", "乙", "甲"])
    cache.prime("FR", ["autre", "texte supprime"])

    assert len(calls) == 2
    assert any(language == "FR" and sorted(texts) == ["autre", "texte supprime"] for language, texts in calls)
    assert any(language == "ZH" and sorted(texts) == ["乙", "甲"] for language, texts in calls)
    assert set(cache.vectors_by_language["FR"]) == {"autre", "texte supprime"}
    assert set(cache.vectors_by_language["ZH"]) == {"甲", "乙"}
