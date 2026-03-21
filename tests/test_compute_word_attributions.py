from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.helpers import make_temp_test_dir


WORD_ATTRIBUTION_REQUIRED_COLUMNS = {
    "subject_id",
    "model",
    "target_language",
    "roi_family",
    "triplet_id",
    "representative_block",
    "representative_state",
    "condition",
    "sentence_language",
    "word_index",
    "word_text",
    "word_char_start",
    "word_char_end",
    "upos",
    "token_class",
    "attribution_score",
    "mapping_status",
    "n_subwords_deleted",
}


def _load_module():
    spec = importlib.util.find_spec("src.attribution.compute_word_attributions")
    if spec is None:
        pytest.skip("src.attribution.compute_word_attributions has not landed yet")
    return importlib.import_module("src.attribution.compute_word_attributions")


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


def _build_semantic_and_auditory_representative_states() -> pd.DataFrame:
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
            },
            {
                "model": "xlmr",
                "language": "EN",
                "roi_family": "AUDITORY",
                "block": 7,
                "state": "POST_ATTN",
                "selection_metric": "max_mean_subject_shared_minus_specific",
                "delta_z_mean": 0.4,
                "tie_break_notes": "synthetic",
            },
        ]
    )


def _build_synthetic_triplets() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "triplet_id": "t001",
                "en_text": "alpha beta",
                "fr_text": "alpha beta",
                "zh_text": "甲 乙",
                "en_onset_sec": 0.0,
                "en_offset_sec": 2.0,
                "fr_onset_sec": 0.0,
                "fr_offset_sec": 2.0,
                "zh_onset_sec": 0.0,
                "zh_offset_sec": 2.0,
                "canonical_run": 1,
            }
        ]
    )


def test_compute_word_attributions_writes_shared_rows_and_mapping_failures(monkeypatch) -> None:
    module = _load_module()
    base_path = make_temp_test_dir("compute_word_attributions")
    outputs_root = base_path / "outputs"
    manifests_root = base_path / "data" / "manifests"
    outputs_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)

    representative_states_path = outputs_root / "group_results" / "representative_states.parquet"
    representative_states_path.parent.mkdir(parents=True, exist_ok=True)
    _build_synthetic_representative_states().to_parquet(representative_states_path, index=False)
    _build_synthetic_triplets().to_parquet(manifests_root / "triplets.parquet", index=False)

    config = {
        "paths": {
            "local_outputs_root": outputs_root.as_posix(),
            "local_manifests_root": manifests_root.as_posix(),
        },
        "project": {"seed": 1337},
        "attribution": {"matched_random_draws": 100, "deletion_ks": [1, 2, 3]},
    }

    word_rows = pd.DataFrame(
        [
            {
                "subject_id": "S01",
                "model": "xlmr",
                "target_language": "EN",
                "roi_family": "SEMANTIC",
                "triplet_id": "t001",
                "representative_block": 11,
                "representative_state": "FFN",
                "condition": "SHARED",
                "sentence_language": "FR",
                "word_index": 0,
                "word_text": "alpha",
                "word_char_start": 0,
                "word_char_end": 5,
                "upos": "NOUN",
                "token_class": "CONTENT",
                "attribution_score": 0.42,
                "mapping_status": "OK",
                "n_subwords_deleted": 1,
            },
            {
                "subject_id": "S01",
                "model": "xlmr",
                "target_language": "EN",
                "roi_family": "SEMANTIC",
                "triplet_id": "t001",
                "representative_block": 11,
                "representative_state": "FFN",
                "condition": "SHARED",
                "sentence_language": "EN",
                "word_index": 1,
                "word_text": "beta",
                "word_char_start": 6,
                "word_char_end": 10,
                "upos": "VERB",
                "token_class": "CONTENT",
                "attribution_score": 0.18,
                "mapping_status": "OK",
                "n_subwords_deleted": 1,
            },
        ]
    )
    failure_rows = pd.DataFrame(
        [
            {
                "subject_id": "S01",
                "model": "xlmr",
                "target_language": "EN",
                "roi_family": "SEMANTIC",
                "triplet_id": "t001",
                "representative_block": 11,
                "representative_state": "FFN",
                "condition": "SHARED",
                "sentence_language": "ZH",
                "word_index": 0,
                "word_text": "甲",
                "word_char_start": 0,
                "word_char_end": 1,
                "failure_reason": "offset_mapping_failed",
                "mapping_status": "FAILED",
            }
        ]
    )

    def _fake_loader(*_args, **_kwargs):
        return _build_synthetic_triplets()

    def _fake_compute_for_setting(*_args, **_kwargs):
        return word_rows.copy(), failure_rows.copy()

    monkeypatch.setattr(module, "_load_triplets", _fake_loader, raising=False)
    monkeypatch.setattr(module, "_load_representative_states", lambda *_args, **_kwargs: _build_synthetic_representative_states(), raising=False)
    monkeypatch.setattr(module, "_compute_for_representative_setting", _fake_compute_for_setting, raising=False)
    monkeypatch.setattr(module, "_compute_shuffled_word_attributions", _fake_compute_for_setting, raising=False)
    monkeypatch.setattr(module, "_write_parquet", lambda df, path: pd.DataFrame(df).to_parquet(path, index=False), raising=False)

    try:
        outputs = module.compute_word_attributions(config)

        word_attributions_path = outputs["word_attributions"]
        failure_path = outputs["mapping_failures"]
        assert word_attributions_path.exists()
        assert failure_path.exists()

        word_df = pd.read_parquet(word_attributions_path)
        failure_df = pd.read_parquet(failure_path)

        assert WORD_ATTRIBUTION_REQUIRED_COLUMNS.issubset(set(word_df.columns))
        assert not word_df.empty
        assert (word_df["condition"].astype(str) == "SHARED").all()
        assert "mapping_status" in word_df.columns

        assert not failure_df.empty
        assert (failure_df["mapping_status"].astype(str) == "FAILED").any()
        assert "failure_reason" in failure_df.columns
        assert "offset_mapping_failed" in set(failure_df["failure_reason"].astype(str))
    finally:
        for path in sorted(base_path.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()


def test_compute_word_attributions_filters_auditory_scope_and_writes_token_subset(monkeypatch) -> None:
    module = _load_module()
    base_path = make_temp_test_dir("compute_word_scope")
    outputs_root = base_path / "outputs"
    manifests_root = base_path / "data" / "manifests"
    outputs_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)

    representative_states_path = outputs_root / "group_results" / "representative_states.parquet"
    representative_states_path.parent.mkdir(parents=True, exist_ok=True)
    _build_semantic_and_auditory_representative_states().to_parquet(representative_states_path, index=False)
    _build_synthetic_triplets().to_parquet(manifests_root / "triplets.parquet", index=False)

    config = {
        "paths": {
            "local_outputs_root": outputs_root.as_posix(),
            "local_manifests_root": manifests_root.as_posix(),
        },
        "dataset": {"languages": ["EN", "FR", "ZH"]},
        "project": {"seed": 1337},
        "attribution": {
            "token_subset": {"name": "strict_1_1_1_v1", "triplets_per_run": 15},
            "token_scope": {
                "include_semantic": True,
                "include_auditory_if_primary_supported": True,
                "support_p_threshold": 0.05,
                "require_positive_effect": True,
            },
        },
    }

    calls: list[tuple[str, str]] = []

    def _fake_compute_for_setting(*_args, **kwargs):
        representative_row = kwargs["representative_row"]
        calls.append((str(representative_row["roi_family"]), str(representative_row["language"])))
        return (
            pd.DataFrame(columns=module.WORD_ATTRIBUTION_COLUMNS),
            pd.DataFrame(columns=module.MAPPING_FAILURE_COLUMNS),
        )

    monkeypatch.setattr(module, "_compute_for_representative_setting", _fake_compute_for_setting, raising=False)
    monkeypatch.setattr(module, "_write_parquet", lambda df, path: pd.DataFrame(df).to_parquet(path, index=False), raising=False)

    try:
        outputs = module.compute_word_attributions(config)

        token_subset_path = outputs_root / "provenance" / "token_triplet_subset.parquet"
        assert outputs["provenance"].exists()
        assert token_subset_path.exists()
        subset_df = pd.read_parquet(token_subset_path)
        provenance_text = outputs["provenance"].read_text(encoding="utf-8")

        assert calls == [("SEMANTIC", "EN")]
        assert len(subset_df) == 3
        assert set(subset_df["target_language"].astype(str)) == {"EN", "FR", "ZH"}
        assert "token_subset_name" in provenance_text
        assert "representative_settings_used" in provenance_text
    finally:
        for path in sorted(base_path.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()


def test_segment_words_supports_deterministic_zh_fallback() -> None:
    module = _load_module()

    words = module._segment_words("甲乙，丙", language="ZH")

    assert [word["word_text"] for word in words] == ["甲", "乙", "，", "丙"]
    assert [(word["word_char_start"], word["word_char_end"]) for word in words] == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert words[0]["token_class"] == "EDGE"
    assert words[1]["token_class"] == "CONTENT"
    assert words[2]["token_class"] == "PUNCT"
    assert words[3]["token_class"] == "EDGE"


def test_load_or_create_token_subset_manifest_rebuilds_stale_subset() -> None:
    module = _load_module()
    base_path = make_temp_test_dir("token_subset_rebuild")
    outputs_root = base_path / "outputs"
    manifests_root = base_path / "data" / "manifests"
    outputs_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)

    config = {
        "paths": {
            "local_outputs_root": outputs_root.as_posix(),
            "local_manifests_root": manifests_root.as_posix(),
        },
        "dataset": {"languages": ["EN", "FR", "ZH"]},
        "attribution": {
            "token_subset": {"name": "strict_1_1_1_v1", "triplets_per_run": 15},
        },
    }
    triplets = _build_synthetic_triplets()

    stale_path = outputs_root / "provenance" / "token_triplet_subset.parquet"
    stale_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "target_language": "EN",
                "canonical_run": 1,
                "triplet_id": "wrong",
                "triplet_row_index": 999,
                "token_subset_name": "stale_subset",
                "token_subset_rank": 1,
            }
        ]
    ).to_parquet(stale_path, index=False)

    try:
        subset = module._load_or_create_token_subset_manifest(config, triplets)

        rebuilt = pd.read_parquet(stale_path).copy()
        assert len(subset) == 3
        assert len(rebuilt) == 3
        assert set(rebuilt["target_language"].astype(str)) == {"EN", "FR", "ZH"}
        assert set(rebuilt["token_subset_name"].astype(str)) == {"strict_1_1_1_v1"}
        assert set(rebuilt["triplet_id"].astype(str)) == {"t001"}
    finally:
        for path in sorted(base_path.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()


def test_subset_triplets_normalizes_triplet_id_dtype_before_merge() -> None:
    module = _load_module()

    triplets = pd.DataFrame(
        [
            {
                "triplet_id": 1,
                "triplet_row_index": 0,
                "canonical_run": 1,
                "en_text": "alpha beta",
                "fr_text": "alpha beta",
                "zh_text": "甲乙",
            }
        ]
    )
    subset_manifest = pd.DataFrame(
        [
            {
                "target_language": "EN",
                "canonical_run": 1,
                "triplet_id": "1",
                "triplet_row_index": 0,
                "token_subset_name": "strict_1_1_1_v1",
                "token_subset_rank": 1,
            }
        ]
    )

    merged = module._subset_triplets_for_target_language(
        triplets,
        subset_manifest,
        target_language="EN",
    )

    assert len(merged) == 1
    assert merged.loc[0, "triplet_id"] == "1"
    assert merged.loc[0, "triplet_row_index"] == 0
    assert merged.loc[0, "canonical_run"] == 1
    assert merged.loc[0, "en_text"] == "alpha beta"


def test_prepare_subject_family_aligns_runwise_nuisance_columns(monkeypatch) -> None:
    module = _load_module()
    base_path = make_temp_test_dir("attr_subject_family_align")
    outputs_root = base_path / "outputs"
    manifests_root = base_path / "data" / "manifests"
    outputs_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)

    triplets = pd.DataFrame(
        [
            {
                "triplet_id": "1",
                "canonical_run": 1,
                "en_text": "alpha",
                "fr_text": "alpha",
                "zh_text": "甲",
                "en_onset_sec": 0.0,
                "en_offset_sec": 1.0,
                "fr_onset_sec": 0.0,
                "fr_offset_sec": 1.0,
                "zh_onset_sec": 0.0,
                "zh_offset_sec": 1.0,
            },
            {
                "triplet_id": "2",
                "canonical_run": 2,
                "en_text": "beta",
                "fr_text": "beta",
                "zh_text": "乙",
                "en_onset_sec": 0.0,
                "en_offset_sec": 1.0,
                "fr_onset_sec": 0.0,
                "fr_offset_sec": 1.0,
                "zh_onset_sec": 0.0,
                "zh_offset_sec": 1.0,
            },
        ]
    )
    sample_manifest = pd.DataFrame(
        [
            {"language": "EN", "subject_id": "S01", "included": True},
        ]
    )
    roi_metadata = pd.DataFrame(
        [
            {"roi_index": 1, "roi_name": "AG_L", "roi_family": "SEMANTIC"},
        ]
    )
    roi_arrays_root = outputs_root / "synthetic_roi_arrays"
    roi_arrays_root.mkdir(parents=True, exist_ok=True)
    run1 = roi_arrays_root / "run1.npy"
    run2 = roi_arrays_root / "run2.npy"
    np.save(run1, np.asarray([[0.1], [0.2]], dtype=np.float32))
    np.save(run2, np.asarray([[0.3], [0.4]], dtype=np.float32))
    roi_manifest = pd.DataFrame(
        [
            {
                "language": "EN",
                "subject_id": "S01",
                "canonical_run": 1,
                "roi_timeseries_path": run1.as_posix(),
            },
            {
                "language": "EN",
                "subject_id": "S01",
                "canonical_run": 2,
                "roi_timeseries_path": run2.as_posix(),
            },
        ]
    )

    triplets.to_parquet(manifests_root / "triplets.parquet", index=False)
    sample_manifest.to_parquet(manifests_root / "sample_manifest.parquet", index=False)
    roi_manifest.to_parquet(manifests_root / "roi_manifest.parquet", index=False)
    roi_metadata.to_parquet(manifests_root / "roi_metadata.parquet", index=False)

    config = {
        "paths": {
            "local_outputs_root": outputs_root.as_posix(),
            "local_manifests_root": manifests_root.as_posix(),
        },
        "dataset": {"tr_seconds": 2.0},
        "features": {"fine_hz": 10.0},
        "nuisance": {"highpass_cutoff_sec": 128.0},
    }

    class _StubOperator:
        def __init__(self) -> None:
            self.hrf_kernel = np.asarray([1.0], dtype=np.float32)

    monkeypatch.setattr(module, "build_language_sentence_to_bold_operator", lambda *args, **kwargs: _StubOperator())
    monkeypatch.setattr(module, "_load_annotation_tables", lambda *args, **kwargs: (pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(module, "_run_scan_counts", lambda *_args, **_kwargs: {1: 2, 2: 2})

    def _fake_build_run_nuisance_and_acoustic(*_args, **kwargs):
        run_index = int(kwargs["run_index"])
        n_scans = int(kwargs["n_scans"])
        nuisance_df = pd.DataFrame({"intercept": np.ones(n_scans, dtype=np.float32)})
        if run_index == 2:
            nuisance_df["highpass_00"] = np.zeros(n_scans, dtype=np.float32)
        acoustic_df = pd.DataFrame({"rms": np.zeros(n_scans, dtype=np.float32)})
        return nuisance_df, acoustic_df

    monkeypatch.setattr(module, "_build_run_nuisance_and_acoustic", _fake_build_run_nuisance_and_acoustic)

    try:
        prepared = module._prepare_subject_family(
            config,
            target_language="EN",
            subject_id="S01",
            roi_family="SEMANTIC",
        )

        assert prepared.nuisance_by_run[1].shape == prepared.nuisance_by_run[2].shape
        assert prepared.nuisance_by_run[1].shape[1] == 3
    finally:
        for path in sorted(base_path.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()


def test_prediction_update_matches_full_recompute_under_shared_transform_path() -> None:
    module = _load_module()
    ridge_module = importlib.import_module("src.encoding.multi_target_ridge")

    X_train = np.asarray(
        [
            [0.2, 1.0],
            [0.4, 0.8],
            [0.7, 0.6],
            [1.0, 0.2],
        ],
        dtype=np.float32,
    )
    y_train = np.asarray([0.3, 0.5, 0.9, 1.1], dtype=np.float32)
    background = np.asarray(
        [
            [0.1, 0.3],
            [0.2, 0.1],
        ],
        dtype=np.float32,
    )
    basis = np.asarray(
        [
            [1.0],
            [0.5],
        ],
        dtype=np.float32,
    )
    feature_old = np.asarray([0.4, 0.8], dtype=np.float32)
    feature_new = np.asarray([0.9, 0.3], dtype=np.float32)
    X_test_old = background + (basis @ feature_old[None, :]).astype(np.float32, copy=False)
    X_test_new = background + (basis @ feature_new[None, :]).astype(np.float32, copy=False)

    X_train_std, X_test_old_std, scaler = ridge_module.standardize_train_test(X_train, X_test_old)
    X_train_pca, X_test_old_pca, transform = ridge_module.fit_pca_train_test(
        X_train_std,
        X_test_old_std,
        max_components=2,
        variance_threshold=0.95,
        random_state=1337,
    )
    beta = ridge_module.ridge_coefficients(X_train_pca, y_train, alpha=10.0)
    base_prediction = X_test_old_pca @ beta

    fold = module.FoldAttributionModel(
        held_out_run=1,
        row_to_local_index={7: 0},
        basis=basis,
        scale_std=scaler.std.astype(np.float32, copy=False),
        pca_components=transform.components.astype(np.float32, copy=False),
        ridge_beta=np.asarray(beta, dtype=np.float32),
        base_prediction=np.asarray(base_prediction, dtype=np.float32),
        actual=np.zeros_like(base_prediction, dtype=np.float32),
        window_by_triplet={7: np.asarray([0, 1], dtype=np.int64)},
    )

    base_contribution = module._prediction_contribution(
        fold,
        triplet_row_index=7,
        feature_vector=feature_old,
    )
    mutated_contribution = module._prediction_contribution(
        fold,
        triplet_row_index=7,
        feature_vector=feature_new,
    )
    updated_prediction = fold.base_prediction - base_contribution + mutated_contribution

    X_test_new_std = scaler.transform(X_test_new)
    full_recompute = transform.transform(X_test_new_std) @ beta

    assert np.allclose(updated_prediction, full_recompute, atol=1.0e-5)
