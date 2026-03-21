from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.plots.generate_all_figures import generate_all_figures
from tests.helpers import make_temp_test_dir


def test_generate_all_figures_builds_available_outputs() -> None:
    base_path = make_temp_test_dir("generate_all_figures")
    outputs_root = base_path / "outputs"
    manifests_root = base_path / "data" / "manifests"
    figures_root = outputs_root / "figures"
    group_root = outputs_root / "group_results"
    subject_root = outputs_root / "subject_results"
    tables_root = outputs_root / "tables"
    figures_root.mkdir(parents=True, exist_ok=True)
    group_root.mkdir(parents=True, exist_ok=True)
    subject_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)

    group_df = pd.DataFrame(
        [
            {"language": "EN", "model": "xlmr", "block": 0, "block_depth_norm": 0.0, "state": "ATTN", "condition": "SHARED", "roi": "L_AG", "roi_family": "SEMANTIC", "mean_z": 0.5},
            {"language": "EN", "model": "xlmr", "block": 0, "block_depth_norm": 0.0, "state": "ATTN", "condition": "SPECIFIC", "roi": "L_AG", "roi_family": "SEMANTIC", "mean_z": 0.1},
            {"language": "EN", "model": "xlmr", "block": 1, "block_depth_norm": 1.0, "state": "FFN", "condition": "SHARED", "roi": "R_IFGtri", "roi_family": "SEMANTIC", "mean_z": 0.9},
            {"language": "EN", "model": "xlmr", "block": 1, "block_depth_norm": 1.0, "state": "FFN", "condition": "SPECIFIC", "roi": "R_IFGtri", "roi_family": "SEMANTIC", "mean_z": 0.2},
            {"language": "FR", "model": "nllb", "block": 0, "block_depth_norm": 0.0, "state": "POST_ATTN", "condition": "SHARED", "roi": "L_pSTG", "roi_family": "AUDITORY", "mean_z": 0.8},
            {"language": "FR", "model": "nllb", "block": 0, "block_depth_norm": 0.0, "state": "POST_ATTN", "condition": "SPECIFIC", "roi": "L_pSTG", "roi_family": "AUDITORY", "mean_z": 0.3},
            {"language": "FR", "model": "nllb", "block": 1, "block_depth_norm": 1.0, "state": "OUTPUT", "condition": "SHARED", "roi": "R_Heschl", "roi_family": "AUDITORY", "mean_z": 0.7},
            {"language": "FR", "model": "nllb", "block": 1, "block_depth_norm": 1.0, "state": "OUTPUT", "condition": "SPECIFIC", "roi": "R_Heschl", "roi_family": "AUDITORY", "mean_z": 0.4},
        ]
    )
    group_df.to_parquet(group_root / "state_sweep_group_results.parquet", index=False)
    pd.DataFrame(
        [
            {"model": "xlmr", "language": "EN", "roi_family": "SEMANTIC", "shared_minus_specific": 0.4},
            {"model": "nllb", "language": "FR", "roi_family": "AUDITORY", "shared_minus_specific": 0.2},
        ]
    ).to_parquet(group_root / "output_layer_family_deltas.parquet", index=False)

    primary_df = pd.DataFrame(
        [
            {"model": "xlmr", "language": "EN", "hypothesis": "H1_semantic_ffn_preference", "effect_mean": 0.4, "ci_low": 0.2, "ci_high": 0.6, "p_raw": 0.01, "p_holm": 0.02, "n_subjects": 4},
            {"model": "nllb", "language": "FR", "hypothesis": "H2_auditory_attention_preference", "effect_mean": 0.3, "ci_low": 0.1, "ci_high": 0.5, "p_raw": 0.03, "p_holm": 0.04, "n_subjects": 4},
        ]
    )
    primary_df.to_parquet(group_root / "primary_effect_tables.parquet", index=False)
    pd.DataFrame(
        [
            {"model": "xlmr", "language": "EN", "hypothesis": "H1_semantic_ffn_preference", "subject_id": "S01", "positive_mean": 0.8, "negative_mean": 0.4, "effect_value": 0.4},
            {"model": "nllb", "language": "FR", "hypothesis": "H2_auditory_attention_preference", "subject_id": "S02", "positive_mean": 0.7, "negative_mean": 0.4, "effect_value": 0.3},
        ]
    ).to_parquet(group_root / "primary_subject_effects.parquet", index=False)

    pd.DataFrame(
        [
            {"model": "xlmr", "language": "EN", "roi_family": "SEMANTIC", "condition": "SHARED", "class": "CONTENT", "positive_mass": 0.6, "content_bias": 0.4},
            {"model": "xlmr", "language": "EN", "roi_family": "SEMANTIC", "condition": "SHARED", "class": "FUNCTION", "positive_mass": 0.2, "content_bias": 0.4},
            {"model": "nllb", "language": "FR", "roi_family": "AUDITORY", "condition": "SHARED", "class": "CONTENT", "positive_mass": 0.3, "content_bias": 0.1},
            {"model": "nllb", "language": "FR", "roi_family": "AUDITORY", "condition": "SHARED", "class": "EDGE", "positive_mass": 0.25, "content_bias": 0.1},
        ]
    ).to_csv(tables_root / "table05_token_class_attribution.csv", index=False)

    pd.DataFrame(
        [
            {"model": "xlmr", "language": "EN", "roi_family": "SEMANTIC", "k": 1, "effect_topk": 0.5, "effect_random": 0.2, "diff": 0.3, "ci_low": 0.1, "ci_high": 0.5},
            {"model": "xlmr", "language": "EN", "roi_family": "SEMANTIC", "k": 2, "effect_topk": 0.6, "effect_random": 0.25, "diff": 0.35, "ci_low": 0.15, "ci_high": 0.55},
            {"model": "nllb", "language": "FR", "roi_family": "AUDITORY", "k": 1, "effect_topk": 0.35, "effect_random": 0.18, "diff": 0.17, "ci_low": 0.05, "ci_high": 0.29},
        ]
    ).to_csv(tables_root / "table06_deletion_validation.csv", index=False)

    pd.DataFrame(
        [
            {
                "model": "xlmr",
                "language": "EN",
                "roi_family": "SEMANTIC",
                "block": 1,
                "state": "FFN",
                "selection_metric": "synthetic",
                "delta_z_mean": 0.5,
                "tie_break_notes": "test",
            },
            {
                "model": "nllb",
                "language": "FR",
                "roi_family": "AUDITORY",
                "block": 0,
                "state": "POST_ATTN",
                "selection_metric": "synthetic",
                "delta_z_mean": 0.2,
                "tie_break_notes": "test",
            },
        ]
    ).to_parquet(group_root / "representative_states.parquet", index=False)

    pd.DataFrame(
        [
            {
                "subject_id": "S01",
                "model": "xlmr",
                "target_language": "EN",
                "roi_family": "SEMANTIC",
                "triplet_id": "t001",
                "representative_block": 1,
                "representative_state": "FFN",
                "condition": "SHARED",
                "sentence_language": "FR",
                "word_index": 0,
                "word_text": "bonjour",
                "word_char_start": 0,
                "word_char_end": 7,
                "upos": "NOUN",
                "token_class": "CONTENT",
                "attribution_score": 0.6,
                "mapping_status": "OK",
                "n_subwords_deleted": 1,
            },
            {
                "subject_id": "S01",
                "model": "xlmr",
                "target_language": "EN",
                "roi_family": "SEMANTIC",
                "triplet_id": "t001",
                "representative_block": 1,
                "representative_state": "FFN",
                "condition": "SHARED",
                "sentence_language": "ZH",
                "word_index": 0,
                "word_text": "nihao",
                "word_char_start": 0,
                "word_char_end": 5,
                "upos": "NOUN",
                "token_class": "CONTENT",
                "attribution_score": 0.4,
                "mapping_status": "OK",
                "n_subwords_deleted": 1,
            },
            {
                "subject_id": "S02",
                "model": "nllb",
                "target_language": "FR",
                "roi_family": "AUDITORY",
                "triplet_id": "t002",
                "representative_block": 0,
                "representative_state": "POST_ATTN",
                "condition": "SHARED",
                "sentence_language": "EN",
                "word_index": 0,
                "word_text": "hello",
                "word_char_start": 0,
                "word_char_end": 5,
                "upos": "NOUN",
                "token_class": "EDGE",
                "attribution_score": 0.3,
                "mapping_status": "OK",
                "n_subwords_deleted": 1,
            },
            {
                "subject_id": "S02",
                "model": "nllb",
                "target_language": "FR",
                "roi_family": "AUDITORY",
                "triplet_id": "t002",
                "representative_block": 0,
                "representative_state": "POST_ATTN",
                "condition": "SHARED",
                "sentence_language": "ZH",
                "word_index": 0,
                "word_text": "zao",
                "word_char_start": 0,
                "word_char_end": 3,
                "upos": "ADV",
                "token_class": "FUNCTION",
                "attribution_score": 0.2,
                "mapping_status": "OK",
                "n_subwords_deleted": 1,
            },
        ]
    ).to_parquet(subject_root / "word_attributions.parquet", index=False)

    pd.DataFrame(
        [
            {"triplet_id": "t001", "en_text": "hello world", "fr_text": "bonjour monde", "zh_text": "nihao shijie"},
            {"triplet_id": "t002", "en_text": "hello again", "fr_text": "bonjour encore", "zh_text": "zao an"},
        ]
    ).to_parquet(manifests_root / "triplets.parquet", index=False)

    config = {
        "paths": {
            "local_outputs_root": outputs_root.as_posix(),
            "local_manifests_root": manifests_root.as_posix(),
        }
    }

    try:
        outputs = generate_all_figures(config)
        expected = {
            "fig01_png",
            "fig01_pdf",
            "fig02_png",
            "fig02_pdf",
            "fig03_png",
            "fig03_pdf",
            "fig04_png",
            "fig04_pdf",
            "fig05_png",
            "fig05_pdf",
            "fig06_png",
            "fig06_pdf",
            "fig07_png",
            "fig07_pdf",
            "fig08_png",
            "fig08_pdf",
            "fig09_png",
            "fig09_pdf",
        }
        assert expected.issubset(set(outputs))
        for key in expected:
            assert outputs[key].exists()
    finally:
        for path in sorted(base_path.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()
