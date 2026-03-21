from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.attribution.select_representative_states import select_representative_states
from tests.helpers import make_temp_test_dir


def _synthetic_subject_results() -> pd.DataFrame:
    semantic_delta = {
        (0, "ATTN"): 0.2,
        (0, "POST_ATTN"): 0.3,
        (0, "FFN"): 0.8,
        (0, "OUTPUT"): 0.8,
        (1, "ATTN"): 0.4,
        (1, "POST_ATTN"): 0.4,
        (1, "FFN"): 0.4,
        (1, "OUTPUT"): 0.4,
    }
    auditory_delta = {
        (0, "ATTN"): 0.5,
        (0, "POST_ATTN"): 0.7,
        (0, "FFN"): 0.2,
        (0, "OUTPUT"): 0.1,
        (1, "ATTN"): 0.7,
        (1, "POST_ATTN"): 0.6,
        (1, "FFN"): 0.3,
        (1, "OUTPUT"): 0.2,
    }

    rows: list[dict[str, object]] = []
    for subject_id in ("S01", "S02"):
        base = 0.25 if subject_id == "S01" else 0.35
        for block in (0, 1):
            for state in ("ATTN", "POST_ATTN", "FFN", "OUTPUT"):
                for roi, roi_family, delta_lookup in (
                    ("AG", "SEMANTIC", semantic_delta),
                    ("pSTG", "AUDITORY", auditory_delta),
                ):
                    delta = delta_lookup[(block, state)]
                    rows.append(
                        {
                            "subject_id": subject_id,
                            "language": "EN",
                            "model": "xlmr",
                            "block": block,
                            "state": state,
                            "condition": "SHARED",
                            "roi": roi,
                            "roi_family": roi_family,
                            "z_mean": base + (delta / 2.0),
                        }
                    )
                    rows.append(
                        {
                            "subject_id": subject_id,
                            "language": "EN",
                            "model": "xlmr",
                            "block": block,
                            "state": state,
                            "condition": "SPECIFIC",
                            "roi": roi,
                            "roi_family": roi_family,
                            "z_mean": base - (delta / 2.0),
                        }
                    )
    return pd.DataFrame(rows)


def _synthetic_group_results() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for roi_family, roi in (("SEMANTIC", "AG"), ("AUDITORY", "pSTG")):
        for condition, mean_z in (("SHARED", 0.6), ("SPECIFIC", 0.2)):
            rows.append(
                {
                    "language": "EN",
                    "model": "xlmr",
                    "block": 0,
                    "state": "FFN" if roi_family == "SEMANTIC" else "POST_ATTN",
                    "condition": condition,
                    "roi": roi,
                    "roi_family": roi_family,
                    "mean_z": mean_z,
                }
            )
    return pd.DataFrame(rows)


def test_select_representative_states_prefers_shallower_then_state_order() -> None:
    base_path = make_temp_test_dir("select_representative_states")
    outputs_root = base_path / "outputs"
    subject_results_path = outputs_root / "subject_results" / "state_sweep_subject_results.parquet"
    subject_results_path.parent.mkdir(parents=True, exist_ok=True)
    _synthetic_subject_results().to_parquet(subject_results_path, index=False)

    config = {"paths": {"local_outputs_root": outputs_root.as_posix()}}

    try:
        outputs = select_representative_states(config)
        candidate_path = outputs["candidates"]
        representative_path = outputs["representative_states"]
        provenance_path = outputs["provenance"]

        assert candidate_path.exists()
        assert representative_path.exists()
        assert provenance_path.exists()

        representative_df = pd.read_parquet(representative_path).sort_values("roi_family").reset_index(drop=True)
        assert len(representative_df) == 2
        assert set(representative_df["roi_family"]) == {"AUDITORY", "SEMANTIC"}

        semantic_row = representative_df.loc[representative_df["roi_family"] == "SEMANTIC"].iloc[0]
        auditory_row = representative_df.loc[representative_df["roi_family"] == "AUDITORY"].iloc[0]

        assert int(semantic_row["block"]) == 0
        assert str(semantic_row["state"]) == "FFN"
        assert float(semantic_row["delta_z_mean"]) == 0.8

        assert int(auditory_row["block"]) == 0
        assert str(auditory_row["state"]) == "POST_ATTN"
        assert float(auditory_row["delta_z_mean"]) == 0.7

        provenance = provenance_path.read_text(encoding="utf-8")
        assert "source_level: `subject`" in provenance
        assert "ties broken toward shallower block then state order" in provenance
    finally:
        for path in sorted(base_path.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()


def test_select_representative_states_requires_subject_results_by_default() -> None:
    base_path = make_temp_test_dir("select_representative_states_require_subject")
    outputs_root = base_path / "outputs"
    group_results_path = outputs_root / "group_results" / "state_sweep_group_results.parquet"
    group_results_path.parent.mkdir(parents=True, exist_ok=True)
    _synthetic_group_results().to_parquet(group_results_path, index=False)

    config = {"paths": {"local_outputs_root": outputs_root.as_posix()}}

    try:
        try:
            select_representative_states(config)
        except FileNotFoundError as exc:
            message = str(exc)
        else:
            raise AssertionError("Expected strict representative-state selection to require subject-level results.")

        assert "subject-level state sweep results" in message
    finally:
        for path in sorted(base_path.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()
