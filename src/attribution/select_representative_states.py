from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import load_config


STATE_ORDER = ("ATTN", "POST_ATTN", "FFN", "OUTPUT")
STATE_RANK = {state: index for index, state in enumerate(STATE_ORDER)}
REPRESENTATIVE_COLUMNS = [
    "model",
    "language",
    "roi_family",
    "block",
    "state",
    "selection_metric",
    "delta_z_mean",
    "tie_break_notes",
]


def _representative_state_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("attribution", {}).get("representative_states", {}))


def _require_subject_level(config: dict[str, Any]) -> bool:
    return bool(_representative_state_config(config).get("require_subject_level", True))


def _local_outputs_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["local_outputs_root"]).resolve()


def _subject_results_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "subject_results" / "state_sweep_subject_results.parquet"


def _group_results_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "group_results" / "state_sweep_group_results.parquet"


def _representative_states_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "group_results" / "representative_states.parquet"


def _candidate_states_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "group_results" / "representative_state_candidates.parquet"


def _provenance_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "provenance" / "representative_states.md"


def _subject_family_deltas(subject_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "subject_id",
        "language",
        "model",
        "block",
        "state",
        "condition",
        "roi",
        "roi_family",
        "z_mean",
    }
    missing_columns = sorted(required_columns.difference(subject_df.columns))
    if missing_columns:
        raise KeyError(
            "State sweep subject results are missing required columns: "
            + ", ".join(missing_columns)
        )

    filtered = subject_df.loc[
        subject_df["condition"].astype(str).isin(["SHARED", "SPECIFIC"])
        & subject_df["state"].astype(str).isin(STATE_ORDER)
    ].copy()
    if filtered.empty:
        return pd.DataFrame(columns=["model", "language", "roi_family", "block", "state", "delta_z_mean", "source_level"])

    key_cols = ["subject_id", "language", "model", "block", "state", "roi", "roi_family"]
    pivot = (
        filtered.pivot_table(index=key_cols, columns="condition", values="z_mean", aggfunc="first")
        .reset_index()
        .rename_axis(columns=None)
    )
    if "SHARED" not in pivot.columns or "SPECIFIC" not in pivot.columns:
        raise RuntimeError("State sweep subject results must include both SHARED and SPECIFIC rows.")
    pivot["delta_z"] = pivot["SHARED"].astype(float) - pivot["SPECIFIC"].astype(float)
    subject_family = (
        pivot.groupby(["subject_id", "language", "model", "roi_family", "block", "state"], as_index=False)["delta_z"]
        .mean()
    )
    candidates = (
        subject_family.groupby(["model", "language", "roi_family", "block", "state"], as_index=False)["delta_z"]
        .mean()
        .rename(columns={"delta_z": "delta_z_mean"})
        .sort_values(["model", "language", "roi_family", "block", "state"])
        .reset_index(drop=True)
    )
    candidates["source_level"] = "subject"
    return candidates


def _group_family_deltas(group_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"language", "model", "block", "state", "condition", "roi_family", "mean_z"}
    missing_columns = sorted(required_columns.difference(group_df.columns))
    if missing_columns:
        raise KeyError(
            "State sweep group results are missing required columns: "
            + ", ".join(missing_columns)
        )

    filtered = group_df.loc[
        group_df["condition"].astype(str).isin(["SHARED", "SPECIFIC"])
        & group_df["state"].astype(str).isin(STATE_ORDER)
    ].copy()
    if filtered.empty:
        return pd.DataFrame(columns=["model", "language", "roi_family", "block", "state", "delta_z_mean", "source_level"])

    pivot = (
        filtered.pivot_table(
            index=["model", "language", "roi_family", "block", "state", "roi"],
            columns="condition",
            values="mean_z",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    if "SHARED" not in pivot.columns or "SPECIFIC" not in pivot.columns:
        raise RuntimeError("State sweep group results must include both SHARED and SPECIFIC rows.")
    pivot["delta_z"] = pivot["SHARED"].astype(float) - pivot["SPECIFIC"].astype(float)
    candidates = (
        pivot.groupby(["model", "language", "roi_family", "block", "state"], as_index=False)["delta_z"]
        .mean()
        .rename(columns={"delta_z": "delta_z_mean"})
        .sort_values(["model", "language", "roi_family", "block", "state"])
        .reset_index(drop=True)
    )
    candidates["source_level"] = "group"
    return candidates


def _load_candidate_table(
    config: dict[str, Any],
    *,
    require_subject_level: bool,
) -> tuple[pd.DataFrame, str]:
    subject_path = _subject_results_path(config)
    if subject_path.exists():
        return _subject_family_deltas(pd.read_parquet(subject_path).copy()), "subject"

    if require_subject_level:
        raise FileNotFoundError(
            f"Representative-state selection requires subject-level state sweep results in full-run mode: {subject_path}"
        )

    group_path = _group_results_path(config)
    if group_path.exists():
        return _group_family_deltas(pd.read_parquet(group_path).copy()), "group"

    raise FileNotFoundError(
        "Representative-state selection requires either state_sweep_subject_results.parquet "
        "or state_sweep_group_results.parquet."
    )


def _choose_representatives(candidate_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_df.empty:
        raise RuntimeError("No representative-state candidates were available for selection.")

    ranked = candidate_df.copy()
    ranked["state_rank"] = ranked["state"].astype(str).map(STATE_RANK).fillna(len(STATE_RANK)).astype(int)
    selected = (
        ranked.sort_values(
            ["model", "language", "roi_family", "delta_z_mean", "block", "state_rank"],
            ascending=[True, True, True, False, True, True],
        )
        .groupby(["model", "language", "roi_family"], as_index=False)
        .first()
        .reset_index(drop=True)
    )
    selected["selection_metric"] = "max_mean_subject_shared_minus_specific"
    selected["tie_break_notes"] = "ties broken toward shallower block then state order ATTN, POST_ATTN, FFN, OUTPUT"
    return selected.loc[:, REPRESENTATIVE_COLUMNS]


def _write_markdown(
    *,
    source_level: str,
    candidate_path: Path,
    representative_path: Path,
    representatives: pd.DataFrame,
) -> str:
    lines = [
        "# Representative States",
        "",
        f"- source_level: `{source_level}`",
        f"- candidate_table: `{candidate_path}`",
        f"- representative_states: `{representative_path}`",
        "",
        "## Selected representative settings",
        "",
        representatives.to_string(index=False),
        "",
    ]
    if source_level != "subject":
        lines.extend(
            [
                "## Fallback",
                "",
                "- subject-level state sweep results were unavailable, so selection fell back to group-level deltas.",
                "",
            ]
        )
    return "\n".join(lines)


def select_representative_states(
    config: dict[str, Any],
    *,
    require_subject_level: bool | None = None,
) -> dict[str, Path]:
    candidate_df, source_level = _load_candidate_table(
        config,
        require_subject_level=_require_subject_level(config) if require_subject_level is None else bool(require_subject_level),
    )
    representatives = _choose_representatives(candidate_df)

    candidate_path = _candidate_states_path(config)
    representative_path = _representative_states_path(config)
    provenance_path = _provenance_path(config)
    for path in (candidate_path, representative_path, provenance_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    candidate_df.to_parquet(candidate_path, index=False)
    representatives.to_parquet(representative_path, index=False)
    provenance_path.write_text(
        _write_markdown(
            source_level=source_level,
            candidate_path=candidate_path,
            representative_path=representative_path,
            representatives=representatives,
        ),
        encoding="utf-8",
    )

    return {
        "candidates": candidate_path,
        "representative_states": representative_path,
        "provenance": provenance_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select representative block/state settings for attribution and deletion.")
    parser.add_argument("--config", default="conf/base.yaml", help="Path to sequel config.")
    parser.add_argument(
        "--allow-group-fallback",
        action="store_true",
        help="Permit fallback to group-level sweep results when subject-level results are unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    outputs = select_representative_states(config, require_subject_level=not args.allow_group_fallback)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
