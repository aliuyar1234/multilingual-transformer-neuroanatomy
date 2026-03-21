from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import load_config


REPRESENTATIVE_ROI_STEMS = {
    "SEMANTIC": ("AG", "IFGtri"),
    "AUDITORY": ("pSTG", "Heschl"),
}


def _local_outputs_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["local_outputs_root"]).resolve()


def _group_results_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "group_results" / "state_sweep_group_results.parquet"


def _representative_states_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "group_results" / "representative_states.parquet"


def _word_attributions_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "subject_results" / "word_attributions.parquet"


def _deletion_validation_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "subject_results" / "deletion_validation.parquet"


def _table_path(config: dict[str, Any], filename: str) -> Path:
    return _local_outputs_root(config) / "tables" / filename


def _roi_stem(roi_name: str) -> str:
    roi_name = str(roi_name)
    if roi_name.startswith(("L_", "R_")):
        return roi_name.split("_", 1)[1]
    return roi_name


def build_table04_representative_roi_summaries(config: dict[str, Any]) -> Path:
    representative_path = _representative_states_path(config)
    group_path = _group_results_path(config)
    if not representative_path.exists():
        raise FileNotFoundError(
            f"Representative states missing: {representative_path}. Run src.attribution.select_representative_states first."
        )
    if not group_path.exists():
        raise FileNotFoundError(
            f"State sweep group results missing: {group_path}. Run src.encoding.run_state_sweep first."
        )

    representatives = pd.read_parquet(representative_path).copy()
    group_df = pd.read_parquet(group_path).copy()
    filtered = group_df.loc[group_df["condition"].astype(str).isin(["SHARED", "SPECIFIC"])].copy()
    filtered["roi_stem"] = filtered["roi"].astype(str).map(_roi_stem)

    rows: list[dict[str, object]] = []
    for rep in representatives.itertuples(index=False):
        roi_stems = REPRESENTATIVE_ROI_STEMS.get(str(rep.roi_family).upper(), ())
        rep_slice = filtered.loc[
            (filtered["model"].astype(str) == str(rep.model))
            & (filtered["language"].astype(str) == str(rep.language))
            & (filtered["roi_family"].astype(str) == str(rep.roi_family))
            & (filtered["block"].astype(int) == int(rep.block))
            & (filtered["state"].astype(str) == str(rep.state))
            & (filtered["roi_stem"].astype(str).isin(roi_stems))
        ].copy()
        if rep_slice.empty:
            continue
        pivot = (
            rep_slice.groupby(["roi_stem", "condition"], as_index=False)["mean_z"]
            .mean()
            .pivot_table(index="roi_stem", columns="condition", values="mean_z", aggfunc="first")
            .reset_index()
            .rename_axis(columns=None)
        )
        for roi_row in pivot.itertuples(index=False):
            shared = float(getattr(roi_row, "SHARED", np.nan))
            specific = float(getattr(roi_row, "SPECIFIC", np.nan))
            rows.append(
                {
                    "model": str(rep.model),
                    "language": str(rep.language),
                    "roi_family": str(rep.roi_family),
                    "representative_block": int(rep.block),
                    "representative_state": str(rep.state),
                    "roi": str(roi_row.roi_stem),
                    "shared_mean_z": shared,
                    "specific_mean_z": specific,
                    "delta_z_mean": shared - specific,
                }
            )

    table04_path = _table_path(config, "table04_representative_roi_summaries.csv")
    table04_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(
        ["model", "language", "roi_family", "roi"]
    ).reset_index(drop=True).to_csv(table04_path, index=False)
    return table04_path


def build_table05_token_class_attribution(config: dict[str, Any]) -> Path:
    attribution_path = _word_attributions_path(config)
    if not attribution_path.exists():
        raise FileNotFoundError(
            f"Word attributions missing: {attribution_path}. Run src.attribution.compute_word_attributions first."
        )

    df = pd.read_parquet(attribution_path).copy()
    language_column = "language" if "language" in df.columns else "target_language"
    class_column = "class" if "class" in df.columns else "token_class"
    df["positive_mass"] = df["attribution_score"].astype(float).clip(lower=0.0)
    summary = (
        df.groupby(["model", language_column, "roi_family", "condition", class_column], as_index=False)["positive_mass"]
        .sum()
        .rename(columns={language_column: "language", class_column: "class"})
    )
    totals = (
        summary.groupby(["model", "language", "roi_family", "condition"], as_index=False)["positive_mass"]
        .sum()
        .rename(columns={"positive_mass": "positive_mass_total"})
    )
    content = (
        summary.loc[summary["class"].astype(str) == "CONTENT", ["model", "language", "roi_family", "condition", "positive_mass"]]
        .rename(columns={"positive_mass": "content_mass"})
    )
    summary = summary.merge(totals, on=["model", "language", "roi_family", "condition"], how="left")
    summary = summary.merge(content, on=["model", "language", "roi_family", "condition"], how="left")
    summary["content_bias"] = np.where(
        summary["positive_mass_total"].astype(float) > 0.0,
        summary["content_mass"].fillna(0.0).astype(float) / summary["positive_mass_total"].astype(float),
        0.0,
    )
    summary = summary.loc[:, ["model", "language", "roi_family", "condition", "class", "positive_mass", "content_bias"]]

    table05_path = _table_path(config, "table05_token_class_attribution.csv")
    table05_path.parent.mkdir(parents=True, exist_ok=True)
    summary.sort_values(["model", "language", "roi_family", "condition", "class"]).reset_index(drop=True).to_csv(
        table05_path,
        index=False,
    )
    return table05_path


def build_table06_deletion_validation(config: dict[str, Any]) -> Path:
    deletion_path = _deletion_validation_path(config)
    if not deletion_path.exists():
        raise FileNotFoundError(
            f"Deletion validation missing: {deletion_path}. Run src.attribution.run_deletion_validation first."
        )

    df = pd.read_parquet(deletion_path).copy()
    summary = (
        df.groupby(["model", "language", "roi_family", "k"], as_index=False)
        .agg(
            effect_topk=("delta_topk", "mean"),
            effect_random=("delta_random", "mean"),
            diff=("delta_diff", "mean"),
            ci_low=("delta_diff", lambda series: float(np.quantile(series, 0.025))),
            ci_high=("delta_diff", lambda series: float(np.quantile(series, 0.975))),
        )
        .sort_values(["model", "language", "roi_family", "k"])
        .reset_index(drop=True)
    )

    table06_path = _table_path(config, "table06_deletion_validation.csv")
    table06_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(table06_path, index=False)
    return table06_path


def generate_all_tables(config: dict[str, Any]) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    table03_path = _table_path(config, "table03_primary_confirmatory_stats.csv")
    if table03_path.exists():
        outputs["table03"] = table03_path
    if _representative_states_path(config).exists() and _group_results_path(config).exists():
        outputs["table04"] = build_table04_representative_roi_summaries(config)
    if _word_attributions_path(config).exists():
        outputs["table05"] = build_table05_token_class_attribution(config)
    if _deletion_validation_path(config).exists():
        outputs["table06"] = build_table06_deletion_validation(config)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate all currently available sequel tables.")
    parser.add_argument("--config", default="conf/base.yaml", help="Path to sequel config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    outputs = generate_all_tables(config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
