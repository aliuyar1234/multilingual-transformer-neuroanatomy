from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from src.attribution.compute_word_attributions import LANGUAGE_TEXT_COLUMNS
from src.utils.config import load_config


STATE_DISPLAY = {
    "INPUT": "Input",
    "ATTN": "Attn",
    "POST_ATTN": "Post-Attn",
    "FFN": "FFN",
    "OUTPUT": "Output",
}
HYPOTHESIS_DISPLAY = {
    "H1_semantic_ffn_preference": "H1 Semantic FFN Preference",
    "H2_auditory_attention_preference": "H2 Auditory Attention Preference",
}
ROI_STEMS = {"AG", "IFGtri", "pSTG", "Heschl"}


def _multilingual_text_font_family() -> list[str]:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ["SimSun", "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]:
        if candidate in available:
            return [candidate, "DejaVu Sans"]
    return ["DejaVu Sans"]


def _local_outputs_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["local_outputs_root"]).resolve()


def _local_manifests_root(config: dict[str, Any]) -> Path | None:
    root = config.get("paths", {}).get("local_manifests_root")
    return Path(root).resolve() if root else None


def _figures_root(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "figures"


def _group_results_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "group_results" / "state_sweep_group_results.parquet"


def _refresh_results_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "group_results" / "refresh_output_results.parquet"


def _refresh_family_fallback_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "group_results" / "output_layer_family_deltas.parquet"


def _primary_effect_tables_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "group_results" / "primary_effect_tables.parquet"


def _primary_subject_effects_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "group_results" / "primary_subject_effects.parquet"


def _representative_states_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "group_results" / "representative_states.parquet"


def _word_attributions_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "subject_results" / "word_attributions.parquet"


def _triplets_path(config: dict[str, Any]) -> Path | None:
    root = _local_manifests_root(config)
    return None if root is None else root / "triplets.parquet"


def _table05_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "tables" / "table05_token_class_attribution.csv"


def _table06_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "tables" / "table06_deletion_validation.csv"


def _save_figure(fig: plt.Figure, path_png: Path) -> dict[str, Path]:
    path_pdf = path_png.with_suffix(".pdf")
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=200, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": path_png, "pdf": path_pdf}


def _roi_stem(roi_name: str) -> str:
    roi_name = str(roi_name)
    if roi_name.startswith(("L_", "R_")):
        return roi_name.split("_", 1)[1]
    return roi_name


def _delta_group_results(group_df: pd.DataFrame) -> pd.DataFrame:
    filtered = group_df.loc[group_df["condition"].astype(str).isin(["SHARED", "SPECIFIC"])].copy()
    pivot = (
        filtered.pivot_table(
            index=["language", "model", "roi", "roi_family", "block", "block_depth_norm", "state"],
            columns="condition",
            values="mean_z",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    pivot["delta_z_mean"] = pivot["SHARED"].astype(float) - pivot["SPECIFIC"].astype(float)
    pivot["roi_stem"] = pivot["roi"].astype(str).map(_roi_stem)
    return pivot


def _generate_fig01_pipeline_bridge(config: dict[str, Any]) -> dict[str, Path]:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    panels = [
        ("A", "Prior Paper", "Triplets\n-> Hidden States\n-> Shared/Specific\n-> ROI Encoding"),
        ("B", "Sequel", "Triplets\n-> Internal States\n-> Shared/Specific by State\n-> ROI Encoding\n-> Attribution\n-> Deletion"),
        ("C", "Hypotheses", "Semantic: FFN-side stronger\nAuditory: ATTN-side stronger"),
    ]
    for axis, (label, title, body) in zip(axes, panels, strict=True):
        axis.axis("off")
        axis.text(0.02, 0.95, label, fontsize=14, fontweight="bold", va="top")
        axis.text(0.5, 0.78, title, fontsize=13, fontweight="bold", ha="center")
        axis.text(
            0.5,
            0.42,
            body,
            fontsize=11,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f6f0e8", "edgecolor": "#c8b8a6"},
        )
    fig.suptitle("From Prior Baseline to Mechanistic Sequel", fontsize=15, fontweight="bold")
    return _save_figure(fig, _figures_root(config) / "fig01_pipeline_bridge.png")


def _refresh_delta_frame(config: dict[str, Any]) -> pd.DataFrame | None:
    refresh_path = _refresh_results_path(config)
    fallback_path = _refresh_family_fallback_path(config)
    if refresh_path.exists():
        df = pd.read_parquet(refresh_path).copy()
        if {"condition", "roi_family"}.issubset(df.columns):
            value_column = "mean_z" if "mean_z" in df.columns else "z_mean"
            if value_column in df.columns:
                filtered = df.loc[df["condition"].astype(str).isin(["SHARED", "SPECIFIC"])].copy()
                pivot = (
                    filtered.groupby(["model", "language", "roi_family", "condition"], as_index=False)[value_column]
                    .mean()
                    .pivot_table(
                        index=["model", "language", "roi_family"],
                        columns="condition",
                        values=value_column,
                        aggfunc="first",
                    )
                    .reset_index()
                    .rename_axis(columns=None)
                )
                pivot["delta_z_mean"] = pivot["SHARED"].astype(float) - pivot["SPECIFIC"].astype(float)
                return pivot.loc[:, ["model", "language", "roi_family", "delta_z_mean"]]
        if "delta_z_mean" in df.columns:
            return (
                df.groupby(["model", "language", "roi_family"], as_index=False)["delta_z_mean"]
                .mean()
                .sort_values(["model", "language", "roi_family"])
                .reset_index(drop=True)
            )
    if fallback_path.exists():
        df = pd.read_parquet(fallback_path).copy()
        if "shared_minus_specific" in df.columns:
            summary = (
                df.groupby(["model", "language", "roi_family"], as_index=False)["shared_minus_specific"]
                .mean()
                .rename(columns={"shared_minus_specific": "delta_z_mean"})
                .sort_values(["model", "language", "roi_family"])
                .reset_index(drop=True)
            )
            return summary
    return None


def _generate_fig02_refresh_baseline(config: dict[str, Any]) -> dict[str, Path] | None:
    delta_df = _refresh_delta_frame(config)
    if delta_df is None or delta_df.empty:
        return None
    families = ["SEMANTIC", "AUDITORY"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, family in zip(axes, families, strict=True):
        subset = delta_df.loc[delta_df["roi_family"].astype(str) == family].copy()
        if subset.empty:
            axis.axis("off")
            continue
        subset["label"] = subset["model"].astype(str).str.upper() + " " + subset["language"].astype(str)
        subset = subset.sort_values(["model", "language"]).reset_index(drop=True)
        colors = ["#c2633f" if model == "xlmr" else "#2a6f8f" for model in subset["model"].astype(str)]
        axis.bar(subset["label"], subset["delta_z_mean"].astype(float).to_numpy(), color=colors)
        axis.axhline(0.0, color="#888888", linewidth=1.0, linestyle="--")
        axis.set_title(f"{family.title()} Family", fontsize=11, fontweight="bold")
        axis.set_ylabel("Shared - Specific")
        axis.tick_params(axis="x", labelrotation=45)
    fig.suptitle("Refreshed OUTPUT-State Bridge Baseline", fontsize=15, fontweight="bold")
    return _save_figure(fig, _figures_root(config) / "fig02_refresh_baseline.png")


def _generate_fig03_state_depth_heatmaps(config: dict[str, Any]) -> dict[str, Path] | None:
    path = _group_results_path(config)
    if not path.exists():
        return None
    delta_df = _delta_group_results(pd.read_parquet(path).copy())
    families = ["SEMANTIC", "AUDITORY"]
    models = sorted(delta_df["model"].astype(str).unique().tolist())
    languages = sorted(delta_df["language"].astype(str).unique().tolist())
    fig, axes = plt.subplots(len(models) * len(families), len(languages), figsize=(4 * len(languages), 2.6 * len(models) * len(families)))
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([[axes]])
    axes = np.atleast_2d(axes)
    state_order = ["INPUT", "ATTN", "POST_ATTN", "FFN", "OUTPUT"]
    vmax = float(np.nanmax(np.abs(delta_df["delta_z_mean"].to_numpy(dtype=float)))) if not delta_df.empty else 1.0
    vmax = max(vmax, 1.0e-6)
    for model_index, model in enumerate(models):
        for family_index, family in enumerate(families):
            row_index = model_index * len(families) + family_index
            for col_index, language in enumerate(languages):
                axis = axes[row_index, col_index]
                subset = delta_df.loc[
                    (delta_df["model"].astype(str) == model)
                    & (delta_df["language"].astype(str) == language)
                    & (delta_df["roi_family"].astype(str) == family)
                ].copy()
                heatmap = np.full((len(state_order), max(1, subset["block"].astype(int).max() + 1 if not subset.empty else 1)), np.nan)
                for row in subset.itertuples(index=False):
                    heatmap[state_order.index(str(row.state)), int(row.block)] = float(row.delta_z_mean)
                image = axis.imshow(heatmap, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
                axis.set_title(f"{model.upper()} {language} {family.title()}", fontsize=10)
                axis.set_yticks(range(len(state_order)))
                axis.set_yticklabels([STATE_DISPLAY.get(state, state) for state in state_order], fontsize=8)
                axis.set_xlabel("Block", fontsize=8)
                axis.set_xticks(range(heatmap.shape[1]))
                axis.tick_params(axis="x", labelsize=7)
        fig.colorbar(image, ax=axes[:, :], shrink=0.6, label="Shared - Specific (mean z)")
    fig.suptitle("State-by-Depth Shared Advantage", fontsize=15, fontweight="bold")
    return _save_figure(fig, _figures_root(config) / "fig03_state_depth_heatmaps.png")


def _generate_fig04_primary_state_family_tests(config: dict[str, Any]) -> dict[str, Path] | None:
    summary_path = _primary_effect_tables_path(config)
    if not summary_path.exists():
        return None
    summary_df = pd.read_parquet(summary_path).copy()
    subject_path = _primary_subject_effects_path(config)
    subject_df = pd.read_parquet(subject_path).copy() if subject_path.exists() else pd.DataFrame()
    hypotheses = ["H1_semantic_ffn_preference", "H2_auditory_attention_preference"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for axis, hypothesis in zip(axes, hypotheses, strict=True):
        subset = summary_df.loc[summary_df["hypothesis"].astype(str) == hypothesis].copy().reset_index(drop=True)
        positions = np.arange(len(subset))
        axis.axvline(0.0, color="#888888", linewidth=1.0, linestyle="--")
        for index, row in enumerate(subset.itertuples(index=False)):
            if not subject_df.empty:
                subject_subset = subject_df.loc[
                    (subject_df["hypothesis"].astype(str) == hypothesis)
                    & (subject_df["model"].astype(str) == str(row.model))
                    & (subject_df["language"].astype(str) == str(row.language))
                ]
                axis.scatter(
                    subject_subset["effect_value"].astype(float).to_numpy(),
                    np.full(len(subject_subset), positions[index], dtype=float),
                    alpha=0.35,
                    color="#8c8c8c",
                    s=14,
                )
            axis.errorbar(
                x=float(row.effect_mean),
                y=positions[index],
                xerr=np.array([[float(row.effect_mean) - float(row.ci_low)], [float(row.ci_high) - float(row.effect_mean)]]),
                fmt="o",
                color="#c2633f" if "H1" in hypothesis else "#2a6f8f",
                capsize=3,
            )
            if float(row.p_holm) < 0.05:
                axis.text(float(row.ci_high) + 0.01, positions[index], "*", va="center", fontsize=12, fontweight="bold")
        axis.set_yticks(positions)
        axis.set_yticklabels([f"{row.model.upper()} {row.language}" for row in subset.itertuples(index=False)], fontsize=9)
        axis.set_title(HYPOTHESIS_DISPLAY[hypothesis], fontsize=11, fontweight="bold")
        axis.set_xlabel("Effect Mean", fontsize=9)
    fig.suptitle("Primary Confirmatory State-Family Tests", fontsize=15, fontweight="bold")
    return _save_figure(fig, _figures_root(config) / "fig04_primary_state_family_tests.png")


def _generate_fig05_representative_roi_curves(config: dict[str, Any]) -> dict[str, Path] | None:
    path = _group_results_path(config)
    if not path.exists():
        return None
    delta_df = _delta_group_results(pd.read_parquet(path).copy())
    delta_df = delta_df.loc[delta_df["roi_stem"].astype(str).isin(ROI_STEMS)].copy()
    if delta_df.empty:
        return None
    rois = ["AG", "IFGtri", "pSTG", "Heschl"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=False)
    state_order = ["ATTN", "POST_ATTN", "FFN", "OUTPUT"]
    color_map = {"ATTN": "#2a6f8f", "POST_ATTN": "#5f9ea0", "FFN": "#c2633f", "OUTPUT": "#d9a441"}
    for axis, roi in zip(axes.flatten(), rois, strict=True):
        subset = delta_df.loc[delta_df["roi_stem"].astype(str) == roi].copy()
        summary = (
            subset.groupby(["state", "block_depth_norm"], as_index=False)["delta_z_mean"]
            .mean()
            .sort_values(["state", "block_depth_norm"])
        )
        for state in state_order:
            state_df = summary.loc[summary["state"].astype(str) == state]
            if state_df.empty:
                continue
            axis.plot(
                state_df["block_depth_norm"].astype(float).to_numpy(),
                state_df["delta_z_mean"].astype(float).to_numpy(),
                label=STATE_DISPLAY.get(state, state),
                color=color_map[state],
                linewidth=2,
            )
        axis.axhline(0.0, color="#888888", linewidth=1.0, linestyle="--")
        axis.set_title(roi, fontsize=11, fontweight="bold")
        axis.set_xlabel("Normalized Depth", fontsize=9)
        axis.set_ylabel("Shared - Specific", fontsize=9)
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Representative ROI Curves Across Depth", fontsize=15, fontweight="bold", y=0.98)
    return _save_figure(fig, _figures_root(config) / "fig05_representative_roi_curves.png")


def _format_attribution_sentence(sentence_df: pd.DataFrame, *, original_text: str) -> str:
    if sentence_df.empty:
        return original_text
    ordered = sentence_df.sort_values(["word_char_start", "word_index"]).reset_index(drop=True)
    positive = ordered["attribution_score"].astype(float).clip(lower=0.0)
    max_positive = float(positive.max()) if len(positive) else 0.0
    tokens: list[str] = []
    for row in ordered.itertuples(index=False):
        magnitude = max(float(row.attribution_score), 0.0)
        if max_positive <= 0.0 or magnitude <= 0.0:
            marker = "-"
        elif magnitude >= 0.75 * max_positive:
            marker = "+++"
        elif magnitude >= 0.40 * max_positive:
            marker = "++"
        else:
            marker = "+"
        tokens.append(f"{row.word_text}({marker})")
    return f"{original_text}\n" + " ".join(tokens)


def _generate_fig06_multilingual_token_attribution_examples(config: dict[str, Any]) -> dict[str, Path] | None:
    word_path = _word_attributions_path(config)
    reps_path = _representative_states_path(config)
    triplets_path = _triplets_path(config)
    if not word_path.exists() or not reps_path.exists() or triplets_path is None or not triplets_path.exists():
        return None
    word_df = pd.read_parquet(word_path).copy()
    reps_df = pd.read_parquet(reps_path).copy()
    triplets = pd.read_parquet(triplets_path).copy()
    if word_df.empty or reps_df.empty or triplets.empty:
        return None
    merged = word_df.merge(
        reps_df.rename(columns={"language": "target_language", "block": "representative_block", "state": "representative_state"}),
        on=["model", "target_language", "roi_family", "representative_block", "representative_state"],
        how="inner",
    )
    if merged.empty:
        return None
    merged["positive_score"] = merged["attribution_score"].astype(float).clip(lower=0.0)
    candidate_examples = (
        merged.groupby(
            ["roi_family", "model", "target_language", "triplet_id", "representative_block", "representative_state"],
            as_index=False,
        )["positive_score"]
        .mean()
        .sort_values(["roi_family", "positive_score"], ascending=[True, False])
    )
    families = [family for family in ["SEMANTIC", "AUDITORY"] if family in set(candidate_examples["roi_family"].astype(str))]
    if not families:
        return None
    fig, axes = plt.subplots(len(families), 1, figsize=(12, 3.8 * len(families)))
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    triplet_lookup = triplets.assign(triplet_id_key=triplets["triplet_id"].astype(str)).set_index("triplet_id_key", drop=False)
    for axis, family in zip(np.atleast_1d(axes), families, strict=True):
        axis.axis("off")
        selected = candidate_examples.loc[candidate_examples["roi_family"].astype(str) == family].iloc[0]
        example_rows = merged.loc[
            (merged["roi_family"].astype(str) == family)
            & (merged["model"].astype(str) == str(selected.model))
            & (merged["target_language"].astype(str) == str(selected.target_language))
            & (merged["triplet_id"].astype(str) == str(selected.triplet_id))
        ].copy()
        triplet_row = triplet_lookup.loc[str(selected.triplet_id)]
        if isinstance(triplet_row, pd.DataFrame):
            triplet_row = triplet_row.iloc[0]
        header = (
            f"{family.title()} | {str(selected.model).upper()} {str(selected.target_language)} | "
            f"Block {int(selected.representative_block)} {str(selected.representative_state)}"
        )
        lines = [header, ""]
        for sentence_language, sentence_rows in example_rows.groupby("sentence_language", sort=True):
            original_text = str(triplet_row[LANGUAGE_TEXT_COLUMNS[str(sentence_language)]])
            lines.append(f"{sentence_language}: {_format_attribution_sentence(sentence_rows, original_text=original_text)}")
            lines.append("")
        axis.text(
            0.02,
            0.98,
            "\n".join(lines).strip(),
            va="top",
            ha="left",
            fontsize=10,
            fontfamily=_multilingual_text_font_family(),
            bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f7f3ea", "edgecolor": "#c8b8a6"},
        )
    fig.suptitle("Representative Multilingual Token-Attribution Examples", fontsize=15, fontweight="bold")
    return _save_figure(fig, _figures_root(config) / "fig06_multilingual_token_attribution_examples.png")


def _generate_fig07_token_class_mass(config: dict[str, Any]) -> dict[str, Path] | None:
    table_path = _table05_path(config)
    if not table_path.exists():
        return None
    df = pd.read_csv(table_path).copy()
    if df.empty:
        return None
    classes = ["CONTENT", "FUNCTION", "PUNCT", "EDGE"]
    summary = (
        df.groupby(["roi_family", "class"], as_index=False)["positive_mass"]
        .mean()
        .sort_values(["roi_family", "class"])
    )
    families = sorted(summary["roi_family"].astype(str).unique().tolist())
    x = np.arange(len(classes))
    width = 0.35
    fig, axis = plt.subplots(figsize=(9, 5))
    for offset, family in enumerate(families):
        family_df = summary.loc[summary["roi_family"].astype(str) == family].copy()
        values = [float(family_df.loc[family_df["class"].astype(str) == cls, "positive_mass"].mean()) if cls in set(family_df["class"].astype(str)) else 0.0 for cls in classes]
        axis.bar(x + (offset - (len(families) - 1) / 2.0) * width, values, width=width, label=family.title())
    axis.set_xticks(x)
    axis.set_xticklabels(classes)
    axis.set_ylabel("Positive Attribution Mass")
    axis.set_title("Token-Class Attribution Mass", fontsize=14, fontweight="bold")
    axis.legend(frameon=False)
    return _save_figure(fig, _figures_root(config) / "fig07_token_class_mass.png")


def _generate_fig08_deletion_validation(config: dict[str, Any]) -> dict[str, Path] | None:
    table_path = _table06_path(config)
    if not table_path.exists():
        return None
    df = pd.read_csv(table_path).copy()
    if df.empty:
        return None
    fig, axis = plt.subplots(figsize=(8, 5))
    for roi_family, family_df in df.groupby("roi_family", sort=True):
        family_df = family_df.sort_values("k")
        axis.plot(family_df["k"], family_df["effect_topk"], marker="o", label=f"{roi_family.title()} top-k")
        axis.plot(family_df["k"], family_df["effect_random"], marker="s", linestyle="--", label=f"{roi_family.title()} random")
    axis.set_xlabel("k Deleted Words")
    axis.set_ylabel("Drop in Held-Out Score")
    axis.set_title("Deletion Validation", fontsize=14, fontweight="bold")
    axis.legend(frameon=False, ncol=2)
    return _save_figure(fig, _figures_root(config) / "fig08_deletion_validation.png")


def _generate_fig09_roi_preference_map(config: dict[str, Any]) -> dict[str, Path] | None:
    path = _group_results_path(config)
    if not path.exists():
        return None
    delta_df = _delta_group_results(pd.read_parquet(path).copy())
    if delta_df.empty:
        return None
    summary = (
        delta_df.groupby(["roi_family", "roi"], as_index=False)["delta_z_mean"]
        .mean()
        .sort_values(["roi_family", "delta_z_mean"], ascending=[True, False])
    )
    families = sorted(summary["roi_family"].astype(str).unique().tolist())
    fig, axes = plt.subplots(1, len(families), figsize=(6 * max(1, len(families)), 5), sharex=False)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    for axis, family in zip(np.atleast_1d(axes), families, strict=True):
        family_df = summary.loc[summary["roi_family"].astype(str) == family].copy().sort_values("delta_z_mean")
        axis.barh(family_df["roi"].astype(str), family_df["delta_z_mean"].astype(float), color="#c2633f" if family == "SEMANTIC" else "#2a6f8f")
        axis.axvline(0.0, color="#888888", linewidth=1.0, linestyle="--")
        axis.set_title(family.title(), fontsize=11, fontweight="bold")
        axis.set_xlabel("Shared - Specific")
    fig.suptitle("Descriptive ROI Preference Map", fontsize=15, fontweight="bold")
    return _save_figure(fig, _figures_root(config) / "fig09_roi_preference_map.png")


def generate_all_figures(config: dict[str, Any]) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    fig01 = _generate_fig01_pipeline_bridge(config)
    outputs["fig01_png"] = fig01["png"]
    outputs["fig01_pdf"] = fig01["pdf"]
    for key, generator in (
        ("fig02", _generate_fig02_refresh_baseline),
        ("fig03", _generate_fig03_state_depth_heatmaps),
        ("fig04", _generate_fig04_primary_state_family_tests),
        ("fig05", _generate_fig05_representative_roi_curves),
        ("fig06", _generate_fig06_multilingual_token_attribution_examples),
        ("fig07", _generate_fig07_token_class_mass),
        ("fig08", _generate_fig08_deletion_validation),
        ("fig09", _generate_fig09_roi_preference_map),
    ):
        figure_paths = generator(config)
        if figure_paths is None:
            continue
        outputs[f"{key}_png"] = figure_paths["png"]
        outputs[f"{key}_pdf"] = figure_paths["pdf"]
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate all currently available sequel figures.")
    parser.add_argument("--config", default="conf/base.yaml", help="Path to sequel config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    outputs = generate_all_figures(config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
