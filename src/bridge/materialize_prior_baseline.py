from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import load_config


RAW_STATS_ARTIFACTS = (
    "outputs/stats/subject_level_roi_results.parquet",
    "outputs/stats/group_level_roi_results.parquet",
    "outputs/stats/roi_condition_stats.parquet",
    "outputs/stats/roi_family_effect_panels.parquet",
)

RAW_SMALL_ARTIFACTS = (
    "outputs/tables/table01_dataset_summary.csv",
    "outputs/tables/table03_main_confirmatory_stats.csv",
    "outputs/figures/fig04_main_confirmatory_roi_families.png",
    "outputs/manuscript/claim_evidence_map.md",
)

CONDITION_MAP = {
    "acoustic_only": "ACOUSTIC_ONLY",
    "full": "FULL",
    "mismatched_shared": "MISMATCHED_SHARED",
    "raw": "RAW",
    "shared": "SHARED",
    "specific": "SPECIFIC",
}

MODEL_MAP = {
    "nllb_encoder": "nllb",
    "xlmr": "xlmr",
}


def _prior_repo_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["prior_repo_root"]).resolve()


def _local_bridge_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["local_outputs_root"]).resolve() / "bridge" / "prior_release"


def _representative_mismatch_shuffles(config: dict[str, Any]) -> int:
    refresh_cfg = dict(config.get("bridge", {}).get("prior_refresh", {}))
    return int(refresh_cfg.get("representative_mismatch_shuffles_default", 5))


def _copy_artifacts(prior_root: Path, bridge_root: Path) -> list[Path]:
    copied: list[Path] = []
    for relative in (*RAW_STATS_ARTIFACTS, *RAW_SMALL_ARTIFACTS):
        src = prior_root / relative
        if not src.exists():
            continue
        dst = bridge_root / "raw" / relative.replace("outputs/", "", 1)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def _load_prior_tables(prior_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subject = pd.read_parquet(prior_root / "outputs" / "stats" / "subject_level_roi_results.parquet").copy()
    group = pd.read_parquet(prior_root / "outputs" / "stats" / "group_level_roi_results.parquet").copy()
    feature_manifest = pd.read_parquet(prior_root / "data" / "processed" / "features" / "feature_manifest.parquet").copy()
    return subject, group, feature_manifest


def _legacy_status(model: str, condition: str) -> str:
    if model == "nllb" and condition == "MISMATCHED_SHARED":
        return "refresh_required_dense_nllb_and_20_shuffle_mismatch"
    if model == "nllb":
        return "refresh_required_dense_nllb"
    if condition == "MISMATCHED_SHARED":
        return "refresh_required_20_shuffle_mismatch"
    return "reusable_bridge_reference"


def _mismatch_shuffle_count(feature_manifest: pd.DataFrame) -> int:
    mismatch = feature_manifest.loc[feature_manifest["condition"].astype(str) == "mismatched_shared"].copy()
    if mismatch.empty or "shuffle_index" not in mismatch.columns:
        return 0
    shuffle_values = mismatch["shuffle_index"].dropna().astype(int)
    if shuffle_values.empty:
        return 0
    return int(shuffle_values.max()) + 1


def _normalize_subject_bridge(subject: pd.DataFrame) -> pd.DataFrame:
    metric_pivot = (
        subject.pivot_table(
            index=[
                "subject_id",
                "language",
                "model",
                "roi_name",
                "roi_family",
                "layer_index",
                "layer_depth",
                "condition",
            ],
            columns="metric_name",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    metric_pivot["language"] = metric_pivot["language"].astype(str).str.upper()
    metric_pivot["model"] = metric_pivot["model"].astype(str).map(MODEL_MAP).fillna(metric_pivot["model"].astype(str))
    metric_pivot["condition"] = (
        metric_pivot["condition"].astype(str).map(CONDITION_MAP).fillna(metric_pivot["condition"].astype(str).str.upper())
    )
    metric_pivot["roi_family"] = metric_pivot["roi_family"].astype(str).str.upper()
    metric_pivot["state"] = "OUTPUT"
    metric_pivot["block"] = metric_pivot["layer_index"].astype(int)
    metric_pivot["block_depth_norm"] = metric_pivot["layer_depth"].astype(float)
    metric_pivot["legacy_status"] = [
        _legacy_status(model=model, condition=condition)
        for model, condition in metric_pivot.loc[:, ["model", "condition"]].itertuples(index=False)
    ]
    metric_pivot["source_release"] = "prior_paper_release"
    return metric_pivot.rename(
        columns={
            "roi_name": "roi",
            "r": "r_mean",
            "z": "z_mean",
            "r2": "r2_mean",
        }
    )[
        [
            "subject_id",
            "language",
            "model",
            "block",
            "block_depth_norm",
            "state",
            "condition",
            "roi",
            "roi_family",
            "r_mean",
            "z_mean",
            "r2_mean",
            "legacy_status",
            "source_release",
        ]
    ]


def _normalize_group_bridge(group: pd.DataFrame) -> pd.DataFrame:
    bridge = group.copy()
    bridge["language"] = bridge["language"].astype(str).str.upper()
    bridge["model"] = bridge["model"].astype(str).map(MODEL_MAP).fillna(bridge["model"].astype(str))
    bridge["condition"] = bridge["condition"].astype(str).map(CONDITION_MAP).fillna(bridge["condition"].astype(str).str.upper())
    bridge["roi_family"] = bridge["roi_family"].astype(str).str.upper()
    bridge["state"] = "OUTPUT"
    bridge["block"] = bridge["layer_index"].astype(int)
    bridge["block_depth_norm"] = bridge["layer_depth"].astype(float)
    bridge["legacy_status"] = [
        _legacy_status(model=model, condition=condition)
        for model, condition in bridge.loc[:, ["model", "condition"]].itertuples(index=False)
    ]
    bridge["source_release"] = "prior_paper_release"
    return bridge.rename(columns={"roi_name": "roi"})[
        [
            "language",
            "model",
            "block",
            "block_depth_norm",
            "state",
            "condition",
            "roi",
            "roi_family",
            "mean_z",
            "mean_r",
            "mean_r2",
            "se_z",
            "n_subjects",
            "legacy_status",
            "source_release",
        ]
    ]


def _build_coverage_table(subject_bridge: pd.DataFrame, *, mismatch_shuffles_legacy: int) -> pd.DataFrame:
    coverage = (
        subject_bridge.groupby(["model", "language", "condition"], as_index=False)
        .agg(
            layer_count=("block", "nunique"),
            layer_min=("block", "min"),
            layer_max=("block", "max"),
            n_subjects=("subject_id", "nunique"),
            n_rois=("roi", "nunique"),
            legacy_status=("legacy_status", "first"),
        )
        .sort_values(["model", "language", "condition"])
        .reset_index(drop=True)
    )
    coverage["legacy_mismatch_shuffle_count"] = mismatch_shuffles_legacy
    coverage["bridge_use"] = [
        "reference_and_schema_bridge" if status == "reusable_bridge_reference" else "reference_only_until_local_refresh"
        for status in coverage["legacy_status"].astype(str)
    ]
    return coverage


def _build_markdown(
    *,
    config: dict[str, Any],
    prior_root: Path,
    copied_artifacts: list[Path],
    coverage: pd.DataFrame,
    mismatch_shuffles_legacy: int,
    bridge_root: Path,
) -> str:
    xlmr_reusable = coverage.loc[
        (coverage["model"] == "xlmr") & (coverage["bridge_use"] == "reference_and_schema_bridge")
    ].copy()
    nllb_rows = coverage.loc[coverage["model"] == "nllb"].copy()
    lines = [
        "# Prior Release Baseline Bridge",
        "",
        f"- Prior repo root: `{prior_root}`",
        f"- Local bridge root: `{bridge_root}`",
        f"- Copied artifacts: `{len(copied_artifacts)}`",
        f"- Legacy mismatch shuffle count detected in prior feature manifest: `{mismatch_shuffles_legacy}`",
        "",
        "## Reuse decision",
        "",
        "- The prior paper release is imported locally as a trusted bridge/reference surface.",
        "- XLM-R OUTPUT results remain directly reusable as a bridge baseline reference because the released surface is dense across layers 0-12.",
        "- NLLB OUTPUT results remain reusable as legacy reference only; they do not satisfy the sequel refresh because the release is sparse at layers 4/6/8.",
        f"- `MISMATCHED_SHARED` remains reference-only until replaced by the local repeated-shuffle refresh outputs at the representative default depth `K = {_representative_mismatch_shuffles(config)}`.",
        "",
        "## Coverage summary",
        "",
    ]
    if coverage.empty:
        lines.append("- No coverage rows were generated.")
    else:
        lines.append(coverage.to_string(index=False))
    lines.extend(
        [
            "",
            "## Quick checks",
            "",
            f"- XLM-R reusable bridge cells: `{len(xlmr_reusable)}`",
            f"- NLLB legacy reference cells: `{len(nllb_rows)}`",
            "",
            "## Copied raw artifacts",
            "",
        ]
    )
    if not copied_artifacts:
        lines.append("- No raw artifacts were copied.")
    else:
        lines.extend(f"- `{path}`" for path in copied_artifacts)
    lines.append("")
    return "\n".join(lines)


def materialize_prior_baseline(config: dict[str, Any]) -> dict[str, Path]:
    prior_root = _prior_repo_root(config)
    bridge_root = _local_bridge_root(config)
    bridge_root.mkdir(parents=True, exist_ok=True)

    copied_artifacts = _copy_artifacts(prior_root, bridge_root)
    subject, group, feature_manifest = _load_prior_tables(prior_root)

    subject_bridge = _normalize_subject_bridge(subject)
    group_bridge = _normalize_group_bridge(group)
    mismatch_shuffles_legacy = _mismatch_shuffle_count(feature_manifest)
    coverage = _build_coverage_table(subject_bridge, mismatch_shuffles_legacy=mismatch_shuffles_legacy)

    subject_bridge_path = bridge_root / "prior_release_subject_output_bridge.parquet"
    group_bridge_path = bridge_root / "prior_release_group_output_bridge.parquet"
    coverage_path = bridge_root / "prior_release_coverage.parquet"
    subject_bridge.to_parquet(subject_bridge_path, index=False)
    group_bridge.to_parquet(group_bridge_path, index=False)
    coverage.to_parquet(coverage_path, index=False)

    provenance_path = Path(config["paths"]["local_outputs_root"]).resolve() / "provenance" / "prior_release_baseline_bridge.md"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        _build_markdown(
            config=config,
            prior_root=prior_root,
            copied_artifacts=copied_artifacts,
            coverage=coverage,
            mismatch_shuffles_legacy=mismatch_shuffles_legacy,
            bridge_root=bridge_root,
        ),
        encoding="utf-8",
    )

    return {
        "subject_bridge": subject_bridge_path,
        "group_bridge": group_bridge_path,
        "coverage": coverage_path,
        "provenance": provenance_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import and normalize the prior-paper release baseline locally.")
    parser.add_argument("--config", default="conf/base.yaml", help="Path to sequel config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    outputs = materialize_prior_baseline(config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
