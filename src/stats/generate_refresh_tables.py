from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import load_config


OUTPUT_EQUIVALENCE_TOLERANCE = 1.0e-5


def _local_outputs_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["local_outputs_root"]).resolve()


def _local_manifests_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["local_manifests_root"]).resolve()


def _sample_manifest_path(config: dict[str, Any]) -> Path:
    return _local_manifests_root(config) / "sample_manifest.parquet"


def _roi_manifest_path(config: dict[str, Any]) -> Path:
    return _local_manifests_root(config) / "roi_manifest.parquet"


def _state_cache_manifest_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "caches" / "features" / "state_cache_manifest.parquet"


def _state_extraction_qc_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "qc" / "state_extraction_equivalence.parquet"


def _table_path(config: dict[str, Any], filename: str) -> Path:
    return _local_outputs_root(config) / "tables" / filename


def build_table01_sample_summary(config: dict[str, Any]) -> Path:
    sample_path = _sample_manifest_path(config)
    roi_path = _roi_manifest_path(config)
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample manifest missing: {sample_path}")
    if not roi_path.exists():
        raise FileNotFoundError(f"ROI manifest missing: {roi_path}")

    sample_df = pd.read_parquet(sample_path).copy()
    roi_df = pd.read_parquet(roi_path).copy()
    tr_seconds = float(config.get("dataset", {}).get("tr_seconds", 2.0))

    rows: list[dict[str, object]] = []
    for language, language_df in sample_df.groupby("language", sort=True):
        included_df = language_df.loc[language_df["included"].astype(bool)].copy()
        excluded_df = language_df.loc[~language_df["included"].astype(bool)].copy()
        roi_subset = roi_df.loc[
            (roi_df["language"].astype(str) == str(language))
            & (roi_df["subject_id"].astype(str).isin(included_df["subject_id"].astype(str)))
        ].copy()
        exclusions = excluded_df["exclusion_reason"].astype(str).str.strip()
        exclusions = exclusions.loc[exclusions.astype(str) != ""]
        rows.append(
            {
                "language": str(language),
                "n_subjects": int(included_df["subject_id"].astype(str).nunique()),
                "n_runs": int(included_df["n_canonical_runs"].astype(int).max()) if not included_df.empty else 0,
                "mean_duration": float(roi_subset["n_scans"].astype(float).mean() * tr_seconds) if not roi_subset.empty else 0.0,
                "TR": tr_seconds,
                "any_exclusions": "; ".join(sorted(exclusions.unique().tolist())) if not exclusions.empty else "none",
            }
        )

    table_path = _table_path(config, "table01_sample_summary.csv")
    table_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("language").reset_index(drop=True).to_csv(table_path, index=False)
    return table_path


def build_table02_model_state_inventory(config: dict[str, Any]) -> Path:
    model_cfg = config.get("models", {})
    state_manifest_path = _state_cache_manifest_path(config)
    qc_path = _state_extraction_qc_path(config)

    state_manifest = pd.read_parquet(state_manifest_path).copy() if state_manifest_path.exists() else pd.DataFrame()
    qc_df = pd.read_parquet(qc_path).copy() if qc_path.exists() else pd.DataFrame()

    rows: list[dict[str, object]] = []
    for model_name, settings in sorted(model_cfg.items()):
        subset = state_manifest.loc[state_manifest["model"].astype(str) == str(model_name)].copy() if not state_manifest.empty else pd.DataFrame()
        qc_subset = qc_df.loc[qc_df["model"].astype(str) == str(model_name)].copy() if not qc_df.empty else pd.DataFrame()
        extracted_states = (
            ",".join(sorted(subset["state"].astype(str).unique().tolist()))
            if not subset.empty
            else ",".join(str(state) for state in settings.get("state_names", []))
        )
        n_blocks = int(subset["block"].astype(int).max() + 1) if not subset.empty else 0
        cache_size_bytes = 0
        if not subset.empty and "source_array_path" in subset.columns:
            for source in subset["source_array_path"].astype(str):
                path = Path(source)
                if path.exists():
                    cache_size_bytes += path.stat().st_size
        if not qc_subset.empty and "max_abs_diff" in qc_subset.columns:
            output_equivalence = "PASS" if float(qc_subset["max_abs_diff"].astype(float).max()) <= OUTPUT_EQUIVALENCE_TOLERANCE else "FAIL"
        else:
            output_equivalence = "NOT_RUN"
        rows.append(
            {
                "model": str(model_name),
                "hidden_size": int(settings.get("hidden_size", 0)),
                "n_blocks": n_blocks,
                "states_extracted": extracted_states,
                "output_equivalence": output_equivalence,
                "cache_size_summary": f"{cache_size_bytes / (1024 * 1024):.2f} MB" if cache_size_bytes > 0 else "0.00 MB",
            }
        )

    table_path = _table_path(config, "table02_model_state_inventory.csv")
    table_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("model").reset_index(drop=True).to_csv(table_path, index=False)
    return table_path


def generate_refresh_tables(config: dict[str, Any]) -> dict[str, Path]:
    return {
        "table01": build_table01_sample_summary(config),
        "table02": build_table02_model_state_inventory(config),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate bridge/refresh descriptive tables.")
    parser.add_argument("--config", default="conf/base.yaml", help="Path to sequel config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    outputs = generate_refresh_tables(config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
