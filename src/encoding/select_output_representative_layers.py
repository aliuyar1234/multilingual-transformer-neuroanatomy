from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import load_config


def _local_outputs_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["local_outputs_root"]).resolve()


def _representative_roi_families(config: dict[str, Any]) -> tuple[str, ...]:
    configured = config.get("bridge", {}).get("prior_refresh", {}).get("representative_roi_families", ["SEMANTIC", "AUDITORY"])
    families = tuple(str(value).upper() for value in configured)
    return families or ("SEMANTIC", "AUDITORY")


def _expected_representative_row_count(config: dict[str, Any]) -> int:
    n_models = len(config.get("models", {})) or 2
    n_languages = len(config.get("dataset", {}).get("languages", ["EN", "FR", "ZH"]))
    n_families = len(_representative_roi_families(config))
    return int(n_models * n_languages * n_families)


def _default_group_sources(config: dict[str, Any]) -> list[Path]:
    output_root = _local_outputs_root(config)
    sources = [
        output_root / "bridge" / "prior_release" / "raw" / "stats" / "group_level_roi_results.parquet",
        output_root / "stats" / "nllb_group_level_roi_results__sequel_refresh_output_nllb_dense.parquet",
        output_root / "group_results" / "state_sweep_group_results.parquet",
    ]
    return [path for path in sources if path.exists()]


def _normalize_group_source(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    if {"block", "block_depth_norm", "state", "roi", "roi_family", "condition", "mean_z", "model", "language"}.issubset(frame.columns):
        normalized = frame.loc[frame["state"].astype(str).str.upper() == "OUTPUT"].copy()
        normalized["layer_index"] = normalized["block"].astype(int)
        normalized["layer_depth"] = normalized["block_depth_norm"].astype(float)
        normalized["roi_name"] = normalized["roi"].astype(str)
        normalized["condition"] = normalized["condition"].astype(str).str.lower()
        return normalized
    if {"layer_index", "layer_depth", "roi_name", "roi_family", "condition", "mean_z", "model", "language"}.issubset(frame.columns):
        normalized = frame.copy()
        normalized["condition"] = normalized["condition"].astype(str).str.lower()
        return normalized
    raise KeyError(
        "Representative-layer selection source has unsupported columns. "
        f"Path: {path}; columns: {sorted(frame.columns.tolist())}"
    )


def _load_group_tables(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        raise FileNotFoundError("No group-level result tables were available for representative-layer selection.")

    frames: list[pd.DataFrame] = []
    for priority, path in enumerate(paths):
        frame = _normalize_group_source(path)
        frame["source_path"] = str(path.resolve())
        frame["source_priority"] = priority
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)

    key_cols = ["language", "model", "roi_name", "roi_family", "layer_index", "condition"]
    combined = combined.sort_values("source_priority").drop_duplicates(subset=key_cols, keep="last")
    return combined.reset_index(drop=True)


def _family_delta_table(group_df: pd.DataFrame) -> pd.DataFrame:
    filtered = group_df.loc[group_df["condition"].astype(str).isin(["shared", "specific"])].copy()
    filtered["language"] = filtered["language"].astype(str).str.upper()
    filtered["roi_family"] = filtered["roi_family"].astype(str).str.upper()
    filtered["model"] = filtered["model"].astype(str).replace({"nllb_encoder": "nllb"})
    summary = (
        filtered.groupby(["model", "language", "roi_family", "layer_index", "layer_depth", "condition"], as_index=False)[
            "mean_z"
        ]
        .mean()
        .pivot_table(
            index=["model", "language", "roi_family", "layer_index", "layer_depth"],
            columns="condition",
            values="mean_z",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    summary["shared_minus_specific"] = summary["shared"] - summary["specific"]
    return summary.sort_values(["model", "language", "roi_family", "layer_index"]).reset_index(drop=True)


def _representative_layers(config: dict[str, Any], delta_df: pd.DataFrame) -> pd.DataFrame:
    allowed_families = set(_representative_roi_families(config))
    filtered = delta_df.loc[delta_df["roi_family"].astype(str).isin(allowed_families)].copy()
    if filtered.empty:
        raise RuntimeError("No semantic/auditory OUTPUT rows were available for representative-layer selection.")
    representatives = (
        filtered.sort_values(
            ["model", "language", "roi_family", "shared_minus_specific", "layer_index"],
            ascending=[True, True, True, False, True],
        )
        .groupby(["model", "language", "roi_family"], as_index=False)
        .first()
        .reset_index(drop=True)
    )
    representatives["state"] = "OUTPUT"
    representatives["block"] = representatives["layer_index"].astype(int)
    representatives["block_depth_norm"] = representatives["layer_depth"].astype(float)
    representatives["selection_metric"] = "max_group_mean_shared_minus_specific"
    representatives["delta_z_mean"] = representatives["shared_minus_specific"].astype(float)
    representatives["tie_break_notes"] = "ties broken toward shallower layer"
    representatives = representatives[
        [
            "model",
            "language",
            "roi_family",
            "block",
            "block_depth_norm",
            "state",
            "selection_metric",
            "delta_z_mean",
            "tie_break_notes",
        ]
    ]
    expected_rows = _expected_representative_row_count(config)
    if len(representatives) != expected_rows:
        raise RuntimeError(
            "Representative OUTPUT layer selection must yield the fixed semantic/auditory bridge grid: "
            f"expected {expected_rows} rows, got {len(representatives)}."
        )
    return representatives


def _write_markdown(
    *,
    group_sources: list[Path],
    delta_path: Path,
    representative_path: Path,
    representatives: pd.DataFrame,
) -> str:
    lines = [
        "# Output Representative Layers",
        "",
        "## Sources",
        "",
    ]
    lines.extend(f"- `{path}`" for path in group_sources)
    lines.extend(
        [
            "",
            f"- delta table: `{delta_path}`",
            f"- representative layers: `{representative_path}`",
            "",
            "## Selected layers",
            "",
            representatives.to_string(index=False),
            "",
        ]
    )
    return "\n".join(lines)


def select_output_representative_layers(
    config: dict[str, Any],
    *,
    group_paths: list[Path] | None = None,
) -> dict[str, Path]:
    sources = group_paths or _default_group_sources(config)
    group_df = _load_group_tables(sources)
    delta_df = _family_delta_table(group_df)
    representatives = _representative_layers(config, delta_df)

    output_root = _local_outputs_root(config)
    group_results_root = output_root / "group_results"
    provenance_root = output_root / "provenance"
    group_results_root.mkdir(parents=True, exist_ok=True)
    provenance_root.mkdir(parents=True, exist_ok=True)

    delta_path = group_results_root / "output_layer_family_deltas.parquet"
    representative_path = group_results_root / "output_representative_layers.parquet"
    markdown_path = provenance_root / "output_representative_layers.md"

    delta_df.to_parquet(delta_path, index=False)
    representatives.to_parquet(representative_path, index=False)
    markdown_path.write_text(
        _write_markdown(
            group_sources=sources,
            delta_path=delta_path,
            representative_path=representative_path,
            representatives=representatives,
        ),
        encoding="utf-8",
    )

    return {
        "delta": delta_path,
        "representative_layers": representative_path,
        "provenance": markdown_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select representative OUTPUT layers for semantic and auditory ROI families.")
    parser.add_argument("--config", default="conf/base.yaml", help="Path to sequel config.")
    parser.add_argument(
        "--group-path",
        action="append",
        default=None,
        help="Optional group-level parquet path. May be passed multiple times; later paths override earlier duplicates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    group_paths = None if not args.group_path else [Path(path).resolve() for path in args.group_path]
    outputs = select_output_representative_layers(config, group_paths=group_paths)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
