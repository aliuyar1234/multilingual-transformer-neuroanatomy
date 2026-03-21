from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from src.bridge.prior_runtime import prior_python, prior_repo_root, run_prior_repo_python
from src.utils.config import load_config


MODEL_TO_PRIOR = {
    "nllb": "nllb_encoder",
    "xlmr": "xlmr",
}


def _local_prior_subject_path(config: dict[str, Any]) -> Path:
    local_path = (
        Path(config["paths"]["local_outputs_root"]).resolve()
        / "bridge"
        / "prior_release"
        / "raw"
        / "stats"
        / "subject_level_roi_results.parquet"
    )
    if local_path.exists():
        return local_path
    return prior_repo_root(config) / "outputs" / "stats" / "subject_level_roi_results.parquet"


def _mirror_stats_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["prior_mirror_root"]).resolve() / "outputs" / "stats"


def _local_stats_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["local_outputs_root"]).resolve() / "stats"


def _local_staging_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["local_outputs_root"]).resolve() / "staging"


def _collect_chunk_paths(config: dict[str, Any], patterns: tuple[str, ...]) -> tuple[Path, ...]:
    stats_root = _mirror_stats_root(config)
    matched: list[Path] = []
    for pattern in patterns:
        pattern_path = Path(pattern)
        if pattern_path.is_absolute():
            if pattern_path.exists():
                matched.append(pattern_path.resolve())
            continue
        matched.extend(sorted(path.resolve() for path in stats_root.glob(pattern)))
    deduped = tuple(dict.fromkeys(matched))
    if not deduped:
        raise FileNotFoundError(f"No local mirror chunk files matched: {patterns!r}")
    return deduped


def _combined_input_path(config: dict[str, Any], *, model: str, output_tag: str) -> Path:
    root = _local_staging_root(config)
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{model}_subject_level_roi_results__{output_tag}__combined_input.parquet"


def _mirror_output_paths(config: dict[str, Any], *, model: str, output_tag: str) -> dict[str, Path]:
    stem = "nllb" if model == "nllb" else model
    root = _mirror_stats_root(config)
    return {
        "subject": root / f"{stem}_subject_level_roi_results__{output_tag}.parquet",
        "group": root / f"{stem}_group_level_roi_results__{output_tag}.parquet",
        "confirmatory": root / f"{stem}_confirmatory_effects__{output_tag}.parquet",
    }


def _local_output_paths(config: dict[str, Any], *, model: str, output_tag: str) -> dict[str, Path]:
    stem = "nllb" if model == "nllb" else model
    root = _local_stats_root(config)
    root.mkdir(parents=True, exist_ok=True)
    return {
        "subject": root / f"{stem}_subject_level_roi_results__{output_tag}.parquet",
        "group": root / f"{stem}_group_level_roi_results__{output_tag}.parquet",
        "confirmatory": root / f"{stem}_confirmatory_effects__{output_tag}.parquet",
    }


def _replace_exact_rows(
    prior_df: pd.DataFrame,
    local_df: pd.DataFrame,
    *,
    replace_acoustic_only: bool,
) -> pd.DataFrame:
    if not replace_acoustic_only:
        local_df = local_df.loc[local_df["condition"].astype(str) != "acoustic_only"].copy()
    if local_df.empty:
        return prior_df.copy()

    key_cols = ["subject_id", "language", "model", "roi_name", "layer_index", "condition", "metric_name"]
    local_keys = local_df.loc[:, key_cols].drop_duplicates().assign(_replace=True)
    remaining_prior = prior_df.merge(local_keys, on=key_cols, how="left")
    remaining_prior = remaining_prior.loc[remaining_prior["_replace"].isna()].drop(columns="_replace")
    combined = pd.concat([remaining_prior, local_df], ignore_index=True)
    return combined.sort_values(key_cols).reset_index(drop=True)


def _merge_and_finalize(config: dict[str, Any], *, model: str, combined_input_path: Path, output_tag: str) -> dict[str, Path]:
    execution_root = prior_repo_root(config)
    python_exe = prior_python(execution_root, config)
    code = (
        "import sys\n"
        f"sys.path.insert(0, {repr(str((execution_root / 'src').resolve()))})\n"
        "import brain_subspace_paper.encoding.xlmr_roi_pipeline as pipeline_module\n"
        f"summary = pipeline_module.merge_subject_result_chunks(\n"
        f"    model_name={repr(model)},\n"
        f"    input_paths=({repr(str(combined_input_path.resolve()))},),\n"
        f"    output_tag={repr(output_tag)},\n"
        ")\n"
        "print(summary)\n"
    )
    completed = run_prior_repo_python(
        code,
        prior_repo_root=execution_root,
        python_exe=python_exe,
    )
    print(completed.stdout.strip())
    if completed.stderr.strip():
        print(completed.stderr.strip())

    mirror_outputs = _mirror_output_paths(config, model=model, output_tag=output_tag)
    local_outputs = _local_output_paths(config, model=model, output_tag=output_tag)
    for name, src in mirror_outputs.items():
        dst = local_outputs[name]
        if not src.exists():
            raise FileNotFoundError(f"Expected finalized {name} output was not written: {src}")
        shutil.copy2(src, dst)
    return local_outputs


def synthesize_refresh_baseline(
    config: dict[str, Any],
    *,
    model: str,
    chunk_patterns: tuple[str, ...],
    output_tag: str,
    replace_acoustic_only: bool = False,
) -> dict[str, Path]:
    prior_model = MODEL_TO_PRIOR[model]
    prior_subject = pd.read_parquet(_local_prior_subject_path(config)).copy()
    prior_subject = prior_subject.loc[prior_subject["model"].astype(str) == prior_model].copy()

    chunk_paths = _collect_chunk_paths(config, chunk_patterns)
    local_frames = [pd.read_parquet(path).copy() for path in chunk_paths]
    local_subject = pd.concat(local_frames, ignore_index=True)
    local_subject = local_subject.loc[local_subject["model"].astype(str) == prior_model].copy()
    if local_subject.empty:
        raise RuntimeError(f"No local rows matched model={model!r} in the selected chunk files.")

    combined = _replace_exact_rows(
        prior_subject,
        local_subject,
        replace_acoustic_only=replace_acoustic_only,
    )
    combined_input_path = _combined_input_path(config, model=model, output_tag=output_tag)
    combined.to_parquet(combined_input_path, index=False)
    finalized = _merge_and_finalize(config, model=model, combined_input_path=combined_input_path, output_tag=output_tag)
    finalized["combined_input"] = combined_input_path
    return finalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blend prior-release subject rows with local mirror refresh chunks and finalize local outputs."
    )
    parser.add_argument("--config", default="conf/base.yaml", help="Path to sequel config.")
    parser.add_argument("--model", required=True, choices=["xlmr", "nllb"])
    parser.add_argument("--chunk-glob", nargs="+", required=True, help="Mirror stats glob(s) or absolute parquet path(s).")
    parser.add_argument("--output-tag", required=True, help="Output tag for the synthesized local baseline.")
    parser.add_argument(
        "--replace-acoustic-only",
        action="store_true",
        help="Replace prior acoustic_only rows when the local chunk also contains them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    outputs = synthesize_refresh_baseline(
        config,
        model=args.model,
        chunk_patterns=tuple(args.chunk_glob),
        output_tag=args.output_tag,
        replace_acoustic_only=args.replace_acoustic_only,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
