from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import load_config


def _coverage_path(config: dict[str, Any]) -> Path:
    return (
        Path(config["paths"]["local_outputs_root"]).resolve()
        / "bridge"
        / "prior_release"
        / "prior_release_coverage.parquet"
    )


def _output_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["local_outputs_root"]).resolve() / "provenance"


def _prior_refresh_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("bridge", {}).get("prior_refresh", {}))


def _build_refresh_jobs(config: dict[str, Any], coverage: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    nllb_legacy_layers = (4, 6, 8)
    nllb_missing_layers = tuple(layer for layer in range(13) if layer not in nllb_legacy_layers)
    refresh_cfg = _prior_refresh_config(config)
    missing_layer_mismatch_shuffles = int(refresh_cfg.get("missing_layer_mismatch_shuffles_default", 1))
    representative_mismatch_shuffles = int(refresh_cfg.get("representative_mismatch_shuffles_default", 5))
    representative_families = tuple(str(value).upper() for value in refresh_cfg.get("representative_roi_families", ["SEMANTIC", "AUDITORY"]))

    languages = tuple(sorted(coverage["language"].astype(str).unique().tolist()))
    for language in languages:
        rows.append(
            {
                "job_id": f"nllb_{language.lower()}_missing_layers_refresh",
                "model": "nllb",
                "language": language,
                "job_type": "local_refresh",
                "refresh_mode": "full",
                "roi_family": pd.NA,
                "expected_representative_rows": pd.NA,
                "layer_indices": ",".join(str(layer) for layer in nllb_missing_layers),
                "mismatch_shuffles": missing_layer_mismatch_shuffles,
                "reuse_surface": "prior_release_layers_4_6_8_only",
                "reason": "Prior NLLB OUTPUT release is sparse at layers 4/6/8, so the missing layers must be filled locally. The one-shuffle mismatch rows produced during this step are byproducts, not the final repeated-shuffle control.",
                "output_tag": f"sequel_refresh_nllb_missing_layers_{language.lower()}",
                "resume_strategy": "subject_only_resume_same_tag_until_all_subjects_complete",
                "synthesis_role": "append_missing_layers_and_replace_exact_rows_if_overlap",
            }
        )
        rows.append(
            {
                "job_id": f"{language.lower()}_representative_layer_selection",
                "model": "xlmr+nllb",
                "language": language,
                "job_type": "selection",
                "refresh_mode": "representative_output_layer",
                "roi_family": ",".join(representative_families),
                "expected_representative_rows": len(representative_families),
                "layer_indices": "from_group_results",
                "mismatch_shuffles": pd.NA,
                "reuse_surface": "xlmr_prior_release_plus_nllb_dense_refresh",
                "reason": "Repeated mismatch shuffles are restricted to the fixed semantic/auditory representative OUTPUT grid rather than every layer.",
                "output_tag": f"output_representative_layers_{language.lower()}",
                "resume_strategy": "not_applicable",
                "synthesis_role": "choose_semantic_and_auditory_representative_layers",
            }
        )
        for roi_family in representative_families:
            rows.append(
                {
                    "job_id": f"xlmr_{language.lower()}_{roi_family.lower()}_representative_mismatch_refresh",
                    "model": "xlmr",
                    "language": language,
                    "job_type": "local_refresh",
                    "refresh_mode": "mismatch_only_representative_layers",
                    "roi_family": roi_family,
                    "expected_representative_rows": 1,
                    "layer_indices": "from_output_representative_layers",
                    "mismatch_shuffles": representative_mismatch_shuffles,
                    "reuse_surface": "prior_release_dense_output_surface",
                    "reason": "XLM-R layer coverage is already dense, so only the representative-layer repeated-shuffle mismatch control needs local refresh at the K=5 default depth unless instability forces expansion.",
                    "output_tag": f"sequel_refresh_xlmr_mismatch_representative_{language.lower()}",
                    "resume_strategy": "subject_only_resume_same_tag_until_all_subjects_complete",
                    "synthesis_role": "build_mismatch_representative_results",
                }
            )
            rows.append(
                {
                    "job_id": f"nllb_{language.lower()}_{roi_family.lower()}_representative_mismatch_refresh",
                    "model": "nllb",
                    "language": language,
                    "job_type": "local_refresh",
                    "refresh_mode": "mismatch_only_representative_layers",
                    "roi_family": roi_family,
                    "expected_representative_rows": 1,
                    "layer_indices": "from_output_representative_layers",
                    "mismatch_shuffles": representative_mismatch_shuffles,
                    "reuse_surface": "dense_nllb_output_surface_after_refresh",
                    "reason": "After dense NLLB coverage is restored, repeated-shuffle mismatch is refreshed only at the representative OUTPUT layers at the K=5 default depth unless instability forces expansion.",
                    "output_tag": f"sequel_refresh_nllb_mismatch_representative_{language.lower()}",
                    "resume_strategy": "subject_only_resume_same_tag_until_all_subjects_complete",
                    "synthesis_role": "build_mismatch_representative_results",
                }
            )

    rows.extend(
        [
            {
                "job_id": "refresh_output_bridge_materialization",
                "model": "xlmr+nllb",
                "language": "ALL",
                "job_type": "synthesis",
                "refresh_mode": "bridge_materialization",
                "roi_family": pd.NA,
                "expected_representative_rows": pd.NA,
                "layer_indices": "all_available_output_layers",
                "mismatch_shuffles": pd.NA,
                "reuse_surface": "prior_release_plus_dense_nllb_refresh",
                "reason": "Build the refreshed OUTPUT bridge table used for Figure 2 from the strongest available local surfaces.",
                "output_tag": "refresh_output_results",
                "resume_strategy": "not_applicable",
                "synthesis_role": "write_outputs/group_results/refresh_output_results.parquet",
            },
            {
                "job_id": "nllb_finalize_dense_refresh",
                "model": "nllb",
                "language": "ALL",
                "job_type": "synthesis",
                "refresh_mode": "blend_prior_and_local",
                "roi_family": pd.NA,
                "expected_representative_rows": pd.NA,
                "layer_indices": "0,1,2,3,4,5,6,7,8,9,10,11,12",
                "mismatch_shuffles": pd.NA,
                "reuse_surface": "prior_release_plus_local_missing_layers_refresh",
                "reason": "Blend prior NLLB release with local missing-layer refresh chunks to create dense OUTPUT coverage.",
                "output_tag": "sequel_refresh_output_nllb_dense",
                "resume_strategy": "not_applicable",
                "synthesis_role": "replace_exact_rows_and_finalize",
            },
            {
                "job_id": "mismatch_representative_materialization",
                "model": "xlmr+nllb",
                "language": "ALL",
                "job_type": "synthesis",
                "refresh_mode": "representative_mismatch_aggregation",
                "roi_family": ",".join(representative_families),
                "expected_representative_rows": len(languages) * len(representative_families) * 2,
                "layer_indices": "from_output_representative_layers",
                "mismatch_shuffles": representative_mismatch_shuffles,
                "reuse_surface": "representative_mismatch_refresh_chunks_plus_output_baseline",
                "reason": "Aggregate representative-layer repeated-shuffle mismatch outputs into the dedicated control table required by the sequel bridge claim.",
                "output_tag": "mismatch_representative_results",
                "resume_strategy": "not_applicable",
                "synthesis_role": "write_outputs/group_results/mismatch_representative_results.parquet",
            },
        ]
    )

    return pd.DataFrame(rows)


def _build_markdown(jobs: pd.DataFrame, *, coverage_path: Path, manifest_path: Path) -> str:
    lines = [
        "# Refresh Output Job Manifest",
        "",
        f"- Coverage source: `{coverage_path}`",
        f"- Manifest path: `{manifest_path}`",
        "",
        "## Strategy",
        "",
        "- Keep as much of the prior release as possible.",
        "- Refresh only the pieces the sequel contract forces us to change.",
        "- Recombine local refresh chunks with prior-release subject rows by exact row keys, then finalize with the inherited prior aggregation code.",
        "",
        "## Jobs",
        "",
        jobs.to_string(index=False),
        "",
    ]
    return "\n".join(lines)


def build_refresh_job_manifest(config: dict[str, Any]) -> dict[str, Path]:
    coverage_path = _coverage_path(config)
    coverage = pd.read_parquet(coverage_path).copy()
    jobs = _build_refresh_jobs(config, coverage)

    output_root = _output_root(config)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "refresh_output_job_manifest.parquet"
    markdown_path = output_root / "refresh_output_job_manifest.md"
    jobs.to_parquet(manifest_path, index=False)
    markdown_path.write_text(
        _build_markdown(jobs, coverage_path=coverage_path, manifest_path=manifest_path),
        encoding="utf-8",
    )
    return {
        "manifest": manifest_path,
        "markdown": markdown_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the local refresh job manifest for the OUTPUT-state baseline.")
    parser.add_argument("--config", default="conf/base.yaml", help="Path to sequel config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    outputs = build_refresh_job_manifest(config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
