from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import load_config


LANGUAGE_TO_UPPER = {"en": "EN", "fr": "FR", "zh": "ZH"}
UPPER_TO_LANGUAGE = {value: key for key, value in LANGUAGE_TO_UPPER.items()}


@dataclass(frozen=True)
class ManifestOutputs:
    dataset_manifest_path: Path
    canonical_run_manifest_path: Path
    sample_manifest_path: Path
    triplets_manifest_path: Path
    roi_manifest_path: Path
    sample_provenance_path: Path
    dataset_provenance_path: Path


def _now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _prior_repo_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["prior_repo_root"]).resolve()


def _prior_data_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["prior_data_root"]).resolve()


def _prior_outputs_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["prior_outputs_root"]).resolve()


def _local_manifests_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["local_manifests_root"]).resolve()


def _local_outputs_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["local_outputs_root"]).resolve()


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required prior artifact not found: {path}")
    return pd.read_parquet(path).copy()


def _absolute_prior_path(prior_repo_root: Path, path_value: str) -> str:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return str(candidate.resolve())
    return str((prior_repo_root / candidate).resolve())


def _build_dataset_manifest(config: dict[str, Any]) -> pd.DataFrame:
    prior_repo_root = _prior_repo_root(config)
    prior_data_root = _prior_data_root(config)
    prior_outputs_root = _prior_outputs_root(config)
    rows = [
        {
            "dataset_name": "LPPC",
            "openneuro_accession": config["dataset"]["accession"],
            "language_set": ",".join(config["dataset"]["languages"]),
            "tr_seconds": float(config["dataset"]["tr_seconds"]),
            "source_kind": "prior_repo_external_baseline",
            "source_root": str(prior_repo_root),
            "data_root": str(prior_data_root),
            "outputs_root": str(prior_outputs_root),
            "exists": all(path.exists() for path in (prior_repo_root, prior_data_root, prior_outputs_root)),
            "version_note": "Using the external prior repo as the baseline source of truth for salvage.",
            "salvage_timestamp_utc": _now_utc(),
        }
    ]
    return pd.DataFrame(rows)


def _transform_canonical_run_manifest(config: dict[str, Any]) -> pd.DataFrame:
    prior_repo_root = _prior_repo_root(config)
    path = _prior_data_root(config) / "interim" / "lppc_run_manifest.parquet"
    run_df = _read_parquet(path)
    transformed = run_df.rename(
        columns={
            "subject_id": "subject_id",
            "language": "language",
            "task_name": "task_name",
            "original_run_label": "raw_run_label",
            "canonical_run_index": "canonical_run",
            "n_volumes": "n_trs",
            "filepath": "prior_relative_bold_path",
        }
    ).copy()
    transformed["language"] = transformed["language"].map(LANGUAGE_TO_UPPER)
    transformed["story_section"] = transformed["canonical_run"].astype(int)
    transformed["bold_path"] = transformed["prior_relative_bold_path"].map(
        lambda value: _absolute_prior_path(prior_repo_root, value)
    )
    transformed["valid_preproc_bold"] = transformed["is_preproc"].astype(bool)
    transformed["valid_n_trs"] = transformed["n_trs"].astype(int) > 0
    transformed["is_valid"] = transformed["valid_preproc_bold"] & transformed["valid_n_trs"]
    transformed["source_artifact"] = str(path)
    column_order = [
        "subject_id",
        "language",
        "task_name",
        "raw_run_label",
        "canonical_run",
        "story_section",
        "n_trs",
        "space",
        "prior_relative_bold_path",
        "bold_path",
        "valid_preproc_bold",
        "valid_n_trs",
        "is_valid",
        "source_artifact",
    ]
    return transformed.loc[:, column_order].sort_values(["language", "subject_id", "canonical_run"]).reset_index(drop=True)


def _transform_sentence_spans(config: dict[str, Any], language: str) -> pd.DataFrame:
    lower = UPPER_TO_LANGUAGE[language]
    path = _prior_data_root(config) / "interim" / f"sentence_spans_{lower}.parquet"
    spans = _read_parquet(path)
    transformed = pd.DataFrame(
        {
            "language": language,
            "story_section": spans["section_index"].astype(int),
            "canonical_run": spans["section_index"].astype(int),
            "section_local_sentence_id": spans["section_sentence_index"].astype(int),
            "global_sentence_id": spans["language_sentence_index"].astype(int),
            "text": spans["text"].astype(str),
            "onset_sec": spans["onset_sec"].astype(float),
            "offset_sec": spans["offset_sec"].astype(float),
            "duration_sec": spans["duration_sec"].astype(float),
            "merge_size": 1,
            "source_annotation_ids": spans.apply(
                lambda row: json.dumps(
                    {
                        "first_word_idx": int(row["first_word_idx"]),
                        "last_word_idx": int(row["last_word_idx"]),
                    },
                    ensure_ascii=False,
                ),
                axis=1,
            ),
            "quality_flag": spans["alignment_cost"].map(lambda value: "ok" if float(value) == 0.0 else "alignment_cost_nonzero"),
            "n_words": spans["n_words"].astype(int),
            "first_word_idx": spans["first_word_idx"].astype(int),
            "last_word_idx": spans["last_word_idx"].astype(int),
            "alignment_cost": spans["alignment_cost"].astype(float),
            "source_artifact": str(path),
        }
    )
    return transformed.sort_values(["canonical_run", "section_local_sentence_id"]).reset_index(drop=True)


def _sentence_id_json(first_idx: pd.Series, last_idx: pd.Series) -> pd.Series:
    return pd.Series(
        [
            json.dumps(list(range(int(start), int(stop) + 1)), ensure_ascii=False)
            for start, stop in zip(first_idx.tolist(), last_idx.tolist(), strict=True)
        ]
    )


def _transform_triplets(config: dict[str, Any]) -> pd.DataFrame:
    triplet_path = _prior_data_root(config) / "processed" / "alignment_triplets.parquet"
    qc_path = _prior_data_root(config) / "processed" / "alignment_triplets_qc.parquet"
    triplets = _read_parquet(triplet_path)
    qc = _read_parquet(qc_path)
    merged = triplets.merge(qc, on=["triplet_id", "section_index"], how="left")
    transformed = pd.DataFrame(
        {
            "triplet_id": merged["triplet_id"].astype(int),
            "story_section": merged["section_index"].astype(int),
            "canonical_run": merged["section_index"].astype(int),
            "merge_pattern": merged["merge_pattern"].astype(str),
            "review_flag": merged["needs_manual_review"].fillna(False).astype(bool),
            "repair_flag": merged["manual_status"].fillna("approved").astype(str).ne("approved"),
            "en_text": merged["en_text"].astype(str),
            "fr_text": merged["fr_text"].astype(str),
            "zh_text": merged["zh_text"].astype(str),
            "en_sentence_ids": _sentence_id_json(merged["en_first_sentence_idx"], merged["en_last_sentence_idx"]),
            "fr_sentence_ids": _sentence_id_json(merged["fr_first_sentence_idx"], merged["fr_last_sentence_idx"]),
            "zh_sentence_ids": _sentence_id_json(merged["zh_first_sentence_idx"], merged["zh_last_sentence_idx"]),
            "en_onset_sec": merged["en_onset_sec"].astype(float),
            "en_offset_sec": merged["en_offset_sec"].astype(float),
            "fr_onset_sec": merged["fr_onset_sec"].astype(float),
            "fr_offset_sec": merged["fr_offset_sec"].astype(float),
            "zh_onset_sec": merged["zh_onset_sec"].astype(float),
            "zh_offset_sec": merged["zh_offset_sec"].astype(float),
            "section_local_triplet_id": merged["section_triplet_index"].astype(int),
            "qc_manual_status": merged["manual_status"].fillna("missing").astype(str),
            "qc_notes": merged["notes"].fillna("").astype(str),
            "source_artifact_triplets": str(triplet_path),
            "source_artifact_qc": str(qc_path),
        }
    )
    return transformed.sort_values(["canonical_run", "section_local_triplet_id"]).reset_index(drop=True)


def _combine_roi_manifests(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    prior_data_root = _prior_data_root(config)
    roi_root = prior_data_root / "interim" / "roi"
    roi_metadata_path = roi_root / "roi_metadata.parquet"
    atlas_path = roi_root / "harvard_oxford_cortical_resampled_to_lppc_bold.nii.gz"
    roi_metadata = _read_parquet(roi_metadata_path).rename(
        columns={"family": "roi_family"}
    ).copy()
    roi_metadata["roi_family"] = roi_metadata["roi_family"].astype(str).str.upper()
    roi_metadata["roi_code"] = roi_metadata["roi_name"].astype(str)
    roi_metadata["source_artifact"] = str(roi_metadata_path)
    manifests: list[pd.DataFrame] = []
    for lower, upper in LANGUAGE_TO_UPPER.items():
        path = roi_root / f"{lower}_roi_target_manifest.parquet"
        manifest = _read_parquet(path).copy()
        manifest["language"] = upper
        manifest["roi_timeseries_path"] = manifest["filepath"].map(str)
        manifest["bold_path"] = manifest["bold_filepath"].map(str)
        manifest["atlas_path"] = str(atlas_path)
        manifest["roi_metadata_path"] = str(roi_metadata_path)
        manifest["source_artifact"] = str(path)
        manifests.append(
            manifest.loc[
                :,
                [
                    "subject_id",
                    "language",
                    "canonical_run_index",
                    "roi_timeseries_path",
                    "bold_path",
                    "n_scans",
                    "n_rois",
                    "atlas_path",
                    "roi_metadata_path",
                    "source_artifact",
                ],
            ].rename(columns={"canonical_run_index": "canonical_run"})
        )
    combined = pd.concat(manifests, ignore_index=True).sort_values(
        ["language", "subject_id", "canonical_run"]
    ).reset_index(drop=True)
    return combined, roi_metadata.sort_values("roi_index").reset_index(drop=True)


def _build_sample_manifest(canonical_run_df: pd.DataFrame, roi_manifest_df: pd.DataFrame) -> pd.DataFrame:
    run_summary = (
        canonical_run_df.groupby(["language", "subject_id"], as_index=False)
        .agg(
            n_canonical_runs=("canonical_run", "nunique"),
            all_runs_valid=("is_valid", "all"),
            n_trs_total=("n_trs", "sum"),
        )
    )
    roi_summary = (
        roi_manifest_df.groupby(["language", "subject_id"], as_index=False)
        .agg(
            n_roi_runs=("canonical_run", "nunique"),
            n_rois_observed=("n_rois", "max"),
            n_scans_total=("n_scans", "sum"),
        )
    )
    sample = run_summary.merge(roi_summary, on=["language", "subject_id"], how="outer")
    sample["n_canonical_runs"] = sample["n_canonical_runs"].fillna(0).astype(int)
    sample["n_roi_runs"] = sample["n_roi_runs"].fillna(0).astype(int)
    sample["all_runs_valid"] = sample["all_runs_valid"].fillna(False).astype(bool)
    sample["n_rois_observed"] = sample["n_rois_observed"].fillna(0).astype(int)
    sample["n_trs_total"] = sample["n_trs_total"].fillna(0).astype(int)
    sample["n_scans_total"] = sample["n_scans_total"].fillna(0).astype(int)
    sample["included"] = (
        (sample["n_canonical_runs"] == 9)
        & (sample["n_roi_runs"] == 9)
        & sample["all_runs_valid"]
        & (sample["n_rois_observed"] > 0)
    )
    sample["exclusion_reason"] = sample.apply(_exclusion_reason, axis=1)
    return sample.sort_values(["language", "subject_id"]).reset_index(drop=True)


def _exclusion_reason(row: pd.Series) -> str:
    if bool(row["included"]):
        return ""
    reasons: list[str] = []
    if int(row["n_canonical_runs"]) != 9:
        reasons.append(f"canonical_runs={int(row['n_canonical_runs'])}")
    if int(row["n_roi_runs"]) != 9:
        reasons.append(f"roi_runs={int(row['n_roi_runs'])}")
    if not bool(row["all_runs_valid"]):
        reasons.append("invalid_run_manifest_rows")
    if int(row["n_rois_observed"]) <= 0:
        reasons.append("missing_roi_targets")
    return ";".join(reasons)


def _write_provenance_notes(
    config: dict[str, Any],
    *,
    canonical_run_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    outputs: ManifestOutputs,
) -> None:
    prior_repo_root = _prior_repo_root(config)
    dataset_note_lines = [
        "# Dataset Fetch Provenance",
        "",
        f"- generated_utc: `{_now_utc()}`",
        f"- dataset: `LPPC / {config['dataset']['accession']}`",
        f"- baseline_source: `{prior_repo_root}`",
        "- mode: `path-based salvage from prior repo`",
        "- note: The sequel repo is reusing the externally stored prior baseline data and outputs rather than duplicating large assets locally.",
        "",
    ]
    outputs.dataset_provenance_path.write_text("\n".join(dataset_note_lines), encoding="utf-8")

    included_counts = (
        sample_df.loc[sample_df["included"]]
        .groupby("language")["subject_id"]
        .nunique()
        .to_dict()
    )
    total_counts = sample_df.groupby("language")["subject_id"].nunique().to_dict()
    sample_note_lines = [
        "# Sample Reconstruction",
        "",
        f"- generated_utc: `{_now_utc()}`",
        f"- prior_repo_root: `{prior_repo_root}`",
        "- reconstruction_rule: `reuse prior ROI target manifests plus canonical run manifest`",
        "- note: No single frozen sample manifest file was found in the prior repo; the sequel sample manifest was reconstructed from the prior run manifest and per-language ROI target manifests.",
        "",
        "## Counts",
        "",
    ]
    for language in ("EN", "FR", "ZH"):
        sample_note_lines.append(
            f"- {language}: included `{included_counts.get(language, 0)}` / discovered `{total_counts.get(language, 0)}`"
        )
    sample_note_lines.extend(
        [
            "",
            "## Canonical run assumptions",
            "",
            "- `story_section` was carried forward from the prior repo's `section_index`.",
            "- sentence-span and triplet tables were mapped to `canonical_run = section_index` because the prior repo's released artifacts are organized by the same nine LPPC sections used as canonical runs.",
            "",
            "## Included subject rule",
            "",
            "- included if and only if the subject had 9 canonical run-manifest rows, 9 ROI target rows, all run-manifest rows valid, and a positive ROI count.",
        ]
    )
    outputs.sample_provenance_path.write_text("\n".join(sample_note_lines), encoding="utf-8")


def build_manifests(config: dict[str, Any]) -> ManifestOutputs:
    manifests_root = _local_manifests_root(config)
    outputs_root = _local_outputs_root(config)
    manifests_root.mkdir(parents=True, exist_ok=True)
    (outputs_root / "provenance").mkdir(parents=True, exist_ok=True)

    dataset_manifest_path = manifests_root / "dataset_manifest.parquet"
    canonical_run_manifest_path = manifests_root / "canonical_run_manifest.parquet"
    sample_manifest_path = manifests_root / "sample_manifest.parquet"
    triplets_manifest_path = manifests_root / "triplets.parquet"
    roi_manifest_path = manifests_root / "roi_manifest.parquet"
    sample_provenance_path = outputs_root / "provenance" / "sample_reconstruction.md"
    dataset_provenance_path = outputs_root / "provenance" / "dataset_fetch.md"

    dataset_manifest_df = _build_dataset_manifest(config)
    canonical_run_df = _transform_canonical_run_manifest(config)
    triplets_df = _transform_triplets(config)
    roi_manifest_df, roi_metadata_df = _combine_roi_manifests(config)
    sample_df = _build_sample_manifest(canonical_run_df, roi_manifest_df)

    dataset_manifest_df.to_parquet(dataset_manifest_path, index=False)
    canonical_run_df.to_parquet(canonical_run_manifest_path, index=False)
    sample_df.to_parquet(sample_manifest_path, index=False)
    triplets_df.to_parquet(triplets_manifest_path, index=False)
    roi_manifest_df.to_parquet(roi_manifest_path, index=False)
    roi_metadata_df.to_parquet(manifests_root / "roi_metadata.parquet", index=False)

    for language in ("EN", "FR", "ZH"):
        spans_df = _transform_sentence_spans(config, language)
        spans_df.to_parquet(manifests_root / f"sentence_spans_{language}.parquet", index=False)

    outputs = ManifestOutputs(
        dataset_manifest_path=dataset_manifest_path,
        canonical_run_manifest_path=canonical_run_manifest_path,
        sample_manifest_path=sample_manifest_path,
        triplets_manifest_path=triplets_manifest_path,
        roi_manifest_path=roi_manifest_path,
        sample_provenance_path=sample_provenance_path,
        dataset_provenance_path=dataset_provenance_path,
    )
    _write_provenance_notes(config, canonical_run_df=canonical_run_df, sample_df=sample_df, outputs=outputs)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sequel manifests by salvaging the prior repo baseline.")
    parser.add_argument(
        "--config",
        default="conf/base.yaml",
        help="Path to the sequel base config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    outputs = build_manifests(config)
    print(f"Wrote dataset manifest to {outputs.dataset_manifest_path}")
    print(f"Wrote canonical run manifest to {outputs.canonical_run_manifest_path}")
    print(f"Wrote sample manifest to {outputs.sample_manifest_path}")
    print(f"Wrote triplets manifest to {outputs.triplets_manifest_path}")
    print(f"Wrote ROI manifest to {outputs.roi_manifest_path}")
    print(f"Wrote dataset provenance to {outputs.dataset_provenance_path}")
    print(f"Wrote sample provenance to {outputs.sample_provenance_path}")


if __name__ == "__main__":
    main()
