from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.encoding.design_matrix_operator import LanguageSentenceToBoldOperator, build_language_sentence_to_bold_operator
from src.encoding.multi_target_ridge import alpha_grid, run_nested_cv_multi_target
from src.utils.config import load_config


LANGUAGE_SPECS = {
    "EN": {
        "annotation_dir": "EN",
        "prefix": "lppEN",
        "onset_col": "en_onset_sec",
        "offset_col": "en_offset_sec",
        "picture_regressors": True,
    },
    "FR": {
        "annotation_dir": "FR",
        "prefix": "lppFR",
        "onset_col": "fr_onset_sec",
        "offset_col": "fr_offset_sec",
        "picture_regressors": False,
    },
    "ZH": {
        "annotation_dir": "CN",
        "prefix": "lppCN",
        "onset_col": "zh_onset_sec",
        "offset_col": "zh_offset_sec",
        "picture_regressors": True,
    },
}
PRIMARY_CONDITIONS = ("SHARED", "SPECIFIC")
PRIMARY_STATES = ("ATTN", "POST_ATTN", "FFN", "OUTPUT")
GROUP_RESULT_COLUMNS = [
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
    "ci_low",
    "ci_high",
    "n_subjects",
]


@dataclass(frozen=True, slots=True)
class PreparedLanguageSweep:
    language: str
    run_order: tuple[int, ...]
    operator: LanguageSentenceToBoldOperator
    run_ids: np.ndarray
    target_matrix: np.ndarray
    nuisance_matrix: np.ndarray
    target_metadata: pd.DataFrame
    n_timepoints_total: int


def _project_seed(config: dict[str, Any]) -> int:
    return int(config.get("project", {}).get("seed", 1337))


def _local_outputs_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["local_outputs_root"]).resolve()


def _local_manifests_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["local_manifests_root"]).resolve()


def _state_feature_manifest_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "caches" / "derived_features" / "state_feature_manifest.parquet"


def _tagged_path(path: Path, output_tag: str | None) -> Path:
    if not output_tag:
        return path
    suffix = "".join(path.suffixes)
    stem = path.name[: -len(suffix)] if suffix else path.name
    return path.with_name(f"{stem}__{output_tag}{suffix}")


def _subject_results_path(config: dict[str, Any], output_tag: str | None = None) -> Path:
    path = _local_outputs_root(config) / "subject_results" / "state_sweep_subject_results.parquet"
    return _tagged_path(path, output_tag)


def _group_results_path(config: dict[str, Any], output_tag: str | None = None) -> Path:
    path = _local_outputs_root(config) / "group_results" / "state_sweep_group_results.parquet"
    return _tagged_path(path, output_tag)


def _provenance_path(config: dict[str, Any], output_tag: str | None = None) -> Path:
    path = _local_outputs_root(config) / "provenance" / "state_sweep.md"
    return _tagged_path(path, output_tag)


def _triplets_path(config: dict[str, Any]) -> Path:
    return _local_manifests_root(config) / "triplets.parquet"


def _sample_manifest_path(config: dict[str, Any]) -> Path:
    return _local_manifests_root(config) / "sample_manifest.parquet"


def _roi_manifest_path(config: dict[str, Any]) -> Path:
    return _local_manifests_root(config) / "roi_manifest.parquet"


def _roi_metadata_path(config: dict[str, Any]) -> Path:
    return _local_manifests_root(config) / "roi_metadata.parquet"


def _read_csv_auto(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python")


def _annotation_paths(config: dict[str, Any], language: str) -> tuple[Path, Path]:
    spec = LANGUAGE_SPECS[language]
    root = Path(config["paths"]["prior_data_root"]).resolve() / "raw" / "ds003643" / "annotation" / spec["annotation_dir"]
    return root / f"{spec['prefix']}_prosody.csv", root / f"{spec['prefix']}_word_information.csv"


def _load_annotation_tables(config: dict[str, Any], language: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    prosody_path, word_info_path = _annotation_paths(config, language)
    return _read_csv_auto(prosody_path).copy(), _read_csv_auto(word_info_path).copy()


def _run_scan_counts(roi_manifest: pd.DataFrame) -> dict[int, int]:
    counts = roi_manifest.groupby("canonical_run")["n_scans"].unique()
    out: dict[int, int] = {}
    for run_index, values in counts.items():
        unique = np.unique(values.astype(int))
        if len(unique) != 1:
            raise RuntimeError(f"Run {run_index} has inconsistent scan counts across subjects: {unique}")
        out[int(run_index)] = int(unique[0])
    return out


def _impulse_series(onsets: np.ndarray, *, n_scans: int, tr: float, fine_hz: float) -> np.ndarray:
    grid_len = int(np.ceil(n_scans * tr * fine_hz))
    series = np.zeros(grid_len, dtype=np.float32)
    for onset in onsets:
        idx = int(np.clip(np.round(float(onset) * fine_hz), 0, grid_len - 1))
        series[idx] += 1.0
    return series


def _boxcar_series(
    onsets: np.ndarray,
    durations: np.ndarray,
    *,
    n_scans: int,
    tr: float,
    fine_hz: float,
) -> np.ndarray:
    grid_len = int(np.ceil(n_scans * tr * fine_hz))
    series = np.zeros(grid_len, dtype=np.float32)
    for onset, duration in zip(onsets, durations, strict=False):
        start = int(np.clip(np.floor(float(onset) * fine_hz), 0, grid_len - 1))
        stop = int(np.clip(np.ceil((float(onset) + float(duration)) * fine_hz), start + 1, grid_len))
        series[start:stop] = 1.0
    return series


def _continuous_series(
    times: np.ndarray,
    values: np.ndarray,
    *,
    n_scans: int,
    tr: float,
    fine_hz: float,
) -> np.ndarray:
    grid_len = int(np.ceil(n_scans * tr * fine_hz))
    fine_times = np.arange(grid_len, dtype=np.float32) / fine_hz
    return np.interp(fine_times, times.astype(np.float32), values.astype(np.float32), left=0.0, right=0.0)


def _convolve_and_sample(series: np.ndarray, *, n_scans: int, tr: float, fine_hz: float, hrf_kernel: np.ndarray) -> np.ndarray:
    convolved = np.convolve(series, hrf_kernel, mode="full")
    scan_times = np.arange(n_scans, dtype=np.float32) * tr
    sample_idx = np.clip(np.round(scan_times * fine_hz).astype(int), 0, len(convolved) - 1)
    return convolved[sample_idx].astype(np.float32, copy=False)


def _highpass_basis(n_scans: int, tr: float, cutoff_sec: float) -> np.ndarray:
    total_duration = n_scans * tr
    n_harmonics = int(np.floor(2.0 * total_duration / cutoff_sec))
    if n_harmonics <= 0:
        return np.zeros((n_scans, 0), dtype=np.float32)
    n = np.arange(n_scans, dtype=np.float32)[:, None]
    k = np.arange(1, n_harmonics + 1, dtype=np.float32)[None, :]
    return np.cos(np.pi * (n + 0.5) * k / n_scans).astype(np.float32, copy=False)


def _build_run_nuisance_and_acoustic(
    config: dict[str, Any],
    *,
    language: str,
    run_index: int,
    n_scans: int,
    sentence_onsets: np.ndarray,
    prosody: pd.DataFrame,
    word_info: pd.DataFrame,
    hrf_kernel: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = LANGUAGE_SPECS[language]
    tr = float(config["dataset"]["tr_seconds"])
    fine_hz = float(config["features"]["fine_hz"])
    prosody_run = prosody.loc[prosody["section"].astype(int) == run_index].copy()
    word_run = word_info.loc[word_info["section"].astype(int) == run_index].copy()

    rms = _convolve_and_sample(
        _continuous_series(prosody_run["time"].to_numpy(), prosody_run["intensity"].to_numpy(), n_scans=n_scans, tr=tr, fine_hz=fine_hz),
        n_scans=n_scans,
        tr=tr,
        fine_hz=fine_hz,
        hrf_kernel=hrf_kernel,
    )
    f0 = _convolve_and_sample(
        _continuous_series(prosody_run["time"].to_numpy(), prosody_run["f0"].to_numpy(), n_scans=n_scans, tr=tr, fine_hz=fine_hz),
        n_scans=n_scans,
        tr=tr,
        fine_hz=fine_hz,
        hrf_kernel=hrf_kernel,
    )
    word_rate = _convolve_and_sample(
        _impulse_series(word_run["onset"].to_numpy(), n_scans=n_scans, tr=tr, fine_hz=fine_hz),
        n_scans=n_scans,
        tr=tr,
        fine_hz=fine_hz,
        hrf_kernel=hrf_kernel,
    )
    sentence_impulse = _convolve_and_sample(
        _impulse_series(sentence_onsets, n_scans=n_scans, tr=tr, fine_hz=fine_hz),
        n_scans=n_scans,
        tr=tr,
        fine_hz=fine_hz,
        hrf_kernel=hrf_kernel,
    )

    if spec["picture_regressors"] and run_index == 1:
        picture_onsets = np.array([10.0, 35.0, 60.0], dtype=np.float32)
        picture_durations = np.array([15.0, 20.0, 15.0], dtype=np.float32)
        picture_event = _convolve_and_sample(
            _impulse_series(picture_onsets, n_scans=n_scans, tr=tr, fine_hz=fine_hz),
            n_scans=n_scans,
            tr=tr,
            fine_hz=fine_hz,
            hrf_kernel=hrf_kernel,
        )
        picture_block = _convolve_and_sample(
            _boxcar_series(picture_onsets, picture_durations, n_scans=n_scans, tr=tr, fine_hz=fine_hz),
            n_scans=n_scans,
            tr=tr,
            fine_hz=fine_hz,
            hrf_kernel=hrf_kernel,
        )
    else:
        picture_event = np.zeros(n_scans, dtype=np.float32)
        picture_block = np.zeros(n_scans, dtype=np.float32)

    acoustic_df = pd.DataFrame(
        {
            "rms": rms,
            "f0": f0,
            "word_rate": word_rate,
            "sentence_onset": sentence_impulse,
            "picture_event": picture_event,
            "picture_block": picture_block,
        }
    )
    trend = np.linspace(-1.0, 1.0, n_scans, dtype=np.float32)
    highpass = _highpass_basis(
        n_scans,
        tr=tr,
        cutoff_sec=float(config.get("nuisance", {}).get("highpass_cutoff_sec", 128.0)),
    )
    nuisance_columns: dict[str, np.ndarray] = {
        "intercept": np.ones(n_scans, dtype=np.float32),
        "linear_trend": trend,
    }
    for idx in range(highpass.shape[1]):
        nuisance_columns[f"highpass_{idx:02d}"] = highpass[:, idx]
    nuisance_df = pd.DataFrame(nuisance_columns)
    return nuisance_df, acoustic_df


def _align_run_frame_columns(frames_by_run: dict[int, pd.DataFrame]) -> dict[int, pd.DataFrame]:
    if not frames_by_run:
        return {}
    all_columns = sorted({str(column) for frame in frames_by_run.values() for column in frame.columns})
    return {
        int(run_index): frame.reindex(columns=all_columns, fill_value=0.0).copy()
        for run_index, frame in frames_by_run.items()
    }


def _prepare_language_sweep(
    config: dict[str, Any],
    *,
    language: str,
    max_subjects: int | None = None,
) -> PreparedLanguageSweep:
    triplets = pd.read_parquet(_triplets_path(config)).sort_values("triplet_id").reset_index(drop=True)
    triplets["triplet_row_index"] = np.arange(len(triplets), dtype=np.int64)
    sample_manifest = pd.read_parquet(_sample_manifest_path(config)).copy()
    roi_manifest = pd.read_parquet(_roi_manifest_path(config)).copy()
    roi_metadata = pd.read_parquet(_roi_metadata_path(config)).copy()

    included_subjects = (
        sample_manifest.loc[
            (sample_manifest["language"].astype(str) == language) & sample_manifest["included"].astype(bool),
            "subject_id",
        ]
        .astype(str)
        .sort_values()
        .tolist()
    )
    if max_subjects is not None:
        included_subjects = included_subjects[: int(max_subjects)]
    if not included_subjects:
        raise RuntimeError(f"No included subjects found for language={language}")

    roi_manifest = roi_manifest.loc[
        (roi_manifest["language"].astype(str) == language)
        & (roi_manifest["subject_id"].astype(str).isin(included_subjects))
    ].copy()
    roi_metadata = roi_metadata.sort_values("roi_index").reset_index(drop=True)
    scan_counts_by_run = _run_scan_counts(roi_manifest)
    spec = LANGUAGE_SPECS[language]
    operator = build_language_sentence_to_bold_operator(
        triplets,
        language=language,
        scan_counts_by_run=scan_counts_by_run,
        onset_column=spec["onset_col"],
        offset_column=spec["offset_col"],
        tr_seconds=float(config["dataset"]["tr_seconds"]),
        fine_hz=float(config["features"]["fine_hz"]),
    )

    prosody, word_info = _load_annotation_tables(config, language)
    nuisance_frames_by_run: dict[int, pd.DataFrame] = {}
    acoustic_frames_by_run: dict[int, pd.DataFrame] = {}
    run_order = tuple(sorted(scan_counts_by_run))
    for run_index in run_order:
        run_triplets = triplets.loc[triplets["canonical_run"].astype(int) == int(run_index)].copy()
        nuisance_df, acoustic_df = _build_run_nuisance_and_acoustic(
            config,
            language=language,
            run_index=int(run_index),
            n_scans=int(scan_counts_by_run[int(run_index)]),
            sentence_onsets=run_triplets[spec["onset_col"]].to_numpy(dtype=np.float32),
            prosody=prosody,
            word_info=word_info,
            hrf_kernel=operator.hrf_kernel,
        )
        nuisance_frames_by_run[int(run_index)] = nuisance_df
        acoustic_frames_by_run[int(run_index)] = acoustic_df

    nuisance_frames_by_run = _align_run_frame_columns(nuisance_frames_by_run)
    acoustic_frames_by_run = _align_run_frame_columns(acoustic_frames_by_run)
    nuisance_by_run = {
        int(run_index): np.column_stack(
            [
                nuisance_frames_by_run[int(run_index)].to_numpy(dtype=np.float32, copy=False),
                acoustic_frames_by_run[int(run_index)].to_numpy(dtype=np.float32, copy=False),
            ]
        ).astype(np.float32, copy=False)
        for run_index in run_order
    }

    roi_names = roi_metadata["roi_name"].astype(str).tolist()
    roi_families = roi_metadata["roi_family"].astype(str).tolist()
    run_target_blocks: dict[int, list[np.ndarray]] = {run_index: [] for run_index in run_order}
    target_rows: list[dict[str, str]] = []
    for subject_id in included_subjects:
        subject_runs = roi_manifest.loc[roi_manifest["subject_id"].astype(str) == str(subject_id)].sort_values("canonical_run")
        run_arrays = {
            int(row.canonical_run): np.load(row.roi_timeseries_path).astype(np.float32, copy=False)
            for row in subject_runs.itertuples(index=False)
        }
        missing_runs = sorted(set(run_order).difference(run_arrays))
        if missing_runs:
            raise RuntimeError(f"Subject {subject_id} missing ROI run arrays for language={language}: {missing_runs}")
        for run_index in run_order:
            run_target_blocks[run_index].append(run_arrays[run_index])
        for roi_name, roi_family in zip(roi_names, roi_families, strict=True):
            target_rows.append(
                {
                    "subject_id": str(subject_id),
                    "roi": str(roi_name),
                    "roi_family": str(roi_family),
                }
            )

    target_matrix = np.vstack(
        [np.concatenate(run_target_blocks[run_index], axis=1) for run_index in run_order]
    ).astype(np.float32, copy=False)
    nuisance_matrix = np.vstack([nuisance_by_run[run_index] for run_index in run_order]).astype(np.float32, copy=False)
    run_ids = np.concatenate(
        [np.full(int(scan_counts_by_run[run_index]), int(run_index), dtype=np.int64) for run_index in run_order]
    )
    target_metadata = pd.DataFrame(target_rows)
    return PreparedLanguageSweep(
        language=language,
        run_order=run_order,
        operator=operator,
        run_ids=run_ids,
        target_matrix=target_matrix,
        nuisance_matrix=nuisance_matrix,
        target_metadata=target_metadata,
        n_timepoints_total=int(len(run_ids)),
    )


def _condition_design_matrix(
    prepared: PreparedLanguageSweep,
    *,
    feature_array: np.ndarray,
) -> np.ndarray:
    run_designs = prepared.operator.transform_feature_matrix(feature_array)
    return np.vstack([run_designs[run_index] for run_index in prepared.run_order]).astype(np.float32, copy=False)


def _build_subject_rows(
    prepared: PreparedLanguageSweep,
    *,
    model: str,
    block: int,
    block_depth_norm: float,
    state: str,
    condition: str,
    summary: Any,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for target_index, target_row in enumerate(prepared.target_metadata.itertuples(index=False)):
        rows.append(
            {
                "subject_id": str(target_row.subject_id),
                "language": prepared.language,
                "model": model,
                "block": int(block),
                "block_depth_norm": float(block_depth_norm),
                "state": state,
                "condition": condition,
                "roi": str(target_row.roi),
                "roi_family": str(target_row.roi_family),
                "z_mean": float(summary.z_mean[target_index]),
                "r_mean": float(summary.r_mean[target_index]),
                "r2_concat": float(summary.r2_concat[target_index]),
                "alpha_best_mode": float(summary.alpha_best_mode[target_index]),
                "alpha_best_mean_log10": float(summary.alpha_best_mean_log10[target_index]),
                "n_pcs_mean": float(summary.n_pcs_mean[target_index]),
                "n_outer_folds": int(len(summary.outer_runs)),
                "n_timepoints_total": int(prepared.n_timepoints_total),
                "seed": int(seed),
            }
        )
    return rows


def _group_subject_results(subject_df: pd.DataFrame) -> pd.DataFrame:
    if subject_df.empty:
        return pd.DataFrame(columns=GROUP_RESULT_COLUMNS)

    group_rows: list[dict[str, object]] = []
    grouped = subject_df.groupby(
        ["language", "model", "block", "block_depth_norm", "state", "condition", "roi", "roi_family"],
        as_index=False,
    )
    for _, group_df in grouped:
        group_rows.append(
            {
                "language": str(group_df["language"].iloc[0]),
                "model": str(group_df["model"].iloc[0]),
                "block": int(group_df["block"].iloc[0]),
                "block_depth_norm": float(group_df["block_depth_norm"].iloc[0]),
                "state": str(group_df["state"].iloc[0]),
                "condition": str(group_df["condition"].iloc[0]),
                "roi": str(group_df["roi"].iloc[0]),
                "roi_family": str(group_df["roi_family"].iloc[0]),
                "mean_z": float(group_df["z_mean"].mean()),
                "mean_r": float(group_df["r_mean"].mean()),
                "ci_low": np.nan,
                "ci_high": np.nan,
                "n_subjects": int(group_df["subject_id"].nunique()),
            }
        )
    return pd.DataFrame(group_rows, columns=GROUP_RESULT_COLUMNS).sort_values(
        ["language", "model", "roi", "block", "state", "condition"]
    ).reset_index(drop=True)


def _validate_feature_manifest(feature_manifest: pd.DataFrame) -> None:
    required_columns = {
        "model",
        "target_language",
        "state",
        "condition",
        "block",
        "block_depth_norm",
        "source_array_path",
    }
    missing_columns = sorted(required_columns.difference(feature_manifest.columns))
    if missing_columns:
        raise RuntimeError(
            "State feature manifest is missing required columns: "
            f"{', '.join(missing_columns)}"
        )


def run_state_sweep(
    config: dict[str, Any],
    *,
    models: Sequence[str] = ("xlmr", "nllb"),
    languages: Sequence[str] = ("EN", "FR", "ZH"),
    states: Sequence[str] = PRIMARY_STATES,
    conditions: Sequence[str] = PRIMARY_CONDITIONS,
    max_subjects: int | None = None,
    output_tag: str | None = None,
) -> dict[str, Path]:
    state_feature_manifest_path = _state_feature_manifest_path(config)
    if not state_feature_manifest_path.exists():
        raise FileNotFoundError(
            f"State feature manifest missing: {state_feature_manifest_path}. Run src.features.build_shared_specific first."
        )

    feature_manifest = pd.read_parquet(state_feature_manifest_path).copy()
    _validate_feature_manifest(feature_manifest)
    selected_models = tuple(dict.fromkeys(str(model) for model in models))
    selected_languages = tuple(dict.fromkeys(str(language).upper() for language in languages))
    selected_states = tuple(dict.fromkeys(str(state).upper() for state in states))
    selected_conditions = tuple(dict.fromkeys(str(condition).upper() for condition in conditions))

    feature_manifest = feature_manifest.loc[
        feature_manifest["model"].astype(str).isin(selected_models)
        & feature_manifest["target_language"].astype(str).isin(selected_languages)
        & feature_manifest["state"].astype(str).isin(selected_states)
        & feature_manifest["condition"].astype(str).isin(selected_conditions)
    ].copy()
    if feature_manifest.empty:
        raise RuntimeError("No state-feature manifest rows matched the requested sweep scope.")

    prepared_by_language = {
        language: _prepare_language_sweep(config, language=language, max_subjects=max_subjects)
        for language in selected_languages
    }

    alpha_values = alpha_grid(min_log10=-2.0, max_log10=6.0, n_values=15)
    encoding_cfg = config.get("encoding", {})
    seed = _project_seed(config)
    subject_rows: list[dict[str, object]] = []

    ordered_feature_manifest = feature_manifest.sort_values(
        ["model", "target_language", "state", "block", "condition"]
    ).reset_index(drop=True)
    for row in ordered_feature_manifest.itertuples(index=False):
        prepared = prepared_by_language[str(row.target_language)]
        source_array_path = Path(row.source_array_path)
        if not source_array_path.exists():
            raise FileNotFoundError(f"Missing feature array referenced by manifest: {source_array_path}")
        feature_array = np.load(source_array_path).astype(np.float32, copy=False)
        design_matrix = _condition_design_matrix(prepared, feature_array=feature_array)
        summary = run_nested_cv_multi_target(
            run_ids=prepared.run_ids,
            X=design_matrix,
            Y=prepared.target_matrix,
            Z=prepared.nuisance_matrix,
            alpha_values=alpha_values,
            max_pcs=int(encoding_cfg.get("max_pcs", 128)),
            variance_threshold=float(encoding_cfg.get("variance_threshold", 0.95)),
            random_state=seed,
            target_chunk_size=int(encoding_cfg.get("target_chunk_size", 512)),
        )
        subject_rows.extend(
            _build_subject_rows(
                prepared,
                model=str(row.model),
                block=int(row.block),
                block_depth_norm=float(row.block_depth_norm),
                state=str(row.state),
                condition=str(row.condition),
                summary=summary,
                seed=seed,
            )
        )

    if not subject_rows:
        raise RuntimeError("State sweep produced no subject rows after encoding.")

    subject_df = pd.DataFrame(subject_rows).sort_values(
        ["language", "model", "subject_id", "roi", "block", "state", "condition"]
    ).reset_index(drop=True)
    group_df = _group_subject_results(subject_df)

    subject_path = _subject_results_path(config, output_tag)
    group_path = _group_results_path(config, output_tag)
    subject_path.parent.mkdir(parents=True, exist_ok=True)
    group_path.parent.mkdir(parents=True, exist_ok=True)
    subject_df.to_parquet(subject_path, index=False)
    group_df.to_parquet(group_path, index=False)

    provenance_path = _provenance_path(config, output_tag)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        "\n".join(
            [
                "# State Sweep",
                "",
                f"- subject_results: `{subject_path}`",
                f"- group_results: `{group_path}`",
                f"- models: `{', '.join(selected_models)}`",
                f"- languages: `{', '.join(selected_languages)}`",
                f"- states: `{', '.join(selected_states)}`",
                f"- conditions: `{', '.join(selected_conditions)}`",
                f"- max_subjects: `{max_subjects if max_subjects is not None else 'all included'}`",
                f"- output_tag: `{output_tag or 'canonical'}`",
                f"- alpha_grid: `10^[-2,6]` with `15` values",
                "- nuisance: intercept, trend, high-pass, RMS, f0, word rate, sentence onset, picture regressors where applicable",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "subject_results": subject_path,
        "group_results": group_path,
        "provenance": provenance_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the sequel-native internal-state sweep.")
    parser.add_argument("--config", default="conf/base.yaml", help="Path to sequel config.")
    parser.add_argument("--models", nargs="+", default=["xlmr", "nllb"], choices=["xlmr", "nllb"])
    parser.add_argument("--languages", nargs="+", default=["EN", "FR", "ZH"], choices=["EN", "FR", "ZH", "en", "fr", "zh"])
    parser.add_argument("--states", nargs="+", default=list(PRIMARY_STATES), choices=list(PRIMARY_STATES) + ["INPUT"])
    parser.add_argument("--conditions", nargs="+", default=list(PRIMARY_CONDITIONS), choices=["SHARED", "SPECIFIC", "RAW", "FULL"])
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--output-tag", default=None, help="Optional shard tag for subject/group/provenance outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    outputs = run_state_sweep(
        config,
        models=tuple(args.models),
        languages=tuple(args.languages),
        states=tuple(args.states),
        conditions=tuple(args.conditions),
        max_subjects=args.max_subjects,
        output_tag=args.output_tag,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
