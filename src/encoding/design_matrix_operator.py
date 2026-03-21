from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


def default_hrf_kernel(*, fine_hz: float) -> np.ndarray:
    """Build the default HRF kernel for the sentence-to-BOLD operator."""

    from nilearn.glm.first_level.hemodynamic_models import glover_hrf

    return glover_hrf(t_r=1.0 / fine_hz, oversampling=1).astype(np.float32, copy=False)


def _build_run_sentence_basis(
    run_triplets: pd.DataFrame,
    *,
    onset_column: str,
    offset_column: str,
    n_scans: int,
    tr_seconds: float,
    fine_hz: float,
    hrf_kernel: np.ndarray,
) -> np.ndarray:
    """Build the fixed sentence-placement operator for one run."""

    grid_len = int(np.ceil(n_scans * tr_seconds * fine_hz))
    basis = np.zeros((grid_len, len(run_triplets)), dtype=np.float32)
    for column_index, row in enumerate(run_triplets.itertuples(index=False)):
        onset_idx = max(0, int(np.floor(float(getattr(row, onset_column)) * fine_hz)))
        offset_idx = max(onset_idx + 1, int(np.ceil(float(getattr(row, offset_column)) * fine_hz)))
        offset_idx = min(offset_idx, grid_len)
        basis[onset_idx:offset_idx, column_index] = 1.0

    convolved_len = grid_len + len(hrf_kernel) - 1
    convolved = np.empty((convolved_len, basis.shape[1]), dtype=np.float32)
    for column_index in range(basis.shape[1]):
        convolved[:, column_index] = np.convolve(
            basis[:, column_index],
            hrf_kernel,
            mode="full",
        ).astype(np.float32, copy=False)

    scan_times = np.arange(n_scans, dtype=np.float32) * tr_seconds
    sample_idx = np.clip(np.round(scan_times * fine_hz).astype(int), 0, convolved_len - 1)
    return convolved[sample_idx, :]


@dataclass(frozen=True, slots=True)
class LanguageSentenceToBoldOperator:
    """A reusable language-level operator from sentence rows to TR designs."""

    language: str
    onset_column: str
    offset_column: str
    run_column: str
    row_index_column: str
    tr_seconds: float
    fine_hz: float
    hrf_kernel: np.ndarray
    basis_by_run: dict[int, np.ndarray]
    row_indices_by_run: dict[int, np.ndarray]

    def transform_run_features(self, run_index: int, run_feature_matrix: np.ndarray) -> np.ndarray:
        """Apply the operator to one run-specific feature matrix."""

        basis = self.basis_by_run[int(run_index)]
        run_feature_matrix = np.asarray(run_feature_matrix, dtype=np.float32)
        if run_feature_matrix.ndim != 2:
            raise ValueError("Expected run_feature_matrix to be 2D.")
        if run_feature_matrix.shape[0] != basis.shape[1]:
            raise ValueError(
                "Run feature row count does not match the number of sentence rows "
                f"for run {run_index}: {run_feature_matrix.shape[0]} != {basis.shape[1]}"
            )
        return (basis @ run_feature_matrix).astype(np.float32, copy=False)

    def transform_feature_matrix(self, feature_matrix: np.ndarray) -> dict[int, np.ndarray]:
        """Apply the operator to a global sentence-feature matrix."""

        feature_matrix = np.asarray(feature_matrix, dtype=np.float32)
        if feature_matrix.ndim != 2:
            raise ValueError("Expected feature_matrix to be 2D.")

        run_designs: dict[int, np.ndarray] = {}
        for run_index, row_indices in self.row_indices_by_run.items():
            run_designs[int(run_index)] = self.transform_run_features(
                int(run_index),
                feature_matrix[row_indices, :],
            )
        return run_designs

    def transform_feature_matrices(
        self,
        feature_matrices: Mapping[str, np.ndarray],
    ) -> dict[str, dict[int, np.ndarray]]:
        """Apply the operator to many feature matrices that share sentence rows."""

        return {
            name: self.transform_feature_matrix(feature_matrix)
            for name, feature_matrix in feature_matrices.items()
        }


def build_language_sentence_to_bold_operator(
    triplets: pd.DataFrame,
    *,
    language: str,
    scan_counts_by_run: Mapping[int, int],
    onset_column: str,
    offset_column: str,
    tr_seconds: float,
    fine_hz: float,
    run_column: str = "canonical_run",
    row_index_column: str = "triplet_row_index",
    hrf_kernel: np.ndarray | None = None,
) -> LanguageSentenceToBoldOperator:
    """Precompute the fixed sentence-to-BOLD operator for one language."""

    if run_column not in triplets.columns:
        raise KeyError(f"Triplet table is missing run column {run_column!r}.")
    if onset_column not in triplets.columns:
        raise KeyError(f"Triplet table is missing onset column {onset_column!r}.")
    if offset_column not in triplets.columns:
        raise KeyError(f"Triplet table is missing offset column {offset_column!r}.")

    working_triplets = triplets.copy()
    if row_index_column not in working_triplets.columns:
        working_triplets[row_index_column] = np.arange(len(working_triplets), dtype=np.int64)

    kernel = (
        np.asarray(hrf_kernel, dtype=np.float32)
        if hrf_kernel is not None
        else default_hrf_kernel(fine_hz=fine_hz)
    )

    basis_by_run: dict[int, np.ndarray] = {}
    row_indices_by_run: dict[int, np.ndarray] = {}
    for run_index in sorted(int(value) for value in scan_counts_by_run):
        run_triplets = (
            working_triplets.loc[working_triplets[run_column].astype(int) == int(run_index)]
            .sort_values(row_index_column)
            .reset_index(drop=True)
        )
        if run_triplets.empty:
            continue

        basis_by_run[int(run_index)] = _build_run_sentence_basis(
            run_triplets,
            onset_column=onset_column,
            offset_column=offset_column,
            n_scans=int(scan_counts_by_run[int(run_index)]),
            tr_seconds=tr_seconds,
            fine_hz=fine_hz,
            hrf_kernel=kernel,
        )
        row_indices_by_run[int(run_index)] = run_triplets[row_index_column].to_numpy(dtype=np.int64)

    if not basis_by_run:
        raise RuntimeError(f"No sentence-to-BOLD operator rows were built for language={language!r}.")

    return LanguageSentenceToBoldOperator(
        language=language,
        onset_column=onset_column,
        offset_column=offset_column,
        run_column=run_column,
        row_index_column=row_index_column,
        tr_seconds=float(tr_seconds),
        fine_hz=float(fine_hz),
        hrf_kernel=kernel,
        basis_by_run=basis_by_run,
        row_indices_by_run=row_indices_by_run,
    )
