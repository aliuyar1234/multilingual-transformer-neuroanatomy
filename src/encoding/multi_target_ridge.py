from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class FeatureStandardization:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return ((X - self.mean) / self.std).astype(np.float32, copy=False)

    def transform_delta(self, delta: np.ndarray) -> np.ndarray:
        delta = np.asarray(delta, dtype=np.float32)
        return (delta / self.std).astype(np.float32, copy=False)


@dataclass(frozen=True, slots=True)
class PCATransform:
    components: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return (X @ self.components.T).astype(np.float32, copy=False)


def alpha_grid(
    *,
    min_log10: float = -2.0,
    max_log10: float = 6.0,
    n_values: int = 15,
) -> np.ndarray:
    """Return the sequel ridge alpha grid."""

    return np.logspace(float(min_log10), float(max_log10), int(n_values)).astype(np.float64)


def _as_2d_targets(targets: np.ndarray) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.float32)
    if targets.ndim == 1:
        return targets[:, None]
    return targets


def fit_feature_standardization(X_train: np.ndarray) -> FeatureStandardization:
    X_train = np.asarray(X_train, dtype=np.float32)
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std < 1.0e-6, 1.0, std)
    return FeatureStandardization(
        mean=mean.astype(np.float32, copy=False),
        std=std.astype(np.float32, copy=False),
    )


def standardize_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, FeatureStandardization]:
    scaler = fit_feature_standardization(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), scaler


def _standardize_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    X_train_std, X_test_std, _ = standardize_train_test(X_train, X_test)
    return X_train_std, X_test_std


def residualize_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    Z_train: np.ndarray,
    Z_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Y_train = _as_2d_targets(Y_train)
    Y_test = _as_2d_targets(Y_test)
    beta_x, *_ = np.linalg.lstsq(Z_train, X_train, rcond=None)
    beta_y, *_ = np.linalg.lstsq(Z_train, Y_train, rcond=None)
    X_train_res = X_train - Z_train @ beta_x
    X_test_res = X_test - Z_test @ beta_x
    Y_train_res = Y_train - (Z_train @ beta_y)
    Y_test_res = Y_test - (Z_test @ beta_y)
    return X_train_res, X_test_res, Y_train_res, Y_test_res


def _residualize_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    Z_train: np.ndarray,
    Z_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return residualize_train_test(X_train, X_test, Y_train, Y_test, Z_train, Z_test)


def fit_pca_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    max_components: int,
    variance_threshold: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, PCATransform]:
    max_components = max(1, min(int(max_components), X_train.shape[0] - 1, X_train.shape[1]))
    try:
        from sklearn.decomposition import PCA

        pca = PCA(
            n_components=max_components,
            svd_solver="randomized",
            iterated_power=3,
            random_state=int(random_state),
        )
        X_train_full = pca.fit_transform(X_train).astype(np.float32, copy=False)
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        keep = int(np.searchsorted(cumulative, variance_threshold) + 1)
        keep = max(1, min(keep, max_components))
        components = pca.components_[:keep, :].astype(np.float32, copy=False)
        transform = PCATransform(components=components)
        return (
            X_train_full[:, :keep].astype(np.float32, copy=False),
            transform.transform(X_test),
            transform,
        )
    except ModuleNotFoundError:
        U, singular_values, Vt = np.linalg.svd(X_train, full_matrices=False)
        explained = (singular_values * singular_values) / max(1, X_train.shape[0] - 1)
        explained_ratio = explained / np.clip(explained.sum(), 1.0e-12, None)
        cumulative = np.cumsum(explained_ratio)
        keep = int(np.searchsorted(cumulative, variance_threshold) + 1)
        keep = max(1, min(keep, max_components, Vt.shape[0]))
        components = Vt[:keep, :].astype(np.float32, copy=False)
        transform = PCATransform(components=components)
        X_train_proj = (U[:, :keep] * singular_values[:keep]).astype(np.float32, copy=False)
        X_test_proj = transform.transform(X_test)
        return X_train_proj, X_test_proj, transform


def _fit_transform_pca(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    max_components: int,
    variance_threshold: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    X_train_pca, X_test_pca, _ = fit_pca_train_test(
        X_train,
        X_test,
        max_components=max_components,
        variance_threshold=variance_threshold,
        random_state=random_state,
    )
    return X_train_pca, X_test_pca


def _pearson_r_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = _as_2d_targets(y_true).astype(np.float64, copy=False)
    y_pred = _as_2d_targets(y_pred).astype(np.float64, copy=False)
    y_true = y_true - y_true.mean(axis=0, keepdims=True)
    y_pred = y_pred - y_pred.mean(axis=0, keepdims=True)
    denom = np.sqrt(np.sum(y_true * y_true, axis=0) * np.sum(y_pred * y_pred, axis=0))
    numer = np.sum(y_true * y_pred, axis=0)
    r = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)
    return np.clip(r, -0.999999, 0.999999)


def _ridge_svd_cache(X_train: np.ndarray, Y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    U, singular_values, Vt = np.linalg.svd(X_train, full_matrices=False)
    UtY = U.T @ Y_train
    return (
        singular_values.astype(np.float32, copy=False),
        Vt.astype(np.float32, copy=False),
        UtY.astype(np.float32, copy=False),
    )


def _ridge_predict_from_cache(
    cache: tuple[np.ndarray, np.ndarray, np.ndarray],
    X_test: np.ndarray,
    alpha: float,
    *,
    target_indices: np.ndarray | None = None,
) -> np.ndarray:
    singular_values, Vt, UtY = cache
    UtY_use = UtY if target_indices is None else UtY[:, target_indices]
    shrink = (singular_values / ((singular_values * singular_values) + float(alpha)))[:, None]
    coefficients = Vt.T @ (shrink * UtY_use)
    return (X_test @ coefficients).astype(np.float32, copy=False)


def ridge_coefficients(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    Y_train_2d = _as_2d_targets(Y_train)
    singular_values, Vt, UtY = _ridge_svd_cache(np.asarray(X_train, dtype=np.float32), Y_train_2d)
    shrink = (singular_values / ((singular_values * singular_values) + float(alpha)))[:, None]
    coefficients = (Vt.T @ (shrink * UtY)).astype(np.float32, copy=False)
    if np.asarray(Y_train).ndim == 1:
        return coefficients[:, 0]
    return coefficients


def _target_column_slices(n_targets: int, *, target_chunk_size: int) -> tuple[slice, ...]:
    return tuple(
        slice(start, min(start + target_chunk_size, n_targets))
        for start in range(0, n_targets, target_chunk_size)
    )


def _select_alpha_per_target(
    *,
    run_ids: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    alpha_values: np.ndarray,
    max_pcs: int,
    variance_threshold: float,
    random_state: int,
    target_chunk_size: int,
) -> np.ndarray:
    Y = _as_2d_targets(Y)
    target_slices = _target_column_slices(Y.shape[1], target_chunk_size=target_chunk_size)
    score_sum = np.zeros((len(alpha_values), Y.shape[1]), dtype=np.float64)
    inner_runs = sorted(np.unique(run_ids).tolist())

    for held_out_run in inner_runs:
        train_mask = run_ids != held_out_run
        test_mask = run_ids == held_out_run
        X_train_res, X_test_res, Y_train_res, Y_test_res = _residualize_train_test(
            X[train_mask],
            X[test_mask],
            Y[train_mask],
            Y[test_mask],
            Z[train_mask],
            Z[test_mask],
        )
        X_train_std, X_test_std = _standardize_train_test(X_train_res, X_test_res)
        X_train_pca, X_test_pca = _fit_transform_pca(
            X_train_std,
            X_test_std,
            max_components=max_pcs,
            variance_threshold=variance_threshold,
            random_state=random_state,
        )
        for target_slice in target_slices:
            Y_train_chunk = Y_train_res[:, target_slice]
            Y_test_chunk = Y_test_res[:, target_slice]
            cache = _ridge_svd_cache(X_train_pca, Y_train_chunk)
            for alpha_index, alpha in enumerate(alpha_values):
                predictions = _ridge_predict_from_cache(cache, X_test_pca, float(alpha))
                score_sum[alpha_index, target_slice] += np.arctanh(
                    _pearson_r_per_target(Y_test_chunk, predictions)
                )

    score_matrix = score_sum / len(inner_runs)
    alpha_by_target = np.empty(Y.shape[1], dtype=np.float64)
    for target_index in range(Y.shape[1]):
        best_score = float(score_matrix[:, target_index].max())
        alpha_by_target[target_index] = max(
            float(alpha)
            for alpha, score in zip(alpha_values, score_matrix[:, target_index], strict=False)
            if float(score) == best_score
        )
    return alpha_by_target


def select_alpha_per_target(
    *,
    run_ids: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    alpha_values: np.ndarray,
    max_pcs: int,
    variance_threshold: float,
    random_state: int,
    target_chunk_size: int,
) -> np.ndarray:
    return _select_alpha_per_target(
        run_ids=run_ids,
        X=X,
        Y=Y,
        Z=Z,
        alpha_values=alpha_values,
        max_pcs=max_pcs,
        variance_threshold=variance_threshold,
        random_state=random_state,
        target_chunk_size=target_chunk_size,
    )


def _mode_with_large_tie_break(values: np.ndarray) -> float:
    unique, counts = np.unique(values.astype(np.float64), return_counts=True)
    max_count = counts.max()
    return float(unique[counts == max_count].max())


@dataclass(frozen=True, slots=True)
class NestedMultiTargetRidgeSummary:
    """Summary outputs for one design matrix solved against many targets."""

    outer_runs: tuple[int, ...]
    r_mean: np.ndarray
    z_mean: np.ndarray
    r2_concat: np.ndarray
    alpha_best_mode: np.ndarray
    alpha_best_mean_log10: np.ndarray
    n_pcs_mean: np.ndarray
    fold_r: np.ndarray
    fold_z: np.ndarray
    fold_alpha: np.ndarray
    fold_n_pcs: np.ndarray


def run_nested_cv_multi_target(
    *,
    run_ids: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    alpha_values: Iterable[float] | None = None,
    max_pcs: int = 128,
    variance_threshold: float = 0.95,
    random_state: int = 1337,
    target_chunk_size: int = 512,
) -> NestedMultiTargetRidgeSummary:
    """Run the sequel nested CV ridge solve once for many targets."""

    run_ids = np.asarray(run_ids, dtype=np.int64)
    X = np.asarray(X, dtype=np.float32)
    Y = _as_2d_targets(Y)
    Z = np.asarray(Z, dtype=np.float32)
    if X.shape[0] != Y.shape[0] or X.shape[0] != Z.shape[0]:
        raise ValueError("X, Y, and Z must share the same number of rows.")
    if target_chunk_size < 1:
        raise ValueError("target_chunk_size must be positive.")

    alpha_values_array = (
        alpha_grid()
        if alpha_values is None
        else np.asarray(tuple(alpha_values), dtype=np.float64)
    )
    outer_runs = tuple(sorted(np.unique(run_ids).tolist()))
    target_slices = _target_column_slices(Y.shape[1], target_chunk_size=target_chunk_size)

    fold_r_rows: list[np.ndarray] = []
    fold_z_rows: list[np.ndarray] = []
    fold_alpha_rows: list[np.ndarray] = []
    fold_n_pcs: list[int] = []

    Y_sum = np.zeros(Y.shape[1], dtype=np.float64)
    Y_sq_sum = np.zeros(Y.shape[1], dtype=np.float64)
    sse_sum = np.zeros(Y.shape[1], dtype=np.float64)
    n_observations = 0

    for held_out_run in outer_runs:
        train_mask = run_ids != held_out_run
        test_mask = run_ids == held_out_run
        alpha_by_target = _select_alpha_per_target(
            run_ids=run_ids[train_mask],
            X=X[train_mask],
            Y=Y[train_mask],
            Z=Z[train_mask],
            alpha_values=alpha_values_array,
            max_pcs=max_pcs,
            variance_threshold=variance_threshold,
            random_state=random_state,
            target_chunk_size=target_chunk_size,
        )
        X_train_res, X_test_res, Y_train_res, Y_test_res = _residualize_train_test(
            X[train_mask],
            X[test_mask],
            Y[train_mask],
            Y[test_mask],
            Z[train_mask],
            Z[test_mask],
        )
        X_train_std, X_test_std = _standardize_train_test(X_train_res, X_test_res)
        X_train_pca, X_test_pca = _fit_transform_pca(
            X_train_std,
            X_test_std,
            max_components=max_pcs,
            variance_threshold=variance_threshold,
            random_state=random_state,
        )
        fold_n_pcs.append(int(X_train_pca.shape[1]))
        fold_predictions = np.empty_like(Y_test_res)
        for target_slice in target_slices:
            Y_train_chunk = Y_train_res[:, target_slice]
            alpha_chunk = alpha_by_target[target_slice]
            cache = _ridge_svd_cache(X_train_pca, Y_train_chunk)
            prediction_chunk = np.empty_like(Y_test_res[:, target_slice])
            for alpha in sorted({float(value) for value in alpha_chunk.tolist()}):
                target_indices = np.flatnonzero(alpha_chunk == alpha)
                prediction_chunk[:, target_indices] = _ridge_predict_from_cache(
                    cache,
                    X_test_pca,
                    alpha,
                    target_indices=target_indices,
                )
            fold_predictions[:, target_slice] = prediction_chunk

        fold_r = _pearson_r_per_target(Y_test_res, fold_predictions)
        fold_z = np.arctanh(fold_r)
        fold_r_rows.append(fold_r.astype(np.float64, copy=False))
        fold_z_rows.append(fold_z.astype(np.float64, copy=False))
        fold_alpha_rows.append(alpha_by_target.astype(np.float64, copy=False))

        Y_test_64 = Y_test_res.astype(np.float64, copy=False)
        predictions_64 = fold_predictions.astype(np.float64, copy=False)
        Y_sum += np.sum(Y_test_64, axis=0)
        Y_sq_sum += np.sum(Y_test_64 * Y_test_64, axis=0)
        residual = Y_test_64 - predictions_64
        sse_sum += np.sum(residual * residual, axis=0)
        n_observations += int(Y_test_res.shape[0])

    fold_r_array = np.vstack(fold_r_rows)
    fold_z_array = np.vstack(fold_z_rows)
    fold_alpha_array = np.vstack(fold_alpha_rows)
    fold_n_pcs_array = np.asarray(fold_n_pcs, dtype=np.float64)

    z_mean = fold_z_array.mean(axis=0)
    r_mean = np.tanh(z_mean)
    denom = Y_sq_sum - ((Y_sum * Y_sum) / float(n_observations))
    r2_concat = np.where(denom <= 0, 0.0, 1.0 - (sse_sum / denom))
    alpha_best_mode = np.asarray(
        [_mode_with_large_tie_break(fold_alpha_array[:, target_index]) for target_index in range(Y.shape[1])],
        dtype=np.float64,
    )
    alpha_best_mean_log10 = np.log10(fold_alpha_array).mean(axis=0)
    n_pcs_mean = np.full(Y.shape[1], fold_n_pcs_array.mean(), dtype=np.float64)

    return NestedMultiTargetRidgeSummary(
        outer_runs=outer_runs,
        r_mean=r_mean.astype(np.float64, copy=False),
        z_mean=z_mean.astype(np.float64, copy=False),
        r2_concat=r2_concat.astype(np.float64, copy=False),
        alpha_best_mode=alpha_best_mode,
        alpha_best_mean_log10=alpha_best_mean_log10.astype(np.float64, copy=False),
        n_pcs_mean=n_pcs_mean,
        fold_r=fold_r_array.astype(np.float64, copy=False),
        fold_z=fold_z_array.astype(np.float64, copy=False),
        fold_alpha=fold_alpha_array.astype(np.float64, copy=False),
        fold_n_pcs=fold_n_pcs_array,
    )
