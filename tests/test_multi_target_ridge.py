from __future__ import annotations

import numpy as np

from src.encoding.multi_target_ridge import run_nested_cv_multi_target


def test_nested_multi_target_ridge_returns_expected_summary_shapes() -> None:
    rng = np.random.default_rng(1337)
    n_runs = 4
    per_run = 30
    n_samples = n_runs * per_run
    n_features = 8
    n_targets = 3

    run_ids = np.repeat(np.arange(1, n_runs + 1), per_run)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    nuisance = np.column_stack(
        [
            np.ones(n_samples, dtype=np.float32),
            np.linspace(-1.0, 1.0, n_samples, dtype=np.float32),
        ]
    )
    coefficients = rng.normal(size=(n_features, n_targets)).astype(np.float32)
    Y = (X @ coefficients) + 0.05 * rng.normal(size=(n_samples, n_targets)).astype(np.float32)

    summary = run_nested_cv_multi_target(
        run_ids=run_ids,
        X=X,
        Y=Y,
        Z=nuisance,
        max_pcs=6,
        target_chunk_size=2,
    )

    assert summary.fold_r.shape == (n_runs, n_targets)
    assert summary.fold_z.shape == (n_runs, n_targets)
    assert summary.fold_alpha.shape == (n_runs, n_targets)
    assert summary.r_mean.shape == (n_targets,)
    assert summary.z_mean.shape == (n_targets,)
    assert summary.r2_concat.shape == (n_targets,)
    assert summary.alpha_best_mode.shape == (n_targets,)
    assert summary.alpha_best_mean_log10.shape == (n_targets,)
    assert summary.n_pcs_mean.shape == (n_targets,)
    assert all(value > 0.5 for value in summary.r_mean.tolist())
