from __future__ import annotations

import numpy as np
import pandas as pd

from src.encoding.design_matrix_operator import build_language_sentence_to_bold_operator


def test_language_sentence_to_bold_operator_reuses_fixed_basis() -> None:
    triplets = pd.DataFrame(
        {
            "canonical_run": [1, 1, 2],
            "triplet_row_index": [0, 1, 2],
            "en_onset_sec": [0.0, 2.0, 1.0],
            "en_offset_sec": [2.0, 4.0, 3.0],
        }
    )
    operator = build_language_sentence_to_bold_operator(
        triplets,
        language="EN",
        scan_counts_by_run={1: 5, 2: 4},
        onset_column="en_onset_sec",
        offset_column="en_offset_sec",
        tr_seconds=1.0,
        fine_hz=1.0,
        hrf_kernel=np.array([1.0], dtype=np.float32),
    )

    feature_matrix = np.asarray(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ],
        dtype=np.float32,
    )
    run_designs = operator.transform_feature_matrix(feature_matrix)

    assert sorted(run_designs) == [1, 2]
    np.testing.assert_allclose(
        run_designs[1],
        np.asarray(
            [
                [1.0, 10.0],
                [1.0, 10.0],
                [2.0, 20.0],
                [2.0, 20.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        run_designs[2],
        np.asarray(
            [
                [0.0, 0.0],
                [3.0, 30.0],
                [3.0, 30.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
