from __future__ import annotations

import numpy as np
import pandas as pd

from margin.attribution.metrics import (
    cluster_bootstrap_mean,
    normalize_log_probabilities,
    rowwise_jsd,
    rowwise_topk_overlap,
)


def test_jsd_is_symmetric_and_zero_for_identical_distributions() -> None:
    left = normalize_log_probabilities(np.array([[2.0, 0.0, -1.0], [0.0, 1.0, 2.0]]))
    right = normalize_log_probabilities(np.array([[0.0, 2.0, -1.0], [2.0, 1.0, 0.0]]))
    np.testing.assert_allclose(rowwise_jsd(left, left), 0.0, atol=1e-12)
    np.testing.assert_allclose(rowwise_jsd(left, right), rowwise_jsd(right, left))
    assert (rowwise_jsd(left, right) > 0).all()


def test_topk_overlap_uses_candidate_sets() -> None:
    left = np.array([[5.0, 4.0, 3.0, 2.0]])
    right = np.array([[5.0, 1.0, 3.0, 4.0]])
    np.testing.assert_allclose(rowwise_topk_overlap(left, right, 2), [0.5])


def test_cluster_mean_weights_domains_not_positions() -> None:
    table = pd.DataFrame(
        {
            "domain_id": ["large"] * 100 + ["small"],
            "value": [1.0] * 100 + [3.0],
        }
    )
    estimate = cluster_bootstrap_mean(table, "value", "domain_id", 200, 0.95, 7)
    assert estimate["estimate"] == 2.0
    assert estimate["n_domains"] == 2
    assert estimate["n_rows"] == 101
