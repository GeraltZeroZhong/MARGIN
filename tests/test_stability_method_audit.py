from __future__ import annotations

import numpy as np
import pandas as pd

from margin.studies.stability.method_audit import (
    TEACHERS,
    _blosum_action,
    _exact_teacher_shapley,
)


def test_blosum_action_is_anchored_to_wild_type() -> None:
    frame = pd.DataFrame(
        {
            "wild_type": ["A", "W", "D"],
            "mutant": ["A", "W", "E"],
        }
    )
    values = _blosum_action(frame)
    assert values[0] == 0
    assert values[1] == 0
    assert np.isfinite(values[2])


def test_exact_shapley_sums_to_grand_coalition_gain() -> None:
    subset_methods = {frozenset(): "none"}
    rows = []
    for mask in range(1 << len(TEACHERS)):
        subset = frozenset(teacher for index, teacher in enumerate(TEACHERS) if mask & (1 << index))
        method = "none" if not subset else "+".join(sorted(subset))
        subset_methods[subset] = method
        value = float(len(subset))
        rows.append(
            {
                "domain_id": "domain",
                "evaluation_population": "megascale_stability_dense",
                "method": method,
                "spearman": value,
                "ndcg_at_10_percent": 2 * value,
            }
        )
    result = _exact_teacher_shapley(pd.DataFrame(rows), subset_methods)
    for metric, expected in (("spearman", 3.0), ("ndcg_at_10_percent", 6.0)):
        selected = result.loc[result["metric"].eq(metric)]
        assert np.isclose(selected["shapley_value"].sum(), expected)
        assert np.allclose(selected["grand_coalition_gain"], expected)
