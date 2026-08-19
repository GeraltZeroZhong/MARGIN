from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from margin.constants import AA_TO_INDEX
from margin.studies.stability.position_specificity import evaluate_position_specificity


def test_final_cplus_position_shuffle_preserves_components_and_detects_position_signal(
    tmp_path: Path,
) -> None:
    query_rows = []
    variant_rows = []
    values = []
    for domain_index, (domain, stratum) in enumerate(
        (("domain_a", "natural"), ("domain_b", "de_novo"))
    ):
        for position in range(8):
            query_rows.append(
                {
                    "domain_id": domain,
                    "position": position,
                    "wild_type": "A",
                }
            )
            effect = float(position + domain_index / 10)
            variant_rows.append(
                {
                    "domain_id": domain,
                    "position": position,
                    "wild_type": "A",
                    "mutant": "C",
                    "effect": effect,
                    "stratum": stratum,
                    "evaluation_population": "megascale_stability_dense",
                    "sequence_action": 0.0,
                    "joint_temperature_native_nll_action": effect,
                }
            )
            values.append(effect)

    shape = (len(query_rows), len(AA_TO_INDEX))
    g = np.zeros(shape, dtype=float)
    c_plus = np.zeros(shape, dtype=float)
    u_plus = np.zeros(shape, dtype=float)
    u_plus[:, AA_TO_INDEX["C"]] = np.asarray(values)
    matrices = tmp_path / "components.npz"
    np.savez_compressed(
        matrices,
        consensus_a=g + c_plus + u_plus,
        consensus_g=g,
        consensus_c_plus=c_plus,
        consensus_u_plus=u_plus,
    )

    tables = evaluate_position_specificity(
        pd.DataFrame(query_rows),
        pd.DataFrame(variant_rows),
        matrices,
        shuffle_repeats=20,
        bootstrap_replicates=200,
        seed=17,
    )

    assert len(tables["position_shuffle_repeats"]) == 40
    assert len(tables["position_shuffle_domains"]) == 2
    summary = tables["position_shuffle_summary"]
    spearman = summary.loc[
        summary["scope"].eq("all") & summary["metric"].eq("spearman_margin")
    ].iloc[0]
    assert spearman["estimate"] > 0.5
    assert spearman["positive_domains"] == 2
