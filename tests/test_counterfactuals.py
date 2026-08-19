from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from margin.studies.counterfactuals.config import load_counterfactual_config
from margin.studies.counterfactuals.evaluation import (
    ROUTE_A_PRIMARY,
    ROUTE_B_PRIMARY,
    _ndcg,
    _spearman,
    _stabilizing_topk_recall,
    stratified_domain_bootstrap,
)
from margin.studies.counterfactuals.mechanism import read_aaindex1


def test_counterfactual_config_preserves_route_and_inference_lock() -> None:
    config = load_counterfactual_config(Path("configs/counterfactuals.yaml"))
    assert config.counterfactuals.primary_role == "contact_rewired_5"
    assert config.counterfactuals.replication_role == "circular_permuted"
    assert config.models.residual_alpha == 1.0
    assert config.inference.bootstrap_replicates == 5000
    assert ROUTE_A_PRIMARY != ROUTE_B_PRIMARY


def test_counterfactual_ranking_metrics_have_registered_direction() -> None:
    observed = np.array([-2.0, -1.0, 0.5, 1.0, 2.0])
    perfect = observed.copy()
    reversed_score = observed[::-1]
    assert np.isclose(_spearman(perfect, observed), 1.0)
    assert _ndcg(perfect, observed) > _ndcg(reversed_score, observed)
    assert _stabilizing_topk_recall(perfect, observed, 0.4) == 1.0


def test_stratified_bootstrap_uses_equal_domain_point_estimate() -> None:
    frame = pd.DataFrame(
        {
            "domain_id": ["n1", "n2", "d1", "d1"],
            "stratum": ["natural", "natural", "de_novo", "de_novo"],
            "value": [1.0, 3.0, -2.0, 2.0],
        }
    )
    result = stratified_domain_bootstrap(
        frame,
        "value",
        replicates=100,
        confidence_level=0.95,
        seed=7,
    )
    assert result["n_domains"] == 3
    assert np.isclose(result["estimate"], (1.0 + 3.0 + 0.0) / 3.0)


def test_aaindex_parser_restores_paired_header_order(tmp_path: Path) -> None:
    path = tmp_path / "aaindex1"
    path.write_text(
        "\n".join(
            [
                "H TEST000001",
                "D Minimal parser fixture",
                "I A/R N/D C/Q E/G H/I L/K M/F P/S T/W Y/V",
                " 1 2 3 4 5 6 7 8 9 10",
                " 11 12 13 14 15 16 17 18 19 20",
                "//",
            ]
        )
    )
    table = read_aaindex1(path)
    first = table.loc[table["accession"].eq("TEST000001")].iloc[0]
    assert first["description"] == "Minimal parser fixture"
    assert np.isclose(first["A"], 1.0)
    assert np.isclose(first["R"], 2.0)
    assert np.isclose(first["L"], 11.0)
    assert np.isclose(first["K"], 12.0)
