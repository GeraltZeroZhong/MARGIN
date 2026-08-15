from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from margin.constants import AA_ALPHABET
from margin.studies.action_validation.config import load_action_validation_config
from margin.studies.action_validation.evaluation import (
    PRIMARY_POPULATION,
    REPLICATION_POPULATION,
    _agreement_gated_u,
    _anchor,
    _anchored_rmse,
    _decision,
    _global_component,
)


def test_action_validation_config_preserves_the_frozen_estimand_and_inference() -> None:
    config = load_action_validation_config(Path("configs/action_validation.yaml"))
    assert config.decomposition.teacher_ids == ["mif", "esm_if1", "proteinmpnn"]
    assert config.decomposition.final_training_splits == [
        "development_train",
        "development_validation",
    ]
    assert config.decomposition.rrr_ranks == [4, 8, 16]
    assert config.decomposition.ridge_alphas == [1.0, 10.0, 100.0]
    assert config.decomposition.shuffled_u_repeats == 20
    assert config.inference.bootstrap_replicates == 5000


def test_global_component_is_wild_type_conditioned_and_anchored() -> None:
    first = np.arange(len(AA_ALPHABET), dtype=float)
    second = 2.0 * first
    third = 100.0 + first
    training = np.stack([first, second, third])
    training_wild = np.array([0, 0, 1])
    target_wild = np.array([0, 1, 2])

    result = _global_component(training, training_wild, target_wild)
    fallback = training.mean(axis=0)
    expected = _anchor(np.stack([1.5 * first, third, fallback]), target_wild)

    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(result[np.arange(3), target_wild], 0.0)


def test_anchored_rmse_uses_only_non_wild_type_actions() -> None:
    predicted = np.zeros((2, len(AA_ALPHABET)))
    observed = np.zeros_like(predicted)
    observed[0, 1] = 3.0
    observed[1, 2] = 4.0
    wild = np.array([0, 1])

    expected = np.sqrt((3.0**2 + 4.0**2) / (2 * (len(AA_ALPHABET) - 1)))
    assert np.isclose(_anchored_rmse(predicted, observed, wild), expected)


def test_agreement_gate_requires_position_reliability_and_three_matching_signs() -> None:
    config = load_action_validation_config(Path("configs/action_validation.yaml"))
    consensus = np.ones((2, len(AA_ALPHABET)))
    positive = np.ones_like(consensus)
    mixed = positive.copy()
    mixed[0, 1] = -1.0
    scaled = {
        "mif": {"u": positive},
        "esm_if1": {"u": positive},
        "proteinmpnn": {"u": mixed},
    }
    agreement = pd.DataFrame({"median_pairwise_u_spearman": [0.5, 0.1]})

    gated, gate = _agreement_gated_u(consensus, scaled, agreement, config)

    assert gate[0, 0]
    assert not gate[0, 1]
    assert not gate[1].any()
    np.testing.assert_array_equal(gated, consensus * gate)


def test_action_validation_decision_requires_every_frozen_gate() -> None:
    config = load_action_validation_config(Path("configs/action_validation.yaml"))
    rows = []

    def add(
        teacher: str,
        population: str,
        stratum: str,
        metric: str,
        *,
        estimate: float = 0.1,
        ci_low: float = 0.05,
        n_domains: int = 8,
        positive_fraction: float = 0.75,
    ) -> None:
        rows.append(
            {
                "teacher_id": teacher,
                "evaluation_population": population,
                "stratum": stratum,
                "metric": metric,
                "estimate": estimate,
                "ci_low": ci_low,
                "ci_high": estimate + 0.05,
                "n_domains": n_domains,
                "positive_domain_fraction": positive_fraction,
            }
        )

    add("consensus", PRIMARY_POPULATION, "all", "spearman_margin")
    add("consensus", PRIMARY_POPULATION, "all", "ndcg_at_10_percent_margin")
    add("consensus", PRIMARY_POPULATION, "natural", "spearman_margin")
    add("consensus", PRIMARY_POPULATION, "de_novo", "spearman_margin")
    add("consensus", REPLICATION_POPULATION, "all", "spearman_margin")
    for teacher in config.decomposition.teacher_ids:
        add(teacher, PRIMARY_POPULATION, "all", "spearman_margin")
    summary = pd.DataFrame(rows)
    shuffle = pd.DataFrame(
        [
            {
                "evaluation_population": PRIMARY_POPULATION,
                "metric": "spearman_actual_minus_shuffled",
                "estimate": 0.1,
                "ci_low": 0.05,
                "ci_high": 0.15,
            }
        ]
    )

    gates, decision = _decision(summary, shuffle, config)
    assert len(gates) == 9
    assert gates["passed"].all()
    assert decision.iloc[0]["decision"] == ("MULTI_TEACHER_STRUCTURE_UNIQUE_ACTION_CONFIRMED")

    failed = summary.copy()
    failed.loc[failed["metric"].eq("ndcg_at_10_percent_margin"), "ci_low"] = -0.01
    failed_gates, failed_decision = _decision(failed, shuffle, config)
    assert not failed_gates["passed"].all()
    assert failed_decision.iloc[0]["decision"] == (
        "TEACHER_SPECIFIC_STRUCTURE_INCREMENT_WITHOUT_CONSENSUS"
    )
