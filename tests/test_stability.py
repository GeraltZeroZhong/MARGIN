from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from margin.constants import AA_ALPHABET
from margin.studies.stability.calibration import apply_calibration, native_metrics
from margin.studies.stability.config import load_stability_config
from margin.studies.stability.evaluation import _decision, _position_bootstrap
from margin.studies.stability.prepare import EXTERNAL_POPULATION, PRIMARY_POPULATION
from margin.studies.stability.profiles import profile_queries


def test_stability_config_preserves_frozen_calibration_control_and_inference() -> None:
    config = load_stability_config(Path("configs/stability.yaml"))
    assert config.calibration.teacher_ids == ["mif", "esm_if1", "proteinmpnn"]
    assert config.calibration.schemes == [
        "unscaled_equal",
        "action_rms_matched",
        "joint_temperature_native_nll",
        "rowwise_rank_normalized",
    ]
    assert config.strong_control.representation_models == [
        "carp_640M",
        "esm2_650M",
        "esm1b_650M",
    ]
    assert config.inference.bootstrap_replicates == 5000
    assert config.inference.external_position_bootstrap_replicates == 5000
    assert config.paths.storage_dir == (Path.cwd() / "data/workspaces/stability").resolve()


def test_temperature_calibration_averages_teacher_actions_after_scaling() -> None:
    actions = {
        "mif": np.array([[0.0, 2.0]]),
        "esm_if1": np.array([[0.0, 4.0]]),
        "proteinmpnn": np.array([[0.0, 8.0]]),
    }
    parameters = {"temperatures": {"mif": 1.0, "esm_if1": 2.0, "proteinmpnn": 4.0}}
    result = apply_calibration(
        actions,
        "joint_temperature_native_nll",
        parameters,
        ["mif", "esm_if1", "proteinmpnn"],
    )
    np.testing.assert_allclose(result, [[0.0, 2.0]])

    metrics = native_metrics(
        np.log(np.array([[0.75, 0.25], [0.25, 0.75]])),
        np.zeros((2, 2)),
        np.array([0, 1]),
    )
    assert np.isclose(metrics["native_nll"], -np.log(0.75))
    assert metrics["native_aar"] == 1.0
    assert metrics["native_mrr"] == 1.0


def test_sequence_profile_excludes_near_identity_hits_and_normalizes_rows() -> None:
    config = load_stability_config(Path("configs/stability.yaml"))
    queries = pd.DataFrame(
        {
            "state_id": ["query"] * 4,
            "domain_id": ["query"] * 4,
            "position": np.arange(4),
            "sequence": ["ACDE"] * 4,
        }
    )
    alignments = pd.DataFrame(
        [
            {
                "query": "query",
                "target": "self",
                "qstart": 1,
                "qaln": "ACDE",
                "taln": "ACDE",
                "evalue": 0.0,
            },
            {
                "query": "query",
                "target": "accepted",
                "qstart": 1,
                "qaln": "ACDE",
                "taln": "AFGE",
                "evalue": 1e-5,
            },
        ]
    )
    result = profile_queries(queries, alignments, config)
    profile_columns = [f"profile_{amino_acid}" for amino_acid in AA_ALPHABET]

    assert result["accepted_homolog_hits"].eq(1).all()
    assert result["homolog_observations"].eq(1).all()
    np.testing.assert_allclose(result[profile_columns].sum(axis=1), 1.0)


def test_external_position_bootstrap_is_grouped_and_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "position": np.repeat(np.arange(4), 3),
            "variant_row": np.arange(12),
            "effect": np.arange(12, dtype=float),
        }
    )
    method = np.arange(12, dtype=float)
    comparator = -method
    first = _position_bootstrap(
        frame,
        method,
        comparator,
        metric="spearman",
        replicates=100,
        confidence_level=0.95,
        seed=7,
    )
    second = _position_bootstrap(
        frame,
        method,
        comparator,
        metric="spearman",
        replicates=100,
        confidence_level=0.95,
        seed=7,
    )
    assert first == second
    assert first["n_positions"] == 4
    assert np.isclose(first["estimate"], 2.0)


def test_stability_decision_separates_paired_action_from_calibration_value() -> None:
    config = load_stability_config(Path("configs/stability.yaml"))
    rows: list[dict[str, object]] = []

    def add(
        contrast: str,
        population: str,
        stratum: str,
        metric: str,
        *,
        estimate: float = 0.10,
        ci_low: float = 0.05,
    ) -> None:
        rows.append(
            {
                "contrast": contrast,
                "evaluation_population": population,
                "stratum": stratum,
                "metric": metric,
                "estimate": estimate,
                "ci_low": ci_low,
                "ci_high": estimate + 0.05,
            }
        )

    add("selected_consensus_vs_sequence", PRIMARY_POPULATION, "all", "spearman")
    add(
        "selected_consensus_vs_sequence",
        PRIMARY_POPULATION,
        "all",
        "ndcg_at_10_percent",
    )
    add("selected_consensus_vs_sequence", PRIMARY_POPULATION, "natural", "spearman")
    add("selected_consensus_vs_sequence", PRIMARY_POPULATION, "de_novo", "spearman")
    add(
        "selected_consensus_vs_sequence",
        EXTERNAL_POPULATION,
        "external_single_protein",
        "spearman",
    )
    add(
        "selected_consensus_vs_sequence",
        EXTERNAL_POPULATION,
        "external_single_protein",
        "ndcg_at_10_percent",
    )
    add("selected_consensus_vs_Cplus", PRIMARY_POPULATION, "all", "spearman")
    add(
        "selected_consensus_vs_unscaled",
        PRIMARY_POPULATION,
        "all",
        "spearman",
        estimate=-0.01,
        ci_low=-0.02,
    )
    for teacher in config.calibration.teacher_ids:
        add(f"{teacher}_vs_sequence", PRIMARY_POPULATION, "all", "spearman")
    quality = pd.DataFrame({"passed": [True, True, True, True]})

    gates, decision = _decision(pd.DataFrame(rows), quality, config)

    assert len(gates) == 8
    assert gates["passed"].all()
    row = decision.iloc[0]
    assert row.project_decision == ("STABILITY_STRUCTURE_CONDITIONED_ACTION_CONFIRMED_BEYOND_CPLUS")
    assert row.paired_action_decision == ("PAIRED_ACTION_CONFIRMED_CALIBRATION_NOT_ADDITIVE")
    assert not bool(row.calibration_additive_on_primary_spearman)
