from __future__ import annotations

from pathlib import Path

import numpy as np

from margin.studies.mechanisms.config import load_mechanism_config
from margin.studies.mechanisms.counterfactuals import smooth_coordinate_perturbation
from margin.studies.mechanisms.evaluation import _ndcg


def test_mechanism_config_keeps_project_decision_and_dense_metric_lock() -> None:
    config = load_mechanism_config(Path("configs/mechanisms.yaml"))
    assert config.panel.natural_domains == 16
    assert config.panel.de_novo_domains_per_family * len(config.panel.de_novo_families) == 16
    assert config.panel.minimum_single_variants == 500
    assert config.panel.minimum_unique_positions == 30
    assert config.inference.top_fraction == 0.10
    assert config.inference.bootstrap_replicates == 5000


def test_mechanism_ndcg_at_k_respects_registered_direction() -> None:
    observed = np.linspace(-3.0, 3.0, 100)
    perfect = observed.copy()
    reversed_score = observed[::-1]
    assert np.isclose(_ndcg(perfect, observed, k=10), 1.0)
    assert _ndcg(perfect, observed, k=10) > _ndcg(reversed_score, observed, k=10)


def test_smooth_coordinate_perturbation_preserves_residue_frames_and_chain_bound() -> None:
    length = 50
    coordinates = np.zeros((length, 4, 3), dtype=np.float32)
    coordinates[:, :, 0] = np.arange(length)[:, None] * 3.8
    coordinates[:, 0, 1] = -1.2
    coordinates[:, 2, 1] = 1.2
    coordinates[:, 3, 2] = 1.0
    perturbed, diagnostics = smooth_coordinate_perturbation(
        coordinates,
        target_rmsd=1.0,
        modes=3,
        maximum_adjacent_change=0.25,
        rng=np.random.default_rng(7),
    )
    native_internal = coordinates[:, :, :] - coordinates[:, 1:2, :]
    perturbed_internal = perturbed[:, :, :] - perturbed[:, 1:2, :]
    np.testing.assert_allclose(native_internal, perturbed_internal, atol=1e-5)
    assert diagnostics["maximum_adjacent_ca_distance_change_angstrom"] <= 0.250001
    assert diagnostics["achieved_ca_rmsd_angstrom"] > 0
