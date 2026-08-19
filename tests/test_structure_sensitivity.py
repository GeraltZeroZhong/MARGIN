from pathlib import Path

import numpy as np

from margin.studies.structure_sensitivity.panel import (
    _unique_subsequence,
    kabsch_rmsd,
    load_structure_sensitivity_config,
    smooth_perturbation,
)


def test_structure_sensitivity_protocol_is_descriptive() -> None:
    config = load_structure_sensitivity_config(Path("configs/structure_sensitivity.yaml"))
    assert config.inference.confirmatory is False
    assert config.inference.routing_allowed is False


def test_smooth_perturbation_has_requested_ca_displacement() -> None:
    coordinates = np.zeros((30, 4, 3), dtype=float)
    perturbed = smooth_perturbation(coordinates, 0.5, 9, seed=7)
    displacement = perturbed[:, 1] - coordinates[:, 1]
    rms = np.sqrt(np.mean(np.sum(displacement**2, axis=1)))
    assert np.isclose(rms, 0.5)
    assert np.allclose(perturbed[:, 0] - perturbed[:, 1], 0.0)


def test_kabsch_and_unique_crop() -> None:
    reference = np.arange(30, dtype=float).reshape(10, 3)
    mobile = reference + np.array([4.0, -2.0, 1.0])
    assert kabsch_rmsd(reference, mobile) < 1e-10
    assert _unique_subsequence("XXACDEYY", "ACDE") == 2
