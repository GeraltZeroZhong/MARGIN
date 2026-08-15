from __future__ import annotations

import numpy as np
import pandas as pd

from margin.studies.stability.publication_audit import _measurement_table
from margin.studies.structure_sensitivity.audit import (
    _align_to_reference,
    _backbone_arrays,
    _frame_angle_degrees,
)


def _coordinates() -> np.ndarray:
    ca = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [3.8, 0.2, 0.1],
            [7.5, -0.1, 0.3],
            [11.2, 0.3, -0.2],
        ]
    )
    offsets = np.asarray(
        [
            [-1.2, 0.3, 0.0],
            [0.0, 0.0, 0.0],
            [1.3, 0.1, 0.1],
            [1.8, 0.9, 0.0],
        ]
    )
    return ca[:, None, :] + offsets[None, :, :]


def test_rigid_alignment_recovers_reference_backbone() -> None:
    reference = _coordinates()
    angle = np.deg2rad(37.0)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    mobile = reference @ rotation + np.asarray([4.0, -2.0, 7.0])
    aligned = _align_to_reference(reference, mobile)
    assert np.allclose(aligned, reference, atol=1e-10)
    assert _frame_angle_degrees(reference[1], aligned[1]) < 1e-5


def test_uniform_translation_preserves_backbone_geometry() -> None:
    reference = _coordinates()
    shifted = reference + np.asarray([2.0, -3.0, 1.0])
    left = _backbone_arrays(reference)
    right = _backbone_arrays(shifted)
    for name in left:
        assert np.allclose(left[name], right[name])


def test_measurement_consistency_rule_is_explicit() -> None:
    frame = pd.DataFrame(
        {
            "domain_id": ["a", "a", "b", "b", "c"],
            "position": [0, 0, 1, 1, 2],
            "wild_type": ["A", "A", "V", "V", "G"],
            "mutant": ["V", "V", "A", "A", "A"],
            "ddG": [0.2, 0.4, -0.1, 0.1, 0.3],
            "is_curated": [True, True, False, False, True],
            "pH": [7.0, 7.0, 6.0, 8.0, 7.0],
            "method": ["m", "m", "m", "n", "m"],
            "technique": ["t", "t", "t", "u", "t"],
            "publication_doi": ["d", "d", "e", "f", "g"],
        }
    )
    specification = {
        "fireprot_measurement_audit": {
            "repeated_measurement_minimum": 2,
            "high_consistency_maximum_ddg_range_kcal_mol": 0.5,
            "high_consistency_requires_sign_agreement": True,
        }
    }
    result = _measurement_table(frame, specification).set_index("domain_id")
    assert bool(result.loc["a", "high_consistency_eligible"])
    assert not bool(result.loc["b", "high_consistency_eligible"])
    assert not bool(result.loc["c", "high_consistency_eligible"])
