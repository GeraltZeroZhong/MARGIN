from __future__ import annotations

import pandas as pd
import pytest

from margin.attribution.decision import _branch
from margin.config import ProjectConfig

CRITERIA = [
    "core_action_value",
    "regular_secondary_structure_action_value",
    "paired_beats_matched_decoy",
    "independent_dms_ranking",
    "cath_h_frozen_linear_residual_accessibility",
    "high_value_high_observability_environment",
    "teacher_action_valid_radius",
    "structure_teacher_directional_consistency",
    "on_policy_incremental_value",
]


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, "GO"),
        ({"core_action_value": "FAIL"}, "NO_GO"),
        ({"regular_secondary_structure_action_value": "FAIL"}, "NO_GO"),
        ({"paired_beats_matched_decoy": "FAIL"}, "NO_GO"),
        ({"independent_dms_ranking": "FAIL"}, "NO_GO"),
        (
            {"cath_h_frozen_linear_residual_accessibility": "FAIL"},
            "PIVOT_STRUCTURE_CONDITIONED",
        ),
        ({"high_value_high_observability_environment": "FAIL"}, "PIVOT_STRUCTURE_CONDITIONED"),
        ({"teacher_action_valid_radius": "FAIL"}, "NO_GO"),
        ({"structure_teacher_directional_consistency": "FAIL"}, "NO_GO"),
        ({"on_policy_incremental_value": "FAIL"}, "DROP_ON_POLICY"),
        ({"on_policy_incremental_value": "INCOMPLETE"}, "INCOMPLETE"),
        ({"independent_dms_ranking": "INCOMPLETE"}, "INCOMPLETE"),
    ],
)
def test_fixed_gate_branch_routing(
    synthetic_config: ProjectConfig,
    overrides: dict[str, str],
    expected: str,
) -> None:
    statuses = {criterion: "PASS" for criterion in CRITERIA}
    statuses.update(overrides)
    criteria = pd.DataFrame(
        [{"criterion": criterion, "status": status} for criterion, status in statuses.items()]
    )

    decision, _ = _branch(criteria, synthetic_config)

    assert decision == expected
