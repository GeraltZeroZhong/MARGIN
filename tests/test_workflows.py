from pathlib import Path

import pytest

from margin.workflows import WORKFLOWS, get_workflow, missing_workflow_paths


def test_workflows_have_unique_semantic_names_and_entry_points() -> None:
    names = [workflow.name for workflow in WORKFLOWS]
    assert len(names) == len(set(names))
    assert names == [
        "foundation",
        "observability",
        "generalization",
        "counterfactuals",
        "mechanisms",
        "action_validation",
        "stability",
        "external_validation",
        "structure_sensitivity",
    ]
    assert missing_workflow_paths(Path.cwd()) == {}


def test_get_workflow_rejects_unknown_names() -> None:
    assert get_workflow("stability").package == "margin.studies.stability"
    with pytest.raises(KeyError, match="unknown workflow"):
        get_workflow("unknown-study")
