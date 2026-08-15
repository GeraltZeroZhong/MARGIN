"""Discoverable registry for the repository's computational workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Workflow:
    """One analysis workflow and its repository entry points."""

    name: str
    purpose: str
    package: str
    config: Path
    scripts: Path


WORKFLOWS: tuple[Workflow, ...] = (
    Workflow(
        "foundation",
        "Teacher value, specificity, observability, and on-policy audit.",
        "margin.pipeline",
        Path("configs/foundation.yaml"),
        Path("scripts/workflows/foundation"),
    ),
    Workflow(
        "observability",
        "Sequence-representation learnability and independent replication.",
        "margin.studies.observability",
        Path("configs/observability.yaml"),
        Path("scripts/workflows/observability"),
    ),
    Workflow(
        "generalization",
        "Architecture, lineage, environment, and DMS transfer analyses.",
        "margin.studies.generalization",
        Path("configs/generalization.yaml"),
        Path("scripts/workflows/generalization"),
    ),
    Workflow(
        "counterfactuals",
        "Counterfactual structure-residual validation.",
        "margin.studies.counterfactuals",
        Path("configs/counterfactuals.yaml"),
        Path("scripts/workflows/counterfactuals"),
    ),
    Workflow(
        "mechanisms",
        "In-distribution counterfactual and denoising mechanism audit.",
        "margin.studies.mechanisms",
        Path("configs/mechanisms.yaml"),
        Path("scripts/workflows/mechanisms"),
    ),
    Workflow(
        "action_validation",
        "Structure-unique mutation-action decomposition and validation.",
        "margin.studies.action_validation",
        Path("configs/action_validation.yaml"),
        Path("scripts/workflows/action_validation"),
    ),
    Workflow(
        "stability",
        "Calibrated paired-action evaluation and strong sequence controls.",
        "margin.studies.stability",
        Path("configs/stability.yaml"),
        Path("scripts/workflows/stability"),
    ),
    Workflow(
        "external_validation",
        "Independent cross-platform confirmation.",
        "margin.studies.external_validation",
        Path("configs/external_validation.yaml"),
        Path("scripts/workflows/external_validation"),
    ),
    Workflow(
        "structure_sensitivity",
        "Matched experimental and predicted-backbone sensitivity analysis.",
        "margin.studies.structure_sensitivity",
        Path("configs/structure_sensitivity.yaml"),
        Path("scripts/workflows/structure_sensitivity"),
    ),
)


def get_workflow(name: str) -> Workflow:
    """Return one workflow by its stable semantic name."""

    for workflow in WORKFLOWS:
        if workflow.name == name:
            return workflow
    choices = ", ".join(item.name for item in WORKFLOWS)
    raise KeyError(f"unknown workflow {name!r}; choose one of: {choices}")


def missing_workflow_paths(root: Path) -> dict[str, list[Path]]:
    """Report missing configured entry points."""

    missing: dict[str, list[Path]] = {}
    for workflow in WORKFLOWS:
        absent = [
            path for path in (workflow.config, workflow.scripts) if not (root / path).exists()
        ]
        if absent:
            missing[workflow.name] = absent
    return missing
