"""Explicit foundation decision logic with branch outcomes and missing-evidence handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from margin.attribution.distillability import DistillabilityAudit
from margin.attribution.metrics import cluster_bootstrap_mean
from margin.attribution.observability import ObservabilityAudit
from margin.attribution.on_policy import OnPolicyAudit
from margin.attribution.teacher_value import TeacherValueAudit
from margin.config import ProjectConfig

DecisionCode = Literal[
    "GO",
    "NO_GO",
    "PIVOT_STRUCTURE_CONDITIONED",
    "DROP_ON_POLICY",
    "NARROW_STABILITY",
    "INCOMPLETE",
    "SYNTHETIC_ONLY",
]


@dataclass(frozen=True)
class FoundationDecision:
    decision: DecisionCode
    criteria: pd.DataFrame
    decision_record: dict[str, Any]


def evaluate_foundation_decision(
    teacher: TeacherValueAudit,
    observability: ObservabilityAudit,
    on_policy: OnPolicyAudit,
    distillability: DistillabilityAudit,
    config: ProjectConfig,
) -> FoundationDecision:
    """Evaluate the fixed foundation audit criteria and route the project branch."""

    criteria = pd.DataFrame(
        [
            _environment_criterion("core_action_value", teacher, config, "burial", ["buried"]),
            _environment_criterion(
                "regular_secondary_structure_action_value",
                teacher,
                config,
                "secondary_structure",
                ["helix", "strand"],
            ),
            _specificity_criterion(teacher, config),
            _dms_criterion(teacher, config),
            _observability_criterion(observability, config),
            _distillable_environment_criterion(distillability, config),
            _radius_criterion(distillability, config),
            _teacher_consistency_criterion(teacher, config),
            _on_policy_criterion(on_policy, config),
        ]
    )
    decision, rationale = _branch(criteria, config)
    evidence_scope = _experimental_evidence_scope(teacher, config)
    if decision == "GO" and evidence_scope == "stability_only":
        decision = "NARROW_STABILITY"
        rationale = (
            "All method gates pass, but independent experimental evidence is confined to "
            "stability assays."
        )
    if config.data_mode == "synthetic" or not config.decision.allow_real_decision:
        decision = "SYNTHETIC_ONLY"
        rationale = (
            "Synthetic fixtures validate software behavior but cannot support a scientific "
            "foundation decision."
        )
    record = {
        "decision": decision,
        "rationale": rationale,
        "data_mode": config.data_mode,
        "allow_real_decision": config.decision.allow_real_decision,
        "decision_analysis_role": config.audit.decision_analysis_role,
        "experimental_evidence_scope": evidence_scope,
        "criteria_passed": int((criteria["status"] == "PASS").sum()),
        "criteria_failed": int((criteria["status"] == "FAIL").sum()),
        "criteria_incomplete": int((criteria["status"] == "INCOMPLETE").sum()),
    }
    return FoundationDecision(decision=decision, criteria=criteria, decision_record=record)


def _environment_criterion(
    criterion: str,
    audit: TeacherValueAudit,
    config: ProjectConfig,
    axis: str,
    environments: list[str],
) -> dict[str, Any]:
    table = _decision_scope(audit.position_metrics, config)
    selected = table.loc[
        (table["teacher_id"] == config.audit.primary_teacher_id)
        & (table["structure_role"] == config.audit.paired_role)
        & (table[axis].isin(environments))
        & (
            table["requested_corruption_ratio"]
            <= config.decision.minimum_teacher_action_valid_radius + 1e-12
        )
    ]
    estimate = cluster_bootstrap_mean(
        selected,
        "advantage_nats",
        config.audit.cluster_column,
        config.audit.bootstrap_replicates,
        config.audit.confidence_level,
        config.seed + (1200 if axis == "burial" else 1201),
    )
    if not estimate["n_rows"]:
        return _criterion(
            criterion, "INCOMPLETE", estimate, config.decision.minimum_environment_advantage_nats
        )
    passed = estimate["estimate"] >= config.decision.minimum_environment_advantage_nats
    if config.decision.require_positive_ci_lower_bound:
        passed &= estimate["ci_low"] > 0
    return _criterion(
        criterion,
        "PASS" if passed else "FAIL",
        estimate,
        config.decision.minimum_environment_advantage_nats,
    )


def _specificity_criterion(audit: TeacherValueAudit, config: ProjectConfig) -> dict[str, Any]:
    specificity = _decision_scope(audit.specificity_positions, config)
    selected = specificity.loc[
        (specificity.get("teacher_id", pd.Series(dtype=str)) == config.audit.primary_teacher_id)
        & (specificity.get("decoy_role", pd.Series(dtype=str)) == "matched_cath")
    ]
    estimate = cluster_bootstrap_mean(
        selected,
        "paired_decoy_lift_nats",
        config.audit.cluster_column,
        config.audit.bootstrap_replicates,
        config.audit.confidence_level,
        config.seed + 1202,
    )
    if not estimate["n_rows"]:
        return _criterion(
            "paired_beats_matched_decoy",
            "INCOMPLETE",
            estimate,
            config.decision.minimum_paired_decoy_lift_nats,
        )
    passed = estimate["estimate"] >= config.decision.minimum_paired_decoy_lift_nats
    if config.decision.require_positive_ci_lower_bound:
        passed &= estimate["ci_low"] > 0
    return _criterion(
        "paired_beats_matched_decoy",
        "PASS" if passed else "FAIL",
        estimate,
        config.decision.minimum_paired_decoy_lift_nats,
    )


def _dms_criterion(audit: TeacherValueAudit, config: ProjectConfig) -> dict[str, Any]:
    coverage = audit.dms_coverage
    if config.audit.decision_analysis_role != "all" and not coverage.empty:
        coverage = coverage.loc[
            coverage["analysis_role"].isin([config.audit.decision_analysis_role, "unknown"])
        ]
    required_teachers = {
        config.audit.primary_teacher_id,
        config.audit.sequence_teacher_id,
    }
    required_coverage = coverage.loc[
        coverage.get("teacher_id", pd.Series(dtype=str)).isin(required_teachers)
    ]
    covered_teachers = set(required_coverage.get("teacher_id", pd.Series(dtype=str)))
    complete_coverage = (
        not required_coverage.empty
        and covered_teachers == required_teachers
        and (required_coverage["status"] == "complete").all()
    )
    if not complete_coverage:
        return _criterion(
            "independent_dms_ranking",
            "INCOMPLETE",
            {
                "dms_coverage_rows": int(len(required_coverage)),
                "incomplete_coverage_rows": int(
                    (required_coverage.get("status", pd.Series(dtype=str)) != "complete").sum()
                ),
                "covered_teachers": sorted(covered_teachers),
            },
            config.decision.minimum_dms_spearman,
        )
    summary = _decision_scope(audit.dms_summary, config)
    selected = summary.loc[
        (summary.get("scope", pd.Series(dtype=str)) == "pooled")
        & (summary.get("teacher_id", pd.Series(dtype=str)) == config.audit.primary_teacher_id)
    ]
    if selected.empty:
        return _criterion(
            "independent_dms_ranking",
            "INCOMPLETE",
            {},
            config.decision.minimum_dms_spearman,
        )
    row = selected.iloc[0]
    sequence = summary.loc[
        (summary["scope"] == "pooled") & (summary["teacher_id"] == config.audit.sequence_teacher_id)
    ]
    if sequence.empty:
        return _criterion(
            "independent_dms_ranking",
            "INCOMPLETE",
            {"reason": "sequence baseline lacks a pooled DMS estimate"},
            config.decision.minimum_dms_spearman,
        )
    sequence_estimate = float(sequence.iloc[0]["estimate"])
    passed = float(row["estimate"]) >= config.decision.minimum_dms_spearman
    if np.isfinite(sequence_estimate):
        passed &= float(row["estimate"]) > sequence_estimate
    if config.decision.require_positive_ci_lower_bound:
        passed &= float(row["ci_low"]) > 0
    values = {
        "estimate": float(row["estimate"]),
        "ci_low": float(row["ci_low"]),
        "ci_high": float(row["ci_high"]),
        "n_rows": int(row["n_rows"]),
        "n_domains": int(row["n_domains"]),
        "sequence_estimate": sequence_estimate,
    }
    return _criterion(
        "independent_dms_ranking",
        "PASS" if passed else "FAIL",
        values,
        config.decision.minimum_dms_spearman,
    )


def _observability_criterion(audit: ObservabilityAudit, config: ProjectConfig) -> dict[str, Any]:
    summary = _decision_scope(audit.summary, config)
    if summary.empty:
        return _criterion("cath_h_frozen_linear_residual_accessibility", "INCOMPLETE", {}, None)
    selected = summary.loc[
        (summary["group_level"] == "cath_h") & (summary["control"] == "observed")
    ]
    jsd = selected.loc[selected["metric"] == "jsd_reduction_nats"]
    cosine = selected.loc[selected["metric"] == "residual_cosine"]
    if jsd.empty or cosine.empty:
        return _criterion("cath_h_frozen_linear_residual_accessibility", "INCOMPLETE", {}, None)
    jsd_value = float(jsd.iloc[0]["estimate"])
    cosine_value = float(cosine.iloc[0]["estimate"])
    jsd_ci_low = float(jsd.iloc[0]["ci_low"])
    jsd_ci_high = float(jsd.iloc[0]["ci_high"])
    shuffled = summary.loc[
        (summary["group_level"] == "cath_h")
        & (summary["control"] == "shuffled")
        & (summary["metric"] == "jsd_reduction_nats")
    ]
    shuffled_jsd = float(shuffled.iloc[0]["estimate"]) if not shuffled.empty else float("nan")
    passed = (
        jsd_value >= config.decision.minimum_observability_jsd_reduction
        and cosine_value >= config.decision.minimum_observability_cosine
        and (not np.isfinite(shuffled_jsd) or jsd_value > shuffled_jsd)
    )
    if config.decision.require_positive_ci_lower_bound:
        passed &= jsd_ci_low > 0
    return _criterion(
        "cath_h_frozen_linear_residual_accessibility",
        "PASS" if passed else "FAIL",
        {
            "estimate": jsd_value,
            "ci_low": jsd_ci_low,
            "ci_high": jsd_ci_high,
            "n_rows": int(jsd.iloc[0]["n_rows"]),
            "n_domains": int(jsd.iloc[0]["n_domains"]),
            "residual_cosine": cosine_value,
            "shuffled_jsd_reduction": shuffled_jsd,
        },
        config.decision.minimum_observability_jsd_reduction,
    )


def _distillable_environment_criterion(
    audit: DistillabilityAudit, config: ProjectConfig
) -> dict[str, Any]:
    table = _decision_scope(audit.map_table, config)
    if table.empty:
        return _criterion("high_value_high_observability_environment", "INCOMPLETE", {}, 1)
    count = int(table["distillable"].sum())
    return _criterion(
        "high_value_high_observability_environment",
        "PASS" if count > 0 else "FAIL",
        {"estimate": count, "n_rows": len(table)},
        1,
    )


def _radius_criterion(audit: DistillabilityAudit, config: ProjectConfig) -> dict[str, Any]:
    radius = _decision_scope(audit.action_valid_radius, config)
    selected = radius.loc[
        (
            (radius.get("environment_axis", pd.Series(dtype=str)) == "burial")
            & (radius.get("environment", pd.Series(dtype=str)) == "buried")
        )
        | (
            (radius.get("environment_axis", pd.Series(dtype=str)) == "secondary_structure")
            & (radius.get("environment", pd.Series(dtype=str)).isin(["helix", "strand"]))
        )
    ]
    if selected.empty:
        return _criterion(
            "teacher_action_valid_radius",
            "INCOMPLETE",
            {},
            config.decision.minimum_teacher_action_valid_radius,
        )
    core = selected.loc[selected["environment_axis"] == "burial", "teacher_action_valid_radius"]
    regular = selected.loc[
        selected["environment_axis"] == "secondary_structure", "teacher_action_valid_radius"
    ]
    observed = min(
        float(core.max()) if len(core) else 0.0, float(regular.max()) if len(regular) else 0.0
    )
    return _criterion(
        "teacher_action_valid_radius",
        "PASS" if observed >= config.decision.minimum_teacher_action_valid_radius else "FAIL",
        {"estimate": observed},
        config.decision.minimum_teacher_action_valid_radius,
    )


def _teacher_consistency_criterion(
    audit: TeacherValueAudit, config: ProjectConfig
) -> dict[str, Any]:
    positions = _decision_scope(audit.position_metrics, config)
    selected = positions.loc[
        (positions["structure_role"] == config.audit.paired_role)
        & (positions["teacher_id"] != config.audit.sequence_teacher_id)
    ]
    required = config.decision.minimum_directionally_consistent_structure_teachers
    if selected.empty or selected["teacher_id"].nunique() < required:
        return _criterion(
            "structure_teacher_directional_consistency",
            "INCOMPLETE",
            {"available_teachers": int(selected["teacher_id"].nunique())},
            required,
        )
    positive = 0
    estimates: dict[str, float] = {}
    for index, (teacher_id, frame) in enumerate(selected.groupby("teacher_id", observed=True)):
        estimate = cluster_bootstrap_mean(
            frame,
            "advantage_nats",
            config.audit.cluster_column,
            config.audit.bootstrap_replicates,
            config.audit.confidence_level,
            config.seed + 1300 + index,
        )
        estimates[str(teacher_id)] = float(estimate["estimate"])
        directional = estimate["estimate"] > 0
        if config.decision.require_positive_ci_lower_bound:
            directional &= estimate["ci_low"] > 0
        positive += int(directional)
    return _criterion(
        "structure_teacher_directional_consistency",
        "PASS" if positive >= required else "FAIL",
        {"estimate": positive, "teacher_estimates": estimates},
        required,
    )


def _on_policy_criterion(audit: OnPolicyAudit, config: ProjectConfig) -> dict[str, Any]:
    summary = _decision_scope(audit.effect_summary, config)
    selected = summary.loc[
        (summary.get("comparison", pd.Series(dtype=str)) == "on_policy_vs_model_aware")
        & (summary.get("metric", pd.Series(dtype=str)) == "teacher_advantage_difference_nats")
    ]
    if selected.empty:
        return _criterion(
            "on_policy_incremental_value",
            "INCOMPLETE",
            {},
            config.decision.minimum_on_policy_advantage_nats,
        )
    row = selected.iloc[0]
    if not bool(row["matching_quality_pass"]):
        return _criterion(
            "on_policy_incremental_value",
            "INCOMPLETE",
            {
                "estimate": float(row["estimate"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
                "n_rows": int(row["n_rows"]),
                "n_domains": int(row["n_domains"]),
                "match_fraction": float(row["match_fraction"]),
                "maximum_absolute_smd": float(row["maximum_absolute_smd"]),
            },
            config.decision.minimum_on_policy_advantage_nats,
        )
    passed = float(row["estimate"]) >= config.decision.minimum_on_policy_advantage_nats
    if config.decision.require_positive_ci_lower_bound:
        passed &= float(row["ci_low"]) > 0
    return _criterion(
        "on_policy_incremental_value",
        "PASS" if passed else "FAIL",
        {
            "estimate": float(row["estimate"]),
            "ci_low": float(row["ci_low"]),
            "ci_high": float(row["ci_high"]),
            "n_rows": int(row["n_rows"]),
            "n_domains": int(row["n_domains"]),
            "match_fraction": float(row["match_fraction"]),
            "maximum_absolute_smd": float(row["maximum_absolute_smd"]),
        },
        config.decision.minimum_on_policy_advantage_nats,
    )


def _branch(criteria: pd.DataFrame, config: ProjectConfig) -> tuple[DecisionCode, str]:
    status = criteria.set_index("criterion")["status"].to_dict()
    required = [
        "core_action_value",
        "regular_secondary_structure_action_value",
        "paired_beats_matched_decoy",
        "independent_dms_ranking",
        "cath_h_frozen_linear_residual_accessibility",
        "high_value_high_observability_environment",
        "teacher_action_valid_radius",
        "structure_teacher_directional_consistency",
    ]
    if any(status.get(item) == "INCOMPLETE" for item in required):
        return "INCOMPLETE", "One or more mandatory scientific criteria lack sufficient evidence."
    if (
        status.get("core_action_value") == "FAIL"
        or status.get("regular_secondary_structure_action_value") == "FAIL"
    ):
        return (
            "NO_GO",
            "The paired structure teacher lacks stable action value in required environments.",
        )
    if status.get("paired_beats_matched_decoy") == "FAIL":
        return "NO_GO", "Correctly paired structure does not beat the matched-structure control."
    if status.get("independent_dms_ranking") == "FAIL":
        return "NO_GO", "Teacher advantage does not transfer to independent experimental ranking."
    if (
        status.get("cath_h_frozen_linear_residual_accessibility") == "FAIL"
        or status.get("high_value_high_observability_environment") == "FAIL"
    ):
        return (
            "PIVOT_STRUCTURE_CONDITIONED",
            "Teacher signal is useful, but its residual is not linearly accessible from the "
            "configured frozen sequence features under cross-domain transfer.",
        )
    if (
        status.get("teacher_action_valid_radius") == "FAIL"
        or status.get("structure_teacher_directional_consistency") == "FAIL"
    ):
        return "NO_GO", "Scaffold reliability or cross-teacher consistency is insufficient."
    if status.get("on_policy_incremental_value") == "INCOMPLETE":
        return "INCOMPLETE", "On-policy matching quality is insufficient for the named claim."
    if status.get("on_policy_incremental_value") == "FAIL":
        return (
            "DROP_ON_POLICY",
            "Residuals are linearly accessible, but on-policy states add no matched value.",
        )
    return "GO", "All fixed foundation decision criteria pass."


def _criterion(
    name: str,
    status: Literal["PASS", "FAIL", "INCOMPLETE"],
    values: dict[str, Any],
    threshold: float | int | None,
) -> dict[str, Any]:
    return {
        "criterion": name,
        "status": status,
        "estimate": values.get("estimate", float("nan")),
        "ci_low": values.get("ci_low", float("nan")),
        "ci_high": values.get("ci_high", float("nan")),
        "threshold": threshold,
        "n_rows": values.get("n_rows", 0),
        "n_domains": values.get("n_domains", 0),
        "details": {
            key: value
            for key, value in values.items()
            if key not in {"estimate", "ci_low", "ci_high", "n_rows", "n_domains"}
        },
    }


def _experimental_evidence_scope(audit: TeacherValueAudit, config: ProjectConfig) -> str:
    predictions = _decision_scope(audit.dms_predictions, config)
    if predictions.empty:
        return "missing"
    assay_types = {
        str(value).strip().lower()
        for value in predictions.get("assay_type", pd.Series(dtype=str)).dropna()
        if str(value).strip() and str(value).strip().lower() != "unspecified"
    }
    if not assay_types:
        return "unspecified"
    stability_names = {"stability", "ddg", "folding_stability"}
    return "stability_only" if assay_types <= stability_names else "multi_assay"


def _decision_scope(table: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    """Select the preregistered analysis population for every scientific criterion."""

    if table.empty or config.audit.decision_analysis_role == "all":
        return table
    if "analysis_role" not in table.columns:
        return table.iloc[0:0]
    return table.loc[table["analysis_role"] == config.audit.decision_analysis_role]
