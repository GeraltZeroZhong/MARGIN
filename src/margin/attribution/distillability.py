"""Combine action value, linear accessibility, specificity, and action support."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from margin.attribution.metrics import grouped_cluster_means
from margin.attribution.observability import ObservabilityAudit
from margin.attribution.teacher_value import TeacherValueAudit
from margin.config import ProjectConfig
from margin.state_sampling.bank import StateBank


@dataclass(frozen=True)
class DistillabilityAudit:
    map_table: pd.DataFrame
    action_valid_radius: pd.DataFrame


def build_distillability_map(
    teacher: TeacherValueAudit,
    observability: ObservabilityAudit,
    bank: StateBank,
    config: ProjectConfig,
) -> DistillabilityAudit:
    """Build the declared map without collapsing incompatible environments."""

    value = teacher.environment_summary.loc[
        teacher.environment_summary.get("teacher_id", pd.Series(dtype=str))
        == config.audit.primary_teacher_id
    ].copy()
    if value.empty:
        return DistillabilityAudit(pd.DataFrame(), pd.DataFrame())
    keys = [
        "analysis_role",
        "state_kind",
        "requested_corruption_ratio",
        "environment_axis",
        "environment",
    ]
    value = value.rename(
        columns={
            "estimate": "teacher_advantage_nats",
            "ci_low": "teacher_advantage_ci_low",
            "ci_high": "teacher_advantage_ci_high",
            "n_rows": "value_rows",
            "n_domains": "value_domains",
        }
    )
    specificity = _specificity_for_map(teacher, config, keys)
    observable = _observability_for_map(observability, keys)
    reliability = _reliability_for_map(bank, config)
    result = value.merge(specificity, on=keys, how="left", validate="one_to_one")
    result = result.merge(observable, on=keys, how="left", validate="one_to_one")
    result = result.merge(reliability, on=keys, how="left", validate="one_to_one")
    positive_ci = (
        result["teacher_advantage_ci_low"] > 0
        if config.decision.require_positive_ci_lower_bound
        else np.ones(len(result), dtype=bool)
    )
    result["high_action_value"] = (
        result["teacher_advantage_nats"] >= config.decision.minimum_environment_advantage_nats
    ) & positive_ci
    result["target_specific"] = (
        result["matched_decoy_lift_nats"] >= config.decision.minimum_paired_decoy_lift_nats
    )
    result["sequence_observable"] = (
        result["observability_jsd_reduction_nats"]
        >= config.decision.minimum_observability_jsd_reduction
    ) & (result["observability_residual_cosine"] >= config.decision.minimum_observability_cosine)
    result["distillable"] = (
        result["high_action_value"] & result["target_specific"] & result["sequence_observable"]
    )
    result["map_class"] = np.select(
        [
            result["distillable"],
            result["high_action_value"] & ~result["sequence_observable"],
            ~result["target_specific"],
        ],
        ["distillable", "valuable_but_unobservable", "nonspecific"],
        default="low_value",
    )
    radius = _teacher_action_valid_radius(result, config)
    return DistillabilityAudit(result, radius)


def _specificity_for_map(
    audit: TeacherValueAudit, config: ProjectConfig, keys: list[str]
) -> pd.DataFrame:
    table = audit.specificity_summary
    if table.empty:
        return pd.DataFrame(columns=[*keys, "matched_decoy_lift_nats"])
    selected = table.loc[
        (table["teacher_id"] == config.audit.primary_teacher_id)
        & (table["decoy_role"] == "matched_cath")
    ]
    return selected[[*keys, "estimate", "ci_low", "ci_high"]].rename(
        columns={
            "estimate": "matched_decoy_lift_nats",
            "ci_low": "matched_decoy_lift_ci_low",
            "ci_high": "matched_decoy_lift_ci_high",
        }
    )


def _observability_for_map(audit: ObservabilityAudit, keys: list[str]) -> pd.DataFrame:
    table = audit.environment_summary
    if table.empty:
        return pd.DataFrame(
            columns=[
                *keys,
                "observability_jsd_reduction_nats",
                "observability_residual_cosine",
            ]
        )
    preferred = "cath_h" if "cath_h" in set(table["group_level"]) else table["group_level"].iloc[0]
    selected = table.loc[table["group_level"] == preferred]
    if "sufficient_rows" in selected:
        selected = selected.loc[selected["sufficient_rows"]]
    pivot = selected.pivot_table(
        index=keys,
        columns="metric",
        values="estimate",
        aggfunc="first",
    ).reset_index()
    return pivot.rename(
        columns={
            "jsd_reduction_nats": "observability_jsd_reduction_nats",
            "residual_cosine": "observability_residual_cosine",
            "topk_overlap": "observability_topk_overlap",
        }
    )


def _reliability_for_map(bank: StateBank, config: ProjectConfig) -> pd.DataFrame:
    positions = bank.positions[
        [
            "state_id",
            "domain_id",
            "analysis_role",
            "burial",
            "secondary_structure",
            "contact_class",
            "conservation_class",
        ]
    ].merge(
        bank.states[
            ["state_id", "state_kind", "requested_corruption_ratio", "scaffold_compatibility"]
        ],
        on="state_id",
        validate="many_to_one",
    )
    long = positions.melt(
        id_vars=[
            "state_id",
            "domain_id",
            "analysis_role",
            "state_kind",
            "requested_corruption_ratio",
            "scaffold_compatibility",
        ],
        value_vars=[
            "burial",
            "secondary_structure",
            "contact_class",
            "conservation_class",
        ],
        var_name="environment_axis",
        value_name="environment",
    )
    summary = grouped_cluster_means(
        long,
        [
            "analysis_role",
            "state_kind",
            "requested_corruption_ratio",
            "environment_axis",
            "environment",
        ],
        "scaffold_compatibility",
        config.audit.cluster_column,
        config.audit.bootstrap_replicates,
        config.audit.confidence_level,
        config.seed + 1100,
    )
    return summary.rename(
        columns={
            "estimate": "scaffold_reliability",
            "ci_low": "scaffold_reliability_ci_low",
            "ci_high": "scaffold_reliability_ci_high",
            "n_rows": "reliability_rows",
            "n_domains": "reliability_domains",
        }
    )


def _teacher_action_valid_radius(table: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (analysis_role, axis, environment), frame in table.groupby(
        ["analysis_role", "environment_axis", "environment"], observed=True
    ):
        by_ratio = (
            frame.groupby("requested_corruption_ratio", observed=True)
            .agg(
                advantage_nats=("teacher_advantage_nats", "mean"),
                ci_low=("teacher_advantage_ci_low", "min"),
                state_kinds=("state_kind", "nunique"),
            )
            .reset_index()
            .sort_values("requested_corruption_ratio")
        )
        passing = by_ratio["advantage_nats"] >= config.decision.minimum_environment_advantage_nats
        if config.decision.require_positive_ci_lower_bound:
            passing &= by_ratio["ci_low"] > 0
        radius = (
            float(by_ratio.loc[passing, "requested_corruption_ratio"].max())
            if passing.any()
            else 0.0
        )
        rows.append(
            {
                "analysis_role": analysis_role,
                "environment_axis": axis,
                "environment": environment,
                "teacher_action_valid_radius": radius,
                "tested_maximum_radius": float(by_ratio["requested_corruption_ratio"].max()),
                "passing_corruption_levels": int(passing.sum()),
                "tested_corruption_levels": int(len(by_ratio)),
            }
        )
    return pd.DataFrame(rows)
