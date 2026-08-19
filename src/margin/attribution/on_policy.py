"""Matched audit of on-policy, model-aware, and reference offline states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

from margin.attribution.metrics import cluster_bootstrap_mean, grouped_cluster_means
from margin.config import ProjectConfig
from margin.state_sampling.bank import StateBank
from margin.teachers.cache import TeacherScoreCache


@dataclass(frozen=True)
class OnPolicyAudit:
    state_metrics: pd.DataFrame
    matches: pd.DataFrame
    effect_summary: pd.DataFrame
    balance: pd.DataFrame
    timestep_summary: pd.DataFrame
    compute_summary: pd.DataFrame


def audit_on_policy(
    position_metrics: pd.DataFrame,
    bank: StateBank,
    cache: TeacherScoreCache,
    config: ProjectConfig,
) -> OnPolicyAudit:
    """Compare state sources after matching declared distribution-shift covariates."""

    states = _state_metrics(position_metrics, bank, config)
    match_tables: list[pd.DataFrame] = []
    balance_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    comparisons = {
        "on_policy_vs_reference_offline": config.on_policy.reference_offline_kinds,
        "on_policy_vs_model_aware": [config.on_policy.model_aware_kind],
    }
    for role_index, (analysis_role, role_states) in enumerate(
        states.groupby("analysis_role", observed=True)
    ):
        for comparison_index, (comparison, control_kinds) in enumerate(comparisons.items()):
            on_policy = role_states.loc[
                role_states["state_kind"] == config.on_policy.on_policy_kind
            ].copy()
            controls = role_states.loc[role_states["state_kind"].isin(control_kinds)].copy()
            matches = _greedy_match(on_policy, controls, comparison, analysis_role, config)
            match_tables.append(matches)
            balance = _balance_table(
                on_policy, controls, matches, comparison, analysis_role, config
            )
            balance_tables.append(balance)
            summary_rows.extend(
                _effect_rows(
                    matches,
                    comparison,
                    analysis_role,
                    len(on_policy),
                    balance,
                    config,
                    config.seed + 900 + role_index * 10_000 + comparison_index * 100,
                )
            )
    matches = pd.concat(match_tables, ignore_index=True) if match_tables else pd.DataFrame()
    balance = pd.concat(balance_tables, ignore_index=True) if balance_tables else pd.DataFrame()
    timestep = _timestep_summary(states, config)
    compute = _compute_summary(bank, cache, config)
    return OnPolicyAudit(
        state_metrics=states,
        matches=matches,
        effect_summary=pd.DataFrame(summary_rows),
        balance=balance,
        timestep_summary=timestep,
        compute_summary=compute,
    )


def _state_metrics(positions: pd.DataFrame, bank: StateBank, config: ProjectConfig) -> pd.DataFrame:
    paired = positions.loc[
        (positions["teacher_id"] == config.audit.primary_teacher_id)
        & (positions["structure_role"] == config.audit.paired_role)
    ].copy()
    if paired.empty:
        raise ValueError("on-policy audit lacks primary paired-teacher scores")
    edited = paired.loc[paired["is_corrupted"]].copy()
    if edited.empty:
        raise ValueError("on-policy audit requires states with edited or masked positions")
    edited["high_confidence_error"] = (
        edited["student_top1_margin"] >= config.on_policy.high_confidence_margin
    ).astype(float)
    aggregate = (
        edited.groupby(["state_id", "domain_id"], observed=True)
        .agg(
            teacher_advantage_nats=("advantage_nats", "mean"),
            high_confidence_error_fraction=("high_confidence_error", "mean"),
            edited_positions=("position", "size"),
        )
        .reset_index()
    )
    state_columns = [
        "state_id",
        "domain_id",
        "dataset",
        "analysis_role",
        "eligible_for_training",
        "state_kind",
        "requested_corruption_ratio",
        "corruption_ratio",
        "edit_distance_fraction",
        "mask_count",
        "student_entropy",
        "student_top1_margin",
        "timestep",
        "scaffold_compatibility",
        "sequence_policy_calls",
    ]
    return bank.states[state_columns].merge(
        aggregate, on=["state_id", "domain_id"], validate="one_to_one"
    )


def _greedy_match(
    on_policy: pd.DataFrame,
    controls: pd.DataFrame,
    comparison: str,
    analysis_role: str,
    config: ProjectConfig,
) -> pd.DataFrame:
    match_columns = config.on_policy.match_columns
    required = set(match_columns) - set(on_policy.columns)
    if required:
        raise ValueError(f"unknown on-policy match columns: {sorted(required)}")
    pooled = pd.concat([on_policy[match_columns], controls[match_columns]], ignore_index=True)
    scaler = StandardScaler().fit(pooled.to_numpy(dtype=float))
    on_scaled = scaler.transform(on_policy[match_columns].to_numpy(dtype=float))
    control_scaled = scaler.transform(controls[match_columns].to_numpy(dtype=float))
    on_lookup = {index: row for index, row in enumerate(on_policy.itertuples(index=False))}
    control_lookup = {index: row for index, row in enumerate(controls.itertuples(index=False))}
    rows: list[dict[str, Any]] = []
    strata = on_policy[["domain_id", "requested_corruption_ratio"]].drop_duplicates()
    for stratum in strata.itertuples(index=False):
        on_indices = np.flatnonzero(
            (on_policy["domain_id"].to_numpy() == stratum.domain_id)
            & np.isclose(
                on_policy["requested_corruption_ratio"].to_numpy(dtype=float),
                float(stratum.requested_corruption_ratio),
            )
        )
        control_indices = np.flatnonzero(
            (controls["domain_id"].to_numpy() == stratum.domain_id)
            & np.isclose(
                controls["requested_corruption_ratio"].to_numpy(dtype=float),
                float(stratum.requested_corruption_ratio),
            )
        )
        if not len(on_indices) or not len(control_indices):
            continue
        distances = cdist(on_scaled[on_indices], control_scaled[control_indices])
        candidates = sorted(
            (
                (float(distances[left, right]), int(on_indices[left]), int(control_indices[right]))
                for left in range(len(on_indices))
                for right in range(len(control_indices))
                if distances[left, right] <= config.on_policy.match_caliper
            ),
            key=lambda item: (item[0], item[1], item[2]),
        )
        used_on: set[int] = set()
        used_control: set[int] = set()
        for distance, on_index, control_index in candidates:
            if on_index in used_on or control_index in used_control:
                continue
            used_on.add(on_index)
            used_control.add(control_index)
            on_row = on_lookup[on_index]
            control_row = control_lookup[control_index]
            row: dict[str, Any] = {
                "comparison": comparison,
                "analysis_role": analysis_role,
                "domain_id": on_row.domain_id,
                "requested_corruption_ratio": on_row.requested_corruption_ratio,
                "on_state_id": on_row.state_id,
                "control_state_id": control_row.state_id,
                "control_state_kind": control_row.state_kind,
                "match_distance": distance,
                "on_teacher_advantage_nats": on_row.teacher_advantage_nats,
                "control_teacher_advantage_nats": control_row.teacher_advantage_nats,
                "teacher_advantage_difference_nats": (
                    on_row.teacher_advantage_nats - control_row.teacher_advantage_nats
                ),
                "on_high_confidence_error_fraction": on_row.high_confidence_error_fraction,
                "control_high_confidence_error_fraction": (
                    control_row.high_confidence_error_fraction
                ),
                "high_confidence_error_fraction_difference": (
                    on_row.high_confidence_error_fraction
                    - control_row.high_confidence_error_fraction
                ),
                "on_timestep": on_row.timestep,
            }
            for column in match_columns:
                row[f"on_{column}"] = getattr(on_row, column)
                row[f"control_{column}"] = getattr(control_row, column)
            rows.append(row)
    return pd.DataFrame(rows)


def _balance_table(
    on_policy: pd.DataFrame,
    controls: pd.DataFrame,
    matches: pd.DataFrame,
    comparison: str,
    analysis_role: str,
    config: ProjectConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in config.on_policy.match_columns:
        before = _standardized_mean_difference(on_policy[column], controls[column])
        after = (
            _standardized_mean_difference(matches[f"on_{column}"], matches[f"control_{column}"])
            if not matches.empty
            else float("nan")
        )
        rows.append(
            {
                "comparison": comparison,
                "analysis_role": analysis_role,
                "variable": column,
                "smd_before": before,
                "smd_after": after,
                "absolute_smd_after": abs(after),
                "threshold": config.on_policy.maximum_standardized_mean_difference,
                "passes": bool(
                    np.isfinite(after)
                    and abs(after) <= config.on_policy.maximum_standardized_mean_difference
                ),
            }
        )
    return pd.DataFrame(rows)


def _effect_rows(
    matches: pd.DataFrame,
    comparison: str,
    analysis_role: str,
    total_on_policy: int,
    balance: pd.DataFrame,
    config: ProjectConfig,
    seed: int,
) -> list[dict[str, Any]]:
    match_fraction = len(matches) / max(1, total_on_policy)
    maximum_smd = float(balance["absolute_smd_after"].max()) if not balance.empty else float("nan")
    quality_pass = bool(
        match_fraction >= config.on_policy.minimum_match_fraction
        and np.isfinite(maximum_smd)
        and maximum_smd <= config.on_policy.maximum_standardized_mean_difference
    )
    rows: list[dict[str, Any]] = []
    for metric_index, metric in enumerate(
        [
            "teacher_advantage_difference_nats",
            "high_confidence_error_fraction_difference",
        ]
    ):
        estimate = cluster_bootstrap_mean(
            matches,
            metric,
            config.audit.cluster_column,
            config.audit.bootstrap_replicates,
            config.audit.confidence_level,
            seed + metric_index,
        )
        rows.append(
            {
                "comparison": comparison,
                "analysis_role": analysis_role,
                "metric": metric,
                **estimate,
                "matched_pairs": len(matches),
                "eligible_on_policy_states": total_on_policy,
                "match_fraction": match_fraction,
                "maximum_absolute_smd": maximum_smd,
                "matching_quality_pass": quality_pass,
            }
        )
    return rows


def _timestep_summary(states: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    on_policy = states.loc[states["state_kind"] == config.on_policy.on_policy_kind]
    if on_policy.empty:
        return pd.DataFrame()
    return grouped_cluster_means(
        on_policy,
        ["analysis_role", "timestep", "requested_corruption_ratio"],
        "teacher_advantage_nats",
        config.audit.cluster_column,
        config.audit.bootstrap_replicates,
        config.audit.confidence_level,
        config.seed + 1000,
    )


def _compute_summary(
    bank: StateBank, cache: TeacherScoreCache, config: ProjectConfig
) -> pd.DataFrame:
    score_requests = (
        cache.scores.loc[cache.scores["teacher_id"] == config.audit.primary_teacher_id]
        .groupby(["state_id", "structure_role", "structure_id"], observed=True, dropna=False)[
            ["wall_seconds", "forward_calls"]
        ]
        .max()
        .reset_index()
    )
    score_requests = score_requests.merge(
        bank.states[["state_id", "state_kind", "analysis_role"]],
        on="state_id",
        validate="many_to_one",
    )
    kinds = {
        "reference_offline": config.on_policy.reference_offline_kinds,
        "model_aware_offline": [config.on_policy.model_aware_kind],
        "on_policy": [config.on_policy.on_policy_kind],
    }
    rows: list[dict[str, Any]] = []
    for analysis_role in sorted(bank.states["analysis_role"].unique()):
        for condition, state_kinds in kinds.items():
            states = bank.states.loc[
                (bank.states["analysis_role"] == analysis_role)
                & (bank.states["state_kind"].isin(state_kinds))
            ]
            requests = score_requests.loc[
                (score_requests["analysis_role"] == analysis_role)
                & (score_requests["state_kind"].isin(state_kinds))
            ]
            rows.append(
                {
                    "analysis_role": analysis_role,
                    "condition": condition,
                    "state_kinds": state_kinds,
                    "states": len(states),
                    "domains": states["domain_id"].nunique(),
                    "sequence_policy_calls": int(states["sequence_policy_calls"].sum()),
                    "teacher_requests": len(requests),
                    "teacher_forward_calls": int(requests["forward_calls"].sum()),
                    "teacher_wall_seconds": float(requests["wall_seconds"].sum()),
                }
            )
    return pd.DataFrame(rows)


def _standardized_mean_difference(left: pd.Series, right: pd.Series) -> float:
    left_values = left.to_numpy(dtype=float)
    right_values = right.to_numpy(dtype=float)
    if not len(left_values) or not len(right_values):
        return float("nan")
    pooled_variance = (
        (np.var(left_values, ddof=1) if len(left_values) > 1 else 0.0)
        + (np.var(right_values, ddof=1) if len(right_values) > 1 else 0.0)
    ) / 2.0
    difference = float(np.mean(left_values) - np.mean(right_values))
    if pooled_variance == 0:
        return 0.0 if np.isclose(difference, 0.0) else float("inf")
    return difference / np.sqrt(pooled_variance)
