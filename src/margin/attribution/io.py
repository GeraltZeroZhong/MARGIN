"""Persist audit evidence, machine-readable results, and figure source data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from margin.attribution.decision import FoundationDecision
from margin.attribution.distillability import DistillabilityAudit
from margin.attribution.observability import ObservabilityAudit
from margin.attribution.on_policy import OnPolicyAudit
from margin.attribution.teacher_value import TeacherValueAudit
from margin.config import ProjectConfig
from margin.provenance import (
    runtime_manifest,
    sha256_file,
    table_manifest,
    write_csv,
    write_json,
    write_parquet,
)


def write_audit_bundle(
    teacher: TeacherValueAudit,
    observability: ObservabilityAudit,
    on_policy: OnPolicyAudit,
    distillability: DistillabilityAudit,
    decision_result: FoundationDecision,
    config: ProjectConfig,
    upstream_manifests: list[Path],
) -> dict[str, Any]:
    """Write every declared audit table and a manifest that links its inputs."""

    directory = config.paths.audit_dir
    directory.mkdir(parents=True, exist_ok=True)
    tables = {
        "teacher_position_metrics": teacher.position_metrics,
        "native_nll_summary": teacher.nll_summary,
        "candidate_ranking_summary": teacher.ranking_summary,
        "teacher_environment_summary": teacher.environment_summary,
        "specificity_positions": teacher.specificity_positions,
        "paired_decoy_summary": teacher.specificity_summary,
        "teacher_agreement_summary": teacher.agreement_summary,
        "dms_predictions": teacher.dms_predictions,
        "dms_summary": teacher.dms_summary,
        "dms_coverage": teacher.dms_coverage,
        "observability_rows": observability.row_metrics,
        "observability_summary": observability.summary,
        "observability_environment_summary": observability.environment_summary,
        "observability_feature_manifest": observability.feature_manifest,
        "on_policy_state_metrics": on_policy.state_metrics,
        "on_policy_matches": on_policy.matches,
        "on_policy_effect_summary": on_policy.effect_summary,
        "on_policy_balance": on_policy.balance,
        "on_policy_timestep_summary": on_policy.timestep_summary,
        "compute_summary": on_policy.compute_summary,
        "distillability_map": distillability.map_table,
        "teacher_action_valid_radius": distillability.action_valid_radius,
        "decision_criteria": _serializable_criteria(decision_result.criteria),
    }
    table_records: dict[str, Any] = {}
    for name, table in tables.items():
        path = directory / f"{name}.parquet"
        write_parquet(path, table)
        table_records[name] = table_manifest(path, table)
    decision_path = directory / "foundation_decision.json"
    write_json(decision_path, decision_result.decision_record)
    result_table = _machine_result_table(decision_result, config)
    result_path = directory / "audit_result_table.parquet"
    write_parquet(result_path, result_table)
    table_records["audit_result_table"] = table_manifest(result_path, result_table)

    source_records = _write_source_data(
        teacher, observability, on_policy, distillability, decision_result, config
    )
    manifest = {
        **runtime_manifest(config.paths.project_root),
        "schema_version": config.schema_version,
        "data_mode": config.data_mode,
        "decision": decision_result.decision,
        "parameters": {
            "audit": config.audit.model_dump(mode="json"),
            "observability": config.observability.model_dump(mode="json"),
            "on_policy": config.on_policy.model_dump(mode="json"),
            "decision": config.decision.model_dump(mode="json"),
        },
        "tables": table_records,
        "source_data": source_records,
        "upstream_manifests": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in upstream_manifests
            if path.exists()
        ],
    }
    write_json(directory / "manifest.json", manifest)
    return manifest


def _write_source_data(
    teacher: TeacherValueAudit,
    observability: ObservabilityAudit,
    on_policy: OnPolicyAudit,
    distillability: DistillabilityAudit,
    decision_result: FoundationDecision,
    config: ProjectConfig,
) -> dict[str, Any]:
    directory = config.paths.source_data_dir
    directory.mkdir(parents=True, exist_ok=True)
    sources = {
        "figure_1_distillability_map": distillability.map_table,
        "figure_2_paired_decoy": teacher.specificity_summary,
        "figure_3_observability": observability.environment_summary,
        "figure_4_on_policy": on_policy.effect_summary,
        "decision_criteria": _serializable_criteria(decision_result.criteria),
    }
    records: dict[str, Any] = {}
    for name, table in sources.items():
        parquet_path = directory / f"{name}.parquet"
        csv_path = directory / f"{name}.csv"
        write_parquet(parquet_path, table)
        write_csv(csv_path, table)
        records[name] = {
            "parquet": table_manifest(parquet_path, table),
            "csv_path": str(csv_path),
            "csv_sha256": sha256_file(csv_path),
        }
    return records


def _serializable_criteria(criteria: pd.DataFrame) -> pd.DataFrame:
    table = criteria.copy()
    if "details" in table:
        table["details"] = table["details"].map(
            lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False)
        )
    return table


def _machine_result_table(
    decision_result: FoundationDecision, config: ProjectConfig
) -> pd.DataFrame:
    table = _serializable_criteria(decision_result.criteria)
    table.insert(0, "project", config.project_name)
    table.insert(1, "schema_version", config.schema_version)
    table.insert(2, "data_mode", config.data_mode)
    table.insert(3, "decision", decision_result.decision)
    return table
