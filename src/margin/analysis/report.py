"""Generate a source-linked foundation decision report and run handoff documentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from margin.attribution.decision import FoundationDecision
from margin.attribution.observability import ObservabilityAudit
from margin.attribution.on_policy import OnPolicyAudit
from margin.attribution.teacher_value import TeacherValueAudit
from margin.config import ProjectConfig
from margin.data_registry.leakage import LeakageAudit
from margin.decoys.generate import DecoyArtifacts
from margin.provenance import sha256_file, write_text
from margin.state_sampling.bank import StateBank
from margin.teachers.cache import TeacherScoreCache


def write_foundation_report(
    decision_result: FoundationDecision,
    leakage: LeakageAudit,
    bank: StateBank,
    decoys: DecoyArtifacts,
    cache: TeacherScoreCache,
    teacher: TeacherValueAudit,
    observability: ObservabilityAudit,
    on_policy: OnPolicyAudit,
    config: ProjectConfig,
) -> Path:
    """Write the decision report with direct links to every evidence class."""

    criteria_rows = "\n".join(
        f"| {row.criterion} | {row.status} | {_number(row.estimate)} | "
        f"{_interval(row.ci_low, row.ci_high)} | {_number(row.threshold)} |"
        for row in decision_result.criteria.itertuples(index=False)
    )
    coverage = (
        teacher.position_metrics.groupby(
            ["teacher_id", "structure_role", "analysis_role"], observed=True
        )
        .agg(
            rows=("position", "size"),
            states=("state_id", "nunique"),
            domains=("domain_id", "nunique"),
        )
        .reset_index()
    )
    coverage_rows = "\n".join(
        f"| {row.teacher_id} | {row.structure_role} | {row.analysis_role} | {row.rows:,} | "
        f"{row.states:,} | {row.domains:,} |"
        for row in coverage.itertuples(index=False)
    )
    decision_note = (
        "This is a software-validation result only. Synthetic inputs and synthetic DMS labels "
        "must not be cited as biological evidence."
        if decision_result.decision == "SYNTHETIC_ONLY"
        else decision_result.decision_record["rationale"]
    )
    complete_dms_cells = int(
        (teacher.dms_coverage.get("status", pd.Series(dtype=str)) == "complete").sum()
    )
    dms_note = (
        f"{len(teacher.dms_predictions):,} scored teacher-variant rows; "
        f"{complete_dms_cells:,}/{len(teacher.dms_coverage):,} teacher-assay-domain coverage "
        f"cells complete; {len(teacher.dms_summary):,} assay/pooled summaries."
        if not teacher.dms_predictions.empty
        else "No DMS input was available; the independent experimental criterion is incomplete."
    )
    domain_roles = (
        bank.states[["domain_id", "analysis_role"]]
        .drop_duplicates()
        .groupby("analysis_role", observed=True)["domain_id"]
        .nunique()
        .to_dict()
    )
    gate_on_policy = on_policy.effect_summary
    if config.audit.decision_analysis_role != "all" and not gate_on_policy.empty:
        gate_on_policy = gate_on_policy.loc[
            gate_on_policy["analysis_role"] == config.audit.decision_analysis_role
        ]
    gate_on_policy = gate_on_policy.loc[
        gate_on_policy.get("metric", pd.Series(dtype=str)) == "teacher_advantage_difference_nats"
    ]
    gate_matches = on_policy.matches
    if config.audit.decision_analysis_role != "all" and not gate_matches.empty:
        gate_matches = gate_matches.loc[
            gate_matches["analysis_role"] == config.audit.decision_analysis_role
        ]
    incomplete_features = observability.feature_manifest.loc[
        observability.feature_manifest.get("status", pd.Series(dtype=str)) != "complete"
    ]
    observability_note = (
        "All configured group levels completed cross-fitting."
        if incomplete_features.empty
        else "; ".join(incomplete_features["reason"].astype(str))
    )
    report = f"""# MARGIN foundation decision report

Decision: **{decision_result.decision}**
Data mode: `{config.data_mode}`
Schema: `{config.schema_version}`
Seed: `{config.seed}`

{decision_note}

## Decision path

```mermaid
flowchart TD
    A[Paired teacher action value] -->|absent| N[NO_GO]
    A -->|present| S[Paired beats matched decoy]
    S -->|no| N
    S -->|yes| D[Independent DMS ranking]
    D -->|missing| I[INCOMPLETE]
    D -->|yes| O[CATH-H residual observable]
    O -->|no| P[PIVOT_STRUCTURE_CONDITIONED]
    O -->|yes| R[Reliable radius and teacher consistency]
    R -->|no| N
    R -->|yes| Q[On-policy adds matched value]
    Q -->|no| X[DROP_ON_POLICY]
    Q -->|yes| E[Experimental evidence scope]
    E -->|stability only| Y[NARROW_STABILITY]
    E -->|multiple assay classes| G[GO]
```

The synthetic guard is evaluated after this logic and forces `SYNTHETIC_ONLY` for fixture runs.

## Five foundation audit questions

| Question | Evidence |
|---|---|
| Does structure improve residue actions? | Core and regular-secondary-structure native-NLL reductions with domain-clustered CIs. |
| Is the gain target-specific? | Paired-minus-matched-CATH and destructive correspondence controls. |
| Can sequence recover the residual? | Frozen-feature Ridge probes cross-fitted by CATH-H and CATH-T, with shuffled-target controls. |
| How far is the scaffold reliable? | Environment-specific maximum corruption level retaining positive action value. |
| Is on-policy necessary? | Within-domain, corruption-matched comparisons against reference and model-aware offline states, with SMD balance diagnostics. |

## Fixed criteria

| Criterion | Status | Estimate | 95% CI | Threshold |
|---|---:|---:|---:|---:|
{criteria_rows}

Full machine-readable criteria are in [`../audit/decision_criteria.parquet`](../audit/decision_criteria.parquet), and the decision record is in [`../audit/foundation_decision.json`](../audit/foundation_decision.json).

## Data and leakage

- Candidate domains: {leakage.summary["total_domains"]:,}
- Eligible training domains: {leakage.summary["eligible_domains"]:,}
- Benchmark-excluded candidate domains: {leakage.summary["excluded_domains"]:,}
- External audit domains in the unified registry: {domain_roles.get("external_benchmark", 0):,}
- Final registry domains: {sum(domain_roles.values()):,}
- Recorded leakage relations: {leakage.summary["relations"]:,}
- State bank: {len(bank.states):,} states and {len(bank.positions):,} state-position rows
- Decoys: {len(decoys.decoys):,} declarations; {len(decoys.skipped):,} skipped matched controls
- Gate analysis population: `{config.audit.decision_analysis_role}`

The registry and leakage evidence are in [`../registry/manifest.json`](../registry/manifest.json) and [`../registry/leakage/leakage_manifest.json`](../registry/leakage/leakage_manifest.json).

## Teacher matrix coverage

| Teacher | Structure role | Analysis role | Rows | States | Domains |
|---|---|---|---:|---:|---:|
{coverage_rows}

All imported scores are normalized natural-log probabilities over the canonical amino-acid order `ACDEFGHIKLMNPQRSTVWY`. Model-specific conditioning semantics remain explicit in the cache.

## External and observability evidence

- DMS: {dms_note}
- DMS coverage table: [`../audit/dms_coverage.parquet`](../audit/dms_coverage.parquet)
- Experimental evidence scope: `{decision_result.decision_record.get("experimental_evidence_scope", "unspecified")}`
- Observability: {observability_note}
- Matched on-policy pairs in the Gate population: {len(gate_matches):,}
- Matching comparisons passing the predeclared quality rule: {int(gate_on_policy.get("matching_quality_pass", pd.Series(dtype=bool)).sum()):,}/{len(gate_on_policy):,}

## Figures and source data

- [`../figures/figure_1_distillability_map.pdf`](../figures/figure_1_distillability_map.pdf)
- [`../figures/figure_2_audit_overview.pdf`](../figures/figure_2_audit_overview.pdf)
- [`../source_data/figure_1_distillability_map.csv`](../source_data/figure_1_distillability_map.csv)
- [`../source_data/figure_2_paired_decoy.csv`](../source_data/figure_2_paired_decoy.csv)
- [`../source_data/figure_3_observability.csv`](../source_data/figure_3_observability.csv)
- [`../source_data/figure_4_on_policy.csv`](../source_data/figure_4_on_policy.csv)

## Interpretation boundary

The Gate reports effect sizes and domain-cluster bootstrap intervals; positions from the same protein are not treated as independent replicates. A passing software fixture validates schemas, joins, controls, fitting isolation, and branch logic. Only a real-data run with frozen student embeddings, external benchmark labels, complete teacher coverage, and acceptable matching balance can support a biological decision.

The resolved configuration and run manifest record the executable workflow used for this run.
"""
    path = config.paths.report_dir / "foundation_report.md"
    write_text(path, report)
    return path


def write_run_index(config: ProjectConfig, artifacts: list[Path]) -> Path:
    rows = "\n".join(
        f"- [`{path.relative_to(config.paths.run_dir)}`]"
        f"(../{path.relative_to(config.paths.run_dir)}) "
        f"— SHA-256 `{sha256_file(path)}`"
        for path in artifacts
        if path.exists() and path.is_file()
    )
    payload = f"""# MARGIN foundation-audit run index

Data mode: `{config.data_mode}`
Configuration schema: `{config.schema_version}`

## Primary artifacts

{rows}
"""
    path = config.paths.report_dir / "run_index.md"
    write_text(path, payload)
    return path


def _number(value: Any) -> str:
    if value is None:
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:.4g}" if np.isfinite(numeric) else "—"


def _interval(low: Any, high: Any) -> str:
    if low is None or high is None:
        return "—"
    try:
        low_value, high_value = float(low), float(high)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(low_value) or not np.isfinite(high_value):
        return "—"
    return f"[{low_value:.4g}, {high_value:.4g}]"
