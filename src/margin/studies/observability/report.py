"""Decision logic and source-linked report for the locked observability study replication."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from margin.provenance import (
    runtime_manifest,
    table_manifest,
    write_json,
    write_parquet,
    write_text,
)
from margin.studies.observability.config import ObservabilityStudyConfig

CONTROL_NAMES = (
    "global",
    "within_domain",
    "within_wild_type",
    "within_environment",
    "within_corruption",
    "fully_conditioned",
)


def build_observability_report(config: ObservabilityStudyConfig) -> dict[str, Path]:
    """Evaluate the frozen success rule and write the canonical observability study handoff."""

    root = config.paths.run_dir
    required = _required_inputs(root)
    missing = [path for path in required if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"observability study report inputs are incomplete:\n{formatted}")

    decisions = _locked_probe_decisions(config)
    environment_decisions = _locked_environment_decisions(config)
    decision, rationale = _select_decision(decisions, environment_decisions)
    environment_overview = _environment_overview(config, environment_decisions)

    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    probe_path = report_dir / "locked_probe_decisions.parquet"
    environment_path = report_dir / "candidate_environment_decisions.parquet"
    overview_path = report_dir / "candidate_environment_overview.parquet"
    write_parquet(probe_path, decisions)
    write_parquet(environment_path, environment_decisions)
    write_parquet(overview_path, environment_overview)

    decision_record = _decision_record(
        config, decision, rationale, decisions, environment_decisions
    )
    decision_path = report_dir / "observability_decision.json"
    write_json(decision_path, decision_record)
    report_path = report_dir / "observability_report.md"
    write_text(
        report_path,
        _render_report(
            config,
            decision_record,
            decisions,
            environment_overview,
        ),
    )
    manifest_path = report_dir / "manifest.json"
    tables = {
        "locked_probe_decisions": (probe_path, decisions),
        "candidate_environment_decisions": (
            environment_path,
            environment_decisions,
        ),
        "candidate_environment_overview": (overview_path, environment_overview),
    }
    write_json(
        manifest_path,
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "decision": decision,
            "historical_gate_modified": False,
            "report": str(report_path),
            "decision_record": str(decision_path),
            "tables": {name: table_manifest(path, table) for name, (path, table) in tables.items()},
            "inputs": [str(path) for path in required],
        },
    )
    return {
        "report": report_path,
        "decision": decision_path,
        "probe_decisions": probe_path,
        "environment_decisions": environment_path,
        "environment_overview": overview_path,
        "manifest": manifest_path,
    }


def evaluate_probe_summary(
    summary: pd.DataFrame,
    config: ObservabilityStudyConfig,
    *,
    source_id: str,
    model_id: str,
    allowed_probes: set[str],
    decision_eligible: bool,
    scope: str,
) -> pd.DataFrame:
    """Apply the registered metric, interval, and control rule to probe summaries."""

    primary = config.residual_targets.primary
    observed = summary.loc[
        summary["target_id"].eq(primary)
        & summary["metric"].eq("jsd_reduction_nats")
        & summary["control"].eq("observed")
        & summary["probe"].isin(allowed_probes)
    ].copy()
    records: list[dict[str, Any]] = []
    group_columns = [
        column
        for column in (
            "target_id",
            "probe",
            "feature_kind",
            "layer",
            "target_rank",
            "target_coordinates",
            "evaluation_split",
            "environment_id",
        )
        if column in summary
    ]
    for row in observed.itertuples(index=False):
        group = summary
        for column in group_columns:
            value = getattr(row, column)
            group = group.loc[_matches(group[column], value)]
        cosine = _single_metric(group, "residual_cosine", "observed")
        cross_entropy = _single_metric(group, "cross_entropy_reduction_nats", "observed")
        rank_agreement = _single_metric(group, "candidate_rank_agreement", "observed")
        expected = _expected_controls(row.probe)
        control_rows = group.loc[
            group["metric"].eq("jsd_reduction_nats") & group["control"].isin(expected)
        ]
        counts = control_rows.groupby("control", observed=True)["repeat"].nunique()
        controls_complete = all(
            int(counts.get(name, 0)) == config.probes.control_repeats for name in expected
        )
        maximum_control = (
            float(control_rows["estimate"].max()) if not control_rows.empty else np.nan
        )
        jsd_pass = float(row.estimate) >= config.inference.minimum_jsd_reduction_nats
        cosine_pass = np.isfinite(cosine) and cosine >= config.inference.minimum_residual_cosine
        ci_pass = (
            float(row.wild_ci_low) > 0 if config.inference.require_positive_ci_lower_bound else True
        )
        controls_pass = controls_complete and float(row.estimate) > maximum_control
        if not controls_complete or not np.isfinite(cosine):
            status = "INCOMPLETE"
        elif jsd_pass and cosine_pass and ci_pass and controls_pass:
            status = "PASS"
        else:
            status = "FAIL"
        rank = getattr(row, "target_rank", np.nan)
        feature = getattr(row, "feature_kind", "none")
        layer = getattr(row, "layer", np.nan)
        environment = getattr(row, "environment_id", None)
        records.append(
            {
                "route_id": _route_id(model_id, row.probe, feature, layer, rank, environment),
                "scope": scope,
                "source_id": source_id,
                "model_id": model_id,
                "probe": row.probe,
                "feature_kind": feature,
                "layer": layer,
                "target_rank": rank,
                "environment_id": environment,
                "decision_eligible": decision_eligible,
                "status": status,
                "jsd_reduction_nats": float(row.estimate),
                "jsd_ci_low": float(row.wild_ci_low),
                "jsd_ci_high": float(row.wild_ci_high),
                "cross_entropy_reduction_nats": cross_entropy,
                "residual_cosine": cosine,
                "candidate_rank_agreement": rank_agreement,
                "maximum_control_jsd_reduction_nats": maximum_control,
                "controls_complete": controls_complete,
                "jsd_threshold_pass": jsd_pass,
                "cosine_threshold_pass": cosine_pass,
                "positive_ci_pass": ci_pass,
                "controls_pass": controls_pass,
                "positive_domains": int(row.positive_domains),
                "negative_domains": int(row.negative_domains),
                "n_domains": int(row.n_domains),
                "n_rows": int(row.n_rows),
            }
        )
    return pd.DataFrame.from_records(records)


def select_observability_decision(
    decisions: pd.DataFrame, environment_decisions: pd.DataFrame
) -> tuple[str, str]:
    """Deterministic form of the observability-study branch logic."""

    return _select_decision(decisions, environment_decisions)


def _locked_probe_decisions(config: ObservabilityStudyConfig) -> pd.DataFrame:
    root = config.paths.run_dir
    specifications = (
        (
            "replication_final_layer",
            "esm2_150m",
            {"fixed_final_layer_ridge"},
            False,
            "probe_summary.parquet",
        ),
        (
            "layerwise_replication",
            "esm2_150m",
            {"layerwise_ridge", "reduced_rank_ridge", "bottleneck_mlp"},
            True,
            "summary.parquet",
        ),
        (
            "carp_replication",
            "carp_640m",
            {"layerwise_ridge", "reduced_rank_ridge", "bottleneck_mlp"},
            True,
            "summary.parquet",
        ),
        (
            "lora_replication",
            "esm2_150m_lora",
            {"esm2_lora"},
            True,
            "summary.parquet",
        ),
    )
    tables = []
    for source_id, model_id, probes, eligible, filename in specifications:
        summary = pd.read_parquet(root / source_id / filename)
        tables.append(
            evaluate_probe_summary(
                summary,
                config,
                source_id=source_id,
                model_id=model_id,
                allowed_probes=probes,
                decision_eligible=eligible,
                scope="global",
            )
        )
    return pd.concat(tables, ignore_index=True)


def _locked_environment_decisions(config: ObservabilityStudyConfig) -> pd.DataFrame:
    root = config.paths.run_dir
    specifications = (
        (
            "layerwise_replication",
            "esm2_150m",
            {"layerwise_ridge", "reduced_rank_ridge", "bottleneck_mlp"},
        ),
        (
            "carp_replication",
            "carp_640m",
            {"layerwise_ridge", "reduced_rank_ridge", "bottleneck_mlp"},
        ),
        ("lora_replication", "esm2_150m_lora", {"esm2_lora"}),
    )
    tables = []
    for source_id, model_id, probes in specifications:
        summary = pd.read_parquet(root / source_id / "environment_summary.parquet")
        tables.append(
            evaluate_probe_summary(
                summary,
                config,
                source_id=source_id,
                model_id=model_id,
                allowed_probes=probes,
                decision_eligible=True,
                scope="candidate_environment",
            )
        )
    return pd.concat(tables, ignore_index=True)


def _select_decision(
    decisions: pd.DataFrame, environment_decisions: pd.DataFrame
) -> tuple[str, str]:
    passed = decisions.loc[
        decisions["decision_eligible"].astype(bool) & decisions["status"].eq("PASS")
    ]
    global_ridge = passed.loc[
        passed["model_id"].eq("esm2_150m") & passed["probe"].eq("layerwise_ridge")
    ]
    if not global_ridge.empty:
        return (
            "NEW_INDEPENDENT_SEQUENCE_ONLY_PROTOCOL_REQUIRED",
            "The validation-selected ESM2 Ridge probe passed every locked global criterion; "
            "the foundation decision remains unchanged, and any sequence-only continuation "
            "requires a new independent protocol.",
        )
    if not passed.empty:
        return (
            "REPRESENTATION_ACCESSIBILITY_LIMITATION",
            "At least one adapted, nonlinear, reduced-rank, or alternate sequence route passed "
            "while the registered ESM2 Ridge route did not.",
        )
    environment_passed = environment_decisions.loc[environment_decisions["status"].eq("PASS")]
    if not environment_passed.empty:
        return (
            "SELECTIVE_HYBRID",
            "No global sequence route passed, but at least one predeclared candidate "
            "environment passed the full locked success rule.",
        )
    return (
        "RETAIN_STRUCTURE_AS_INFERENCE_MODALITY",
        "No executed global probe or predeclared candidate environment passed the full "
        "locked success rule.",
    )


def _environment_overview(config: ObservabilityStudyConfig, locked: pd.DataFrame) -> pd.DataFrame:
    root = config.paths.run_dir
    distillability = pd.read_parquet(
        config.paths.foundation_run / "audit" / "distillability_map.parquet"
    )
    current_summary = pd.read_parquet(root / "candidate_environments_current" / "summary.parquet")
    current_coverage = pd.read_parquet(root / "candidate_environments_current" / "coverage.parquet")
    dms = pd.read_parquet(root / "dms_residual" / "environment_increment_summary.parquet")
    records = []
    for environment in config.candidate_environments:
        foundation = distillability.loc[
            distillability["teacher_id"].eq(config.residual_targets.primary)
            & distillability["analysis_role"].eq("external_benchmark")
            & distillability["state_kind"].eq(environment.state_kind)
            & distillability["requested_corruption_ratio"].eq(
                environment.requested_corruption_ratio
            )
            & distillability["environment_axis"].eq(environment.axis)
            & distillability["environment"].eq(environment.value)
        ]
        current = current_summary.loc[
            current_summary["environment_id"].eq(environment.environment_id)
            & current_summary["target_id"].eq(config.residual_targets.primary)
            & current_summary["probe"].eq("final_layer_ridge")
            & current_summary["control"].eq("observed")
            & current_summary["metric"].eq("jsd_reduction_nats")
        ]
        coverage = current_coverage.loc[
            current_coverage["environment_id"].eq(environment.environment_id)
        ]
        dms_row = dms.loc[
            dms["teacher_id"].eq(config.residual_targets.primary)
            & dms["decoy_role"].eq("matched_cath")
            & dms["method"].eq("sequence_plus_structural_residual")
            & dms["environment_axis"].eq(environment.axis)
            & dms["environment"].eq(environment.value)
        ]
        locked_rows = locked.loc[
            locked["environment_id"].eq(environment.environment_id)
        ].sort_values(["status", "jsd_reduction_nats"], ascending=[False, False])
        best = _best_locked_environment(locked_rows)
        foundation_row = _one(foundation, "foundation audit distillability environment")
        current_row = _one(current, "current environment sensitivity")
        coverage_row = _one(coverage, "current environment coverage")
        dms_value = _one(dms_row, "DMS residual environment")
        records.append(
            {
                "environment_id": environment.environment_id,
                "state_kind": environment.state_kind,
                "requested_corruption_ratio": environment.requested_corruption_ratio,
                "environment_axis": environment.axis,
                "environment_value": environment.value,
                "current_domains": int(coverage_row.observed_domains),
                "action_value_nats": float(foundation_row.teacher_advantage_nats),
                "action_value_ci_low": float(foundation_row.teacher_advantage_ci_low),
                "action_value_ci_high": float(foundation_row.teacher_advantage_ci_high),
                "specificity_nats": float(foundation_row.matched_decoy_lift_nats),
                "specificity_ci_low": float(foundation_row.matched_decoy_lift_ci_low),
                "specificity_ci_high": float(foundation_row.matched_decoy_lift_ci_high),
                "current_observability_jsd_nats": float(current_row.estimate),
                "current_observability_ci_low": float(current_row.wild_ci_low),
                "current_observability_ci_high": float(current_row.wild_ci_high),
                "dms_residual_spearman_increment": float(dms_value.estimate),
                "dms_residual_ci_low": float(dms_value.wild_ci_low),
                "dms_residual_ci_high": float(dms_value.wild_ci_high),
                "current_positive_domains": int(current_row.positive_domains),
                "requires_structure_label": (
                    environment.state_kind == "core_targeted"
                    or environment.axis in {"burial", "secondary_structure", "contact_class"}
                ),
                **best,
            }
        )
    return pd.DataFrame.from_records(records)


def _best_locked_environment(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {
            "locked_status": "INCOMPLETE",
            "locked_best_route": None,
            "locked_best_jsd_nats": np.nan,
            "locked_best_ci_low": np.nan,
            "locked_best_ci_high": np.nan,
            "locked_best_residual_cosine": np.nan,
            "locked_best_max_control_jsd_nats": np.nan,
            "locked_domains": 0,
        }
    passed = rows.loc[rows["status"].eq("PASS")]
    selected = (
        (passed if not passed.empty else rows)
        .sort_values("jsd_reduction_nats", ascending=False)
        .iloc[0]
    )
    return {
        "locked_status": "PASS" if not passed.empty else str(selected.status),
        "locked_best_route": selected.route_id,
        "locked_best_jsd_nats": float(selected.jsd_reduction_nats),
        "locked_best_ci_low": float(selected.jsd_ci_low),
        "locked_best_ci_high": float(selected.jsd_ci_high),
        "locked_best_residual_cosine": float(selected.residual_cosine),
        "locked_best_max_control_jsd_nats": float(selected.maximum_control_jsd_reduction_nats),
        "locked_domains": int(selected.n_domains),
    }


def _decision_record(
    config: ObservabilityStudyConfig,
    decision: str,
    rationale: str,
    probes: pd.DataFrame,
    environments: pd.DataFrame,
) -> dict[str, Any]:
    passed = probes.loc[probes["decision_eligible"].astype(bool) & probes["status"].eq("PASS")]
    bounded_sequence_result = {
        "NEW_INDEPENDENT_SEQUENCE_ONLY_PROTOCOL_REQUIRED": (
            "SUPPORTED_BY_LOCKED_ESM2_LINEAR_REPLICATION"
        ),
        "REPRESENTATION_ACCESSIBILITY_LIMITATION": (
            "SUPPORTED_BY_LOCKED_ALTERNATE_SEQUENCE_REPRESENTATION"
        ),
        "SELECTIVE_HYBRID": "SUPPORTED_ONLY_IN_LOCKED_CANDIDATE_ENVIRONMENTS",
        "RETAIN_STRUCTURE_AS_INFERENCE_MODALITY": (
            "NOT_RECOVERED_BY_EXECUTED_OBSERVABILITY_PROBES"
        ),
    }[decision]
    return {
        **runtime_manifest(config.paths.project_root),
        "schema_version": config.schema_version,
        "observability_decision": decision,
        "rationale": rationale,
        "historical_gate_modified": False,
        "primary_registered_route": "PIVOT_SELECTIVE_STRUCTURE_CONDITIONED",
        "sequence_only_status": "NOT_LINEARLY_ACCESSIBLE_IN_CURRENT_FROZEN_ESM2_AUDIT",
        "general_sequence_learnability": "UNRESOLVED",
        "bounded_sequence_result": bounded_sequence_result,
        "on_policy_status": "NOT_ESTIMABLE",
        "biological_scope": "FIXED_SCAFFOLD_STABILITY",
        "teacher_pretraining_exposure": "NOT_IDENTIFIABLE_FROM_RELEASED_CHECKPOINTS",
        "scope": "COMPUTATIONAL_MODEL_EVALUATION_ONLY",
        "thresholds": {
            "minimum_jsd_reduction_nats": config.inference.minimum_jsd_reduction_nats,
            "minimum_residual_cosine": config.inference.minimum_residual_cosine,
            "require_positive_domain_cluster_ci_lower_bound": (
                config.inference.require_positive_ci_lower_bound
            ),
            "registered_shuffle_controls": list(CONTROL_NAMES),
            "control_repeats": config.probes.control_repeats,
        },
        "global_passing_routes": passed["route_id"].tolist(),
        "candidate_environment_passing_routes": environments.loc[
            environments["status"].eq("PASS"), "route_id"
        ].tolist(),
        "executed_global_routes": int(probes["decision_eligible"].astype(bool).sum()),
        "executed_candidate_environment_routes": int(len(environments)),
    }


def _render_report(
    config: ObservabilityStudyConfig,
    record: dict[str, Any],
    probes: pd.DataFrame,
    environments: pd.DataFrame,
) -> str:
    root = config.paths.run_dir
    domains = pd.read_parquet(root / "replication" / "registry" / "domains.parquet")
    states = pd.read_parquet(root / "replication" / "state_bank" / "states.parquet")
    skipped = pd.read_parquet(root / "replication" / "state_bank" / "skipped_states.parquet")
    scores = pd.read_parquet(root / "replication" / "teacher_cache" / "scores.parquet")
    split_counts = domains.groupby("observability_split", observed=True)["domain_id"].nunique()
    state_counts = (
        states.merge(domains[["domain_id", "observability_split"]], on="domain_id")
        .groupby("observability_split", observed=True)["state_id"]
        .nunique()
    )
    skip_counts = skipped.groupby("requested_corruption_ratio", observed=True).size()
    current_rows = _current_sensitivity_rows(config)
    dms_rows = _dms_rows(config)
    semantic = pd.read_parquet(root / "proteinmpnn" / "order_variance.parquet")
    compositional = pd.read_parquet(root / "compositional_replication" / "summary.parquet")
    probe_rows = "\n".join(_probe_markdown_rows(probes))
    environment_rows = "\n".join(_environment_markdown_rows(environments))
    current_table = "\n".join(current_rows)
    dms_table = "\n".join(dms_rows)
    clr = _one(
        compositional.loc[
            compositional["metric"].eq("jsd_reduction_nats")
            & compositional["target_coordinates"].eq("clr20")
        ],
        "replication CLR sensitivity",
    )
    ilr = _one(
        compositional.loc[
            compositional["metric"].eq("jsd_reduction_nats")
            & compositional["target_coordinates"].eq("ilr19")
        ],
        "replication ILR sensitivity",
    )
    decision = record["observability_decision"]
    train_domains = int(split_counts.get("development_train", 0))
    validation_domains = int(split_counts.get("development_validation", 0))
    test_domains = int(split_counts.get("locked_test", 0))
    train_states = int(state_counts.get("development_train", 0))
    validation_states = int(state_counts.get("development_validation", 0))
    test_states = int(state_counts.get("locked_test", 0))
    state_positions = int(scores[["state_id", "domain_id", "position"]].drop_duplicates().shape[0])
    order_difference_min = semantic["maximum_absolute_order_difference"].min()
    order_difference_max = semantic["maximum_absolute_order_difference"].max()
    order_jsd = semantic["backbone_only_jsd_nats"].mean()
    clr_interval = _interval(clr.estimate, clr.wild_ci_low, clr.wild_ci_high)
    ilr_interval = _interval(ilr.estimate, ilr.wild_ci_low, ilr.wild_ci_high)
    dms_header = (
        "| Teacher residual | Spearman increment over sequence | "
        "95% wild-cluster interval | Positive domains |"
    )
    environment_header = (
        "| Environment | Current domains | Action | Specificity | Current ΔJSD | "
        "DMS increment | Locked status | Locked best ΔJSD | Locked domains | "
        "Structure label |"
    )
    return f"""# MARGIN observability study sequence-learnability report

Decision: **`{decision}`**
Registered primary route: `PIVOT_SELECTIVE_STRUCTURE_CONDITIONED`
Historical foundation decision modified: **no**
Scope: computational fixed-scaffold stability only

{record["rationale"]}

## Registered interpretation

| Status field | Final value |
|---|---|
| Primary registered route | `PIVOT_SELECTIVE_STRUCTURE_CONDITIONED` |
| foundation audit frozen ESM2 status | `NOT_LINEARLY_ACCESSIBLE_IN_CURRENT_FROZEN_ESM2_AUDIT` |
| observability study boundary result | `{decision}` |
| Bounded sequence result | `{record["bounded_sequence_result"]}` |
| General sequence learnability | `UNRESOLVED` |
| On-policy status | `NOT_ESTIMABLE` |
| Biological scope | `FIXED_SCAFFOLD_STABILITY` |

```mermaid
flowchart TD
    accTitle: Locked observability study decision result
    accDescr: Locked replication tests Ridge, adapted probes, and frozen environments.

    foundation[Foundation decision fixed] --> ridge{{Validation-selected ESM2 Ridge passes?}}
    ridge -->|yes| newp[New independent sequence-only protocol]
    ridge -->|no| adapted{{Any nonlinear, adapted, reduced-rank, or alternate route passes?}}
    adapted -->|yes| access[Representation accessibility limitation]
    adapted -->|no| env{{Any predeclared environment passes?}}
    env -->|yes| hybrid[Selective hybrid]
    env -->|no| retain[Retain structure at inference]

    classDef decision fill:#fef9c3,stroke:#ca8a04,color:#713f12
    classDef outcome fill:#dcfce7,stroke:#16a34a,color:#14532d
    class ridge,adapted,env decision
    class newp,access,hybrid,retain outcome
```

## Locked replication population

- Domains: {len(domains):,} total; {train_domains:,} train, {validation_domains:,}
  validation, and {test_domains:,} locked test.
- States: {len(states):,} total; {train_states:,} train, {validation_states:,}
  validation, and {test_states:,} locked test.
- State-position rows: {state_positions:,}; canonical teacher-score rows: {len(scores):,}.
- Explicitly skipped core-targeted states: {len(skipped):,} ({_skip_text(skip_counts)}),
  all because the declared buried target set was too small.
- CATH-T overlaps across splits: zero. foundation audit domain/CATH-H/CATH-T overlaps: zero.
  Benchmark homology relations in the 600-domain candidate pool: zero.
- Released-teacher pretraining membership is not identifiable from the checkpoints;
  this report claims project-registry separation only.

## Existing-data sensitivity

| Target or probe | JSD reduction (nats) | 95% wild-cluster interval | Positive domains |
|---|---:|---:|---:|
{current_table}

The all-layer ESM2 validation sweep selected layer 0/query, which is the constant mask
embedding and therefore behaves as an intercept. CARP-640M did not produce a global positive
result on the ten seen external domains. CLR and ILR gave the same substantive conclusion, so
the original negative result was not a compositional-coordinate artifact.

## DMS residual audit

{dms_header}
|---|---:|---:|---:|
{dms_table}

These are residual increments, not whole-teacher correlations. They support the stability
relevance of the paired structural residual while leaving broader biological function outside
scope.

ProteinMPNN order semantics were not interchangeable: across {len(semantic)} external domains,
the maximum per-domain absolute order difference ranged from {order_difference_min:.3f} to
{order_difference_max:.3f} nats, and backbone-only order-ensemble JSD averaged {order_jsd:.4f}
nats. The replication therefore used eight shared decoding orders and averaged probabilities.

## Locked global probes

| Route | Status | ΔJSD | 95% interval | Cosine | Max control | Domains |
|---|---:|---:|---:|---:|---:|---:|
{probe_rows}

`PASS` requires ΔJSD ≥ {config.inference.minimum_jsd_reduction_nats:.2f} nats, residual cosine ≥
{config.inference.minimum_residual_cosine:.2f}, a positive domain-cluster lower bound, and an
estimate above every one of the six registered controls across
{config.probes.control_repeats} repeats. The fixed-final-layer row is retained as a non-decision
sensitivity; the validation-selected all-layer route is the registered global Ridge test.

## Predeclared candidate environments

{environment_header}
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{environment_rows}

The four environments were frozen before replication scoring. Environment success uses the
same full success rule as the global analysis; current-data values remain exploratory and
cannot substitute for locked replication.

## Coordinate and exposure limits

- Replication compositional sensitivity: CLR ΔJSD {clr_interval}; ILR ΔJSD {ilr_interval}.
- Project-level domain, topology, homology, and split leakage were excluded before locking.
- Teacher checkpoint pretraining membership remains unresolved and is not relabeled as zero leakage.
- `teacher_action_valid_radius` denotes teacher usefulness under the supplied scaffold; it is
  not evidence that edited sequences physically retain that scaffold.

## Authoritative artifacts

- [Machine-readable decision](./observability_decision.json)
- [Locked probe decisions](./locked_probe_decisions.parquet)
- [Candidate environment overview](./candidate_environment_overview.parquet)
- [Resolved run configuration](../config.resolved.yaml)
- [Historical foundation decision report](../../foundation/reports/foundation_report.md)
- [Replication lock](../replication/replication_lock.json)
- [Teacher exposure limitation](../replication/leakage/teacher_pretraining_exposure.json)
"""


def _current_sensitivity_rows(config: ObservabilityStudyConfig) -> list[str]:
    root = config.paths.run_dir
    current = pd.read_parquet(root / "current_sensitivity" / "probe_summary.parquet")
    rows = current.loc[
        current["metric"].eq("jsd_reduction_nats")
        & current["probe"].eq("final_layer_ridge")
        & current["control"].eq("observed")
    ].sort_values("target_id")
    result = [
        f"| Final ESM2 / {row.target_id} | {row.estimate:.4f} | "
        f"[{row.wild_ci_low:.4f}, {row.wild_ci_high:.4f}] | "
        f"{int(row.positive_domains)}/{int(row.n_domains)} |"
        for row in rows.itertuples(index=False)
    ]
    for directory, label in (
        ("layerwise_current", "All-layer ESM2 selected Ridge"),
        ("carp_current", "CARP-640M selected Ridge"),
    ):
        summary = pd.read_parquet(root / directory / "summary.parquet")
        row = _one(
            summary.loc[
                summary["metric"].eq("jsd_reduction_nats")
                & summary["probe"].eq("layerwise_ridge")
                & summary["control"].eq("observed")
                & summary["target_id"].eq(config.residual_targets.primary)
            ],
            label,
        )
        result.append(
            f"| {label} | {row.estimate:.4f} | "
            f"[{row.wild_ci_low:.4f}, {row.wild_ci_high:.4f}] | "
            f"{int(row.positive_domains)}/{int(row.n_domains)} |"
        )
    return result


def _dms_rows(config: ObservabilityStudyConfig) -> list[str]:
    summary = pd.read_parquet(config.paths.run_dir / "dms_residual" / "lodo_summary.parquet")
    routes = (
        ("mifst", "matched_cath", "sequence_plus_structural_residual", "MIF-ST paired"),
        ("mifst", "matched_cath", "sequence_plus_paired_decoy", "MIF-ST paired-minus-decoy"),
        ("mifst", "matched_cath", "sequence_plus_both_residuals", "MIF-ST both residuals"),
        ("esm_if1", "matched_cath", "sequence_plus_structural_residual", "ESM-IF1 paired"),
        (
            "proteinmpnn_mc8",
            "not_scored",
            "sequence_plus_structural_residual",
            "ProteinMPNN MC8 paired",
        ),
    )
    result = []
    for teacher, decoy, method, label in routes:
        row = _one(
            summary.loc[
                summary["teacher_id"].eq(teacher)
                & summary["decoy_role"].eq(decoy)
                & summary["method"].eq(method)
            ],
            label,
        )
        result.append(
            f"| {label} | {row.estimate:.4f} | "
            f"[{row.wild_ci_low:.4f}, {row.wild_ci_high:.4f}] | "
            f"{int(row.positive_domains)}/{int(row.n_domains)} |"
        )
    return result


def _probe_markdown_rows(probes: pd.DataFrame) -> list[str]:
    rows = []
    for row in probes.sort_values(
        ["decision_eligible", "model_id", "probe", "target_rank"],
        ascending=[False, True, True, True],
    ).itertuples(index=False):
        status = row.status if row.controls_complete else "INCOMPLETE"
        rows.append(
            f"| `{row.route_id}` | {status} | {row.jsd_reduction_nats:.4f} | "
            f"[{row.jsd_ci_low:.4f}, {row.jsd_ci_high:.4f}] | "
            f"{row.residual_cosine:.4f} | "
            f"{row.maximum_control_jsd_reduction_nats:.4f} | {row.n_domains} |"
        )
    return rows


def _environment_markdown_rows(environments: pd.DataFrame) -> list[str]:
    rows = []
    for row in environments.itertuples(index=False):
        label = (
            f"{row.state_kind} {row.requested_corruption_ratio:.0%}; "
            f"{row.environment_axis}={row.environment_value}"
        )
        rows.append(
            f"| {label} | {row.current_domains} | {row.action_value_nats:.3f} | "
            f"{row.specificity_nats:.3f} | {row.current_observability_jsd_nats:.4f} | "
            f"{row.dms_residual_spearman_increment:.3f} | {row.locked_status} | "
            f"{row.locked_best_jsd_nats:.4f} | {row.locked_domains} | "
            f"{'yes' if row.requires_structure_label else 'no'} |"
        )
    return rows


def _required_inputs(root: Path) -> list[Path]:
    return [
        root / "protocol_lock.json",
        root / "replication" / "replication_lock.json",
        root / "replication" / "teacher_cache" / "manifest.json",
        root / "replication" / "teacher_cache" / "scores.parquet",
        root / "replication_final_layer" / "probe_summary.parquet",
        root / "layerwise_replication" / "summary.parquet",
        root / "layerwise_replication" / "environment_summary.parquet",
        root / "carp_replication" / "summary.parquet",
        root / "carp_replication" / "environment_summary.parquet",
        root / "lora_replication" / "summary.parquet",
        root / "lora_replication" / "environment_summary.parquet",
        root / "compositional_replication" / "summary.parquet",
    ]


def _expected_controls(probe: str) -> tuple[str, ...]:
    prefix = (
        "prediction_"
        if probe
        in {
            "reduced_rank_ridge",
            "bottleneck_mlp",
            "esm2_lora",
        }
        else ""
    )
    return tuple(f"{prefix}{control}" for control in CONTROL_NAMES)


def _single_metric(group: pd.DataFrame, metric: str, control: str) -> float:
    rows = group.loc[group["metric"].eq(metric) & group["control"].eq(control)]
    if len(rows) != 1:
        return np.nan
    return float(rows.iloc[0]["estimate"])


def _matches(series: pd.Series, value: Any) -> pd.Series:
    return series.isna() if pd.isna(value) else series.eq(value)


def _route_id(
    model: str,
    probe: str,
    feature: Any,
    layer: Any,
    rank: Any,
    environment: Any,
) -> str:
    parts = [model, probe]
    if feature is not None and not pd.isna(feature):
        parts.append(str(feature))
    if layer is not None and not pd.isna(layer):
        parts.append(f"layer{int(layer)}")
    if rank is not None and not pd.isna(rank):
        parts.append(f"rank{int(rank)}")
    if environment is not None and not pd.isna(environment):
        parts.append(str(environment))
    return "/".join(parts)


def _one(frame: pd.DataFrame, label: str):
    if len(frame) != 1:
        raise ValueError(f"Expected one {label} row, found {len(frame)}")
    return next(frame.itertuples(index=False))


def _interval(estimate: Any, low: Any, high: Any) -> str:
    return f"{float(estimate):.4f} [{float(low):.4f}, {float(high):.4f}]"


def _skip_text(counts: pd.Series) -> str:
    return ", ".join(f"{float(level):.0%}: {int(count)}" for level, count in counts.items())
