"""Build the generalization study decision record and completion report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from margin.provenance import runtime_manifest, write_json, write_text
from margin.studies.generalization.config import GeneralizationStudyConfig


def build_generalization_report(config: GeneralizationStudyConfig) -> dict[str, Path]:
    """Assemble all locked results into one explicit project decision."""

    cath_root = config.paths.run_dir / "cath_audit"
    dms_root = config.paths.run_dir / "dms_transfer"
    environment_root = config.paths.run_dir / "environment_audit"
    architecture = pd.read_parquet(cath_root / "architecture_decisions.parquet")
    lineage = pd.read_parquet(cath_root / "lineage_summary.parquet")
    dms = pd.read_parquet(dms_root / "decision.parquet").iloc[0]
    dms_summary = pd.read_parquet(dms_root / "increment_summary.parquet")
    routes = pd.read_parquet(environment_root / "route_summary.parquet")
    route_deltas = pd.read_parquet(environment_root / "route_delta_summary.parquet")
    carp = architecture.loc[architecture["model_id"].eq("carp_640M")].iloc[0]
    lineage_pass = bool(carp["passed"])
    dms_pass = bool(dms["passed"])
    advance = lineage_pass and dms_pass
    decision = (
        "ADVANCE_OFFLINE_CARP_RESIDUAL_DISTILLATION_SELECTIVE_ROUTING"
        if advance
        else "RETAIN_CARP_AS_REPRESENTATIONAL_AUDIT_ONLY"
    )
    status = {
        "PRIMARY_ROUTE": "PIVOT_SELECTIVE_STRUCTURE_CONDITIONED",
        "SECONDARY_ROUTE": decision,
        "ACCESSIBILITY_RESULT": "MODEL_FAMILY_AND_SCALE_MATRIX_COMPLETED",
        "ESM2_RESULT": _model_result(architecture, "esm2_150M", "esm2_650M"),
        "CARP_RESULT": _model_result(architecture, "carp_76M", "carp_640M"),
        "SEQUENCE_PREDICTED_DMS_UTILITY": (
            "ESTABLISHED_ON_LOCKED_STABILITY_PANEL" if dms_pass else "NOT_ESTABLISHED"
        ),
        "ON_POLICY": "NOT_ESTIMABLE",
        "BIOLOGICAL_SCOPE": "FIXED_SCAFFOLD_STABILITY",
    }
    output = config.paths.run_dir / "reports"
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "generalization_report.md"
    decision_path = output / "generalization_decision.json"
    report = _report_markdown(
        decision,
        status,
        architecture,
        lineage,
        dms,
        dms_summary,
        routes,
        route_deltas,
    )
    write_text(report_path, report)
    write_json(
        decision_path,
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "decision": decision,
            "lineage_independent_cath_pass": lineage_pass,
            "control_unique_margin_pass": bool(carp["pass_control_margin"]),
            "locked_external_dms_transfer_pass": dms_pass,
            "foundation_decision_modified": False,
            "status": status,
        },
    )
    completion = config.paths.project_root / "reports" / "generalization_completion.md"
    write_text(
        completion,
        _completion_markdown(decision, status, report_path, decision_path),
    )
    return {"report": report_path, "decision": decision_path, "completion": completion}


def _model_result(frame: pd.DataFrame, *model_ids: str) -> str:
    selected = frame.loc[frame["model_id"].isin(model_ids)]
    passed = selected.loc[selected["passed"], "model_id"].tolist()
    return "PASS:" + ",".join(passed) if passed else "NO_REGISTERED_MODEL_PASSED"


def _report_markdown(
    decision: str,
    status: dict[str, str],
    architecture: pd.DataFrame,
    lineage: pd.DataFrame,
    dms: pd.Series,
    dms_summary: pd.DataFrame,
    routes: pd.DataFrame,
    route_deltas: pd.DataFrame,
) -> str:
    architecture_rows = _architecture_table_rows(architecture)
    lineage_jsd = lineage.loc[
        lineage["metric"].eq("jsd_reduction_nats") & lineage["control"].eq("observed")
    ].sort_values("target_id")
    lineage_rows = "\n".join(
        f"| **{row.target_id}** | {row.estimate:.4f} | "
        f"[{row.wild_ci_low:.4f}, {row.wild_ci_high:.4f}] | "
        f"{int(row.positive_domains)}/{int(row.n_domains)} |"
        for row in lineage_jsd.itertuples(index=False)
    )
    primary_dms = dms_summary.loc[dms_summary["method"].eq("primary_hybrid")]
    dms_rows = "\n".join(
        f"| **{row.metric}** | {row.estimate:.4f} | "
        f"[{row.wild_ci_low:.4f}, {row.wild_ci_high:.4f}] |"
        for row in primary_dms.itertuples(index=False)
    )
    dms_comparison_rows = _dms_comparison_table_rows(dms_summary)
    route_jsd = routes.loc[routes["metric"].eq("jsd_reduction_nats")].sort_values(
        "environment_route"
    )
    route_rows = "\n".join(
        f"| **{row.environment_route}** | {row.estimate:.4f} | "
        f"[{row.wild_ci_low:.4f}, {row.wild_ci_high:.4f}] |"
        for row in route_jsd.itertuples(index=False)
    )
    delta_rows = "\n".join(
        f"| **{row.environment_route}** | {row.estimate:.4f} | "
        f"[{row.wild_ci_low:.4f}, {row.wild_ci_high:.4f}] |"
        for row in route_deltas.sort_values("environment_route").itertuples(index=False)
    )
    status_lines = "\n".join(f"- `{key} {value}`" for key, value in status.items())
    carp = architecture.loc[architecture["model_id"].eq("carp_640M")].iloc[0]
    cath_pass = bool(carp["pass_jsd"] and carp["pass_cosine"])
    control_pass = bool(carp["pass_control_margin"])
    dms_pass = bool(dms["passed"])
    cath_label = (
        f"{'' if cath_pass else ''} CATH accessibility {'passed' if cath_pass else 'failed'}"
    )
    control_label = (
        f"{'' if control_pass else ''} CARP margin {'passed' if control_pass else 'failed'}"
    )
    dms_label = f"{'' if dms_pass else ''} DMS transfer {'passed' if dms_pass else 'failed'}"
    accessible_description = (
        "Three registered generalization study gates feed a joint decision; the observed gate states "
        "determine whether CARP advances to offline sequence-control branch or remains audit-only"
    )
    return f"""# generalization study residual transfer and confound audit

_Final locked analysis · generated from frozen generalization study artifacts_

---

## Outcome

The registered decision is `{decision}`.

{status_lines}

```mermaid
flowchart LR
    accTitle: generalization study observed decision
    accDescr: {accessible_description}

    cath_gate[{cath_label}] --> joint_gate{{ All gates pass?}}
    control_gate[{control_label}] --> joint_gate
    dms_gate[{dms_label}] --> joint_gate
    joint_gate -->|Yes| advance([ Advance offline sequence-control branch])
    joint_gate -->|No| retain([ Retain CARP as audit-only])
```

The historical foundation decision is unchanged. The previously locked 48-domain CATH test split was
reused after the observability study decision and is therefore supporting evidence, not a new independent
test. The decisive independent evidence is the pre-excluded 52-assay stability DMS panel.

## Teacher-lineage transfer

CARP-640M rank-16 prediction was evaluated against teacher-specific and consensus targets.

| Target | Mean ΔJSD | 95% domain CI | Positive domains |
| --- | ---: | ---: | ---: |
{lineage_rows}

The leave-MIF-ST-out target directly tests whether the observability study result survives removal of the
CARP-containing MIF-ST teacher. The plain-MIF paired-minus-rewired target is a separate
structure-specific diagnostic.

## Architecture and control audit

All models used the same native query rows, strict masking, rank-16 probe, target, split, and
training-target shuffle controls.

| Model | Family | Mean ΔJSD | Control margin | Decision |
| --- | --- | ---: | ---: | --- |
{architecture_rows}

The control margin is paired by domain against the globally strongest registered control-repeat.
Model size and family effects should be read from this matrix rather than inferred from the
initial CARP-versus-ESM2 comparison alone.

## Locked external DMS transfer

The DMS panel was frozen before model inference from ProteinGym v1.3 Tsuboyama stability
assays.[^1] No DMS labels entered CATH predictor training or alpha selection.

| Primary hybrid metric | Mean increment | 95% assay CI |
| --- | ---: | ---: |
{dms_rows}

- Mean Spearman increment: `{float(dms["mean_spearman_increment"]):.4f}`
- Positive-assay fraction: `{float(dms["positive_assay_fraction"]):.3f}`
- Mean NDCG increment: `{float(dms["mean_ndcg_increment"]):.4f}`
- Registered DMS gate: `{"PASS" if bool(dms["passed"]) else "FAIL"}`

| Route | Alpha | Mean ΔSpearman | 95% assay CI | Mean ΔNDCG |
| --- | ---: | ---: | ---: | ---: |
{dms_comparison_rows}

The paired-minus-rewired route is a predeclared supportive diagnostic. Its positive transfer
does not replace the frozen primary `alpha=1.0` result, and the alpha sensitivities were not used
to revise the decision.

## Environment deployability

| Route | Mean ΔJSD | 95% domain CI |
| --- | ---: | ---: |
{route_rows}

| Route versus no labels | Mean ΔΔJSD | 95% domain CI |
| --- | ---: | ---: |
{delta_rows}

True structural labels are an oracle route. MSA conservation and sequence-predicted environment
features are deployable sequence-side alternatives. The primary external DMS score uses no
environment labels, so this audit explains routing but cannot rescue a failed DMS gate.

## Interpretation and limits

- The result concerns offline residual reconstruction and fixed-scaffold stability only
- It does not establish on-policy utility, general function prediction, or a historical foundation decision
  reversal
- CATH results reuse the observability study test population after its opening
- The 52-assay DMS panel is independent of the ten historical assays and excludes exact or
  registered-threshold observability study sequence overlap

## Artifacts and references

- [Resolved run configuration](../config.resolved.yaml)
- [Runtime implementation errata](../protocol_errata.md)
- [Architecture decisions](../cath_audit/architecture_decisions.parquet)
- [DMS decision](../dms_transfer/decision.parquet)
- [Environment summary](../environment_audit/route_summary.parquet)
- [Machine-readable decision](generalization_decision.json)

[^1]: ProteinGym. (2024). "ProteinGym benchmark resources." https://proteingym.org/
"""


def _architecture_table_rows(architecture: pd.DataFrame) -> str:
    return "\n".join(
        f"| **{row.model_id}** | {row.family} | {row.jsd_reduction_nats:.4f} | "
        f"{row.control_unique_margin_nats:.4f} | "
        f"{'PASS' if row.passed else 'FAIL'} |"
        for row in architecture.itertuples(index=False)
    )


def _dms_comparison_table_rows(summary: pd.DataFrame) -> str:
    labels = {
        "alpha_0.25": "Alpha sensitivity",
        "alpha_0.5": "Alpha sensitivity",
        "primary_hybrid": "Primary hybrid",
        "alpha_2": "Alpha sensitivity",
        "paired_decoy_hybrid": "Paired-minus-rewired diagnostic",
    }
    order = list(labels)
    spearman = summary.loc[summary["metric"].eq("spearman_increment")].set_index("method")
    ndcg = summary.loc[summary["metric"].eq("ndcg_increment")].set_index("method")
    return "\n".join(
        f"| **{labels[method]}** | {float(spearman.loc[method, 'alpha']):g} | "
        f"{float(spearman.loc[method, 'estimate']):.4f} | "
        f"[{float(spearman.loc[method, 'wild_ci_low']):.4f}, "
        f"{float(spearman.loc[method, 'wild_ci_high']):.4f}] | "
        f"{float(ndcg.loc[method, 'estimate']):.4f} |"
        for method in order
    )


def _completion_markdown(
    decision: str,
    status: dict[str, str],
    report_path: Path,
    decision_path: Path,
) -> str:
    status_lines = "\n".join(f"- `{key} {value}`" for key, value in status.items())
    return f"""# generalization study completion

_MARGIN residual transfer and confound audit_

---

## Final decision

`{decision}`

{status_lines}

## Deliverables

- [Full generalization study report](../{report_path.relative_to(report_path.parents[3])})
- [Machine-readable decision](../{decision_path.relative_to(decision_path.parents[3])})
- [Resolved run configuration](../runs/generalization/config.resolved.yaml)
- [Runtime implementation errata](../runs/generalization/protocol_errata.md)

## Scope

generalization study is complete for computational fixed-scaffold stability evaluation. Historical foundation decision and
the selective structure-conditioned primary route remain unchanged; on-policy utility remains
not estimable.
"""
