"""Predeclared candidate-environment sensitivity summaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from margin.provenance import runtime_manifest, table_manifest, write_json, write_parquet
from margin.studies.observability.config import ObservabilityStudyConfig
from margin.studies.observability.current import summarize_probe_rows


def run_current_environment_audit(config: ObservabilityStudyConfig) -> dict[str, Path]:
    """Re-evaluate the four foundation audit hypotheses against every registered control."""

    source = config.paths.run_dir / "current_sensitivity" / "probe_rows.parquet"
    rows = pd.read_parquet(source)
    rows = rows.loc[rows["target_id"].eq(config.residual_targets.primary)]
    output = config.paths.run_dir / "candidate_environments_current"
    output.mkdir(parents=True, exist_ok=True)
    selected_frames = []
    coverage = []
    for environment in config.candidate_environments:
        selected = rows.loc[
            rows["state_kind"].eq(environment.state_kind)
            & rows["requested_corruption_ratio"].eq(environment.requested_corruption_ratio)
            & rows[environment.axis].eq(environment.value)
        ].copy()
        selected["environment_id"] = environment.environment_id
        selected["environment_axis"] = environment.axis
        selected["environment_value"] = environment.value
        selected_frames.append(selected)
        observed = selected.loc[
            selected["control"].eq("observed") & selected["probe"].eq("final_layer_ridge")
        ]
        coverage.append(
            {
                "environment_id": environment.environment_id,
                "state_kind": environment.state_kind,
                "requested_corruption_ratio": environment.requested_corruption_ratio,
                "environment_axis": environment.axis,
                "environment_value": environment.value,
                "observed_rows": len(observed),
                "observed_domains": observed["domain_id"].nunique(),
            }
        )
    selected_rows = pd.concat(selected_frames, ignore_index=True)
    summaries = []
    domains = []
    for environment_id, frame in selected_rows.groupby("environment_id", observed=True):
        summary, domain = summarize_probe_rows(frame, config)
        summary["environment_id"] = environment_id
        domain["environment_id"] = environment_id
        summaries.append(summary)
        domains.append(domain)
    tables = {
        "coverage": pd.DataFrame(coverage),
        "rows": selected_rows,
        "summary": pd.concat(summaries, ignore_index=True),
        "domain_estimates": pd.concat(domains, ignore_index=True),
    }
    paths = {name: output / f"{name}.parquet" for name in tables}
    for name, table in tables.items():
        write_parquet(paths[name], table)
    manifest_path = output / "manifest.json"
    write_json(
        manifest_path,
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "status": "exploratory_hypothesis_definition",
            "source": str(source),
            "artifacts": [table_manifest(paths[name], table) for name, table in tables.items()],
        },
    )
    paths["manifest"] = manifest_path
    return paths
