#!/usr/bin/env python
"""Freeze the action-validation workflow before locked-panel scoring."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from margin.provenance import runtime_manifest, sha256_file, table_manifest, write_json
from margin.studies.action_validation.config import load_action_validation_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/action_validation.yaml"))
    arguments = parser.parse_args()
    config = load_action_validation_config(arguments.config)
    panel = config.paths.run_dir / "panel"
    scoring_artifacts = [
        config.paths.run_dir / "teacher_scores" / "scores.parquet",
        config.paths.storage_dir / "representations" / "carp_640M" / "manifest.json",
        config.paths.storage_dir / "representations" / "esm2_150M" / "manifest.json",
        config.paths.run_dir / "evaluation" / "project_decision.parquet",
    ]
    existing = [str(path) for path in scoring_artifacts if path.exists()]
    if existing:
        raise RuntimeError(f"cannot freeze after action-validation study panel scoring: {existing}")
    required = [
        panel / "candidate_scan.parquet",
        panel / "domains.parquet",
        panel / "variants.parquet",
        panel / "residues.parquet",
        panel / "query_rows.parquet",
        panel / "exclusions.parquet",
        panel / "prior_homology_hits.parquet",
        panel / "cross_panel_homology_hits.parquet",
        panel / "manifest.json",
        config.paths.run_dir / "teacher_requests" / "requests.parquet",
        config.paths.run_dir / "teacher_requests" / "structures.parquet",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"action-validation study protocol inputs missing: {missing}")

    domains = pd.read_parquet(panel / "domains.parquet")
    dense = domains.loc[domains["evaluation_population"].eq("megascale_dense")]
    sparse = domains.loc[domains["evaluation_population"].eq("s669_sparse_cross_platform")]
    if dense["stratum"].value_counts().to_dict() != {"natural": 16, "de_novo": 16}:
        raise RuntimeError(
            "action-validation study dense panel does not match the frozen 16/16 strata"
        )
    family_counts = dense.loc[dense["stratum"].eq("de_novo"), "design_family"].value_counts()
    expected_families = set(config.panel.de_novo_families)
    if set(family_counts.index) != expected_families or not family_counts.eq(2).all():
        raise RuntimeError(
            "action-validation study de novo family allocation is not two per family"
        )
    if len(sparse) < config.panel.s669_minimum_selected_domains:
        raise RuntimeError(
            "action-validation study lacks the frozen minimum S669 replication domains"
        )

    selected = set(domains["domain_id"].astype(str))
    for name in ("prior_homology_hits", "cross_panel_homology_hits"):
        hits = pd.read_parquet(panel / f"{name}.parquet")
        strict = hits.loc[
            hits["domain_id"].isin(selected)
            & hits["sequence_identity"].ge(config.panel.near_duplicate_identity)
            & hits[["query_coverage", "target_coverage"]]
            .min(axis=1)
            .ge(config.panel.near_duplicate_minimum_coverage)
        ]
        if not strict.empty:
            raise RuntimeError(
                f"selected action-validation study domains retain near duplicates in {name}"
            )

    frozen = config.paths.run_dir / "frozen_protocol"
    frozen.mkdir(parents=True, exist_ok=True)
    copies = []
    for source in (arguments.config.resolve(),):
        target = frozen / source.name
        shutil.copy2(source, target)
        copies.append({"path": str(target), "sha256": sha256_file(target)})
    tables = []
    for name in (
        "candidate_scan",
        "domains",
        "variants",
        "residues",
        "query_rows",
        "exclusions",
        "prior_homology_hits",
        "cross_panel_homology_hits",
    ):
        path = panel / f"{name}.parquet"
        tables.append(table_manifest(path, pd.read_parquet(path)))
    for name in ("requests", "structures"):
        path = config.paths.run_dir / "teacher_requests" / f"{name}.parquet"
        tables.append(table_manifest(path, pd.read_parquet(path)))
    lock_path = config.paths.run_dir / "protocol_lock.json"
    write_json(
        lock_path,
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "status": "FROZEN_BEFORE_ACTION_VALIDATION_PANEL_MODEL_SCORING",
            "frozen_files": copies,
            "locked_tables": tables,
            "selection_uses_outcome_magnitudes": False,
            "primary_dense_domains": int(len(dense)),
            "sparse_cross_platform_domains": int(len(sparse)),
            "immutable_counterfactuals_decision": "RETAIN_GENERALIZATION_CLOSE_COUNTERFACTUALS",
            "counterfactual_search_closed": True,
            "registered_route": "PIVOT_SELECTIVE_STRUCTURE_CONDITIONED",
            "currently_supported_implementation": "CALIBRATED_PAIRED_STRUCTURE_CONDITIONED",
            "selective_routing": "NOT_YET_ESTABLISHED",
            "selective_routing_authorized": False,
        },
    )
    print(f"locked={lock_path}")


if __name__ == "__main__":
    main()
