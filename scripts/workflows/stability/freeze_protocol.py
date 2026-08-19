#!/usr/bin/env python
"""Freeze the calibrated paired-action workflow before panel scoring."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from margin.provenance import read_json, runtime_manifest, write_json
from margin.studies.stability.config import load_stability_config
from margin.studies.stability.prepare import EXTERNAL_POPULATION, PRIMARY_POPULATION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/stability.yaml"))
    arguments = parser.parse_args()
    config = load_stability_config(arguments.config)
    calibration = config.paths.run_dir / "calibration" / "selection.json"
    panel = config.paths.run_dir / "panel"
    scoring_artifacts = [
        config.paths.run_dir / "teacher_scores" / "scores.parquet",
        config.paths.storage_dir / "representations" / "esm2_150M" / "manifest.json",
        config.paths.run_dir / "evaluation" / "project_decision.parquet",
    ]
    existing = [str(path) for path in scoring_artifacts if path.exists()]
    if existing:
        raise RuntimeError(f"cannot freeze after stability study panel scoring: {existing}")
    required = [
        calibration,
        panel / "candidate_scan.parquet",
        panel / "domains.parquet",
        panel / "variants.parquet",
        panel / "residues.parquet",
        panel / "query_rows.parquet",
        panel / "exclusions.parquet",
        panel / "prior_homology_hits.parquet",
        panel / "within_pool_homology_hits.parquet",
        panel / "cross_platform_homology_hits.parquet",
        panel / "manifest.json",
        config.paths.run_dir / "teacher_requests" / "requests.parquet",
        config.paths.run_dir / "teacher_requests" / "structures.parquet",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"stability study protocol inputs are missing: {missing}")

    domains = pd.read_parquet(panel / "domains.parquet")
    variants = pd.read_parquet(panel / "variants.parquet")
    primary = domains.loc[domains["evaluation_population"].eq(PRIMARY_POPULATION)]
    external = domains.loc[domains["evaluation_population"].eq(EXTERNAL_POPULATION)]
    if primary["stratum"].value_counts().to_dict() != {"natural": 16, "de_novo": 16}:
        raise RuntimeError("stability study primary panel does not match the frozen 16/16 strata")
    family_counts = primary.loc[primary["stratum"].eq("de_novo"), "design_family"].value_counts()
    if (
        set(family_counts.index) != set(config.panel.de_novo_families)
        or not family_counts.eq(config.panel.de_novo_domains_per_family).all()
    ):
        raise RuntimeError("stability study de novo family allocation is not two per family")
    if len(external) != 1 or str(external.iloc[0]["wt_name"]) != config.panel.external_assay_id:
        raise RuntimeError("stability study external assay does not match the frozen identity")
    external_variants = variants.loc[variants["evaluation_population"].eq(EXTERNAL_POPULATION)]
    if len(external_variants) != 2172 or external_variants["position"].nunique() != 168:
        raise RuntimeError("stability study external assay coverage changed before freezing")

    selected = set(domains["domain_id"].astype(str))
    prior = pd.read_parquet(panel / "prior_homology_hits.parquet")
    cross = pd.read_parquet(panel / "cross_platform_homology_hits.parquet")
    within = pd.read_parquet(panel / "within_pool_homology_hits.parquet")
    for name, hits, require_selected_target in (
        ("prior", prior, False),
        ("cross_platform", cross, False),
        ("within_panel", within, True),
    ):
        mask = (
            hits["domain_id"].isin(selected)
            & hits["sequence_identity"].ge(config.panel.near_duplicate_identity)
            & hits[["query_coverage", "target_coverage"]]
            .min(axis=1)
            .ge(config.panel.near_duplicate_minimum_coverage)
            & hits["domain_id"].ne(hits["target_id"])
        )
        if require_selected_target:
            mask &= hits["target_id"].isin(selected)
        if mask.any():
            raise RuntimeError(
                f"selected stability study identities retain near duplicates in {name}"
            )

    selected_calibration = read_json(calibration)
    if selected_calibration["selected_scheme"] != "joint_temperature_native_nll":
        raise RuntimeError("frozen protocol and outcome-free calibration selection disagree")
    frozen = config.paths.run_dir / "frozen_protocol"
    frozen.mkdir(parents=True, exist_ok=True)
    copied = []
    for source in (arguments.config.resolve(), calibration):
        target = frozen / source.name
        shutil.copy2(source, target)
        copied.append(str(target))
    table_names = [
        "candidate_scan",
        "domains",
        "variants",
        "residues",
        "query_rows",
        "exclusions",
        "prior_homology_hits",
        "within_pool_homology_hits",
        "cross_platform_homology_hits",
    ]
    tables = []
    for name in table_names:
        path = panel / f"{name}.parquet"
        table = pd.read_parquet(path)
        tables.append({"path": str(path), "rows": len(table), "columns": list(table.columns)})
    for name in ("requests", "structures"):
        path = config.paths.run_dir / "teacher_requests" / f"{name}.parquet"
        table = pd.read_parquet(path)
        tables.append({"path": str(path), "rows": len(table), "columns": list(table.columns)})
    lock_path = config.paths.run_dir / "protocol_lock.json"
    write_json(
        lock_path,
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "status": "FROZEN_BEFORE_STABILITY_PANEL_MODEL_SCORING",
            "frozen_files": copied,
            "locked_tables": tables,
            "selection_uses_outcome_magnitudes": False,
            "calibration_stability_labels_used": False,
            "selected_calibration": selected_calibration["selected_scheme"],
            "final_calibration_parameters": selected_calibration["final_parameters"],
            "primary_domains": len(primary),
            "external_domains": len(external),
            "registered_route": "PIVOT_SELECTIVE_STRUCTURE_CONDITIONED",
            "current_supported_model": "CALIBRATED_PAIRED_STRUCTURE_CONDITIONED",
            "selective_routing": "NOT_ESTABLISHED",
            "sequence_only_residual_transfer": "CLOSED",
            "counterfactual_subtraction": "CLOSED",
            "structure_sensitivity": "DEFERRED_SEPARATE_PROTOCOL",
        },
    )
    print(f"locked={lock_path}")


if __name__ == "__main__":
    main()
