#!/usr/bin/env python
"""Freeze the mechanism workflow before any audit-panel model scoring."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from margin.provenance import runtime_manifest, sha256_file, table_manifest, write_json
from margin.studies.mechanisms.config import load_mechanism_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/mechanisms.yaml"))
    arguments = parser.parse_args()
    config = load_mechanism_config(arguments.config)
    panel = config.paths.run_dir / "panel"
    scoring_artifacts = [
        config.paths.run_dir / "mif" / "scores.parquet",
        config.paths.storage_dir / "representations" / "carp_640M" / "manifest.json",
        config.paths.storage_dir / "representations" / "esm2_150M" / "manifest.json",
    ]
    existing = [str(path) for path in scoring_artifacts if path.exists()]
    if existing:
        raise RuntimeError(f"cannot freeze after mechanism study model scoring: {existing}")
    required = [
        panel / "domains.parquet",
        panel / "variants.parquet",
        panel / "residues.parquet",
        panel / "query_rows.parquet",
        panel / "matched_real_decoys.parquet",
        panel / "manifest.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"mechanism study protocol inputs missing: {missing}")
    domains = pd.read_parquet(panel / "domains.parquet")
    if domains["stratum"].value_counts().to_dict() != {"de_novo": 16, "natural": 16}:
        raise RuntimeError("mechanism study panel does not match the frozen 16/16 strata")

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
        "donor_pool",
        "matched_real_decoys",
    ):
        path = panel / f"{name}.parquet"
        table = pd.read_parquet(path)
        tables.append(table_manifest(path, table))
    lock_path = config.paths.run_dir / "protocol_lock.json"
    write_json(
        lock_path,
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "status": "FROZEN_BEFORE_MECHANISMS_MODEL_SCORING",
            "frozen_files": copies,
            "locked_tables": tables,
            "selection_uses_outcome_magnitudes": False,
            "immutable_counterfactuals_decision": "RETAIN_GENERALIZATION_CLOSE_COUNTERFACTUALS",
            "selective_routing_authorized": False,
        },
    )
    print(f"locked={lock_path}")


if __name__ == "__main__":
    main()
