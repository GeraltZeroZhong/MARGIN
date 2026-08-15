#!/usr/bin/env python
"""Freeze the generalization workflow and outcome-blind inputs before inference."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from margin.provenance import read_json, runtime_manifest, sha256_file, write_json
from margin.studies.generalization.config import load_generalization_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/generalization.yaml"))
    arguments = parser.parse_args()
    config_path = arguments.config.resolve()
    config = load_generalization_config(config_path)
    project = config.paths.project_root
    sources = [
        config_path,
        config.paths.run_dir / "input_manifest.json",
        config.paths.run_dir / "architecture" / "query_rows.parquet",
        config.paths.run_dir / "architecture" / "query_manifest.json",
        config.paths.run_dir / "dms" / "assays.parquet",
        config.paths.run_dir / "dms" / "variants.parquet",
        config.paths.run_dir / "dms" / "query_rows.parquet",
        config.paths.run_dir / "dms" / "exclusions.parquet",
        config.paths.run_dir / "dms" / "observability_homology_hits.parquet",
        config.paths.run_dir / "dms" / "panel_manifest.json",
        config.paths.run_dir / "mif_requests" / "requests.parquet",
        config.paths.run_dir / "mif_requests" / "structures.parquet",
        config.paths.run_dir / "mif_requests" / "manifest.json",
        config.paths.run_dir / "mif_decoys" / "manifest.json",
    ]
    missing = [source for source in sources if not source.is_file()]
    if missing:
        raise FileNotFoundError(f"generalization study lock inputs are missing: {missing}")
    records = [
        {"source": str(source), "sha256": sha256_file(source), "bytes": source.stat().st_size}
        for source in sources
    ]
    lock_path = config.paths.run_dir / "protocol_lock.json"
    if lock_path.exists():
        locked = read_json(lock_path)
        if locked.get("artifacts") != records:
            raise ValueError(
                "generalization study protocol or input tables differ from the frozen lock"
            )
        print(f"protocol_lock={lock_path}")
        return
    snapshot = config.paths.run_dir / "frozen_protocol"
    snapshot.mkdir(parents=True, exist_ok=True)
    for source in sources[:1]:
        shutil.copy2(source, snapshot / source.name)
    write_json(
        lock_path,
        {
            **runtime_manifest(project),
            "schema_version": config.schema_version,
            "lock_status": "FROZEN_BEFORE_GENERALIZATION_MODEL_INFERENCE",
            "artifacts": records,
        },
    )
    print(f"protocol_lock={lock_path}")


if __name__ == "__main__":
    main()
