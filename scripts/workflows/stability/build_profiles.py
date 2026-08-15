#!/usr/bin/env python
"""Build homolog profiles for the frozen strengthened sequence control."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.provenance import read_json, runtime_manifest, write_json
from margin.studies.stability.config import load_stability_config
from margin.studies.stability.profiles import build_profiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/stability.yaml"))
    arguments = parser.parse_args()
    config = load_stability_config(arguments.config)
    lock = read_json(config.paths.run_dir / "protocol_lock.json")
    if lock.get("status") != "FROZEN_BEFORE_STABILITY_PANEL_MODEL_SCORING":
        raise RuntimeError("stability study protocol lock is missing")
    profiles = build_profiles(config)
    output = config.paths.run_dir / "strong_control"
    write_json(
        output / "profiles_manifest.json",
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "stability_labels_used": False,
            "database": str(config.paths.cath_fasta),
            "filters": {
                "minimum_identity": config.strong_control.profile_minimum_identity,
                "maximum_identity": config.strong_control.profile_maximum_identity,
                "minimum_query_coverage": (config.strong_control.profile_minimum_query_coverage),
            },
            "tables": {
                name: {
                    "path": str(output / f"{name}_profiles.parquet"),
                    "rows": len(table),
                    "columns": list(table.columns),
                }
                for name, table in profiles.items()
            },
        },
    )
    for name, table in profiles.items():
        print(
            f"{name}_rows={len(table)} covered={table['profile_covered'].mean():.4f} "
            f"median_observations={table['homolog_observations'].median():.1f}"
        )


if __name__ == "__main__":
    main()
