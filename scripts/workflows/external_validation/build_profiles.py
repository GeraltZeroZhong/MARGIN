#!/usr/bin/env python
"""Build CATH-homolog profiles for the frozen cross-platform panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from margin.provenance import read_json, runtime_manifest, write_json, write_parquet
from margin.studies.external_validation.panel import load_external_validation_config
from margin.studies.stability.config import load_stability_config
from margin.studies.stability.profiles import _read_alignments, _search_panel, profile_queries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/external_validation.yaml"),
    )
    arguments = parser.parse_args()
    config = load_external_validation_config(arguments.protocol)
    lock = read_json(config.paths.run_dir / "protocol_lock.json")
    if lock.get("status") != config.status:
        raise RuntimeError("cross-platform protocol lock is missing")
    stability = load_stability_config(config.paths.stability_config)
    queries = pd.read_parquet(config.paths.run_dir / "panel/query_rows.parquet")
    output = config.paths.run_dir / "strong_control"
    output.mkdir(parents=True, exist_ok=True)
    alignment_path = output / "panel_alignments.tsv"
    profile_path = output / "panel_profiles.parquet"
    if not alignment_path.exists():
        _search_panel(queries, alignment_path, stability)
    profiles = profile_queries(queries, _read_alignments(alignment_path), stability)
    write_parquet(profile_path, profiles)
    write_json(
        output / "profile_manifest.json",
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "stability_labels_read": False,
            "queries": len(queries),
            "profile_rows": len(profiles),
            "accepted_hit_domain_count": int(
                profiles.loc[profiles["accepted_homolog_hits"].gt(0), "domain_id"].nunique()
            ),
            "profile_path": str(profile_path),
        },
    )
    print(f"profiles={profile_path} rows={len(profiles)}")


if __name__ == "__main__":
    main()
