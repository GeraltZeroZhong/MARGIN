#!/usr/bin/env python
"""Snapshot the observability protocol before replication teacher scoring."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from margin.provenance import read_json, runtime_manifest, sha256_file, write_json
from margin.studies.observability.config import load_observability_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/observability.yaml"))
    arguments = parser.parse_args()
    config_path = arguments.config.resolve()
    config = load_observability_config(config_path)
    sources = [config_path, config.paths.replication_config]
    output = config.paths.run_dir / "frozen_protocol"
    output.mkdir(parents=True, exist_ok=True)
    lock_path = config.paths.run_dir / "protocol_lock.json"
    records = [
        {
            "source": str(source),
            "snapshot": str(output / source.name),
            "sha256": sha256_file(source),
        }
        for source in sources
    ]
    if lock_path.exists():
        locked = read_json(lock_path)
        if locked.get("artifacts") != records:
            raise ValueError("observability configuration differs from the pre-scoring lock")
        print(f"protocol_lock={lock_path}")
        return
    for source in sources:
        shutil.copy2(source, output / source.name)
    write_json(
        lock_path,
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "lock_status": "FROZEN_BEFORE_REPLICATION_TEACHER_SCORING",
            "artifacts": records,
        },
    )
    print(f"protocol_lock={lock_path}")


if __name__ == "__main__":
    main()
