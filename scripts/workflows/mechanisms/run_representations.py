#!/usr/bin/env python
"""Export frozen CARP features and ESM2 scores for mechanism study."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from margin.provenance import read_json, runtime_manifest, sha256_file, write_json
from margin.studies.mechanisms.config import load_mechanism_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/mechanisms.yaml"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", choices=["all", "carp_640M", "esm2_150M"], default="all")
    arguments = parser.parse_args()
    config = load_mechanism_config(arguments.config)
    lock = read_json(config.paths.run_dir / "protocol_lock.json")
    if lock.get("status") != "FROZEN_BEFORE_MECHANISMS_MODEL_SCORING":
        raise RuntimeError("mechanism study protocol lock is missing")
    queries = config.paths.run_dir / "panel" / "query_rows.parquet"
    output = config.paths.storage_dir / "representations"
    output.mkdir(parents=True, exist_ok=True)
    requested = ["carp_640M", "esm2_150M"] if arguments.model == "all" else [arguments.model]
    environment = os.environ.copy()
    project_src = str(config.paths.project_root / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        [project_src, environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    for model_id in requested:
        command = [
            sys.executable,
            str(
                config.paths.project_root
                / "scripts/workflows/generalization/export_representations.py"
            ),
            "--config",
            str(config.paths.generalization_config),
            "--model-id",
            model_id,
            "--queries",
            str(queries),
            "--output",
            str(output / model_id),
            "--device",
            arguments.device,
            "--force-inference",
        ]
        if model_id == "esm2_150M":
            command.append("--save-logp")
        subprocess.run(command, cwd=config.paths.project_root, env=environment, check=True)
    manifests = {}
    for model_id in ("carp_640M", "esm2_150M"):
        path = output / model_id / "manifest.json"
        if path.exists():
            manifests[model_id] = {"path": str(path), "sha256": sha256_file(path)}
    if set(manifests) == {"carp_640M", "esm2_150M"}:
        write_json(
            config.paths.run_dir / "representations_manifest.json",
            {
                **runtime_manifest(config.paths.project_root),
                "schema_version": config.schema_version,
                "conditioning": "strict_leave_one_position_out",
                "locked_panel_outcomes_used_for_training": False,
                "queries": {"path": str(queries), "sha256": sha256_file(queries)},
                "models": manifests,
            },
        )
    print(f"complete_models={','.join(sorted(manifests))}")


if __name__ == "__main__":
    main()
