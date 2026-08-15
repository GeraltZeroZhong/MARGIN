#!/usr/bin/env python
"""Export frozen CARP context features and ESM2 actions for action-validation study."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from margin.provenance import read_json, runtime_manifest, write_json
from margin.studies.action_validation.config import load_action_validation_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/action_validation.yaml"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", choices=["all", "carp_640M", "esm2_150M"], default="all")
    arguments = parser.parse_args()
    config = load_action_validation_config(arguments.config)
    lock = read_json(config.paths.run_dir / "protocol_lock.json")
    if lock.get("status") != "FROZEN_BEFORE_ACTION_VALIDATION_PANEL_MODEL_SCORING":
        raise RuntimeError("action-validation study protocol lock is missing")
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
    manifests = {
        model_id: str(output / model_id / "manifest.json")
        for model_id in ("carp_640M", "esm2_150M")
        if (output / model_id / "manifest.json").exists()
    }
    if set(manifests) == {"carp_640M", "esm2_150M"}:
        write_json(
            config.paths.run_dir / "representations_manifest.json",
            {
                **runtime_manifest(config.paths.project_root),
                "schema_version": config.schema_version,
                "conditioning": "strict_leave_one_position_out",
                "panel_stability_labels_used": False,
                "queries": str(queries),
                "models": manifests,
            },
        )
    print(f"complete_models={','.join(sorted(manifests))}")


if __name__ == "__main__":
    main()
