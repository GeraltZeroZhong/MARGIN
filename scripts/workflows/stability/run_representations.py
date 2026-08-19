#!/usr/bin/env python
"""Export locked sequence features and logits for paired-action and sequence-control branches."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from margin.provenance import read_json, runtime_manifest, write_json
from margin.studies.stability.config import load_stability_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/stability.yaml"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--model",
        choices=["all", "esm2_150M", "carp_640M", "esm2_650M", "esm1b_650M"],
        default="all",
    )
    arguments = parser.parse_args()
    config = load_stability_config(arguments.config)
    lock = read_json(config.paths.run_dir / "protocol_lock.json")
    if lock.get("status") != "FROZEN_BEFORE_STABILITY_PANEL_MODEL_SCORING":
        raise RuntimeError("stability study protocol lock is missing")
    queries = config.paths.run_dir / "panel" / "query_rows.parquet"
    output = config.paths.storage_dir / "representations"
    output.mkdir(parents=True, exist_ok=True)
    model_ids = ["esm2_150M", "carp_640M", "esm2_650M", "esm1b_650M"]
    requested = model_ids if arguments.model == "all" else [arguments.model]
    environment = os.environ.copy()
    project_src = str(config.paths.project_root / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        [project_src, environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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
        if model_id == "esm1b_650M":
            command.extend(
                [
                    "--loader",
                    "hf_esm",
                    "--checkpoint",
                    str(config.paths.esm1b_hf_checkpoint),
                ]
            )
        subprocess.run(
            command,
            cwd=config.paths.project_root,
            env=environment,
            check=True,
        )
    manifests = {
        model_id: str(output / model_id / "manifest.json")
        for model_id in model_ids
        if (output / model_id / "manifest.json").exists()
    }
    if set(manifests) == set(model_ids):
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
