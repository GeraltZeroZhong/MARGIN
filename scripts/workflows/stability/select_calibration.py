#!/usr/bin/env python
"""Select and refit stability study teacher calibration without stability labels."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.provenance import runtime_manifest, write_json, write_parquet
from margin.studies.stability.calibration import select_calibration
from margin.studies.stability.config import load_stability_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/stability.yaml"))
    arguments = parser.parse_args()
    config = load_stability_config(arguments.config)
    output = config.paths.run_dir / "calibration"
    output.mkdir(parents=True, exist_ok=True)
    result = select_calibration(config)
    validation_path = output / "scheme_validation.parquet"
    audit_path = output / "selected_scheme_audit.parquet"
    write_parquet(validation_path, result["validation"])
    write_parquet(audit_path, result["audit"])
    selection_path = output / "selection.json"
    write_json(
        selection_path,
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "status": "SELECTED_WITH_CATH_NATIVE_RESIDUES_ONLY",
            "stability_labels_used": False,
            "selection_metric": config.calibration.selection_metric,
            "selected_scheme": result["selected_scheme"],
            "training_parameters": result["training_parameters"],
            "final_parameters": result["final_parameters"],
            "validation": {
                "path": str(validation_path),
                "rows": len(result["validation"]),
                "columns": list(result["validation"].columns),
            },
            "audit": {
                "path": str(audit_path),
                "rows": len(result["audit"]),
                "columns": list(result["audit"].columns),
            },
        },
    )
    print(f"selected_scheme={result['selected_scheme']}")
    print(result["validation"].to_string(index=False))
    print(f"selection={selection_path}")


if __name__ == "__main__":
    main()
