#!/usr/bin/env python
"""Evaluate position specificity of U+ under the final strong sequence control."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.stability.config import load_stability_config
from margin.studies.stability.position_specificity import (
    DEFAULT_POPULATION,
    run_position_specificity_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/stability.yaml"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--component-matrices", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--population", default=DEFAULT_POPULATION)
    arguments = parser.parse_args()
    config = load_stability_config(arguments.config)
    outputs = run_position_specificity_audit(
        config,
        run_dir=arguments.run_dir,
        component_matrices=arguments.component_matrices,
        output_dir=arguments.output_dir,
        population=arguments.population,
    )
    for name, path in outputs.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
