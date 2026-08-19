#!/usr/bin/env python
"""Prepare all outcome-blind generalization study inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.generalization.config import load_generalization_config
from margin.studies.generalization.prepare import prepare_generalization_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/generalization.yaml"))
    arguments = parser.parse_args()
    paths = prepare_generalization_inputs(load_generalization_config(arguments.config))
    for name, path in paths.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
