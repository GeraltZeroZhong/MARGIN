#!/usr/bin/env python
"""Run the generalization study environment-label deployability audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.generalization.config import load_generalization_config
from margin.studies.generalization.environments import run_environment_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/generalization.yaml"))
    arguments = parser.parse_args()
    for name, value in run_environment_audit(load_generalization_config(arguments.config)).items():
        print(f"{name}={value}")


if __name__ == "__main__":
    main()
