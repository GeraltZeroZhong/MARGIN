#!/usr/bin/env python
"""Run the locked label-free generalization study stability-DMS transfer audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.generalization.config import load_generalization_config
from margin.studies.generalization.dms import run_dms_transfer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/generalization.yaml"))
    arguments = parser.parse_args()
    results = run_dms_transfer(load_generalization_config(arguments.config))
    for name, value in results.items():
        print(f"{name}={value}")


if __name__ == "__main__":
    main()
