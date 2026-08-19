#!/usr/bin/env python
"""Audit the four predeclared observability study candidate environments."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.observability.config import load_observability_config
from margin.studies.observability.environments import run_current_environment_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/observability.yaml"))
    arguments = parser.parse_args()
    for name, path in run_current_environment_audit(
        load_observability_config(arguments.config)
    ).items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
