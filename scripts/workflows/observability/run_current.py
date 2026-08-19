#!/usr/bin/env python
"""Run the frozen-artifact observability study sensitivity audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.observability.config import load_observability_config
from margin.studies.observability.current import run_current_sensitivity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/observability.yaml"))
    arguments = parser.parse_args()
    artifacts = run_current_sensitivity(load_observability_config(arguments.config))
    for name, path in artifacts.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
