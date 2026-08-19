#!/usr/bin/env python
"""Compare CLR and orthonormal ILR residual coordinates."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.observability.compositional import run_compositional_audit
from margin.studies.observability.config import load_observability_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/observability.yaml"))
    parser.add_argument("--dataset", choices=["current", "replication"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    for name, path in run_compositional_audit(
        load_observability_config(arguments.config), arguments.dataset, arguments.output
    ).items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
