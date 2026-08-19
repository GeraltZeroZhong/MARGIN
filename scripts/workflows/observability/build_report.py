#!/usr/bin/env python
"""Build the machine-readable decision and source-linked observability study report."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.observability.config import load_observability_config
from margin.studies.observability.report import build_observability_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/observability.yaml"))
    arguments = parser.parse_args()
    artifacts = build_observability_report(load_observability_config(arguments.config))
    for name, path in artifacts.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
