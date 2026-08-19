#!/usr/bin/env python
"""Build stability study publication figures and source-data CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.stability.config import load_stability_config
from margin.studies.stability.plots import build_stability_figures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/stability.yaml"))
    arguments = parser.parse_args()
    paths = build_stability_figures(load_stability_config(arguments.config))
    for name, path in paths.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
