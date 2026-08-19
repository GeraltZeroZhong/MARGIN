#!/usr/bin/env python
"""Build mechanism study publication figures and source data."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.mechanisms.config import load_mechanism_config
from margin.studies.mechanisms.plots import build_mechanism_figures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/mechanisms.yaml"))
    arguments = parser.parse_args()
    artifacts = build_mechanism_figures(load_mechanism_config(arguments.config))
    for name, path in artifacts.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
