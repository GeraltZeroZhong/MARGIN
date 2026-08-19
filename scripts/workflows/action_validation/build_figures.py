#!/usr/bin/env python
"""Build action-validation study publication figures and source-data tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.action_validation.config import load_action_validation_config
from margin.studies.action_validation.plots import build_action_validation_figures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/action_validation.yaml"))
    arguments = parser.parse_args()
    paths = build_action_validation_figures(load_action_validation_config(arguments.config))
    for name, path in paths.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
