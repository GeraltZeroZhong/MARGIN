#!/usr/bin/env python
"""Build the complete counterfactual study Markdown report."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.counterfactuals.config import load_counterfactual_config
from margin.studies.counterfactuals.report import build_counterfactual_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/counterfactuals.yaml"))
    arguments = parser.parse_args()
    paths = build_counterfactual_report(load_counterfactual_config(arguments.config))
    for name, path in paths.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
