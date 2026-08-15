#!/usr/bin/env python
"""Run exploratory counterfactual study mechanism and OOD analyses after the frozen gate."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.counterfactuals.config import load_counterfactual_config
from margin.studies.counterfactuals.mechanism import analyze_counterfactual_mechanisms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/counterfactuals.yaml"))
    arguments = parser.parse_args()
    paths = analyze_counterfactual_mechanisms(load_counterfactual_config(arguments.config))
    for name, path in paths.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
