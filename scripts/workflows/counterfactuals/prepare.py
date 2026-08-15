#!/usr/bin/env python
"""Prepare the label-uninspected counterfactual study validation panel."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.counterfactuals.config import load_counterfactual_config
from margin.studies.counterfactuals.prepare import prepare_counterfactual_panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/counterfactuals.yaml"))
    arguments = parser.parse_args()
    outputs = prepare_counterfactual_panel(load_counterfactual_config(arguments.config))
    for name, path in outputs.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
