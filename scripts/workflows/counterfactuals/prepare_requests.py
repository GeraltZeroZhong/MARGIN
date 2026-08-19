#!/usr/bin/env python
"""Generate the frozen counterfactual study MIF counterfactual requests."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.counterfactuals.config import load_counterfactual_config
from margin.studies.counterfactuals.counterfactuals import build_counterfactual_requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/counterfactuals.yaml"))
    arguments = parser.parse_args()
    artifacts = build_counterfactual_requests(load_counterfactual_config(arguments.config))
    for name, path in artifacts.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
