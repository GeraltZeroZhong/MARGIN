#!/usr/bin/env python
"""Generate the frozen mechanism study MIF request set."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.mechanisms.config import load_mechanism_config
from margin.studies.mechanisms.counterfactuals import build_mechanism_counterfactuals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/mechanisms.yaml"))
    arguments = parser.parse_args()
    artifacts = build_mechanism_counterfactuals(load_mechanism_config(arguments.config))
    for name, path in artifacts.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
