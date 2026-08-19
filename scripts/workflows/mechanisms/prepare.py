#!/usr/bin/env python
"""Prepare the outcome-blind dense mechanism study panel."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.mechanisms.config import load_mechanism_config
from margin.studies.mechanisms.prepare import prepare_mechanism_panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/mechanisms.yaml"))
    arguments = parser.parse_args()
    artifacts = prepare_mechanism_panel(load_mechanism_config(arguments.config))
    for name, path in artifacts.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
