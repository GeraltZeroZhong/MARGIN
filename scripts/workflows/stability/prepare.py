#!/usr/bin/env python
"""Prepare the outcome-blind stability study panel and paired teacher requests."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.stability.config import load_stability_config
from margin.studies.stability.prepare import prepare_stability_panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/stability.yaml"))
    arguments = parser.parse_args()
    paths = prepare_stability_panel(load_stability_config(arguments.config))
    for name, path in paths.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
