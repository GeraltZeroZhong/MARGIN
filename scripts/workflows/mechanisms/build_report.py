#!/usr/bin/env python
"""Build the complete mechanism study Markdown report."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.mechanisms.config import load_mechanism_config
from margin.studies.mechanisms.report import build_mechanism_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/mechanisms.yaml"))
    arguments = parser.parse_args()
    paths = build_mechanism_report(load_mechanism_config(arguments.config))
    for name, path in paths.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
