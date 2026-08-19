#!/usr/bin/env python
"""Build the complete locked stability study scientific report."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.stability.config import load_stability_config
from margin.studies.stability.report import build_stability_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/stability.yaml"))
    arguments = parser.parse_args()
    for name, path in build_stability_report(load_stability_config(arguments.config)).items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
