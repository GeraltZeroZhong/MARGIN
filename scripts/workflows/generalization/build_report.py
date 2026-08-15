#!/usr/bin/env python
"""Build the final generalization study report and decision record."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.generalization.config import load_generalization_config
from margin.studies.generalization.report import build_generalization_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/generalization.yaml"))
    arguments = parser.parse_args()
    for name, path in build_generalization_report(
        load_generalization_config(arguments.config)
    ).items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
