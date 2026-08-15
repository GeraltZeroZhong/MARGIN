#!/usr/bin/env python
"""Run generalization study teacher-lineage and architecture-scale CATH audits."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.generalization.audit import run_cath_audits
from margin.studies.generalization.config import load_generalization_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/generalization.yaml"))
    arguments = parser.parse_args()
    for name, path in run_cath_audits(load_generalization_config(arguments.config)).items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
