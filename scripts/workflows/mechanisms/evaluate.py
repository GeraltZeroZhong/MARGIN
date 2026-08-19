#!/usr/bin/env python
"""Run the frozen mechanism study mechanism audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.mechanisms.config import load_mechanism_config
from margin.studies.mechanisms.evaluation import evaluate_mechanisms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/mechanisms.yaml"))
    arguments = parser.parse_args()
    result = evaluate_mechanisms(load_mechanism_config(arguments.config))
    for name, value in result.items():
        print(f"{name}={value}")


if __name__ == "__main__":
    main()
