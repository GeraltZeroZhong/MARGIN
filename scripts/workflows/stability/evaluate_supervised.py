#!/usr/bin/env python
"""Evaluate completed supervised stability study upper bounds by official split."""

from pathlib import Path

from margin.studies.stability.config import load_stability_config
from margin.studies.stability.supervised_evaluation import evaluate_supervised_upper_bounds


def main() -> None:
    config = load_stability_config(Path("configs/stability.yaml"))
    outputs = evaluate_supervised_upper_bounds(config)
    for name, path in outputs.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
