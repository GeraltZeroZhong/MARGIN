#!/usr/bin/env python
"""Run the frozen action-validation study G/C/U evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.action_validation.config import load_action_validation_config
from margin.studies.action_validation.evaluation import evaluate_action_validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/action_validation.yaml"))
    arguments = parser.parse_args()
    result = evaluate_action_validation(load_action_validation_config(arguments.config))
    print(f"decision={result['decision']}")
    print(f"confirmed={result['confirmed']}")
    print(f"manifest={result['manifest']}")


if __name__ == "__main__":
    main()
