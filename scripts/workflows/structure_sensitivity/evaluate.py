#!/usr/bin/env python
"""Run the frozen descriptive structure-sensitivity study matched-structure evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.structure_sensitivity.evaluation import evaluate_structure_sensitivity
from margin.studies.structure_sensitivity.panel import load_structure_sensitivity_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/structure_sensitivity.yaml"),
    )
    arguments = parser.parse_args()
    result = evaluate_structure_sensitivity(load_structure_sensitivity_config(arguments.protocol))
    print(f"decision={result['decision_name']}")
    print(f"manifest={result['manifest']}")


if __name__ == "__main__":
    main()
