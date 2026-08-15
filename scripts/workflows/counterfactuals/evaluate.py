#!/usr/bin/env python
"""Run the frozen counterfactual study route evaluation and decision rules."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.counterfactuals.config import load_counterfactual_config
from margin.studies.counterfactuals.evaluation import evaluate_counterfactuals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/counterfactuals.yaml"))
    arguments = parser.parse_args()
    results = evaluate_counterfactuals(load_counterfactual_config(arguments.config))
    print(f"route_a_passed={results['route_a_passed']}")
    print(f"route_b_passed={results['route_b_passed']}")
    print(f"decision={results['decision']}")
    print(f"manifest={results['manifest']}")


if __name__ == "__main__":
    main()
