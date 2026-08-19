#!/usr/bin/env python
"""Run the locked stability evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.stability.config import load_stability_config
from margin.studies.stability.evaluation import evaluate_stability


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/stability.yaml"))
    arguments = parser.parse_args()
    result = evaluate_stability(load_stability_config(arguments.config))
    print(f"decision={result['decision']}")
    print(f"paired_action_confirmed={result['paired_action_confirmed']}")
    print(f"selective_routing_confirmed={result['selective_routing_confirmed']}")
    print(f"manifest={result['manifest']}")


if __name__ == "__main__":
    main()
