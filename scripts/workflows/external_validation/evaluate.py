#!/usr/bin/env python
"""Materialize frozen C+ scores and then evaluate the cross-platform endpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.external_validation.evaluation import (
    build_external_validation_scores,
    evaluate_external_validation,
)
from margin.studies.external_validation.panel import load_external_validation_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/external_validation.yaml"),
    )
    parser.add_argument("--scores-only", action="store_true")
    arguments = parser.parse_args()
    config = load_external_validation_config(arguments.protocol)
    score_paths = build_external_validation_scores(config)
    print(f"frozen_scores={score_paths['components']}", flush=True)
    if arguments.scores_only:
        return
    result = evaluate_external_validation(config)
    print(f"decision={result['decision_name']}")
    print(f"manifest={result['manifest']}")


if __name__ == "__main__":
    main()
