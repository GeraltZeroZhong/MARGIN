#!/usr/bin/env python
"""Build the complete action-validation study scientific report."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.action_validation.config import load_action_validation_config
from margin.studies.action_validation.report import build_action_validation_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/action_validation.yaml"))
    arguments = parser.parse_args()
    print(
        f"report={build_action_validation_report(load_action_validation_config(arguments.config))}"
    )


if __name__ == "__main__":
    main()
