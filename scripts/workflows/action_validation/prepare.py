#!/usr/bin/env python
"""Prepare the outcome-blind action-validation study two-platform panel."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.action_validation.config import load_action_validation_config
from margin.studies.action_validation.prepare import prepare_action_validation_panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/action_validation.yaml"))
    arguments = parser.parse_args()
    artifacts = prepare_action_validation_panel(load_action_validation_config(arguments.config))
    for name, path in artifacts.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
