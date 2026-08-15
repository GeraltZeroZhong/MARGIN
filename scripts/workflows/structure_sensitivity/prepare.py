#!/usr/bin/env python
"""Prepare matched experimental and predicted structure-sensitivity study structures."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.structure_sensitivity.panel import (
    load_structure_sensitivity_config,
    prepare_structure_sensitivity_panel,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/structure_sensitivity.yaml"),
    )
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    paths = prepare_structure_sensitivity_panel(
        load_structure_sensitivity_config(arguments.protocol), force=arguments.force
    )
    print(f"protocol_lock={paths['lock']}")
    print(f"requests={paths['requests']}")


if __name__ == "__main__":
    main()
