#!/usr/bin/env python
"""Prepare and freeze the outcome-blind external-validation panel."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.external_validation.panel import (
    load_external_validation_config,
    prepare_external_validation_panel,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/external_validation.yaml"),
    )
    arguments = parser.parse_args()
    config = load_external_validation_config(arguments.protocol)
    paths = prepare_external_validation_panel(config)
    print(f"protocol_lock={paths['lock']}")
    print(f"domains={paths['domains']}")
    print(f"queries={paths['queries']}")


if __name__ == "__main__":
    main()
