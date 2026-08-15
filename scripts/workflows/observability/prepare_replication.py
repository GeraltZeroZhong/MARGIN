#!/usr/bin/env python
"""Prepare and freeze the large-sample observability study replication registry."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.observability.config import load_observability_config
from margin.studies.observability.prepare import prepare_replication_registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/observability.yaml"))
    arguments = parser.parse_args()
    registry = prepare_replication_registry(load_observability_config(arguments.config))
    print(registry.domains["observability_split"].value_counts().to_string())


if __name__ == "__main__":
    main()
