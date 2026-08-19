#!/usr/bin/env python
"""Run the fixed-final-layer audit on the locked observability study replication."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.observability.config import load_observability_config
from margin.studies.observability.current import run_replication_final_layer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/observability.yaml"))
    arguments = parser.parse_args()
    for name, path in run_replication_final_layer(
        load_observability_config(arguments.config)
    ).items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
