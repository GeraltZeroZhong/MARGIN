#!/usr/bin/env python
"""Run ThermoMPNN as a separately reported supervised stability study upper bound."""

import argparse
from pathlib import Path

from margin.studies.stability.config import load_stability_config
from margin.studies.stability.supervised import run_thermompnn_upper_bound


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path("external/repositories/ThermoMPNN"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("external/repositories/ThermoMPNN/models/thermoMPNN_default.pt"),
    )
    arguments = parser.parse_args()
    config = load_stability_config(Path("configs/stability.yaml"))
    outputs = run_thermompnn_upper_bound(
        config,
        repository=arguments.repository,
        checkpoint=arguments.checkpoint,
        device=arguments.device,
    )
    for name, path in outputs.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
