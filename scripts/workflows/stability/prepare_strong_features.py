#!/usr/bin/env python
"""Build frozen outcome-free features for the strengthened sequence control."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.provenance import read_json
from margin.studies.stability.config import load_stability_config
from margin.studies.stability.strong_control import build_strong_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/stability.yaml"))
    arguments = parser.parse_args()
    config = load_stability_config(arguments.config)
    lock = read_json(config.paths.run_dir / "protocol_lock.json")
    if lock.get("status") != "FROZEN_BEFORE_STABILITY_PANEL_MODEL_SCORING":
        raise RuntimeError("stability study protocol lock is missing")
    paths = build_strong_features(config)
    for name, path in paths.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
