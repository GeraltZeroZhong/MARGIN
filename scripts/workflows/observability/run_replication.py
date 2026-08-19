#!/usr/bin/env python
"""Run resumable stages of the locked observability study replication."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.observability.config import load_observability_config
from margin.studies.observability.replication import (
    build_replication_state_bank,
    export_replication_requests,
    replication_paths,
    score_replication_teachers,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["state-bank", "requests", "teachers", "paths"])
    parser.add_argument("--config", type=Path, default=Path("configs/observability.yaml"))
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    config = load_observability_config(arguments.config)
    if arguments.stage == "state-bank":
        print(f"state_bank={build_replication_state_bank(config)}")
    elif arguments.stage == "requests":
        print(f"requests={export_replication_requests(config)}")
    elif arguments.stage == "teachers":
        print(f"teacher_cache={score_replication_teachers(config, device=arguments.device)}")
    else:
        for name, path in replication_paths(config).items():
            print(f"{name}={path}")


if __name__ == "__main__":
    main()
