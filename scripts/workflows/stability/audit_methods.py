#!/usr/bin/env python
"""Run the retrospective stability study baseline, simplification, and cost audit."""

from pathlib import Path

from margin.studies.stability.config import load_stability_config
from margin.studies.stability.method_audit import audit_stability_methods


def main() -> None:
    config = load_stability_config(Path("configs/stability.yaml"))
    outputs = audit_stability_methods(config)
    for name, path in outputs.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
