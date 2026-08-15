#!/usr/bin/env python
"""Run all-layer observability study probes on current or replication artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.observability.config import load_observability_config
from margin.studies.observability.layerwise import (
    current_split_indices,
    replication_split_indices,
    run_layerwise_audit,
)
from margin.studies.observability.targets import (
    load_foundation_residual_dataset,
    load_replication_residual_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/observability.yaml"))
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=["current", "replication"], default="current")
    parser.add_argument("--replication-run", type=Path)
    arguments = parser.parse_args()
    config = load_observability_config(arguments.config)
    if arguments.dataset == "replication":
        dataset = load_replication_residual_dataset(config, arguments.replication_run)
        selection_train, selection_validation, final_train, final_test = replication_split_indices(
            dataset
        )
        final_label = "locked_test"
    else:
        dataset = load_foundation_residual_dataset(config)
        selection_train, selection_validation, final_train, final_test = current_split_indices(
            dataset, config.seed
        )
        final_label = "foundation_external_benchmark"
    artifacts = run_layerwise_audit(
        dataset,
        config,
        arguments.representations,
        arguments.output,
        selection_train=selection_train,
        selection_validation=selection_validation,
        final_train=final_train,
        final_test=final_test,
        final_split_label=final_label,
    )
    for name, path in artifacts.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
