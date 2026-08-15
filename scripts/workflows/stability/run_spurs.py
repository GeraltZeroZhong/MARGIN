#!/usr/bin/env python
"""Run SPURS as a separately reported supervised stability study upper bound."""

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

from margin.studies.stability.config import load_stability_config
from margin.studies.stability.supervised import run_spurs_upper_bound

MODEL_REPOSITORY = "cyclization9/SPURS"
MODEL_REVISION = "0cc7a565af8f31eb122819f95a9d16e27b3d1596"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path("external/repositories/SPURS"),
    )
    arguments = parser.parse_args()
    model_config = Path(
        hf_hub_download(
            repo_id=MODEL_REPOSITORY,
            revision=MODEL_REVISION,
            filename="spurs/.hydra/config.yaml",
        )
    )
    checkpoint = Path(
        hf_hub_download(
            repo_id=MODEL_REPOSITORY,
            revision=MODEL_REVISION,
            filename="spurs/checkpoints/best.ckpt",
        )
    )
    config = load_stability_config(Path("configs/stability.yaml"))
    outputs = run_spurs_upper_bound(
        config,
        repository=arguments.repository,
        model_config_path=model_config,
        checkpoint=checkpoint,
        model_revision=MODEL_REVISION,
        device=arguments.device,
    )
    for name, path in outputs.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
