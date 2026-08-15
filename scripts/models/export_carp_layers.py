#!/usr/bin/env python
"""Export identity-safe query and local-window features from CARP-640M."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from margin.provenance import read_json, runtime_manifest, write_json, write_parquet
from margin.state_sampling.bank import load_state_bank
from margin.studies.observability.config import load_observability_config


def main() -> None:
    arguments = parse_arguments()
    config = load_observability_config(arguments.config)
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        print(f"complete={manifest_path}")
        return
    sys.path.insert(0, str(config.paths.carp_repository))
    from sequence_models.collaters import SimpleCollater
    from sequence_models.constants import PROTEIN_ALPHABET
    from sequence_models.pretrained import load_carp

    checkpoint = torch.load(
        config.paths.carp_model,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    hidden_size = int(checkpoint["d_model"])
    model = load_carp(checkpoint)
    del checkpoint
    device = torch.device(arguments.device if arguments.device != "auto" else "cuda:0")
    model = model.eval().to(device)
    collater = SimpleCollater(PROTEIN_ALPHABET, pad=True)
    bank = load_state_bank(arguments.state_bank)
    keys, offsets = _keys_and_offsets(bank.states)
    _validate_keys(keys, bank.positions)
    key_path = output / "keys.parquet"
    array_path = output / "representations.npy"
    progress_path = output / "progress.json"
    if not key_path.exists():
        write_parquet(key_path, keys)
    layer = config.probes.alternate_model_layer
    shape = (len(keys), 1, 2, hidden_size)
    if array_path.exists():
        representations = np.load(array_path, mmap_mode="r+")
        if representations.shape != shape or representations.dtype != np.float16:
            raise ValueError("existing CARP representation array has incompatible shape or dtype")
    else:
        representations = np.lib.format.open_memmap(
            array_path, mode="w+", dtype=np.float16, shape=shape
        )
    next_state = int(read_json(progress_path)["next_state"]) if progress_path.exists() else 0
    with torch.inference_mode():
        for state_index, state in enumerate(bank.states.itertuples(index=False)):
            if state_index < next_state:
                continue
            values = _encode_state(
                model,
                collater,
                str(state.state_sequence),
                layer,
                hidden_size,
                config.probes.alternate_model_batch_size,
                config.probes.local_window_radius,
                device,
            )
            start, stop = offsets[state.state_id]
            representations[start:stop, 0] = values
            checkpoint = (state_index + 1) % 10 == 0 or state_index + 1 == len(bank.states)
            if checkpoint:
                representations.flush()
                write_json(
                    progress_path,
                    {"next_state": state_index + 1, "states": len(bank.states)},
                )
                print(f"states={state_index + 1}/{len(bank.states)}", flush=True)
    write_json(
        manifest_path,
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": "observability.alternate_representations.v1",
            "model_id": config.probes.alternate_model_id,
            "architecture": "ByteNet_CNN",
            "model_path": str(config.paths.carp_model),
            "model_bytes": config.paths.carp_model.stat().st_size,
            "conditioning": "strict_leave_one_position_out",
            "layers": [layer],
            "feature_kinds": [
                "query",
                f"local_mean_radius_{config.probes.local_window_radius}",
            ],
            "dtype": "float16",
            "flush_interval_states": 10,
            "shape": list(shape),
            "keys": {"path": str(key_path), "rows": len(keys)},
            "representations": {"path": str(array_path), "bytes": array_path.stat().st_size},
        },
    )
    progress_path.unlink(missing_ok=True)


def _encode_state(
    model,
    collater,
    sequence: str,
    layer: int,
    hidden_size: int,
    batch_size: int,
    radius: int,
    device: torch.device,
) -> np.ndarray:
    length = len(sequence)
    result = np.empty((length, 2, hidden_size), dtype=np.float16)
    base = ["#" if token == "X" else token for token in sequence]
    for start in range(0, length, batch_size):
        positions = list(range(start, min(length, start + batch_size)))
        masked_sequences = []
        for position in positions:
            masked = base.copy()
            masked[position] = "#"
            masked_sequences.append("".join(masked))
        tokens = collater([[sequence] for sequence in masked_sequences])[0].to(device)
        output = model(tokens, repr_layers=[layer], logits=False)["representations"][layer]
        for batch_index, position in enumerate(positions):
            lower = max(0, position - radius)
            upper = min(length, position + radius + 1)
            result[position, 0] = output[batch_index, position].float().cpu().numpy()
            result[position, 1] = output[batch_index, lower:upper].mean(0).float().cpu().numpy()
    return result


def _keys_and_offsets(states: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, tuple[int, int]]]:
    frames = []
    offsets = {}
    start = 0
    for state in states.itertuples(index=False):
        length = len(state.state_sequence)
        frames.append(
            pd.DataFrame(
                {
                    "state_id": state.state_id,
                    "domain_id": state.domain_id,
                    "position": np.arange(length, dtype=int),
                }
            )
        )
        offsets[state.state_id] = (start, start + length)
        start += length
    return pd.concat(frames, ignore_index=True), offsets


def _validate_keys(keys: pd.DataFrame, positions: pd.DataFrame) -> None:
    columns = ["state_id", "domain_id", "position"]
    expected = keys.sort_values(columns, ignore_index=True)
    observed = positions[columns].sort_values(columns, ignore_index=True)
    if not expected.equals(observed):
        raise ValueError("state-bank positions do not match exported CARP representation keys")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/observability.yaml"))
    parser.add_argument("--state-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    main()
