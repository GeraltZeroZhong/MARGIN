#!/usr/bin/env python
"""Export identity-safe ESM2 query and local-window features from every hidden layer."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmForMaskedLM

from margin.config import load_config
from margin.provenance import read_json, runtime_manifest, sha256_file, write_json, write_parquet
from margin.state_sampling.bank import load_state_bank


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)
    bank = load_state_bank(arguments.state_bank)
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_path = config.student_policy.model_path
    expected_hash = config.student_policy.weights_sha256
    if model_path is None or expected_hash is None:
        raise ValueError("ESM2 layer export requires a local model path and weight identity")
    weights = model_path / "model.safetensors"
    if sha256_file(weights) != expected_hash:
        raise ValueError("ESM2 checkpoint does not match the frozen configuration")
    device = torch.device(arguments.device if arguments.device != "auto" else "cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = EsmForMaskedLM.from_pretrained(model_path, local_files_only=True).eval().to(device)
    layers = list(range(int(model.config.num_hidden_layers) + 1))
    hidden_size = int(model.config.hidden_size)
    keys, offsets = _keys_and_offsets(bank.states)
    _validate_keys(keys, bank.positions)
    key_path = output / "keys.parquet"
    array_path = output / "representations.npy"
    progress_path = output / "progress.json"
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        print(f"complete={manifest_path}")
        return
    if not key_path.exists():
        write_parquet(key_path, keys)
    shape = (len(keys), len(layers), 2, hidden_size)
    if array_path.exists():
        representations = np.load(array_path, mmap_mode="r+")
        if representations.shape != shape or representations.dtype != np.float16:
            raise ValueError("existing representation array has incompatible shape or dtype")
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
                tokenizer,
                str(state.state_sequence),
                layers,
                arguments.batch_size,
                arguments.local_radius,
                device,
            )
            start, stop = offsets[state.state_id]
            representations[start:stop] = values
            representations.flush()
            write_json(progress_path, {"next_state": state_index + 1, "states": len(bank.states)})
            if (state_index + 1) % 10 == 0 or state_index + 1 == len(bank.states):
                print(f"states={state_index + 1}/{len(bank.states)}", flush=True)
    write_json(
        manifest_path,
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": "observability.representations.v1",
            "policy_id": config.student_policy.policy_id,
            "model_revision": config.student_policy.model_revision,
            "weights_sha256": expected_hash,
            "conditioning": "strict_leave_one_position_out",
            "layers": layers,
            "feature_kinds": ["query", f"local_mean_radius_{arguments.local_radius}"],
            "dtype": "float16",
            "shape": list(shape),
            "keys": {"path": str(key_path), "rows": len(keys)},
            "representations": {"path": str(array_path), "bytes": array_path.stat().st_size},
        },
    )
    progress_path.unlink(missing_ok=True)


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
        raise ValueError("state-bank positions do not match exported representation keys")


def _encode_state(
    model,
    tokenizer,
    sequence: str,
    layers: list[int],
    batch_size: int,
    radius: int,
    device: torch.device,
) -> np.ndarray:
    length = len(sequence)
    hidden_size = int(model.config.hidden_size)
    result = np.empty((length, len(layers), 2, hidden_size), dtype=np.float16)
    base = [tokenizer.mask_token if token == "X" else token for token in sequence]
    for start in range(0, length, batch_size):
        positions = list(range(start, min(length, start + batch_size)))
        masked_sequences = []
        for position in positions:
            masked = base.copy()
            masked[position] = tokenizer.mask_token
            masked_sequences.append("".join(masked))
        encoded = tokenizer(
            masked_sequences,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        output = model(**encoded, output_hidden_states=True, return_dict=True)
        batch = np.empty((len(positions), len(layers), 2, hidden_size), dtype=np.float16)
        for layer_offset, layer in enumerate(layers):
            hidden = output.hidden_states[layer]
            for batch_index, position in enumerate(positions):
                token_index = position + 1
                lower = max(0, position - radius) + 1
                upper = min(length, position + radius + 1) + 1
                batch[batch_index, layer_offset, 0] = (
                    hidden[batch_index, token_index].float().cpu().numpy()
                )
                batch[batch_index, layer_offset, 1] = (
                    hidden[batch_index, lower:upper].mean(dim=0).float().cpu().numpy()
                )
        result[positions] = batch
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--local-radius", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    main()
