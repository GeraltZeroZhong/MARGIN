#!/usr/bin/env python
"""Isolated ESM-IF1 autoregressive candidate-distribution runner."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from margin.teachers.runner_cache import (
    completed_request_ids,
    finalize_parts,
    part_directory,
    write_request_part,
)

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def main() -> None:
    arguments = parse_arguments()
    if arguments.repository:
        sys.path.insert(0, str(Path(arguments.repository).resolve()))
    import biotite.structure

    if not hasattr(biotite.structure, "filter_backbone"):
        biotite.structure.filter_backbone = biotite.structure.filter_peptide_backbone
    import esm
    from esm.inverse_folding.util import CoordBatchConverter

    device = choose_device(arguments.device)
    if arguments.weights is not None:
        # The official ESM-IF1 checkpoint stores its model arguments as an
        # argparse.Namespace. PyTorch 2.6's weights-only loader requires that
        # metadata type to be explicitly allowlisted.
        with torch.serialization.safe_globals([argparse.Namespace]):
            model, alphabet = esm.pretrained.load_model_and_alphabet_local(arguments.weights)
    else:
        loader = getattr(esm.pretrained, arguments.model)
        model, alphabet = loader()
    model = model.eval().to(device)
    canonical_indices = torch.tensor([alphabet.get_idx(aa) for aa in AA_ALPHABET], device=device)
    batch_converter = CoordBatchConverter(alphabet)
    requests = pd.read_parquet(arguments.requests)
    requests = requests.loc[requests["input_kind"] == "coordinates"]
    if arguments.limit is not None:
        requests = requests.head(arguments.limit)
    parts = part_directory(arguments.output, arguments.run_key)
    completed = completed_request_ids(parts)
    with torch.no_grad():
        for ordinal, request in enumerate(requests.itertuples(index=False)):
            if str(request.request_id) in completed:
                continue
            started = time.perf_counter()
            coordinates = np.load(request.input_path)["coordinates"][:, :3, :]
            converted_coords, confidence, _strings, tokens, padding_mask = batch_converter(
                [(coordinates, None, request.state_sequence)], device=device
            )
            previous_tokens = tokens[:, :-1]
            logits, _ = model.forward(
                converted_coords,
                padding_mask,
                confidence,
                previous_tokens,
            )
            if logits.shape[1] == len(alphabet):
                position_logits = logits[0, :, : request.length].transpose(0, 1)
            elif logits.shape[-1] == len(alphabet):
                position_logits = logits[0, : request.length, :]
            else:
                raise ValueError(f"unexpected ESM-IF1 logits shape: {tuple(logits.shape)}")
            canonical = position_logits.index_select(-1, canonical_indices).log_softmax(dim=-1)
            elapsed = time.perf_counter() - started
            write_request_part(
                parts,
                ordinal,
                score_rows(
                    request.request_id,
                    canonical.cpu().numpy(),
                    "autoregressive_prefix_backbone_conditional",
                    str(device),
                    elapsed,
                    1,
                ),
            )
            if (ordinal + 1) % 10 == 0 or ordinal + 1 == len(requests):
                print(f"requests={ordinal + 1}/{len(requests)}", flush=True)
    finalize_parts(
        arguments.output,
        parts,
        requests["request_id"].astype(str).tolist(),
    )


def score_rows(
    request_id: str,
    scores: np.ndarray,
    conditioning: str,
    device: str,
    wall_seconds: float,
    forward_calls: int,
) -> list[dict[str, object]]:
    rows = []
    for position, values in enumerate(scores):
        row: dict[str, object] = {
            "request_id": request_id,
            "position": position,
            "conditioning": conditioning,
            "device": device,
            "wall_seconds": wall_seconds,
            "forward_calls": forward_calls,
        }
        row.update({f"score_{aa}": float(values[index]) for index, aa in enumerate(AA_ALPHABET)})
        rows.append(row)
    return rows


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--model", default="esm_if1_gvp4_t16_142M_UR50")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-key", default="unkeyed")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main()
