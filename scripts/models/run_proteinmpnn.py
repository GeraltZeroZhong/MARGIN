#!/usr/bin/env python
"""Isolated ProteinMPNN conditional-probability runner."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from margin.teachers.runner_cache import (
    completed_request_ids,
    finalize_parts,
    part_directory,
    write_request_part,
)

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def main() -> None:
    arguments = parse_arguments()
    repository = Path(arguments.repository).resolve()
    sys.path.insert(0, str(repository))
    from protein_mpnn_utils import ProteinMPNN, tied_featurize

    device = choose_device(arguments.device)
    checkpoint_path = (
        arguments.weights or repository / "vanilla_model_weights" / f"{arguments.model}.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ProteinMPNN(
        ca_only=False,
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        augment_eps=0.0,
        k_neighbors=checkpoint["num_edges"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    requests = pd.read_parquet(arguments.requests)
    requests = requests.loc[requests["input_kind"] == "coordinates"]
    if arguments.limit is not None:
        requests = requests.head(arguments.limit)
    state_seed_index = {
        state_id: index for index, state_id in enumerate(sorted(requests["state_id"].unique()))
    }
    parts = part_directory(arguments.output, arguments.run_key)
    completed = completed_request_ids(parts)
    with torch.no_grad():
        for ordinal, request in enumerate(requests.itertuples(index=False)):
            if str(request.request_id) in completed:
                continue
            started = time.perf_counter()
            coordinates = np.load(request.input_path)["coordinates"].astype(float)
            protein = protein_record(request.request_id, request.state_sequence, coordinates)
            chain_dictionary = {request.request_id: (["A"], [])}
            (
                x,
                sequence_tokens,
                mask,
                _lengths,
                chain_mask,
                chain_encoding,
                _chain_list,
                _visible,
                _masked,
                _masked_lengths,
                chain_position_mask,
                _omit_mask,
                residue_index,
                _dihedral_mask,
                _tied_positions,
                _pssm_coefficient,
                _pssm_bias,
                _pssm_log_odds,
                _bias_by_residue,
                _tied_beta,
            ) = tied_featurize([protein], device, chain_dictionary)
            log_probabilities = monte_carlo_conditional_probabilities(
                model,
                x,
                sequence_tokens,
                mask,
                chain_mask,
                chain_position_mask,
                residue_index,
                chain_encoding,
                device,
                arguments.seed,
                state_seed_index[request.state_id],
                arguments.order_repeats,
            )
            elapsed = time.perf_counter() - started
            write_request_part(
                parts,
                ordinal,
                score_rows(
                    request.request_id,
                    log_probabilities,
                    (
                        "full_sequence_backbone_conditional"
                        if arguments.order_repeats == 1
                        else (
                            "full_sequence_backbone_conditional_"
                            f"mc_probability_mean_{arguments.order_repeats}"
                        )
                    ),
                    str(device),
                    elapsed,
                    arguments.order_repeats,
                ),
            )
            if (ordinal + 1) % 10 == 0 or ordinal + 1 == len(requests):
                print(f"requests={ordinal + 1}/{len(requests)}", flush=True)
    finalize_parts(
        arguments.output,
        parts,
        requests["request_id"].astype(str).tolist(),
    )


def monte_carlo_conditional_probabilities(
    model,
    x: Tensor,
    sequence_tokens: Tensor,
    mask: Tensor,
    chain_mask: Tensor,
    chain_position_mask: Tensor,
    residue_index: Tensor,
    chain_encoding: Tensor,
    device: torch.device,
    seed: int,
    state_index: int,
    repeats: int,
) -> np.ndarray:
    """Average probabilities over orders shared by every structure role of a state."""

    draws = []
    for repeat in range(repeats):
        generator = torch.Generator(device=device).manual_seed(
            seed + state_index * repeats + repeat
        )
        random_order = torch.randn(chain_mask.shape, generator=generator, device=device)
        values = model.conditional_probs(
            x,
            sequence_tokens,
            mask,
            chain_mask * chain_position_mask,
            residue_index,
            chain_encoding,
            random_order,
            False,
        )[0, :, : len(AA_ALPHABET)].log_softmax(dim=-1)
        draws.append(values)
    stacked = torch.stack(draws, dim=0)
    averaged = torch.logsumexp(stacked, dim=0) - np.log(repeats)
    return averaged.log_softmax(dim=-1).cpu().numpy()


def protein_record(name: str, sequence: str, coordinates: np.ndarray) -> dict[str, object]:
    return {
        "name": name,
        "num_of_chains": 1,
        "seq": sequence,
        "seq_chain_A": sequence,
        "coords_chain_A": {
            "N_chain_A": coordinates[:, 0, :].tolist(),
            "CA_chain_A": coordinates[:, 1, :].tolist(),
            "C_chain_A": coordinates[:, 2, :].tolist(),
            "O_chain_A": coordinates[:, 3, :].tolist(),
        },
    }


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
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--model", default="v_48_020")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--order-repeats", type=int, default=1)
    parser.add_argument("--run-key", default="unkeyed")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main()
