#!/usr/bin/env python
"""Isolated MIF/MIF-ST runner for canonical MARGIN teacher requests."""

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
    from sequence_models.pdb_utils import process_coords

    device = choose_device(arguments.device)
    model, collater = load_mifst(arguments)
    model = model.eval().to(device)
    requests = pd.read_parquet(arguments.requests)
    requests = requests.loc[
        requests["input_kind"].isin(
            ["coordinates", "contact_graph", "contact_deletion", "contact_reassignment"]
        )
    ]
    if arguments.limit is not None:
        requests = requests.head(arguments.limit)
    parts = part_directory(arguments.output, arguments.run_key)
    completed = completed_request_ids(parts)
    with torch.no_grad():
        for ordinal, request in enumerate(requests.itertuples(index=False)):
            if str(request.request_id) in completed:
                continue
            started = time.perf_counter()
            input_data = np.load(request.input_path)
            coordinates = input_data["coordinates"][:, :3, :]
            coordinate_dict = {
                "N": coordinates[:, 0],
                "CA": coordinates[:, 1],
                "C": coordinates[:, 2],
            }
            distance, omega, theta, phi = process_coords(coordinate_dict)
            conditioning = (
                "leave_one_out_masked_bidirectional_structure"
                if arguments.model == "mif"
                else "leave_one_out_masked_bidirectional_structure_sequence"
            )
            if request.input_kind == "contact_graph":
                edges = input_data["edges"]
                distance, omega, theta, phi = contact_graph_geometry(
                    distance, omega, theta, phi, edges
                )
                conditioning = "leave_one_out_masked_sequence_rewired_contact_graph"
            elif request.input_kind == "contact_deletion":
                distance, omega, theta, phi = contact_deletion_geometry(
                    distance,
                    omega,
                    theta,
                    phi,
                    input_data["removed_edges"],
                )
                conditioning = "leave_one_out_masked_sequence_native_geometry_contact_deletion"
            elif request.input_kind == "contact_reassignment":
                distance, omega, theta, phi = contact_reassignment_geometry(
                    distance,
                    omega,
                    theta,
                    phi,
                    input_data["removed_edges"],
                    input_data["added_edges"],
                    input_data["source_edges"],
                )
                conditioning = (
                    "leave_one_out_masked_sequence_native_geometry_constrained_reassignment"
                )
            log_probabilities = leave_one_out_log_probabilities(
                model,
                collater,
                request.state_sequence,
                distance,
                omega,
                theta,
                phi,
                device,
                arguments.batch_size,
            )
            elapsed = time.perf_counter() - started
            write_request_part(
                parts,
                ordinal,
                score_rows(
                    request.request_id,
                    log_probabilities,
                    conditioning,
                    str(device),
                    elapsed,
                    int(np.ceil(request.length / arguments.batch_size)),
                ),
            )
            if (ordinal + 1) % 10 == 0 or ordinal + 1 == len(requests):
                print(f"requests={ordinal + 1}/{len(requests)}", flush=True)
    finalize_parts(
        arguments.output,
        parts,
        requests["request_id"].astype(str).tolist(),
    )


def leave_one_out_log_probabilities(
    model,
    collater,
    state_sequence: str,
    distance: np.ndarray,
    omega: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Mask each queried position before reading its candidate distribution."""

    sequence = state_sequence.replace("X", "#")
    output = np.empty((len(sequence), len(AA_ALPHABET)), dtype=float)
    for start in range(0, len(sequence), batch_size):
        positions = list(range(start, min(len(sequence), start + batch_size)))
        batch = []
        for position in positions:
            masked = list(sequence)
            masked[position] = "#"
            batch.append(
                [
                    "".join(masked),
                    torch.tensor(distance.copy(), dtype=torch.float32),
                    torch.tensor(omega.copy(), dtype=torch.float32),
                    torch.tensor(theta.copy(), dtype=torch.float32),
                    torch.tensor(phi.copy(), dtype=torch.float32),
                ]
            )
        src, nodes, edge_features, connections, edge_mask = collater(batch)
        logits = model(
            src.to(device),
            nodes.to(device),
            edge_features.to(device),
            connections.to(device),
            edge_mask.to(device),
            result="logits",
        )[:, :, : len(AA_ALPHABET)]
        position_index = torch.tensor(positions, device=logits.device)
        batch_index = torch.arange(len(positions), device=logits.device)
        selected = logits[batch_index, position_index].log_softmax(dim=-1)
        output[positions] = selected.cpu().numpy()
    return output


def contact_graph_geometry(
    distance: np.ndarray,
    omega: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Use rewired adjacency, neutral pair angles, and original local node frames."""

    result = np.full_like(distance, 20.0)
    np.fill_diagonal(result, 0.0)
    for left, right in np.asarray(edges, dtype=int):
        result[left, right] = 6.0
        result[right, left] = 6.0
    neutral_angles = []
    for values in (omega, theta, phi):
        neutral = np.zeros_like(values)
        for offset in (-1, 1):
            rows = np.arange(max(0, -offset), min(len(values), len(values) - offset))
            columns = rows + offset
            neutral[rows, columns] = values[rows, columns]
        neutral_angles.append(neutral)
    return result, *neutral_angles


def contact_deletion_geometry(
    distance: np.ndarray,
    omega: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    removed_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove selected nonlocal contacts while preserving every other native feature."""

    outputs = [values.copy() for values in (distance, omega, theta, phi)]
    for left, right in np.asarray(removed_edges, dtype=int).reshape(-1, 2):
        outputs[0][left, right] = outputs[0][right, left] = 20.0
        for values in outputs[1:]:
            values[left, right] = values[right, left] = 0.0
    return tuple(outputs)


def contact_reassignment_geometry(
    distance: np.ndarray,
    omega: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    removed_edges: np.ndarray,
    added_edges: np.ndarray,
    source_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Move native edge features to topology-matched nonedges for a constrained graph edit."""

    native = [values.copy() for values in (distance, omega, theta, phi)]
    outputs = [values.copy() for values in native]
    removed = np.asarray(removed_edges, dtype=int).reshape(-1, 2)
    added = np.asarray(added_edges, dtype=int).reshape(-1, 2)
    sources = np.asarray(source_edges, dtype=int).reshape(-1, 2)
    if not (len(removed) == len(added) == len(sources)):
        raise ValueError("contact reassignment arrays must have identical edge counts")
    for left, right in removed:
        outputs[0][left, right] = outputs[0][right, left] = 20.0
        for values in outputs[1:]:
            values[left, right] = values[right, left] = 0.0
    for (left, right), (source_left, source_right) in zip(added, sources, strict=True):
        for output, values in zip(outputs, native, strict=True):
            output[left, right] = values[source_left, source_right]
            output[right, left] = values[source_right, source_left]
    return tuple(outputs)


def load_mifst(arguments: argparse.Namespace):
    """Load pinned local MIF or MIF-ST weights, otherwise use the upstream downloader."""

    if arguments.weights is None and arguments.auxiliary_weights is None:
        from sequence_models.pretrained import load_model_and_alphabet

        return load_model_and_alphabet(arguments.model)
    if arguments.weights is None:
        raise ValueError("local MIF loading requires --weights")
    if arguments.model == "mifst" and arguments.auxiliary_weights is None:
        raise ValueError("local MIF-ST loading requires --auxiliary-weights")
    if arguments.model not in {"mif", "mifst"}:
        raise ValueError("local loading supports only --model mif or mifst")
    from sequence_models.collaters import SimpleCollater, StructureCollater
    from sequence_models.constants import PROTEIN_ALPHABET
    from sequence_models.pretrained import MIF, load_carp, load_gnn

    gnn_data = torch.load(arguments.weights, map_location="cpu", weights_only=False, mmap=True)
    cnn = None
    if arguments.model == "mifst":
        carp_data = torch.load(
            arguments.auxiliary_weights,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        cnn = load_carp(carp_data)
    model = MIF(load_gnn(gnn_data), cnn=cnn)
    collater = StructureCollater(SimpleCollater(PROTEIN_ALPHABET, pad=True), n_connections=30)
    return model, collater


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
    parser.add_argument("--auxiliary-weights", type=Path)
    parser.add_argument("--model", default="mifst")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--run-key", default="unkeyed")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main()
