#!/usr/bin/env python
"""Empirically audit ProteinMPNN conditional-score semantics and order variance."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from margin.provenance import runtime_manifest, table_manifest, write_json, write_parquet

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def main() -> None:
    arguments = parse_arguments()
    repository = arguments.repository.resolve()
    sys.path.insert(0, str(repository))
    from protein_mpnn_utils import ProteinMPNN, tied_featurize

    device = torch.device(arguments.device if arguments.device != "auto" else "cuda:0")
    checkpoint = torch.load(arguments.weights, map_location=device, weights_only=False)
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
    registry = pd.read_parquet(arguments.registry / "domains.parquet")
    external_domains = set(
        registry.loc[registry["analysis_role"].eq("external_benchmark"), "domain_id"]
    )
    requests = requests.loc[
        requests["structure_role"].eq("paired")
        & requests["state_id"].str.contains(":native_reference:")
        & requests["input_kind"].eq("coordinates")
        & requests["domain_id"].isin(external_domains)
    ].head(arguments.domains)
    rows = []
    score_rows = []
    started = time.perf_counter()
    with torch.inference_mode():
        for request in requests.itertuples(index=False):
            coordinates = np.load(request.input_path)["coordinates"].astype(float)
            protein = protein_record(request.request_id, request.state_sequence, coordinates)
            chain_dictionary = {request.request_id: (["A"], [])}
            features = tied_featurize([protein], device, chain_dictionary)
            x, sequence_tokens, mask, _lengths, chain_mask = features[:5]
            chain_encoding = features[5]
            chain_position_mask = features[10]
            residue_index = features[12]
            reference = None
            maximum_difference = 0.0
            repeat_entropies = []
            draws = []
            for repeat in range(arguments.repeats):
                generator = torch.Generator(device=device).manual_seed(arguments.seed + repeat)
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
                )[0, :, : len(AA_ALPHABET)]
                values = values.log_softmax(dim=-1)
                if reference is None:
                    reference = values
                else:
                    maximum_difference = max(
                        maximum_difference, float(torch.max(torch.abs(values - reference)))
                    )
                repeat_entropies.append(
                    float(torch.mean(-torch.sum(values.exp() * values, dim=-1)))
                )
                draws.append(values)
            assert reference is not None
            averaged = torch.logsumexp(torch.stack(draws, dim=0), dim=0) - np.log(arguments.repeats)
            averaged = averaged.log_softmax(dim=-1).cpu().numpy()
            for position, values in enumerate(averaged):
                score_rows.append(
                    {
                        "domain_id": request.domain_id,
                        "state_id": request.state_id,
                        "position": position,
                        **{
                            f"logp_{amino_acid}": float(values[index])
                            for index, amino_acid in enumerate(AA_ALPHABET)
                        },
                    }
                )
            random_order = torch.zeros_like(chain_mask)
            backbone_only = model.conditional_probs(
                x,
                sequence_tokens,
                mask,
                chain_mask * chain_position_mask,
                residue_index,
                chain_encoding,
                random_order,
                True,
            )[0, :, : len(AA_ALPHABET)].log_softmax(dim=-1)
            jsd = rowwise_jsd(reference.cpu().numpy(), backbone_only.cpu().numpy())
            rows.append(
                {
                    "domain_id": request.domain_id,
                    "length": int(request.length),
                    "order_repeats": arguments.repeats,
                    "maximum_absolute_order_difference": maximum_difference,
                    "conditional_entropy": float(np.mean(repeat_entropies)),
                    "backbone_only_jsd_nats": float(jsd.mean()),
                }
            )
    table = pd.DataFrame(rows)
    scores = pd.DataFrame(score_rows)
    output = arguments.output.resolve()
    write_parquet(output / "order_variance.parquet", table)
    write_parquet(output / "mc_scores.parquet", scores)
    write_json(
        output / "semantics.json",
        {
            **runtime_manifest(arguments.project_root.resolve()),
            "adapter_mode": "conditional_probs(backbone_only=False)",
            "interpretation": "P(residue_i | sequence_except_i, backbone)",
            "candidate_order": "all canonical candidates share one conditional vector",
            "wt_mutant_order_confounded": False,
            "random_decoding_order_used": True,
            "random_decoding_order_changes_scores": bool(
                (table["maximum_absolute_order_difference"] > arguments.tolerance).any()
            ),
            "canonical_normalization": "renormalized over 20 canonical amino acids",
            "temperature_rank_effect": "positive scalar temperature preserves within-site ranks",
            "population": "foundation_external_benchmark_native_reference",
            "elapsed_seconds": time.perf_counter() - started,
            "artifact": table_manifest(output / "order_variance.parquet", table),
            "mc_scores": table_manifest(output / "mc_scores.parquet", scores),
        },
    )
    print(table.to_string(index=False))


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


def rowwise_jsd(left_logp: np.ndarray, right_logp: np.ndarray) -> np.ndarray:
    midpoint = np.logaddexp(left_logp, right_logp) - np.log(2.0)
    left = np.sum(np.exp(left_logp) * (left_logp - midpoint), axis=1)
    right = np.sum(np.exp(right_logp) * (right_logp - midpoint), axis=1)
    return 0.5 * (left + right)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--requests",
        type=Path,
        default=Path("runs/foundation/teacher_cache/requests/requests.parquet"),
    )
    parser.add_argument("--registry", type=Path, default=Path("runs/foundation/registry"))
    parser.add_argument(
        "--repository", type=Path, default=Path("external/repositories/ProteinMPNN")
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("external/repositories/ProteinMPNN/vanilla_model_weights/v_48_020.pt"),
    )
    parser.add_argument("--output", type=Path, default=Path("runs/observability/proteinmpnn"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--domains", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    return parser.parse_args()


if __name__ == "__main__":
    main()
