#!/usr/bin/env python
"""Export strict leave-one-position-out query features for a generalization study model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from margin.constants import AA_ALPHABET
from margin.provenance import read_json, runtime_manifest, write_json, write_parquet
from margin.studies.generalization.config import (
    ArchitectureModel,
    GeneralizationStudyConfig,
    load_generalization_config,
)


def main() -> None:
    arguments = parse_arguments()
    config = load_generalization_config(arguments.config)
    specification = next(
        model for model in config.architecture.models if model.model_id == arguments.model_id
    )
    queries = pd.read_parquet(arguments.queries).sort_values(
        ["state_id", "position"], ignore_index=True
    )
    validate_queries(queries)
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        print(f"complete={manifest_path}")
        return
    if specification.reuse_observability_store and not arguments.force_inference:
        reuse_observability_features(config, specification, queries, output)
        return
    infer(config, specification, queries, output, arguments)


def reuse_observability_features(
    config: GeneralizationStudyConfig,
    specification: ArchitectureModel,
    queries: pd.DataFrame,
    output: Path,
) -> None:
    source = (
        config.paths.observability_carp_representations
        if specification.reuse_observability_store == "carp"
        else config.paths.observability_esm2_representations
    )
    source_manifest = read_json(source / "manifest.json")
    source_keys = pd.read_parquet(source / "keys.parquet")
    source_keys = source_keys.copy()
    source_keys["source_row"] = np.arange(len(source_keys), dtype=int)
    key_columns = ["state_id", "domain_id", "position"]
    aligned = queries[key_columns].merge(source_keys, on=key_columns, validate="one_to_one")
    if len(aligned) != len(queries):
        raise ValueError(
            "observability study representation store does not cover generalization study queries"
        )
    layers = [int(layer) for layer in source_manifest["layers"]]
    layer_offset = layers.index(specification.layer)
    source_store = np.load(source / "representations.npy", mmap_mode="r")
    features = np.asarray(
        source_store[aligned["source_row"].to_numpy(dtype=int), layer_offset, 0, :],
        dtype=np.float16,
    )
    array_path = output / "representations.npy"
    np.save(array_path, features)
    key_path = output / "keys.parquet"
    write_parquet(key_path, queries[key_columns])
    write_json(
        output / "manifest.json",
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "model_id": specification.model_id,
            "family": specification.family,
            "scale_millions": specification.scale_millions,
            "conditioning": "strict_leave_one_position_out",
            "feature": "final_layer_query",
            "layer": specification.layer,
            "dtype": "float16",
            "shape": list(features.shape),
            "source": str(source),
            "keys": str(key_path),
            "representations": str(array_path),
        },
    )
    print(f"complete={output / 'manifest.json'}")


def infer(
    config: GeneralizationStudyConfig,
    specification: ArchitectureModel,
    queries: pd.DataFrame,
    output: Path,
    arguments: argparse.Namespace,
) -> None:
    device = torch.device(arguments.device if arguments.device != "auto" else "cuda:0")
    configured_checkpoint = getattr(config.paths, specification.checkpoint_path_key)
    configured_loader = specification.loader
    checkpoint = (
        arguments.checkpoint.resolve()
        if arguments.checkpoint is not None
        else configured_checkpoint
    )
    if arguments.loader is not None:
        specification = specification.model_copy(update={"loader": arguments.loader})
    encoder, hidden_size = load_encoder(config, specification, checkpoint, device)
    key_columns = ["state_id", "domain_id", "position"]
    key_path = output / "keys.parquet"
    write_parquet(key_path, queries[key_columns])
    representation_path = output / "representations.npy"
    score_path = output / "log_probabilities.npy"
    progress_path = output / "progress.json"
    features = open_store(representation_path, (len(queries), hidden_size))
    scores = (
        open_store(score_path, (len(queries), len(AA_ALPHABET))) if arguments.save_logp else None
    )
    next_group = int(read_json(progress_path)["next_group"]) if progress_path.exists() else 0
    grouped = list(queries.groupby("state_id", sort=False, observed=True))
    with torch.inference_mode():
        for group_index, (_, frame) in enumerate(grouped):
            if group_index < next_group:
                continue
            sequence_values = frame["sequence"].drop_duplicates()
            if len(sequence_values) != 1:
                raise ValueError("every query state must have exactly one sequence")
            positions = frame["position"].to_numpy(dtype=int)
            state_features, state_scores = encoder(
                str(sequence_values.iloc[0]),
                positions,
                specification.batch_size,
                arguments.save_logp,
            )
            rows = frame.index.to_numpy(dtype=int)
            features[rows] = state_features
            if scores is not None and state_scores is not None:
                scores[rows] = state_scores
            if (group_index + 1) % 5 == 0 or group_index + 1 == len(grouped):
                features.flush()
                if scores is not None:
                    scores.flush()
                write_json(
                    progress_path,
                    {"next_group": group_index + 1, "groups": len(grouped)},
                )
                print(f"groups={group_index + 1}/{len(grouped)}", flush=True)
    manifest = {
        **runtime_manifest(config.paths.project_root),
        "schema_version": config.schema_version,
        "model_id": specification.model_id,
        "family": specification.family,
        "scale_millions": specification.scale_millions,
        "loader": specification.loader,
        "configured_loader": configured_loader,
        "checkpoint": str(checkpoint),
        "configured_checkpoint": str(configured_checkpoint),
        "runtime_loader_override": arguments.loader is not None,
        "runtime_checkpoint_override": arguments.checkpoint is not None,
        "checkpoint_bytes": _path_bytes(checkpoint),
        "conditioning": "strict_leave_one_position_out",
        "feature": "final_layer_query",
        "layer": specification.layer,
        "dtype": "float16",
        "shape": list(features.shape),
        "keys": str(key_path),
        "representations": str(representation_path),
        "log_probabilities": str(score_path) if scores is not None else None,
    }
    write_json(output / "manifest.json", manifest)
    progress_path.unlink(missing_ok=True)
    print(f"complete={output / 'manifest.json'}")


def load_encoder(
    config: GeneralizationStudyConfig,
    specification: ArchitectureModel,
    checkpoint: Path,
    device: torch.device,
):
    if specification.loader == "carp":
        sys.path.insert(0, str(config.paths.sequence_models_repository))
        from sequence_models.collaters import SimpleCollater
        from sequence_models.constants import PROTEIN_ALPHABET
        from sequence_models.pretrained import load_carp

        model_data = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        if int(model_data["n_layers"]) != specification.layer:
            raise ValueError(
                f"{specification.model_id} layer mismatch: "
                f"checkpoint={model_data['n_layers']} config={specification.layer}"
            )
        hidden_size = int(model_data["d_model"])
        model = load_carp(model_data).eval().to(device)
        del model_data
        collater = SimpleCollater(PROTEIN_ALPHABET, pad=True)

        def encode(sequence: str, positions: np.ndarray, batch_size: int, save_logp: bool):
            result = np.empty((len(positions), hidden_size), dtype=np.float16)
            score_result = (
                np.empty((len(positions), len(AA_ALPHABET)), dtype=np.float16)
                if save_logp
                else None
            )
            base = ["#" if token == "X" else token for token in sequence]
            aa_indices = [PROTEIN_ALPHABET.index(aa) for aa in AA_ALPHABET]
            for start in range(0, len(positions), batch_size):
                selected = positions[start : start + batch_size]
                masked_sequences = []
                for position in selected:
                    masked = base.copy()
                    masked[int(position)] = "#"
                    masked_sequences.append("".join(masked))
                tokens = collater([[value] for value in masked_sequences])[0].to(device)
                output = model(
                    tokens,
                    repr_layers=[specification.layer],
                    logits=save_logp,
                )
                batch_index = torch.arange(len(selected), device=device)
                token_index = torch.as_tensor(selected, device=device)
                hidden = output["representations"][specification.layer][batch_index, token_index]
                result[start : start + len(selected)] = hidden.float().cpu().numpy()
                if score_result is not None:
                    logits = output["logits"][batch_index, token_index][:, aa_indices]
                    score_result[start : start + len(selected)] = (
                        logits.log_softmax(dim=-1).float().cpu().numpy()
                    )
            return result, score_result

        return encode, hidden_size

    if specification.loader == "hf_esm":
        from transformers import AutoTokenizer, EsmForMaskedLM

        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        model_dtype = torch.float16 if device.type == "cuda" else torch.float32
        model = EsmForMaskedLM.from_pretrained(
            checkpoint,
            local_files_only=True,
            low_cpu_mem_usage=True,
            torch_dtype=model_dtype,
        )
        model = model.eval().to(device)
        hidden_size = int(model.config.hidden_size)
        aa_indices = torch.tensor(tokenizer.convert_tokens_to_ids(list(AA_ALPHABET)), device=device)

        def encode(sequence: str, positions: np.ndarray, batch_size: int, save_logp: bool):
            result = np.empty((len(positions), hidden_size), dtype=np.float16)
            score_result = (
                np.empty((len(positions), len(AA_ALPHABET)), dtype=np.float16)
                if save_logp
                else None
            )
            base = [tokenizer.mask_token if token == "X" else token for token in sequence]
            for start in range(0, len(positions), batch_size):
                selected = positions[start : start + batch_size]
                masked_sequences = []
                for position in selected:
                    masked = base.copy()
                    masked[int(position)] = tokenizer.mask_token
                    masked_sequences.append("".join(masked))
                encoded = tokenizer(
                    masked_sequences,
                    add_special_tokens=True,
                    padding=True,
                    return_tensors="pt",
                )
                encoded = {name: value.to(device) for name, value in encoded.items()}
                output_values = model.esm(**encoded, output_hidden_states=False, return_dict=True)
                batch_index = torch.arange(len(selected), device=device)
                token_index = torch.as_tensor(selected, device=device) + 1
                hidden = output_values.last_hidden_state[batch_index, token_index]
                result[start : start + len(selected)] = hidden.float().cpu().numpy()
                if score_result is not None:
                    logits = model.lm_head(output_values.last_hidden_state)
                    logits = logits[batch_index, token_index].index_select(-1, aa_indices)
                    score_result[start : start + len(selected)] = (
                        logits.log_softmax(dim=-1).float().cpu().numpy()
                    )
            return result, score_result

        return encode, hidden_size

    if specification.loader == "fair_esm":
        from margin.config import load_config

        foundation = load_config(config.paths.observability_replication_config)
        fair_esm_teacher = next(
            teacher
            for teacher in foundation.teacher_cache.teachers
            if teacher.teacher_id == "esm_if1"
        )
        if fair_esm_teacher.repository is None:
            raise ValueError(
                "observability study ESM-IF1 specification lacks the pinned fair-esm repository"
            )
        sys.path.insert(0, str(fair_esm_teacher.repository))
        import esm

        model_data = load_checkpoint(checkpoint)
        regression_path = checkpoint.with_name(f"{checkpoint.stem}-contact-regression.pt")
        regression_data = load_checkpoint(regression_path) if regression_path.exists() else None
        model, alphabet = esm.pretrained.load_model_and_alphabet_core(
            checkpoint.stem,
            model_data,
            regression_data,
        )
        del model_data, regression_data
        model = model.eval().half().to(device)
        hidden_size = int(model.embed_dim)
        aa_indices = torch.tensor([alphabet.get_idx(aa) for aa in AA_ALPHABET], device=device)
        _, _, base_tokens = alphabet.get_batch_converter()([("query", "")])
        del base_tokens

        def encode(sequence: str, positions: np.ndarray, batch_size: int, save_logp: bool):
            result = np.empty((len(positions), hidden_size), dtype=np.float16)
            score_result = (
                np.empty((len(positions), len(AA_ALPHABET)), dtype=np.float16)
                if save_logp
                else None
            )
            _, _, encoded = alphabet.get_batch_converter()([("query", sequence)])
            encoded = encoded[0]
            for start in range(0, len(positions), batch_size):
                selected = positions[start : start + batch_size]
                tokens = encoded.unsqueeze(0).repeat(len(selected), 1).to(device)
                batch_index = torch.arange(len(selected), device=device)
                token_index = torch.as_tensor(selected, device=device) + 1
                tokens[batch_index, token_index] = alphabet.mask_idx
                output_values = model(
                    tokens,
                    repr_layers=[specification.layer],
                    return_contacts=False,
                )
                hidden = output_values["representations"][specification.layer][
                    batch_index, token_index
                ]
                result[start : start + len(selected)] = hidden.float().cpu().numpy()
                if score_result is not None:
                    logits = output_values["logits"][batch_index, token_index].index_select(
                        -1, aa_indices
                    )
                    score_result[start : start + len(selected)] = (
                        logits.log_softmax(dim=-1).float().cpu().numpy()
                    )
            return result, score_result

        return encode, hidden_size
    raise ValueError(f"unsupported loader: {specification.loader}")


def load_checkpoint(path: Path):
    """Memory-map modern checkpoints and load the reachable legacy ESM-1b format normally."""

    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError as error:
        if "mmap can only be used with files saved" not in str(error):
            raise
        return torch.load(path, map_location="cpu", weights_only=False)


def open_store(path: Path, shape: tuple[int, int]) -> np.memmap:
    if path.exists():
        store = np.load(path, mmap_mode="r+")
        if store.shape != shape or store.dtype != np.float16:
            raise ValueError(f"incompatible partial store: {path}")
        return store
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)


def validate_queries(queries: pd.DataFrame) -> None:
    required = {"state_id", "domain_id", "position", "sequence"}
    missing = required - set(queries.columns)
    if missing:
        raise ValueError(f"query table lacks columns: {sorted(missing)}")
    if queries.duplicated(["state_id", "position"]).any():
        raise ValueError("query state-position keys must be unique")
    lengths = queries["sequence"].str.len()
    if (~queries["position"].between(0, lengths - 1)).any():
        raise ValueError("query position lies outside its sequence")


def _path_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/generalization.yaml"))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--loader", choices=["carp", "hf_esm", "fair_esm"])
    parser.add_argument("--save-logp", action="store_true")
    parser.add_argument("--force-inference", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
