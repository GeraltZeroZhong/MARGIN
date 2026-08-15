"""Supervised stability upper-bound adapters for the post-lock stability study audit."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from margin.provenance import runtime_manifest, write_json, write_parquet
from margin.studies.stability.config import StabilityStudyConfig


def run_thermompnn_upper_bound(
    config: StabilityStudyConfig,
    *,
    repository: Path,
    checkpoint: Path,
    device: str,
) -> dict[str, Path]:
    """Score the stability study panel with the official ThermoMPNN checkpoint.

    The output is an upper-bound audit, not a zero-shot comparator.  Exact split
    overlap is attached to every prediction so downstream tables cannot silently
    pool training, validation, and test domains.
    """

    repository = repository.resolve()
    checkpoint = checkpoint.resolve()
    _add_thermompnn_imports(repository)
    from datasets import Mutation
    from protein_mpnn_utils import alt_parse_PDB
    from train_thermompnn import TransferModelPL

    run = config.paths.run_dir
    output = run / "supervised" / "thermompnn"
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "predictions.parquet"
    timing_path = output / "domain_timing.parquet"
    variants = pd.read_parquet(run / "panel" / "variants.parquet").reset_index(drop=True)
    variants["variant_row"] = np.arange(len(variants), dtype=int)
    domains = pd.read_parquet(run / "panel" / "domains.parquet")
    overlap_path = run / "method_audit" / "supervised_split_overlap.parquet"
    overlap = pd.read_parquet(overlap_path)[["domain_id", "thermompnn_split"]]
    domain_metadata = domains.merge(overlap, on="domain_id", how="left", validate="one_to_one")
    domain_metadata["thermompnn_split"] = domain_metadata["thermompnn_split"].fillna(
        "external_not_megascale"
    )

    model_config = OmegaConf.create(
        {
            "training": {
                "num_workers": 0,
                "learn_rate": 0.001,
                "epochs": 100,
                "lr_schedule": True,
            },
            "model": {
                "hidden_dims": [64, 32],
                "subtract_mut": True,
                "num_final_layers": 2,
                "freeze_weights": True,
                "load_pretrained": True,
                "lightattn": True,
                "lr_schedule": True,
            },
            "platform": {"thermompnn_dir": str(repository)},
        }
    )
    model = (
        TransferModelPL.load_from_checkpoint(str(checkpoint), cfg=model_config)
        .model.eval()
        .to(device)
    )
    completed = (
        pd.read_parquet(prediction_path)
        if prediction_path.exists()
        else pd.DataFrame(columns=_thermompnn_prediction_columns())
    )
    timing = (
        pd.read_parquet(timing_path)
        if timing_path.exists()
        else pd.DataFrame(columns=_thermompnn_timing_columns())
    )
    completed_domains = set(completed["domain_id"].astype(str))
    for domain in domain_metadata.itertuples(index=False):
        if domain.domain_id in completed_domains:
            continue
        frame = variants.loc[variants["domain_id"].eq(domain.domain_id)].copy()
        parsed = alt_parse_PDB(str(domain.structure_path), str(domain.chain_id))
        parsed_sequence = str(parsed[0]["seq"])
        expected_sequence = str(domain.sequence)
        if parsed_sequence != expected_sequence:
            raise ValueError(
                f"ThermoMPNN PDB/query sequence mismatch for {domain.domain_id}: "
                f"{len(parsed_sequence)} != {len(expected_sequence)} or residues differ"
            )
        mutations = [
            Mutation(
                position=int(row.position),
                wildtype=str(row.wild_type),
                mutation=str(row.mutant),
                pdb=str(parsed[0]["name"]),
            )
            for row in frame.itertuples(index=False)
        ]
        started = time.perf_counter()
        with torch.inference_mode():
            predictions, _ = model(parsed, mutations)
        elapsed = time.perf_counter() - started
        raw = [float(value["ddG"].detach().cpu().item()) for value in predictions]
        scored = frame[["variant_row", "domain_id", "position", "wild_type", "mutant"]].copy()
        scored["thermompnn_raw_ddg"] = raw
        scored["predicted_stability"] = -scored["thermompnn_raw_ddg"]
        scored["thermompnn_split"] = str(domain.thermompnn_split)
        completed = pd.concat([completed, scored], ignore_index=True)
        timing = pd.concat(
            [
                timing,
                pd.DataFrame(
                    [
                        {
                            "domain_id": domain.domain_id,
                            "thermompnn_split": domain.thermompnn_split,
                            "length": int(domain.length),
                            "n_variants": int(len(frame)),
                            "wall_seconds": elapsed,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        write_parquet(
            prediction_path,
            completed.sort_values("variant_row", ignore_index=True),
        )
        write_parquet(timing_path, timing.sort_values("domain_id", ignore_index=True))
        print(
            f"thermompnn_domain={domain.domain_id} variants={len(frame)} seconds={elapsed:.3f}",
            flush=True,
        )
    completed = completed.sort_values("variant_row", ignore_index=True)
    if len(completed) != len(variants) or completed["variant_row"].duplicated().any():
        raise ValueError(
            f"ThermoMPNN coverage mismatch: predictions={len(completed)} variants={len(variants)}"
        )
    if not np.array_equal(
        completed["variant_row"].to_numpy(dtype=int), np.arange(len(variants), dtype=int)
    ):
        raise ValueError("ThermoMPNN variant order is incomplete")
    manifest = output / "manifest.json"
    write_json(
        manifest,
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": "stability.supervised_upper_bound.v1",
            "analysis_role": "supervised_upper_bound_separate_from_zero_shot",
            "model": "ThermoMPNN",
            "repository": str(repository),
            "repository_revision": _git_revision(repository),
            "checkpoint": str(checkpoint),
            "device": device,
            "prediction_sign": (
                "predicted_stability=-official_raw_ddG because the official Megascale "
                "loader trains on -ddG_ML"
            ),
            "split_policy": {
                "train": "transductive upper bound only",
                "val": "model-selection overlap; not untouched test",
                "test": "official exact-name held-out test",
                "not_listed": "independence not inferred",
                "external_not_megascale": "external dataset; training homology not asserted",
            },
            "predictions": str(prediction_path),
            "timing": str(timing_path),
            "n_predictions": int(len(completed)),
            "n_domains": int(completed["domain_id"].nunique()),
        },
    )
    return {"predictions": prediction_path, "timing": timing_path, "manifest": manifest}


def run_spurs_upper_bound(
    config: StabilityStudyConfig,
    *,
    repository: Path,
    model_config_path: Path,
    checkpoint: Path,
    model_revision: str,
    device: str,
) -> dict[str, Path]:
    """Score the panel with the official SPURS full-matrix inference path."""

    repository = repository.resolve()
    model_config_path = model_config_path.resolve()
    checkpoint = checkpoint.resolve()
    value = str(repository)
    if value not in sys.path:
        sys.path.insert(0, value)
    from spurs.inference import parse_pdb
    from spurs.models.stability.spurs import SPURS

    run = config.paths.run_dir
    output = run / "supervised" / "spurs"
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "predictions.parquet"
    timing_path = output / "domain_timing.parquet"
    variants = pd.read_parquet(run / "panel" / "variants.parquet").reset_index(drop=True)
    variants["variant_row"] = np.arange(len(variants), dtype=int)
    domains = pd.read_parquet(run / "panel" / "domains.parquet")
    overlap = pd.read_parquet(run / "method_audit" / "supervised_split_overlap.parquet")[
        ["domain_id", "thermompnn_split"]
    ]
    domain_metadata = domains.merge(overlap, on="domain_id", how="left", validate="one_to_one")
    domain_metadata["thermompnn_split"] = domain_metadata["thermompnn_split"].fillna(
        "external_not_megascale"
    )
    model_config = OmegaConf.load(model_config_path)
    del model_config["model"]["_target_"]
    seed = int(model_config["train"]["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    load_started = time.perf_counter()
    model = SPURS(model_config["model"]).eval().to(device)
    container = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    state = {key[6:]: tensor for key, tensor in container["state_dict"].items() if "model." in key}
    incompatibility = model.load_state_dict(state, strict=False)
    del state, container
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise ValueError(
            "SPURS checkpoint mismatch: "
            f"missing={incompatibility.missing_keys} "
            f"unexpected={incompatibility.unexpected_keys}"
        )
    model_load_seconds = time.perf_counter() - load_started
    completed = (
        pd.read_parquet(prediction_path)
        if prediction_path.exists()
        else pd.DataFrame(columns=_spurs_prediction_columns())
    )
    timing = (
        pd.read_parquet(timing_path)
        if timing_path.exists()
        else pd.DataFrame(columns=_thermompnn_timing_columns())
    )
    completed_domains = set(completed["domain_id"].astype(str))
    for domain in domain_metadata.itertuples(index=False):
        if domain.domain_id in completed_domains:
            continue
        frame = variants.loc[variants["domain_id"].eq(domain.domain_id)].copy()
        batch = parse_pdb(
            str(domain.structure_path),
            str(domain.domain_id),
            str(domain.chain_id),
            model_config,
            device,
        )
        parsed_sequence = str(batch["seq"])
        expected_sequence = str(domain.sequence)
        if parsed_sequence != expected_sequence:
            raise ValueError(
                f"SPURS PDB/query sequence mismatch for {domain.domain_id}: "
                f"{len(parsed_sequence)} != {len(expected_sequence)} or residues differ"
            )
        started = time.perf_counter()
        with torch.inference_mode():
            matrix = model(batch, return_logist=True).detach().cpu().numpy()
        elapsed = time.perf_counter() - started
        if matrix.shape != (len(expected_sequence), 20) or not np.isfinite(matrix).all():
            raise ValueError(f"SPURS matrix contract failed for {domain.domain_id}: {matrix.shape}")
        position = frame["position"].to_numpy(dtype=int)
        mutant = frame["mutant"].map(_spurs_aa_index()).to_numpy(dtype=int)
        raw = matrix[position, mutant]
        scored = frame[["variant_row", "domain_id", "position", "wild_type", "mutant"]].copy()
        scored["spurs_raw_ddg"] = raw
        scored["predicted_stability"] = -scored["spurs_raw_ddg"]
        scored["thermompnn_split"] = str(domain.thermompnn_split)
        completed = pd.concat([completed, scored], ignore_index=True)
        timing = pd.concat(
            [
                timing,
                pd.DataFrame(
                    [
                        {
                            "domain_id": domain.domain_id,
                            "thermompnn_split": domain.thermompnn_split,
                            "length": int(domain.length),
                            "n_variants": int(len(frame)),
                            "wall_seconds": elapsed,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        write_parquet(
            prediction_path,
            completed.sort_values("variant_row", ignore_index=True),
        )
        write_parquet(timing_path, timing.sort_values("domain_id", ignore_index=True))
        print(
            f"spurs_domain={domain.domain_id} variants={len(frame)} seconds={elapsed:.3f}",
            flush=True,
        )
    completed = completed.sort_values("variant_row", ignore_index=True)
    if len(completed) != len(variants) or completed["variant_row"].duplicated().any():
        raise ValueError(f"SPURS coverage mismatch: {len(completed)}/{len(variants)}")
    if not np.array_equal(
        completed["variant_row"].to_numpy(dtype=int), np.arange(len(variants), dtype=int)
    ):
        raise ValueError("SPURS variant order is incomplete")
    manifest = output / "manifest.json"
    write_json(
        manifest,
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": "stability.supervised_upper_bound.v1",
            "analysis_role": "supervised_upper_bound_separate_from_zero_shot",
            "model": "SPURS",
            "repository": str(repository),
            "repository_revision": _git_revision(repository),
            "model_revision": model_revision,
            "model_config": str(model_config_path),
            "checkpoint": str(checkpoint),
            "checkpoint_bytes": int(checkpoint.stat().st_size),
            "device": device,
            "model_load_seconds": model_load_seconds,
            "prediction_sign": (
                "predicted_stability=-official_raw_ddG because the official Megascale "
                "loader trains on -ddG_ML"
            ),
            "split_policy": {
                "train": "transductive upper bound only",
                "val": "model-selection overlap; not untouched test",
                "test": "official exact-name held-out test",
                "not_listed": "independence not inferred",
                "external_not_megascale": "external dataset; training homology not asserted",
            },
            "predictions": str(prediction_path),
            "timing": str(timing_path),
            "n_predictions": int(len(completed)),
            "n_domains": int(completed["domain_id"].nunique()),
        },
    )
    return {"predictions": prediction_path, "timing": timing_path, "manifest": manifest}


def _add_thermompnn_imports(repository: Path) -> None:
    for path in (repository / "analysis", repository):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _git_revision(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _thermompnn_prediction_columns() -> list[str]:
    return [
        "variant_row",
        "domain_id",
        "position",
        "wild_type",
        "mutant",
        "thermompnn_raw_ddg",
        "predicted_stability",
        "thermompnn_split",
    ]


def _thermompnn_timing_columns() -> list[str]:
    return ["domain_id", "thermompnn_split", "length", "n_variants", "wall_seconds"]


def _spurs_prediction_columns() -> list[str]:
    return [
        "variant_row",
        "domain_id",
        "position",
        "wild_type",
        "mutant",
        "spurs_raw_ddg",
        "predicted_stability",
        "thermompnn_split",
    ]


def _spurs_aa_index() -> dict[str, int]:
    return {aa: index for index, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
