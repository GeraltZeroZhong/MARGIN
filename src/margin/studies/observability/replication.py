"""Executable stages for the locked observability study replication."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from margin.config import load_config
from margin.data_registry.registry import load_registry
from margin.decoys.generate import (
    DECOY_COLUMNS,
    DECOY_RESIDUE_COLUMNS,
    EDGE_COLUMNS,
    DecoyArtifacts,
    write_decoys,
)
from margin.provenance import read_json, sha256_file
from margin.state_sampling.bank import build_state_bank, load_state_bank, write_state_bank
from margin.student.esm2 import create_policy
from margin.studies.observability.config import ObservabilityStudyConfig
from margin.teachers.cache import (
    load_teacher_cache,
    merge_score_tables,
    sequence_policy_scores,
    write_teacher_cache,
)
from margin.teachers.external import run_external_teacher
from margin.teachers.requests import export_teacher_requests, load_teacher_requests


def build_replication_state_bank(config: ObservabilityStudyConfig) -> Path:
    """Build fixed states and frozen final-layer features for all locked domains."""

    _verify_protocol_lock(config)
    replication_config = load_config(config.paths.replication_config)
    manifest = replication_config.paths.state_bank_dir / "manifest.json"
    embeddings = replication_config.paths.embeddings_input
    if manifest.exists() and embeddings is not None and embeddings.exists():
        load_state_bank(replication_config.paths.state_bank_dir)
        return replication_config.paths.state_bank_dir
    registry = load_registry(replication_config.paths.registry_dir)
    policy = create_policy(replication_config, registry)
    bank, skipped = build_state_bank(registry, policy, replication_config)
    write_state_bank(
        replication_config.paths.state_bank_dir, bank, policy, replication_config, skipped
    )
    if embeddings is None:
        raise ValueError("replication configuration must declare paths.embeddings_input")
    policy.export_embeddings(bank, embeddings, replication_config)
    return replication_config.paths.state_bank_dir


def export_replication_requests(config: ObservabilityStudyConfig) -> Path:
    """Export paired-backbone requests; structural decoys are outside this workflow."""

    _verify_protocol_lock(config)
    replication_config = load_config(config.paths.replication_config)
    request_dir = replication_config.paths.teacher_cache_dir / "requests"
    request_path = request_dir / "requests.parquet"
    if (request_dir / "manifest.json").exists() and request_path.exists():
        load_teacher_requests(request_dir)
        return request_path
    registry = load_registry(replication_config.paths.registry_dir)
    bank = load_state_bank(replication_config.paths.state_bank_dir)
    empty_decoys = _empty_decoys()
    write_decoys(replication_config.paths.decoy_dir, empty_decoys, replication_config)
    export_teacher_requests(request_dir, bank, registry, empty_decoys, replication_config)
    return request_dir / "requests.parquet"


def score_replication_teachers(config: ObservabilityStudyConfig, *, device: str = "auto") -> Path:
    """Score every frozen teacher and materialize the canonical paired cache."""

    _verify_protocol_lock(config)
    replication_config = load_config(config.paths.replication_config)
    cache_manifest = replication_config.paths.teacher_cache_dir / "manifest.json"
    if cache_manifest.exists():
        load_teacher_cache(replication_config.paths.teacher_cache_dir)
        return cache_manifest
    bank = load_state_bank(replication_config.paths.state_bank_dir)
    requests = load_teacher_requests(replication_config.paths.teacher_cache_dir / "requests")
    enabled = [teacher for teacher in replication_config.teacher_cache.teachers if teacher.enabled]
    sequence = [teacher for teacher in enabled if teacher.role == "sequence"]
    if len(sequence) != 1:
        raise ValueError("replication requires exactly one sequence-policy score source")
    tables = [sequence_policy_scores(bank, sequence[0], replication_config)]
    request_path = replication_config.paths.teacher_cache_dir / "requests" / "requests.parquet"
    for teacher in enabled:
        if teacher.role == "sequence":
            continue
        raw_path = (
            replication_config.paths.teacher_cache_dir / "raw" / f"{teacher.teacher_id}.parquet"
        )
        tables.append(
            run_external_teacher(
                teacher,
                request_path,
                raw_path,
                replication_config,
                device=device,
            )
        )
    cache = merge_score_tables(tables)
    upstream = [
        replication_config.paths.registry_dir / "manifest.json",
        replication_config.paths.state_bank_dir / "manifest.json",
        replication_config.paths.decoy_dir / "manifest.json",
        replication_config.paths.teacher_cache_dir / "requests" / "manifest.json",
    ]
    write_teacher_cache(
        replication_config.paths.teacher_cache_dir,
        cache,
        replication_config,
        upstream,
        requests=requests.requests,
    )
    return cache_manifest


def replication_paths(config: ObservabilityStudyConfig) -> dict[str, Path]:
    """Return the canonical inputs used by the external long-running stages."""

    replication_config = load_config(config.paths.replication_config)
    return {
        "run": replication_config.paths.run_dir,
        "registry": replication_config.paths.registry_dir,
        "state_bank": replication_config.paths.state_bank_dir,
        "requests": replication_config.paths.teacher_cache_dir / "requests" / "requests.parquet",
        "representations": config.paths.storage_dir / "replication_layers",
    }


def _empty_decoys() -> DecoyArtifacts:
    return DecoyArtifacts(
        decoys=pd.DataFrame(columns=DECOY_COLUMNS),
        residues=pd.DataFrame(columns=DECOY_RESIDUE_COLUMNS),
        edges=pd.DataFrame(columns=EDGE_COLUMNS),
        skipped=pd.DataFrame(columns=["target_domain_id", "decoy_type", "reason"]),
    )


def _verify_protocol_lock(config: ObservabilityStudyConfig) -> None:
    lock_path = config.paths.run_dir / "protocol_lock.json"
    if not lock_path.exists():
        raise FileNotFoundError(
            "observability study protocol must be frozen before replication stages"
        )
    lock = read_json(lock_path)
    for artifact in lock.get("artifacts", []):
        source = Path(artifact["source"])
        if not source.exists() or sha256_file(source) != artifact["sha256"]:
            raise ValueError(f"observability study source differs from protocol lock: {source}")
