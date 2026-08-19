"""Canonical teacher-score cache with coverage and provenance manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from margin.config import ProjectConfig, TeacherSpec
from margin.constants import AA_ALPHABET
from margin.provenance import (
    canonical_json_hash,
    runtime_manifest,
    sha256_file,
    table_manifest,
    write_json,
    write_parquet,
)
from margin.state_sampling.bank import StateBank
from margin.teachers.schema import logp_columns, validate_score_table

ADAPTER_INPUT_KINDS = {
    "mifst": {
        "coordinates",
        "contact_graph",
        "contact_deletion",
        "contact_reassignment",
    },
    "proteinmpnn": {"coordinates"},
    "esm_if1_candidates": {"coordinates"},
    "synthetic": {"coordinates", "contact_graph"},
}


@dataclass(frozen=True)
class TeacherScoreCache:
    scores: pd.DataFrame


def sequence_policy_scores(
    bank: StateBank,
    teacher: TeacherSpec,
    config: ProjectConfig,
) -> pd.DataFrame:
    """Promote the exact rollout policy distributions into the shared teacher matrix."""

    rows = bank.positions[
        ["state_id", "domain_id", "position", *[f"student_logp_{aa}" for aa in AA_ALPHABET]]
    ].copy()
    rows["teacher_id"] = teacher.teacher_id
    rows["teacher_role"] = teacher.role
    rows["structure_role"] = "sequence_only"
    rows["structure_id"] = ""
    rows["input_score_type"] = "log_probability"
    rows["conditioning"] = "sequence_only_state"
    rows["model_name"] = teacher.model_name
    rows["model_revision"] = teacher.model_revision
    rows["device"] = "policy"
    rows["wall_seconds"] = 0.0
    rows["forward_calls"] = 0
    rows["data_mode"] = config.data_mode
    rows = rows.rename(columns={f"student_logp_{aa}": f"logp_{aa}" for aa in AA_ALPHABET})
    validate_score_table(rows)
    return rows


def canonicalize_external_scores(
    raw: pd.DataFrame,
    requests: pd.DataFrame,
    teacher: TeacherSpec,
    config: ProjectConfig,
) -> pd.DataFrame:
    """Normalize model-specific score rows and attach immutable request metadata."""

    raw_columns = [f"score_{aa}" for aa in AA_ALPHABET]
    required = {"request_id", "position", *raw_columns, "conditioning", "device", "wall_seconds"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"external teacher output is missing columns: {sorted(missing)}")
    raw = raw.copy()
    if "forward_calls" not in raw:
        raw["forward_calls"] = 1
    joined = raw.merge(
        requests[
            ["request_id", "state_id", "domain_id", "structure_role", "structure_id", "length"]
        ],
        on="request_id",
        how="left",
        validate="many_to_one",
    )
    if joined["state_id"].isna().any():
        raise ValueError("external teacher output contains unknown request IDs")
    if (joined["position"] >= joined["length"]).any():
        raise ValueError("external teacher output contains out-of-range positions")
    scores = joined[raw_columns].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("external teacher output contains non-finite candidate scores")
    scores = np.clip(scores, -config.teacher_cache.score_clip, config.teacher_cache.score_clip)
    scores = scores / config.teacher_cache.normalization_temperature
    scores = scores - logsumexp(scores, axis=1, keepdims=True)
    result = joined[
        [
            "state_id",
            "domain_id",
            "position",
            "structure_role",
            "structure_id",
            "conditioning",
            "device",
            "wall_seconds",
            "forward_calls",
        ]
    ].copy()
    result["teacher_id"] = teacher.teacher_id
    result["teacher_role"] = teacher.role
    result["input_score_type"] = teacher.score_type
    result["model_name"] = teacher.model_name
    result["model_revision"] = teacher.model_revision
    result["data_mode"] = config.data_mode
    for index, column in enumerate(logp_columns()):
        result[column] = scores[:, index]
    validate_score_table(result)
    return result


def merge_score_tables(tables: list[pd.DataFrame]) -> TeacherScoreCache:
    if not tables:
        raise ValueError("at least one teacher score table is required")
    scores = pd.concat(tables, ignore_index=True)
    validate_score_table(scores)
    return TeacherScoreCache(scores=scores)


def write_teacher_cache(
    directory: Path,
    cache: TeacherScoreCache,
    config: ProjectConfig,
    upstream_manifests: list[Path] | None = None,
    requests: pd.DataFrame | None = None,
) -> dict[str, Any]:
    validate_score_table(cache.scores)
    directory.mkdir(parents=True, exist_ok=True)
    score_path = directory / "scores.parquet"
    write_parquet(score_path, cache.scores)
    coverage = (
        cache.scores.groupby(["teacher_id", "structure_role", "conditioning"], observed=True)
        .agg(
            rows=("position", "size"),
            states=("state_id", "nunique"),
            domains=("domain_id", "nunique"),
        )
        .reset_index()
    )
    coverage_path = directory / "coverage.parquet"
    write_parquet(coverage_path, coverage)
    coverage_audit = _coverage_audit(cache.scores, requests, config)
    coverage_audit_path = directory / "coverage_audit.parquet"
    write_parquet(coverage_audit_path, coverage_audit)
    missing_required = coverage_audit.loc[
        coverage_audit["required"] & (coverage_audit["coverage_fraction"] < 1.0 - 1e-12)
    ]
    if not missing_required.empty:
        details = missing_required[
            ["teacher_id", "structure_role", "expected_rows", "observed_rows"]
        ].to_dict(orient="records")
        raise ValueError(f"teacher cache lacks required adapter coverage: {details}")
    shard_directory = directory / "shards"
    shard_directory.mkdir(parents=True, exist_ok=True)
    shard_records: list[dict[str, Any]] = []
    for shard_index, start in enumerate(
        range(0, len(cache.scores), config.teacher_cache.shard_rows)
    ):
        shard = cache.scores.iloc[start : start + config.teacher_cache.shard_rows].copy()
        shard_path = shard_directory / f"scores-{shard_index:05d}.parquet"
        write_parquet(shard_path, shard)
        shard_records.append(table_manifest(shard_path, shard))
    upstream = [
        {"path": str(path), "sha256": sha256_file(path)}
        for path in (upstream_manifests or [])
        if path.exists()
    ]
    model_artifacts = _model_artifacts(config)
    compatibility_payload = {
        "schema_version": config.schema_version,
        "alphabet": AA_ALPHABET,
        "normalization_temperature": config.teacher_cache.normalization_temperature,
        "score_clip": config.teacher_cache.score_clip,
        "teachers": [teacher.model_dump(mode="json") for teacher in config.teacher_cache.teachers],
        "model_artifacts": model_artifacts,
        "upstream": upstream,
    }
    manifest = {
        **runtime_manifest(config.paths.project_root),
        "schema_version": config.schema_version,
        "data_mode": config.data_mode,
        "alphabet": AA_ALPHABET,
        "normalization_temperature": config.teacher_cache.normalization_temperature,
        "scores": table_manifest(score_path, cache.scores),
        "shards": shard_records,
        "coverage": table_manifest(coverage_path, coverage),
        "coverage_audit": table_manifest(coverage_audit_path, coverage_audit),
        "teachers": [teacher.model_dump(mode="json") for teacher in config.teacher_cache.teachers],
        "model_artifacts": model_artifacts,
        "upstream_manifests": upstream,
        "cache_compatibility_key": canonical_json_hash(compatibility_payload),
    }
    write_json(directory / "manifest.json", manifest)
    return manifest


def _model_artifacts(config: ProjectConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for teacher in config.teacher_cache.teachers:
        for role, path in (
            ("weights", teacher.weights),
            ("auxiliary_weights", teacher.auxiliary_weights),
        ):
            if path is None:
                continue
            if not path.exists():
                raise FileNotFoundError(f"configured teacher artifact does not exist: {path}")
            rows.append(
                {
                    "teacher_id": teacher.teacher_id,
                    "artifact_role": role,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return rows


def _coverage_audit(
    scores: pd.DataFrame,
    requests: pd.DataFrame | None,
    config: ProjectConfig,
) -> pd.DataFrame:
    columns = [
        "teacher_id",
        "structure_role",
        "input_kind",
        "required",
        "support_status",
        "expected_rows",
        "observed_rows",
        "coverage_fraction",
    ]
    if requests is None:
        return pd.DataFrame(columns=columns)
    expected = (
        requests.groupby(["structure_role", "input_kind"], observed=True)
        .agg(expected_rows=("length", "sum"))
        .reset_index()
    )
    observed = (
        scores.loc[scores["teacher_role"] != "sequence"]
        .groupby(["teacher_id", "structure_role"], observed=True)
        .size()
        .to_dict()
    )
    rows: list[dict[str, Any]] = []
    for teacher in config.teacher_cache.teachers:
        if not teacher.enabled or teacher.role == "sequence":
            continue
        supported = ADAPTER_INPUT_KINDS.get(teacher.adapter, set())
        for request in expected.itertuples(index=False):
            required = request.input_kind in supported
            observed_rows = int(observed.get((teacher.teacher_id, request.structure_role), 0))
            expected_rows = int(request.expected_rows)
            rows.append(
                {
                    "teacher_id": teacher.teacher_id,
                    "structure_role": request.structure_role,
                    "input_kind": request.input_kind,
                    "required": required,
                    "support_status": "supported" if required else "unsupported_by_adapter",
                    "expected_rows": expected_rows,
                    "observed_rows": observed_rows,
                    "coverage_fraction": (
                        observed_rows / expected_rows
                        if required and expected_rows
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def load_teacher_cache(directory: Path) -> TeacherScoreCache:
    cache = TeacherScoreCache(scores=pd.read_parquet(directory / "scores.parquet"))
    validate_score_table(cache.scores)
    return cache
