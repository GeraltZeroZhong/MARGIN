"""Export state/structure pairs once so isolated teacher environments share identical inputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from margin.config import ProjectConfig
from margin.data_registry.registry import RegistryTables
from margin.decoys.generate import DecoyArtifacts
from margin.provenance import (
    runtime_manifest,
    sha256_file,
    table_manifest,
    write_json,
    write_parquet,
)
from margin.state_sampling.bank import StateBank

REQUEST_COLUMNS = (
    "request_id",
    "state_id",
    "domain_id",
    "state_sequence",
    "structure_role",
    "structure_id",
    "input_kind",
    "input_path",
    "length",
)


@dataclass(frozen=True)
class TeacherRequests:
    requests: pd.DataFrame
    structure_index: pd.DataFrame


def export_teacher_requests(
    directory: Path,
    bank: StateBank,
    registry: RegistryTables,
    decoys: DecoyArtifacts,
    config: ProjectConfig,
) -> TeacherRequests:
    """Write backbone/graph NPZ artifacts and a canonical request table."""

    directory.mkdir(parents=True, exist_ok=True)
    input_directory = directory / "inputs"
    input_directory.mkdir(parents=True, exist_ok=True)
    paired_residues = {
        domain_id: frame.sort_values("position").reset_index(drop=True)
        for domain_id, frame in registry.residues.groupby("domain_id", sort=False, observed=True)
    }
    decoy_residues = {
        decoy_id: frame.sort_values("position").reset_index(drop=True)
        for decoy_id, frame in decoys.residues.groupby("decoy_id", sort=False, observed=True)
    }
    decoy_edges = {
        decoy_id: frame.sort_values(["source", "target"]).reset_index(drop=True)
        for decoy_id, frame in decoys.edges.groupby("decoy_id", sort=False, observed=True)
    }
    structures: list[dict[str, Any]] = []
    input_lookup: dict[tuple[str, str], tuple[str, Path]] = {}

    for domain_id, residues in paired_residues.items():
        structure_id = domain_id
        path = input_directory / f"{_safe_name(structure_id)}.npz"
        _write_coordinate_npz(path, residues)
        input_lookup[("paired", structure_id)] = ("coordinates", path)
        structures.append(_structure_row("paired", structure_id, domain_id, "coordinates", path))

    for decoy in decoys.decoys.itertuples(index=False):
        if bool(decoy.supports_coordinate_teacher):
            path = input_directory / f"{_safe_name(decoy.decoy_id)}.npz"
            _write_coordinate_npz(path, decoy_residues[decoy.decoy_id])
            input_kind = "coordinates"
        else:
            path = input_directory / f"{_safe_name(decoy.decoy_id)}.npz"
            edge_frame = decoy_edges.get(decoy.decoy_id, pd.DataFrame(columns=["source", "target"]))
            np.savez_compressed(
                path,
                coordinates=np.stack(
                    [
                        paired_residues[decoy.target_domain_id][
                            [f"{atom}_{axis}" for axis in "xyz"]
                        ].to_numpy(dtype=np.float32)
                        for atom in ("n", "ca", "c", "o")
                    ],
                    axis=1,
                ),
                edges=edge_frame[["source", "target"]].to_numpy(dtype=np.int32),
                length=np.array([int(decoy.target_length)], dtype=np.int32),
            )
            input_kind = "contact_graph"
        input_lookup[(decoy.decoy_type, decoy.decoy_id)] = (input_kind, path)
        structures.append(
            _structure_row(
                decoy.decoy_type,
                decoy.decoy_id,
                decoy.target_domain_id,
                input_kind,
                path,
            )
        )

    decoys_by_target = {
        domain_id: frame
        for domain_id, frame in decoys.decoys.groupby("target_domain_id", sort=False, observed=True)
    }
    requests: list[dict[str, Any]] = []
    for state in bank.states.itertuples(index=False):
        paired_kind, paired_path = input_lookup[("paired", state.domain_id)]
        requests.append(
            _request_row(
                state,
                "paired",
                state.domain_id,
                paired_kind,
                paired_path,
            )
        )
        for decoy in decoys_by_target.get(state.domain_id, pd.DataFrame()).itertuples(index=False):
            input_kind, path = input_lookup[(decoy.decoy_type, decoy.decoy_id)]
            requests.append(
                _request_row(
                    state,
                    decoy.decoy_type,
                    decoy.decoy_id,
                    input_kind,
                    path,
                )
            )
    request_table = pd.DataFrame(requests, columns=REQUEST_COLUMNS)
    structure_table = pd.DataFrame(structures)
    if request_table["request_id"].duplicated().any():
        raise ValueError("teacher request IDs must be unique")
    request_path = directory / "requests.parquet"
    structure_path = directory / "structures.parquet"
    write_parquet(request_path, request_table)
    write_parquet(structure_path, structure_table)
    manifest = {
        **runtime_manifest(config.paths.project_root),
        "schema_version": config.schema_version,
        "data_mode": config.data_mode,
        "decoy_parameters": config.decoys.model_dump(mode="json"),
        "requests": table_manifest(request_path, request_table),
        "structures": table_manifest(structure_path, structure_table),
        "input_files": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in sorted(input_directory.glob("*.npz"))
        ],
    }
    write_json(directory / "manifest.json", manifest)
    return TeacherRequests(requests=request_table, structure_index=structure_table)


def load_teacher_requests(directory: Path) -> TeacherRequests:
    requests = pd.read_parquet(directory / "requests.parquet")
    structures = pd.read_parquet(directory / "structures.parquet")
    missing = set(REQUEST_COLUMNS) - set(requests.columns)
    if missing:
        raise ValueError(f"teacher request table is missing columns: {sorted(missing)}")
    return TeacherRequests(requests=requests, structure_index=structures)


def _write_coordinate_npz(path: Path, residues: pd.DataFrame) -> None:
    coordinates = np.stack(
        [
            residues[[f"{atom}_{axis}" for axis in "xyz"]].to_numpy(dtype=np.float32)
            for atom in ("n", "ca", "c", "o")
        ],
        axis=1,
    )
    np.savez_compressed(path, coordinates=coordinates)


def _structure_row(
    structure_role: str,
    structure_id: str,
    target_domain_id: str,
    input_kind: str,
    path: Path,
) -> dict[str, Any]:
    return {
        "structure_role": structure_role,
        "structure_id": structure_id,
        "target_domain_id": target_domain_id,
        "input_kind": input_kind,
        "input_path": str(path),
        "sha256": sha256_file(path),
    }


def _request_row(
    state: Any,
    structure_role: str,
    structure_id: str,
    input_kind: str,
    path: Path,
) -> dict[str, Any]:
    request_id = f"{state.state_id}|{structure_role}|{structure_id}"
    return {
        "request_id": request_id,
        "state_id": state.state_id,
        "domain_id": state.domain_id,
        "state_sequence": state.state_sequence,
        "structure_role": structure_role,
        "structure_id": structure_id,
        "input_kind": input_kind,
        "input_path": str(path),
        "length": len(state.state_sequence),
    }


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value)
