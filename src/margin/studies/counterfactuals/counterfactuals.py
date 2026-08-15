"""Build the frozen counterfactual study MIF counterfactual request set."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from margin.decoys.graph import contact_edges, degree_preserving_rewire, degrees
from margin.provenance import (
    read_json,
    runtime_manifest,
    sha256_file,
    table_manifest,
    write_json,
    write_parquet,
)
from margin.studies.counterfactuals.config import CounterfactualStudyConfig
from margin.teachers.requests import REQUEST_COLUMNS

STRUCTURE_COLUMNS = (
    "structure_role",
    "structure_id",
    "target_domain_id",
    "input_kind",
    "input_path",
    "sha256",
    "analysis_population",
    "rewiring_swaps_per_edge",
    "original_edge_count",
    "requested_edge_swaps",
    "completed_edge_swaps",
    "circular_shift",
    "achieved_displacement_fraction",
)


def build_counterfactual_requests(config: CounterfactualStudyConfig) -> dict[str, Path]:
    """Export locked-panel and CATH-training MIF requests without reading outcomes."""

    _require_frozen_protocol(config)
    output = config.paths.run_dir / "mif_requests"
    input_directory = output / "inputs"
    input_directory.mkdir(parents=True, exist_ok=True)

    domains = pd.read_parquet(config.paths.run_dir / "panel" / "domains.parquet")
    residues = pd.read_parquet(config.paths.run_dir / "panel" / "residues.parquet")
    query_rows = pd.read_parquet(config.paths.run_dir / "panel" / "query_rows.parquet")
    query_states = query_rows[["state_id", "domain_id", "sequence"]].drop_duplicates()
    if query_states["domain_id"].duplicated().any() or len(query_states) != len(domains):
        raise ValueError("the locked panel must provide exactly one native state per domain")

    requests: list[dict[str, Any]] = []
    structures: list[dict[str, Any]] = []
    residue_groups = {
        domain_id: frame.sort_values("position").reset_index(drop=True)
        for domain_id, frame in residues.groupby("domain_id", sort=False, observed=True)
    }
    states = query_states.set_index("domain_id")
    for domain_ordinal, domain in enumerate(
        domains.sort_values("domain_id").itertuples(index=False)
    ):
        frame = residue_groups[domain.domain_id]
        coordinates = _coordinates(frame)
        if len(coordinates) != len(domain.sequence):
            raise ValueError(f"coordinate/sequence length mismatch for {domain.domain_id}")
        state = states.loc[domain.domain_id]
        paired_path = input_directory / f"panel__{_safe_name(domain.domain_id)}__paired.npz"
        np.savez_compressed(paired_path, coordinates=coordinates)
        _add_structure_and_request(
            requests,
            structures,
            state_id=str(state.state_id),
            domain_id=domain.domain_id,
            sequence=str(state.sequence),
            role="paired",
            structure_id=f"{domain.domain_id}:paired",
            input_kind="coordinates",
            path=paired_path,
            population="counterfactuals_locked_panel",
        )

        original_edges = contact_edges(
            coordinates[:, 1, :],
            config.counterfactuals.contact_distance_angstrom,
            config.counterfactuals.contact_minimum_sequence_separation,
        )
        original_degrees = degrees(original_edges, len(coordinates))
        for strength_ordinal, strength in enumerate(config.counterfactuals.rewiring_swaps_per_edge):
            requested = int(round(len(original_edges) * strength))
            rng = np.random.default_rng(
                config.seed + 10_000 * (domain_ordinal + 1) + strength_ordinal
            )
            rewired, completed = degree_preserving_rewire(
                original_edges,
                len(coordinates),
                requested,
                config.counterfactuals.rewire_max_attempts_per_swap,
                rng,
            )
            if not np.array_equal(degrees(rewired, len(coordinates)), original_degrees):
                raise ValueError(f"degree preservation failed for {domain.domain_id}/{strength}")
            role = f"contact_rewired_{strength:g}"
            graph_path = input_directory / (
                f"panel__{_safe_name(domain.domain_id)}__{_safe_name(role)}.npz"
            )
            np.savez_compressed(
                graph_path,
                coordinates=coordinates,
                edges=np.asarray(sorted(rewired), dtype=np.int32).reshape(-1, 2),
                length=np.asarray([len(coordinates)], dtype=np.int32),
            )
            _add_structure_and_request(
                requests,
                structures,
                state_id=str(state.state_id),
                domain_id=domain.domain_id,
                sequence=str(state.sequence),
                role=role,
                structure_id=f"{domain.domain_id}:{role}",
                input_kind="contact_graph",
                path=graph_path,
                population="counterfactuals_locked_panel",
                rewiring_swaps_per_edge=float(strength),
                original_edge_count=len(original_edges),
                requested_edge_swaps=requested,
                completed_edge_swaps=completed,
            )

        circular, shift, fraction = _circular_permutation(coordinates)
        circular_path = input_directory / (
            f"panel__{_safe_name(domain.domain_id)}__circular_permuted.npz"
        )
        np.savez_compressed(circular_path, coordinates=circular)
        _add_structure_and_request(
            requests,
            structures,
            state_id=str(state.state_id),
            domain_id=domain.domain_id,
            sequence=str(state.sequence),
            role=config.counterfactuals.replication_role,
            structure_id=f"{domain.domain_id}:{config.counterfactuals.replication_role}",
            input_kind="coordinates",
            path=circular_path,
            population="counterfactuals_locked_panel",
            circular_shift=shift,
            achieved_displacement_fraction=fraction,
        )

    _add_cath_circular_requests(config, input_directory, requests, structures)
    request_table = pd.DataFrame(requests, columns=REQUEST_COLUMNS)
    structure_table = pd.DataFrame(structures, columns=STRUCTURE_COLUMNS)
    _validate_request_tables(request_table, structure_table, len(domains))
    request_path = output / "requests.parquet"
    structure_path = output / "structures.parquet"
    write_parquet(request_path, request_table)
    write_parquet(structure_path, structure_table)
    manifest_path = output / "manifest.json"
    write_json(
        manifest_path,
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "protocol_lock": str(config.paths.run_dir / "protocol_lock.json"),
            "locked_panel_outcomes_read": False,
            "circular_rule": "maximum_feasible_cyclic_displacement",
            "requests": table_manifest(request_path, request_table),
            "structures": table_manifest(structure_path, structure_table),
            "counts": {
                "by_population": {
                    str(key): int(value)
                    for key, value in structure_table["analysis_population"].value_counts().items()
                },
                "by_role": {
                    str(key): int(value)
                    for key, value in structure_table["structure_role"].value_counts().items()
                },
            },
        },
    )
    return {
        "requests": request_path,
        "structures": structure_path,
        "manifest": manifest_path,
    }


def _add_cath_circular_requests(
    config: CounterfactualStudyConfig,
    input_directory: Path,
    requests: list[dict[str, Any]],
    structures: list[dict[str, Any]],
) -> None:
    generalization_requests = pd.read_parquet(
        config.paths.generalization_run / "mif_requests" / "requests.parquet"
    )
    paired = generalization_requests.loc[
        generalization_requests["structure_role"].eq("paired")
    ].copy()
    if paired["domain_id"].duplicated().any():
        raise ValueError(
            "generalization study CATH MIF requests must contain one paired state per domain"
        )
    for row in paired.sort_values("domain_id").itertuples(index=False):
        with np.load(row.input_path) as data:
            coordinates = np.asarray(data["coordinates"], dtype=np.float32)
        circular, shift, fraction = _circular_permutation(coordinates)
        path = input_directory / f"cath__{_safe_name(row.domain_id)}__circular_permuted.npz"
        np.savez_compressed(path, coordinates=circular)
        _add_structure_and_request(
            requests,
            structures,
            state_id=str(row.state_id),
            domain_id=str(row.domain_id),
            sequence=str(row.state_sequence),
            role="circular_permuted",
            structure_id=f"cath:{row.domain_id}:circular_permuted",
            input_kind="coordinates",
            path=path,
            population="generalization_cath_training",
            circular_shift=shift,
            achieved_displacement_fraction=fraction,
        )


def _coordinates(residues: pd.DataFrame) -> np.ndarray:
    return np.stack(
        [
            residues[[f"{atom}_{axis}" for axis in "xyz"]].to_numpy(dtype=np.float32)
            for atom in ("n", "ca", "c", "o")
        ],
        axis=1,
    )


def _circular_permutation(coordinates: np.ndarray) -> tuple[np.ndarray, int, float]:
    length = len(coordinates)
    if length < 2:
        raise ValueError("circular counterfactual requires at least two residues")
    shift = length // 2
    fraction = min(shift, length - shift) / length
    return np.roll(coordinates, shift, axis=0), shift, fraction


def _add_structure_and_request(
    requests: list[dict[str, Any]],
    structures: list[dict[str, Any]],
    *,
    state_id: str,
    domain_id: str,
    sequence: str,
    role: str,
    structure_id: str,
    input_kind: str,
    path: Path,
    population: str,
    rewiring_swaps_per_edge: float = np.nan,
    original_edge_count: int = 0,
    requested_edge_swaps: int = 0,
    completed_edge_swaps: int = 0,
    circular_shift: int = 0,
    achieved_displacement_fraction: float = np.nan,
) -> None:
    structures.append(
        {
            "structure_role": role,
            "structure_id": structure_id,
            "target_domain_id": domain_id,
            "input_kind": input_kind,
            "input_path": str(path),
            "sha256": sha256_file(path),
            "analysis_population": population,
            "rewiring_swaps_per_edge": rewiring_swaps_per_edge,
            "original_edge_count": original_edge_count,
            "requested_edge_swaps": requested_edge_swaps,
            "completed_edge_swaps": completed_edge_swaps,
            "circular_shift": circular_shift,
            "achieved_displacement_fraction": achieved_displacement_fraction,
        }
    )
    requests.append(
        {
            "request_id": f"{state_id}|{role}|{structure_id}",
            "state_id": state_id,
            "domain_id": domain_id,
            "state_sequence": sequence,
            "structure_role": role,
            "structure_id": structure_id,
            "input_kind": input_kind,
            "input_path": str(path),
            "length": len(sequence),
        }
    )


def _validate_request_tables(
    requests: pd.DataFrame,
    structures: pd.DataFrame,
    locked_domain_count: int,
) -> None:
    if requests["request_id"].duplicated().any():
        raise ValueError("counterfactual study MIF request IDs must be unique")
    if structures["structure_id"].duplicated().any():
        raise ValueError("counterfactual study structure IDs must be unique")
    joined = requests.merge(
        structures[["structure_id", "input_path", "input_kind"]],
        on=["structure_id", "input_path", "input_kind"],
        validate="one_to_one",
    )
    if len(joined) != len(requests):
        raise ValueError("MIF requests and structures do not align one-to-one")
    locked = structures.loc[structures["analysis_population"].eq("counterfactuals_locked_panel")]
    role_counts = locked["structure_role"].value_counts()
    expected_roles = {"paired", "circular_permuted"} | {
        f"contact_rewired_{value:g}" for value in (0.5, 1.0, 2.0, 5.0)
    }
    if set(role_counts.index) != expected_roles or not role_counts.eq(locked_domain_count).all():
        raise ValueError("locked-panel request roles do not cover every domain exactly once")
    circular = structures.loc[
        structures["structure_role"].eq("circular_permuted"),
        ["structure_id", "achieved_displacement_fraction"],
    ].merge(requests[["structure_id", "length"]], on="structure_id", validate="one_to_one")
    feasible = (circular["length"] // 2) / circular["length"]
    if not np.allclose(circular["achieved_displacement_fraction"], feasible):
        raise ValueError("circular counterfactual did not use maximum feasible displacement")


def _require_frozen_protocol(config: CounterfactualStudyConfig) -> None:
    lock = read_json(config.paths.run_dir / "protocol_lock.json")
    if lock.get("status") != "FROZEN_BEFORE_LOCKED_PANEL_MODEL_SCORING":
        raise RuntimeError(
            "counterfactual study protocol must be frozen before counterfactual generation"
        )


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value)
