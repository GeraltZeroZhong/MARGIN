"""Build the frozen mechanism study paired and counterfactual MIF request set."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

from margin.constants import BACKBONE_ATOMS
from margin.decoys.graph import contact_edges, degree_preserving_rewire, degrees
from margin.provenance import (
    read_json,
    runtime_manifest,
    sha256_file,
    table_manifest,
    write_json,
    write_parquet,
)
from margin.studies.mechanisms.config import MechanismStudyConfig
from margin.teachers.requests import REQUEST_COLUMNS

STRUCTURE_COLUMNS = (
    "structure_role",
    "counterfactual_family",
    "condition_level",
    "seed_index",
    "structure_id",
    "target_domain_id",
    "input_kind",
    "input_path",
    "sha256",
    "analysis_population",
    "native_edge_count",
    "requested_edge_count",
    "completed_edge_count",
    "requested_swap_count",
    "completed_swap_count",
    "target_ca_rmsd_angstrom",
    "achieved_ca_rmsd_angstrom",
    "maximum_adjacent_ca_distance_change_angstrom",
    "maximum_peptide_cn_distance_change_angstrom",
    "donor_domain_id",
    "donor_source_length",
    "donor_crop_start",
)


def build_mechanism_counterfactuals(config: MechanismStudyConfig) -> dict[str, Path]:
    """Export all frozen requests without reading mechanism study stability outcomes."""

    _require_frozen_protocol(config)
    output = config.paths.run_dir / "mif_requests"
    inputs = output / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    domains = pd.read_parquet(config.paths.run_dir / "panel" / "domains.parquet")
    residues = pd.read_parquet(config.paths.run_dir / "panel" / "residues.parquet")
    queries = pd.read_parquet(config.paths.run_dir / "panel" / "query_rows.parquet")
    matched = pd.read_parquet(config.paths.run_dir / "panel" / "matched_real_decoys.parquet")
    states = queries[["state_id", "domain_id", "sequence"]].drop_duplicates().set_index("domain_id")
    residue_groups = {
        str(domain_id): frame.sort_values("position").reset_index(drop=True)
        for domain_id, frame in residues.groupby("domain_id", sort=False, observed=True)
    }
    match_groups = {
        str(domain_id): frame.sort_values("decoy_ordinal")
        for domain_id, frame in matched.groupby("target_domain_id", observed=True)
    }
    requests: list[dict[str, Any]] = []
    structures: list[dict[str, Any]] = []
    for domain_ordinal, domain in enumerate(
        domains.sort_values("domain_id").itertuples(index=False)
    ):
        domain_id = str(domain.domain_id)
        sequence = str(domain.sequence)
        state = states.loc[domain_id]
        frame = residue_groups[domain_id]
        coordinates = _coordinates(frame)
        secondary_structure = frame["secondary_structure"].astype(str).to_numpy()
        native_edges = contact_edges(
            coordinates[:, 1, :],
            config.counterfactuals.contact_distance_angstrom,
            config.counterfactuals.contact_minimum_sequence_separation,
        )
        if not native_edges:
            raise ValueError(f"no nonlocal contacts for {domain_id}")

        paired_path = inputs / f"{_safe_name(domain_id)}__paired.npz"
        np.savez_compressed(paired_path, coordinates=coordinates)
        _add_request(
            requests,
            structures,
            state_id=str(state.state_id),
            domain_id=domain_id,
            sequence=sequence,
            role="paired",
            family="paired",
            level="native",
            seed_index=-1,
            input_kind="coordinates",
            path=paired_path,
            native_edge_count=len(native_edges),
        )

        rigid_rng = np.random.default_rng(config.seed + 1_000_000 + domain_ordinal)
        rigid_coordinates = _rigid_transform(coordinates, rigid_rng)
        rigid_path = inputs / f"{_safe_name(domain_id)}__rigid_transform_qc.npz"
        np.savez_compressed(rigid_path, coordinates=rigid_coordinates)
        _add_request(
            requests,
            structures,
            state_id=str(state.state_id),
            domain_id=domain_id,
            sequence=sequence,
            role="rigid_transform_qc",
            family="geometry_invariance_qc",
            level="rigid",
            seed_index=0,
            input_kind="coordinates",
            path=rigid_path,
            native_edge_count=len(native_edges),
        )

        legacy_rng = np.random.default_rng(config.seed + 2_000_000 + domain_ordinal)
        legacy_requested = int(
            round(len(native_edges) * config.counterfactuals.legacy_rewiring_swaps_per_edge)
        )
        legacy_edges, legacy_completed = degree_preserving_rewire(
            native_edges,
            len(coordinates),
            legacy_requested,
            config.counterfactuals.legacy_rewire_max_attempts_per_swap,
            legacy_rng,
        )
        legacy_path = inputs / f"{_safe_name(domain_id)}__legacy_rewired_5.npz"
        np.savez_compressed(
            legacy_path,
            coordinates=coordinates,
            edges=_edge_array(legacy_edges),
            length=np.asarray([len(coordinates)], dtype=np.int32),
        )
        _add_request(
            requests,
            structures,
            state_id=str(state.state_id),
            domain_id=domain_id,
            sequence=sequence,
            role="legacy_rewired_5",
            family="legacy_ood_rewiring",
            level="5_swaps_per_edge",
            seed_index=0,
            input_kind="contact_graph",
            path=legacy_path,
            native_edge_count=len(native_edges),
            requested_swap_count=legacy_requested,
            completed_swap_count=legacy_completed,
        )

        for level_ordinal, fraction in enumerate(config.counterfactuals.contact_deletion_fractions):
            requested = max(1, int(round(len(native_edges) * fraction)))
            for seed_index, seed in enumerate(config.counterfactuals.seeds):
                rng = np.random.default_rng(
                    config.seed + 3_000_000 + domain_ordinal * 10_000 + level_ordinal * 100 + seed
                )
                removed = _sample_edges(native_edges, requested, rng)
                role = f"contact_deletion_{fraction:g}_seed_{seed_index}"
                path = inputs / f"{_safe_name(domain_id)}__{role}.npz"
                np.savez_compressed(
                    path,
                    coordinates=coordinates,
                    removed_edges=_edge_array(removed),
                )
                _add_request(
                    requests,
                    structures,
                    state_id=str(state.state_id),
                    domain_id=domain_id,
                    sequence=sequence,
                    role=role,
                    family="contact_deletion",
                    level=f"{fraction:g}",
                    seed_index=seed_index,
                    input_kind="contact_deletion",
                    path=path,
                    native_edge_count=len(native_edges),
                    requested_edge_count=requested,
                    completed_edge_count=len(removed),
                )

        for level_ordinal, target_rmsd in enumerate(
            config.counterfactuals.coordinate_rmsd_angstrom
        ):
            for seed_index, seed in enumerate(config.counterfactuals.seeds):
                rng = np.random.default_rng(
                    config.seed + 4_000_000 + domain_ordinal * 10_000 + level_ordinal * 100 + seed
                )
                perturbed, diagnostics = smooth_coordinate_perturbation(
                    coordinates,
                    target_rmsd=float(target_rmsd),
                    modes=config.counterfactuals.coordinate_low_frequency_modes,
                    maximum_adjacent_change=(
                        config.counterfactuals.maximum_adjacent_ca_distance_change_angstrom
                    ),
                    rng=rng,
                )
                role = f"smooth_coordinate_{target_rmsd:g}_seed_{seed_index}"
                path = inputs / f"{_safe_name(domain_id)}__{role}.npz"
                np.savez_compressed(path, coordinates=perturbed)
                _add_request(
                    requests,
                    structures,
                    state_id=str(state.state_id),
                    domain_id=domain_id,
                    sequence=sequence,
                    role=role,
                    family="smooth_coordinate",
                    level=f"{target_rmsd:g}",
                    seed_index=seed_index,
                    input_kind="coordinates",
                    path=path,
                    native_edge_count=len(native_edges),
                    target_ca_rmsd_angstrom=float(target_rmsd),
                    achieved_ca_rmsd_angstrom=diagnostics["achieved_ca_rmsd_angstrom"],
                    maximum_adjacent_ca_distance_change_angstrom=diagnostics[
                        "maximum_adjacent_ca_distance_change_angstrom"
                    ],
                    maximum_peptide_cn_distance_change_angstrom=diagnostics[
                        "maximum_peptide_cn_distance_change_angstrom"
                    ],
                )

        requested_swaps = max(
            1,
            int(
                np.ceil(
                    len(native_edges)
                    * config.counterfactuals.constrained_reassignment_fraction
                    / 2.0
                )
            ),
        )
        for seed_index, seed in enumerate(config.counterfactuals.seeds):
            rng = np.random.default_rng(config.seed + 5_000_000 + domain_ordinal * 10_000 + seed)
            reassignment = constrained_contact_reassignment(
                native_edges,
                coordinates[:, 1, :],
                secondary_structure,
                requested_swaps=requested_swaps,
                maximum_attempts=(
                    requested_swaps * config.counterfactuals.constrained_max_attempts_per_swap
                ),
                rng=rng,
            )
            role = f"constrained_reassignment_0.1_seed_{seed_index}"
            path = inputs / f"{_safe_name(domain_id)}__{role}.npz"
            np.savez_compressed(
                path,
                coordinates=coordinates,
                removed_edges=_edge_array(reassignment["removed_edges"]),
                added_edges=_edge_array(reassignment["added_edges"]),
                source_edges=_edge_array(reassignment["source_edges"]),
            )
            _add_request(
                requests,
                structures,
                state_id=str(state.state_id),
                domain_id=domain_id,
                sequence=sequence,
                role=role,
                family="constrained_reassignment",
                level="0.1",
                seed_index=seed_index,
                input_kind="contact_reassignment",
                path=path,
                native_edge_count=len(native_edges),
                requested_edge_count=2 * requested_swaps,
                completed_edge_count=len(reassignment["removed_edges"]),
                requested_swap_count=requested_swaps,
                completed_swap_count=int(reassignment["completed_swaps"]),
            )

        for match in match_groups[domain_id].itertuples(index=False):
            donor_coordinates = _pdb_coordinates(
                Path(match.donor_structure_path), str(match.donor_chain_id)
            )
            start = int(match.crop_start)
            donor_coordinates = donor_coordinates[start : start + len(sequence)]
            if len(donor_coordinates) != len(sequence):
                raise ValueError(f"matched donor crop length mismatch for {domain_id}")
            role = f"matched_real_seed_{int(match.decoy_ordinal)}"
            path = inputs / f"{_safe_name(domain_id)}__{role}.npz"
            np.savez_compressed(path, coordinates=donor_coordinates)
            _add_request(
                requests,
                structures,
                state_id=str(state.state_id),
                domain_id=domain_id,
                sequence=sequence,
                role=role,
                family="matched_real_structure",
                level="descriptor_matched",
                seed_index=int(match.decoy_ordinal),
                input_kind="coordinates",
                path=path,
                native_edge_count=len(native_edges),
                donor_domain_id=str(match.donor_domain_id),
                donor_source_length=int(match.source_length),
                donor_crop_start=start,
            )

    request_table = pd.DataFrame(requests, columns=REQUEST_COLUMNS)
    structure_table = pd.DataFrame(structures, columns=STRUCTURE_COLUMNS)
    _validate_requests(request_table, structure_table, domains, config)
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
            "requests": table_manifest(request_path, request_table),
            "structures": table_manifest(structure_path, structure_table),
            "counts_by_family": {
                str(key): int(value)
                for key, value in structure_table["counterfactual_family"].value_counts().items()
            },
            "counts_by_input_kind": {
                str(key): int(value)
                for key, value in structure_table["input_kind"].value_counts().items()
            },
        },
    )
    return {"requests": request_path, "structures": structure_path, "manifest": manifest_path}


def smooth_coordinate_perturbation(
    coordinates: np.ndarray,
    *,
    target_rmsd: float,
    modes: int,
    maximum_adjacent_change: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply a low-frequency per-residue translation with explicit chain-continuity bounds."""

    length = len(coordinates)
    x = np.linspace(0.0, 1.0, length)
    displacement = np.zeros((length, 3), dtype=float)
    for mode in range(1, modes + 1):
        coefficients = rng.normal(size=(2, 3)) / mode
        displacement += np.sin(np.pi * mode * x)[:, None] * coefficients[0]
        displacement += np.cos(np.pi * mode * x)[:, None] * coefficients[1]
    displacement -= displacement.mean(axis=0, keepdims=True)
    rms = float(np.sqrt(np.mean(np.sum(displacement**2, axis=1))))
    if rms == 0:
        raise ValueError("degenerate smooth coordinate displacement")
    displacement *= target_rmsd / rms
    native_ca = coordinates[:, 1, :].astype(float)

    def adjacent_change(scale: float) -> float:
        perturbed_ca = native_ca + displacement * scale
        native_distance = np.linalg.norm(np.diff(native_ca, axis=0), axis=1)
        perturbed_distance = np.linalg.norm(np.diff(perturbed_ca, axis=0), axis=1)
        return float(np.max(np.abs(perturbed_distance - native_distance)))

    scale = 1.0
    if adjacent_change(scale) > maximum_adjacent_change:
        low, high = 0.0, 1.0
        for _ in range(40):
            middle = (low + high) / 2.0
            if adjacent_change(middle) <= maximum_adjacent_change:
                low = middle
            else:
                high = middle
        scale = low
    displacement *= scale
    perturbed = coordinates.astype(float) + displacement[:, None, :]
    native_cn = np.linalg.norm(coordinates[:-1, 2, :] - coordinates[1:, 0, :], axis=1)
    perturbed_cn = np.linalg.norm(perturbed[:-1, 2, :] - perturbed[1:, 0, :], axis=1)
    diagnostics = {
        "achieved_ca_rmsd_angstrom": float(np.sqrt(np.mean(np.sum(displacement**2, axis=1)))),
        "maximum_adjacent_ca_distance_change_angstrom": adjacent_change(1.0),
        "maximum_peptide_cn_distance_change_angstrom": float(
            np.max(np.abs(perturbed_cn - native_cn))
        ),
    }
    return perturbed.astype(np.float32), diagnostics


def constrained_contact_reassignment(
    native_edges: set[tuple[int, int]],
    ca_coordinates: np.ndarray,
    secondary_structure: np.ndarray,
    *,
    requested_swaps: int,
    maximum_attempts: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Degree-preserving swaps matched on local topology and transferred edge geometry."""

    native = {_ordered(edge) for edge in native_edges}
    current = set(native)
    available = set(native)
    removed: list[tuple[int, int]] = []
    added: list[tuple[int, int]] = []
    sources: list[tuple[int, int]] = []
    completed = 0
    attempts = 0
    while completed < requested_swaps and attempts < maximum_attempts and len(available) >= 2:
        attempts += 1
        edge_list = tuple(sorted(available))
        first_index, second_index = rng.choice(len(edge_list), size=2, replace=False)
        first = edge_list[int(first_index)]
        second = edge_list[int(second_index)]
        a, b = first
        c, d = second
        if len({a, b, c, d}) < 4:
            continue
        proposals = [
            (_ordered((a, d)), _ordered((c, b))),
            (_ordered((a, c)), _ordered((b, d))),
        ]
        if rng.random() < 0.5:
            proposals.reverse()
        accepted: tuple[tuple[int, int], tuple[int, int]] | None = None
        mapping: tuple[tuple[int, int], tuple[int, int]] | None = None
        for proposal in proposals:
            if any(left == right for left, right in proposal):
                continue
            if len(set(proposal)) != 2 or set(proposal) & current or set(proposal) & native:
                continue
            for source_order in ((first, second), (second, first)):
                if all(
                    _topology_signature(new, len(ca_coordinates), secondary_structure)
                    == _topology_signature(source, len(ca_coordinates), secondary_structure)
                    for new, source in zip(proposal, source_order, strict=True)
                ):
                    accepted = proposal
                    mapping = source_order
                    break
            if accepted is not None:
                break
        if accepted is None or mapping is None:
            continue
        current.remove(first)
        current.remove(second)
        available.remove(first)
        available.remove(second)
        current.update(accepted)
        removed.extend([first, second])
        added.extend(accepted)
        sources.extend(mapping)
        completed += 1
    if not np.array_equal(
        degrees(current, len(ca_coordinates)), degrees(native, len(ca_coordinates))
    ):
        raise ValueError("constrained reassignment failed to preserve degree")
    for source, new in zip(sources, added, strict=True):
        source_distance = float(
            np.linalg.norm(ca_coordinates[source[0]] - ca_coordinates[source[1]])
        )
        if _distance_bin(source_distance) not in {"under_6", "6_to_7", "7_to_8"}:
            raise ValueError("source contact has an unexpected distance bin")
        if _topology_signature(source, len(ca_coordinates), secondary_structure) != (
            _topology_signature(new, len(ca_coordinates), secondary_structure)
        ):
            raise ValueError("constrained reassignment signature mismatch")
    return {
        "removed_edges": removed,
        "added_edges": added,
        "source_edges": sources,
        "completed_swaps": completed,
        "attempts": attempts,
    }


def _topology_signature(
    edge: tuple[int, int], length: int, secondary_structure: np.ndarray
) -> tuple[str, str, str]:
    left, right = edge
    separation = abs(left - right)
    if separation <= 5:
        separation_bin = "3_to_5"
    elif separation <= 11:
        separation_bin = "6_to_11"
    elif separation <= 23:
        separation_bin = "12_to_23"
    else:
        separation_bin = "24_plus"
    normalized = separation / length
    if normalized <= 0.10:
        contact_order_bin = "local"
    elif normalized <= 0.25:
        contact_order_bin = "mesoscale"
    else:
        contact_order_bin = "long_range"
    pair = "__".join(sorted((str(secondary_structure[left]), str(secondary_structure[right]))))
    return separation_bin, pair, contact_order_bin


def _distance_bin(distance: float) -> str:
    if distance < 6.0:
        return "under_6"
    if distance < 7.0:
        return "6_to_7"
    if distance <= 8.0:
        return "7_to_8"
    return "noncontact"


def _sample_edges(
    edges: set[tuple[int, int]], count: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    ordered = tuple(sorted(edges))
    indices = rng.choice(len(ordered), size=min(count, len(ordered)), replace=False)
    return [ordered[int(index)] for index in np.sort(indices)]


def _rigid_transform(coordinates: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    matrix, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(matrix) < 0:
        matrix[:, 0] *= -1
    center = coordinates[:, 1, :].mean(axis=0)
    transformed = (coordinates - center) @ matrix + np.asarray([11.0, -7.0, 3.0])
    return transformed.astype(np.float32)


def _coordinates(residues: pd.DataFrame) -> np.ndarray:
    return np.stack(
        [
            residues[[f"{atom}_{axis}" for axis in "xyz"]].to_numpy(dtype=np.float32)
            for atom in ("n", "ca", "c", "o")
        ],
        axis=1,
    )


def _pdb_coordinates(path: Path, requested_chain: str) -> np.ndarray:
    model = next(PDBParser(QUIET=True).get_structure(path.stem, str(path)).get_models())
    chains = list(model.get_chains())
    chain = next((item for item in chains if item.id == requested_chain), None)
    if chain is None and len(chains) == 1:
        chain = chains[0]
    if chain is None:
        raise ValueError(f"donor chain {requested_chain!r} unavailable in {path}")
    rows = []
    for residue in chain.get_residues():
        if residue.id[0].strip():
            continue
        if all(atom in residue for atom in BACKBONE_ATOMS):
            rows.append([residue[atom].coord.astype(np.float32) for atom in BACKBONE_ATOMS])
    if not rows:
        raise ValueError(f"no complete donor backbone in {path}")
    return np.asarray(rows, dtype=np.float32)


def _add_request(
    requests: list[dict[str, Any]],
    structures: list[dict[str, Any]],
    *,
    state_id: str,
    domain_id: str,
    sequence: str,
    role: str,
    family: str,
    level: str,
    seed_index: int,
    input_kind: str,
    path: Path,
    native_edge_count: int,
    requested_edge_count: int = 0,
    completed_edge_count: int = 0,
    requested_swap_count: int = 0,
    completed_swap_count: int = 0,
    target_ca_rmsd_angstrom: float = np.nan,
    achieved_ca_rmsd_angstrom: float = np.nan,
    maximum_adjacent_ca_distance_change_angstrom: float = np.nan,
    maximum_peptide_cn_distance_change_angstrom: float = np.nan,
    donor_domain_id: str = "",
    donor_source_length: int = 0,
    donor_crop_start: int = 0,
) -> None:
    structure_id = f"{domain_id}:{role}"
    structures.append(
        {
            "structure_role": role,
            "counterfactual_family": family,
            "condition_level": level,
            "seed_index": seed_index,
            "structure_id": structure_id,
            "target_domain_id": domain_id,
            "input_kind": input_kind,
            "input_path": str(path),
            "sha256": sha256_file(path),
            "analysis_population": "mechanisms_locked_dense_panel",
            "native_edge_count": native_edge_count,
            "requested_edge_count": requested_edge_count,
            "completed_edge_count": completed_edge_count,
            "requested_swap_count": requested_swap_count,
            "completed_swap_count": completed_swap_count,
            "target_ca_rmsd_angstrom": target_ca_rmsd_angstrom,
            "achieved_ca_rmsd_angstrom": achieved_ca_rmsd_angstrom,
            "maximum_adjacent_ca_distance_change_angstrom": (
                maximum_adjacent_ca_distance_change_angstrom
            ),
            "maximum_peptide_cn_distance_change_angstrom": (
                maximum_peptide_cn_distance_change_angstrom
            ),
            "donor_domain_id": donor_domain_id,
            "donor_source_length": donor_source_length,
            "donor_crop_start": donor_crop_start,
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


def _validate_requests(
    requests: pd.DataFrame,
    structures: pd.DataFrame,
    domains: pd.DataFrame,
    config: MechanismStudyConfig,
) -> None:
    if requests["request_id"].duplicated().any() or structures["structure_id"].duplicated().any():
        raise ValueError("mechanism study request and structure identifiers must be unique")
    expected_per_domain = (
        3
        + len(config.counterfactuals.contact_deletion_fractions) * len(config.counterfactuals.seeds)
        + len(config.counterfactuals.coordinate_rmsd_angstrom) * len(config.counterfactuals.seeds)
        + len(config.counterfactuals.seeds)
        + config.panel.matched_real_decoys
    )
    counts = structures.groupby("target_domain_id", observed=True).size()
    if not counts.eq(expected_per_domain).all() or len(counts) != len(domains):
        raise ValueError("mechanism study request families do not cover each domain exactly")
    smooth = structures.loc[structures["counterfactual_family"].eq("smooth_coordinate")]
    if (
        smooth["maximum_adjacent_ca_distance_change_angstrom"]
        > config.counterfactuals.maximum_adjacent_ca_distance_change_angstrom + 1e-6
    ).any():
        raise ValueError("smooth coordinate continuity bound was exceeded")
    deletion = structures.loc[structures["counterfactual_family"].eq("contact_deletion")]
    if not deletion["requested_edge_count"].eq(deletion["completed_edge_count"]).all():
        raise ValueError("contact deletion did not remove its requested edge count")
    legacy = structures.loc[structures["counterfactual_family"].eq("legacy_ood_rewiring")]
    if not legacy["requested_swap_count"].eq(legacy["completed_swap_count"]).all():
        raise ValueError("legacy rewiring did not complete the frozen swap count")
    reassignment = structures.loc[
        structures["counterfactual_family"].eq("constrained_reassignment")
    ]
    if (reassignment["completed_swap_count"] <= 0).any():
        raise ValueError("constrained reassignment failed to complete any swap")


def _edge_array(edges: Any) -> np.ndarray:
    return np.asarray(sorted(edges), dtype=np.int32).reshape(-1, 2)


def _ordered(edge: tuple[int, int]) -> tuple[int, int]:
    left, right = edge
    return (left, right) if left < right else (right, left)


def _require_frozen_protocol(config: MechanismStudyConfig) -> None:
    lock = read_json(config.paths.run_dir / "protocol_lock.json")
    if lock.get("status") != "FROZEN_BEFORE_MECHANISMS_MODEL_SCORING":
        raise RuntimeError("mechanism study protocol must be frozen before request generation")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value)
