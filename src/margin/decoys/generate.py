"""Generate matched CATH, frame-permuted, rewired, and shuffled controls."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from margin.config import ProjectConfig
from margin.data_registry.registry import RegistryTables
from margin.data_registry.schema import coordinate_columns
from margin.decoys.graph import contact_edges, degree_preserving_rewire
from margin.provenance import runtime_manifest, table_manifest, write_json, write_parquet

DECOY_COLUMNS = (
    "decoy_id",
    "target_domain_id",
    "decoy_type",
    "source_domain_id",
    "target_length",
    "source_length",
    "target_cath_h",
    "source_cath_h",
    "helix_fraction_difference",
    "strand_fraction_difference",
    "supports_coordinate_teacher",
    "supports_graph_teacher",
    "preserved_statistics",
    "mapping",
    "requested_edge_swaps",
    "completed_edge_swaps",
)

DECOY_RESIDUE_COLUMNS = (
    "decoy_id",
    "target_domain_id",
    "position",
    "source_domain_id",
    "source_position",
    "dssp",
    "secondary_structure",
    "rsa",
    "contact_degree",
    *coordinate_columns(),
)

EDGE_COLUMNS = ("decoy_id", "target_domain_id", "source", "target")


@dataclass(frozen=True)
class DecoyArtifacts:
    decoys: pd.DataFrame
    residues: pd.DataFrame
    edges: pd.DataFrame
    skipped: pd.DataFrame


def build_decoys(
    registry: RegistryTables,
    config: ProjectConfig,
    target_domain_ids: set[str] | None = None,
    source_domain_ids: set[str] | None = None,
) -> DecoyArtifacts:
    rng = np.random.default_rng(config.seed + 1)
    domains = registry.domains.sort_values("domain_id").reset_index(drop=True)
    targets = (
        domains
        if target_domain_ids is None
        else domains.loc[domains["domain_id"].isin(target_domain_ids)]
    )
    sources = (
        domains
        if source_domain_ids is None
        else domains.loc[domains["domain_id"].isin(source_domain_ids)]
    )
    if targets.empty or sources.empty:
        raise ValueError("decoy generation requires non-empty target and matched-source pools")
    residue_groups = {
        domain_id: frame.sort_values("position").reset_index(drop=True)
        for domain_id, frame in registry.residues.groupby("domain_id", sort=False, observed=True)
    }
    decoy_rows: list[dict[str, Any]] = []
    residue_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for target in targets.itertuples(index=False):
        target_residues = residue_groups[target.domain_id]
        matched = _matched_candidates(target, sources, config)
        if len(matched) < config.decoys.matched_decoys_per_domain:
            skipped.append(
                {
                    "target_domain_id": target.domain_id,
                    "decoy_type": "matched_cath",
                    "reason": "no_candidate_satisfies_declared_constraints",
                }
            )
        for index, source in enumerate(
            matched.head(config.decoys.matched_decoys_per_domain).itertuples(index=False)
        ):
            decoy_id = f"{target.domain_id}:matched_cath:{index:03d}"
            mapping = _length_mapping(int(target.length), int(source.length))
            decoy_rows.append(
                _decoy_row(
                    decoy_id,
                    target,
                    "matched_cath",
                    source,
                    mapping,
                    supports_coordinate=True,
                    supports_graph=True,
                    preserved=["length", "secondary_structure_composition_within_tolerance"],
                )
            )
            residue_rows.extend(
                _mapped_residues(
                    decoy_id,
                    target.domain_id,
                    source.domain_id,
                    residue_groups[source.domain_id],
                    mapping,
                )
            )

        circular_mapping = _circular_mapping(
            int(target.length), config.decoys.permutation_minimum_displacement_fraction, rng
        )
        circular_id = f"{target.domain_id}:permuted:000"
        decoy_rows.append(
            _decoy_row(
                circular_id,
                target,
                "permuted",
                target,
                circular_mapping,
                supports_coordinate=True,
                supports_graph=True,
                preserved=[
                    "backbone_atom_inventory",
                    "pairwise_distance_multiset",
                    "all_pair_contact_degree_multiset",
                ],
            )
        )
        residue_rows.extend(
            _mapped_residues(
                circular_id,
                target.domain_id,
                target.domain_id,
                target_residues,
                circular_mapping,
            )
        )

        shuffled_mapping = rng.permutation(int(target.length)).astype(int).tolist()
        shuffled_id = f"{target.domain_id}:shuffled_residue:000"
        decoy_rows.append(
            _decoy_row(
                shuffled_id,
                target,
                "shuffled_residue",
                target,
                shuffled_mapping,
                supports_coordinate=True,
                supports_graph=True,
                preserved=[
                    "backbone_atom_inventory",
                    "pairwise_distance_multiset",
                    "all_pair_contact_degree_multiset",
                ],
            )
        )
        residue_rows.extend(
            _mapped_residues(
                shuffled_id,
                target.domain_id,
                target.domain_id,
                target_residues,
                shuffled_mapping,
            )
        )

        ca = target_residues[[f"ca_{axis}" for axis in "xyz"]].to_numpy(dtype=float)
        paired_edges = contact_edges(
            ca,
            config.registry.contact_distance_angstrom,
            config.registry.contact_minimum_sequence_separation,
        )
        requested = int(round(len(paired_edges) * config.decoys.contact_rewire_swaps_per_edge))
        rewired, completed = degree_preserving_rewire(
            paired_edges,
            int(target.length),
            requested,
            config.decoys.contact_rewire_max_attempts_per_swap,
            rng,
        )
        rewired_id = f"{target.domain_id}:contact_rewired:000"
        decoy_rows.append(
            _decoy_row(
                rewired_id,
                target,
                "contact_rewired",
                target,
                None,
                supports_coordinate=False,
                supports_graph=True,
                preserved=["node_count", "edge_count", "node_degree_sequence"],
                requested_swaps=requested,
                completed_swaps=completed,
            )
        )
        edge_rows.extend(
            {
                "decoy_id": rewired_id,
                "target_domain_id": target.domain_id,
                "source": left,
                "target": right,
            }
            for left, right in sorted(rewired)
        )

    artifacts = DecoyArtifacts(
        decoys=pd.DataFrame(decoy_rows, columns=DECOY_COLUMNS),
        residues=pd.DataFrame(residue_rows, columns=DECOY_RESIDUE_COLUMNS),
        edges=pd.DataFrame(edge_rows, columns=EDGE_COLUMNS),
        skipped=pd.DataFrame(skipped, columns=["target_domain_id", "decoy_type", "reason"]),
    )
    validate_decoys(artifacts)
    return artifacts


def write_decoys(
    directory: Path, artifacts: DecoyArtifacts, config: ProjectConfig
) -> dict[str, Any]:
    validate_decoys(artifacts)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "decoys": directory / "decoys.parquet",
        "residues": directory / "decoy_residues.parquet",
        "edges": directory / "contact_rewired_edges.parquet",
        "skipped": directory / "skipped_decoys.parquet",
    }
    write_parquet(paths["decoys"], artifacts.decoys)
    write_parquet(paths["residues"], artifacts.residues)
    write_parquet(paths["edges"], artifacts.edges)
    write_parquet(paths["skipped"], artifacts.skipped)
    manifest = {
        **runtime_manifest(config.paths.project_root),
        "schema_version": config.schema_version,
        "seed": config.seed + 1,
        "parameters": config.decoys.model_dump(mode="json"),
        "statistics": {
            "counts_by_type": {
                str(key): int(value)
                for key, value in artifacts.decoys["decoy_type"].value_counts().items()
            },
            "requested_edge_swaps": int(artifacts.decoys["requested_edge_swaps"].sum()),
            "completed_edge_swaps": int(artifacts.decoys["completed_edge_swaps"].sum()),
            "coordinate_decoys": int(artifacts.decoys["supports_coordinate_teacher"].sum()),
            "graph_decoys": int(artifacts.decoys["supports_graph_teacher"].sum()),
        },
        "decoys": table_manifest(paths["decoys"], artifacts.decoys),
        "residues": table_manifest(paths["residues"], artifacts.residues),
        "edges": table_manifest(paths["edges"], artifacts.edges),
        "skipped": table_manifest(paths["skipped"], artifacts.skipped),
    }
    write_json(directory / "manifest.json", manifest)
    return manifest


def load_decoys(directory: Path) -> DecoyArtifacts:
    artifacts = DecoyArtifacts(
        decoys=pd.read_parquet(directory / "decoys.parquet"),
        residues=pd.read_parquet(directory / "decoy_residues.parquet"),
        edges=pd.read_parquet(directory / "contact_rewired_edges.parquet"),
        skipped=pd.read_parquet(directory / "skipped_decoys.parquet"),
    )
    validate_decoys(artifacts)
    return artifacts


def validate_decoys(artifacts: DecoyArtifacts) -> None:
    if set(DECOY_COLUMNS) - set(artifacts.decoys.columns):
        raise ValueError("decoy table schema mismatch")
    if set(DECOY_RESIDUE_COLUMNS) - set(artifacts.residues.columns):
        raise ValueError("decoy residue table schema mismatch")
    if set(EDGE_COLUMNS) - set(artifacts.edges.columns):
        raise ValueError("decoy edge table schema mismatch")
    if artifacts.decoys["decoy_id"].duplicated().any():
        raise ValueError("decoy_id values must be unique")
    coordinate_ids = set(
        artifacts.decoys.loc[artifacts.decoys["supports_coordinate_teacher"], "decoy_id"]
    )
    residue_counts = artifacts.residues.groupby("decoy_id", observed=True).size().to_dict()
    expected_lengths = artifacts.decoys.set_index("decoy_id")["target_length"].astype(int).to_dict()
    if any(
        residue_counts.get(decoy_id, 0) != expected_lengths[decoy_id] for decoy_id in coordinate_ids
    ):
        raise ValueError("coordinate decoys must have one residue row per target position")
    if set(artifacts.residues["decoy_id"]) - coordinate_ids:
        raise ValueError("non-coordinate decoys must not contain fabricated coordinates")


def _matched_candidates(target: Any, domains: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    candidates = domains.loc[domains["domain_id"] != target.domain_id].copy()
    if config.decoys.require_exact_length:
        candidates = candidates.loc[candidates["length"] == target.length]
    else:
        candidates = candidates.loc[
            (candidates["length"] - target.length).abs() <= config.decoys.maximum_length_difference
        ]
    if config.decoys.require_different_cath_h:
        candidates = candidates.loc[candidates["cath_h"] != target.cath_h]
    candidates["helix_difference"] = (candidates["helix_fraction"] - target.helix_fraction).abs()
    candidates["strand_difference"] = (candidates["strand_fraction"] - target.strand_fraction).abs()
    candidates = candidates.loc[
        (candidates["helix_difference"] <= config.decoys.maximum_helix_fraction_difference)
        & (candidates["strand_difference"] <= config.decoys.maximum_strand_fraction_difference)
    ]
    return candidates.sort_values(
        ["helix_difference", "strand_difference", "domain_id"], kind="stable"
    )


def _circular_mapping(
    length: int, minimum_displacement_fraction: float, rng: np.random.Generator
) -> list[int]:
    minimum = max(1, int(np.ceil(length * minimum_displacement_fraction)))
    possible = np.array(
        [shift for shift in range(1, length) if min(shift, length - shift) >= minimum],
        dtype=int,
    )
    if not len(possible):
        possible = np.arange(1, length, dtype=int)
    shift = int(rng.choice(possible))
    return np.roll(np.arange(length), shift).astype(int).tolist()


def _length_mapping(target_length: int, source_length: int) -> list[int]:
    """Map a permitted near-length control onto every target position."""

    if target_length == source_length:
        return list(range(target_length))
    return np.rint(np.linspace(0, source_length - 1, target_length)).astype(int).tolist()


def _mapped_residues(
    decoy_id: str,
    target_domain_id: str,
    source_domain_id: str,
    source: pd.DataFrame,
    mapping: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_position, source_position in enumerate(mapping):
        residue = source.iloc[source_position]
        row = {
            "decoy_id": decoy_id,
            "target_domain_id": target_domain_id,
            "position": target_position,
            "source_domain_id": source_domain_id,
            "source_position": source_position,
            "dssp": residue["dssp"],
            "secondary_structure": residue["secondary_structure"],
            "rsa": float(residue["rsa"]),
            "contact_degree": int(residue["contact_degree"]),
        }
        for column in coordinate_columns():
            row[column] = float(residue[column])
        rows.append(row)
    return rows


def _decoy_row(
    decoy_id: str,
    target: Any,
    decoy_type: str,
    source: Any,
    mapping: list[int] | None,
    supports_coordinate: bool,
    supports_graph: bool,
    preserved: list[str],
    requested_swaps: int = 0,
    completed_swaps: int = 0,
) -> dict[str, Any]:
    return {
        "decoy_id": decoy_id,
        "target_domain_id": target.domain_id,
        "decoy_type": decoy_type,
        "source_domain_id": source.domain_id,
        "target_length": int(target.length),
        "source_length": int(source.length),
        "target_cath_h": target.cath_h,
        "source_cath_h": source.cath_h,
        "helix_fraction_difference": abs(float(target.helix_fraction - source.helix_fraction)),
        "strand_fraction_difference": abs(float(target.strand_fraction - source.strand_fraction)),
        "supports_coordinate_teacher": supports_coordinate,
        "supports_graph_teacher": supports_graph,
        "preserved_statistics": preserved,
        "mapping": mapping,
        "requested_edge_swaps": requested_swaps,
        "completed_edge_swaps": completed_swaps,
    }
