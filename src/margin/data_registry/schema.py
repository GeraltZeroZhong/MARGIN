"""Canonical foundation audit data-registry schemas."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

from margin.constants import AA_ALPHABET, BACKBONE_ATOMS

DOMAIN_COLUMNS = (
    "domain_id",
    "pdb_id",
    "chain_id",
    "sequence",
    "length",
    "cath_c",
    "cath_a",
    "cath_t",
    "cath_h",
    "resolution_angstrom",
    "structure_path",
    "structure_sha256",
    "source_name",
    "source_version",
    "source_url",
    "is_experimental",
    "dataset",
    "analysis_role",
    "eligible_for_training",
    "missing_residue_count",
    "missing_fraction",
    "helix_fraction",
    "strand_fraction",
)

RESIDUE_COLUMNS = (
    "domain_id",
    "position",
    "residue",
    "pdb_residue_id",
    "is_resolved",
    "has_complete_backbone",
    "dssp",
    "secondary_structure",
    "rsa",
    "burial",
    "contact_degree",
    "contact_class",
    "conservation_score",
    "conservation_class",
    *tuple(f"{atom.lower()}_{axis}" for atom in BACKBONE_ATOMS for axis in "xyz"),
)

BENCHMARK_COLUMNS = (
    "benchmark_id",
    "domain_id",
    "dataset",
    "sequence",
    "pdb_id",
    "chain_id",
    "cath_t",
    "cath_h",
)

HOMOLOGY_COLUMNS = (
    "domain_id",
    "benchmark_id",
    "sequence_identity",
    "query_coverage",
    "target_coverage",
)


def validate_domains(table: pd.DataFrame) -> None:
    _require_columns(table, DOMAIN_COLUMNS, "domains")
    if table["domain_id"].duplicated().any():
        duplicates = sorted(table.loc[table["domain_id"].duplicated(), "domain_id"].unique())
        raise ValueError(f"duplicate domain_id values: {duplicates[:5]}")
    bad_lengths = table["sequence"].str.len().to_numpy() != table["length"].to_numpy()
    if bad_lengths.any():
        raise ValueError("domain sequence length does not match the length column")
    bad_sequence = table["sequence"].map(lambda sequence: bool(set(sequence) - set(AA_ALPHABET)))
    if bad_sequence.any():
        raise ValueError("domain sequences must use only the canonical 20 amino acids")
    allowed_roles = {"training_candidate", "external_benchmark"}
    if not set(table["analysis_role"]).issubset(allowed_roles):
        raise ValueError(f"analysis_role must be one of {sorted(allowed_roles)}")
    for column in ("is_experimental", "eligible_for_training"):
        if not is_bool_dtype(table[column]):
            raise ValueError(f"{column} must have boolean dtype")
    eligible = table["eligible_for_training"]
    if (eligible & (table["analysis_role"] != "training_candidate")).any():
        raise ValueError("external benchmark domains cannot be eligible for training")


def validate_residues(domains: pd.DataFrame, residues: pd.DataFrame) -> None:
    _require_columns(residues, RESIDUE_COLUMNS, "residues")
    duplicate = residues.duplicated(["domain_id", "position"])
    if duplicate.any():
        raise ValueError("residue table contains duplicate domain_id/position keys")
    expected = domains.set_index("domain_id")["length"].astype(int)
    observed = residues.groupby("domain_id", observed=True).size()
    if not observed.reindex(expected.index, fill_value=0).equals(expected):
        raise ValueError("each domain must have exactly one residue row per sequence position")
    unknown_domains = set(residues["domain_id"]) - set(domains["domain_id"])
    if unknown_domains:
        raise ValueError(f"residue table references unknown domains: {sorted(unknown_domains)[:5]}")
    ordered = residues.sort_values(["domain_id", "position"])
    for domain_id, frame in ordered.groupby("domain_id", observed=True):
        positions = frame["position"].to_numpy(dtype=int)
        if not np.array_equal(positions, np.arange(len(frame))):
            raise ValueError(f"residue positions must be zero-based and contiguous: {domain_id}")
    observed_sequences = ordered.groupby("domain_id", observed=True)["residue"].agg("".join)
    expected_sequences = domains.set_index("domain_id")["sequence"]
    if not observed_sequences.reindex(expected_sequences.index).equals(expected_sequences):
        raise ValueError("residue identities do not reproduce the canonical domain sequence")
    coordinates = coordinate_columns()
    resolved = residues["is_resolved"].to_numpy(dtype=bool)
    if np.isfinite(residues.loc[~resolved, coordinates].to_numpy(dtype=float)).any():
        raise ValueError("unresolved residues must not contain finite backbone coordinates")
    conservation = residues["conservation_score"].to_numpy(dtype=float)
    unavailable = residues["conservation_class"].to_numpy(dtype=str) == "unavailable"
    if np.isfinite(conservation[unavailable]).any():
        raise ValueError("unavailable conservation rows must not contain a finite score")
    available = ~unavailable
    if (
        not np.isfinite(conservation[available]).all()
        or ((conservation[available] < 0) | (conservation[available] > 1)).any()
    ):
        raise ValueError("available conservation scores must be finite and in [0, 1]")


def validate_benchmarks(table: pd.DataFrame) -> None:
    _require_columns(table, BENCHMARK_COLUMNS, "benchmarks")
    if table["benchmark_id"].duplicated().any():
        raise ValueError("benchmark_id values must be unique")
    if (
        table["benchmark_id"].isna().any()
        or table["benchmark_id"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError("benchmark_id values must be non-empty")
    bad_sequence = table["sequence"].map(
        lambda sequence: (
            not isinstance(sequence, str) or not sequence or bool(set(sequence) - set(AA_ALPHABET))
        )
    )
    if bad_sequence.any():
        raise ValueError("benchmark sequences must be non-empty and use canonical amino acids")


def validate_homology_hits(table: pd.DataFrame) -> None:
    _require_columns(table, HOMOLOGY_COLUMNS, "homology hits")
    if table.duplicated(["domain_id", "benchmark_id"]).any():
        raise ValueError("homology domain/benchmark keys must be unique")
    for column in ("sequence_identity", "query_coverage", "target_coverage"):
        values = table[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"{column} must be finite and in [0, 1]")


def coordinate_columns() -> list[str]:
    return [f"{atom.lower()}_{axis}" for atom in BACKBONE_ATOMS for axis in "xyz"]


def _require_columns(table: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = set(required) - set(table.columns)
    if missing:
        raise ValueError(f"{label} table is missing columns: {sorted(missing)}")
