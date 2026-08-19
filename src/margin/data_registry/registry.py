"""Build, validate, and persist the unified foundation audit data registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from margin.config import ProjectConfig
from margin.data_registry.cath import (
    locate_domain_structure,
    read_cath_domain_list,
    read_cath_fasta,
)
from margin.data_registry.schema import validate_domains, validate_residues
from margin.preprocessing.structure import preprocess_domain_structure
from margin.provenance import (
    runtime_manifest,
    sha256_file,
    table_manifest,
    write_json,
    write_parquet,
)

CATH_SOURCE_URL = "https://download.cathdb.info/cath/releases/all-releases/v4_4_0/"


@dataclass(frozen=True)
class RegistryTables:
    domains: pd.DataFrame
    residues: pd.DataFrame


def build_cath_registry(config: ProjectConfig) -> tuple[RegistryTables, pd.DataFrame]:
    """Build an experimental CATH registry and an explicit exclusion table."""

    paths = config.paths
    if paths.cath_domain_list is None or paths.cath_fasta is None or paths.structures_dir is None:
        raise ValueError("CATH registry build requires domain list, FASTA, and structures_dir")
    classification = read_cath_domain_list(paths.cath_domain_list)
    sequences = read_cath_fasta(paths.cath_fasta)
    domain_rows: list[dict[str, Any]] = []
    residue_tables: list[pd.DataFrame] = []
    excluded: list[dict[str, str]] = []

    for record in classification.itertuples(index=False):
        reason = _pre_filter_reason(record, sequences, config)
        if reason is not None:
            excluded.append({"domain_id": record.domain_id, "stage": "metadata", "reason": reason})
            continue
        sequence = sequences[record.domain_id]
        structure_path = locate_domain_structure(paths.structures_dir, record.domain_id)
        if structure_path is None:
            excluded.append(
                {"domain_id": record.domain_id, "stage": "structure", "reason": "structure_missing"}
            )
            continue
        try:
            residues, summary = preprocess_domain_structure(
                record.domain_id,
                sequence,
                structure_path,
                record.chain_id,
                config.registry,
            )
        except (ValueError, RuntimeError, OSError) as error:
            excluded.append(
                {
                    "domain_id": record.domain_id,
                    "stage": "structure",
                    "reason": f"preprocessing_failed:{error}",
                }
            )
            continue
        if summary["missing_fraction"] > config.registry.max_missing_fraction:
            excluded.append(
                {
                    "domain_id": record.domain_id,
                    "stage": "structure",
                    "reason": "too_many_missing_backbone_residues",
                }
            )
            continue
        domain_rows.append(
            {
                "domain_id": record.domain_id,
                "pdb_id": record.pdb_id,
                "chain_id": summary["selected_chain"],
                "sequence": sequence,
                "length": len(sequence),
                "cath_c": record.cath_c,
                "cath_a": record.cath_a,
                "cath_t": record.cath_t,
                "cath_h": record.cath_h,
                "resolution_angstrom": record.resolution_angstrom,
                "structure_path": str(structure_path.resolve()),
                "structure_sha256": sha256_file(structure_path),
                "source_name": config.registry.source_name,
                "source_version": config.registry.source_version,
                "source_url": CATH_SOURCE_URL,
                "is_experimental": True,
                "dataset": config.registry.source_name,
                "analysis_role": "training_candidate",
                "eligible_for_training": True,
                "missing_residue_count": summary["missing_residue_count"],
                "missing_fraction": summary["missing_fraction"],
                "helix_fraction": summary["helix_fraction"],
                "strand_fraction": summary["strand_fraction"],
            }
        )
        residue_tables.append(residues)

    domains = pd.DataFrame(domain_rows)
    residues = pd.concat(residue_tables, ignore_index=True) if residue_tables else pd.DataFrame()
    tables = RegistryTables(domains=domains, residues=residues)
    if not domains.empty:
        validate_domains(domains)
        validate_residues(domains, residues)
    return tables, pd.DataFrame(excluded, columns=["domain_id", "stage", "reason"])


def write_registry(
    directory: Path,
    tables: RegistryTables,
    config: ProjectConfig,
    exclusions: pd.DataFrame | None = None,
    input_files: list[Path] | None = None,
) -> dict[str, Any]:
    validate_domains(tables.domains)
    validate_residues(tables.domains, tables.residues)
    directory.mkdir(parents=True, exist_ok=True)
    domain_path = directory / "domains.parquet"
    residue_path = directory / "residues.parquet"
    write_parquet(domain_path, tables.domains)
    write_parquet(residue_path, tables.residues)
    manifest: dict[str, Any] = {
        **runtime_manifest(config.paths.project_root),
        "schema_version": config.schema_version,
        "data_mode": config.data_mode,
        "seed": config.seed,
        "source": {
            "name": config.registry.source_name,
            "version": config.registry.source_version,
        },
        "parameters": config.registry.model_dump(mode="json"),
        "input_files": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in (input_files or [])
            if path.exists() and path.is_file()
        ],
        "domains": table_manifest(domain_path, tables.domains),
        "residues": table_manifest(residue_path, tables.residues),
    }
    if exclusions is not None:
        exclusion_path = directory / "preprocessing_exclusions.parquet"
        write_parquet(exclusion_path, exclusions)
        manifest["preprocessing_exclusions"] = table_manifest(exclusion_path, exclusions)
    write_json(directory / "manifest.json", manifest)
    return manifest


def load_registry(directory: Path) -> RegistryTables:
    domains = pd.read_parquet(directory / "domains.parquet")
    residues = pd.read_parquet(directory / "residues.parquet")
    validate_domains(domains)
    validate_residues(domains, residues)
    return RegistryTables(domains=domains, residues=residues)


def registry_from_canonical_input(path: Path) -> RegistryTables:
    """Load a canonical domain table and its sibling residue table."""

    if path.is_dir():
        return load_registry(path)
    raise ValueError("canonical registry input must be a directory containing both parquet tables")


def _pre_filter_reason(record: Any, sequences: dict[str, str], config: ProjectConfig) -> str | None:
    sequence = sequences.get(record.domain_id)
    if sequence is None:
        return "sequence_missing"
    if not config.registry.min_length <= len(sequence) <= config.registry.max_length:
        return "length_out_of_range"
    if set(sequence) - set(config.registry.allowed_amino_acids):
        return "noncanonical_sequence"
    resolution = float(record.resolution_angstrom)
    if np.isfinite(resolution) and resolution > config.registry.max_resolution_angstrom:
        return "resolution_above_limit"
    return None
