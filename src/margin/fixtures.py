"""Deterministic fixtures that exercise foundation audit without making scientific claims."""

from __future__ import annotations

import numpy as np
import pandas as pd

from margin.config import ProjectConfig
from margin.constants import AA_ALPHABET, AA_TO_INDEX
from margin.data_registry.registry import RegistryTables
from margin.data_registry.schema import BENCHMARK_COLUMNS, HOMOLOGY_COLUMNS
from margin.teachers.cache import TeacherScoreCache
from margin.teachers.schema import logp_columns


def build_synthetic_registry(config: ProjectConfig, domain_count: int = 9) -> RegistryTables:
    """Create compact CATH-like domains with valid coordinates and environment labels."""

    if config.data_mode != "synthetic":
        raise ValueError("synthetic registry is forbidden in real-data mode")
    length = max(config.registry.min_length, min(config.registry.max_length, 60))
    rng = np.random.default_rng(config.seed + 10)
    domain_rows: list[dict[str, object]] = []
    residue_rows: list[dict[str, object]] = []
    hydrophobic = np.array(list("AVILMFWY"))
    polar = np.array(list("DENQKRST"))
    all_amino_acids = np.array(list(AA_ALPHABET))
    for domain_index in range(domain_count):
        domain_id = f"SYN{domain_index:03d}A00"
        ca = _compact_backbone(length, angle_offset=domain_index * 0.17)
        degree = _contact_degree(ca, config)
        sequence: list[str] = []
        per_domain_rows: list[dict[str, object]] = []
        for position in range(length):
            rsa = (0.10, 0.16, 0.34, 0.42, 0.62, 0.74)[position % 6]
            burial = (
                "buried"
                if rsa <= config.registry.buried_rsa_max
                else "exposed"
                if rsa >= config.registry.exposed_rsa_min
                else "intermediate"
            )
            pool = (
                hydrophobic
                if burial == "buried"
                else polar
                if burial == "exposed"
                else all_amino_acids
            )
            residue = str(rng.choice(pool))
            sequence.append(residue)
            section = position * 3 // length
            secondary = ("helix", "strand", "turn_or_coil")[section]
            dssp = {"helix": "H", "strand": "E", "turn_or_coil": "C"}[secondary]
            row: dict[str, object] = {
                "domain_id": domain_id,
                "position": position,
                "residue": residue,
                "pdb_residue_id": str(position + 1),
                "is_resolved": True,
                "has_complete_backbone": True,
                "dssp": dssp,
                "secondary_structure": secondary,
                "rsa": rsa,
                "burial": burial,
                "contact_degree": int(degree[position]),
                "contact_class": (
                    "high_contact"
                    if degree[position] >= config.registry.high_contact_degree_min
                    else "low_contact"
                ),
                "conservation_score": (0.90, 0.55, 0.20)[position % 3],
                "conservation_class": (
                    "conserved"
                    if position % 3 == 0
                    else "variable"
                    if position % 3 == 2
                    else "intermediate"
                ),
            }
            atoms = {
                "n": ca[position] + np.array([-1.15, 0.10, 0.00]),
                "ca": ca[position],
                "c": ca[position] + np.array([1.20, -0.08, 0.03]),
                "o": ca[position] + np.array([1.65, -0.75, 0.05]),
            }
            for atom, coordinate in atoms.items():
                for axis, value in zip("xyz", coordinate, strict=True):
                    row[f"{atom}_{axis}"] = float(value)
            per_domain_rows.append(row)
        sequence_text = "".join(sequence)
        domain_rows.append(
            {
                "domain_id": domain_id,
                "pdb_id": f"S{domain_index:03d}",
                "chain_id": "A",
                "sequence": sequence_text,
                "length": length,
                "cath_c": "1",
                "cath_a": f"1.{domain_index + 1}",
                "cath_t": f"1.{domain_index + 1}.10",
                "cath_h": f"1.{domain_index + 1}.10.10",
                "resolution_angstrom": 2.0,
                "structure_path": f"synthetic://{domain_id}",
                "structure_sha256": f"synthetic-fixture-{domain_index:03d}",
                "source_name": config.registry.source_name,
                "source_version": config.registry.source_version,
                "source_url": "synthetic://margin-fixture",
                "is_experimental": False,
                "dataset": "synthetic-CATH",
                "analysis_role": "training_candidate",
                "eligible_for_training": True,
                "missing_residue_count": 0,
                "missing_fraction": 0.0,
                "helix_fraction": float(
                    sum(row["secondary_structure"] == "helix" for row in per_domain_rows) / length
                ),
                "strand_fraction": float(
                    sum(row["secondary_structure"] == "strand" for row in per_domain_rows) / length
                ),
            }
        )
        residue_rows.extend(per_domain_rows)
    return RegistryTables(pd.DataFrame(domain_rows), pd.DataFrame(residue_rows))


def build_synthetic_leakage_inputs(
    registry: RegistryTables,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create three structured benchmarks and one homology-only exclusion."""

    first, second, third, fourth = list(registry.domains.itertuples(index=False))[:4]
    benchmarks = pd.DataFrame(
        [
            {
                "benchmark_id": f"SYNTHETIC_BENCH_{index}",
                "domain_id": domain.domain_id,
                "dataset": "synthetic-control",
                "sequence": domain.sequence,
                "pdb_id": domain.pdb_id,
                "chain_id": domain.chain_id,
                "cath_t": domain.cath_t,
                "cath_h": domain.cath_h,
            }
            for index, domain in enumerate((first, second, third))
        ]
        + [
            {
                "benchmark_id": "SYNTHETIC_BENCH_HOMOLOG",
                "domain_id": None,
                "dataset": "synthetic-control",
                "sequence": fourth.sequence[::-1],
                "pdb_id": "EXT0",
                "chain_id": "B",
                "cath_t": "9.9.9",
                "cath_h": "9.9.9.9",
            },
        ],
        columns=BENCHMARK_COLUMNS,
    )
    homology = pd.DataFrame(
        [
            {
                "domain_id": fourth.domain_id,
                "benchmark_id": "SYNTHETIC_BENCH_HOMOLOG",
                "sequence_identity": 0.45,
                "query_coverage": 0.95,
                "target_coverage": 0.95,
            }
        ],
        columns=HOMOLOGY_COLUMNS,
    )
    return benchmarks, homology


def build_synthetic_dms(
    cache: TeacherScoreCache,
    registry: RegistryTables,
    config: ProjectConfig,
) -> pd.DataFrame:
    """Generate a labeled stability fixture from the primary teacher plus fixed noise."""

    external_domains = set(
        registry.domains.loc[registry.domains["analysis_role"] == "external_benchmark", "domain_id"]
    )
    primary = cache.scores.loc[
        (cache.scores["teacher_id"] == config.audit.primary_teacher_id)
        & (cache.scores["structure_role"] == config.audit.paired_role)
        & (cache.scores["domain_id"].isin(external_domains))
        & (cache.scores["state_id"].str.contains(":native_reference:c000:r000", regex=False))
    ]
    averaged = (
        primary.groupby(["domain_id", "position"], observed=True)[logp_columns()]
        .mean()
        .reset_index()
    )
    native_lookup = registry.residues.set_index(["domain_id", "position"])["residue"]
    rng = np.random.default_rng(config.seed + 11)
    rows: list[dict[str, object]] = []
    for domain_id, frame in averaged.groupby("domain_id", observed=True):
        for row in frame.iloc[::2].itertuples(index=False):
            values = np.array([getattr(row, column) for column in logp_columns()], dtype=float)
            wild_type = str(native_lookup.loc[(domain_id, int(row.position))])
            wild_index = AA_TO_INDEX[wild_type]
            alternative_indices = [
                index for index in range(len(AA_ALPHABET)) if index != wild_index
            ]
            mutant_index = int(rng.choice(alternative_indices))
            predicted = float(values[mutant_index] - values[wild_index])
            noise = 0.05 * np.sin((int(row.position) + 1) * (len(rows) + 1))
            rows.append(
                {
                    "assay_id": f"{domain_id}:stability",
                    "domain_id": domain_id,
                    "position": int(row.position),
                    "wild_type": wild_type,
                    "mutant": AA_ALPHABET[mutant_index],
                    "effect": predicted + noise,
                    "assay_type": "stability",
                    "source": "synthetic-fixture-not-independent-evidence",
                }
            )
    return pd.DataFrame(rows)


def _compact_backbone(length: int, angle_offset: float) -> np.ndarray:
    position = np.arange(length, dtype=float)
    angle = 2.0 * np.pi * position / 15.0 + angle_offset
    layer = np.floor(position / 15.0)
    return np.column_stack(
        [
            6.0 * np.cos(angle),
            6.0 * np.sin(angle),
            3.5 * layer + 0.25 * np.sin(angle * 2.0),
        ]
    )


def _contact_degree(ca: np.ndarray, config: ProjectConfig) -> np.ndarray:
    distance = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
    separation = np.abs(np.arange(len(ca))[:, None] - np.arange(len(ca))[None, :])
    contact = (distance <= config.registry.contact_distance_angstrom) & (
        separation >= config.registry.contact_minimum_sequence_separation
    )
    return contact.sum(axis=1).astype(int)
