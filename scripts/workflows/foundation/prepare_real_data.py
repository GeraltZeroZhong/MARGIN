#!/usr/bin/env python
"""Prepare the fixed public-data inputs for the real foundation audit audit."""

from __future__ import annotations

import argparse
import io
import re
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from margin.config import ProjectConfig, load_config
from margin.constants import AA_ALPHABET
from margin.data_registry.cath import read_cath_domain_list, read_cath_fasta
from margin.data_registry.conservation import attach_conservation
from margin.data_registry.registry import RegistryTables, write_registry
from margin.preprocessing.structure import preprocess_domain_structure
from margin.provenance import (
    runtime_manifest,
    sha256_file,
    table_manifest,
    write_json,
    write_parquet,
    write_text,
)


@dataclass(frozen=True)
class ExternalDomain:
    pdb_id: str
    chain_id: str
    domain_id: str


EXTERNAL_DOMAINS = (
    ExternalDomain("4G3O", "A", "4g3oA00"),
    ExternalDomain("1TG0", "A", "1tg0A00"),
    ExternalDomain("1F0M", "A", "1f0mA00"),
    ExternalDomain("2CJJ", "A", "2cjjA00"),
    ExternalDomain("1ORC", "A", "1orcA00"),
    ExternalDomain("1A32", "A", "1a32A00"),
    ExternalDomain("1I2T", "A", "1i2tA00"),
    ExternalDomain("1LP1", "A", "1lp1A00"),
    ExternalDomain("3L1X", "A", "3l1xA01"),
    ExternalDomain("1YU5", "X", "1yu5X00"),
)

# Two matched donors are retained where the exact-length S40 pool permits it.
# Every entry passed experimental-backbone, DSSP/RSA, missingness, CATH-H/T,
# and secondary-structure-composition screening before this list was frozen.
TRAINING_DOMAIN_IDS = (
    "2q35A02",
    "4k12A00",
    "3aiiA02",
    "6rnzA00",
    "2xz2A00",
    "4dooA02",
    "1z8fA02",
    "4mtdA02",
    "2ra1A03",
    "4tq1A03",
    "2ii2A02",
    "7myqB02",
    "1wr8A02",
    "3vpbA02",
    "5lxvB00",
    "2x0sA04",
)

PROTEINGYM_DATASET = "ProteinGym_v1.3/Tsuboyama_2023"
PROTEINGYM_URL = "https://github.com/OATML-Markslab/ProteinGym"
RCSB_URL = "https://www.rcsb.org/"
MUTATION_PATTERN = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)
    raw = arguments.raw_root.resolve()
    work_root = arguments.work_root.resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    cath_root = raw / "cath" / "v4_4_0"
    cath_list = cath_root / "cath-domain-list-v4_4_0.txt"
    cath_fasta = cath_root / "cath-domain-seqs-v4_4_0.fa"
    candidate_structure_root = cath_root / "s40_exact_pool"
    external_structure_root = raw / "tsuboyama" / "experimental_pdb"
    reference_path = raw / "proteingym_DMS_substitutions.csv"
    dms_zip_path = raw / "DMS_ProteinGym_substitutions_v1.3.zip"
    msa_zip_path = raw / "DMS_msa_files_v1.3.zip"
    required = (
        cath_list,
        cath_fasta,
        candidate_structure_root,
        external_structure_root,
        reference_path,
        dms_zip_path,
        msa_zip_path,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing raw foundation audit inputs: {missing}")

    classifications = read_cath_domain_list(cath_list)
    sequences = read_cath_fasta(cath_fasta)
    classification_index = classifications.set_index("domain_id")
    reference = pd.read_csv(reference_path)
    external_metadata = _external_metadata(reference)

    _materialize_candidate_inputs(
        cath_list,
        sequences,
        candidate_structure_root,
        config,
    )
    external_structure_paths = _materialize_external_structures(
        external_structure_root,
        config,
    )
    benchmark = _build_benchmark_table(external_metadata, classification_index)
    dms = _build_dms_table(dms_zip_path, external_metadata)
    write_parquet(config.paths.benchmark_input, benchmark)
    write_parquet(config.paths.dms_input, dms)

    candidate_conservation, alignment_path = _candidate_conservation(
        sequences,
        cath_fasta,
        config,
        work_root,
    )
    external_conservation = _external_conservation(msa_zip_path, external_metadata)
    conservation = pd.concat(
        [candidate_conservation, external_conservation],
        ignore_index=True,
    ).sort_values(["domain_id", "position"], ignore_index=True)
    write_parquet(config.paths.conservation_input, conservation)

    audit_registry = _build_external_registry(
        external_metadata,
        classification_index,
        external_structure_paths,
        conservation,
        config,
    )
    write_registry(
        config.paths.audit_domain_input,
        audit_registry,
        config,
        input_files=[reference_path, *external_structure_paths.values()],
    )
    _write_preparation_manifest(
        raw,
        config,
        benchmark,
        dms,
        conservation,
        alignment_path,
        (cath_list, cath_fasta, reference_path, dms_zip_path, msa_zip_path),
    )
    print(f"training_domains={len(TRAINING_DOMAIN_IDS)}")
    print(f"external_domains={len(EXTERNAL_DOMAINS)}")
    print(f"dms_variants={len(dms)}")
    print(f"conservation_rows={len(conservation)}")


def _external_metadata(reference: pd.DataFrame) -> dict[str, pd.Series]:
    result: dict[str, pd.Series] = {}
    for specification in EXTERNAL_DOMAINS:
        matches = reference.loc[
            reference["DMS_id"].astype(str).str.endswith(f"_{specification.pdb_id}")
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one ProteinGym assay for {specification.pdb_id}, found {len(matches)}"
            )
        row = matches.iloc[0]
        sequence = str(row["target_seq"])
        if not sequence or set(sequence) - set(AA_ALPHABET):
            raise ValueError(f"invalid ProteinGym target sequence for {specification.pdb_id}")
        result[specification.domain_id] = row
    return result


def _materialize_candidate_inputs(
    source_domain_list: Path,
    sequences: dict[str, str],
    source_structure_root: Path,
    config: ProjectConfig,
) -> None:
    assert config.paths.cath_domain_list is not None
    assert config.paths.cath_fasta is not None
    assert config.paths.structures_dir is not None
    selected = set(TRAINING_DOMAIN_IDS)
    lines = []
    for line in source_domain_list.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and stripped.split()[0] in selected:
            lines.append(line)
    observed = {line.split()[0] for line in lines}
    if observed != selected:
        raise ValueError(
            f"CATH classification lacks selected domains: {sorted(selected - observed)}"
        )
    write_text(config.paths.cath_domain_list, "\n".join(lines) + "\n")

    fasta_lines: list[str] = []
    for domain_id in TRAINING_DOMAIN_IDS:
        sequence = sequences.get(domain_id)
        if sequence is None:
            raise ValueError(f"CATH FASTA lacks selected domain: {domain_id}")
        fasta_lines.extend([f">{domain_id}", sequence])
    write_text(config.paths.cath_fasta, "\n".join(fasta_lines) + "\n")

    config.paths.structures_dir.mkdir(parents=True, exist_ok=True)
    for domain_id in TRAINING_DOMAIN_IDS:
        source = source_structure_root / f"{domain_id}.pdb"
        if not source.is_file():
            raise FileNotFoundError(f"selected CATH structure is missing: {source}")
        shutil.copy2(source, config.paths.structures_dir / source.name)


def _materialize_external_structures(
    source_root: Path,
    config: ProjectConfig,
) -> dict[str, Path]:
    assert config.paths.benchmark_input is not None
    target_root = config.paths.benchmark_input.parent / "structures"
    target_root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for specification in EXTERNAL_DOMAINS:
        source = source_root / f"{specification.pdb_id}.pdb"
        if not source.is_file():
            raise FileNotFoundError(f"external experimental structure is missing: {source}")
        target = target_root / source.name
        shutil.copy2(source, target)
        result[specification.domain_id] = target.resolve()
    return result


def _build_benchmark_table(
    metadata: dict[str, pd.Series],
    classifications: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for specification in EXTERNAL_DOMAINS:
        assay = metadata[specification.domain_id]
        cath = classifications.loc[specification.domain_id]
        rows.append(
            {
                "benchmark_id": str(assay["DMS_id"]),
                "domain_id": specification.domain_id,
                "dataset": PROTEINGYM_DATASET,
                "sequence": str(assay["target_seq"]),
                "pdb_id": specification.pdb_id.lower(),
                "chain_id": specification.chain_id,
                "cath_t": cath["cath_t"],
                "cath_h": cath["cath_h"],
            }
        )
    return pd.DataFrame(rows)


def _build_dms_table(
    archive_path: Path,
    metadata: dict[str, pd.Series],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(archive_path) as archive:
        for specification in EXTERNAL_DOMAINS:
            assay = metadata[specification.domain_id]
            sequence = str(assay["target_seq"])
            archive_name = f"DMS_ProteinGym_substitutions/{assay['DMS_filename']}"
            with archive.open(archive_name) as handle:
                table = pd.read_csv(handle)
            for mutation, effect in zip(table["mutant"], table["DMS_score"], strict=True):
                match = MUTATION_PATTERN.fullmatch(str(mutation))
                if match is None or not np.isfinite(float(effect)):
                    continue
                wild_type, position_text, mutant = match.groups()
                position = int(position_text) - 1
                if position < 0 or position >= len(sequence):
                    raise ValueError(f"out-of-range DMS mutation: {mutation}")
                if sequence[position] != wild_type:
                    raise ValueError(
                        f"DMS wild type mismatch for {specification.domain_id}: {mutation}"
                    )
                rows.append(
                    {
                        "assay_id": str(assay["DMS_id"]),
                        "domain_id": specification.domain_id,
                        "position": position,
                        "wild_type": wild_type,
                        "mutant": mutant,
                        "effect": float(effect),
                        "assay_type": "stability",
                        "source": PROTEINGYM_DATASET,
                    }
                )
    result = pd.DataFrame(rows)
    if result.duplicated(["assay_id", "domain_id", "position", "mutant"]).any():
        raise ValueError("ProteinGym single-mutant keys are not unique")
    return result.sort_values(["assay_id", "position", "mutant"], ignore_index=True)


def _candidate_conservation(
    sequences: dict[str, str],
    target_fasta: Path,
    config: ProjectConfig,
    work_root: Path,
) -> tuple[pd.DataFrame, Path]:
    assert config.paths.conservation_input is not None
    output_root = config.paths.conservation_input.parent
    query_path = output_root / "cath_selected_queries.fasta"
    alignment_path = output_root / "cath_mmseqs_alignments.tsv"
    query_payload = "".join(
        f">{domain_id}\n{sequences[domain_id]}\n" for domain_id in TRAINING_DOMAIN_IDS
    )
    write_text(query_path, query_payload)
    with tempfile.TemporaryDirectory(prefix="margin-conservation-", dir=work_root) as temporary:
        command = [
            config.homology.executable,
            "easy-search",
            str(query_path),
            str(target_fasta),
            str(alignment_path),
            temporary,
            "-s",
            str(config.homology.sensitivity),
            "-e",
            str(config.homology.evalue),
            "--threads",
            str(config.homology.threads),
            "--max-seqs",
            "500",
            "--format-output",
            "query,target,qstart,qaln,taln,evalue",
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"MMseqs2 conservation search failed: {completed.stderr}")
    alignments = pd.read_csv(
        alignment_path,
        sep="\t",
        names=["query", "target", "qstart", "qaln", "taln", "evalue"],
    )
    rows: list[dict[str, object]] = []
    for domain_id in TRAINING_DOMAIN_IDS:
        sequence = sequences[domain_id]
        matches = np.zeros(len(sequence), dtype=np.int64)
        observations = np.zeros(len(sequence), dtype=np.int64)
        domain_hits = alignments.loc[alignments["query"] == domain_id]
        for hit in domain_hits.itertuples(index=False):
            if _cath_identifier(str(hit.target)) == domain_id:
                continue
            query_position = int(hit.qstart) - 2
            for query_aa, target_aa in zip(str(hit.qaln), str(hit.taln), strict=True):
                if query_aa == "-":
                    continue
                query_position += 1
                if query_position < 0 or query_position >= len(sequence):
                    raise ValueError(f"MMseqs2 query alignment exceeds {domain_id}")
                target_aa = target_aa.upper()
                if target_aa not in AA_ALPHABET:
                    continue
                observations[query_position] += 1
                matches[query_position] += int(target_aa == sequence[query_position])
        scores = np.divide(
            matches,
            observations,
            out=np.full(len(sequence), 0.5, dtype=float),
            where=observations > 0,
        )
        rows.extend(
            {
                "domain_id": domain_id,
                "position": position,
                "conservation_score": float(scores[position]),
                "homolog_observations": int(observations[position]),
                "conservation_method": "MMseqs2 CATH-v4.4 native-residue frequency",
            }
            for position in range(len(sequence))
        )
    return pd.DataFrame(rows), alignment_path


def _external_conservation(
    archive_path: Path,
    metadata: dict[str, pd.Series],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(archive_path) as archive:
        for specification in EXTERNAL_DOMAINS:
            assay = metadata[specification.domain_id]
            target = str(assay["target_seq"])
            archive_name = f"DMS_msa_files/{assay['MSA_filename']}"
            with archive.open(archive_name) as binary:
                records = _fasta_records(io.TextIOWrapper(binary, encoding="utf-8"))
                _, query = next(records)
                query_letters = "".join(character for character in query if character.isalpha())
                if query_letters.upper() != target:
                    raise ValueError(f"ProteinGym MSA query does not reproduce {assay['DMS_id']}")
                column_to_position: list[int] = []
                target_position = -1
                for character in query:
                    if character.isalpha():
                        target_position += 1
                    if not character.islower() and character != ".":
                        column_to_position.append(target_position if character != "-" else -1)
                matches = np.zeros(len(target), dtype=np.int64)
                observations = np.zeros(len(target), dtype=np.int64)
                for _, homolog in records:
                    aligned = "".join(
                        character.upper()
                        for character in homolog
                        if not character.islower() and character != "."
                    )
                    if len(aligned) != len(column_to_position):
                        raise ValueError(f"inconsistent A2M width in {archive_name}")
                    for column, amino_acid in enumerate(aligned):
                        position = column_to_position[column]
                        if position < 0 or amino_acid not in AA_ALPHABET:
                            continue
                        observations[position] += 1
                        matches[position] += int(amino_acid == target[position])
                scores = np.divide(
                    matches,
                    observations,
                    out=np.full(len(target), 0.5, dtype=float),
                    where=observations > 0,
                )
                rows.extend(
                    {
                        "domain_id": specification.domain_id,
                        "position": position,
                        "conservation_score": float(scores[position]),
                        "homolog_observations": int(observations[position]),
                        "conservation_method": (
                            "ProteinGym-v1.3 A2M native-residue frequency; "
                            "uncovered query insertions neutral"
                        ),
                    }
                    for position in range(len(target))
                )
    return pd.DataFrame(rows)


def _fasta_records(handle: io.TextIOBase) -> Iterator[tuple[str, str]]:
    name: str | None = None
    chunks: list[str] = []
    for line in handle:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(">"):
            if name is not None:
                yield name, "".join(chunks)
            name = stripped[1:]
            chunks = []
        else:
            chunks.append(stripped)
    if name is not None:
        yield name, "".join(chunks)


def _build_external_registry(
    metadata: dict[str, pd.Series],
    classifications: pd.DataFrame,
    structure_paths: dict[str, Path],
    conservation: pd.DataFrame,
    config: ProjectConfig,
) -> RegistryTables:
    domain_rows: list[dict[str, object]] = []
    residue_tables: list[pd.DataFrame] = []
    for specification in EXTERNAL_DOMAINS:
        assay = metadata[specification.domain_id]
        cath = classifications.loc[specification.domain_id]
        sequence = str(assay["target_seq"])
        structure_path = structure_paths[specification.domain_id]
        residues, summary = preprocess_domain_structure(
            specification.domain_id,
            sequence,
            structure_path,
            specification.chain_id,
            config.registry,
        )
        if summary["missing_fraction"] > config.registry.max_missing_fraction:
            raise ValueError(f"external structure has excess missing residues: {specification}")
        domain_rows.append(
            {
                "domain_id": specification.domain_id,
                "pdb_id": specification.pdb_id.lower(),
                "chain_id": summary["selected_chain"],
                "sequence": sequence,
                "length": len(sequence),
                "cath_c": cath["cath_c"],
                "cath_a": cath["cath_a"],
                "cath_t": cath["cath_t"],
                "cath_h": cath["cath_h"],
                "resolution_angstrom": float(cath["resolution_angstrom"]),
                "structure_path": str(structure_path),
                "structure_sha256": sha256_file(structure_path),
                "source_name": "ProteinGym/RCSB/CATH",
                "source_version": "ProteinGym-v1.3;CATH-v4.4.0",
                "source_url": PROTEINGYM_URL,
                "is_experimental": True,
                "dataset": PROTEINGYM_DATASET,
                "analysis_role": "external_benchmark",
                "eligible_for_training": False,
                "missing_residue_count": summary["missing_residue_count"],
                "missing_fraction": summary["missing_fraction"],
                "helix_fraction": summary["helix_fraction"],
                "strand_fraction": summary["strand_fraction"],
            }
        )
        residue_tables.append(residues)
    registry = RegistryTables(
        pd.DataFrame(domain_rows),
        pd.concat(residue_tables, ignore_index=True),
    )
    return attach_conservation(registry, conservation, config)


def _write_preparation_manifest(
    raw_root: Path,
    config: ProjectConfig,
    benchmark: pd.DataFrame,
    dms: pd.DataFrame,
    conservation: pd.DataFrame,
    alignment_path: Path,
    raw_files: tuple[Path, ...],
) -> None:
    assert config.paths.benchmark_input is not None
    assert config.paths.dms_input is not None
    assert config.paths.conservation_input is not None
    manifest = {
        **runtime_manifest(config.paths.project_root),
        "schema_version": config.schema_version,
        "raw_root": str(raw_root),
        "scope": "public computational datasets only",
        "sources": {
            "cath": "https://download.cathdb.info/cath/releases/all-releases/v4_4_0/",
            "proteingym": PROTEINGYM_URL,
            "rcsb": RCSB_URL,
        },
        "selected_training_domains": list(TRAINING_DOMAIN_IDS),
        "selected_external_domains": [item.domain_id for item in EXTERNAL_DOMAINS],
        "raw_files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in raw_files
        ],
        "benchmark": table_manifest(config.paths.benchmark_input, benchmark),
        "dms": table_manifest(config.paths.dms_input, dms),
        "conservation": table_manifest(config.paths.conservation_input, conservation),
        "conservation_alignments": {
            "path": str(alignment_path),
            "sha256": sha256_file(alignment_path),
        },
    }
    write_json(config.paths.benchmark_input.parent / "preparation_manifest.json", manifest)


def _cath_identifier(value: str) -> str:
    fields = value.split("|")
    return fields[-1].split("/", maxsplit=1)[0]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/foundation.yaml"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/external"))
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path("/tmp/margin"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
