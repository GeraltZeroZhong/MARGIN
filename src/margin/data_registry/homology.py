"""Reproducible MMseqs2 search from candidate domains to external benchmarks."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from margin.config import ProjectConfig
from margin.data_registry.schema import (
    HOMOLOGY_COLUMNS,
    validate_benchmarks,
    validate_domains,
    validate_homology_hits,
)
from margin.provenance import (
    runtime_manifest,
    sha256_file,
    table_manifest,
    write_json,
    write_parquet,
    write_text,
)


def build_mmseqs_homology(
    domains: pd.DataFrame,
    benchmarks: pd.DataFrame,
    output_path: Path,
    config: ProjectConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Search every candidate sequence against all benchmark sequences."""

    validate_domains(domains)
    validate_benchmarks(benchmarks)
    if domains.empty or benchmarks.empty:
        raise ValueError("MMseqs2 homology search requires non-empty domain and benchmark tables")
    executable = shutil.which(config.homology.executable)
    if executable is None:
        raise FileNotFoundError(f"MMseqs2 executable not found: {config.homology.executable}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    query_fasta = output_path.with_suffix(".queries.fasta")
    target_fasta = output_path.with_suffix(".targets.fasta")
    write_text(
        query_fasta,
        _fasta(domains["domain_id"].astype(str), domains["sequence"].astype(str)),
    )
    write_text(
        target_fasta,
        _fasta(benchmarks["benchmark_id"].astype(str), benchmarks["sequence"].astype(str)),
    )
    with tempfile.TemporaryDirectory(prefix="margin-mmseqs-", dir=output_path.parent) as name:
        temporary = Path(name)
        raw_output = temporary / "hits.tsv"
        mmseqs_tmp = temporary / "work"
        command = [
            executable,
            "easy-search",
            str(query_fasta),
            str(target_fasta),
            str(raw_output),
            str(mmseqs_tmp),
            "--format-output",
            "query,target,fident,qcov,tcov",
            "-s",
            str(config.homology.sensitivity),
            "-e",
            str(config.homology.evalue),
            "--max-seqs",
            str(len(benchmarks)),
            "--threads",
            str(config.homology.threads),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                f"MMseqs2 failed with exit code {completed.returncode}: {completed.stderr[-2000:]}"
            )
        hits = parse_mmseqs_hits(raw_output)
    query_hash = sha256_file(query_fasta)
    target_hash = sha256_file(target_fasta)
    write_parquet(output_path, hits)
    version = subprocess.run(
        [executable, "version"], check=False, capture_output=True, text=True
    ).stdout.strip()
    manifest = {
        **runtime_manifest(config.paths.project_root),
        "schema_version": config.schema_version,
        "engine": "MMseqs2",
        "engine_version": version,
        "command": command,
        "parameters": config.homology.model_dump(mode="json"),
        "query_fasta_sha256": query_hash,
        "target_fasta_sha256": target_hash,
        "query_fasta_path": str(query_fasta),
        "target_fasta_path": str(target_fasta),
        "hits": table_manifest(output_path, hits),
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    write_json(manifest_path, manifest)
    return hits, manifest


def parse_mmseqs_hits(path: Path) -> pd.DataFrame:
    """Parse the fixed MMseqs2 format and normalize percent-style fields."""

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=HOMOLOGY_COLUMNS)
    raw = pd.read_csv(
        path,
        sep="\t",
        names=[
            "domain_id",
            "benchmark_id",
            "sequence_identity",
            "query_coverage",
            "target_coverage",
        ],
    )
    for column in ("sequence_identity", "query_coverage", "target_coverage"):
        values = raw[column].astype(float)
        raw[column] = values.where(values <= 1.0, values / 100.0)
    raw["_minimum_coverage"] = raw[["query_coverage", "target_coverage"]].min(axis=1)
    hits = (
        raw.sort_values(
            ["domain_id", "benchmark_id", "sequence_identity", "_minimum_coverage"],
            ascending=[True, True, False, False],
            kind="stable",
        )
        .drop_duplicates(["domain_id", "benchmark_id"])
        .drop(columns="_minimum_coverage")
        .reset_index(drop=True)
    )
    validate_homology_hits(hits)
    return hits


def _fasta(identifiers: pd.Series, sequences: pd.Series) -> str:
    return "".join(
        f">{identifier}\n{sequence}\n"
        for identifier, sequence in zip(identifiers, sequences, strict=True)
    )
