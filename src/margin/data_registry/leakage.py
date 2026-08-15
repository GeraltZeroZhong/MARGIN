"""Auditable exclusion of benchmark proteins and their homologous CATH groups."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from margin.config import ProjectConfig
from margin.data_registry.schema import (
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
)


@dataclass(frozen=True)
class LeakageAudit:
    eligibility: pd.DataFrame
    relations: pd.DataFrame
    summary: dict[str, int]


def audit_benchmark_leakage(
    domains: pd.DataFrame,
    benchmarks: pd.DataFrame,
    homology_hits: pd.DataFrame,
    config: ProjectConfig,
) -> LeakageAudit:
    """Apply exact-ID/sequence, identity, CATH-H, and CATH-T exclusions."""

    validate_domains(domains)
    validate_benchmarks(benchmarks)
    validate_homology_hits(homology_hits)
    unknown = set(homology_hits["domain_id"]) - set(domains["domain_id"])
    if unknown:
        raise ValueError(f"homology hits reference unknown domains: {sorted(unknown)[:5]}")
    unknown_benchmarks = set(homology_hits["benchmark_id"]) - set(benchmarks["benchmark_id"])
    if unknown_benchmarks:
        raise ValueError(
            f"homology hits reference unknown benchmarks: {sorted(unknown_benchmarks)[:5]}"
        )

    relation_rows: list[dict[str, object]] = []
    benchmark_by_pdb_chain = {
        (str(row.pdb_id).lower(), str(row.chain_id)): row.benchmark_id
        for row in benchmarks.itertuples(index=False)
        if pd.notna(row.pdb_id)
        and pd.notna(row.chain_id)
        and str(row.pdb_id).strip()
        and str(row.chain_id).strip()
    }
    benchmark_by_sequence = {
        str(row.sequence): row.benchmark_id
        for row in benchmarks.itertuples(index=False)
        if pd.notna(row.sequence) and str(row.sequence).strip()
    }
    benchmark_h = {str(value) for value in benchmarks["cath_h"].dropna() if str(value).strip()}
    benchmark_t = {str(value) for value in benchmarks["cath_t"].dropna() if str(value).strip()}
    for row in domains.itertuples(index=False):
        exact = benchmark_by_pdb_chain.get((str(row.pdb_id).lower(), str(row.chain_id)))
        if config.registry.exclude_exact_benchmark_ids and exact is not None:
            relation_rows.append(_relation(row.domain_id, exact, "exact_pdb_chain", 1.0))
        exact_sequence = benchmark_by_sequence.get(str(row.sequence))
        if exact_sequence is not None:
            relation_rows.append(
                _relation(row.domain_id, exact_sequence, "exact_sequence", 1.0, 1.0, 1.0)
            )
        if config.registry.exclude_cath_h and str(row.cath_h) in benchmark_h:
            relation_rows.append(_relation(row.domain_id, "*", "same_cath_h", None))
        if config.registry.exclude_cath_t_for_topology_ood and str(row.cath_t) in benchmark_t:
            relation_rows.append(_relation(row.domain_id, "*", "same_cath_t", None))

    high_identity = homology_hits.loc[
        homology_hits["sequence_identity"] >= config.registry.benchmark_identity_threshold
    ]
    for hit in high_identity.itertuples(index=False):
        relation_rows.append(
            _relation(
                hit.domain_id,
                hit.benchmark_id,
                "sequence_identity",
                float(hit.sequence_identity),
                float(hit.query_coverage),
                float(hit.target_coverage),
            )
        )

    relations = pd.DataFrame(
        relation_rows,
        columns=[
            "domain_id",
            "benchmark_id",
            "reason",
            "sequence_identity",
            "query_coverage",
            "target_coverage",
        ],
    ).drop_duplicates()
    reasons = (
        relations.groupby("domain_id", observed=True)["reason"]
        .agg(lambda values: sorted(set(values)))
        .to_dict()
        if not relations.empty
        else {}
    )
    eligibility = domains[["domain_id", "cath_t", "cath_h"]].copy()
    eligibility["eligible_for_training"] = ~eligibility["domain_id"].isin(reasons)
    eligibility["exclusion_reasons"] = (
        eligibility["domain_id"]
        .map(reasons)
        .map(lambda value: value if isinstance(value, list) else [])
    )
    summary = {
        "total_domains": int(len(domains)),
        "eligible_domains": int(eligibility["eligible_for_training"].sum()),
        "excluded_domains": int((~eligibility["eligible_for_training"]).sum()),
        "relations": int(len(relations)),
    }
    for reason, count in relations["reason"].value_counts().items():
        summary[f"relations_{reason}"] = int(count)
    return LeakageAudit(eligibility=eligibility, relations=relations, summary=summary)


def write_leakage_audit(
    directory: Path,
    audit: LeakageAudit,
    config: ProjectConfig,
    input_files: list[Path] | None = None,
) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    eligibility_path = directory / "training_eligibility.parquet"
    relations_path = directory / "leakage_relations.parquet"
    write_parquet(eligibility_path, audit.eligibility)
    write_parquet(relations_path, audit.relations)
    manifest: dict[str, object] = {
        **runtime_manifest(config.paths.project_root),
        "schema_version": config.schema_version,
        "thresholds": {
            "sequence_identity": config.registry.benchmark_identity_threshold,
            "exclude_cath_h": config.registry.exclude_cath_h,
            "exclude_cath_t_for_topology_ood": config.registry.exclude_cath_t_for_topology_ood,
        },
        "summary": audit.summary,
        "input_files": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in (input_files or [])
            if path.exists() and path.is_file()
        ],
        "eligibility": table_manifest(eligibility_path, audit.eligibility),
        "relations": table_manifest(relations_path, audit.relations),
    }
    write_json(directory / "leakage_manifest.json", manifest)
    return manifest


def _relation(
    domain_id: str,
    benchmark_id: str,
    reason: str,
    identity: float | None,
    query_coverage: float | None = None,
    target_coverage: float | None = None,
) -> dict[str, object]:
    return {
        "domain_id": domain_id,
        "benchmark_id": benchmark_id,
        "reason": reason,
        "sequence_identity": identity,
        "query_coverage": query_coverage,
        "target_coverage": target_coverage,
    }
