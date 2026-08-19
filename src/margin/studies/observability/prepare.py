"""Prepare and lock the independent CATH S40 replication registry."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from margin.config import ProjectConfig, load_config
from margin.constants import AA_ALPHABET
from margin.data_registry.cath import read_cath_domain_list, read_cath_fasta
from margin.data_registry.conservation import attach_conservation
from margin.data_registry.homology import build_mmseqs_homology
from margin.data_registry.leakage import audit_benchmark_leakage, write_leakage_audit
from margin.data_registry.registry import RegistryTables, write_registry
from margin.preprocessing.structure import preprocess_domain_structure
from margin.provenance import (
    read_json,
    runtime_manifest,
    sha256_file,
    table_manifest,
    write_json,
    write_parquet,
    write_text,
)
from margin.studies.observability.config import ObservabilityStudyConfig


def prepare_replication_registry(config: ObservabilityStudyConfig) -> RegistryTables:
    """Quality-filter, leakage-filter, split, annotate, and freeze 240 new domains."""

    replication_config = load_config(config.paths.replication_config)
    output = replication_config.paths.run_dir
    registry_dir = output / "registry"
    if (registry_dir / "manifest.json").exists():
        from margin.data_registry.registry import load_registry

        return load_registry(registry_dir)

    classification = read_cath_domain_list(config.paths.cath_domain_list)
    sequences = read_cath_fasta(config.paths.cath_fasta)
    s40_ids = {
        line.strip()
        for line in config.paths.cath_s40_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    foundation_domains = pd.read_parquet(
        config.paths.foundation_run / "registry" / "domains.parquet"
    )
    candidates = _metadata_candidates(
        classification, sequences, s40_ids, foundation_domains, config
    )
    processed, exclusions = _preprocess_candidate_pool(
        candidates, sequences, replication_config, config
    )
    benchmark = pd.read_parquet(config.paths.benchmark_registry)
    homology_path = output / "leakage" / "homology_hits.parquet"
    hits, _ = build_mmseqs_homology(processed.domains, benchmark, homology_path, replication_config)
    leakage = audit_benchmark_leakage(processed.domains, benchmark, hits, replication_config)
    write_leakage_audit(output / "leakage", leakage, replication_config, [homology_path])
    eligible_ids = set(
        leakage.eligibility.loc[leakage.eligibility["eligible_for_training"], "domain_id"]
    )
    eligible = processed.domains.loc[processed.domains["domain_id"].isin(eligible_ids)].copy()
    if len(eligible) < config.replication.total_domains:
        raise ValueError(
            f"only {len(eligible)} replication domains remain after leakage filtering; "
            f"need {config.replication.total_domains}"
        )
    split_domains = _lock_splits(eligible, config)
    selected_ids = set(split_domains["domain_id"])
    selected = RegistryTables(
        domains=split_domains,
        residues=processed.residues.loc[
            processed.residues["domain_id"].isin(selected_ids)
        ].reset_index(drop=True),
    )
    conservation, alignments = _build_conservation(selected.domains, replication_config, config)
    selected = attach_conservation(selected, conservation, replication_config)
    write_registry(
        registry_dir,
        selected,
        replication_config,
        exclusions=exclusions,
        input_files=[
            config.paths.cath_domain_list,
            config.paths.cath_fasta,
            config.paths.cath_s40_list,
            homology_path,
            alignments,
        ],
    )
    _write_split_manifest(selected, leakage, conservation, config, replication_config)
    _write_teacher_exposure_manifest(config, replication_config)
    return selected


def _metadata_candidates(
    classification: pd.DataFrame,
    sequences: dict[str, str],
    s40_ids: set[str],
    foundation_domains: pd.DataFrame,
    config: ObservabilityStudyConfig,
) -> pd.DataFrame:
    rules = config.replication
    table = classification.loc[classification["domain_id"].isin(s40_ids)].copy()
    case_counts = Counter(domain_id.lower() for domain_id in s40_ids)
    case_collisions = {domain_id for domain_id in s40_ids if case_counts[domain_id.lower()] > 1}
    table = table.loc[~table["domain_id"].isin(case_collisions)]
    table["sequence"] = table["domain_id"].map(sequences)
    table = table.loc[table["sequence"].notna()]
    table = table.loc[table["sequence"].map(lambda value: not (set(value) - set(AA_ALPHABET)))]
    table = table.loc[
        table["sequence"].str.len().between(rules.minimum_length, rules.maximum_length)
    ]
    table = table.loc[
        table["resolution_angstrom"].isna()
        | table["resolution_angstrom"].le(rules.maximum_resolution_angstrom)
    ]
    table = table.loc[~table["domain_id"].isin(set(foundation_domains["domain_id"]))]
    if rules.exclude_foundation_cath_h:
        table = table.loc[~table["cath_h"].isin(set(foundation_domains["cath_h"]))]
    if rules.exclude_foundation_cath_t:
        table = table.loc[~table["cath_t"].isin(set(foundation_domains["cath_t"]))]
    # A single directory enumeration avoids issuing one expensive ``stat`` call
    # per S40 domain on mounted filesystems.
    available_structures = set(os.listdir(config.paths.cath_structures_dir))
    table = table.loc[table["domain_id"].isin(available_structures)]
    table["structure_path"] = table["domain_id"].map(
        lambda value: config.paths.cath_structures_dir / value
    )
    rng = np.random.default_rng(config.seed)
    table["selection_order"] = rng.permutation(len(table))
    table = table.sort_values("selection_order")
    if rules.one_domain_per_cath_t:
        table = table.drop_duplicates("cath_t", keep="first")
    return table.reset_index(drop=True)


def _preprocess_candidate_pool(
    candidates: pd.DataFrame,
    sequences: dict[str, str],
    replication_config: ProjectConfig,
    config: ObservabilityStudyConfig,
) -> tuple[RegistryTables, pd.DataFrame]:
    cache = config.paths.storage_dir / "preprocessed"
    cache.mkdir(parents=True, exist_ok=True)
    domain_rows = []
    residue_tables = []
    exclusions = []
    for record in candidates.itertuples(index=False):
        if len(domain_rows) >= config.replication.candidate_pool_size:
            break
        residue_path = cache / f"{record.domain_id}.parquet"
        summary_path = cache / f"{record.domain_id}.json"
        try:
            if residue_path.exists() and summary_path.exists():
                residues = pd.read_parquet(residue_path)
                summary = read_json(summary_path)
            else:
                residues, summary = preprocess_domain_structure(
                    record.domain_id,
                    sequences[record.domain_id],
                    record.structure_path,
                    record.chain_id,
                    replication_config.registry,
                )
                write_parquet(residue_path, residues)
                write_json(summary_path, summary)
        except (ValueError, RuntimeError, OSError) as error:
            exclusions.append(
                {
                    "domain_id": record.domain_id,
                    "stage": "structure",
                    "reason": f"preprocessing_failed:{error}",
                }
            )
            continue
        if summary["missing_fraction"] > config.replication.maximum_missing_fraction:
            exclusions.append(
                {
                    "domain_id": record.domain_id,
                    "stage": "structure",
                    "reason": "too_many_missing_backbone_residues",
                }
            )
            continue
        structure_path = Path(record.structure_path)
        domain_rows.append(
            {
                "domain_id": record.domain_id,
                "pdb_id": record.pdb_id,
                "chain_id": summary["selected_chain"],
                "sequence": sequences[record.domain_id],
                "length": len(sequences[record.domain_id]),
                "cath_c": record.cath_c,
                "cath_a": record.cath_a,
                "cath_t": record.cath_t,
                "cath_h": record.cath_h,
                "resolution_angstrom": record.resolution_angstrom,
                "structure_path": str(structure_path.resolve()),
                "structure_sha256": sha256_file(structure_path),
                "source_name": "CATH",
                "source_version": "4.4.0-S40",
                "source_url": "https://download.cathdb.info/cath/releases/all-releases/v4_4_0/",
                "is_experimental": True,
                "dataset": "CATH_Observability",
                "analysis_role": "training_candidate",
                "eligible_for_training": False,
                "missing_residue_count": summary["missing_residue_count"],
                "missing_fraction": summary["missing_fraction"],
                "helix_fraction": summary["helix_fraction"],
                "strand_fraction": summary["strand_fraction"],
            }
        )
        residue_tables.append(residues)
    if len(domain_rows) < config.replication.total_domains:
        raise ValueError(
            f"only {len(domain_rows)} structures passed preprocessing; "
            f"need at least {config.replication.total_domains}"
        )
    return (
        RegistryTables(pd.DataFrame(domain_rows), pd.concat(residue_tables, ignore_index=True)),
        pd.DataFrame(exclusions, columns=["domain_id", "stage", "reason"]),
    )


def _lock_splits(eligible: pd.DataFrame, config: ObservabilityStudyConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed + 1)
    table = eligible.copy()
    table["split_order"] = rng.permutation(len(table))
    table = table.sort_values(["cath_c", "split_order", "domain_id"])
    class_frames = [frame for _, frame in table.groupby("cath_c", observed=True)]
    interleaved = []
    while any(len(frame) for frame in class_frames):
        for index, frame in enumerate(class_frames):
            if len(frame):
                interleaved.append(frame.iloc[0])
                class_frames[index] = frame.iloc[1:]
    selected = pd.DataFrame(interleaved[: config.replication.total_domains]).reset_index(drop=True)
    counts = (
        config.replication.train_domains,
        config.replication.validation_domains,
        config.replication.locked_test_domains,
    )
    labels = (
        ["development_train"] * counts[0]
        + ["development_validation"] * counts[1]
        + ["locked_test"] * counts[2]
    )
    selected["observability_split"] = labels
    selected["analysis_role"] = np.where(
        selected["observability_split"].eq("locked_test"),
        "external_benchmark",
        "training_candidate",
    )
    selected["eligible_for_training"] = selected["observability_split"].eq("development_train")
    return selected.drop(columns=["split_order"])


def _build_conservation(
    domains: pd.DataFrame,
    replication_config: ProjectConfig,
    config: ObservabilityStudyConfig,
) -> tuple[pd.DataFrame, Path]:
    directory = replication_config.paths.run_dir / "conservation"
    directory.mkdir(parents=True, exist_ok=True)
    query = directory / "queries.fasta"
    alignments = directory / "alignments.tsv"
    write_text(
        query,
        "".join(f">{row.domain_id}\n{row.sequence}\n" for row in domains.itertuples(index=False)),
    )
    executable = shutil.which(replication_config.homology.executable)
    if executable is None:
        raise FileNotFoundError(
            f"MMseqs2 executable not found: {replication_config.homology.executable}"
        )
    with tempfile.TemporaryDirectory(prefix="observability-conservation-") as temporary:
        command = [
            executable,
            "easy-search",
            str(query),
            str(config.paths.cath_fasta),
            str(alignments),
            temporary,
            "-s",
            str(config.replication.homology_sensitivity),
            "-e",
            str(config.replication.homology_evalue),
            "--threads",
            str(config.replication.homology_threads),
            "--max-seqs",
            "500",
            "--format-output",
            "query,target,qstart,qaln,taln,evalue",
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                f"observability study conservation search failed: {completed.stderr[-2000:]}"
            )
    hits = pd.read_csv(
        alignments,
        sep="\t",
        names=["query", "target", "qstart", "qaln", "taln", "evalue"],
    )
    rows = []
    for domain in domains.itertuples(index=False):
        matches = np.zeros(domain.length, dtype=np.int64)
        observations = np.zeros(domain.length, dtype=np.int64)
        for hit in hits.loc[hits["query"].eq(domain.domain_id)].itertuples(index=False):
            if _cath_identifier(str(hit.target)) == domain.domain_id:
                continue
            position = int(hit.qstart) - 2
            for query_aa, target_aa in zip(str(hit.qaln), str(hit.taln), strict=True):
                if query_aa == "-":
                    continue
                position += 1
                target_aa = target_aa.upper()
                if target_aa not in AA_ALPHABET or position < 0 or position >= domain.length:
                    continue
                observations[position] += 1
                matches[position] += int(target_aa == domain.sequence[position])
        scores = np.divide(
            matches,
            observations,
            out=np.full(domain.length, 0.5, dtype=float),
            where=observations > 0,
        )
        rows.extend(
            {
                "domain_id": domain.domain_id,
                "position": position,
                "conservation_score": float(scores[position]),
                "homolog_observations": int(observations[position]),
                "conservation_method": "MMseqs2 CATH-v4.4 native-residue frequency",
            }
            for position in range(domain.length)
        )
    conservation = pd.DataFrame(rows)
    write_parquet(directory / "residue_conservation.parquet", conservation)
    return conservation, alignments


def _write_split_manifest(
    registry: RegistryTables,
    leakage,
    conservation: pd.DataFrame,
    config: ObservabilityStudyConfig,
    replication_config: ProjectConfig,
) -> None:
    split = registry.domains["observability_split"].value_counts().to_dict()
    overlap = registry.domains.groupby("observability_split")["cath_t"].apply(set)
    split_pairs = [
        (left, right) for left in overlap.index for right in overlap.index if left < right
    ]
    write_json(
        replication_config.paths.run_dir / "replication_lock.json",
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "lock_status": "LOCKED_BEFORE_TEACHER_SCORING",
            "seed": config.seed,
            "split_counts": split,
            "cath_t_cross_split_overlaps": {
                f"{left}__{right}": len(overlap[left] & overlap[right])
                for left, right in split_pairs
            },
            "leakage_summary": leakage.summary,
            "conservation_rows": len(conservation),
            "domains": table_manifest(
                replication_config.paths.registry_dir / "domains.parquet", registry.domains
            ),
        },
    )


def _write_teacher_exposure_manifest(
    config: ObservabilityStudyConfig, replication_config: ProjectConfig
) -> None:
    rows = []
    for teacher in replication_config.teacher_cache.teachers:
        if teacher.role == "sequence":
            continue
        rows.append(
            {
                "teacher_id": teacher.teacher_id,
                "model_revision": teacher.model_revision,
                "project_registry_overlap": "excluded_by_observability_rules",
                "teacher_pretraining_membership": "not_identifiable_from_released_checkpoint",
                "claim_allowed": "released_teacher_signal_on_locked_public_domains",
                "claim_not_allowed": "teacher_never_saw_test_protein",
            }
        )
    table = pd.DataFrame(rows)
    path = replication_config.paths.run_dir / "leakage" / "teacher_pretraining_exposure.parquet"
    write_parquet(path, table)
    write_json(
        path.with_suffix(".json"),
        {
            **runtime_manifest(config.paths.project_root),
            "scope": "teacher exposure limitation",
            "artifact": table_manifest(path, table),
        },
    )


def _cath_identifier(value: str) -> str:
    fields = value.split("|")
    return fields[-1].split("/", maxsplit=1)[0]
