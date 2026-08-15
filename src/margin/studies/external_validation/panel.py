"""Outcome-blind construction of the post-lock FireProt C+ confirmation panel."""

from __future__ import annotations

import re
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from margin.config import load_config
from margin.constants import AA_ALPHABET
from margin.provenance import (
    runtime_manifest,
    table_manifest,
    write_json,
    write_parquet,
    write_text,
)
from margin.studies.mechanisms.prepare import (
    _preprocess_silently,
    _read_chain,
    _structure_descriptors,
)
from margin.studies.stability.prepare import _teacher_requests

PDB_ID = re.compile(r"^[0-9A-Z]{4}$")
POPULATION = "fireprot_hf_cross_platform"
IDENTITY_COLUMNS = [
    "experiment_id",
    "protein_name",
    "uniprot_id",
    "pdb_id_corrected",
    "chain",
    "pdb_position",
    "wild_type",
    "mutation",
    "pdb_sequence",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ExternalValidationPaths(StrictModel):
    project_root: Path
    run_dir: Path
    storage_dir: Path
    source_csv: Path
    foundation_config: Path
    stability_config: Path
    mmseqs_executable: Path
    observability_domains: Path
    generalization_assays: Path
    counterfactual_domains: Path
    mechanism_domains: Path
    action_validation_domains: Path
    stability_domains: Path


class ExternalValidationPanel(StrictModel):
    source_population: Literal["FireProtDB_homologue_free"]
    minimum_unique_variants: PositiveInt
    minimum_unique_positions: PositiveInt
    maximum_length: PositiveInt
    require_complete_backbone: bool
    prior_homology_maximum_identity: float = Field(ge=0, le=1)
    prior_homology_minimum_bidirectional_coverage: float = Field(ge=0, le=1)
    homology_evalue: float = Field(gt=0)
    homology_sensitivity: float = Field(gt=0)
    homology_threads: PositiveInt
    minimum_selected_domains: PositiveInt


class ExternalValidationInference(StrictModel):
    primary_score: str
    comparator_score: str
    primary_metric: str
    secondary_metric: str
    bootstrap_replicates: PositiveInt
    confidence_level: float = Field(gt=0, lt=1)
    minimum_positive_domain_fraction: float = Field(ge=0, le=1)
    require_primary_ci_lower_positive: bool
    require_secondary_point_positive: bool
    effect_convention: str
    duplicate_measurement_aggregation: str
    routing_allowed: bool
    changes_primary_decision: bool


class ExternalValidationConfig(StrictModel):
    schema_version: Literal["external_validation.v1"]
    status: Literal["FROZEN_BEFORE_CROSS_PLATFORM_MODEL_SCORING"]
    seed: int
    paths: ExternalValidationPaths
    panel: ExternalValidationPanel
    inference: ExternalValidationInference


def load_external_validation_config(path: Path) -> ExternalValidationConfig:
    """Load the frozen protocol and resolve all filesystem paths."""

    path = path.resolve()
    with path.open(encoding="utf-8") as handle:
        config = ExternalValidationConfig.model_validate(yaml.safe_load(handle))
    project_root = _resolve(path.parent, config.paths.project_root)
    config.paths.project_root = project_root
    for name in ExternalValidationPaths.model_fields:
        if name == "project_root":
            continue
        setattr(config.paths, name, _resolve(project_root, getattr(config.paths, name)))
    return config


def prepare_external_validation_panel(config: ExternalValidationConfig) -> dict[str, Path]:
    """Select and preprocess the panel without loading the FireProt ddG column."""

    panel_dir = config.paths.run_dir / "panel"
    request_dir = config.paths.run_dir / "teacher_requests"
    structure_dir = config.paths.storage_dir / "structures" / "experimental"
    panel_dir.mkdir(parents=True, exist_ok=True)
    request_dir.mkdir(parents=True, exist_ok=True)
    structure_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "candidate_scan": panel_dir / "candidate_scan.parquet",
        "homology_hits": panel_dir / "prior_homology_hits.parquet",
        "exclusions": panel_dir / "exclusions.parquet",
        "domains": panel_dir / "domains.parquet",
        "residues": panel_dir / "residues.parquet",
        "mutation_index": panel_dir / "mutation_index.parquet",
        "queries": panel_dir / "query_rows.parquet",
        "requests": request_dir / "requests.parquet",
        "structures": request_dir / "structures.parquet",
        "manifest": panel_dir / "manifest.json",
        "lock": config.paths.run_dir / "protocol_lock.json",
    }
    if all(path.exists() for path in paths.values()):
        return paths
    _require_inputs(config)
    raw = _read_identity_only(config.paths.source_csv)
    scan, mutation_index, scan_exclusions = _scan_candidates(raw, config)
    eligible = scan.loc[scan["metadata_eligible"]].copy()
    prior = _opened_sequence_registry(config)
    hits = _run_homology(eligible, prior, paths["homology_hits"], config)
    near = set(
        hits.loc[
            hits["sequence_identity"].ge(config.panel.prior_homology_maximum_identity)
            & hits[["query_coverage", "target_coverage"]]
            .min(axis=1)
            .ge(config.panel.prior_homology_minimum_bidirectional_coverage),
            "domain_id",
        ]
    )
    exclusions = [*scan_exclusions]
    exclusions.extend(
        {"domain_id": domain_id, "reason": "homologous_to_previously_opened_sequence"}
        for domain_id in sorted(near)
    )
    structure_candidates = eligible.loc[~eligible["domain_id"].isin(near)].copy()
    domains, residues, structure_exclusions = _materialize_structures(
        structure_candidates, structure_dir, config
    )
    exclusions.extend(structure_exclusions)
    if len(domains) < config.panel.minimum_selected_domains:
        raise RuntimeError(
            f"cross-platform panel requires {config.panel.minimum_selected_domains} proteins; "
            f"only {len(domains)} passed frozen filters"
        )
    selected = set(domains["domain_id"])
    mutation_index = mutation_index.loc[mutation_index["domain_id"].isin(selected)].copy()
    observed = mutation_index.groupby("domain_id", observed=True).agg(
        variant_count=("mutant", "size"), mutated_position_count=("position", "nunique")
    )
    domains = domains.merge(observed, on="domain_id", validate="one_to_one")
    queries = (
        mutation_index[["domain_id", "position", "wild_type"]]
        .drop_duplicates(["domain_id", "position"])
        .merge(domains[["domain_id", "sequence"]], on="domain_id", validate="many_to_one")
        .assign(state_id=lambda frame: frame["domain_id"])[
            ["state_id", "domain_id", "position", "wild_type", "sequence"]
        ]
        .sort_values(["domain_id", "position"], ignore_index=True)
    )
    request_tables = _teacher_requests(domains, residues, request_dir)
    exclusion_table = pd.DataFrame(exclusions, columns=["domain_id", "reason"]).drop_duplicates(
        ignore_index=True
    )
    tables = {
        "candidate_scan": scan,
        "homology_hits": hits,
        "exclusions": exclusion_table,
        "domains": domains.sort_values("domain_id", ignore_index=True),
        "residues": residues.sort_values(["domain_id", "position"], ignore_index=True),
        "mutation_index": mutation_index.sort_values(
            ["domain_id", "position", "mutant"], ignore_index=True
        ),
        "queries": queries,
        "requests": request_tables["requests"],
        "structures": request_tables["structures"],
    }
    for name, table in tables.items():
        write_parquet(paths[name], table)
    lock = {
        **runtime_manifest(config.paths.project_root),
        "schema_version": config.schema_version,
        "status": config.status,
        "protocol": str(config.paths.project_root / "configs/external_validation.yaml"),
        "selection_uses_outcome_magnitudes": False,
        "selection_uses_outcome_availability": True,
        "selected_domain_ids": sorted(selected),
        "selected_domains": len(domains),
        "selected_variants": len(mutation_index),
        "selected_query_positions": len(queries),
        "primary_score": config.inference.primary_score,
        "comparator_score": config.inference.comparator_score,
        "effect_values_opened": False,
        "changes_primary_decision": False,
    }
    write_json(paths["lock"], lock)
    write_json(
        paths["manifest"],
        {
            **lock,
            "source": "ThermoMPNN official fireprot_HF.csv",
            "source_column_read_exclusion": ["ddG", "dTm"],
            "structure_source": "RCSB PDB coordinate download",
            "tables": [table_manifest(paths[name], table) for name, table in tables.items()],
        },
    )
    return paths


def _require_inputs(config: ExternalValidationConfig) -> None:
    required = [
        config.paths.source_csv,
        config.paths.foundation_config,
        config.paths.stability_config,
        config.paths.mmseqs_executable,
        config.paths.observability_domains,
        config.paths.generalization_assays,
        config.paths.counterfactual_domains,
        config.paths.mechanism_domains,
        config.paths.action_validation_domains,
        config.paths.stability_domains,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"cross-platform inputs are missing: {missing}")


def _read_identity_only(path: Path) -> pd.DataFrame:
    columns = list(pd.read_csv(path, nrows=0).columns)
    if "ddG" not in columns:
        raise ValueError("FireProt source lacks the registered ddG endpoint")
    return pd.read_csv(path, usecols=IDENTITY_COLUMNS)


def _scan_candidates(
    raw: pd.DataFrame, config: ExternalValidationConfig
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, str]]]:
    frame = raw.copy()
    frame["pdb_id_corrected"] = frame["pdb_id_corrected"].astype(str).str.upper()
    frame["chain"] = frame["chain"].astype(str)
    frame["pdb_sequence"] = frame["pdb_sequence"].astype(str).str.upper()
    frame["pdb_position"] = pd.to_numeric(frame["pdb_position"], errors="coerce")
    frame["wild_type"] = frame["wild_type"].astype(str).str.upper()
    frame["mutation"] = frame["mutation"].astype(str).str.upper()
    frame["domain_id"] = frame["pdb_id_corrected"] + ":" + frame["chain"]
    alphabet = set(AA_ALPHABET)
    row_valid = (
        frame["pdb_id_corrected"].map(lambda value: bool(PDB_ID.fullmatch(value)))
        & frame["chain"].str.len().eq(1)
        & frame["pdb_sequence"].map(lambda value: bool(value) and set(value) <= alphabet)
        & frame["pdb_sequence"].str.len().le(config.panel.maximum_length)
        & frame["pdb_position"].notna()
        & frame["wild_type"].isin(alphabet)
        & frame["mutation"].isin(alphabet)
        & frame["wild_type"].ne(frame["mutation"])
    )
    valid = frame.loc[row_valid].copy()
    valid["pdb_position"] = valid["pdb_position"].astype(int)
    position_valid = np.asarray(
        [
            0 <= position < len(sequence) and sequence[position] == wild
            for position, sequence, wild in zip(
                valid["pdb_position"],
                valid["pdb_sequence"],
                valid["wild_type"],
                strict=True,
            )
        ]
    )
    valid = valid.loc[position_valid].copy()
    valid["position"] = valid["pdb_position"]
    valid = valid.rename(columns={"mutation": "mutant", "pdb_sequence": "sequence"})
    mutation_index = valid[["domain_id", "position", "wild_type", "mutant"]].drop_duplicates(
        ignore_index=True
    )
    counts = mutation_index.groupby("domain_id", observed=True).agg(
        candidate_variant_count=("mutant", "size"),
        candidate_position_count=("position", "nunique"),
    )
    metadata = (
        valid.groupby("domain_id", observed=True)
        .agg(
            pdb_id=("pdb_id_corrected", "first"),
            chain_id=("chain", "first"),
            sequence=("sequence", "first"),
            sequence_count=("sequence", "nunique"),
            protein_name=("protein_name", "first"),
            uniprot_id=("uniprot_id", "first"),
        )
        .join(counts)
        .reset_index()
    )
    metadata["length"] = metadata["sequence"].str.len()
    metadata["metadata_eligible"] = (
        metadata["sequence_count"].eq(1)
        & metadata["candidate_variant_count"].ge(config.panel.minimum_unique_variants)
        & metadata["candidate_position_count"].ge(config.panel.minimum_unique_positions)
    )
    exclusions: list[dict[str, str]] = []
    for row in metadata.loc[~metadata["metadata_eligible"]].itertuples(index=False):
        if row.sequence_count != 1:
            reason = "inconsistent_aligned_pdb_sequence"
        elif row.candidate_variant_count < config.panel.minimum_unique_variants:
            reason = "too_few_unique_variants"
        else:
            reason = "too_few_unique_positions"
        exclusions.append({"domain_id": row.domain_id, "reason": reason})
    return metadata.sort_values("domain_id", ignore_index=True), mutation_index, exclusions


def _opened_sequence_registry(config: ExternalValidationConfig) -> pd.DataFrame:
    frames = []
    specifications = [
        (config.paths.observability_domains, "domain_id", "observability_cath"),
        (config.paths.generalization_assays, "assay_id", "generalization_dms"),
        (config.paths.counterfactual_domains, "domain_id", "counterfactuals"),
        (config.paths.mechanism_domains, "domain_id", "mechanisms"),
        (config.paths.action_validation_domains, "domain_id", "action_validation"),
        (config.paths.stability_domains, "domain_id", "stability"),
    ]
    for path, identifier, source in specifications:
        table = pd.read_parquet(path, columns=[identifier, "sequence"])
        frames.append(table.rename(columns={identifier: "target_id"}).assign(target_source=source))
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        ["target_id", "sequence", "target_source"], ignore_index=True
    )


def _run_homology(
    candidates: pd.DataFrame,
    targets: pd.DataFrame,
    output_path: Path,
    config: ExternalValidationConfig,
) -> pd.DataFrame:
    query_path = output_path.with_suffix(".queries.fasta")
    target_path = output_path.with_suffix(".targets.fasta")
    write_text(query_path, _fasta(candidates["domain_id"], candidates["sequence"]))
    encoded = [f"target_{index:05d}" for index in range(len(targets))]
    write_text(target_path, _fasta(pd.Series(encoded), targets["sequence"]))
    lookup = targets.assign(encoded_target=encoded)[
        ["encoded_target", "target_id", "target_source"]
    ]
    with tempfile.TemporaryDirectory(
        prefix="external-validation-mmseqs-", dir=output_path.parent
    ) as name:
        temporary = Path(name)
        raw = temporary / "hits.tsv"
        command = [
            str(config.paths.mmseqs_executable),
            "easy-search",
            str(query_path),
            str(target_path),
            str(raw),
            str(temporary / "work"),
            "--format-output",
            "query,target,fident,qcov,tcov,evalue",
            "-s",
            str(config.panel.homology_sensitivity),
            "-e",
            str(config.panel.homology_evalue),
            "--max-seqs",
            str(len(targets)),
            "--threads",
            str(config.panel.homology_threads),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"MMseqs2 failed: {completed.stderr[-2000:]}")
        if raw.exists() and raw.stat().st_size:
            hits = pd.read_csv(
                raw,
                sep="\t",
                names=["domain_id", "encoded_target", "fident", "qcov", "tcov", "evalue"],
            )
        else:
            hits = pd.DataFrame(
                columns=["domain_id", "encoded_target", "fident", "qcov", "tcov", "evalue"]
            )
    for column in ("fident", "qcov", "tcov", "evalue"):
        hits[column] = pd.to_numeric(hits[column], errors="coerce")
    for column in ("fident", "qcov", "tcov"):
        if len(hits) and hits[column].max() > 1:
            hits[column] /= 100.0
    return hits.merge(lookup, on="encoded_target", how="left", validate="many_to_one").rename(
        columns={
            "fident": "sequence_identity",
            "qcov": "query_coverage",
            "tcov": "target_coverage",
        }
    )[
        [
            "domain_id",
            "target_id",
            "target_source",
            "sequence_identity",
            "query_coverage",
            "target_coverage",
            "evalue",
        ]
    ]


def _materialize_structures(
    candidates: pd.DataFrame,
    structure_dir: Path,
    config: ExternalValidationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, str]]]:
    foundation = load_config(config.paths.foundation_config)
    domain_rows: list[dict[str, Any]] = []
    residue_frames = []
    exclusions = []
    for row in candidates.sort_values("domain_id").itertuples(index=False):
        path = structure_dir / f"{row.pdb_id}.pdb"
        try:
            _download_pdb(str(row.pdb_id), path)
            parsed = _read_chain(path, str(row.chain_id))
            if parsed["sequence"] != row.sequence:
                raise ValueError("coordinate_chain_sequence_mismatch")
            if config.panel.require_complete_backbone and not parsed["complete_backbone"]:
                raise ValueError("incomplete_backbone")
            residues, summary = _preprocess_silently(
                str(row.domain_id),
                str(row.sequence),
                path,
                str(row.chain_id),
                foundation.registry,
            )
            if (
                config.panel.require_complete_backbone
                and not residues["has_complete_backbone"].all()
            ):
                raise ValueError("preprocessed_incomplete_backbone")
        except (OSError, RuntimeError, ValueError) as error:
            exclusions.append(
                {
                    "domain_id": str(row.domain_id),
                    "reason": f"structure_failure:{error}",
                }
            )
            continue
        residues["source"] = "FireProtDB_homologue_free"
        residues["evaluation_population"] = POPULATION
        residue_frames.append(residues)
        domain_rows.append(
            {
                "domain_id": row.domain_id,
                "pdb_id": row.pdb_id,
                "chain_id": row.chain_id,
                "protein_name": row.protein_name,
                "uniprot_id": row.uniprot_id,
                "sequence": row.sequence,
                "length": int(row.length),
                "structure_path": str(path.resolve()),
                "source": "FireProtDB_homologue_free",
                "platform": "heterogeneous_experimental_thermodynamics",
                "structure_kind": "experimental_PDB",
                "evaluation_role": "external_validation_cplus_confirmation",
                "evaluation_population": POPULATION,
                "helix_fraction": float(summary["helix_fraction"]),
                "strand_fraction": float(summary["strand_fraction"]),
            }
        )
    if not domain_rows:
        return pd.DataFrame(), pd.DataFrame(), exclusions
    domains = pd.DataFrame(domain_rows)
    residues = pd.concat(residue_frames, ignore_index=True)
    descriptors = _structure_descriptors(residues)
    return domains.merge(descriptors, on="domain_id", validate="one_to_one"), residues, exclusions


def _download_pdb(pdb_id: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "MARGIN/external-validation"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            if not payload.startswith(b"HEADER") and b"ATOM  " not in payload[:20000]:
                raise OSError(f"RCSB response for {pdb_id} is not a PDB coordinate file")
            temporary = path.with_suffix(".pdb.part")
            temporary.write_bytes(payload)
            temporary.replace(path)
            return
        except (OSError, TimeoutError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(1 + attempt)
    raise OSError(f"failed to download {pdb_id}: {last_error}")


def _fasta(identifiers: pd.Series, sequences: pd.Series) -> str:
    return "".join(
        f">{identifier}\n{sequence}\n"
        for identifier, sequence in zip(identifiers, sequences, strict=True)
    )


def _resolve(base: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (base / value).resolve()
