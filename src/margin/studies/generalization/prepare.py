"""Prepare outcome-blind generalization study query sets and teacher controls."""

from __future__ import annotations

import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from margin.config import load_config
from margin.data_registry.homology import parse_mmseqs_hits
from margin.data_registry.registry import load_registry
from margin.decoys.generate import DecoyArtifacts, build_decoys, write_decoys
from margin.provenance import (
    runtime_manifest,
    table_manifest,
    write_json,
    write_parquet,
    write_text,
)
from margin.state_sampling.bank import StateBank, load_state_bank
from margin.studies.generalization.config import GeneralizationStudyConfig
from margin.teachers.requests import export_teacher_requests

AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")
SINGLE_MUTATION = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")


def prepare_generalization_inputs(config: GeneralizationStudyConfig) -> dict[str, Path]:
    """Create all pre-inference query tables without examining aggregate outcomes."""

    run = config.paths.run_dir
    run.mkdir(parents=True, exist_ok=True)
    architecture = prepare_architecture_queries(config)
    dms = prepare_dms_panel(config)
    requests = prepare_mif_requests(config)
    manifest_path = run / "input_manifest.json"
    write_json(
        manifest_path,
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "status": "PREPARED_WITHOUT_AGGREGATE_OUTCOME_INSPECTION",
            "architecture_queries": str(architecture),
            "dms_registry": str(dms["registry"]),
            "dms_variants": str(dms["variants"]),
            "mif_requests": str(requests),
        },
    )
    return {
        "architecture_queries": architecture,
        "mif_requests": requests,
        "manifest": manifest_path,
        **{f"dms_{name}": path for name, path in dms.items()},
    }


def prepare_architecture_queries(config: GeneralizationStudyConfig) -> Path:
    """Select fixed, label-blind native positions for the architecture matrix."""

    output = config.paths.run_dir / "architecture"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "query_rows.parquet"
    if path.exists():
        return path
    root = config.paths.observability_replication_run
    domains = pd.read_parquet(root / "registry" / "domains.parquet")
    positions = pd.read_parquet(root / "state_bank" / "positions.parquet")
    states = pd.read_parquet(root / "state_bank" / "states.parquet")
    native = states.loc[
        states["state_kind"].eq(config.architecture.state_kind),
        ["state_id", "domain_id", "state_sequence"],
    ].rename(columns={"state_sequence": "sequence"})
    rows: list[pd.DataFrame] = []
    metadata_columns = [
        "state_id",
        "domain_id",
        "position",
        "native_aa",
        "current_aa",
        "burial",
        "secondary_structure",
        "contact_class",
        "conservation_score",
        "conservation_class",
    ]
    native_positions = positions[metadata_columns].merge(
        native, on=["state_id", "domain_id"], validate="many_to_one"
    )
    split = domains.set_index("domain_id")["observability_split"]
    for (_, domain_id), frame in native_positions.groupby(
        ["state_id", "domain_id"], sort=True, observed=True
    ):
        frame = frame.sort_values("position")
        count = min(config.architecture.positions_per_domain, len(frame))
        selected = np.floor((np.arange(count) + 0.5) * len(frame) / count).astype(int)
        chosen = frame.iloc[selected].copy()
        chosen["observability_split"] = split.loc[domain_id]
        rows.append(chosen)
    query = pd.concat(rows, ignore_index=True).sort_values(
        ["state_id", "position"], ignore_index=True
    )
    if query.duplicated(["state_id", "position"]).any():
        raise ValueError("architecture query keys are not unique")
    write_parquet(path, query)
    write_json(
        output / "query_manifest.json",
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "selection": "deterministic_equal_width_position_bins_without_target_values",
            "positions_per_domain_maximum": config.architecture.positions_per_domain,
            "domains": int(query["domain_id"].nunique()),
            "rows": int(len(query)),
            "splits": {
                str(key): int(value)
                for key, value in query.groupby("observability_split", observed=True)["domain_id"]
                .nunique()
                .items()
            },
            "table": table_manifest(path, query),
        },
    )
    return path


def prepare_dms_panel(config: GeneralizationStudyConfig) -> dict[str, Path]:
    """Build a Tsuboyama single-mutant panel and exclude foundation and observability overlap."""

    output = config.paths.run_dir / "dms"
    output.mkdir(parents=True, exist_ok=True)
    registry_path = output / "assays.parquet"
    variants_path = output / "variants.parquet"
    query_path = output / "query_rows.parquet"
    exclusion_path = output / "exclusions.parquet"
    hit_path = output / "observability_homology_hits.parquet"
    if all(path.exists() for path in (registry_path, variants_path, query_path, exclusion_path)):
        return {
            "registry": registry_path,
            "variants": variants_path,
            "queries": query_path,
            "exclusions": exclusion_path,
            "homology_hits": hit_path,
        }

    metadata = pd.read_csv(config.paths.protein_gym_metadata)
    candidates = metadata.loc[
        metadata["DMS_id"].astype(str).str.contains(config.dms.source_pattern, regex=False)
    ].copy()
    candidates = candidates.rename(
        columns={
            "DMS_id": "assay_id",
            "DMS_filename": "filename",
            "target_seq": "sequence",
            "seq_len": "length",
        }
    )
    candidates = candidates[["assay_id", "filename", "sequence", "length", "pdb_file"]]
    candidates["sequence"] = candidates["sequence"].astype(str)
    invalid = candidates.loc[
        candidates["sequence"].map(lambda value: bool(set(value) - AA_ALPHABET))
    ]
    candidates = candidates.drop(index=invalid.index).reset_index(drop=True)

    all_variants = _read_single_mutants(config.paths.protein_gym_substitutions, candidates)
    counts = all_variants.groupby("assay_id", observed=True).size()
    too_small = set(counts.index[counts < config.dms.minimum_single_variants])
    candidates = candidates.loc[~candidates["assay_id"].isin(too_small)].copy()
    all_variants = all_variants.loc[all_variants["assay_id"].isin(candidates["assay_id"])]

    cath = pd.read_parquet(
        config.paths.observability_replication_run / "registry" / "domains.parquet"
    )[["domain_id", "sequence"]]
    hits = _run_homology(candidates, cath, hit_path, config)
    homologous = set(
        hits.loc[
            hits["sequence_identity"].ge(config.dms.identity_threshold)
            & hits[["query_coverage", "target_coverage"]]
            .min(axis=1)
            .ge(config.dms.minimum_bidirectional_coverage),
            "domain_id",
        ]
    )
    exact_sequences = set(cath["sequence"].astype(str))
    exact = set(candidates.loc[candidates["sequence"].isin(exact_sequences), "assay_id"])
    historical = set(
        pd.read_parquet(
            config.paths.project_root / "data/raw/benchmarks/benchmark_registry.parquet"
        )["benchmark_id"].astype(str)
    )
    exclusions: list[dict[str, str]] = []
    for assay_id in invalid["assay_id"].astype(str):
        exclusions.append({"assay_id": assay_id, "reason": "noncanonical_sequence"})
    for assay_id in sorted(too_small):
        exclusions.append({"assay_id": assay_id, "reason": "too_few_single_mutants"})
    for assay_id in candidates["assay_id"].astype(str):
        if assay_id in historical:
            reason = "historical_foundation_assay"
        elif assay_id in exact:
            reason = "exact_observability_sequence"
        elif assay_id in homologous:
            reason = "observability_homology_threshold"
        else:
            continue
        exclusions.append({"assay_id": assay_id, "reason": reason})
    excluded_ids = {row["assay_id"] for row in exclusions}
    selected = candidates.loc[~candidates["assay_id"].isin(excluded_ids)].copy()
    selected["variant_count"] = selected["assay_id"].map(counts).astype(int)
    selected["evaluation_role"] = "locked_external_stability_dms"
    selected = selected.sort_values("assay_id", ignore_index=True)
    variants = all_variants.loc[all_variants["assay_id"].isin(selected["assay_id"])].copy()
    variants = variants.sort_values(["assay_id", "position", "mutant"], ignore_index=True)
    queries = (
        variants[["assay_id", "position", "wild_type"]]
        .drop_duplicates(["assay_id", "position"])
        .merge(selected[["assay_id", "sequence"]], on="assay_id", validate="many_to_one")
        .rename(columns={"assay_id": "state_id"})
        .sort_values(["state_id", "position"], ignore_index=True)
    )
    queries["domain_id"] = queries["state_id"]
    exclusions_frame = pd.DataFrame(exclusions, columns=["assay_id", "reason"]).drop_duplicates()
    write_parquet(registry_path, selected)
    write_parquet(variants_path, variants)
    write_parquet(query_path, queries)
    write_parquet(exclusion_path, exclusions_frame)
    write_json(
        output / "panel_manifest.json",
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "source": "ProteinGym_v1.3_Tsuboyama_2023_substitutions",
            "outcome_blinding": "no_aggregate_DMS_metric_computed_during_panel_preparation",
            "candidate_assays": int(
                len(
                    metadata.loc[
                        metadata["DMS_id"]
                        .astype(str)
                        .str.contains(config.dms.source_pattern, regex=False)
                    ]
                )
            ),
            "selected_assays": int(len(selected)),
            "selected_variants": int(len(variants)),
            "homology_rule": {
                "identity": config.dms.identity_threshold,
                "minimum_bidirectional_coverage": config.dms.minimum_bidirectional_coverage,
                "evalue": config.dms.homology_evalue,
            },
            "tables": [
                table_manifest(registry_path, selected),
                table_manifest(variants_path, variants),
                table_manifest(query_path, queries),
                table_manifest(exclusion_path, exclusions_frame),
                table_manifest(hit_path, hits),
            ],
        },
    )
    return {
        "registry": registry_path,
        "variants": variants_path,
        "queries": query_path,
        "exclusions": exclusion_path,
        "homology_hits": hit_path,
    }


def prepare_mif_requests(config: GeneralizationStudyConfig) -> Path:
    """Create native paired and degree-preserving rewired-graph MIF requests."""

    output = config.paths.run_dir / "mif_requests"
    request_path = output / "requests.parquet"
    if request_path.exists() and (output / "manifest.json").exists():
        return request_path
    replication_config = load_config(config.paths.observability_replication_config)
    registry = load_registry(config.paths.observability_replication_run / "registry")
    full_bank = load_state_bank(config.paths.observability_replication_run / "state_bank")
    native_states = full_bank.states.loc[full_bank.states["state_kind"].eq("native_reference")]
    native_ids = set(native_states["state_id"])
    native_bank = StateBank(
        states=native_states.reset_index(drop=True),
        positions=full_bank.positions.loc[
            full_bank.positions["state_id"].isin(native_ids)
        ].reset_index(drop=True),
    )
    generated = build_decoys(registry, replication_config)
    contact_ids = set(
        generated.decoys.loc[generated.decoys["decoy_type"].eq("contact_rewired"), "decoy_id"]
    )
    contact = DecoyArtifacts(
        decoys=generated.decoys.loc[generated.decoys["decoy_id"].isin(contact_ids)].copy(),
        residues=generated.residues.iloc[0:0].copy(),
        edges=generated.edges.loc[generated.edges["decoy_id"].isin(contact_ids)].copy(),
        skipped=generated.skipped.iloc[0:0].copy(),
    )
    write_decoys(config.paths.run_dir / "mif_decoys", contact, replication_config)
    export_teacher_requests(output, native_bank, registry, contact, replication_config)
    return request_path


def _read_single_mutants(zip_path: Path, assays: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        for assay in assays.itertuples(index=False):
            member = f"DMS_ProteinGym_substitutions/{assay.filename}"
            if member not in names:
                raise FileNotFoundError(f"ProteinGym archive lacks {member}")
            with archive.open(member) as handle:
                frame = pd.read_csv(handle)
            parsed = frame["mutant"].astype(str).str.extract(SINGLE_MUTATION)
            keep = parsed.notna().all(axis=1) & frame["DMS_score"].notna()
            parsed = parsed.loc[keep]
            result = pd.DataFrame(
                {
                    "assay_id": assay.assay_id,
                    "position": parsed[1].astype(int).to_numpy() - 1,
                    "wild_type": parsed[0].to_numpy(),
                    "mutant": parsed[2].to_numpy(),
                    "effect": frame.loc[keep, "DMS_score"].astype(float).to_numpy(),
                }
            )
            sequence = str(assay.sequence)
            valid_position = result["position"].between(0, len(sequence) - 1)
            result = result.loc[valid_position]
            expected = pd.Series(
                np.asarray(list(sequence))[result["position"].to_numpy(dtype=int)],
                index=result.index,
            )
            result = result.loc[result["wild_type"].eq(expected)].copy()
            rows.append(result)
    return pd.concat(rows, ignore_index=True)


def _run_homology(
    assays: pd.DataFrame,
    cath: pd.DataFrame,
    output_path: Path,
    config: GeneralizationStudyConfig,
) -> pd.DataFrame:
    if output_path.exists():
        return pd.read_parquet(output_path)
    executable = config.paths.mmseqs_executable
    if not executable.is_file():
        raise FileNotFoundError(f"MMseqs2 executable not found: {executable}")
    query_path = output_path.with_suffix(".queries.fasta")
    target_path = output_path.with_suffix(".targets.fasta")
    write_text(query_path, _fasta(assays["assay_id"], assays["sequence"]))
    write_text(target_path, _fasta(cath["domain_id"], cath["sequence"]))
    with tempfile.TemporaryDirectory(
        prefix="generalization-mmseqs-", dir=output_path.parent
    ) as name:
        temporary = Path(name)
        raw = temporary / "hits.tsv"
        command = [
            str(executable),
            "easy-search",
            str(query_path),
            str(target_path),
            str(raw),
            str(temporary / "work"),
            "--format-output",
            "query,target,fident,qcov,tcov",
            "-s",
            str(config.dms.homology_sensitivity),
            "-e",
            str(config.dms.homology_evalue),
            "--max-seqs",
            str(len(cath)),
            "--threads",
            str(config.dms.homology_threads),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"MMseqs2 failed: {completed.stderr[-2000:]}")
        hits = parse_mmseqs_hits(raw)
    write_parquet(output_path, hits)
    return hits


def _fasta(identifiers: pd.Series, sequences: pd.Series) -> str:
    return "".join(
        f">{identifier}\n{sequence}\n"
        for identifier, sequence in zip(identifiers, sequences, strict=True)
    )
