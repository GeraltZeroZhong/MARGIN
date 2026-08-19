"""Outcome-blind preparation of the dense mechanism study audit panel."""

from __future__ import annotations

import re
import subprocess
import tempfile
import warnings
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1

from margin.config import load_config
from margin.constants import AA_ALPHABET, BACKBONE_ATOMS
from margin.decoys.graph import contact_edges
from margin.preprocessing.structure import preprocess_domain_structure
from margin.provenance import (
    runtime_manifest,
    table_manifest,
    write_json,
    write_parquet,
    write_text,
)
from margin.studies.counterfactuals.prepare import _design_family
from margin.studies.mechanisms.config import MechanismStudyConfig

SINGLE_MUTATION = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")
NATURAL_NAME = re.compile(r"^(?:v2_)?[0-9A-Za-z]{4}\.pdb$")
DE_NOVO_NAME = re.compile(r"^(?:EA\||GG\||XX\||EHEE_|EEHEE_|HHH_|HEEH_|r\d+_.*TrROS_Hall)")


def prepare_mechanism_panel(config: MechanismStudyConfig) -> dict[str, Path]:
    """Select new dense domains without inspecting stability-effect magnitudes."""

    output = config.paths.run_dir / "panel"
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "candidate_scan": output / "candidate_scan.parquet",
        "domains": output / "domains.parquet",
        "variants": output / "variants.parquet",
        "residues": output / "residues.parquet",
        "queries": output / "query_rows.parquet",
        "exclusions": output / "exclusions.parquet",
        "homology_hits": output / "prior_homology_hits.parquet",
        "donor_pool": output / "donor_pool.parquet",
        "matched_decoys": output / "matched_real_decoys.parquet",
        "data_audit": config.paths.run_dir / "reports" / "data_audit.md",
        "manifest": output / "manifest.json",
    }
    if all(path.exists() for path in paths.values()):
        return paths
    _require_inputs(config)

    scan, exclusions = _scan_archive(config)
    candidates = scan.loc[scan["metadata_eligible"]].copy()
    prior = _opened_sequence_registry(config)
    homology = _run_homology(candidates, prior, paths["homology_hits"], config)
    near_duplicate = set(
        homology.loc[
            homology["sequence_identity"].ge(config.panel.near_duplicate_identity)
            & homology[["query_coverage", "target_coverage"]]
            .min(axis=1)
            .ge(config.panel.near_duplicate_minimum_coverage),
            "domain_id",
        ]
    )
    exclusions.extend(
        {"domain_id": domain_id, "reason": "opened_exact_or_near_duplicate_sequence"}
        for domain_id in sorted(near_duplicate)
    )
    eligible = candidates.loc[~candidates["domain_id"].isin(near_duplicate)].copy()
    selected = _select_panel(eligible, config)
    selected_ids = set(selected["domain_id"])
    exclusions.extend(
        {"domain_id": row.domain_id, "reason": "deterministic_stratum_or_family_quota"}
        for row in eligible.loc[~eligible["domain_id"].isin(selected_ids)].itertuples(index=False)
    )

    domains, residues = _materialize_domains_and_residues(selected, config)
    variants = _materialize_variants(selected, config)
    observed = variants.groupby("domain_id", observed=True).agg(
        variant_count=("effect", "size"),
        mutated_position_count=("position", "nunique"),
    )
    domains = domains.merge(observed, on="domain_id", validate="one_to_one")
    if (domains["variant_count"] < config.panel.minimum_single_variants).any():
        raise ValueError("selected panel lost the registered minimum variant count")
    if (domains["mutated_position_count"] < config.panel.minimum_unique_positions).any():
        raise ValueError("selected panel lost the registered minimum mutated-position count")
    queries = (
        variants[["domain_id", "position", "wild_type"]]
        .drop_duplicates(["domain_id", "position"])
        .merge(domains[["domain_id", "sequence"]], on="domain_id", validate="many_to_one")
        .assign(state_id=lambda frame: frame["domain_id"])[
            ["state_id", "domain_id", "position", "wild_type", "sequence"]
        ]
        .sort_values(["domain_id", "position"], ignore_index=True)
    )

    donor_pool, matched_decoys = _match_real_structure_decoys(
        selected,
        scan,
        domains,
        config,
    )
    exclusion_table = pd.DataFrame(exclusions, columns=["domain_id", "reason"]).drop_duplicates(
        ignore_index=True
    )
    tables = {
        "candidate_scan": scan,
        "domains": domains,
        "variants": variants,
        "residues": residues,
        "queries": queries,
        "exclusions": exclusion_table,
        "homology_hits": homology,
        "donor_pool": donor_pool,
        "matched_decoys": matched_decoys,
    }
    for name, table in tables.items():
        write_parquet(paths[name], table)
    write_text(paths["data_audit"], _data_audit_report(scan, domains, variants, queries))
    write_json(
        paths["manifest"],
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "status": "PREPARED_BEFORE_MECHANISMS_MODEL_SCORING",
            "selection_uses_outcome_magnitudes": False,
            "selection_uses_outcome_availability": True,
            "panel_role": "new_domain_dense_in_distribution_mechanism_audit",
            "independent_external_confirmation": False,
            "effect_convention": "effect=ddG_ML; positive means stabilizing",
            "selected_domains": int(len(domains)),
            "selected_domains_by_stratum": {
                str(key): int(value) for key, value in domains["stratum"].value_counts().items()
            },
            "selected_variants": int(len(variants)),
            "selected_query_positions": int(len(queries)),
            "matched_real_decoys": int(len(matched_decoys)),
            "tables": [table_manifest(paths[name], table) for name, table in tables.items()],
        },
    )
    return paths


def _require_inputs(config: MechanismStudyConfig) -> None:
    required = [
        config.paths.megascale_archive,
        config.paths.megascale_structures,
        config.paths.mmseqs_executable,
        config.paths.generalization_run / "dms" / "assays.parquet",
        config.paths.counterfactual_run / "panel" / "domains.parquet",
        config.paths.observability_replication_run / "registry" / "domains.parquet",
        config.paths.counterfactual_run / "evaluation" / "project_decision.parquet",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"mechanism study inputs are missing: {missing}")
    decision = pd.read_parquet(
        config.paths.counterfactual_run / "evaluation" / "project_decision.parquet"
    )
    if str(decision["decision"].iloc[0]) != "RETAIN_GENERALIZATION_CLOSE_COUNTERFACTUALS":
        raise RuntimeError(
            "counterfactual study decision does not match the immutable mechanism study premise"
        )


def _scan_archive(config: MechanismStudyConfig) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    counts: Counter[str] = Counter()
    positions: dict[str, set[int]] = defaultdict(set)
    sequences: dict[str, str] = {}
    clusters: dict[str, str] = {}
    missing_effect: Counter[str] = Counter()
    columns = ["WT_name", "WT_cluster", "aa_seq", "mut_type", "ddG_ML"]
    with (
        zipfile.ZipFile(config.paths.megascale_archive) as archive,
        archive.open(config.panel.megascale_member) as handle,
    ):
        for chunk in pd.read_csv(handle, usecols=columns, chunksize=200_000, low_memory=False):
            wild = chunk.loc[
                chunk["mut_type"].eq("wt"), ["WT_name", "WT_cluster", "aa_seq"]
            ].drop_duplicates()
            for row in wild.itertuples(index=False):
                name = str(row.WT_name)
                sequence = str(row.aa_seq)
                if name in sequences and sequences[name] != sequence:
                    raise ValueError(f"inconsistent WT sequence for {name}")
                sequences[name] = sequence
                clusters[name] = str(row.WT_cluster)
            parsed = chunk["mut_type"].astype(str).str.extract(SINGLE_MUTATION)
            mutation = parsed.notna().all(axis=1)
            available = mutation & pd.to_numeric(chunk["ddG_ML"], errors="coerce").notna()
            counts.update(chunk.loc[available, "WT_name"].astype(str))
            missing_effect.update(chunk.loc[mutation & ~available, "WT_name"].astype(str))
            for name, position in zip(
                chunk.loc[available, "WT_name"].astype(str),
                parsed.loc[available, 1].astype(int),
                strict=True,
            ):
                positions[name].add(int(position))

    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    for name, sequence in sorted(sequences.items()):
        stratum = _stratum(name)
        domain_id = f"megascale:{name}"
        filename = name.replace("|", ":")
        if not filename.endswith(".pdb"):
            filename += ".pdb"
        path = config.paths.megascale_structures / filename
        reasons: list[str] = []
        if stratum == "unsupported":
            reasons.append("name_not_in_registered_natural_or_de_novo_scope")
        if counts[name] < config.panel.minimum_single_variants:
            reasons.append("too_few_usable_single_variants")
        if len(positions[name]) < config.panel.minimum_unique_positions:
            reasons.append("too_few_unique_mutated_positions")
        parsed_structure: dict[str, Any] | None
        if not path.is_file():
            reasons.append("structure_unavailable")
            parsed_structure = None
        else:
            try:
                parsed_structure = _read_chain(path, "A")
            except (ValueError, OSError) as error:
                reasons.append(f"structure_parse_failure:{type(error).__name__}")
                parsed_structure = None
            if parsed_structure is not None:
                if parsed_structure["sequence"] != sequence:
                    reasons.append("sequence_structure_mismatch")
                if (
                    config.panel.require_complete_backbone
                    and not parsed_structure["complete_backbone"]
                ):
                    reasons.append("incomplete_backbone")
        for reason in reasons:
            exclusions.append({"domain_id": domain_id, "reason": reason})
        structure_eligible = bool(
            parsed_structure is not None
            and parsed_structure["sequence"] == sequence
            and (
                not config.panel.require_complete_backbone or parsed_structure["complete_backbone"]
            )
        )
        rows.append(
            {
                "domain_id": domain_id,
                "wt_name": name,
                "stratum": stratum,
                "design_family": _design_family(name) if stratum == "de_novo" else "natural",
                "design_cluster": clusters.get(name, ""),
                "sequence": sequence,
                "length": len(sequence),
                "chain_id": str(parsed_structure["chain_id"]) if parsed_structure else "",
                "structure_path": str(path.resolve()),
                "candidate_variant_count": int(counts[name]),
                "candidate_position_count": int(len(positions[name])),
                "missing_single_effect_count": int(missing_effect[name]),
                "structure_eligible_as_decoy": structure_eligible,
                "metadata_eligible": not reasons,
            }
        )
    return pd.DataFrame(rows), exclusions


def _stratum(name: str) -> str:
    if NATURAL_NAME.fullmatch(name):
        return "natural"
    if DE_NOVO_NAME.match(name):
        return "de_novo"
    return "unsupported"


def _opened_sequence_registry(config: MechanismStudyConfig) -> pd.DataFrame:
    generalization = pd.read_parquet(config.paths.generalization_run / "dms" / "assays.parquet")
    counterfactuals = pd.read_parquet(config.paths.counterfactual_run / "panel" / "domains.parquet")
    observability = pd.read_parquet(
        config.paths.observability_replication_run / "registry" / "domains.parquet"
    )
    frames = [
        generalization[["assay_id", "sequence"]]
        .rename(columns={"assay_id": "target_id"})
        .assign(target_source="generalization_opened_52"),
        counterfactuals[["domain_id", "sequence"]]
        .rename(columns={"domain_id": "target_id"})
        .assign(target_source="counterfactuals_opened_50"),
        observability[["domain_id", "sequence"]]
        .rename(columns={"domain_id": "target_id"})
        .assign(target_source="observability_cath_training"),
    ]
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        ["target_id", "sequence", "target_source"], ignore_index=True
    )


def _run_homology(
    candidates: pd.DataFrame,
    targets: pd.DataFrame,
    output_path: Path,
    config: MechanismStudyConfig,
) -> pd.DataFrame:
    query_path = output_path.with_suffix(".queries.fasta")
    target_path = output_path.with_suffix(".targets.fasta")
    write_text(query_path, _fasta(candidates["domain_id"], candidates["sequence"]))
    encoded = [f"target_{index:05d}" for index in range(len(targets))]
    write_text(target_path, _fasta(pd.Series(encoded), targets["sequence"]))
    lookup = targets.assign(encoded_target=encoded)[
        ["encoded_target", "target_id", "target_source"]
    ]
    with tempfile.TemporaryDirectory(prefix="mechanisms-mmseqs-", dir=output_path.parent) as name:
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
            "query,target,fident,qcov,tcov",
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
                names=["domain_id", "encoded_target", "fident", "qcov", "tcov"],
            )
        else:
            hits = pd.DataFrame(columns=["domain_id", "encoded_target", "fident", "qcov", "tcov"])
    for column in ("fident", "qcov", "tcov"):
        hits[column] = pd.to_numeric(hits[column], errors="coerce")
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
        ]
    ]


def _select_panel(eligible: pd.DataFrame, config: MechanismStudyConfig) -> pd.DataFrame:
    natural = (
        eligible.loc[eligible["stratum"].eq("natural")]
        .sort_values(["design_cluster", "wt_name"], kind="stable")
        .drop_duplicates("design_cluster", keep="first")
        .head(config.panel.natural_domains)
    )
    if len(natural) != config.panel.natural_domains:
        raise ValueError(
            f"need {config.panel.natural_domains} natural domains, found {len(natural)}"
        )
    design_frames = []
    for family in config.panel.de_novo_families:
        frame = (
            eligible.loc[eligible["stratum"].eq("de_novo") & eligible["design_family"].eq(family)]
            .sort_values("wt_name", kind="stable")
            .head(config.panel.de_novo_domains_per_family)
        )
        if len(frame) != config.panel.de_novo_domains_per_family:
            raise ValueError(
                f"need {config.panel.de_novo_domains_per_family} domains for {family}, "
                f"found {len(frame)}"
            )
        design_frames.append(frame)
    selected = pd.concat([natural, *design_frames], ignore_index=True)
    return selected.sort_values(["stratum", "design_family", "wt_name"], ignore_index=True)


def _materialize_domains_and_residues(
    selected: pd.DataFrame, config: MechanismStudyConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    foundation = load_config(config.paths.foundation_config)
    domain_rows = []
    residue_frames = []
    for row in selected.itertuples(index=False):
        residues, summary = _preprocess_silently(
            str(row.domain_id),
            str(row.sequence),
            Path(row.structure_path),
            str(row.chain_id),
            foundation.registry,
        )
        if not residues["has_complete_backbone"].all():
            raise ValueError(f"selected domain lacks complete backbone: {row.domain_id}")
        residues["source"] = "Megascale_2023"
        residues["stratum"] = row.stratum
        residue_frames.append(residues)
        domain_rows.append(
            {
                "domain_id": row.domain_id,
                "wt_name": row.wt_name,
                "sequence": row.sequence,
                "length": int(row.length),
                "structure_path": row.structure_path,
                "chain_id": row.chain_id,
                "source": "Megascale_2023",
                "stratum": row.stratum,
                "design_family": row.design_family,
                "design_cluster": row.design_cluster,
                "platform": "cDNA_display_proteolysis",
                "structure_kind": "released_model_archive",
                "evaluation_role": "mechanisms_locked_dense_in_distribution_audit",
                "helix_fraction": float(summary["helix_fraction"]),
                "strand_fraction": float(summary["strand_fraction"]),
            }
        )
    domains = pd.DataFrame(domain_rows).sort_values("domain_id", ignore_index=True)
    residues = pd.concat(residue_frames, ignore_index=True).sort_values(
        ["domain_id", "position"], ignore_index=True
    )
    descriptors = _structure_descriptors(residues)
    return domains.merge(descriptors, on="domain_id", validate="one_to_one"), residues


def _materialize_variants(selected: pd.DataFrame, config: MechanismStudyConfig) -> pd.DataFrame:
    names = set(selected["wt_name"])
    sequences = selected.set_index("wt_name")["sequence"].to_dict()
    domains = selected.set_index("wt_name")["domain_id"].to_dict()
    strata = selected.set_index("wt_name")["stratum"].to_dict()
    frames = []
    columns = ["WT_name", "mut_type", "ddG_ML"]
    with (
        zipfile.ZipFile(config.paths.megascale_archive) as archive,
        archive.open(config.panel.megascale_member) as handle,
    ):
        for chunk in pd.read_csv(handle, usecols=columns, chunksize=200_000, low_memory=False):
            chunk = chunk.loc[chunk["WT_name"].astype(str).isin(names)].copy()
            if chunk.empty:
                continue
            parsed = chunk["mut_type"].astype(str).str.extract(SINGLE_MUTATION)
            effect = pd.to_numeric(chunk["ddG_ML"], errors="coerce")
            keep = parsed.notna().all(axis=1) & effect.notna()
            if not keep.any():
                continue
            parsed = parsed.loc[keep]
            wt_names = chunk.loc[keep, "WT_name"].astype(str)
            frame = pd.DataFrame(
                {
                    "domain_id": wt_names.map(domains).to_numpy(),
                    "position": parsed[1].astype(int).to_numpy() - 1,
                    "pdb_position": parsed[1].astype(int).to_numpy(),
                    "wild_type": parsed[0].to_numpy(),
                    "mutant": parsed[2].to_numpy(),
                    "effect": effect.loc[keep].astype(float).to_numpy(),
                    "source": "Megascale_2023",
                    "stratum": wt_names.map(strata).to_numpy(),
                    "wt_name": wt_names.to_numpy(),
                }
            )
            expected = np.asarray(
                [
                    sequences[name][position]
                    for name, position in zip(frame["wt_name"], frame["position"], strict=True)
                ]
            )
            frames.append(
                frame.loc[frame["wild_type"].to_numpy() == expected].drop(columns="wt_name")
            )
    result = pd.concat(frames, ignore_index=True)
    keys = [
        "domain_id",
        "position",
        "pdb_position",
        "wild_type",
        "mutant",
        "source",
        "stratum",
    ]
    if result.duplicated(["domain_id", "position", "wild_type", "mutant"]).any():
        result = result.groupby(keys, as_index=False, observed=True)["effect"].median()
    return result.sort_values(["domain_id", "position", "mutant"], ignore_index=True)


def _match_real_structure_decoys(
    selected: pd.DataFrame,
    archive_scan: pd.DataFrame,
    selected_domains: pd.DataFrame,
    config: MechanismStudyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    foundation = load_config(config.paths.foundation_config)
    lengths = set(selected["length"].astype(int))
    donor_candidates = archive_scan.loc[
        archive_scan["structure_eligible_as_decoy"]
        & archive_scan["length"].ge(min(lengths))
        & archive_scan["length"].le(
            max(lengths) + config.panel.matched_real_maximum_source_length_excess
        )
    ].copy()
    donor_rows = []
    donor_residue_lookup: dict[str, pd.DataFrame] = {}
    for row in donor_candidates.sort_values("domain_id").itertuples(index=False):
        try:
            residues, summary = _preprocess_silently(
                str(row.domain_id),
                str(row.sequence),
                Path(row.structure_path),
                str(row.chain_id),
                foundation.registry,
            )
        except (RuntimeError, ValueError, OSError):
            continue
        if not residues["has_complete_backbone"].all():
            continue
        donor_residue_lookup[str(row.domain_id)] = residues.sort_values("position")
        descriptor = _structure_descriptors(residues).iloc[0].to_dict()
        donor_rows.append(
            {
                "domain_id": row.domain_id,
                "wt_name": row.wt_name,
                "stratum": row.stratum,
                "design_family": row.design_family,
                "sequence": row.sequence,
                "length": int(row.length),
                "structure_path": row.structure_path,
                "chain_id": row.chain_id,
                "helix_fraction": float(summary["helix_fraction"]),
                "strand_fraction": float(summary["strand_fraction"]),
                **{key: value for key, value in descriptor.items() if key != "domain_id"},
            }
        )
    donor_pool = pd.DataFrame(donor_rows)
    if donor_pool.empty:
        raise ValueError("no exact-length real-structure donor candidates")

    matches = []
    descriptor_columns = [
        "helix_fraction",
        "strand_fraction",
        "mean_contact_order",
        "radius_of_gyration",
        "mean_contact_degree",
    ]
    scales = donor_pool[descriptor_columns].std(ddof=0).replace(0, 1.0)
    target_lookup = selected_domains.set_index("domain_id")
    for target in selected.sort_values("domain_id").itertuples(index=False):
        window_rows = []
        source_pool = donor_pool.loc[
            donor_pool["domain_id"].ne(target.domain_id)
            & donor_pool["length"].ge(int(target.length))
            & donor_pool["length"].le(
                int(target.length) + config.panel.matched_real_maximum_source_length_excess
            )
        ]
        for donor in source_pool.itertuples(index=False):
            residue_table = donor_residue_lookup.get(str(donor.domain_id))
            if residue_table is None:
                continue
            excess = int(donor.length) - int(target.length)
            for crop_start in range(excess + 1):
                window = residue_table.iloc[crop_start : crop_start + int(target.length)]
                descriptor = _window_descriptor(window)
                donor_sequence = str(donor.sequence)[crop_start : crop_start + int(target.length)]
                window_rows.append(
                    {
                        "domain_id": donor.domain_id,
                        "structure_path": donor.structure_path,
                        "chain_id": donor.chain_id,
                        "stratum": donor.stratum,
                        "design_family": donor.design_family,
                        "source_length": int(donor.length),
                        "crop_start": crop_start,
                        "sequence_identity_hamming": _hamming_identity(
                            str(target.sequence), donor_sequence
                        ),
                        **descriptor,
                    }
                )
        pool = pd.DataFrame(window_rows)
        if pool.empty:
            raise ValueError(f"no real-structure windows for {target.domain_id}")
        dissimilar = pool.loc[pool["sequence_identity_hamming"].lt(0.30)].copy()
        if len(dissimilar) >= config.panel.matched_real_decoys:
            pool = dissimilar
        values = target_lookup.loc[target.domain_id, descriptor_columns].astype(float)
        pool["descriptor_distance"] = np.sqrt(
            (((pool[descriptor_columns] - values) / scales) ** 2).sum(axis=1)
        )
        pool["same_stratum_penalty"] = (~pool["stratum"].eq(target.stratum)).astype(float) * 0.25
        pool["matching_objective"] = pool["descriptor_distance"] + pool["same_stratum_penalty"]
        chosen = (
            pool.sort_values(
                ["matching_objective", "sequence_identity_hamming", "domain_id"], kind="stable"
            )
            .drop_duplicates("domain_id", keep="first")
            .head(config.panel.matched_real_decoys)
        )
        if len(chosen) != config.panel.matched_real_decoys:
            raise ValueError(f"insufficient matched real decoys for {target.domain_id}")
        for ordinal, donor in enumerate(chosen.itertuples(index=False)):
            matches.append(
                {
                    "target_domain_id": target.domain_id,
                    "decoy_ordinal": ordinal,
                    "donor_domain_id": donor.domain_id,
                    "donor_structure_path": donor.structure_path,
                    "donor_chain_id": donor.chain_id,
                    "source_length": int(donor.source_length),
                    "crop_start": int(donor.crop_start),
                    "exact_length": int(target.length),
                    "target_stratum": target.stratum,
                    "donor_stratum": donor.stratum,
                    "donor_design_family": donor.design_family,
                    "sequence_identity_hamming": float(donor.sequence_identity_hamming),
                    "descriptor_distance": float(donor.descriptor_distance),
                    "matching_objective": float(donor.matching_objective),
                }
            )
    return donor_pool.sort_values("domain_id", ignore_index=True), pd.DataFrame(matches)


def _structure_descriptors(residues: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain_id, frame in residues.groupby("domain_id", sort=True, observed=True):
        frame = frame.sort_values("position")
        ca = frame[[f"ca_{axis}" for axis in "xyz"]].to_numpy(dtype=float)
        edges = contact_edges(ca, cutoff=8.0, minimum_sequence_separation=3)
        contact_order = (
            np.mean([abs(left - right) / len(frame) for left, right in edges]) if edges else 0.0
        )
        center = ca.mean(axis=0)
        rows.append(
            {
                "domain_id": domain_id,
                "mean_contact_order": float(contact_order),
                "radius_of_gyration": float(np.sqrt(np.mean(np.sum((ca - center) ** 2, axis=1)))),
                "mean_contact_degree": float(frame["contact_degree"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _window_descriptor(frame: pd.DataFrame) -> dict[str, float]:
    frame = frame.reset_index(drop=True)
    ca = frame[[f"ca_{axis}" for axis in "xyz"]].to_numpy(dtype=float)
    edges = contact_edges(ca, cutoff=8.0, minimum_sequence_separation=3)
    degree = np.zeros(len(frame), dtype=int)
    for left, right in edges:
        degree[left] += 1
        degree[right] += 1
    center = ca.mean(axis=0)
    return {
        "helix_fraction": float(frame["secondary_structure"].eq("helix").mean()),
        "strand_fraction": float(frame["secondary_structure"].eq("strand").mean()),
        "mean_contact_order": float(
            np.mean([abs(left - right) / len(frame) for left, right in edges]) if edges else 0.0
        ),
        "radius_of_gyration": float(np.sqrt(np.mean(np.sum((ca - center) ** 2, axis=1)))),
        "mean_contact_degree": float(degree.mean()),
    }


def _preprocess_silently(
    domain_id: str,
    sequence: str,
    structure_path: Path,
    chain_id: str,
    registry_config: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Suppress verbose DSSP remarks emitted by model-archive PDB comment records."""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Dropped unsupported records:.*")
        return preprocess_domain_structure(
            domain_id,
            sequence,
            structure_path,
            chain_id,
            registry_config,
        )


def _read_chain(path: Path, requested_chain: str) -> dict[str, Any]:
    model = next(PDBParser(QUIET=True).get_structure(path.stem, str(path)).get_models())
    chains = list(model.get_chains())
    chain = next((item for item in chains if item.id == requested_chain), None)
    if chain is None and len(chains) == 1:
        chain = chains[0]
    if chain is None:
        raise ValueError(f"chain {requested_chain!r} unavailable in {path}")
    sequence = []
    complete = True
    for residue in chain.get_residues():
        if residue.id[0].strip():
            continue
        letter = protein_letters_3to1.get(residue.resname.upper())
        if letter is None:
            continue
        sequence.append(letter)
        complete &= all(atom in residue for atom in BACKBONE_ATOMS)
    if not sequence or set(sequence) - set(AA_ALPHABET):
        raise ValueError(f"no canonical protein chain in {path}")
    return {
        "sequence": "".join(sequence),
        "chain_id": chain.id,
        "complete_backbone": bool(complete),
    }


def _hamming_identity(left: str, right: str) -> float:
    if len(left) != len(right):
        return 0.0
    return float(np.mean(np.fromiter((a == b for a, b in zip(left, right, strict=True)), bool)))


def _fasta(identifiers: pd.Series, sequences: pd.Series) -> str:
    return "".join(
        f">{identifier}\n{sequence}\n"
        for identifier, sequence in zip(identifiers, sequences, strict=True)
    )


def _data_audit_report(
    scan: pd.DataFrame,
    domains: pd.DataFrame,
    variants: pd.DataFrame,
    queries: pd.DataFrame,
) -> str:
    candidate_counts = (
        scan.groupby("stratum", observed=True)
        .agg(
            scanned=("domain_id", "size"),
            metadata_eligible=("metadata_eligible", "sum"),
            clusters=("design_cluster", "nunique"),
        )
        .reset_index()
    )
    selected_counts = (
        domains.groupby("stratum", observed=True)
        .agg(
            domains=("domain_id", "size"),
            variants=("variant_count", "sum"),
            query_positions=("mutated_position_count", "sum"),
            minimum_variants=("variant_count", "min"),
            median_variants=("variant_count", "median"),
            minimum_positions=("mutated_position_count", "min"),
            median_length=("length", "median"),
        )
        .reset_index()
    )
    missing = int(variants.isna().sum().sum())
    return f"""# mechanism study data audit

## Dataset overview

This report was generated before mechanism study model scoring. Selection used sequence, structure,
mutation availability, and missingness only; no stability-effect magnitude or aggregate was
used to choose domains. The selected panel contains {len(domains)} domains, {len(variants):,}
single-substitution rows, and {len(queries):,} unique queried positions.

### Candidate availability

{candidate_counts.to_markdown(index=False)}

### Locked panel coverage

{selected_counts.to_markdown(index=False)}

## Data quality

All selected sequences match their released structure chain and have complete N/CA/C/O
backbones. Variant wild-type letters match the locked sequence. The materialized variant
table has {missing:,} missing cells; effect rows with unavailable values were excluded before
counting eligibility. Duplicate measurements, when present, are collapsed by the median only
after deterministic domain selection.

The panel is dense enough for the frozen full-NDCG and NDCG@10% estimands. It is an
in-distribution audit drawn from a platform already used by the project and is therefore not
described as an independent external validation.
"""
