"""Outcome-blind preparation of the locked action-validation study two-platform panel."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from margin.config import load_config
from margin.provenance import (
    runtime_manifest,
    sha256_file,
    table_manifest,
    write_json,
    write_parquet,
    write_text,
)
from margin.studies.action_validation.config import ActionValidationStudyConfig
from margin.studies.counterfactuals.prepare import _materialize_s669_variants
from margin.studies.counterfactuals.prepare import _read_chain as _read_s669_chain
from margin.studies.mechanisms.prepare import (
    _materialize_domains_and_residues,
    _materialize_variants,
    _preprocess_silently,
    _run_homology,
    _scan_archive,
    _select_panel,
    _structure_descriptors,
)

SINGLE_MUTATION = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")


def prepare_action_validation_panel(config: ActionValidationStudyConfig) -> dict[str, Path]:
    """Select new identities and materialize outcomes without summarizing their values."""

    output = config.paths.run_dir / "panel"
    output.mkdir(parents=True, exist_ok=True)
    requests = config.paths.run_dir / "teacher_requests"
    paths = {
        "candidate_scan": output / "candidate_scan.parquet",
        "domains": output / "domains.parquet",
        "variants": output / "variants.parquet",
        "residues": output / "residues.parquet",
        "queries": output / "query_rows.parquet",
        "exclusions": output / "exclusions.parquet",
        "homology_hits": output / "prior_homology_hits.parquet",
        "cross_panel_homology": output / "cross_panel_homology_hits.parquet",
        "manifest": output / "manifest.json",
        "requests": requests / "requests.parquet",
        "structures": requests / "structures.parquet",
        "requests_manifest": requests / "manifest.json",
        "data_audit": config.paths.run_dir / "reports" / "data_audit.md",
    }
    if all(path.exists() for path in paths.values()):
        return paths
    _require_inputs(config)

    megascale_scan, megascale_exclusions = _scan_archive(config)
    s669_rows, s669_scan, s669_private, s669_exclusions = _scan_s669(config)
    scan = pd.concat(
        [megascale_scan.assign(candidate_source="Megascale_2023"), s669_scan],
        ignore_index=True,
        sort=False,
    )
    candidates = scan.loc[scan["metadata_eligible"]].copy()
    opened = _opened_sequence_registry(config)
    prior_homology = _run_homology(candidates, opened, paths["homology_hits"], config)
    prior_near = _near_duplicate_ids(prior_homology, config)

    opened_ids = set(opened["target_id"].astype(str))
    exact_opened = set(candidates.loc[candidates["domain_id"].isin(opened_ids), "domain_id"])
    all_exclusions = [*megascale_exclusions, *s669_exclusions]
    all_exclusions.extend(
        {"domain_id": domain_id, "reason": "previously_opened_domain_identity"}
        for domain_id in sorted(exact_opened)
    )
    all_exclusions.extend(
        {"domain_id": domain_id, "reason": "opened_exact_or_near_duplicate_sequence"}
        for domain_id in sorted(prior_near)
    )
    excluded = prior_near | exact_opened

    eligible_s669 = s669_private.loc[
        s669_private["metadata_eligible"] & ~s669_private["domain_id"].isin(excluded)
    ].copy()
    selected_s669 = (
        eligible_s669.sort_values(
            ["candidate_variant_count", "domain_id"],
            ascending=[False, True],
            kind="stable",
        )
        .head(config.panel.s669_maximum_domains)
        .reset_index(drop=True)
    )
    if len(selected_s669) < config.panel.s669_minimum_selected_domains:
        raise ValueError(
            "insufficient unopened sequence-independent S669 domains: "
            f"{len(selected_s669)}/{config.panel.s669_minimum_selected_domains}"
        )
    selected_s669_ids = set(selected_s669["domain_id"])
    all_exclusions.extend(
        {"domain_id": row.domain_id, "reason": "deterministic_s669_information_quota"}
        for row in eligible_s669.loc[
            ~eligible_s669["domain_id"].isin(selected_s669_ids)
        ].itertuples(index=False)
    )

    eligible_megascale = megascale_scan.loc[
        megascale_scan["metadata_eligible"] & ~megascale_scan["domain_id"].isin(excluded)
    ].copy()
    cross_targets = selected_s669[["domain_id", "sequence"]].rename(
        columns={"domain_id": "target_id"}
    )
    cross_targets["target_source"] = "action_validation_s669_replication"
    cross_homology = _run_homology(
        eligible_megascale,
        cross_targets,
        paths["cross_panel_homology"],
        config,
    )
    cross_near = _near_duplicate_ids(cross_homology, config)
    all_exclusions.extend(
        {"domain_id": domain_id, "reason": "within_action_validation_cross_platform_near_duplicate"}
        for domain_id in sorted(cross_near)
    )
    eligible_megascale = eligible_megascale.loc[
        ~eligible_megascale["domain_id"].isin(cross_near)
    ].copy()
    selected_megascale = _select_panel(eligible_megascale, config)
    selected_megascale_ids = set(selected_megascale["domain_id"])
    all_exclusions.extend(
        {
            "domain_id": row.domain_id,
            "reason": "deterministic_megascale_stratum_or_family_quota",
        }
        for row in eligible_megascale.loc[
            ~eligible_megascale["domain_id"].isin(selected_megascale_ids)
        ].itertuples(index=False)
    )

    mega_domains, mega_residues = _materialize_domains_and_residues(selected_megascale, config)
    mega_domains["evaluation_role"] = "action_validation_locked_dense_confirmation"
    mega_domains["evaluation_population"] = "megascale_dense"
    mega_residues["evaluation_population"] = "megascale_dense"
    s669_domains, s669_residues = _materialize_s669_domains(selected_s669, config)
    domains = pd.concat([mega_domains, s669_domains], ignore_index=True, sort=False).sort_values(
        "domain_id", ignore_index=True
    )
    residues = pd.concat([mega_residues, s669_residues], ignore_index=True, sort=False).sort_values(
        ["domain_id", "position"], ignore_index=True
    )

    mega_variants = _materialize_variants(selected_megascale, config)
    mega_variants["evaluation_population"] = "megascale_dense"
    s669_variants = _materialize_s669_variants(s669_rows, selected_s669)
    s669_variants["stratum"] = "s669_natural"
    s669_variants["evaluation_population"] = "s669_sparse_cross_platform"
    variants = pd.concat([mega_variants, s669_variants], ignore_index=True, sort=False).sort_values(
        ["domain_id", "position", "mutant"], ignore_index=True
    )
    observed = variants.groupby("domain_id", observed=True).agg(
        variant_count=("effect", "size"),
        mutated_position_count=("position", "nunique"),
    )
    domains = domains.drop(columns=["variant_count", "mutated_position_count"], errors="ignore")
    domains = domains.merge(observed, on="domain_id", validate="one_to_one")
    queries = (
        variants[["domain_id", "position", "wild_type"]]
        .drop_duplicates(["domain_id", "position"])
        .merge(domains[["domain_id", "sequence"]], on="domain_id", validate="many_to_one")
        .assign(state_id=lambda frame: frame["domain_id"])[
            ["state_id", "domain_id", "position", "wild_type", "sequence"]
        ]
        .sort_values(["domain_id", "position"], ignore_index=True)
    )
    exclusion_table = pd.DataFrame(all_exclusions, columns=["domain_id", "reason"]).drop_duplicates(
        ignore_index=True
    )
    tables = {
        "candidate_scan": scan.drop(columns="pdb_number_to_position", errors="ignore"),
        "domains": domains,
        "variants": variants,
        "residues": residues,
        "queries": queries,
        "exclusions": exclusion_table,
        "homology_hits": prior_homology,
        "cross_panel_homology": cross_homology,
    }
    for name, table in tables.items():
        write_parquet(paths[name], table)
    request_tables = _teacher_requests(domains, residues, paths["requests"].parent)
    write_parquet(paths["requests"], request_tables["requests"])
    write_parquet(paths["structures"], request_tables["structures"])
    write_json(
        paths["requests_manifest"],
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "role": "paired_structures_only_no_counterfactuals",
            "requests": table_manifest(paths["requests"], request_tables["requests"]),
            "structures": table_manifest(paths["structures"], request_tables["structures"]),
        },
    )
    write_text(paths["data_audit"], _data_audit(domains, variants, queries))
    write_json(
        paths["manifest"],
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "status": "PREPARED_BEFORE_ACTION_VALIDATION_PANEL_MODEL_SCORING",
            "selection_uses_outcome_magnitudes": False,
            "selection_uses_outcome_availability": True,
            "primary_panel_is_cross_platform": False,
            "s669_replication_is_dense": False,
            "s669_replication_is_topology_independent": False,
            "effect_conventions": {
                "megascale_dense": "effect=ddG_ML; positive means stabilizing",
                "s669_sparse_cross_platform": (
                    "effect=-ddG_experimental; positive means stabilizing"
                ),
            },
            "selected_domains": int(len(domains)),
            "selected_domains_by_population": {
                str(key): int(value)
                for key, value in domains["evaluation_population"].value_counts().items()
            },
            "selected_variants": int(len(variants)),
            "selected_query_positions": int(len(queries)),
            "tables": [table_manifest(paths[name], table) for name, table in tables.items()],
        },
    )
    return paths


def _require_inputs(config: ActionValidationStudyConfig) -> None:
    required = [
        config.paths.megascale_archive,
        config.paths.megascale_structures,
        config.paths.s669_root / "s669_ddg_experimental.csv",
        config.paths.s669_root / "pdb",
        config.paths.mmseqs_executable,
        config.paths.observability_replication_run / "registry" / "domains.parquet",
        config.paths.generalization_run / "dms" / "assays.parquet",
        config.paths.counterfactual_run / "panel" / "domains.parquet",
        config.paths.mechanism_run / "panel" / "domains.parquet",
        config.paths.mechanism_run / "evaluation" / "audit_decisions.parquet",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"action-validation study inputs are missing: {missing}")
    decision = pd.read_parquet(
        config.paths.mechanism_run / "evaluation" / "audit_decisions.parquet"
    )
    if bool(decision["selective_routing_authorized"].iloc[0]):
        raise RuntimeError("mechanism study unexpectedly authorized sequence-control branch")


def _scan_s669(
    config: ActionValidationStudyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, str]]]:
    columns = ["pdb_id", "chain", "pre", "pos", "post", "ddG_experimental"]
    rows = pd.read_csv(config.paths.s669_root / "s669_ddg_experimental.csv", usecols=columns)
    rows["chain"] = rows["chain"].astype(str)
    opened_assays = pd.read_parquet(config.paths.generalization_run / "dms" / "assays.parquet")
    opened_pdb = set(opened_assays["assay_id"].astype(str).str.rsplit("_", n=1).str[-1].str.upper())
    records: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    for (pdb_id, requested_chain), frame in rows.groupby(["pdb_id", "chain"], sort=True):
        domain_id = f"s669:{pdb_id}:{requested_chain}"
        structure_path = config.paths.s669_root / "pdb" / f"{pdb_id}.pdb"
        reasons: list[str] = []
        try:
            parsed = _read_s669_chain(structure_path, requested_chain)
        except (ValueError, OSError) as error:
            parsed = None
            reasons.append(f"structure_parse_failure:{type(error).__name__}")
        if len(frame) < config.panel.s669_minimum_variants:
            reasons.append("too_few_domain_variants")
        if str(pdb_id).upper() in opened_pdb:
            reasons.append("generalization_opened_pdb_identity")
        if parsed is not None:
            if config.panel.require_complete_backbone and not parsed["complete_backbone"]:
                reasons.append("incomplete_backbone")
            number_to_position = parsed["pdb_number_to_position"]
            mismatch = any(
                int(row.pos) not in number_to_position
                or parsed["sequence"][number_to_position[int(row.pos)]] != str(row.pre)
                for row in frame.itertuples(index=False)
            )
            if mismatch:
                reasons.append("mutation_structure_mismatch")
        else:
            number_to_position = {}
        for reason in reasons:
            exclusions.append({"domain_id": domain_id, "reason": reason})
        records.append(
            {
                "domain_id": domain_id,
                "pdb_id": str(pdb_id),
                "requested_chain": str(requested_chain),
                "chain_id": str(parsed["chain_id"]) if parsed else "",
                "sequence": str(parsed["sequence"]) if parsed else "",
                "length": len(str(parsed["sequence"])) if parsed else 0,
                "structure_path": str(structure_path.resolve()),
                "candidate_variant_count": int(len(frame)),
                "candidate_position_count": int(frame["pos"].nunique()),
                "stratum": "s669_natural",
                "design_family": "not_applicable",
                "design_cluster": f"s669:{pdb_id}",
                "wt_name": f"{pdb_id}.pdb",
                "structure_eligible_as_decoy": False,
                "missing_single_effect_count": int(frame["ddG_experimental"].isna().sum()),
                "metadata_eligible": not reasons,
                "pdb_number_to_position": number_to_position,
                "candidate_source": "S669",
            }
        )
    private = pd.DataFrame(records)
    public = private.drop(columns="pdb_number_to_position")
    return rows, public, private, exclusions


def _opened_sequence_registry(config: ActionValidationStudyConfig) -> pd.DataFrame:
    cath = pd.read_parquet(
        config.paths.observability_replication_run / "registry" / "domains.parquet"
    )
    generalization = pd.read_parquet(config.paths.generalization_run / "dms" / "assays.parquet")
    counterfactuals = pd.read_parquet(config.paths.counterfactual_run / "panel" / "domains.parquet")
    mechanisms = pd.read_parquet(config.paths.mechanism_run / "panel" / "domains.parquet")
    frames = [
        cath[["domain_id", "sequence"]]
        .rename(columns={"domain_id": "target_id"})
        .assign(target_source="observability_cath_training"),
        generalization[["assay_id", "sequence"]]
        .rename(columns={"assay_id": "target_id"})
        .assign(target_source="generalization_opened_52"),
        counterfactuals[["domain_id", "sequence"]]
        .rename(columns={"domain_id": "target_id"})
        .assign(target_source="counterfactuals_opened_50"),
        mechanisms[["domain_id", "sequence"]]
        .rename(columns={"domain_id": "target_id"})
        .assign(target_source="mechanisms_opened_32"),
    ]
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        ["target_id", "sequence", "target_source"], ignore_index=True
    )


def _near_duplicate_ids(hits: pd.DataFrame, config: ActionValidationStudyConfig) -> set[str]:
    return set(
        hits.loc[
            hits["sequence_identity"].ge(config.panel.near_duplicate_identity)
            & hits[["query_coverage", "target_coverage"]]
            .min(axis=1)
            .ge(config.panel.near_duplicate_minimum_coverage),
            "domain_id",
        ].astype(str)
    )


def _materialize_s669_domains(
    selected: pd.DataFrame, config: ActionValidationStudyConfig
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
            raise ValueError(f"selected S669 domain lacks complete backbone: {row.domain_id}")
        residues["source"] = "S669"
        residues["stratum"] = "s669_natural"
        residues["evaluation_population"] = "s669_sparse_cross_platform"
        residue_frames.append(residues)
        domain_rows.append(
            {
                "domain_id": row.domain_id,
                "wt_name": row.wt_name,
                "sequence": row.sequence,
                "length": int(row.length),
                "structure_path": row.structure_path,
                "chain_id": row.chain_id,
                "pdb_id": row.pdb_id,
                "source": "S669",
                "stratum": "s669_natural",
                "design_family": "not_applicable",
                "design_cluster": row.design_cluster,
                "platform": "direct_experimental_ddG",
                "structure_kind": "experimental_PDB",
                "evaluation_role": "action_validation_locked_sparse_cross_platform_replication",
                "evaluation_population": "s669_sparse_cross_platform",
                "helix_fraction": float(summary["helix_fraction"]),
                "strand_fraction": float(summary["strand_fraction"]),
            }
        )
    domains = pd.DataFrame(domain_rows)
    residues = pd.concat(residue_frames, ignore_index=True)
    descriptors = _structure_descriptors(residues)
    return domains.merge(descriptors, on="domain_id", validate="one_to_one"), residues


def _teacher_requests(
    domains: pd.DataFrame, residues: pd.DataFrame, directory: Path
) -> dict[str, pd.DataFrame]:
    inputs = directory / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    request_rows = []
    structure_rows = []
    for domain in domains.sort_values("domain_id").itertuples(index=False):
        frame = residues.loc[residues["domain_id"].eq(domain.domain_id)].sort_values("position")
        coordinates = np.stack(
            [
                frame[[f"{atom}_{axis}" for axis in "xyz"]].to_numpy(dtype=np.float32)
                for atom in ("n", "ca", "c", "o")
            ],
            axis=1,
        )
        if len(coordinates) != int(domain.length):
            raise ValueError(f"coordinate length mismatch for {domain.domain_id}")
        path = inputs / f"{_safe_name(str(domain.domain_id))}.npz"
        np.savez_compressed(path, coordinates=coordinates)
        request_rows.append(
            {
                "request_id": f"{domain.domain_id}|paired|{domain.domain_id}",
                "state_id": domain.domain_id,
                "domain_id": domain.domain_id,
                "state_sequence": domain.sequence,
                "structure_role": "paired",
                "structure_id": domain.domain_id,
                "input_kind": "coordinates",
                "input_path": str(path.resolve()),
                "length": int(domain.length),
            }
        )
        structure_rows.append(
            {
                "structure_role": "paired",
                "structure_id": domain.domain_id,
                "target_domain_id": domain.domain_id,
                "input_kind": "coordinates",
                "input_path": str(path.resolve()),
                "sha256": sha256_file(path),
                "analysis_population": domain.evaluation_population,
            }
        )
    return {
        "requests": pd.DataFrame(request_rows),
        "structures": pd.DataFrame(structure_rows),
    }


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value)


def _data_audit(domains: pd.DataFrame, variants: pd.DataFrame, queries: pd.DataFrame) -> str:
    counts = (
        domains.groupby("evaluation_population", observed=True)
        .agg(
            domains=("domain_id", "size"),
            variants=("variant_count", "sum"),
            query_positions=("mutated_position_count", "sum"),
            minimum_variants=("variant_count", "min"),
            median_variants=("variant_count", "median"),
        )
        .reset_index()
    )
    rows = [
        "# action-validation study locked-panel data audit",
        "",
        "_Generated without summarizing stability-effect magnitudes._",
        "",
        "##  Coverage",
        "",
        "| Population | Domains | Variants | Query positions | Minimum/domain | Median/domain |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in counts.itertuples(index=False):
        rows.append(
            f"| {row.evaluation_population} | {row.domains} | {row.variants} | "
            f"{row.query_positions} | {row.minimum_variants} | {row.median_variants:.1f} |"
        )
    rows.extend(
        [
            "",
            "Selection used mutation availability, sequence identity, structure completeness, "
            "and deterministic quotas only. The dense Megascale panel and sparse S669 "
            "replication retain distinct evidence roles.",
            "",
            f"Materialized totals: {len(domains)} domains, {len(variants):,} variants, "
            f"and {len(queries):,} unique query positions.",
        ]
    )
    return "\n".join(rows) + "\n"
