"""Outcome-blind preparation of the locked stability study evaluation panel."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from margin.config import load_config
from margin.provenance import (
    runtime_manifest,
    sha256_file,
    write_json,
    write_parquet,
    write_text,
)
from margin.studies.mechanisms.prepare import (
    _materialize_domains_and_residues,
    _materialize_variants,
    _preprocess_silently,
    _read_chain,
    _run_homology,
    _scan_archive,
    _structure_descriptors,
)
from margin.studies.stability.config import StabilityStudyConfig

SINGLE_MUTATION = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")
PRIMARY_POPULATION = "megascale_stability_dense"
EXTERNAL_POPULATION = "proteingym_esta_dense_external"


def prepare_stability_panel(config: StabilityStudyConfig) -> dict[str, Path]:
    """Select unopened identities and materialize labels without using their magnitudes."""

    panel = config.paths.run_dir / "panel"
    requests = config.paths.run_dir / "teacher_requests"
    panel.mkdir(parents=True, exist_ok=True)
    paths = {
        "candidate_scan": panel / "candidate_scan.parquet",
        "domains": panel / "domains.parquet",
        "variants": panel / "variants.parquet",
        "residues": panel / "residues.parquet",
        "queries": panel / "query_rows.parquet",
        "exclusions": panel / "exclusions.parquet",
        "prior_homology": panel / "prior_homology_hits.parquet",
        "within_pool_homology": panel / "within_pool_homology_hits.parquet",
        "cross_platform_homology": panel / "cross_platform_homology_hits.parquet",
        "manifest": panel / "manifest.json",
        "data_audit": config.paths.run_dir / "reports" / "data_audit.md",
        "requests": requests / "requests.parquet",
        "structures": requests / "structures.parquet",
        "requests_manifest": requests / "manifest.json",
    }
    if all(path.exists() for path in paths.values()):
        return paths
    _require_inputs(config)

    megascale_scan, exclusions = _scan_archive(config)
    external_scan = _external_candidate(config)
    scan = pd.concat(
        [
            megascale_scan.assign(candidate_source="Megascale_2023"),
            external_scan,
        ],
        ignore_index=True,
        sort=False,
    )
    candidates = scan.loc[scan["metadata_eligible"]].copy()
    opened = _opened_sequence_registry(config)
    prior_homology = _run_homology(candidates, opened, paths["prior_homology"], config)
    prior_near = _near_duplicate_ids(prior_homology, config)
    exact_opened = set(candidates["domain_id"]) & set(opened["target_id"])
    excluded = prior_near | exact_opened
    exclusions.extend(
        {"domain_id": domain_id, "reason": "opened_exact_or_near_duplicate_sequence"}
        for domain_id in sorted(excluded)
    )

    external_id = str(external_scan.iloc[0]["domain_id"])
    if external_id in excluded:
        raise ValueError("registered external assay is not sequence-independent of opened data")
    eligible_megascale = megascale_scan.loc[
        megascale_scan["metadata_eligible"] & ~megascale_scan["domain_id"].isin(excluded)
    ].copy()
    external_target = external_scan[["domain_id", "sequence"]].rename(
        columns={"domain_id": "target_id"}
    )
    external_target["target_source"] = "stability_external_esta"
    cross_platform = _run_homology(
        eligible_megascale,
        external_target,
        paths["cross_platform_homology"],
        config,
    )
    cross_near = _near_duplicate_ids(cross_platform, config)
    exclusions.extend(
        {"domain_id": domain_id, "reason": "external_validation_near_duplicate"}
        for domain_id in sorted(cross_near)
    )
    eligible_megascale = eligible_megascale.loc[
        ~eligible_megascale["domain_id"].isin(cross_near)
    ].copy()

    pool_targets = eligible_megascale[["domain_id", "sequence"]].rename(
        columns={"domain_id": "target_id"}
    )
    pool_targets["target_source"] = "stability_eligible_pool"
    within_pool = _run_homology(
        eligible_megascale,
        pool_targets,
        paths["within_pool_homology"],
        config,
    )
    selected = _select_megascale(eligible_megascale, within_pool, config)
    selected_ids = set(selected["domain_id"])
    exclusions.extend(
        {"domain_id": domain_id, "reason": "deterministic_stability_quota_or_deduplication"}
        for domain_id in sorted(set(eligible_megascale["domain_id"]) - selected_ids)
    )

    mega_domains, mega_residues = _materialize_domains_and_residues(selected, config)
    mega_domains["evaluation_role"] = "stability_locked_new_domain_primary"
    mega_domains["evaluation_population"] = PRIMARY_POPULATION
    mega_residues["evaluation_population"] = PRIMARY_POPULATION
    mega_variants = _materialize_variants(selected, config)
    mega_variants["evaluation_population"] = PRIMARY_POPULATION
    mega_variants["effect_name"] = "ddG_ML"

    external_domains, external_residues, external_variants = _materialize_external(
        external_scan.iloc[0], config
    )
    domains = pd.concat([mega_domains, external_domains], ignore_index=True, sort=False)
    residues = pd.concat([mega_residues, external_residues], ignore_index=True, sort=False)
    variants = pd.concat([mega_variants, external_variants], ignore_index=True, sort=False)
    observed = variants.groupby("domain_id", observed=True).agg(
        variant_count=("effect", "size"),
        mutated_position_count=("position", "nunique"),
    )
    domains = domains.drop(
        columns=["variant_count", "mutated_position_count"], errors="ignore"
    ).merge(observed, on="domain_id", validate="one_to_one")
    domains = domains.sort_values("domain_id", ignore_index=True)
    residues = residues.sort_values(["domain_id", "position"], ignore_index=True)
    variants = variants.sort_values(["domain_id", "position", "mutant"], ignore_index=True)
    queries = (
        variants[["domain_id", "position", "wild_type"]]
        .drop_duplicates(["domain_id", "position"])
        .merge(domains[["domain_id", "sequence"]], on="domain_id", validate="many_to_one")
        .assign(state_id=lambda frame: frame["domain_id"])[
            ["state_id", "domain_id", "position", "wild_type", "sequence"]
        ]
        .sort_values(["domain_id", "position"], ignore_index=True)
    )
    exclusion_table = pd.DataFrame(exclusions, columns=["domain_id", "reason"]).drop_duplicates(
        ignore_index=True
    )
    tables = {
        "candidate_scan": scan.drop(columns="pdb_number_to_position", errors="ignore"),
        "domains": domains,
        "variants": variants,
        "residues": residues,
        "queries": queries,
        "exclusions": exclusion_table,
        "prior_homology": prior_homology,
        "within_pool_homology": within_pool,
        "cross_platform_homology": cross_platform,
    }
    for name, table in tables.items():
        write_parquet(paths[name], table)
    request_tables = _teacher_requests(domains, residues, requests)
    write_parquet(paths["requests"], request_tables["requests"])
    write_parquet(paths["structures"], request_tables["structures"])
    write_json(
        paths["requests_manifest"],
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "role": "paired_target_structures_only",
            "counterfactuals": False,
            "requests": _table_description(paths["requests"], request_tables["requests"]),
            "structures": _table_description(paths["structures"], request_tables["structures"]),
        },
    )
    write_text(paths["data_audit"], _data_audit(domains, variants, queries))
    write_json(
        paths["manifest"],
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "status": "PREPARED_BEFORE_STABILITY_PANEL_MODEL_SCORING",
            "selection_uses_outcome_magnitudes": False,
            "selection_uses_outcome_availability": True,
            "primary_population": PRIMARY_POPULATION,
            "external_population": EXTERNAL_POPULATION,
            "effect_conventions": {
                PRIMARY_POPULATION: "effect=ddG_ML; positive means stabilizing",
                EXTERNAL_POPULATION: "effect=T50 degrees C; higher means more thermostable",
            },
            "tables": [_table_description(paths[name], table) for name, table in tables.items()],
        },
    )
    return paths


def _require_inputs(config: StabilityStudyConfig) -> None:
    required = [
        config.paths.megascale_archive,
        config.paths.megascale_structures,
        config.paths.protein_gym_metadata,
        config.paths.protein_gym_substitutions,
        config.paths.protein_gym_structures / config.panel.external_structure_file,
        config.paths.mmseqs_executable,
        config.paths.observability_replication_run / "registry" / "domains.parquet",
        config.paths.generalization_run / "dms" / "assays.parquet",
        config.paths.counterfactual_run / "panel" / "domains.parquet",
        config.paths.mechanism_run / "panel" / "domains.parquet",
        config.paths.action_validation_run / "panel" / "domains.parquet",
        config.paths.action_validation_run / "evaluation" / "project_decision.parquet",
        config.paths.run_dir / "calibration" / "selection.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"stability study inputs are missing: {missing}")
    decision = pd.read_parquet(
        config.paths.action_validation_run / "evaluation" / "project_decision.parquet"
    ).iloc[0]
    if str(decision["decision"]) != "MULTI_TEACHER_STRUCTURE_UNIQUE_ACTION_CONFIRMED":
        raise RuntimeError(
            "action-validation study did not authorize structure-conditioned stability study"
        )


def _external_candidate(config: StabilityStudyConfig) -> pd.DataFrame:
    metadata = pd.read_csv(config.paths.protein_gym_metadata)
    selected = metadata.loc[metadata["DMS_id"].eq(config.panel.external_assay_id)]
    if len(selected) != 1:
        raise ValueError("registered ProteinGym external assay must resolve exactly once")
    row = selected.iloc[0]
    sequence = str(row["target_seq"])
    structure_path = config.paths.protein_gym_structures / config.panel.external_structure_file
    parsed = _read_chain(structure_path, "A")
    reasons = []
    if parsed["sequence"] != sequence:
        reasons.append("sequence_structure_mismatch")
    if config.panel.require_complete_backbone and not parsed["complete_backbone"]:
        reasons.append("incomplete_backbone")
    if int(row["DMS_number_single_mutants"]) < config.panel.external_minimum_variants:
        reasons.append("too_few_external_variants")
    region = [int(value) for value in str(row["region_mutated"]).split("-")]
    position_count = region[1] - region[0] + 1
    if position_count < config.panel.external_minimum_positions:
        reasons.append("too_few_external_positions")
    return pd.DataFrame(
        [
            {
                "domain_id": f"proteingym:{config.panel.external_assay_id}",
                "wt_name": config.panel.external_assay_id,
                "stratum": "external_single_protein",
                "design_family": "not_applicable",
                "design_cluster": "ESTA_BACSU",
                "sequence": sequence,
                "length": len(sequence),
                "chain_id": str(parsed["chain_id"]),
                "structure_path": str(structure_path.resolve()),
                "candidate_variant_count": int(row["DMS_number_single_mutants"]),
                "candidate_position_count": position_count,
                "missing_single_effect_count": 0,
                "structure_eligible_as_decoy": False,
                "metadata_eligible": not reasons,
                "candidate_source": "ProteinGym_v1.3_Nutschel_2020",
            }
        ]
    )


def _opened_sequence_registry(config: StabilityStudyConfig) -> pd.DataFrame:
    cath = pd.read_parquet(
        config.paths.observability_replication_run / "registry" / "domains.parquet"
    )
    generalization = pd.read_parquet(config.paths.generalization_run / "dms" / "assays.parquet")
    frames = [
        cath[["domain_id", "sequence"]]
        .rename(columns={"domain_id": "target_id"})
        .assign(target_source="observability_cath_training"),
        generalization[["assay_id", "sequence"]]
        .rename(columns={"assay_id": "target_id"})
        .assign(target_source="generalization_opened"),
    ]
    for workflow_name, path in (
        ("counterfactuals", config.paths.counterfactual_run),
        ("mechanisms", config.paths.mechanism_run),
        ("action_validation", config.paths.action_validation_run),
    ):
        table = pd.read_parquet(path / "panel" / "domains.parquet")
        frames.append(
            table[["domain_id", "sequence"]]
            .rename(columns={"domain_id": "target_id"})
            .assign(target_source=f"{workflow_name}_opened")
        )
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        ["target_id", "sequence", "target_source"], ignore_index=True
    )


def _near_duplicate_ids(hits: pd.DataFrame, config: StabilityStudyConfig) -> set[str]:
    return set(
        hits.loc[
            hits["sequence_identity"].ge(config.panel.near_duplicate_identity)
            & hits[["query_coverage", "target_coverage"]]
            .min(axis=1)
            .ge(config.panel.near_duplicate_minimum_coverage),
            "domain_id",
        ].astype(str)
    )


def _select_megascale(
    eligible: pd.DataFrame,
    homology: pd.DataFrame,
    config: StabilityStudyConfig,
) -> pd.DataFrame:
    strict = homology.loc[
        homology["sequence_identity"].ge(config.panel.near_duplicate_identity)
        & homology[["query_coverage", "target_coverage"]]
        .min(axis=1)
        .ge(config.panel.near_duplicate_minimum_coverage)
        & homology["domain_id"].ne(homology["target_id"])
    ]
    neighbors: dict[str, set[str]] = {}
    for row in strict.itertuples(index=False):
        neighbors.setdefault(str(row.domain_id), set()).add(str(row.target_id))
        neighbors.setdefault(str(row.target_id), set()).add(str(row.domain_id))
    selected: list[pd.Series] = []
    selected_ids: set[str] = set()

    def take(frame: pd.DataFrame, count: int, *, distinct_cluster: bool) -> None:
        used_clusters = {str(row["design_cluster"]) for row in selected}
        for _, row in frame.iterrows():
            domain_id = str(row["domain_id"])
            cluster = str(row["design_cluster"])
            if distinct_cluster and cluster in used_clusters:
                continue
            if neighbors.get(domain_id, set()) & selected_ids:
                continue
            selected.append(row)
            selected_ids.add(domain_id)
            used_clusters.add(cluster)
            if sum(item["stratum"] == row["stratum"] for item in selected[-count:]) >= count:
                return

    natural = eligible.loc[eligible["stratum"].eq("natural")].sort_values(
        ["design_cluster", "wt_name"], kind="stable"
    )
    take(natural, config.panel.natural_domains, distinct_cluster=True)
    natural_selected = sum(str(row["stratum"]) == "natural" for row in selected)
    if natural_selected != config.panel.natural_domains:
        raise ValueError(
            f"need {config.panel.natural_domains} deduplicated natural domains, "
            f"found {natural_selected}"
        )
    for family in config.panel.de_novo_families:
        before = len(selected)
        frame = eligible.loc[
            eligible["stratum"].eq("de_novo") & eligible["design_family"].eq(family)
        ].sort_values(["design_cluster", "wt_name"], kind="stable")
        for _, row in frame.iterrows():
            domain_id = str(row["domain_id"])
            if neighbors.get(domain_id, set()) & selected_ids:
                continue
            selected.append(row)
            selected_ids.add(domain_id)
            if len(selected) - before == config.panel.de_novo_domains_per_family:
                break
        if len(selected) - before != config.panel.de_novo_domains_per_family:
            raise ValueError(f"insufficient deduplicated de novo domains for {family}")
    result = pd.DataFrame(selected)
    return result.sort_values(["stratum", "design_family", "wt_name"], ignore_index=True)


def _materialize_external(
    candidate: pd.Series,
    config: StabilityStudyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    foundation = load_config(config.paths.foundation_config)
    domain_id = str(candidate["domain_id"])
    residues, summary = _preprocess_silently(
        domain_id,
        str(candidate["sequence"]),
        Path(candidate["structure_path"]),
        str(candidate["chain_id"]),
        foundation.registry,
    )
    if not residues["has_complete_backbone"].all():
        raise ValueError("registered external structure lacks a complete backbone")
    residues["source"] = "ProteinGym_v1.3_Nutschel_2020"
    residues["stratum"] = "external_single_protein"
    residues["evaluation_population"] = EXTERNAL_POPULATION
    descriptor = _structure_descriptors(residues).iloc[0]
    domains = pd.DataFrame(
        [
            {
                "domain_id": domain_id,
                "wt_name": config.panel.external_assay_id,
                "sequence": candidate["sequence"],
                "length": int(candidate["length"]),
                "structure_path": candidate["structure_path"],
                "chain_id": candidate["chain_id"],
                "source": "ProteinGym_v1.3_Nutschel_2020",
                "stratum": "external_single_protein",
                "design_family": "not_applicable",
                "design_cluster": "ESTA_BACSU",
                "platform": "experimental_T50_thermostability",
                "structure_kind": "ProteinGym_AF2_structure",
                "evaluation_role": "stability_locked_dense_external_replication",
                "evaluation_population": EXTERNAL_POPULATION,
                "helix_fraction": float(summary["helix_fraction"]),
                "strand_fraction": float(summary["strand_fraction"]),
                "mean_contact_order": float(descriptor["mean_contact_order"]),
                "radius_of_gyration": float(descriptor["radius_of_gyration"]),
                "mean_contact_degree": float(descriptor["mean_contact_degree"]),
            }
        ]
    )
    member = f"DMS_ProteinGym_substitutions/{config.panel.external_assay_id}.csv"
    with (
        zipfile.ZipFile(config.paths.protein_gym_substitutions) as archive,
        archive.open(member) as handle,
    ):
        raw = pd.read_csv(handle, usecols=["mutant", "DMS_score"])
    parsed = raw["mutant"].astype(str).str.extract(SINGLE_MUTATION)
    keep = parsed.notna().all(axis=1) & pd.to_numeric(raw["DMS_score"], errors="coerce").notna()
    parsed = parsed.loc[keep]
    variants = pd.DataFrame(
        {
            "domain_id": domain_id,
            "position": parsed[1].astype(int).to_numpy() - 1,
            "pdb_position": parsed[1].astype(int).to_numpy(),
            "wild_type": parsed[0].to_numpy(),
            "mutant": parsed[2].to_numpy(),
            "effect": pd.to_numeric(raw.loc[keep, "DMS_score"]).to_numpy(dtype=float),
            "source": "ProteinGym_v1.3_Nutschel_2020",
            "stratum": "external_single_protein",
            "evaluation_population": EXTERNAL_POPULATION,
            "effect_name": "T50_Celsius",
        }
    )
    sequence = str(candidate["sequence"])
    expected = np.asarray([sequence[position] for position in variants["position"]])
    variants = variants.loc[variants["wild_type"].to_numpy() == expected].copy()
    if len(variants) < config.panel.external_minimum_variants:
        raise ValueError("external assay lost the registered minimum variant count")
    if variants["position"].nunique() < config.panel.external_minimum_positions:
        raise ValueError("external assay lost the registered minimum position count")
    if variants.duplicated(["domain_id", "position", "mutant"]).any():
        raise ValueError("external assay contains duplicate single substitutions")
    return domains, residues, variants


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


def _table_description(path: Path, table: pd.DataFrame) -> dict[str, Any]:
    return {"path": str(path), "rows": int(len(table)), "columns": list(table.columns)}


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
        "# stability study locked-panel data audit",
        "",
        "_Generated without summarizing outcome magnitudes._",
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
            "Selection used assay identity, mutation availability, sequence identity, "
            "structure completeness, and deterministic quotas only. Megascale and ESTA "
            "retain separate evidence roles and are never pooled into one endpoint.",
            "",
            f"Materialized totals: {len(domains)} proteins/domains, {len(variants):,} "
            f"variants, and {len(queries):,} queried positions.",
        ]
    )
    return "\n".join(rows) + "\n"
