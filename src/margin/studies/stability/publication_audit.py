"""Submission-facing post-lock audit for the completed stability study evidence chain."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from margin.provenance import runtime_manifest, write_json, write_parquet, write_text
from margin.studies.action_validation.evaluation import _ndcg, _spearman
from margin.studies.counterfactuals.evaluation import stratified_domain_bootstrap
from margin.studies.external_validation.panel import load_external_validation_config
from margin.studies.stability.config import load_stability_config
from margin.studies.stability.publication_figure import build_publication_audit_figure
from margin.studies.structure_sensitivity.audit import build_structure_audit_tables

PRIMARY_FIREPROT_METHOD = "temperature_consensus_action"
PRIMARY_FIREPROT_CONTROL = "temperature_consensus_g_plus_c_plus"
FAST_METHOD = "esm_if1_action_only"
ROBUST_METHOD = "unscaled_equal_action_only"


def load_publication_audit_specification(path: Path) -> dict[str, Any]:
    """Load the recorded descriptive specification and resolve its project root."""

    path = path.resolve()
    with path.open(encoding="utf-8") as handle:
        specification = yaml.safe_load(handle)
    expected = "RECORDED_AFTER_OUTCOME_OPENING_BEFORE_EXTENDED_AUDIT_EXECUTION"
    if specification.get("status") != expected:
        raise ValueError("post-lock audit specification has an unexpected status")
    root = Path(specification["paths"].get("project_root", "../.."))
    specification["project_root"] = (path.parent / root).resolve()
    specification["specification_path"] = path
    return specification


def run_publication_audit(specification: Mapping[str, Any]) -> dict[str, Path]:
    """Run the three finite submission audits without changing registered decisions."""

    root = Path(specification["project_root"])
    output = root / specification["paths"]["output_dir"]
    source_data = output / "source_data"
    figures = output / "figures"
    reports = output / "reports"
    source_data.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    stability = load_stability_config(root / specification["paths"]["stability_config"])
    cross = load_external_validation_config(
        root / specification["paths"]["external_validation_protocol"]
    )
    fireprot = _fireprot_tables(root, cross, specification)
    fast_robust = _fast_robust_tables(stability, cross, specification)
    structure_sensitivity = build_structure_audit_tables(root, specification)
    nomenclature = _score_nomenclature(stability, cross)
    tables = {
        "score_nomenclature": nomenclature,
        **fireprot,
        **fast_robust,
        **structure_sensitivity,
    }
    paths: dict[str, Path] = {}
    for name, table in tables.items():
        parquet = source_data / f"{name}.parquet"
        csv = source_data / f"{name}.csv"
        write_parquet(parquet, table)
        table.to_csv(csv, index=False)
        paths[name] = parquet
        paths[f"{name}_csv"] = csv
    figure_paths = build_publication_audit_figure(tables, figures)
    for extension, path in figure_paths.items():
        paths[f"figure_{extension}"] = path
    report = reports / "stability_postlock_submission_audit.md"
    write_text(report, _render_report(tables, specification))
    paths["report"] = report
    manifest = output / "manifest.json"
    write_json(
        manifest,
        {
            **runtime_manifest(root),
            "schema_version": specification["schema_version"],
            "status": "POSTLOCK_SUBMISSION_AUDIT_COMPLETE_NO_GATE_CHANGE",
            "analysis_role": specification["analysis_role"],
            "outcomes_were_open_before_specification": True,
            "changes_registered_gates": False,
            "authorizes_routing": False,
            "specification": str(specification["specification_path"]),
            "report": str(report),
            "figures": {extension: str(path) for extension, path in figure_paths.items()},
            "tables": {
                name: {
                    "path": str(paths[name]),
                    "csv": str(paths[f"{name}_csv"]),
                    "rows": len(table),
                    "columns": list(table.columns),
                }
                for name, table in tables.items()
            },
        },
    )
    paths["manifest"] = manifest
    return paths


def _score_nomenclature(stability, cross) -> pd.DataFrame:
    method_summary = pd.read_parquet(
        stability.paths.run_dir / "method_audit/method_summary.parquet"
    )
    cross_metrics = pd.read_parquet(cross.paths.run_dir / "evaluation/domain_metrics.parquet")

    def megascale(method: str) -> float:
        selected = method_summary.loc[
            method_summary["method"].eq(method)
            & method_summary["evaluation_population"].eq("megascale_stability_dense")
            & method_summary["stratum"].eq("all")
            & method_summary["metric"].eq("spearman"),
            "estimate",
        ]
        return float(selected.iloc[0])

    def fireprot(method: str) -> float:
        return float(cross_metrics.loc[cross_metrics["method"].eq(method), "spearman"].mean())

    rows = [
        {
            "canonical_id": "S_ESM2_150",
            "display_name": "ESM2-150M sequence action",
            "formula": "log p_seq(mut | sequence with site masked) - log p_seq(WT | same context)",
            "analysis_role": "sequence baseline component",
            "panel": "Megascale-32",
            "spearman": megascale("esm2_150M_loo"),
            "registered": True,
        },
        {
            "canonical_id": "A_teacher",
            "display_name": "single-teacher paired action",
            "formula": "log p_teacher(mut | WT backbone) - log p_teacher(WT | WT backbone)",
            "analysis_role": "structure-conditioned action component",
            "panel": "definition",
            "spearman": float("nan"),
            "registered": True,
        },
        {
            "canonical_id": "S_plus_A_temperature",
            "display_name": "registered stability study predictor",
            "formula": "S_ESM2_150 + mean_teacher(A_teacher / T_teacher)",
            "analysis_role": "registered confirmatory predictor",
            "panel": "Megascale-32",
            "spearman": megascale("esm2_150M_plus_temperature_consensus"),
            "registered": True,
        },
        {
            "canonical_id": "A_temperature",
            "display_name": "temperature action consensus",
            "formula": "mean_teacher(A_teacher / T_teacher)",
            "analysis_role": (
                "registered FireProt primary; post-lock Megascale action-only ablation"
            ),
            "panel": "Megascale-32",
            "spearman": megascale("temperature_consensus_action_only"),
            "registered": False,
        },
        {
            "canonical_id": "A_unscaled",
            "display_name": "unscaled action consensus",
            "formula": "mean_teacher(A_teacher)",
            "analysis_role": "post-lock practical robust-tier predictor",
            "panel": "Megascale-32",
            "spearman": megascale("unscaled_equal_action_only"),
            "registered": False,
        },
        {
            "canonical_id": "S_plus_G_Cplus_temperature",
            "display_name": "registered strong sequence control",
            "formula": "S_ESM2_150 + mean_teacher((G_teacher + Cplus_teacher) / T_teacher)",
            "analysis_role": "outcome-free sequence control",
            "panel": "Megascale-32",
            "spearman": megascale("esm2_150M_plus_G_Cplus"),
            "registered": True,
        },
        {
            "canonical_id": "simplified_sequence_prior_sum",
            "display_name": "simplified ESM2 plus ESM-IF1 sum",
            "formula": "S_ESM2_150 + A_ESM_IF1",
            "analysis_role": "simplified sequence-prior sum; not an official free-energy protocol",
            "panel": "Megascale-32",
            "spearman": megascale("esm2_150M_plus_unscaled_esm_if1"),
            "registered": False,
        },
        {
            "canonical_id": "FireProt_A_primary",
            "display_name": "FireProt registered A",
            "formula": "mean_teacher(A_teacher / T_teacher); no sequence term",
            "analysis_role": "registered cross-platform primary",
            "panel": "FireProt-18",
            "spearman": fireprot("temperature_consensus_action"),
            "registered": True,
        },
        {
            "canonical_id": "FireProt_G_Cplus_primary",
            "display_name": "FireProt registered G+Cplus",
            "formula": "mean_teacher((G_teacher + Cplus_teacher) / T_teacher); no sequence term",
            "analysis_role": "registered cross-platform comparator",
            "panel": "FireProt-18",
            "spearman": fireprot("temperature_consensus_g_plus_c_plus"),
            "registered": True,
        },
        {
            "canonical_id": "FireProt_A_unscaled",
            "display_name": "FireProt unscaled A",
            "formula": "mean_teacher(A_teacher); no sequence term",
            "analysis_role": "post-lock simplification sensitivity",
            "panel": "FireProt-18",
            "spearman": fireprot("unscaled_consensus_action"),
            "registered": False,
        },
        {
            "canonical_id": "FireProt_S_plus_A_temperature",
            "display_name": "FireProt matched sequence sensitivity",
            "formula": "S_ESM2_150 + mean_teacher(A_teacher / T_teacher)",
            "analysis_role": "registered matched-sequence sensitivity",
            "panel": "FireProt-18",
            "spearman": fireprot("sequence_plus_temperature_action"),
            "registered": True,
        },
    ]
    table = pd.DataFrame(rows)
    table["stability_labels_used_for_fitting_or_selection"] = False
    return table


def _fireprot_tables(
    project_root: Path,
    cross,
    specification: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    source = cross.paths.source_csv
    raw = pd.read_csv(source)
    variants = pd.read_parquet(cross.paths.run_dir / "evaluation/variants.parquet")
    components = pd.read_parquet(cross.paths.run_dir / "evaluation/variant_components.parquet")
    domain_metrics = pd.read_parquet(cross.paths.run_dir / "evaluation/domain_metrics.parquet")
    domain_contrasts = pd.read_parquet(cross.paths.run_dir / "evaluation/domain_contrasts.parquet")
    domains = pd.read_parquet(cross.paths.run_dir / "panel/domains.parquet")
    selected_raw = _selected_fireprot_rows(raw, variants)
    measurement = _measurement_table(selected_raw, specification)
    method_summary = _fireprot_method_summary(domain_metrics, specification)
    domain_results = _fireprot_domain_results(
        domain_metrics, domain_contrasts, variants, domains, selected_raw
    )
    subset_domain, subset_summary = _fireprot_subset_sensitivity(
        selected_raw,
        measurement,
        components,
        specification,
    )
    structure_audit = pd.DataFrame(
        [_pdb_metadata(row.domain_id, Path(row.structure_path)) for row in domains.itertuples()]
    )
    domain_metadata = _fireprot_domain_metadata(variants, domains, selected_raw, structure_audit)
    failure_case = _fireprot_failure_case(
        domain_contrasts,
        domain_metrics,
        variants,
        selected_raw,
        domain_metadata,
        domains,
    )
    provenance = _fireprot_provenance(
        project_root,
        source,
        raw,
        variants,
        measurement,
        domains,
    )
    measurement_summary = _fireprot_measurement_summary(measurement)
    return {
        "fireprot_provenance": provenance,
        "fireprot_measurement_audit": measurement,
        "fireprot_measurement_summary": measurement_summary,
        "fireprot_method_summary": method_summary,
        "fireprot_domain_results": domain_results,
        "fireprot_subset_domain_metrics": subset_domain,
        "fireprot_subset_summary": subset_summary,
        "fireprot_structure_audit": structure_audit,
        "fireprot_domain_metadata": domain_metadata,
        "fireprot_failure_case": failure_case,
    }


def _selected_fireprot_rows(raw: pd.DataFrame, variants: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame["domain_id"] = (
        frame["pdb_id_corrected"].astype(str).str.upper() + ":" + frame["chain"].astype(str)
    )
    frame["position"] = pd.to_numeric(frame["pdb_position"], errors="coerce")
    frame["wild_type"] = frame["wild_type"].astype(str).str.upper()
    frame["mutant"] = frame["mutation"].astype(str).str.upper()
    frame["ddG"] = pd.to_numeric(frame["ddG"], errors="coerce")
    frame = frame.dropna(subset=["position", "ddG"])
    frame["position"] = frame["position"].astype(int)
    keys = ["domain_id", "position", "wild_type", "mutant"]
    selected = variants[keys].drop_duplicates()
    result = frame.merge(selected, on=keys, validate="many_to_one")
    if result[keys].drop_duplicates().shape[0] != len(selected):
        raise ValueError("full FireProt metadata does not cover every selected variant")
    return result.sort_values([*keys, "experiment_id"], ignore_index=True)


def _measurement_table(
    selected_raw: pd.DataFrame, specification: Mapping[str, Any]
) -> pd.DataFrame:
    keys = ["domain_id", "position", "wild_type", "mutant"]
    rows = []
    settings = specification["fireprot_measurement_audit"]
    minimum = int(settings["repeated_measurement_minimum"])
    maximum_range = float(settings["high_consistency_maximum_ddg_range_kcal_mol"])
    requires_sign_agreement = bool(settings["high_consistency_requires_sign_agreement"])
    for key, frame in selected_raw.groupby(keys, sort=True, observed=True):
        values = frame["ddG"].to_numpy(dtype=float)
        q25, q75 = np.quantile(values, [0.25, 0.75])
        sign_agreement = not (np.any(values < 0) and np.any(values > 0))
        ddg_range = float(values.max() - values.min())
        high_consistency = (
            len(values) >= minimum
            and (sign_agreement or not requires_sign_agreement)
            and ddg_range <= maximum_range
        )
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "measurement_count": len(values),
                "ddg_median": float(np.median(values)),
                "ddg_mean": float(values.mean()),
                "ddg_std": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                "ddg_iqr": float(q75 - q25),
                "ddg_min": float(values.min()),
                "ddg_max": float(values.max()),
                "ddg_range": ddg_range,
                "measurement_direction_consistent": sign_agreement,
                "high_consistency_eligible": high_consistency,
                "all_records_curated": bool(frame["is_curated"].fillna(False).all()),
                "any_record_curated": bool(frame["is_curated"].fillna(False).any()),
                "unique_ph_values": int(frame["pH"].nunique(dropna=True)),
                "unique_methods": int(frame["method"].nunique(dropna=True)),
                "unique_techniques": int(frame["technique"].nunique(dropna=True)),
                "unique_publications": int(frame["publication_doi"].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def _fireprot_method_summary(
    domain_metrics: pd.DataFrame, specification: Mapping[str, Any]
) -> pd.DataFrame:
    rows = []
    seed = int(specification["seed"]) + 60_000
    for method_index, (method, frame) in enumerate(
        domain_metrics.groupby("method", sort=True, observed=True)
    ):
        working = frame.assign(stratum="fireprot")
        for metric_index, metric in enumerate(("spearman", "ndcg10")):
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    **_bootstrap(
                        working,
                        metric,
                        specification,
                        seed + method_index * 10 + metric_index,
                    ),
                }
            )
    return pd.DataFrame(rows)


def _fireprot_domain_results(
    metrics: pd.DataFrame,
    contrasts: pd.DataFrame,
    variants: pd.DataFrame,
    domains: pd.DataFrame,
    selected_raw: pd.DataFrame,
) -> pd.DataFrame:
    counts = variants.groupby("domain_id", observed=True).agg(
        n_variants=("mutant", "size"),
        n_query_positions=("position", "nunique"),
    )
    primary = contrasts.loc[
        contrasts["contrast"].eq("temperature_action_vs_g_plus_c_plus"),
        [
            "domain_id",
            "spearman",
            "comparator_spearman",
            "spearman_margin",
            "ndcg10",
            "comparator_ndcg10",
            "ndcg10_margin",
        ],
    ].rename(
        columns={
            "spearman": "temperature_action_spearman",
            "comparator_spearman": "g_plus_c_plus_spearman",
            "ndcg10": "temperature_action_ndcg10",
            "comparator_ndcg10": "g_plus_c_plus_ndcg10",
        }
    )
    absolute = metrics.pivot(index="domain_id", columns="method", values=["spearman", "ndcg10"])
    absolute.columns = [f"{method}_{metric}" for metric, method in absolute.columns]
    metadata = domains.set_index("domain_id")[["protein_name", "pdb_id", "uniprot_id", "length"]]
    source = selected_raw.groupby("domain_id", observed=True).agg(
        source_structure_methods=("structure_method", _joined),
        source_publications=("publication_doi", "nunique"),
        curated_record_fraction=("is_curated", "mean"),
    )
    return (
        metadata.join(counts)
        .join(primary.set_index("domain_id"))
        .join(absolute)
        .join(source)
        .reset_index()
        .sort_values("domain_id", ignore_index=True)
    )


def _fireprot_subset_sensitivity(
    selected_raw: pd.DataFrame,
    measurement: pd.DataFrame,
    components: pd.DataFrame,
    specification: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["domain_id", "position", "wild_type", "mutant"]
    minimum = int(specification["inference"]["minimum_variants_per_subset_domain"])
    subsets: dict[str, pd.DataFrame] = {
        "all_records": selected_raw,
        "curated_records_only": selected_raw.loc[selected_raw["is_curated"].fillna(False)],
        "uncurated_records_only": selected_raw.loc[~selected_raw["is_curated"].fillna(False)],
    }
    single_keys = measurement.loc[measurement["measurement_count"].eq(1), keys]
    repeated_keys = measurement.loc[measurement["measurement_count"].ge(2), keys]
    consistent_keys = measurement.loc[measurement["high_consistency_eligible"], keys]
    subsets["single_measurement_variants"] = selected_raw.merge(
        single_keys, on=keys, validate="many_to_one"
    )
    subsets["repeated_measurement_variants"] = selected_raw.merge(
        repeated_keys, on=keys, validate="many_to_one"
    )
    subsets["high_consistency_repeated_variants"] = selected_raw.merge(
        consistent_keys, on=keys, validate="many_to_one"
    )
    scores = components[[*keys, PRIMARY_FIREPROT_METHOD, PRIMARY_FIREPROT_CONTROL]].drop_duplicates(
        keys
    )
    domain_rows = []
    summary_rows = []
    seed = int(specification["seed"]) + 70_000
    for subset_index, (subset, raw_subset) in enumerate(subsets.items()):
        labels = raw_subset.groupby(keys, as_index=False, observed=True).agg(ddg=("ddG", "median"))
        labels["effect"] = -labels["ddg"]
        frame = scores.merge(labels, on=keys, validate="one_to_one")
        for domain_id, domain in frame.groupby("domain_id", sort=True, observed=True):
            if len(domain) < minimum:
                continue
            observed = domain["effect"].to_numpy(dtype=float)
            action = domain[PRIMARY_FIREPROT_METHOD].to_numpy(dtype=float)
            control = domain[PRIMARY_FIREPROT_CONTROL].to_numpy(dtype=float)
            k = max(1, int(np.ceil(0.10 * len(domain))))
            action_spearman = _spearman(action, observed)
            control_spearman = _spearman(control, observed)
            action_ndcg = _ndcg(action, observed, k=k)
            control_ndcg = _ndcg(control, observed, k=k)
            domain_rows.append(
                {
                    "subset": subset,
                    "domain_id": domain_id,
                    "n_variants": len(domain),
                    "spearman_margin": action_spearman - control_spearman,
                    "ndcg10_margin": action_ndcg - control_ndcg,
                    "stratum": subset,
                }
            )
        domain_table = pd.DataFrame(domain_rows)
        current = (
            domain_table.loc[domain_table["subset"].eq(subset)]
            if len(domain_table)
            else pd.DataFrame()
        )
        for metric_index, metric in enumerate(("spearman_margin", "ndcg10_margin")):
            if current.empty:
                result = _empty_summary()
            else:
                result = _bootstrap(
                    current,
                    metric,
                    specification,
                    seed + subset_index * 10 + metric_index,
                )
            summary_rows.append(
                {
                    "subset": subset,
                    "metric": metric,
                    "available_variants": int(len(labels)),
                    "minimum_variants_per_domain": minimum,
                    "estimable": bool(result["n_domains"] > 0),
                    **result,
                }
            )
    return pd.DataFrame(domain_rows), pd.DataFrame(summary_rows)


def _fireprot_domain_metadata(
    variants: pd.DataFrame,
    domains: pd.DataFrame,
    selected_raw: pd.DataFrame,
    structure_audit: pd.DataFrame,
) -> pd.DataFrame:
    variant_counts = variants.groupby("domain_id", observed=True).agg(
        variants=("mutant", "size"),
        query_positions=("position", "nunique"),
        repeated_variants=("measurement_count", lambda values: int((values >= 2).sum())),
    )
    assay = selected_raw.groupby("domain_id", observed=True).agg(
        raw_measurement_rows=("ddG", "size"),
        curated_record_fraction=("is_curated", "mean"),
        methods=("method", _joined),
        techniques=("technique", _joined),
        ph_min=("pH", "min"),
        ph_max=("pH", "max"),
        unique_ph=("pH", "nunique"),
        publications=("publication_doi", "nunique"),
        upstream_structure_methods=("structure_method", _joined),
    )
    core = domains.set_index("domain_id")[
        ["protein_name", "pdb_id", "uniprot_id", "length", "structure_path"]
    ]
    return (
        core.join(variant_counts)
        .join(assay)
        .join(structure_audit.set_index("domain_id"))
        .reset_index()
        .sort_values("domain_id", ignore_index=True)
    )


def _fireprot_failure_case(
    contrasts: pd.DataFrame,
    metrics: pd.DataFrame,
    variants: pd.DataFrame,
    selected_raw: pd.DataFrame,
    metadata: pd.DataFrame,
    domains: pd.DataFrame,
) -> pd.DataFrame:
    primary = contrasts.loc[
        contrasts["contrast"].eq("temperature_action_vs_g_plus_c_plus")
        & contrasts["spearman_margin"].lt(0)
    ].sort_values("spearman_margin")
    rows = []
    for contrast in primary.itertuples(index=False):
        domain_id = str(contrast.domain_id)
        domain_variants = variants.loc[variants["domain_id"].eq(domain_id)]
        raw = selected_raw.loc[selected_raw["domain_id"].eq(domain_id)]
        row = metadata.loc[metadata["domain_id"].eq(domain_id)].iloc[0]
        selected_sequence = str(domains.loc[domains["domain_id"].eq(domain_id), "sequence"].iloc[0])
        source_pdb_sequences = {
            str(value) for value in raw["pdb_sequence"].dropna().astype(str).unique()
        }
        variant_mapping_matches = all(
            selected_sequence[int(variant.position)] == str(variant.wild_type)
            for variant in domain_variants.itertuples(index=False)
        )
        method_spearman = metrics.loc[metrics["domain_id"].eq(domain_id)].set_index("method")[
            "spearman"
        ]
        rows.append(
            {
                "domain_id": domain_id,
                "protein_name": row["protein_name"],
                "n_variants": len(domain_variants),
                "n_query_positions": domain_variants["position"].nunique(),
                "temperature_action_spearman": float(
                    method_spearman["temperature_consensus_action"]
                ),
                "g_plus_c_plus_spearman": float(
                    method_spearman["temperature_consensus_g_plus_c_plus"]
                ),
                "spearman_margin": float(contrast.spearman_margin),
                "unscaled_action_spearman": float(method_spearman["unscaled_consensus_action"]),
                "mif_action_spearman": float(method_spearman["mif_action"]),
                "esm_if1_action_spearman": float(method_spearman["esm_if1_action"]),
                "proteinmpnn_action_spearman": float(method_spearman["proteinmpnn_action"]),
                "alanine_mutant_fraction": float(domain_variants["mutant"].eq("A").mean()),
                "unique_methods": _joined(raw["method"]),
                "unique_techniques": _joined(raw["technique"]),
                "ph_min": float(raw["pH"].min()),
                "ph_max": float(raw["pH"].max()),
                "publications": int(raw["publication_doi"].nunique(dropna=True)),
                "experimental_method": row["experimental_method"],
                "engineered_structure": bool(row["engineered_structure"]),
                "pdb_chain_count": int(row["pdb_chain_count"]),
                "hetero_residues": row["hetero_residues"],
                "metal_elements": row["metal_elements"],
                "ssbond_records": int(row["ssbond_records"]),
                "sequence_cysteines": int(row["sequence_cysteines"]),
                "source_pdb_sequence_matches_selected_structure_sequence": bool(
                    source_pdb_sequences
                    and all(value == selected_sequence for value in source_pdb_sequences)
                ),
                "selected_variant_wild_types_match_structure_sequence": variant_mapping_matches,
                "interpretation": (
                    "descriptive failure case only; isolated engineered module and assay "
                    "composition "
                    "are observations, not an assigned causal explanation"
                ),
            }
        )
    return pd.DataFrame(rows)


def _fireprot_provenance(
    project_root: Path,
    source: Path,
    raw: pd.DataFrame,
    variants: pd.DataFrame,
    measurement: pd.DataFrame,
    domains: pd.DataFrame,
) -> pd.DataFrame:
    repository = source.parents[2]
    commit = _git(repository, "rev-parse", "HEAD")
    commit_date = _git(repository, "show", "-s", "--format=%aI", "HEAD")
    mtime = datetime.fromtimestamp(source.stat().st_mtime, tz=timezone.utc).isoformat()
    return pd.DataFrame(
        [
            {
                "source_name": "ThermoMPNN FireProt homologue-free CSV",
                "source_repository": "https://github.com/Kuhlman-Lab/ThermoMPNN",
                "source_repository_commit": commit,
                "source_repository_commit_date": commit_date,
                "upstream_dataset_doi": "https://doi.org/10.5281/zenodo.8169288",
                "embedded_fireprotdb_release_number": "not encoded in source CSV",
                "source_file": str(source),
                "source_file_mtime_utc": mtime,
                "source_rows": len(raw),
                "source_columns": len(raw.columns),
                "selected_domains": domains["domain_id"].nunique(),
                "selected_variants": len(variants),
                "selected_query_positions": variants[["domain_id", "position"]]
                .drop_duplicates()
                .shape[0],
                "selected_raw_measurement_rows": int(measurement["measurement_count"].sum()),
                "endpoint_column": "ddG",
                "unit_interpretation": (
                    "kcal/mol according to upstream FireProt/ThermoMPNN dataset convention; "
                    "the CSV has no explicit unit column"
                ),
                "effect_formula": "effect = -median(ddG) for each unique substitution",
                "positive_effect_meaning": "stabilizing",
                "duplicate_handling": "median across rows sharing domain, position, WT, mutant",
                "selection_protocol": str(project_root / "configs/external_validation.yaml"),
            }
        ]
    )


def _fireprot_measurement_summary(measurement: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("unique_variants", len(measurement)),
        ("single_measurement_variants", int(measurement["measurement_count"].eq(1).sum())),
        ("repeated_measurement_variants", int(measurement["measurement_count"].ge(2).sum())),
        (
            "direction_consistent_repeated_variants",
            int(
                (
                    measurement["measurement_count"].ge(2)
                    & measurement["measurement_direction_consistent"]
                ).sum()
            ),
        ),
        (
            "high_consistency_repeated_variants",
            int(measurement["high_consistency_eligible"].sum()),
        ),
        ("all_records_curated_variants", int(measurement["all_records_curated"].sum())),
        ("any_record_curated_variants", int(measurement["any_record_curated"].sum())),
    ]
    return pd.DataFrame(rows, columns=["quantity", "value"])


def _pdb_metadata(domain_id: str, path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    atom_lines = [line for line in lines if line.startswith(("ATOM  ", "HETATM"))]
    chains = {line[21:22].strip() for line in atom_lines if line[21:22].strip()}
    hetero = {
        line[17:20].strip()
        for line in lines
        if line.startswith("HETATM") and line[17:20].strip() not in {"HOH", "DOD"}
    }
    metals = {
        line[76:78].strip().upper()
        for line in lines
        if line.startswith("HETATM")
        and line[76:78].strip().upper()
        in {"ZN", "FE", "MG", "MN", "CA", "CU", "CO", "NI", "NA", "K", "CD", "HG"}
    }
    method = " ".join(line[10:].strip() for line in lines if line.startswith("EXPDTA"))
    resolution = float("nan")
    for line in lines:
        if line.startswith("REMARK   2 RESOLUTION."):
            match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s+ANGSTROMS", line)
            if match:
                resolution = float(match.group(1))
                break
    sequence = "".join(
        {
            "ALA": "A",
            "ARG": "R",
            "ASN": "N",
            "ASP": "D",
            "CYS": "C",
            "GLN": "Q",
            "GLU": "E",
            "GLY": "G",
            "HIS": "H",
            "ILE": "I",
            "LEU": "L",
            "LYS": "K",
            "MET": "M",
            "PHE": "F",
            "PRO": "P",
            "SER": "S",
            "THR": "T",
            "TRP": "W",
            "TYR": "Y",
            "VAL": "V",
        }.get(residue, "X")
        for line in lines
        if line.startswith("SEQRES")
        for residue in line[19:].split()
    )
    return {
        "domain_id": domain_id,
        "experimental_method": method,
        "resolution_angstrom": resolution,
        "engineered_structure": any(
            line.startswith("COMPND") and "ENGINEERED: YES" in line for line in lines
        ),
        "pdb_chain_count": len(chains),
        "pdb_model_count": max(1, sum(line.startswith("MODEL ") for line in lines)),
        "hetero_residues": _joined(pd.Series(sorted(hetero), dtype="object")),
        "metal_elements": _joined(pd.Series(sorted(metals), dtype="object")),
        "ssbond_records": sum(line.startswith("SSBOND") for line in lines),
        "sequence_cysteines": sequence.count("C"),
    }


def _fast_robust_tables(stability, cross, specification) -> dict[str, pd.DataFrame]:
    mega = pd.read_parquet(stability.paths.run_dir / "method_audit/domain_metrics.parquet")
    mega = mega.loc[mega["evaluation_population"].eq("megascale_stability_dense")]
    mega_fast = mega.loc[
        mega["method"].eq(FAST_METHOD), ["domain_id", "stratum", "spearman", "ndcg_at_10_percent"]
    ].rename(columns={"spearman": "fast_spearman", "ndcg_at_10_percent": "fast_ndcg10"})
    mega_robust = mega.loc[
        mega["method"].eq(ROBUST_METHOD), ["domain_id", "stratum", "spearman", "ndcg_at_10_percent"]
    ].rename(columns={"spearman": "robust_spearman", "ndcg_at_10_percent": "robust_ndcg10"})
    mega_contrast = mega_robust.merge(mega_fast, on=["domain_id", "stratum"], validate="one_to_one")
    mega_contrast["panel"] = "Megascale-32"
    cross_metrics = pd.read_parquet(cross.paths.run_dir / "evaluation/domain_metrics.parquet")
    fire_fast = cross_metrics.loc[
        cross_metrics["method"].eq("esm_if1_action"), ["domain_id", "spearman", "ndcg10"]
    ].rename(columns={"spearman": "fast_spearman", "ndcg10": "fast_ndcg10"})
    fire_robust = cross_metrics.loc[
        cross_metrics["method"].eq("unscaled_consensus_action"),
        ["domain_id", "spearman", "ndcg10"],
    ].rename(columns={"spearman": "robust_spearman", "ndcg10": "robust_ndcg10"})
    fire_contrast = fire_robust.merge(fire_fast, on="domain_id", validate="one_to_one")
    fire_contrast["stratum"] = "fireprot"
    fire_contrast["panel"] = "FireProt-18"
    domain = pd.concat([mega_contrast, fire_contrast], ignore_index=True)
    domain["spearman_margin"] = domain["robust_spearman"] - domain["fast_spearman"]
    domain["ndcg10_margin"] = domain["robust_ndcg10"] - domain["fast_ndcg10"]
    summary = _fast_robust_summary(domain, specification)
    runtime_domain = pd.concat(
        [
            _runtime_domain_table(
                stability.paths.run_dir,
                "Megascale-32",
                pd.read_parquet(stability.paths.run_dir / "panel/variants.parquet").loc[
                    lambda frame: frame["evaluation_population"].eq("megascale_stability_dense")
                ],
            ),
            _runtime_domain_table(
                cross.paths.run_dir,
                "FireProt-18",
                pd.read_parquet(cross.paths.run_dir / "evaluation/variants.parquet"),
            ),
        ],
        ignore_index=True,
    )
    runtime_summary = _runtime_summary(runtime_domain)
    return {
        "fast_robust_domain_contrasts": domain,
        "fast_robust_summary": summary,
        "fast_robust_runtime_domain": runtime_domain,
        "fast_robust_runtime_summary": runtime_summary,
    }


def _fast_robust_summary(domain: pd.DataFrame, specification: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    seed = int(specification["seed"]) + 80_000
    group_index = 0
    for panel, panel_frame in domain.groupby("panel", sort=True, observed=True):
        scopes = [("all", panel_frame)]
        if panel == "Megascale-32":
            scopes.extend(
                (str(stratum), frame)
                for stratum, frame in panel_frame.groupby("stratum", sort=True, observed=True)
            )
        for scope, frame in scopes:
            for metric_index, metric in enumerate(("spearman_margin", "ndcg10_margin")):
                rows.append(
                    {
                        "panel": panel,
                        "scope": scope,
                        "metric": metric,
                        **_bootstrap(
                            frame,
                            metric,
                            specification,
                            seed + group_index * 10 + metric_index,
                        ),
                    }
                )
            group_index += 1
    return pd.DataFrame(rows)


def _runtime_domain_table(run_dir: Path, panel: str, variants: pd.DataFrame) -> pd.DataFrame:
    requests = pd.read_parquet(run_dir / "teacher_requests/requests.parquet")[
        ["request_id", "domain_id", "length"]
    ]
    variant_counts = variants.groupby("domain_id")["mutant"].size().rename("panel_variants")
    requests = requests.loc[requests["domain_id"].isin(variant_counts.index)]
    timings = {}
    for teacher in ("mif", "esm_if1", "proteinmpnn"):
        raw = pd.read_parquet(run_dir / "teacher_scores/raw" / f"{teacher}.parquet")
        timing = raw[["request_id", "wall_seconds"]].drop_duplicates("request_id")
        timings[teacher] = timing.set_index("request_id")["wall_seconds"]
    rows = []
    for request in requests.itertuples(index=False):
        values = {teacher: float(timings[teacher].loc[request.request_id]) for teacher in timings}
        methods = {
            "ESM-IF1 fast tier": values["esm_if1"],
            "unscaled three-teacher robust tier": sum(values.values()),
        }
        for method, seconds in methods.items():
            rows.append(
                {
                    "panel": panel,
                    "domain_id": request.domain_id,
                    "method": method,
                    "length": int(request.length),
                    "panel_variants": int(variant_counts.loc[request.domain_id]),
                    "full_scan_substitutions": int(19 * request.length),
                    "wall_seconds": seconds,
                    "seconds_per_100_residues": seconds / request.length * 100.0,
                    "seconds_per_1000_full_scan_substitutions": seconds
                    / (19 * request.length)
                    * 1000.0,
                }
            )
    return pd.DataFrame(rows)


def _runtime_summary(domain: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (panel, method), frame in domain.groupby(["panel", "method"], sort=True, observed=True):
        rows.append(
            {
                "panel": panel,
                "method": method,
                "domains": len(frame),
                "residues": int(frame["length"].sum()),
                "full_scan_substitutions": int(frame["full_scan_substitutions"].sum()),
                "wall_seconds_total": float(frame["wall_seconds"].sum()),
                "wall_seconds_p50": float(frame["wall_seconds"].quantile(0.50)),
                "wall_seconds_p90": float(frame["wall_seconds"].quantile(0.90)),
                "wall_seconds_max": float(frame["wall_seconds"].max()),
                "seconds_per_100_residues": float(
                    frame["wall_seconds"].sum() / frame["length"].sum() * 100.0
                ),
                "seconds_per_1000_full_scan_substitutions": float(
                    frame["wall_seconds"].sum() / frame["full_scan_substitutions"].sum() * 1000.0
                ),
                "timing_scope": (
                    "adapter scoring of all 19 substitutions; excludes model load and "
                    "structure generation"
                ),
            }
        )
    result = pd.DataFrame(rows)
    ratios = []
    for panel, frame in result.groupby("panel", observed=True):
        indexed = frame.set_index("method")
        ratios.append(
            {
                "panel": panel,
                "method": "robust/fast p50 ratio",
                "domains": int(indexed.iloc[0]["domains"]),
                "residues": int(indexed.iloc[0]["residues"]),
                "full_scan_substitutions": int(indexed.iloc[0]["full_scan_substitutions"]),
                "wall_seconds_total": float("nan"),
                "wall_seconds_p50": float(
                    indexed.loc["unscaled three-teacher robust tier", "wall_seconds_p50"]
                    / indexed.loc["ESM-IF1 fast tier", "wall_seconds_p50"]
                ),
                "wall_seconds_p90": float("nan"),
                "wall_seconds_max": float("nan"),
                "seconds_per_100_residues": float("nan"),
                "seconds_per_1000_full_scan_substitutions": float("nan"),
                "timing_scope": "ratio stored in wall_seconds_p50",
            }
        )
    return pd.concat([result, pd.DataFrame(ratios)], ignore_index=True)


def _render_report(tables: Mapping[str, pd.DataFrame], specification: Mapping[str, Any]) -> str:
    nomenclature = tables["score_nomenclature"]
    fire_summary = tables["fireprot_method_summary"]
    fire_contrast = tables["fireprot_subset_summary"]
    fast = tables["fast_robust_summary"]
    teacher_delta = tables["structure_sensitivity_teacher_delta_summary"]
    geometry = tables["structure_sensitivity_geometry_summary"]
    distribution = tables["structure_sensitivity_teacher_distribution_summary"]
    backbone = tables["structure_sensitivity_backbone_geometry_summary"]
    geometry_action = tables["structure_sensitivity_geometry_action_correlation_summary"]
    provenance = tables["fireprot_provenance"].iloc[0]
    failure = tables["fireprot_failure_case"].iloc[0]
    runtime = tables["fast_robust_runtime_summary"]
    measurement = tables["fireprot_measurement_summary"].set_index("quantity")["value"]

    def score(canonical: str) -> float:
        return float(nomenclature.set_index("canonical_id").loc[canonical, "spearman"])

    def estimate(table: pd.DataFrame, **conditions: str) -> pd.Series:
        mask = pd.Series(True, index=table.index)
        for column, value in conditions.items():
            mask &= table[column].eq(value)
        selected = table.loc[mask]
        if len(selected) != 1:
            raise ValueError(f"report lookup is not unique: {conditions}")
        return selected.iloc[0]

    primary_spearman = estimate(fire_summary, method=PRIMARY_FIREPROT_METHOD, metric="spearman")
    primary_ndcg = estimate(fire_summary, method=PRIMARY_FIREPROT_METHOD, metric="ndcg10")
    gc_spearman = estimate(fire_summary, method=PRIMARY_FIREPROT_CONTROL, metric="spearman")
    full_subset = estimate(fire_contrast, subset="all_records", metric="spearman_margin")
    mega_fast = estimate(fast, panel="Megascale-32", scope="all", metric="spearman_margin")
    fire_fast = estimate(fast, panel="FireProt-18", scope="all", metric="spearman_margin")
    mega_fast_ndcg = estimate(fast, panel="Megascale-32", scope="all", metric="ndcg10_margin")
    fire_fast_ndcg = estimate(fast, panel="FireProt-18", scope="all", metric="ndcg10_margin")

    robustness_lines = []
    subset_labels = {
        "all_records": "All selected records",
        "single_measurement_variants": "Single-measurement only",
        "curated_records_only": "Curated records only",
        "uncurated_records_only": "Uncurated records only",
    }
    for subset, label in subset_labels.items():
        row = estimate(fire_contrast, subset=subset, metric="spearman_margin")
        robustness_lines.append(
            f"| {label} | {int(row['available_variants']):,} | {int(row['n_domains'])} | "
            f"{row['estimate']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
            f"{int(row['positive_domains'])}/{int(row['n_domains'])} | "
            f"[{row['leave_one_domain_out_min']:+.4f}, "
            f"{row['leave_one_domain_out_max']:+.4f}] |"
        )

    runtime_lines = []
    runtime_labels = {
        "ESM-IF1 fast tier": "Fast: ESM-IF1",
        "unscaled three-teacher robust tier": "Robust: three teachers",
    }
    for panel in ("Megascale-32", "FireProt-18"):
        ratio = estimate(runtime, panel=panel, method="robust/fast p50 ratio")
        for method, label in runtime_labels.items():
            row = estimate(runtime, panel=panel, method=method)
            ratio_text = (
                f"{ratio['wall_seconds_p50']:.1f}×" if method.startswith("unscaled") else "—"
            )
            runtime_lines.append(
                f"| {panel} | {label} | {row['wall_seconds_p50']:.3f} | "
                f"{row['wall_seconds_p90']:.3f} | {row['wall_seconds_max']:.3f} | "
                f"{ratio_text} |"
            )

    teacher_lines = []
    for teacher in ("mif", "esm_if1", "proteinmpnn", "registered_temperature_consensus"):
        role_values = []
        for role in ("alphafold", "perturbed_0p5", "perturbed_1p0"):
            row = estimate(
                teacher_delta,
                teacher_id=teacher,
                structure_role=role,
                metric="action_spearman_delta_vs_experimental",
            )
            role_values.append(
                f"{row['estimate']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"
            )
        teacher_lines.append(
            f"| `{teacher}` | {role_values[0]} | {role_values[1]} | {role_values[2]} |"
        )

    local_action = estimate(
        geometry_action,
        teacher_id="registered_temperature_consensus",
        structure_role="alphafold",
        geometry_metric="local_backbone_rmsd_10a",
    )
    frame_action = estimate(
        geometry_action,
        teacher_id="registered_temperature_consensus",
        structure_role="alphafold",
        geometry_metric="local_frame_angular_error_degrees",
    )
    lddt_action = estimate(
        geometry_action,
        teacher_id="registered_temperature_consensus",
        structure_role="alphafold",
        geometry_metric="ca_lddt15",
    )

    geometry_lines = []
    for role in ("alphafold", "perturbed_0p5", "perturbed_1p0"):
        local = estimate(geometry, structure_role=role, metric="local_backbone_rmsd_10a")
        frame = estimate(
            geometry,
            structure_role=role,
            metric="local_frame_angular_error_degrees",
        )
        geometry_lines.append(
            f"| `{role}` | {local['estimate']:.3f} | {local['pooled_position_p90']:.3f} | "
            f"{frame['estimate']:.2f} |"
        )

    distribution_lines = []
    for teacher in ("mif", "esm_if1", "proteinmpnn"):
        for role in ("alphafold", "perturbed_0p5", "perturbed_1p0"):
            jsd = estimate(
                distribution,
                teacher_id=teacher,
                structure_role=role,
                metric="jsd_mean_vs_experimental",
            )
            nll = estimate(
                distribution,
                teacher_id=teacher,
                structure_role=role,
                metric="native_nll_delta_vs_experimental",
            )
            distribution_lines.append(
                f"| `{teacher}` | `{role}` | {jsd['estimate']:.4f} | {nll['estimate']:+.4f} |"
            )

    peptide_lines = []
    for role in ("alphafold", "perturbed_0p5", "perturbed_1p0"):
        cn = estimate(
            backbone,
            structure_role=role,
            metric="peptide_cn_mae_vs_experimental_angstrom",
        )
        omega = estimate(
            backbone,
            structure_role=role,
            metric="omega_mae_vs_experimental_degrees",
        )
        breaks = estimate(
            backbone,
            structure_role=role,
            metric="chain_break_fraction_cn_gt_2a",
        )
        peptide_lines.append(
            f"| `{role}` | {cn['estimate']:.3f} | {omega['estimate']:.2f} | "
            f"{breaks['estimate']:.3f} |"
        )

    figure_markdown = (
        "![FireProt confirmation, ensemble gain, and structure-sensitivity audit]"
        "(../figures/figure4_submission_audit.png)"
    )

    return f"""# stability study post-lock submission audit

_Descriptive audit recorded after outcome opening · seed `{specification["seed"]}` · no gate change_

---

## Executive conclusion

The finite pre-submission audit is complete. The registered stability study and cross-platform
decisions remain unchanged. The defensible central claim is:

> Paired inverse-folding mutation actions provide reproducible,
> stability-label-free ranking information beyond strong outcome-free sequence controls
> across dense Megascale and homology-isolated FireProt proteins. High-quality predicted
> wild-type backbones retain this advantage in the matched panel, while degradation under
> the registered 1 Å smooth perturbation reflects both geometric displacement and a
> measurable shift in teacher input distribution.

This result does not establish a pure structural causal effect, unseen-protein pretraining
generalization, predicted/experimental structure equivalence, mutant relaxation, or routing.

{figure_markdown}
_Figure 4: a, Per-protein FireProt action-minus-G+Cplus margins. b, Absolute FireProt
method performance with 95% protein-bootstrap intervals. c, Paired robust-minus-fast
differences; open circles are proteins and filled points are equal-protein means with 95%
bootstrap intervals. d, Matched structure-source action change versus local backbone error;
both axes show 95% protein-bootstrap intervals._

## Score definitions and canonical names

The three frequently confused Megascale values now have one interpretation each:

| Canonical score | Exact formula | Mean Spearman |
| --- | --- | ---: |
| Registered predictor | `S_ESM2_150 + mean(A_t / T_t)` | {score("S_plus_A_temperature"):.4f} |
| Temperature action-only | `mean(A_t / T_t)` | {score("A_temperature"):.4f} |
| Unscaled action-only | `mean(A_t)` | {score("A_unscaled"):.4f} |

FireProt's registered `A` is the temperature action-only score, with no sequence term. Its
absolute Spearman is {primary_spearman["estimate"]:.4f}
[{primary_spearman["ci_low"]:.4f}, {primary_spearman["ci_high"]:.4f}], versus
{gc_spearman["estimate"]:.4f} for the registered action-only `G+Cplus` comparator. The
registered action NDCG@10% is {primary_ndcg["estimate"]:.4f}
[{primary_ndcg["ci_low"]:.4f}, {primary_ndcg["ci_high"]:.4f}]. Full formulas and roles are
in `source_data/score_nomenclature.csv`.

The former “free-energy sum” wording is retired. The implemented comparator is a
**simplified sequence-prior sum**, `S_ESM2_150 + A_ESM_IF1`; it is not a reproduction of an
official ensemble or importance-sampling free-energy protocol.

## FireProt data and robustness audit

The source is pinned to ThermoMPNN commit `{str(provenance["source_repository_commit"])[:12]}`
and contains {int(provenance["source_rows"]):,} raw rows. The selected panel retains
{int(provenance["selected_raw_measurement_rows"]):,} measurement rows. The upstream
repository identifies Zenodo record
8169288 as the dataset release.[^1][^2] The CSV contains no embedded FireProtDB release
number or unit column, so both facts are stated explicitly rather than inferred silently.
The endpoint is the upstream `ddG` field and the evaluation effect is
`-median(ddG)`, where positive means stabilizing.

Of {int(measurement["unique_variants"]):,} selected variants,
{int(measurement["single_measurement_variants"]):,} have one row and only
{int(measurement["repeated_measurement_variants"])} have two rows. Just
{int(measurement["high_consistency_repeated_variants"])} variant satisfies the recorded
repeat/sign/range definition. A repeated-measurement correlation analysis is therefore not
estimable; claiming one would be misleading.

| Analysis subset | Variants | Proteins | ΔSpearman [95% CI] | Positive proteins | LOPO range |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(robustness_lines)}

Removing the three repeated variants leaves the result unchanged, so duplicate aggregation
is not driving the primary result. The curated-only interval remains above zero; the smaller
uncurated-only interval crosses zero. The all-record leave-one-protein-out range is
[{full_subset["leave_one_domain_out_min"]:+.4f},
{full_subset["leave_one_domain_out_max"]:+.4f}]. Repeated-only and high-consistency-only
effects are not estimable because they contain only three and one variants, respectively.
All absolute methods, NDCG intervals, and per-protein counts are provided as source tables.

The sole negative primary domain, `{failure["domain_id"]}`, is retained: its
{int(failure["n_variants"])} variants give action Spearman
{failure["temperature_action_spearman"]:+.4f}, G+Cplus Spearman
{failure["g_plus_c_plus_spearman"]:+.4f}, and a margin of
{failure["spearman_margin"]:+.4f}. It is an engineered solution-NMR fibronectin module and
{failure["alanine_mutant_fraction"]:.1%} of selected mutations are to alanine. These are
failure-case observations, not an assigned causal explanation; the full audit is retained
in `source_data/fireprot_failure_case.csv`.

## Fast versus robust tier

The unscaled three-teacher action consensus exceeds ESM-IF1 action-only on Megascale by
{mega_fast["estimate"]:+.4f} Spearman
[{mega_fast["ci_low"]:+.4f}, {mega_fast["ci_high"]:+.4f}]. On FireProt the paired difference
is {fire_fast["estimate"]:+.4f}
[{fire_fast["ci_low"]:+.4f}, {fire_fast["ci_high"]:+.4f}]. The corresponding NDCG@10%
differences are {mega_fast_ndcg["estimate"]:+.4f} and {fire_fast_ndcg["estimate"]:+.4f}.

| Panel | Tier | p50, s/protein | p90 | max | Robust/fast p50 |
| --- | --- | ---: | ---: | ---: | ---: |
{chr(10).join(runtime_lines)}

These timings cover adapter scoring of all 19 substitutions and exclude model loading and
structure generation. Natural/de novo strata and full normalization details are retained in
the paired source tables. Product language should therefore follow the intervals: ESM-IF1
is the fast tier; the three-teacher predictor is the robust tier, with ensemble gain and
cost described separately for each panel.

## structure-sensitivity study teacher and geometry audit

### Teacher-specific matched differences

| Teacher | AFDB − experimental | +0.5 Å − experimental | +1.0 Å − experimental |
| --- | ---: | ---: | ---: |
{chr(10).join(teacher_lines)}

These are descriptive paired differences. A confidence interval crossing zero is reported
as no detectable difference, not as equivalence. All three teachers move downward under
the 1 Å perturbation; MIF's interval narrowly crosses zero, while ESM-IF1, ProteinMPNN, and
the registered consensus do not. The teacher rows prevent the consensus from masking
architecture-specific responses.

### Local geometry on one common axis

| Structure role | Mean local RMSD, Å | Position p90, Å | Mean frame error, degrees |
| --- | ---: | ---: | ---: |
{chr(10).join(geometry_lines)}

The table aligns real AFDB–experimental differences and registered synthetic perturbations
under the same metric: after a global Cα rigid alignment, backbone displacement is averaged
over residues whose experimental Cα lies within 10 Å of the queried site. It is not a second
local fit. Cα-lDDT15, contact retention, pLDDT, and within-domain correlations with
action-score change are available in the source tables. These analyses use wild-type
scaffolds only and do not model mutation-induced relaxation.

For AFDB, the registered consensus's mean within-protein Spearman association between local
geometry and action RMSE is {local_action["estimate"]:+.4f}
[{local_action["ci_low"]:+.4f}, {local_action["ci_high"]:+.4f}] for local RMSD,
{frame_action["estimate"]:+.4f}
[{frame_action["ci_low"]:+.4f}, {frame_action["ci_high"]:+.4f}] for frame error, and
{lddt_action["estimate"]:+.4f}
[{lddt_action["ci_low"]:+.4f}, {lddt_action["ci_high"]:+.4f}] for Cα-lDDT15. These weak,
descriptive local associations are not routing thresholds.

### Teacher input-distribution shift

| Teacher | Structure role | Mean JSD vs experimental | Native-NLL delta |
| --- | --- | ---: | ---: |
{chr(10).join(distribution_lines)}

### Backbone validity diagnostics

| Structure role | Peptide C–N MAE, Å | Omega MAE, degrees | C–N > 2 Å fraction |
| --- | ---: | ---: | ---: |
{chr(10).join(peptide_lines)}

The smooth perturbation translates each residue as a rigid unit, so within-residue bond
geometry is preserved, but inter-residue peptide geometry can move away from the teacher
training distribution. The 1 Å performance loss must therefore remain qualified as a result
under this registered perturbation model, not a universal structural-accuracy threshold.

## References

[^1]: Kuhlman Lab. “ThermoMPNN source repository.” <https://github.com/Kuhlman-Lab/ThermoMPNN>

[^2]: Dieckhaus et al. “ThermoMPNN datasets.” Zenodo. <https://doi.org/10.5281/zenodo.8169288>
"""


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _joined(values: pd.Series) -> str:
    return "|".join(sorted({str(value) for value in values.dropna() if str(value)}))


def _bootstrap(
    frame: pd.DataFrame,
    metric: str,
    specification: Mapping[str, Any],
    seed: int,
) -> dict[str, float | int]:
    selected = frame.copy()
    if "stratum" not in selected:
        selected["stratum"] = "postlock_audit"
    return stratified_domain_bootstrap(
        selected,
        metric,
        replicates=int(specification["inference"]["bootstrap_replicates"]),
        confidence_level=float(specification["inference"]["confidence_level"]),
        seed=seed,
    )


def _empty_summary() -> dict[str, float | int]:
    return {
        "estimate": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "positive_domain_fraction": float("nan"),
        "positive_domains": 0,
        "negative_domains": 0,
        "zero_domains": 0,
        "n_domains": 0,
        "leave_one_domain_out_min": float("nan"),
        "leave_one_domain_out_max": float("nan"),
    }
