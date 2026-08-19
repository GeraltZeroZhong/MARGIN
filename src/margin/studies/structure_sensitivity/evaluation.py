"""Descriptive matched-structure evaluation for structure-sensitivity study."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from margin.attribution.metrics import normalize_log_probabilities
from margin.constants import AA_TO_INDEX
from margin.provenance import read_json, runtime_manifest, write_json, write_parquet
from margin.studies.action_validation.evaluation import _anchor, _ndcg, _spearman
from margin.studies.counterfactuals.evaluation import stratified_domain_bootstrap
from margin.studies.external_validation.panel import load_external_validation_config
from margin.studies.stability.config import load_stability_config
from margin.studies.structure_sensitivity.panel import StructureSensitivityConfig
from margin.teachers.schema import logp_columns


def evaluate_structure_sensitivity(config: StructureSensitivityConfig) -> dict[str, Path | str]:
    """Compare frozen structure roles on the already-open FireProt labels."""

    lock = read_json(config.paths.run_dir / "protocol_lock.json")
    if lock.get("status") != config.status:
        raise RuntimeError("structure-sensitivity study protocol lock is missing")
    score_manifest = config.paths.run_dir / "teacher_scores/manifest.json"
    if not score_manifest.exists():
        raise FileNotFoundError("structure-sensitivity study teacher scores are missing")
    score_metadata = read_json(score_manifest)
    cross = load_external_validation_config(config.paths.external_validation_protocol)
    stability = load_stability_config(config.paths.stability_config)
    if score_metadata.get("teacher_inference_seed") != cross.seed:
        raise RuntimeError(
            "structure-sensitivity study teacher scores do not use the frozen cross-platform seed"
        )
    queries = pd.read_parquet(cross.paths.run_dir / "panel/query_rows.parquet")
    variants = pd.read_parquet(cross.paths.run_dir / "evaluation/variants.parquet")
    scores = pd.read_parquet(config.paths.run_dir / "teacher_scores/scores.parquet")
    structures = pd.read_parquet(config.paths.run_dir / "panel/structures.parquet")
    confidence = pd.read_parquet(config.paths.run_dir / "panel/residue_confidence.parquet")
    temperatures = read_json(stability.paths.run_dir / "calibration/selection.json")[
        "final_parameters"
    ]["temperatures"]
    roles = sorted(structures["structure_role"].unique())
    actions = {
        role: _role_consensus(scores, queries, structures, role, temperatures) for role in roles
    }
    reproduction = _experimental_reproduction(actions["experimental"], queries, cross, stability)
    if float(reproduction["maximum_absolute_action_difference"].max()) > 5e-5:
        raise RuntimeError(
            "structure-sensitivity study experimental action does not reproduce the frozen panel"
        )
    component_path = cross.paths.storage_dir / "strong_control/components.npz"
    frozen = np.load(component_path)
    gc = np.asarray(frozen["temperature_consensus_g"], dtype=float) + np.asarray(
        frozen["temperature_consensus_c_plus"], dtype=float
    )
    variant_scores = _variant_scores(variants, queries, structures, confidence, actions, gc, config)
    domain = _domain_metrics(variant_scores)
    domain = _add_experimental_concordance(domain, variant_scores)
    availability = _role_availability(structures, lock, config)
    role_summary = _role_summary(domain, availability, config)
    paired_delta, delta_summary = _paired_experimental_deltas(domain, availability, config)
    confidence_domain, confidence_summary = _confidence_analysis(
        variant_scores, availability, config
    )
    quality = _quality_correlations(domain, paired_delta, structures, availability)
    decision = _completion_state(role_summary, delta_summary, availability)
    output = config.paths.run_dir / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "reproduction_checks": reproduction,
        "role_availability": availability,
        "variant_scores": variant_scores,
        "domain_metrics": domain,
        "role_summary": role_summary,
        "paired_experimental_deltas": paired_delta,
        "paired_delta_summary": delta_summary,
        "confidence_domain_metrics": confidence_domain,
        "confidence_summary": confidence_summary,
        "quality_correlations": quality,
        "decision": decision,
    }
    paths = {name: output / f"{name}.parquet" for name in tables}
    for name, table in tables.items():
        write_parquet(paths[name], table)
    manifest = output / "manifest.json"
    write_json(
        manifest,
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "status": "STRUCTURE_SENSITIVITY_MATCHED_STRUCTURE_SENSITIVITY_COMPLETE",
            "confirmatory": False,
            "outcomes_were_open_before_protocol": True,
            "routing_evaluated": False,
            "changes_primary_decision": False,
            "teacher_inference_seed": cross.seed,
            "analysis_seed": config.seed,
            "decision": str(decision.iloc[0]["decision"]),
            "tables": {
                name: {"path": str(path), "rows": len(tables[name])} for name, path in paths.items()
            },
        },
    )
    return {**paths, "manifest": manifest, "decision_name": str(decision.iloc[0]["decision"])}


def _role_consensus(
    scores: pd.DataFrame,
    queries: pd.DataFrame,
    structures: pd.DataFrame,
    role: str,
    temperatures: dict[str, float],
) -> pd.DataFrame:
    domains = set(structures.loc[structures["structure_role"].eq(role), "domain_id"])
    metadata = queries.loc[queries["domain_id"].isin(domains)].copy().reset_index(drop=True)
    wild = metadata["wild_type"].map(AA_TO_INDEX).to_numpy(dtype=int)
    teacher_actions = []
    for teacher in ("mif", "esm_if1", "proteinmpnn"):
        selected = scores.loc[scores["teacher_id"].eq(teacher) & scores["structure_role"].eq(role)]
        keys = ["state_id", "domain_id", "position"]
        selected = selected[[*keys, *logp_columns()]]
        if selected.duplicated(keys).any():
            raise ValueError(f"duplicate structure-sensitivity study scores for {teacher}/{role}")
        aligned = metadata[keys].merge(selected, on=keys, validate="one_to_one")
        if len(aligned) != len(metadata):
            raise ValueError(f"incomplete structure-sensitivity study scores for {teacher}/{role}")
        logp = normalize_log_probabilities(aligned[logp_columns()].to_numpy(dtype=float))
        teacher_actions.append(_anchor(logp, wild) / float(temperatures[teacher]))
    consensus = np.mean(np.stack(teacher_actions), axis=0)
    result = metadata[["state_id", "domain_id", "position"]].copy()
    result["action_row"] = list(consensus)
    return result


def _experimental_reproduction(
    structure_sensitivity_action: pd.DataFrame,
    queries: pd.DataFrame,
    cross,
    stability,
) -> pd.DataFrame:
    frozen = np.load(cross.paths.storage_dir / "strong_control/components.npz")
    reference = np.asarray(frozen["temperature_consensus_a"], dtype=float)
    aligned = queries[["state_id", "domain_id", "position"]].merge(
        structure_sensitivity_action,
        on=["state_id", "domain_id", "position"],
        validate="one_to_one",
    )
    observed = np.stack(aligned["action_row"].to_numpy())
    rows = []
    for domain_id, indices_value in queries.groupby("domain_id", sort=True).indices.items():
        indices = np.asarray(indices_value, dtype=int)
        rows.append(
            {
                "domain_id": domain_id,
                "maximum_absolute_action_difference": float(
                    np.max(np.abs(observed[indices] - reference[indices]))
                ),
                "mean_absolute_action_difference": float(
                    np.mean(np.abs(observed[indices] - reference[indices]))
                ),
                "temperature_source": str(stability.paths.run_dir / "calibration/selection.json"),
            }
        )
    return pd.DataFrame(rows)


def _variant_scores(
    variants: pd.DataFrame,
    queries: pd.DataFrame,
    structures: pd.DataFrame,
    confidence: pd.DataFrame,
    actions: dict[str, pd.DataFrame],
    gc: np.ndarray,
    config: StructureSensitivityConfig,
) -> pd.DataFrame:
    global_index = queries[["domain_id", "position"]].copy()
    global_index["global_query_row"] = np.arange(len(global_index), dtype=int)
    rows = []
    for role, action_table in actions.items():
        domains = set(structures.loc[structures["structure_role"].eq(role), "domain_id"])
        action_table = action_table.copy().reset_index(drop=True)
        action_table["role_query_row"] = np.arange(len(action_table), dtype=int)
        action_matrix = np.stack(action_table["action_row"].to_numpy())
        selected = variants.loc[variants["domain_id"].isin(domains)].copy()
        selected = selected.merge(
            action_table[["domain_id", "position", "role_query_row"]],
            on=["domain_id", "position"],
            validate="many_to_one",
        ).merge(global_index, on=["domain_id", "position"], validate="many_to_one")
        mutant = selected["mutant"].map(AA_TO_INDEX).to_numpy(dtype=int)
        role_rows = selected["role_query_row"].to_numpy(dtype=int)
        global_rows = selected["global_query_row"].to_numpy(dtype=int)
        selected["structure_role"] = role
        selected["action_score"] = action_matrix[role_rows, mutant]
        selected["g_plus_c_plus_score"] = gc[global_rows, mutant]
        role_confidence = confidence.loc[
            confidence["structure_role"].eq(role),
            ["domain_id", "position", "confidence"],
        ]
        selected = selected.merge(
            role_confidence, on=["domain_id", "position"], how="left", validate="many_to_one"
        )
        selected["confidence_bin"] = _confidence_bin(selected["confidence"], config)
        rows.append(selected.drop(columns=["role_query_row", "global_query_row"]))
    return pd.concat(rows, ignore_index=True).sort_values(
        ["structure_role", "domain_id", "position", "mutant"], ignore_index=True
    )


def _domain_metrics(variant_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (role, domain_id), frame in variant_scores.groupby(
        ["structure_role", "domain_id"], sort=True, observed=True
    ):
        observed = frame["effect"].to_numpy(dtype=float)
        action = frame["action_score"].to_numpy(dtype=float)
        gc = frame["g_plus_c_plus_score"].to_numpy(dtype=float)
        k = max(1, int(np.ceil(len(frame) * 0.10)))
        action_spearman = _spearman(action, observed)
        gc_spearman = _spearman(gc, observed)
        action_ndcg = _ndcg(action, observed, k=k)
        gc_ndcg = _ndcg(gc, observed, k=k)
        rows.append(
            {
                "structure_role": role,
                "domain_id": domain_id,
                "n_variants": len(frame),
                "action_spearman": action_spearman,
                "g_plus_c_plus_spearman": gc_spearman,
                "spearman_margin": action_spearman - gc_spearman,
                "action_ndcg10": action_ndcg,
                "g_plus_c_plus_ndcg10": gc_ndcg,
                "ndcg10_margin": action_ndcg - gc_ndcg,
                "stratum": "matched_structure",
            }
        )
    return pd.DataFrame(rows)


def _add_experimental_concordance(
    domain: pd.DataFrame, variant_scores: pd.DataFrame
) -> pd.DataFrame:
    experimental = variant_scores.loc[
        variant_scores["structure_role"].eq("experimental"),
        ["domain_id", "position", "wild_type", "mutant", "action_score"],
    ].rename(columns={"action_score": "experimental_action_score"})
    concordance = []
    keys = ["domain_id", "position", "wild_type", "mutant"]
    for (role, domain_id), frame in variant_scores.groupby(
        ["structure_role", "domain_id"], sort=True, observed=True
    ):
        paired = frame.merge(experimental, on=keys, validate="one_to_one")
        concordance.append(
            {
                "structure_role": role,
                "domain_id": domain_id,
                "action_concordance_experimental": _spearman(
                    paired["action_score"].to_numpy(dtype=float),
                    paired["experimental_action_score"].to_numpy(dtype=float),
                ),
                "action_mae_experimental": float(
                    np.mean(
                        np.abs(
                            paired["action_score"].to_numpy(dtype=float)
                            - paired["experimental_action_score"].to_numpy(dtype=float)
                        )
                    )
                ),
            }
        )
    return domain.merge(
        pd.DataFrame(concordance), on=["structure_role", "domain_id"], validate="one_to_one"
    )


def _role_availability(
    structures: pd.DataFrame, lock: dict[str, Any], config: StructureSensitivityConfig
) -> pd.DataFrame:
    counts = structures.groupby("structure_role")["domain_id"].nunique().to_dict()
    predictor_eligible = lock["predictor_summary_eligible"]
    rows = []
    for role in config.inference.comparison_roles:
        is_predictor = role in {"alphafold", "esmfold"}
        rows.append(
            {
                "structure_role": role,
                "domains": int(counts.get(role, 0)),
                "minimum_domains": (
                    config.panel.minimum_matched_domains_per_predictor if is_predictor else 1
                ),
                "summary_eligible": (
                    bool(predictor_eligible.get(role, False)) if is_predictor else True
                ),
                "role_kind": "prediction" if is_predictor else "experimental_or_perturbation",
            }
        )
    return pd.DataFrame(rows)


def _role_summary(
    domain: pd.DataFrame, availability: pd.DataFrame, config: StructureSensitivityConfig
) -> pd.DataFrame:
    eligible = set(availability.loc[availability["summary_eligible"], "structure_role"])
    metrics = [
        "action_spearman",
        "g_plus_c_plus_spearman",
        "spearman_margin",
        "action_ndcg10",
        "g_plus_c_plus_ndcg10",
        "ndcg10_margin",
        "action_concordance_experimental",
        "action_mae_experimental",
    ]
    rows = []
    for role_index, role in enumerate(config.inference.comparison_roles):
        if role not in eligible:
            continue
        frame = domain.loc[domain["structure_role"].eq(role)]
        for metric_index, metric in enumerate(metrics):
            rows.append(
                {
                    "structure_role": role,
                    "metric": metric,
                    **stratified_domain_bootstrap(
                        frame,
                        metric,
                        replicates=config.inference.bootstrap_replicates,
                        confidence_level=config.inference.confidence_level,
                        seed=config.seed + role_index * 100 + metric_index,
                    ),
                }
            )
    return pd.DataFrame(rows)


def _paired_experimental_deltas(
    domain: pd.DataFrame, availability: pd.DataFrame, config: StructureSensitivityConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    experimental = domain.loc[
        domain["structure_role"].eq("experimental"),
        ["domain_id", "action_spearman", "action_ndcg10", "spearman_margin", "ndcg10_margin"],
    ].rename(
        columns={
            "action_spearman": "experimental_action_spearman",
            "action_ndcg10": "experimental_action_ndcg10",
            "spearman_margin": "experimental_spearman_margin",
            "ndcg10_margin": "experimental_ndcg10_margin",
        }
    )
    frames = []
    for role in config.inference.comparison_roles:
        selected = domain.loc[domain["structure_role"].eq(role)].merge(
            experimental, on="domain_id", validate="one_to_one"
        )
        selected["action_spearman_delta_vs_experimental"] = (
            selected["action_spearman"] - selected["experimental_action_spearman"]
        )
        selected["action_ndcg10_delta_vs_experimental"] = (
            selected["action_ndcg10"] - selected["experimental_action_ndcg10"]
        )
        selected["spearman_margin_delta_vs_experimental"] = (
            selected["spearman_margin"] - selected["experimental_spearman_margin"]
        )
        selected["ndcg10_margin_delta_vs_experimental"] = (
            selected["ndcg10_margin"] - selected["experimental_ndcg10_margin"]
        )
        frames.append(selected)
    paired = pd.concat(frames, ignore_index=True)
    eligible = set(availability.loc[availability["summary_eligible"], "structure_role"])
    metrics = [
        "action_spearman_delta_vs_experimental",
        "action_ndcg10_delta_vs_experimental",
        "spearman_margin_delta_vs_experimental",
        "ndcg10_margin_delta_vs_experimental",
    ]
    rows = []
    for role_index, role in enumerate(config.inference.comparison_roles):
        if role not in eligible:
            continue
        frame = paired.loc[paired["structure_role"].eq(role)]
        for metric_index, metric in enumerate(metrics):
            rows.append(
                {
                    "structure_role": role,
                    "metric": metric,
                    **stratified_domain_bootstrap(
                        frame,
                        metric,
                        replicates=config.inference.bootstrap_replicates,
                        confidence_level=config.inference.confidence_level,
                        seed=config.seed + 1000 + role_index * 100 + metric_index,
                    ),
                }
            )
    return paired, pd.DataFrame(rows)


def _confidence_analysis(
    variant_scores: pd.DataFrame,
    availability: pd.DataFrame,
    config: StructureSensitivityConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible_predictors = set(
        availability.loc[
            availability["summary_eligible"] & availability["role_kind"].eq("prediction"),
            "structure_role",
        ]
    )
    rows = []
    for (role, confidence_bin, domain_id), frame in variant_scores.loc[
        variant_scores["structure_role"].isin(eligible_predictors)
        & variant_scores["confidence_bin"].ne("not_applicable")
    ].groupby(["structure_role", "confidence_bin", "domain_id"], sort=True, observed=True):
        observed = frame["effect"].to_numpy(dtype=float)
        action = frame["action_score"].to_numpy(dtype=float)
        gc = frame["g_plus_c_plus_score"].to_numpy(dtype=float)
        action_spearman = _spearman(action, observed)
        gc_spearman = _spearman(gc, observed)
        rows.append(
            {
                "structure_role": role,
                "confidence_bin": confidence_bin,
                "domain_id": domain_id,
                "n_variants": len(frame),
                "action_spearman": action_spearman,
                "g_plus_c_plus_spearman": gc_spearman,
                "spearman_margin": action_spearman - gc_spearman,
                "stratum": "confidence_bin",
            }
        )
    domain = pd.DataFrame(rows)
    if domain.empty:
        return domain, pd.DataFrame()
    summary_rows = []
    for group_index, ((role, confidence_bin), frame) in enumerate(
        domain.groupby(["structure_role", "confidence_bin"], sort=True, observed=True)
    ):
        for metric_index, metric in enumerate(
            ("action_spearman", "g_plus_c_plus_spearman", "spearman_margin")
        ):
            summary_rows.append(
                {
                    "structure_role": role,
                    "confidence_bin": confidence_bin,
                    "metric": metric,
                    "variants": int(frame["n_variants"].sum()),
                    **stratified_domain_bootstrap(
                        frame,
                        metric,
                        replicates=config.inference.bootstrap_replicates,
                        confidence_level=config.inference.confidence_level,
                        seed=config.seed + 2000 + group_index * 100 + metric_index,
                    ),
                }
            )
    return domain, pd.DataFrame(summary_rows)


def _quality_correlations(
    domain: pd.DataFrame,
    paired_delta: pd.DataFrame,
    structures: pd.DataFrame,
    availability: pd.DataFrame,
) -> pd.DataFrame:
    eligible_predictors = availability.loc[
        availability["summary_eligible"] & availability["role_kind"].eq("prediction"),
        "structure_role",
    ]
    rows = []
    for role in eligible_predictors:
        frame = paired_delta.loc[paired_delta["structure_role"].eq(role)].merge(
            structures.loc[
                structures["structure_role"].eq(role),
                [
                    "domain_id",
                    "ca_rmsd_to_experimental",
                    "mean_confidence",
                    "minimum_confidence",
                ],
            ],
            on="domain_id",
            validate="one_to_one",
        )
        for quality in (
            "ca_rmsd_to_experimental",
            "mean_confidence",
            "minimum_confidence",
        ):
            for outcome in (
                "action_concordance_experimental",
                "action_spearman_delta_vs_experimental",
                "spearman_margin",
            ):
                rows.append(
                    {
                        "structure_role": role,
                        "quality_variable": quality,
                        "outcome_variable": outcome,
                        "domains": len(frame),
                        "spearman": _spearman(
                            frame[quality].to_numpy(dtype=float),
                            frame[outcome].to_numpy(dtype=float),
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _completion_state(
    role_summary: pd.DataFrame,
    delta_summary: pd.DataFrame,
    availability: pd.DataFrame,
) -> pd.DataFrame:
    def estimate(table: pd.DataFrame, role: str, metric: str) -> float:
        selected = table.loc[
            table["structure_role"].eq(role) & table["metric"].eq(metric), "estimate"
        ]
        return float(selected.iloc[0]) if len(selected) else float("nan")

    return pd.DataFrame(
        [
            {
                "decision": "STRUCTURE_SENSITIVITY_MATCHED_STRUCTURE_SENSITIVITY_COMPLETE",
                "confirmatory": False,
                "stability_decision_modified": False,
                "routing_authorized": False,
                "alphafold_domains": int(
                    availability.set_index("structure_role").loc["alphafold", "domains"]
                ),
                "esmfold_domains": int(
                    availability.set_index("structure_role").loc["esmfold", "domains"]
                ),
                "esmfold_summary_eligible": bool(
                    availability.set_index("structure_role").loc["esmfold", "summary_eligible"]
                ),
                "experimental_spearman_margin": estimate(
                    role_summary, "experimental", "spearman_margin"
                ),
                "alphafold_spearman_margin": estimate(role_summary, "alphafold", "spearman_margin"),
                "alphafold_action_delta_vs_experimental": estimate(
                    delta_summary, "alphafold", "action_spearman_delta_vs_experimental"
                ),
                "perturbed_0p5_action_delta_vs_experimental": estimate(
                    delta_summary, "perturbed_0p5", "action_spearman_delta_vs_experimental"
                ),
                "perturbed_1p0_action_delta_vs_experimental": estimate(
                    delta_summary, "perturbed_1p0", "action_spearman_delta_vs_experimental"
                ),
            }
        ]
    )


def _confidence_bin(values: pd.Series, config: StructureSensitivityConfig) -> pd.Series:
    bins = np.select(
        [
            values.lt(config.panel.confidence_low_upper),
            values.ge(config.panel.confidence_high_lower),
            values.notna(),
        ],
        ["low", "high", "medium"],
        default="not_applicable",
    )
    return pd.Series(bins, index=values.index)
