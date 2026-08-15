"""Locked stability stability evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from margin.attribution.metrics import normalize_log_probabilities
from margin.constants import AA_TO_INDEX
from margin.provenance import read_json, runtime_manifest, write_json, write_parquet
from margin.studies.action_validation.evaluation import (
    METRICS,
    _aligned_store,
    _aligned_teacher,
    _anchor,
    _domain_metrics,
    _ndcg,
    _spearman,
    _stabilizing_recall,
)
from margin.studies.counterfactuals.evaluation import stratified_domain_bootstrap
from margin.studies.stability.calibration import apply_calibration, select_calibration
from margin.studies.stability.config import StabilityStudyConfig
from margin.studies.stability.prepare import EXTERNAL_POPULATION, PRIMARY_POPULATION

SEQUENCE_METHOD = "esm2_150M_sequence_only"
SELECTED_METHOD = "joint_temperature_consensus"
CPLUS_METHOD = "sequence_plus_G_Cplus"


def evaluate_stability(config: StabilityStudyConfig) -> dict[str, Path | str | bool]:
    """Open the locked outcomes once and evaluate all frozen stability study contrasts."""

    _require_inputs(config)
    output = config.paths.run_dir / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    panel = config.paths.run_dir / "panel"
    domains = pd.read_parquet(panel / "domains.parquet")
    variants = pd.read_parquet(panel / "variants.parquet")
    residues = pd.read_parquet(panel / "residues.parquet")
    queries = pd.read_parquet(panel / "query_rows.parquet")
    sequence_logp = normalize_log_probabilities(
        _aligned_store(
            config.paths.storage_dir / "representations" / "esm2_150M",
            queries,
            "log_probabilities.npy",
        ).astype(float)
    )
    wild = queries["wild_type"].map(AA_TO_INDEX).to_numpy(dtype=int)
    sequence_action = _anchor(sequence_logp, wild)
    scores = pd.read_parquet(config.paths.run_dir / "teacher_scores" / "scores.parquet")
    teacher_actions = {
        teacher: _anchor(_aligned_teacher(scores, queries, teacher), wild)
        for teacher in config.calibration.teacher_ids
    }
    calibration = select_calibration(config)
    scheme_actions = {
        scheme: apply_calibration(
            teacher_actions,
            scheme,
            calibration["final_parameters_by_scheme"][scheme],
            config.calibration.teacher_ids,
        )
        for scheme in config.calibration.schemes
    }
    strong = np.load(
        config.paths.storage_dir / "strong_control" / "panel_strong_control_components.npz"
    )
    components, query_rows, mutants = _variant_components(
        variants,
        domains,
        residues,
        queries,
        sequence_action,
        teacher_actions,
        scheme_actions,
        strong,
    )
    methods, registry = _methods(
        components,
        query_rows,
        mutants,
        sequence_action,
        teacher_actions,
        scheme_actions,
        strong,
    )
    domain_metrics = _domain_metrics(components, methods, config.inference.top_fraction)
    domain_metrics.loc[
        domain_metrics["evaluation_population"].eq(EXTERNAL_POPULATION),
        "stabilizing_top_10_percent_recall",
    ] = np.nan
    contrasts = _contrasts(config)
    domain_contrasts = _domain_contrasts(domain_metrics, contrasts)
    contrast_summary = _contrast_summary(domain_contrasts, components, methods, contrasts, config)
    subgroup = _subgroup_analysis(components, methods, config)
    quality = _quality_controls(components, methods, strong, query_rows, mutants)
    gates, decision = _decision(contrast_summary, quality, config)
    tables = {
        "variant_components": components,
        "method_registry": registry,
        "domain_metrics": domain_metrics,
        "domain_contrasts": domain_contrasts,
        "contrast_summary": contrast_summary,
        "subgroup_domain_metrics": subgroup["domain"],
        "subgroup_summary": subgroup["summary"],
        "quality_controls": quality,
        "gate_checks": gates,
        "project_decision": decision,
    }
    paths = {name: output / f"{name}.parquet" for name in tables}
    for name, table in tables.items():
        write_parquet(paths[name], table)
    manifest_path = output / "manifest.json"
    write_json(
        manifest_path,
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "protocol_lock": str(config.paths.run_dir / "protocol_lock.json"),
            "panel_stability_labels_used_for_training_or_model_selection": False,
            "panel_outcomes_opened_only_in_this_evaluation": True,
            "selected_calibration": calibration["selected_scheme"],
            "primary_population": PRIMARY_POPULATION,
            "external_population": EXTERNAL_POPULATION,
            "external_is_single_protein": True,
            "phenotypes_pooled": False,
            "external_stabilizing_recall": (
                "not_applicable_absolute_T50_has_no_in_table_WT_zero_reference"
            ),
            "routing_evaluated": False,
            "artifacts": {
                name: {
                    "path": str(path),
                    "rows": len(tables[name]),
                    "columns": list(tables[name].columns),
                }
                for name, path in paths.items()
            },
            "paired_action_decision": str(decision.iloc[0]["paired_action_decision"]),
            "selective_routing_decision": str(decision.iloc[0]["selective_routing_decision"]),
            "project_decision": str(decision.iloc[0]["project_decision"]),
        },
    )
    return {
        **paths,
        "manifest": manifest_path,
        "decision": str(decision.iloc[0]["project_decision"]),
        "paired_action_confirmed": bool(decision.iloc[0]["paired_action_confirmed"]),
        "selective_routing_confirmed": bool(decision.iloc[0]["selective_routing_confirmed"]),
    }


def _require_inputs(config: StabilityStudyConfig) -> None:
    lock = read_json(config.paths.run_dir / "protocol_lock.json")
    if lock.get("status") != "FROZEN_BEFORE_STABILITY_PANEL_MODEL_SCORING":
        raise RuntimeError("stability study protocol is not frozen")
    required = [
        config.paths.run_dir / "teacher_scores" / "scores.parquet",
        config.paths.storage_dir / "representations" / "esm2_150M" / "manifest.json",
        config.paths.run_dir / "strong_control" / "strong_control_manifest.json",
        config.paths.storage_dir / "strong_control" / "panel_strong_control_components.npz",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"stability study scoring inputs are missing: {missing}")


def _variant_components(
    variants: pd.DataFrame,
    domains: pd.DataFrame,
    residues: pd.DataFrame,
    queries: pd.DataFrame,
    sequence_action: np.ndarray,
    teacher_actions: dict[str, np.ndarray],
    scheme_actions: dict[str, np.ndarray],
    strong: Any,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    query_index = queries[["domain_id", "position"]].copy()
    query_index["query_row"] = np.arange(len(query_index), dtype=int)
    result = variants.copy().reset_index(drop=True)
    result["variant_row"] = np.arange(len(result), dtype=int)
    result = result.merge(query_index, on=["domain_id", "position"], validate="many_to_one")
    query_rows = result["query_row"].to_numpy(dtype=int)
    mutants = result["mutant"].map(AA_TO_INDEX).to_numpy(dtype=int)
    result["sequence_action"] = sequence_action[query_rows, mutants]
    for teacher, action in teacher_actions.items():
        result[f"{teacher}_action"] = action[query_rows, mutants]
    for scheme, action in scheme_actions.items():
        result[f"{scheme}_action"] = action[query_rows, mutants]
    for name in ("a", "g", "c_plus", "u_plus"):
        result[f"consensus_{name}"] = strong[f"consensus_{name}"][query_rows, mutants]
    residue_columns = [
        "domain_id",
        "position",
        "burial",
        "secondary_structure",
        "contact_class",
        "contact_degree",
        "rsa",
    ]
    available = [column for column in residue_columns if column in residues.columns]
    result = result.merge(residues[available], on=["domain_id", "position"], validate="many_to_one")
    result = result.merge(
        domains[["domain_id", "platform", "structure_kind"]],
        on="domain_id",
        validate="many_to_one",
    )
    result["gly_pro_boundary"] = np.where(
        result["wild_type"].isin(["G", "P"]) | result["mutant"].isin(["G", "P"]),
        "involves_glycine_or_proline",
        "other_substitutions",
    )
    return result.drop(columns="query_row"), query_rows, mutants


def _methods(
    components: pd.DataFrame,
    query_rows: np.ndarray,
    mutants: np.ndarray,
    sequence_action: np.ndarray,
    teacher_actions: dict[str, np.ndarray],
    scheme_actions: dict[str, np.ndarray],
    strong: Any,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    sequence = components["sequence_action"].to_numpy(dtype=float)
    methods = {SEQUENCE_METHOD: sequence}
    rows = [
        {
            "method": SEQUENCE_METHOD,
            "method_family": "sequence",
            "test_time_structure": False,
            "calibration": "not_applicable",
        }
    ]
    for teacher, action in teacher_actions.items():
        method = f"sequence_plus_{teacher}_uncalibrated"
        methods[method] = sequence + action[query_rows, mutants]
        rows.append(
            {
                "method": method,
                "method_family": "single_paired_teacher",
                "test_time_structure": True,
                "calibration": "unscaled",
            }
        )
    scheme_names = {
        "unscaled_equal": "unscaled_consensus",
        "action_rms_matched": "rms_matched_consensus",
        "joint_temperature_native_nll": SELECTED_METHOD,
        "rowwise_rank_normalized": "rank_normalized_consensus",
    }
    for scheme, action in scheme_actions.items():
        method = scheme_names[scheme]
        methods[method] = sequence + action[query_rows, mutants]
        rows.append(
            {
                "method": method,
                "method_family": "paired_teacher_consensus",
                "test_time_structure": True,
                "calibration": scheme,
            }
        )
    methods[CPLUS_METHOD] = (
        sequence
        + strong["consensus_g"][query_rows, mutants]
        + strong["consensus_c_plus"][query_rows, mutants]
    )
    rows.append(
        {
            "method": CPLUS_METHOD,
            "method_family": "strong_sequence_control",
            "test_time_structure": False,
            "calibration": "joint_temperature_native_nll",
        }
    )
    return methods, pd.DataFrame(rows)


def _contrasts(config: StabilityStudyConfig) -> list[dict[str, str]]:
    rows = [
        {
            "contrast": f"{teacher}_vs_sequence",
            "method": f"sequence_plus_{teacher}_uncalibrated",
            "comparator": SEQUENCE_METHOD,
            "role": "individual_teacher_replication",
        }
        for teacher in config.calibration.teacher_ids
    ]
    rows.extend(
        [
            {
                "contrast": "unscaled_consensus_vs_sequence",
                "method": "unscaled_consensus",
                "comparator": SEQUENCE_METHOD,
                "role": "calibration_sensitivity",
            },
            {
                "contrast": "rms_consensus_vs_sequence",
                "method": "rms_matched_consensus",
                "comparator": SEQUENCE_METHOD,
                "role": "calibration_sensitivity",
            },
            {
                "contrast": "selected_consensus_vs_sequence",
                "method": SELECTED_METHOD,
                "comparator": SEQUENCE_METHOD,
                "role": "paired_action_primary",
            },
            {
                "contrast": "rank_consensus_vs_sequence",
                "method": "rank_normalized_consensus",
                "comparator": SEQUENCE_METHOD,
                "role": "calibration_sensitivity",
            },
            {
                "contrast": "selected_consensus_vs_unscaled",
                "method": SELECTED_METHOD,
                "comparator": "unscaled_consensus",
                "role": "calibration_value",
            },
            {
                "contrast": "selected_consensus_vs_Cplus",
                "method": SELECTED_METHOD,
                "comparator": CPLUS_METHOD,
                "role": "selective_routing_primary",
            },
        ]
    )
    return rows


def _domain_contrasts(metrics: pd.DataFrame, contrasts: list[dict[str, str]]) -> pd.DataFrame:
    keys = ["domain_id", "evaluation_population", "stratum", "n_variants"]
    rows = []
    for contrast in contrasts:
        method = metrics.loc[metrics["method"].eq(contrast["method"]), [*keys, *METRICS]]
        comparator = metrics.loc[
            metrics["method"].eq(contrast["comparator"]), [*keys, *METRICS]
        ].rename(columns={metric: f"comparator_{metric}" for metric in METRICS})
        merged = method.merge(comparator, on=keys, validate="one_to_one")
        for key, value in contrast.items():
            merged[key] = value
        for metric in METRICS:
            merged[f"{metric}_margin"] = merged[metric] - merged[f"comparator_{metric}"]
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def _contrast_summary(
    domain: pd.DataFrame,
    components: pd.DataFrame,
    methods: dict[str, np.ndarray],
    contrasts: list[dict[str, str]],
    config: StabilityStudyConfig,
) -> pd.DataFrame:
    rows = []
    primary = domain.loc[domain["evaluation_population"].eq(PRIMARY_POPULATION)]
    for contrast_index, contrast in enumerate(contrasts):
        frame = primary.loc[primary["contrast"].eq(contrast["contrast"])]
        scopes = [("all", frame)]
        scopes.extend(
            (str(stratum), group)
            for stratum, group in frame.groupby("stratum", sort=True, observed=True)
        )
        for scope_index, (stratum, scope) in enumerate(scopes):
            for metric_index, metric in enumerate(METRICS):
                column = f"{metric}_margin"
                rows.append(
                    {
                        **contrast,
                        "evaluation_population": PRIMARY_POPULATION,
                        "stratum": stratum,
                        "metric": metric,
                        "interval_unit": "protein_domain",
                        **stratified_domain_bootstrap(
                            scope,
                            column,
                            replicates=config.inference.bootstrap_replicates,
                            confidence_level=config.inference.confidence_level,
                            seed=(
                                config.seed
                                + contrast_index * 1000
                                + scope_index * 100
                                + metric_index
                            ),
                        ),
                    }
                )
    external_domain = domain.loc[domain["evaluation_population"].eq(EXTERNAL_POPULATION)]
    for contrast in contrasts:
        frame = external_domain.loc[external_domain["contrast"].eq(contrast["contrast"])]
        if len(frame) != 1:
            raise ValueError(f"missing external point contrast: {contrast['contrast']}")
        for metric in METRICS:
            rows.append(
                {
                    **contrast,
                    "evaluation_population": EXTERNAL_POPULATION,
                    "stratum": "external_single_protein",
                    "metric": metric,
                    "interval_unit": "point_only",
                    "estimate": float(frame.iloc[0][f"{metric}_margin"]),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "positive_domain_fraction": float(frame.iloc[0][f"{metric}_margin"] > 0),
                    "positive_domains": int(frame.iloc[0][f"{metric}_margin"] > 0),
                    "negative_domains": int(frame.iloc[0][f"{metric}_margin"] < 0),
                    "zero_domains": int(frame.iloc[0][f"{metric}_margin"] == 0),
                    "n_domains": 1,
                    "leave_one_domain_out_min": float("nan"),
                    "leave_one_domain_out_max": float("nan"),
                }
            )
    external = components.loc[components["evaluation_population"].eq(EXTERNAL_POPULATION)]
    for contrast_index, contrast_name in enumerate(
        ["selected_consensus_vs_sequence", "selected_consensus_vs_Cplus"]
    ):
        contrast = next(row for row in contrasts if row["contrast"] == contrast_name)
        bootstrap = _position_bootstrap(
            external,
            methods[contrast["method"]],
            methods[contrast["comparator"]],
            metric="spearman",
            replicates=config.inference.external_position_bootstrap_replicates,
            confidence_level=config.inference.confidence_level,
            seed=config.seed + 100_000 + contrast_index,
        )
        for row in rows:
            if (
                row["contrast"] == contrast_name
                and row["evaluation_population"] == EXTERNAL_POPULATION
                and row["metric"] == "spearman"
            ):
                row.update(bootstrap)
                row["interval_unit"] = "mutated_position"
                break
    return pd.DataFrame(rows)


def _position_bootstrap(
    frame: pd.DataFrame,
    method: np.ndarray,
    comparator: np.ndarray,
    *,
    metric: str,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float | int]:
    variant_rows = frame["variant_row"].to_numpy(dtype=int)
    observed = frame["effect"].to_numpy(dtype=float)
    method_values = np.asarray(method)[variant_rows]
    comparator_values = np.asarray(comparator)[variant_rows]
    groups = [
        np.asarray(indices, dtype=int)
        for indices in frame.reset_index(drop=True).groupby("position", sort=True).indices.values()
    ]
    estimate = _metric_value(method_values, observed, metric) - _metric_value(
        comparator_values, observed, metric
    )
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=float)
    for repeat in range(replicates):
        sampled_groups = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in sampled_groups])
        bootstrap[repeat] = _metric_value(
            method_values[indices], observed[indices], metric
        ) - _metric_value(comparator_values[indices], observed[indices], metric)
    alpha = (1.0 - confidence_level) / 2.0
    ci_low, ci_high = np.quantile(bootstrap, [alpha, 1.0 - alpha])
    return {
        "estimate": float(estimate),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_positions": len(groups),
    }


def _metric_value(predicted: np.ndarray, observed: np.ndarray, metric: str) -> float:
    if metric == "spearman":
        return _spearman(predicted, observed)
    if metric == "full_ndcg":
        return _ndcg(predicted, observed)
    if metric == "ndcg_at_10_percent":
        k = max(1, int(np.ceil(len(observed) * 0.10)))
        return _ndcg(predicted, observed, k=k)
    if metric == "stabilizing_top_10_percent_recall":
        k = max(1, int(np.ceil(len(observed) * 0.10)))
        return _stabilizing_recall(predicted, observed, k)
    raise ValueError(f"unknown metric: {metric}")


def _subgroup_analysis(
    components: pd.DataFrame,
    methods: dict[str, np.ndarray],
    config: StabilityStudyConfig,
) -> dict[str, pd.DataFrame]:
    rows = []
    selected_methods = [SEQUENCE_METHOD, CPLUS_METHOD, SELECTED_METHOD]
    for dimension in ["burial", "secondary_structure", "contact_class", "gly_pro_boundary"]:
        if dimension not in components:
            continue
        for level, level_frame in components.groupby(dimension, sort=True, observed=True):
            grouped = level_frame.groupby("domain_id", sort=True).indices
            for domain_id, indices_value in grouped.items():
                local = level_frame.iloc[np.asarray(indices_value, dtype=int)]
                indices = local["variant_row"].to_numpy(dtype=int)
                if len(indices) < 10:
                    continue
                observed = components.iloc[indices]["effect"].to_numpy(dtype=float)
                metadata = components.iloc[indices[0]]
                for method in selected_methods:
                    predicted = methods[method][indices]
                    rows.append(
                        {
                            "dimension": dimension,
                            "level": str(level),
                            "domain_id": domain_id,
                            "evaluation_population": metadata.evaluation_population,
                            "stratum": metadata.stratum,
                            "method": method,
                            "n_variants": len(indices),
                            "label_variance": float(np.var(observed)),
                            "spearman": _spearman(predicted, observed),
                            "ndcg_at_10_percent": _metric_value(
                                predicted, observed, "ndcg_at_10_percent"
                            ),
                        }
                    )
    domain = pd.DataFrame(rows)
    summaries = []
    if not domain.empty:
        keys = ["dimension", "level", "evaluation_population", "method"]
        for group_index, (values, frame) in enumerate(
            domain.groupby(keys, sort=True, observed=True)
        ):
            for metric_index, metric in enumerate(["spearman", "ndcg_at_10_percent"]):
                summaries.append(
                    {
                        **dict(zip(keys, values, strict=True)),
                        "metric": metric,
                        **stratified_domain_bootstrap(
                            frame,
                            metric,
                            replicates=config.inference.bootstrap_replicates,
                            confidence_level=config.inference.confidence_level,
                            seed=config.seed + 200_000 + group_index * 10 + metric_index,
                        ),
                    }
                )
    return {"domain": domain, "summary": pd.DataFrame(summaries)}


def _quality_controls(
    components: pd.DataFrame,
    methods: dict[str, np.ndarray],
    strong: Any,
    query_rows: np.ndarray,
    mutants: np.ndarray,
) -> pd.DataFrame:
    identity_error = np.max(
        np.abs(
            strong["consensus_a"]
            - strong["consensus_g"]
            - strong["consensus_c_plus"]
            - strong["consensus_u_plus"]
        )
    )
    full_error = np.max(
        np.abs(
            methods[SELECTED_METHOD]
            - (methods[CPLUS_METHOD] + strong["consensus_u_plus"][query_rows, mutants])
        )
    )
    finite = all(np.isfinite(values).all() for values in methods.values())
    return pd.DataFrame(
        [
            {
                "check": "Cplus_decomposition_identity",
                "estimate": float(identity_error),
                "threshold": 5e-6,
                "passed": bool(identity_error <= 5e-6),
            },
            {
                "check": "full_score_equals_Cplus_plus_Uplus",
                "estimate": float(full_error),
                "threshold": 1e-6,
                "passed": bool(full_error <= 1e-6),
            },
            {
                "check": "all_variant_predictions_finite",
                "estimate": float(finite),
                "threshold": 1.0,
                "passed": bool(finite),
            },
            {
                "check": "variant_row_alignment",
                "estimate": float(
                    np.array_equal(
                        components["variant_row"].to_numpy(dtype=int),
                        np.arange(len(components)),
                    )
                ),
                "threshold": 1.0,
                "passed": bool(
                    np.array_equal(
                        components["variant_row"].to_numpy(dtype=int),
                        np.arange(len(components)),
                    )
                ),
            },
        ]
    )


def _decision(
    summary: pd.DataFrame,
    quality: pd.DataFrame,
    config: StabilityStudyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def get(contrast: str, population: str, stratum: str, metric: str) -> pd.Series:
        selected = summary.loc[
            summary["contrast"].eq(contrast)
            & summary["evaluation_population"].eq(population)
            & summary["stratum"].eq(stratum)
            & summary["metric"].eq(metric)
        ]
        if len(selected) != 1:
            raise ValueError(
                "missing unique stability study summary: "
                f"{contrast}/{population}/{stratum}/{metric}"
            )
        return selected.iloc[0]

    primary_spearman = get("selected_consensus_vs_sequence", PRIMARY_POPULATION, "all", "spearman")
    primary_ndcg = get(
        "selected_consensus_vs_sequence",
        PRIMARY_POPULATION,
        "all",
        "ndcg_at_10_percent",
    )
    natural = get("selected_consensus_vs_sequence", PRIMARY_POPULATION, "natural", "spearman")
    de_novo = get("selected_consensus_vs_sequence", PRIMARY_POPULATION, "de_novo", "spearman")
    external_spearman = get(
        "selected_consensus_vs_sequence",
        EXTERNAL_POPULATION,
        "external_single_protein",
        "spearman",
    )
    external_ndcg = get(
        "selected_consensus_vs_sequence",
        EXTERNAL_POPULATION,
        "external_single_protein",
        "ndcg_at_10_percent",
    )
    cplus = get("selected_consensus_vs_Cplus", PRIMARY_POPULATION, "all", "spearman")
    calibration_value = get("selected_consensus_vs_unscaled", PRIMARY_POPULATION, "all", "spearman")
    teacher_passes = {}
    for teacher in config.calibration.teacher_ids:
        row = get(f"{teacher}_vs_sequence", PRIMARY_POPULATION, "all", "spearman")
        teacher_passes[teacher] = bool(row["ci_low"] > 0)
    checks = [
        {
            "branch": "paired_action",
            "gate": "primary_selected_vs_sequence_spearman_ci_lower_positive",
            "estimate": float(primary_spearman["ci_low"]),
            "threshold": 0.0,
            "passed": bool(primary_spearman["ci_low"] > 0),
        },
        {
            "branch": "paired_action",
            "gate": "primary_selected_vs_sequence_ndcg10_ci_lower_positive",
            "estimate": float(primary_ndcg["ci_low"]),
            "threshold": 0.0,
            "passed": bool(primary_ndcg["ci_low"] > 0),
        },
        {
            "branch": "paired_action",
            "gate": "minimum_uncalibrated_teacher_replications",
            "estimate": float(sum(teacher_passes.values())),
            "threshold": float(config.inference.minimum_teacher_replications),
            "passed": sum(teacher_passes.values()) >= config.inference.minimum_teacher_replications,
        },
        {
            "branch": "paired_action",
            "gate": "natural_spearman_point_positive",
            "estimate": float(natural["estimate"]),
            "threshold": 0.0,
            "passed": bool(natural["estimate"] > 0),
        },
        {
            "branch": "paired_action",
            "gate": "de_novo_spearman_point_positive",
            "estimate": float(de_novo["estimate"]),
            "threshold": 0.0,
            "passed": bool(de_novo["estimate"] > 0),
        },
        {
            "branch": "paired_action",
            "gate": "external_spearman_position_ci_lower_positive",
            "estimate": float(external_spearman["ci_low"]),
            "threshold": 0.0,
            "passed": bool(external_spearman["ci_low"] > 0),
        },
        {
            "branch": "paired_action",
            "gate": "external_ndcg10_point_positive",
            "estimate": float(external_ndcg["estimate"]),
            "threshold": 0.0,
            "passed": bool(external_ndcg["estimate"] > 0),
        },
        {
            "branch": "selective_routing",
            "gate": "primary_full_action_vs_Cplus_spearman_ci_lower_positive",
            "estimate": float(cplus["ci_low"]),
            "threshold": 0.0,
            "passed": bool(cplus["ci_low"] > 0),
        },
    ]
    gates = pd.DataFrame(checks)
    paired_action_gates = gates.loc[gates["branch"].eq("paired_action")]
    paired_action_core = paired_action_gates.iloc[:5]["passed"].all()
    paired_action_all = paired_action_gates["passed"].all()
    calibration_additive = bool(calibration_value["ci_low"] > 0)
    if paired_action_all and calibration_additive:
        paired_action_decision = "PAIRED_ACTION_CALIBRATED_PAIRED_ACTION_CONFIRMED"
    elif paired_action_all:
        paired_action_decision = "PAIRED_ACTION_CONFIRMED_CALIBRATION_NOT_ADDITIVE"
    elif paired_action_core:
        paired_action_decision = "PAIRED_ACTION_IN_DISTRIBUTION_ONLY"
    else:
        paired_action_decision = "PAIRED_ACTION_NOT_CONFIRMED"
    paired_action_confirmed = bool(paired_action_all)
    selective_routing_confirmed = bool(cplus["ci_low"] > 0)
    selective_routing_decision = (
        "STRUCTURE_RETAINED_BEYOND_CPLUS"
        if selective_routing_confirmed
        else "CPLUS_ABSORBS_REGISTERED_RESIDUAL_OR_INCONCLUSIVE"
    )
    quality_passed = bool(quality["passed"].all())
    if paired_action_confirmed and selective_routing_confirmed and quality_passed:
        project_decision = "STABILITY_STRUCTURE_CONDITIONED_ACTION_CONFIRMED_BEYOND_CPLUS"
    elif paired_action_confirmed and quality_passed:
        project_decision = "STABILITY_PAIRED_ACTION_CONFIRMED_CPLUS_BOUNDARY_UNRESOLVED"
    elif paired_action_core and quality_passed:
        project_decision = "STABILITY_PRIMARY_ONLY_EXTERNAL_CONFIRMATION_FAILED"
    else:
        project_decision = "STABILITY_STRUCTURE_CONDITIONED_ACTION_NOT_CONFIRMED"
    project = pd.DataFrame(
        [
            {
                "project_decision": project_decision,
                "paired_action_decision": paired_action_decision,
                "selective_routing_decision": selective_routing_decision,
                "paired_action_confirmed": paired_action_confirmed,
                "selective_routing_confirmed": selective_routing_confirmed,
                "calibration_additive_on_primary_spearman": calibration_additive,
                "quality_controls_passed": quality_passed,
                "teacher_replication_count": int(sum(teacher_passes.values())),
                **{f"{teacher}_replicated": value for teacher, value in teacher_passes.items()},
                "primary_spearman_margin": float(primary_spearman["estimate"]),
                "primary_spearman_ci_low": float(primary_spearman["ci_low"]),
                "primary_ndcg10_margin": float(primary_ndcg["estimate"]),
                "primary_ndcg10_ci_low": float(primary_ndcg["ci_low"]),
                "external_spearman_margin": float(external_spearman["estimate"]),
                "external_spearman_ci_low": float(external_spearman["ci_low"]),
                "external_ndcg10_margin": float(external_ndcg["estimate"]),
                "Cplus_spearman_margin": float(cplus["estimate"]),
                "Cplus_spearman_ci_low": float(cplus["ci_low"]),
                "registered_route": "PIVOT_SELECTIVE_STRUCTURE_CONDITIONED",
                "current_supported_model": "CALIBRATED_PAIRED_STRUCTURE_CONDITIONED",
                "selective_routing": "NOT_ESTABLISHED",
                "sequence_only_residual_transfer": "CLOSED",
                "counterfactual_subtraction": "CLOSED",
                "structure_sensitivity": "DEFERRED_SEPARATE_PROTOCOL",
            }
        ]
    )
    return gates, project
