"""Frozen counterfactual study Route A and Route B stability evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

from margin.attribution.metrics import normalize_log_probabilities
from margin.constants import AA_TO_INDEX
from margin.provenance import (
    runtime_manifest,
    sha256_file,
    table_manifest,
    write_json,
    write_parquet,
)
from margin.studies.counterfactuals.config import CounterfactualStudyConfig
from margin.studies.generalization.audit import load_features
from margin.studies.generalization.config import load_generalization_config
from margin.studies.generalization.dms import fit_rrr_predict
from margin.studies.generalization.targets import load_generalization_residual_dataset
from margin.studies.observability.targets import clr
from margin.teachers.schema import logp_columns

SEQUENCE_METHOD = "esm2_150m_sequence_only"
ROUTE_A_PRIMARY = "direct_mif_paired_minus_contact_rewired_5"
ROUTE_A_REPLICATION = "direct_mif_paired_minus_circular_permuted"
ROUTE_B_PRIMARY = "carp_predicted_mif_paired_minus_contact_rewired_5"
ROUTE_B_REPLICATION = "carp_predicted_mif_paired_minus_circular_permuted"
METRICS = (
    "spearman",
    "ndcg",
    "stabilizing_topk_recall",
    "calibration_slope",
)
GATE_METRICS = (
    "spearman_increment",
    "ndcg_increment",
    "stabilizing_topk_recall_increment",
)


def evaluate_counterfactuals(config: CounterfactualStudyConfig) -> dict[str, Path | bool | str]:
    """Run the fixed label-free predictors and locked domain-level inference."""

    output = config.paths.run_dir / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    panel = config.paths.run_dir / "panel"
    domains = pd.read_parquet(panel / "domains.parquet")
    variants = pd.read_parquet(panel / "variants.parquet")
    residues = pd.read_parquet(panel / "residues.parquet")
    queries = pd.read_parquet(panel / "query_rows.parquet")
    mif_scores = pd.read_parquet(config.paths.run_dir / "mif" / "scores.parquet")
    representation_root = config.paths.storage_dir / "representations"
    sequence_logp = _aligned_store(
        representation_root / "esm2_150M",
        queries,
        "log_probabilities.npy",
    ).astype(float)
    carp_features = _aligned_store(
        representation_root / "carp_640M",
        queries,
        "representations.npy",
    ).astype(np.float32)
    paired = _aligned_scores(
        mif_scores,
        queries,
        "paired",
        population="counterfactuals_locked_panel",
    )
    direct_residuals = {
        role: clr(
            paired
            - _aligned_scores(
                mif_scores,
                queries,
                role,
                population="counterfactuals_locked_panel",
            )
        )
        for role in (
            "contact_rewired_0.5",
            "contact_rewired_1",
            "contact_rewired_2",
            "contact_rewired_5",
            "circular_permuted",
        )
    }
    predictions, training_summary = _fit_route_b(config, mif_scores, queries, carp_features)
    direct_residuals["counterfactual_ensemble"] = np.mean(
        [direct_residuals["contact_rewired_5"], direct_residuals["circular_permuted"]],
        axis=0,
    )
    predictions["counterfactual_ensemble"] = np.mean(
        [predictions["contact_rewired_5"], predictions["circular_permuted"]],
        axis=0,
    )
    methods = _method_columns(direct_residuals, predictions)
    components, variant_query_rows = _variant_components(
        variants,
        domains,
        residues,
        queries,
        sequence_logp,
        direct_residuals,
        predictions,
    )
    domain_metrics = _domain_metrics(components, methods, config.inference.top_fraction)
    increments = _metric_increments(domain_metrics)
    summary = _summarize_increments(increments, config)
    random_tables = _matched_random_control(
        components,
        queries,
        residues,
        sequence_logp,
        direct_residuals["contact_rewired_5"],
        variant_query_rows,
        domain_metrics,
        config,
    )
    decisions, project_decision = _decisions(summary, random_tables["summary"], config)

    tables = {
        "route_b_training": training_summary,
        "variant_components": components,
        "domain_metrics": domain_metrics,
        "domain_increments": increments,
        "increment_summary": summary,
        "random_control_metrics": random_tables["metrics"],
        "random_control_increments": random_tables["increments"],
        "random_control_margins": random_tables["margins"],
        "random_control_summary": random_tables["summary"],
        "route_decisions": decisions,
        "project_decision": project_decision,
    }
    paths: dict[str, Path] = {}
    for name, table in tables.items():
        path = output / f"{name}.parquet"
        write_parquet(path, table)
        paths[name] = path
    matrix_path = output / "residual_matrices.npz"
    np.savez_compressed(
        matrix_path,
        sequence_logp=sequence_logp.astype(np.float32),
        paired_mif_logp=paired.astype(np.float32),
        **{f"direct__{key}": value.astype(np.float32) for key, value in direct_residuals.items()},
        **{f"predicted__{key}": value.astype(np.float32) for key, value in predictions.items()},
    )
    paths["residual_matrices"] = matrix_path
    manifest_path = output / "manifest.json"
    write_json(
        manifest_path,
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "protocol_lock": str(config.paths.run_dir / "protocol_lock.json"),
            "training_data": "generalization_CATH_development_train_and_validation_native_queries",
            "counterfactuals_stability_labels_used_for_predictor_training": False,
            "route_a_test_time_structure": True,
            "route_b_test_time_structure": False,
            "residual_alpha": config.models.residual_alpha,
            "artifacts": [table_manifest(paths[name], table) for name, table in tables.items()],
            "residual_matrices": {
                "path": str(matrix_path),
                "sha256": sha256_file(matrix_path),
            },
            "final_decision": str(project_decision["decision"].iloc[0]),
        },
    )
    paths["manifest"] = manifest_path
    return {
        **paths,
        "route_a_passed": bool(
            decisions.loc[decisions["route"].eq("route_a_direct_structure"), "passed"].iloc[0]
        ),
        "route_b_passed": bool(
            decisions.loc[decisions["route"].eq("route_b_sequence_predicted"), "passed"].iloc[0]
        ),
        "decision": str(project_decision["decision"].iloc[0]),
    }


def stratified_domain_bootstrap(
    frame: pd.DataFrame,
    value_column: str,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float | int]:
    """Equal-domain mean with a percentile bootstrap preserving stratum counts."""

    clean = (
        frame[["domain_id", "stratum", value_column]].replace([np.inf, -np.inf], np.nan).dropna()
    )
    domains = (
        clean.groupby(["domain_id", "stratum"], observed=True)[value_column].mean().reset_index()
    )
    if domains.empty:
        return _empty_summary()
    values = domains[value_column].to_numpy(dtype=float)
    estimate = float(values.mean())
    rng = np.random.default_rng(seed)
    total = np.zeros(replicates, dtype=float)
    total_domains = 0
    for _, group in domains.groupby("stratum", sort=True, observed=True):
        group_values = group[value_column].to_numpy(dtype=float)
        sampled = group_values[
            rng.integers(0, len(group_values), size=(replicates, len(group_values)))
        ]
        total += sampled.sum(axis=1)
        total_domains += len(group_values)
    bootstrap = total / total_domains
    alpha = (1.0 - confidence_level) / 2.0
    ci_low, ci_high = np.quantile(bootstrap, [alpha, 1.0 - alpha])
    if len(values) > 1:
        leave_one_out = np.asarray(
            [np.delete(values, index).mean() for index in range(len(values))]
        )
        leave_min = float(leave_one_out.min())
        leave_max = float(leave_one_out.max())
    else:
        leave_min = leave_max = float("nan")
    return {
        "estimate": estimate,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "positive_domain_fraction": float((values > 0).mean()),
        "positive_domains": int((values > 0).sum()),
        "negative_domains": int((values < 0).sum()),
        "zero_domains": int((values == 0).sum()),
        "n_domains": int(len(values)),
        "leave_one_domain_out_min": leave_min,
        "leave_one_domain_out_max": leave_max,
    }


def _fit_route_b(
    config: CounterfactualStudyConfig,
    counterfactuals_mif: pd.DataFrame,
    queries: pd.DataFrame,
    carp_features: np.ndarray,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    generalization = load_generalization_config(config.paths.generalization_config)
    cath = load_generalization_residual_dataset(generalization)
    cath_features = load_features(
        generalization.paths.storage_dir / "architecture" / "carp_640M",
        cath.metadata,
    )
    train = np.flatnonzero(
        cath.metadata["observability_split"]
        .isin(["development_train", "development_validation"])
        .to_numpy()
    )
    generalization_mif = pd.read_parquet(generalization.paths.run_dir / "mif" / "scores.parquet")
    cath_paired = _aligned_scores(generalization_mif, cath.metadata, "paired")
    cath_circular = _aligned_scores(
        counterfactuals_mif,
        cath.metadata,
        "circular_permuted",
        population="generalization_cath_training",
    )
    targets = {
        "contact_rewired_5": cath.residuals["mif_paired_minus_rewired"],
        "circular_permuted": clr(cath_paired - cath_circular),
    }
    predictions = {
        target_id: fit_rrr_predict(
            cath_features[train],
            target[train],
            carp_features,
            rank=config.models.rrr_rank,
            alpha=config.models.ridge_alpha,
        )
        for target_id, target in targets.items()
    }
    summary = pd.DataFrame(
        [
            {
                "target_id": target_id,
                "training_rows": int(len(train)),
                "training_domains": int(cath.metadata.iloc[train]["domain_id"].nunique()),
                "excluded_cath_locked_test_rows": int(len(cath.metadata) - len(train)),
                "rrr_rank": config.models.rrr_rank,
                "ridge_alpha": config.models.ridge_alpha,
                "counterfactuals_stability_labels_used": False,
                "target_rms": float(np.sqrt(np.mean(target[train] ** 2))),
                "prediction_rms": float(np.sqrt(np.mean(predictions[target_id] ** 2))),
            }
            for target_id, target in targets.items()
        ]
    )
    return predictions, summary


def _method_columns(
    direct: dict[str, np.ndarray],
    predicted: dict[str, np.ndarray],
) -> dict[str, str | None]:
    methods: dict[str, str | None] = {SEQUENCE_METHOD: None}
    for role in direct:
        methods[f"direct_mif_paired_minus_{role}"] = f"direct_{role}_effect"
    for role in predicted:
        methods[f"carp_predicted_mif_paired_minus_{role}"] = f"predicted_{role}_effect"
    return methods


def _variant_components(
    variants: pd.DataFrame,
    domains: pd.DataFrame,
    residues: pd.DataFrame,
    queries: pd.DataFrame,
    sequence_logp: np.ndarray,
    direct: dict[str, np.ndarray],
    predicted: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, np.ndarray]:
    query_index = queries[["state_id", "domain_id", "position"]].copy()
    query_index["query_row"] = np.arange(len(query_index), dtype=int)
    result = variants.merge(
        query_index[["domain_id", "position", "query_row"]],
        on=["domain_id", "position"],
        validate="many_to_one",
    )
    result = result.merge(
        residues[
            [
                "domain_id",
                "position",
                "burial",
                "secondary_structure",
                "contact_class",
                "rsa",
            ]
        ],
        on=["domain_id", "position"],
        validate="many_to_one",
    ).merge(
        domains[["domain_id", "length", "design_family", "platform"]],
        on="domain_id",
        validate="many_to_one",
    )
    rows = result["query_row"].to_numpy(dtype=int)
    wild = result["wild_type"].map(AA_TO_INDEX).to_numpy(dtype=int)
    mutant = result["mutant"].map(AA_TO_INDEX).to_numpy(dtype=int)
    result["sequence_effect"] = sequence_logp[rows, mutant] - sequence_logp[rows, wild]
    for role, matrix in direct.items():
        result[f"direct_{role}_effect"] = matrix[rows, mutant] - matrix[rows, wild]
    for role, matrix in predicted.items():
        result[f"predicted_{role}_effect"] = matrix[rows, mutant] - matrix[rows, wild]
    result["substitution_class"] = [
        _substitution_class(wild_type, mutant_type)
        for wild_type, mutant_type in zip(result["wild_type"], result["mutant"], strict=True)
    ]
    result["length_class"] = pd.cut(
        result["length"],
        bins=[-np.inf, 39, 72, np.inf],
        labels=["shorter_than_40", "megascale_range_40_72", "longer_than_72"],
    ).astype(str)
    return result.drop(columns="query_row"), rows


def _domain_metrics(
    components: pd.DataFrame,
    methods: dict[str, str | None],
    top_fraction: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for domain_id, frame in components.groupby("domain_id", sort=True, observed=True):
        observed = frame["effect"].to_numpy(dtype=float)
        sequence = frame["sequence_effect"].to_numpy(dtype=float)
        metadata = {
            "domain_id": domain_id,
            "stratum": str(frame["stratum"].iloc[0]),
            "source": str(frame["source"].iloc[0]),
            "length": int(frame["length"].iloc[0]),
            "n_variants": int(len(frame)),
        }
        for method, residual_column in methods.items():
            predicted = sequence.copy()
            if residual_column is not None:
                predicted += frame[residual_column].to_numpy(dtype=float)
            rows.append(
                {
                    **metadata,
                    "method": method,
                    "spearman": _spearman(predicted, observed),
                    "ndcg": _ndcg(predicted, observed),
                    "stabilizing_topk_recall": _stabilizing_topk_recall(
                        predicted, observed, top_fraction
                    ),
                    "calibration_slope": _calibration_slope(predicted, observed),
                }
            )
    return pd.DataFrame(rows)


def _metric_increments(metrics: pd.DataFrame) -> pd.DataFrame:
    baseline = metrics.loc[metrics["method"].eq(SEQUENCE_METHOD)].drop(columns="method")
    rename = {metric: f"baseline_{metric}" for metric in METRICS}
    baseline = baseline.rename(columns=rename)
    keys = ["domain_id", "stratum", "source", "length", "n_variants"]
    increments = metrics.loc[~metrics["method"].eq(SEQUENCE_METHOD)].merge(
        baseline[[*keys, *rename.values()]],
        on=keys,
        validate="many_to_one",
    )
    for metric in ("spearman", "ndcg", "stabilizing_topk_recall"):
        increments[f"{metric}_increment"] = increments[metric] - increments[f"baseline_{metric}"]
    increments["calibration_error_reduction"] = (
        increments["baseline_calibration_slope"] - 1.0
    ).abs() - (increments["calibration_slope"] - 1.0).abs()
    return increments


def _summarize_increments(
    increments: pd.DataFrame,
    config: CounterfactualStudyConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    value_columns = [*GATE_METRICS, "calibration_error_reduction"]
    for method_index, (method, frame) in enumerate(
        increments.groupby("method", sort=True, observed=True)
    ):
        for metric_index, value_column in enumerate(value_columns):
            estimate = stratified_domain_bootstrap(
                frame,
                value_column,
                replicates=config.inference.bootstrap_replicates,
                confidence_level=config.inference.confidence_level,
                seed=config.seed + 100_000 + method_index * 100 + metric_index,
            )
            rows.append(
                {
                    "method": method,
                    "metric": value_column,
                    "scope": "all_stratum_preserving",
                    "stratum": "all",
                    **estimate,
                }
            )
            for stratum_index, (stratum, group) in enumerate(
                frame.groupby("stratum", sort=True, observed=True)
            ):
                estimate = stratified_domain_bootstrap(
                    group,
                    value_column,
                    replicates=config.inference.bootstrap_replicates,
                    confidence_level=config.inference.confidence_level,
                    seed=(
                        config.seed
                        + 200_000
                        + method_index * 1_000
                        + metric_index * 10
                        + stratum_index
                    ),
                )
                rows.append(
                    {
                        "method": method,
                        "metric": value_column,
                        "scope": "single_stratum",
                        "stratum": str(stratum),
                        **estimate,
                    }
                )
    return pd.DataFrame(rows)


def _matched_random_control(
    components: pd.DataFrame,
    queries: pd.DataFrame,
    residues: pd.DataFrame,
    sequence_logp: np.ndarray,
    residual: np.ndarray,
    variant_query_rows: np.ndarray,
    observed_metrics: pd.DataFrame,
    config: CounterfactualStudyConfig,
) -> dict[str, pd.DataFrame]:
    query_environment = queries[["domain_id", "position"]].copy()
    query_environment["query_row"] = np.arange(len(query_environment), dtype=int)
    query_environment = query_environment.merge(
        residues[["domain_id", "position", "burial"]],
        on=["domain_id", "position"],
        validate="one_to_one",
    )
    wild = components["wild_type"].map(AA_TO_INDEX).to_numpy(dtype=int)
    mutant = components["mutant"].map(AA_TO_INDEX).to_numpy(dtype=int)
    random_metrics: list[pd.DataFrame] = []
    for repeat in range(config.counterfactuals.matched_random_repeats):
        rng = np.random.default_rng(config.seed + 500_000 + repeat)
        shuffled = residual.copy()
        for _, group in query_environment.groupby(
            ["domain_id", "burial"], sort=True, observed=True
        ):
            indices = group["query_row"].to_numpy(dtype=int)
            shuffled[indices] = residual[rng.permutation(indices)]
        random_components = components[
            ["domain_id", "effect", "stratum", "source", "length", "sequence_effect"]
        ].copy()
        random_components["matched_random_effect"] = (
            shuffled[variant_query_rows, mutant] - shuffled[variant_query_rows, wild]
        )
        frame = _domain_metrics(
            random_components,
            {"matched_random_control": "matched_random_effect"},
            config.inference.top_fraction,
        )
        frame["repeat"] = repeat
        random_metrics.append(frame)
    metrics = pd.concat(random_metrics, ignore_index=True)
    baseline = observed_metrics.loc[observed_metrics["method"].eq(SEQUENCE_METHOD)]
    baseline = baseline.rename(columns={metric: f"baseline_{metric}" for metric in METRICS})
    keys = ["domain_id", "stratum", "source", "length", "n_variants"]
    increments = metrics.merge(
        baseline[[*keys, *[f"baseline_{metric}" for metric in METRICS]]],
        on=keys,
        validate="many_to_one",
    )
    for metric in ("spearman", "ndcg", "stabilizing_topk_recall"):
        increments[f"{metric}_increment"] = increments[metric] - increments[f"baseline_{metric}"]
    increments["calibration_error_reduction"] = (
        increments["baseline_calibration_slope"] - 1.0
    ).abs() - (increments["calibration_slope"] - 1.0).abs()
    random_mean = (
        increments.groupby([*keys], observed=True)[[*GATE_METRICS, "calibration_error_reduction"]]
        .mean()
        .reset_index()
        .rename(
            columns={
                column: f"random_mean_{column}"
                for column in [*GATE_METRICS, "calibration_error_reduction"]
            }
        )
    )
    direct = _metric_increments(observed_metrics).loc[
        lambda frame: frame["method"].eq(ROUTE_A_PRIMARY)
    ]
    margins = direct.merge(random_mean, on=keys, validate="one_to_one")
    for metric in [*GATE_METRICS, "calibration_error_reduction"]:
        margins[f"{metric}_margin"] = margins[metric] - margins[f"random_mean_{metric}"]
    summary_rows = []
    for metric_index, metric in enumerate([*GATE_METRICS, "calibration_error_reduction"]):
        column = f"{metric}_margin"
        summary_rows.append(
            {
                "method": "direct_primary_minus_matched_random_control",
                "metric": column,
                **stratified_domain_bootstrap(
                    margins,
                    column,
                    replicates=config.inference.bootstrap_replicates,
                    confidence_level=config.inference.confidence_level,
                    seed=config.seed + 600_000 + metric_index,
                ),
            }
        )
    return {
        "metrics": metrics,
        "increments": increments,
        "margins": margins,
        "summary": pd.DataFrame(summary_rows),
    }


def _decisions(
    summary: pd.DataFrame,
    random_summary: pd.DataFrame,
    config: CounterfactualStudyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def row(method: str, metric: str, stratum: str = "all") -> pd.Series:
        selected = summary.loc[
            summary["method"].eq(method)
            & summary["metric"].eq(metric)
            & summary["stratum"].eq(stratum)
        ]
        if len(selected) != 1:
            raise ValueError(f"missing unique summary row for {method}/{metric}/{stratum}")
        return selected.iloc[0]

    control = random_summary.loc[random_summary["metric"].eq("spearman_increment_margin")].iloc[0]
    route_rows = []
    for route, primary, replication, require_control in (
        ("route_a_direct_structure", ROUTE_A_PRIMARY, ROUTE_A_REPLICATION, True),
        ("route_b_sequence_predicted", ROUTE_B_PRIMARY, ROUTE_B_REPLICATION, False),
    ):
        spearman = row(primary, "spearman_increment")
        ndcg = row(primary, "ndcg_increment")
        topk = row(primary, "stabilizing_topk_recall_increment")
        replication_values = {metric: row(replication, metric) for metric in GATE_METRICS}
        natural = row(primary, "spearman_increment", "natural")
        de_novo = row(primary, "spearman_increment", "de_novo")
        checks = {
            "pass_spearman_ci": bool(spearman["ci_low"] > 0),
            "pass_ndcg_ci": bool(ndcg["ci_low"] > 0),
            "pass_topk_point": bool(topk["estimate"] > 0),
            "pass_majority_domains": bool(
                spearman["positive_domain_fraction"]
                >= config.inference.minimum_positive_domain_fraction
            ),
            "pass_natural_point": bool(natural["estimate"] > 0),
            "pass_de_novo_point": bool(de_novo["estimate"] > 0),
            "pass_replication_spearman": bool(
                replication_values["spearman_increment"]["estimate"] > 0
            ),
            "pass_replication_ndcg": bool(replication_values["ndcg_increment"]["estimate"] > 0),
            "pass_replication_topk": bool(
                replication_values["stabilizing_topk_recall_increment"]["estimate"] > 0
            ),
            "pass_random_control_margin": bool(control["ci_low"] > 0) if require_control else True,
        }
        route_rows.append(
            {
                "route": route,
                "primary_method": primary,
                "replication_method": replication,
                "spearman_increment": float(spearman["estimate"]),
                "spearman_ci_low": float(spearman["ci_low"]),
                "spearman_ci_high": float(spearman["ci_high"]),
                "ndcg_increment": float(ndcg["estimate"]),
                "ndcg_ci_low": float(ndcg["ci_low"]),
                "ndcg_ci_high": float(ndcg["ci_high"]),
                "topk_increment": float(topk["estimate"]),
                "positive_domain_fraction": float(spearman["positive_domain_fraction"]),
                "natural_spearman_increment": float(natural["estimate"]),
                "de_novo_spearman_increment": float(de_novo["estimate"]),
                "random_control_spearman_margin": float(control["estimate"])
                if require_control
                else np.nan,
                "random_control_margin_ci_low": float(control["ci_low"])
                if require_control
                else np.nan,
                **checks,
                "passed": all(checks.values()),
            }
        )
    decisions = pd.DataFrame(route_rows)
    route_a = bool(
        decisions.loc[decisions["route"].eq("route_a_direct_structure"), "passed"].iloc[0]
    )
    route_b = bool(
        decisions.loc[decisions["route"].eq("route_b_sequence_predicted"), "passed"].iloc[0]
    )
    if route_a and route_b:
        decision = "OPEN_SELECTIVE_ROUTING_COUNTERFACTUAL_GEOMETRY_RESIDUAL_TRANSFER"
    elif route_a:
        decision = "RETAIN_STRUCTURE_CONDITIONED_CSAR_CLOSE_SEQUENCE_ONLY"
    else:
        decision = "RETAIN_GENERALIZATION_CLOSE_COUNTERFACTUALS"
    project = pd.DataFrame(
        [
            {
                "decision": decision,
                "route_a_passed": route_a,
                "route_b_passed": route_b,
                "historical_generalization_decision_modified": False,
            }
        ]
    )
    return decisions, project


def _aligned_store(directory: Path, queries: pd.DataFrame, filename: str) -> np.ndarray:
    keys = pd.read_parquet(directory / "keys.parquet").copy()
    keys["array_row"] = np.arange(len(keys), dtype=int)
    columns = ["state_id", "domain_id", "position"]
    aligned = queries[columns].merge(keys, on=columns, validate="one_to_one")
    if len(aligned) != len(queries):
        raise ValueError(
            f"representation store lacks counterfactual study query coverage: {directory}"
        )
    array = np.load(directory / filename, mmap_mode="r")
    result = np.asarray(array[aligned["array_row"].to_numpy(dtype=int)])
    if not np.isfinite(result).all():
        raise ValueError(f"representation store contains non-finite values: {directory}")
    return result


def _aligned_scores(
    scores: pd.DataFrame,
    metadata: pd.DataFrame,
    role: str,
    *,
    population: str | None = None,
) -> np.ndarray:
    selected = scores.loc[scores["structure_role"].eq(role)].copy()
    if population is not None:
        selected = selected.loc[selected["analysis_population"].eq(population)]
    keys = ["state_id", "domain_id", "position"]
    selected = selected[[*keys, *logp_columns()]]
    if selected.duplicated(keys).any():
        raise ValueError(f"duplicate MIF rows for {role}/{population}")
    aligned = metadata[keys].merge(selected, on=keys, validate="one_to_one")
    if len(aligned) != len(metadata):
        raise ValueError(
            f"MIF scores lack coverage for {role}/{population}: {len(aligned)}/{len(metadata)}"
        )
    return normalize_log_probabilities(aligned[logp_columns()].to_numpy(dtype=float))


def _substitution_class(wild_type: str, mutant: str) -> str:
    positive = {"K", "R"}
    negative = {"D", "E"}
    hydrophobic = {"A", "V", "I", "L", "M", "F", "W", "Y"}
    if wild_type in {"G", "P"} or mutant in {"G", "P"}:
        return "involves_glycine_or_proline"
    if (wild_type in positive and mutant in negative) or (
        wild_type in negative and mutant in positive
    ):
        return "charge_reversal"
    if wild_type in hydrophobic and mutant in hydrophobic:
        return "within_hydrophobic"
    if wild_type in hydrophobic and mutant not in hydrophobic:
        return "hydrophobic_to_nonhydrophobic"
    if wild_type not in hydrophobic and mutant in hydrophobic:
        return "nonhydrophobic_to_hydrophobic"
    return "other_polar_or_charged"


def _spearman(predicted: np.ndarray, observed: np.ndarray) -> float:
    if len(predicted) < 2 or np.ptp(predicted) == 0 or np.ptp(observed) == 0:
        return float("nan")
    value = spearmanr(predicted, observed).statistic
    return float(value) if np.isfinite(value) else float("nan")


def _ndcg(predicted: np.ndarray, observed: np.ndarray) -> float:
    relevance = observed - np.min(observed)
    if np.allclose(relevance, 0):
        return float("nan")
    return float(ndcg_score(relevance[None, :], predicted[None, :]))


def _stabilizing_topk_recall(
    predicted: np.ndarray,
    observed: np.ndarray,
    fraction: float,
) -> float:
    k = max(1, int(np.ceil(len(predicted) * fraction)))
    stabilizing = observed > 0
    count = int(stabilizing.sum())
    if count == 0:
        return float("nan")
    predicted_top = np.argpartition(predicted, -k)[-k:]
    return float(stabilizing[predicted_top].sum() / min(k, count))


def _calibration_slope(predicted: np.ndarray, observed: np.ndarray) -> float:
    centered = predicted - np.mean(predicted)
    denominator = float(np.dot(centered, centered))
    if denominator == 0:
        return float("nan")
    return float(np.dot(centered, observed - np.mean(observed)) / denominator)


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
