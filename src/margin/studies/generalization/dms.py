"""Locked external stability-DMS transfer evaluation for generalization study."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import StandardScaler

from margin.constants import AA_TO_INDEX
from margin.provenance import runtime_manifest, table_manifest, write_json, write_parquet
from margin.studies.generalization.audit import load_features
from margin.studies.generalization.config import GeneralizationStudyConfig
from margin.studies.generalization.targets import load_generalization_residual_dataset
from margin.studies.observability.stats import domain_sensitivity_summary


def run_dms_transfer(config: GeneralizationStudyConfig) -> dict[str, object]:
    """Fit only on CATH residuals, then evaluate fixed hybrid scores on locked DMS assays."""

    output = config.paths.run_dir / "dms_transfer"
    output.mkdir(parents=True, exist_ok=True)
    cath = load_generalization_residual_dataset(config)
    cath_features = load_features(
        config.paths.storage_dir / "architecture" / "carp_640M", cath.metadata
    )
    train = np.flatnonzero(
        cath.metadata["observability_split"]
        .isin(["development_train", "development_validation"])
        .to_numpy()
    )
    dms_queries = pd.read_parquet(config.paths.run_dir / "dms" / "query_rows.parquet")
    dms_variants = pd.read_parquet(config.paths.run_dir / "dms" / "variants.parquet")
    dms_features = _aligned_dms_array(
        config.paths.storage_dir / "dms" / "carp_640M", dms_queries, "representations.npy"
    ).astype(np.float32)
    sequence_logp = _aligned_dms_array(
        config.paths.storage_dir / "dms" / "esm2_150M",
        dms_queries,
        "log_probabilities.npy",
    ).astype(float)
    predictions = {
        "consensus_leave_mifst_out": fit_rrr_predict(
            cath_features[train],
            cath.residuals[config.architecture.primary_target][train],
            dms_features,
            rank=config.architecture.rrr_rank,
            alpha=config.architecture.ridge_alpha,
        ),
        "mif_paired_minus_rewired": fit_rrr_predict(
            cath_features[train],
            cath.residuals["mif_paired_minus_rewired"][train],
            dms_features,
            rank=config.architecture.rrr_rank,
            alpha=config.architecture.ridge_alpha,
        ),
    }
    components = _variant_components(
        dms_variants,
        dms_queries,
        sequence_logp,
        predictions,
    )
    assay_metrics = _assay_metrics(components, config)
    increments, summary = _increment_summary(assay_metrics, config)
    decision = _decision(summary, increments, config)
    tables = {
        "variant_components": components,
        "assay_metrics": assay_metrics,
        "assay_increments": increments,
        "increment_summary": summary,
        "decision": decision,
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
            "scope": "fixed_scaffold_stability",
            "training_data": "observability_CATH_native_queries_only",
            "dms_labels_used_for_training": False,
            "primary_alpha": config.dms.primary_alpha,
            "artifacts": [table_manifest(paths[name], table) for name, table in tables.items()],
        },
    )
    return {**paths, "manifest": manifest_path, "passed": bool(decision["passed"].iloc[0])}


def fit_rrr_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    rank: int,
    alpha: float,
) -> np.ndarray:
    """Fit the frozen standardized rank-reduced Ridge and return CLR predictions."""

    scaler = StandardScaler()
    train = scaler.fit_transform(x_train)
    test = scaler.transform(x_test)
    basis = PCA(n_components=min(rank, y_train.shape[1] - 1), svd_solver="full")
    latent = basis.fit_transform(y_train)
    model = Ridge(alpha=alpha, solver="lsqr").fit(train, latent)
    prediction = basis.inverse_transform(model.predict(test))
    return prediction - prediction.mean(axis=1, keepdims=True)


def _aligned_dms_array(directory, queries: pd.DataFrame, filename: str) -> np.ndarray:
    keys = pd.read_parquet(directory / "keys.parquet").copy()
    keys["array_row"] = np.arange(len(keys), dtype=int)
    columns = ["state_id", "domain_id", "position"]
    aligned = queries[columns].merge(keys, on=columns, validate="one_to_one")
    if len(aligned) != len(queries):
        raise ValueError(f"DMS store lacks query coverage: {directory}")
    array = np.load(directory / filename, mmap_mode="r")
    return np.asarray(array[aligned["array_row"].to_numpy(dtype=int)])


def _variant_components(
    variants: pd.DataFrame,
    queries: pd.DataFrame,
    sequence_logp: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    index = queries[["state_id", "position"]].copy()
    index["query_row"] = np.arange(len(index), dtype=int)
    result = variants.merge(
        index.rename(columns={"state_id": "assay_id"}),
        on=["assay_id", "position"],
        validate="many_to_one",
    )
    rows = result["query_row"].to_numpy(dtype=int)
    wild = result["wild_type"].map(AA_TO_INDEX).to_numpy(dtype=int)
    mutant = result["mutant"].map(AA_TO_INDEX).to_numpy(dtype=int)
    result["sequence_effect"] = sequence_logp[rows, mutant] - sequence_logp[rows, wild]
    for target_id, values in predictions.items():
        column = (
            "predicted_residual_effect"
            if target_id == "consensus_leave_mifst_out"
            else "predicted_paired_decoy_effect"
        )
        result[column] = values[rows, mutant] - values[rows, wild]
    return result.drop(columns="query_row")


def _assay_metrics(components: pd.DataFrame, config: GeneralizationStudyConfig) -> pd.DataFrame:
    methods = [("sequence_only", 0.0, "predicted_residual_effect")]
    methods.append(("primary_hybrid", config.dms.primary_alpha, "predicted_residual_effect"))
    methods.extend(
        (f"alpha_{alpha:g}", alpha, "predicted_residual_effect")
        for alpha in config.dms.sensitivity_alphas
    )
    methods.append(
        ("paired_decoy_hybrid", config.dms.primary_alpha, "predicted_paired_decoy_effect")
    )
    rows: list[dict[str, object]] = []
    for assay_id, frame in components.groupby("assay_id", sort=True, observed=True):
        observed = frame["effect"].to_numpy(dtype=float)
        for method, alpha, residual_column in methods:
            predicted = frame["sequence_effect"].to_numpy(dtype=float)
            if method != "sequence_only":
                predicted = predicted + alpha * frame[residual_column].to_numpy(dtype=float)
            rows.append(
                {
                    "assay_id": assay_id,
                    "method": method,
                    "alpha": alpha,
                    "target_component": residual_column,
                    "spearman": _spearman(predicted, observed),
                    "ndcg": _ndcg(predicted, observed),
                    "stabilizing_topk_recall": _stabilizing_topk_recall(
                        predicted, observed, config.dms.top_fraction
                    ),
                    "calibration_slope": _calibration_slope(predicted, observed),
                    "n_variants": int(len(frame)),
                }
            )
    return pd.DataFrame(rows)


def _increment_summary(
    metrics: pd.DataFrame,
    config: GeneralizationStudyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = metrics.loc[
        metrics["method"].eq("sequence_only"),
        [
            "assay_id",
            "spearman",
            "ndcg",
            "stabilizing_topk_recall",
            "calibration_slope",
        ],
    ].rename(columns=lambda name: name if name == "assay_id" else f"baseline_{name}")
    increments = metrics.loc[metrics["method"].ne("sequence_only")].merge(
        baseline, on="assay_id", validate="many_to_one"
    )
    for metric in ("spearman", "ndcg", "stabilizing_topk_recall"):
        increments[f"{metric}_increment"] = increments[metric] - increments[f"baseline_{metric}"]
    increments["calibration_error_reduction"] = (
        increments["baseline_calibration_slope"] - 1.0
    ).abs() - (increments["calibration_slope"] - 1.0).abs()
    summaries: list[pd.DataFrame] = []
    value_columns = [
        "spearman_increment",
        "ndcg_increment",
        "stabilizing_topk_recall_increment",
        "calibration_error_reduction",
    ]
    for group_index, ((method, alpha), frame) in enumerate(
        increments.groupby(["method", "alpha"], observed=True)
    ):
        for metric_index, value_column in enumerate(value_columns):
            input_frame = frame.rename(columns={"assay_id": "domain_id", value_column: "value"})
            _, summary = domain_sensitivity_summary(
                input_frame,
                "value",
                confidence_level=config.inference.confidence_level,
                wild_replicates=config.inference.bootstrap_replicates,
                seed=config.seed + 400_000 + group_index * 100 + metric_index,
            )
            summary["method"] = method
            summary["alpha"] = alpha
            summary["metric"] = value_column
            summaries.append(summary)
    return increments, pd.concat(summaries, ignore_index=True)


def _decision(
    summary: pd.DataFrame,
    increments: pd.DataFrame,
    config: GeneralizationStudyConfig,
) -> pd.DataFrame:
    primary = summary.loc[summary["method"].eq("primary_hybrid")]
    spearman = primary.loc[primary["metric"].eq("spearman_increment")].iloc[0]
    ndcg = primary.loc[primary["metric"].eq("ndcg_increment")].iloc[0]
    assay_values = increments.loc[increments["method"].eq("primary_hybrid")]
    positive_fraction = float((assay_values["spearman_increment"] > 0).mean())
    pass_spearman = bool(
        spearman["estimate"] >= config.dms.minimum_mean_spearman_increment
        and (not config.dms.require_positive_cluster_ci_lower or spearman["wild_ci_low"] > 0)
    )
    pass_positive = positive_fraction >= config.dms.minimum_positive_assay_fraction
    pass_ndcg = bool(not config.dms.require_positive_mean_ndcg_increment or ndcg["estimate"] > 0)
    return pd.DataFrame(
        [
            {
                "decision": (
                    "DMS_TRANSFER_PASS"
                    if pass_spearman and pass_positive and pass_ndcg
                    else "DMS_TRANSFER_FAIL"
                ),
                "mean_spearman_increment": float(spearman["estimate"]),
                "spearman_ci_low": float(spearman["wild_ci_low"]),
                "spearman_ci_high": float(spearman["wild_ci_high"]),
                "positive_assay_fraction": positive_fraction,
                "mean_ndcg_increment": float(ndcg["estimate"]),
                "pass_spearman": pass_spearman,
                "pass_positive_assay_fraction": pass_positive,
                "pass_ndcg": pass_ndcg,
                "passed": pass_spearman and pass_positive and pass_ndcg,
            }
        ]
    )


def _spearman(predicted: np.ndarray, observed: np.ndarray) -> float:
    result = spearmanr(predicted, observed).statistic
    return float(result) if np.isfinite(result) else np.nan


def _ndcg(predicted: np.ndarray, observed: np.ndarray) -> float:
    relevance = observed - np.nanmin(observed)
    if np.allclose(relevance, 0):
        return np.nan
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
        return np.nan
    predicted_top = np.argpartition(predicted, -k)[-k:]
    return float(stabilizing[predicted_top].sum() / min(k, count))


def _calibration_slope(predicted: np.ndarray, observed: np.ndarray) -> float:
    predicted_std = float(np.std(predicted))
    observed_std = float(np.std(observed))
    if predicted_std == 0 or observed_std == 0:
        return np.nan
    x = (predicted - np.mean(predicted)) / predicted_std
    y = (observed - np.mean(observed)) / observed_std
    return float(np.dot(x, y) / np.dot(x, x))
