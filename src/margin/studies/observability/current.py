"""Exploratory observability study sensitivity audit over the frozen foundation audit artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from margin.config import load_config
from margin.provenance import (
    runtime_manifest,
    table_manifest,
    write_json,
    write_parquet,
)
from margin.studies.observability.config import ObservabilityStudyConfig
from margin.studies.observability.probes import (
    intercept_predict,
    load_aligned_parquet_features,
    prediction_metrics,
    ridge_predict,
    shuffled_target,
)
from margin.studies.observability.stats import domain_sensitivity_summary
from margin.studies.observability.targets import (
    load_foundation_residual_dataset,
    load_replication_residual_dataset,
)

METRICS = (
    "jsd_reduction_nats",
    "cross_entropy_reduction_nats",
    "residual_cosine",
    "candidate_rank_agreement",
    "top3_overlap",
)


def run_current_sensitivity(config: ObservabilityStudyConfig) -> dict[str, Path]:
    """Run the registered current-data probes without modifying foundation decision artifacts."""

    output = config.paths.run_dir / "current_sensitivity"
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_foundation_residual_dataset(config)
    feature_path = (
        config.paths.project_root / "data/interim/student/state_position_embeddings.parquet"
    )
    features = load_aligned_parquet_features(feature_path, dataset.metadata)
    train = np.flatnonzero(dataset.metadata["eligible_for_training"].astype(bool).to_numpy())
    test = np.flatnonzero(~dataset.metadata["eligible_for_training"].astype(bool).to_numpy())
    rows: list[pd.DataFrame] = []

    for target_id in dataset.residuals:
        prediction = ridge_predict(
            features[train],
            dataset.residuals[target_id][train],
            features[test],
            alpha=config.probes.ridge_alpha,
        )
        rows.append(
            prediction_metrics(
                dataset,
                target_id,
                test,
                prediction,
                probe="final_layer_ridge",
                control="observed",
                repeat=0,
                moved_fraction=1.0,
            )
        )
    primary = config.residual_targets.primary
    rows.extend(_intercept_rows(dataset, primary, train, test))
    rng = np.random.default_rng(config.seed + 9100)
    for control in config.probes.shuffle_controls:
        for repeat in range(config.probes.control_repeats):
            shuffled, moved = shuffled_target(
                dataset.metadata,
                dataset.residuals[primary],
                train,
                control,
                rng,
            )
            prediction = ridge_predict(
                features[train],
                shuffled,
                features[test],
                alpha=config.probes.ridge_alpha,
            )
            rows.append(
                prediction_metrics(
                    dataset,
                    primary,
                    test,
                    prediction,
                    probe="final_layer_ridge",
                    control=control,
                    repeat=repeat,
                    moved_fraction=moved,
                )
            )

    metrics = pd.concat(rows, ignore_index=True)
    metric_path = output / "probe_rows.parquet"
    write_parquet(metric_path, metrics)
    summary, domains = summarize_probe_rows(metrics, config)
    summary_path = output / "probe_summary.parquet"
    domain_path = output / "domain_estimates.parquet"
    write_parquet(summary_path, summary)
    write_parquet(domain_path, domains)
    temperature_path = output / "teacher_temperatures.parquet"
    write_parquet(temperature_path, dataset.temperatures)
    manifest_path = output / "manifest.json"
    write_json(
        manifest_path,
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "status": "exploratory_follow_up",
            "historical_gate_modified": False,
            "feature_path": str(feature_path),
            "artifacts": [
                table_manifest(metric_path, metrics),
                table_manifest(summary_path, summary),
                table_manifest(domain_path, domains),
                table_manifest(temperature_path, dataset.temperatures),
            ],
        },
    )
    return {
        "rows": metric_path,
        "summary": summary_path,
        "domains": domain_path,
        "temperatures": temperature_path,
        "manifest": manifest_path,
    }


def run_replication_final_layer(config: ObservabilityStudyConfig) -> dict[str, Path]:
    """Evaluate the fixed final ESM2 layer once on the untouched replication test."""

    output = config.paths.run_dir / "replication_final_layer"
    output.mkdir(parents=True, exist_ok=True)
    replication_config = load_config(config.paths.replication_config)
    dataset = load_replication_residual_dataset(config, replication_config.paths.run_dir)
    if replication_config.paths.embeddings_input is None:
        raise ValueError("replication final-layer audit requires paths.embeddings_input")
    features = load_aligned_parquet_features(
        replication_config.paths.embeddings_input, dataset.metadata
    )
    role = dataset.metadata["analysis_role"]
    train = np.flatnonzero(role.isin(["development_train", "development_validation"]).to_numpy())
    test = np.flatnonzero(role.eq("locked_test").to_numpy())
    rows: list[pd.DataFrame] = []
    for target_id in dataset.residuals:
        prediction = ridge_predict(
            features[train],
            dataset.residuals[target_id][train],
            features[test],
            alpha=config.probes.ridge_alpha,
        )
        rows.append(
            prediction_metrics(
                dataset,
                target_id,
                test,
                prediction,
                probe="fixed_final_layer_ridge",
                evaluation_split="locked_test",
                control="observed",
                repeat=0,
                moved_fraction=1.0,
            )
        )
    primary = config.residual_targets.primary
    intercept_rows = _intercept_rows(dataset, primary, train, test)
    for frame in intercept_rows:
        frame["evaluation_split"] = "locked_test"
    rows.extend(intercept_rows)
    rng = np.random.default_rng(config.seed + 9300)
    for control in config.probes.shuffle_controls:
        for repeat in range(config.probes.control_repeats):
            shuffled, moved = shuffled_target(
                dataset.metadata,
                dataset.residuals[primary],
                train,
                control,
                rng,
            )
            prediction = ridge_predict(
                features[train],
                shuffled,
                features[test],
                alpha=config.probes.ridge_alpha,
            )
            rows.append(
                prediction_metrics(
                    dataset,
                    primary,
                    test,
                    prediction,
                    probe="fixed_final_layer_ridge",
                    evaluation_split="locked_test",
                    control=control,
                    repeat=repeat,
                    moved_fraction=moved,
                )
            )
    metrics = pd.concat(rows, ignore_index=True)
    summary, domains = summarize_probe_rows(metrics, config)
    tables = {
        "probe_rows": metrics,
        "probe_summary": summary,
        "domain_estimates": domains,
        "teacher_temperatures": dataset.temperatures,
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
            "status": "locked_replication_fixed_final_layer",
            "feature_path": str(replication_config.paths.embeddings_input),
            "artifacts": [table_manifest(paths[name], table) for name, table in tables.items()],
        },
    )
    paths["manifest"] = manifest_path
    return paths


def summarize_probe_rows(
    rows: pd.DataFrame, config: ObservabilityStudyConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[pd.DataFrame] = []
    domain_tables: list[pd.DataFrame] = []
    groups = [
        column
        for column in (
            "target_id",
            "probe",
            "feature_kind",
            "layer",
            "target_rank",
            "target_coordinates",
            "evaluation_split",
            "control",
            "repeat",
        )
        if column in rows
    ]
    for group_index, (key, frame) in enumerate(rows.groupby(groups, observed=True, dropna=False)):
        labels = dict(zip(groups, key, strict=True))
        for metric_index, metric in enumerate(METRICS):
            domains, summary = domain_sensitivity_summary(
                frame,
                metric,
                confidence_level=config.inference.confidence_level,
                wild_replicates=config.inference.bootstrap_replicates,
                seed=config.seed + 10000 * group_index + 100 * metric_index,
            )
            for name, value in labels.items():
                summary[name] = value
                domains[name] = value
            summary["metric"] = metric
            domains["metric"] = metric
            summaries.append(summary)
            domain_tables.append(domains)
    return pd.concat(summaries, ignore_index=True), pd.concat(domain_tables, ignore_index=True)


def _intercept_rows(dataset, target_id: str, train: np.ndarray, test: np.ndarray):
    definitions = {
        "global_intercept": [],
        "environment_intercept": [
            "burial",
            "secondary_structure",
            "contact_class",
            "conservation_class",
            "state_kind",
            "requested_corruption_ratio",
        ],
    }
    result = []
    for probe, strata in definitions.items():
        prediction = intercept_predict(
            dataset.metadata,
            dataset.residuals[target_id],
            train,
            test,
            strata,
        )
        result.append(
            prediction_metrics(
                dataset,
                target_id,
                test,
                prediction,
                probe=probe,
                control="observed",
                repeat=0,
                moved_fraction=1.0,
            )
        )
    return result
