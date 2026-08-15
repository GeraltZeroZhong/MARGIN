"""Environment-label deployability audit for the CARP residual predictor."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from margin.provenance import runtime_manifest, table_manifest, write_json, write_parquet
from margin.studies.generalization.audit import load_features, summarize
from margin.studies.generalization.config import GeneralizationStudyConfig
from margin.studies.generalization.targets import load_generalization_residual_dataset
from margin.studies.observability.probes import prediction_metrics, ridge_predict
from margin.studies.observability.stats import domain_sensitivity_summary

STRUCTURAL_COLUMNS = ["burial", "secondary_structure", "contact_class"]


def run_environment_audit(config: GeneralizationStudyConfig) -> dict[str, object]:
    """Compare oracle structural, MSA, predicted, and absent environment features."""

    output = config.paths.run_dir / "environment_audit"
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_generalization_residual_dataset(config)
    sequence_features = load_features(
        config.paths.storage_dir / "architecture" / "carp_640M", dataset.metadata
    )
    role = dataset.metadata["observability_split"]
    train = np.flatnonzero(role.isin(["development_train", "development_validation"]).to_numpy())
    test = np.flatnonzero(role.eq("locked_test").to_numpy())
    metadata = dataset.metadata
    oracle_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    oracle_train = oracle_encoder.fit_transform(metadata.iloc[train][STRUCTURAL_COLUMNS])
    oracle_test = oracle_encoder.transform(metadata.iloc[test][STRUCTURAL_COLUMNS])
    msa_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    msa_train_class = msa_encoder.fit_transform(metadata.iloc[train][["conservation_class"]])
    msa_test_class = msa_encoder.transform(metadata.iloc[test][["conservation_class"]])
    msa_train = np.column_stack(
        [metadata.iloc[train]["conservation_score"].to_numpy(dtype=float), msa_train_class]
    )
    msa_test = np.column_stack(
        [metadata.iloc[test]["conservation_score"].to_numpy(dtype=float), msa_test_class]
    )
    predicted_train, predicted_test, accuracy = _predicted_environment_features(
        sequence_features, metadata, train, test, config
    )
    routes = {
        "no_environment_labels": (sequence_features[train], sequence_features[test]),
        "true_structural_environment": (
            np.column_stack([sequence_features[train], oracle_train]),
            np.column_stack([sequence_features[test], oracle_test]),
        ),
        "msa_conservation": (
            np.column_stack([sequence_features[train], msa_train]),
            np.column_stack([sequence_features[test], msa_test]),
        ),
        "sequence_predicted_environment": (
            np.column_stack([sequence_features[train], predicted_train]),
            np.column_stack([sequence_features[test], predicted_test]),
        ),
    }
    rows = []
    target_id = config.architecture.primary_target
    for route, (x_train, x_test) in routes.items():
        prediction = ridge_predict(
            x_train,
            dataset.residuals[target_id][train],
            x_test,
            alpha=config.architecture.ridge_alpha,
            rank=config.architecture.rrr_rank,
        )
        rows.append(
            prediction_metrics(
                dataset,
                target_id,
                test,
                prediction,
                model_id="carp_640M",
                probe="environment_route_rrr",
                environment_route=route,
                evaluation_split="observability_locked_test_reused_postdecision",
                control="observed",
                repeat=0,
            )
        )
    route_rows = pd.concat(rows, ignore_index=True)
    route_summary, route_domains = summarize(route_rows, config, seed_offset=500_000)
    deltas, delta_summary = _route_deltas(route_rows, config)
    tables = {
        "route_rows": route_rows,
        "route_summary": route_summary,
        "route_domain_estimates": route_domains,
        "route_deltas": deltas,
        "route_delta_summary": delta_summary,
        "environment_prediction_accuracy": accuracy,
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
            "test_environment_labels_used_for_training": False,
            "main_dms_route_requires_environment_labels": False,
            "artifacts": [table_manifest(paths[name], table) for name, table in tables.items()],
        },
    )
    return {**paths, "manifest": manifest_path}


def _predicted_environment_features(
    features: np.ndarray,
    metadata: pd.DataFrame,
    train: np.ndarray,
    test: np.ndarray,
    config: GeneralizationStudyConfig,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(features[train])
    x_test = scaler.transform(features[test])
    train_blocks = []
    test_blocks = []
    accuracy_rows = []
    for column in [*STRUCTURAL_COLUMNS, "conservation_class"]:
        classifier = RidgeClassifier(alpha=config.architecture.ridge_alpha)
        y_train = metadata.iloc[train][column].astype(str).to_numpy()
        y_test = metadata.iloc[test][column].astype(str).to_numpy()
        classifier.fit(x_train, y_train)
        train_score = classifier.decision_function(x_train)
        test_score = classifier.decision_function(x_test)
        if train_score.ndim == 1:
            train_score = np.column_stack([-train_score, train_score])
            test_score = np.column_stack([-test_score, test_score])
        train_blocks.append(softmax(train_score, axis=1))
        test_blocks.append(softmax(test_score, axis=1))
        predicted = classifier.predict(x_test)
        accuracy_rows.append(
            {
                "environment_axis": column,
                "accuracy": float(np.mean(predicted == y_test)),
                "classes": int(len(classifier.classes_)),
                "test_rows": int(len(test)),
            }
        )
    return (
        np.column_stack(train_blocks),
        np.column_stack(test_blocks),
        pd.DataFrame(accuracy_rows),
    )


def _route_deltas(
    rows: pd.DataFrame,
    config: GeneralizationStudyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    domain = (
        rows.groupby(["environment_route", "domain_id"], observed=True)["jsd_reduction_nats"]
        .mean()
        .reset_index()
    )
    baseline = domain.loc[
        domain["environment_route"].eq("no_environment_labels"),
        ["domain_id", "jsd_reduction_nats"],
    ].rename(columns={"jsd_reduction_nats": "baseline_jsd_reduction_nats"})
    selected = domain.loc[domain["environment_route"].ne("no_environment_labels")].merge(
        baseline, on="domain_id", validate="many_to_one"
    )
    selected["jsd_delta_vs_no_environment_nats"] = (
        selected["jsd_reduction_nats"] - selected["baseline_jsd_reduction_nats"]
    )
    summaries = []
    for route_index, (route, frame) in enumerate(
        selected.groupby("environment_route", observed=True)
    ):
        _, summary = domain_sensitivity_summary(
            frame.rename(columns={"jsd_delta_vs_no_environment_nats": "value"}),
            "value",
            confidence_level=config.inference.confidence_level,
            wild_replicates=config.inference.bootstrap_replicates,
            seed=config.seed + 600_000 + route_index,
        )
        summary["environment_route"] = route
        summaries.append(summary)
    return selected, pd.concat(summaries, ignore_index=True)
