"""CLR-versus-ILR representation sensitivity for residual probes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from margin.config import load_config
from margin.provenance import runtime_manifest, table_manifest, write_json, write_parquet
from margin.studies.observability.config import ObservabilityStudyConfig
from margin.studies.observability.current import summarize_probe_rows
from margin.studies.observability.probes import (
    load_aligned_parquet_features,
    prediction_metrics,
    ridge_predict,
)
from margin.studies.observability.targets import (
    ilr,
    inverse_ilr,
    load_foundation_residual_dataset,
    load_replication_residual_dataset,
)


def run_compositional_audit(
    config: ObservabilityStudyConfig, dataset_kind: str, output: Path
) -> dict[str, Path]:
    """Verify that orthonormal ILR coordinates do not alter the Ridge conclusion."""

    if dataset_kind == "current":
        dataset = load_foundation_residual_dataset(config)
        feature_path = (
            config.paths.project_root / "data/interim/student/state_position_embeddings.parquet"
        )
        train = np.flatnonzero(dataset.metadata["eligible_for_training"].astype(bool).to_numpy())
        test = np.flatnonzero(~dataset.metadata["eligible_for_training"].astype(bool).to_numpy())
        split = "foundation_external_benchmark"
    elif dataset_kind == "replication":
        replication_config = load_config(config.paths.replication_config)
        dataset = load_replication_residual_dataset(config, replication_config.paths.run_dir)
        if replication_config.paths.embeddings_input is None:
            raise ValueError("replication ILR audit requires paths.embeddings_input")
        feature_path = replication_config.paths.embeddings_input
        role = dataset.metadata["analysis_role"]
        train = np.flatnonzero(
            role.isin(["development_train", "development_validation"]).to_numpy()
        )
        test = np.flatnonzero(role.eq("locked_test").to_numpy())
        split = "locked_test"
    else:
        raise ValueError("dataset_kind must be current or replication")
    features = load_aligned_parquet_features(feature_path, dataset.metadata)
    target_id = config.residual_targets.primary
    target = dataset.residuals[target_id]
    clr_prediction = ridge_predict(
        features[train], target[train], features[test], alpha=config.probes.ridge_alpha
    )
    scaler = StandardScaler()
    train_features = scaler.fit_transform(features[train])
    test_features = scaler.transform(features[test])
    model = Ridge(alpha=config.probes.ridge_alpha, solver="lsqr")
    model.fit(train_features, ilr(target[train]))
    ilr_prediction = inverse_ilr(model.predict(test_features))
    rows = pd.concat(
        [
            prediction_metrics(
                dataset,
                target_id,
                test,
                clr_prediction,
                probe="ridge_compositional_sensitivity",
                target_coordinates="clr20",
                evaluation_split=split,
                control="observed",
                repeat=0,
            ),
            prediction_metrics(
                dataset,
                target_id,
                test,
                ilr_prediction,
                probe="ridge_compositional_sensitivity",
                target_coordinates="ilr19",
                evaluation_split=split,
                control="observed",
                repeat=0,
            ),
        ],
        ignore_index=True,
    )
    summary, domains = summarize_probe_rows(rows, config)
    equivalence = pd.DataFrame(
        [
            {
                "dataset": dataset_kind,
                "maximum_absolute_prediction_difference": float(
                    np.max(np.abs(clr_prediction - ilr_prediction))
                ),
                "root_mean_squared_prediction_difference": float(
                    np.sqrt(np.mean((clr_prediction - ilr_prediction) ** 2))
                ),
            }
        ]
    )
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "rows": rows,
        "summary": summary,
        "domain_estimates": domains,
        "equivalence": equivalence,
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
            "dataset": dataset_kind,
            "feature_path": str(feature_path),
            "artifacts": [table_manifest(paths[name], table) for name, table in tables.items()],
        },
    )
    paths["manifest"] = manifest_path
    return paths
