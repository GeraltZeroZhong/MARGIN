"""Layerwise, local-window, reduced-rank, and bottleneck observability study probes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from margin.provenance import read_json, runtime_manifest, table_manifest, write_json, write_parquet
from margin.studies.observability.config import ObservabilityStudyConfig
from margin.studies.observability.current import summarize_probe_rows
from margin.studies.observability.probes import (
    intercept_predict,
    mlp_predict,
    prediction_metrics,
    ridge_predict,
    shuffled_target,
)
from margin.studies.observability.targets import ResidualDataset


def run_layerwise_audit(
    dataset: ResidualDataset,
    config: ObservabilityStudyConfig,
    representation_dir: Path,
    output: Path,
    *,
    selection_train: np.ndarray,
    selection_validation: np.ndarray,
    final_train: np.ndarray,
    final_test: np.ndarray,
    final_split_label: str,
) -> dict[str, Path]:
    """Select a layer on development data, then evaluate every probe on untouched rows."""

    output.mkdir(parents=True, exist_ok=True)
    manifest = read_json(representation_dir / "manifest.json")
    keys = pd.read_parquet(representation_dir / "keys.parquet")
    feature_rows = _aligned_feature_rows(keys, dataset.metadata)
    store = np.load(representation_dir / "representations.npy", mmap_mode="r")
    if tuple(manifest["shape"]) != store.shape:
        raise ValueError("representation manifest shape does not match the array")
    layers = [int(layer) for layer in manifest["layers"]]
    primary = config.residual_targets.primary
    target = dataset.residuals[primary]
    selection_records = []
    feature_kinds = {0: "query", 1: "local_mean"}

    for kind_index, kind in feature_kinds.items():
        for layer_offset, layer in enumerate(layers):
            features = _feature_matrix(store, feature_rows, layer_offset, kind_index)
            validation_prediction = ridge_predict(
                features[selection_train],
                target[selection_train],
                features[selection_validation],
                alpha=config.probes.ridge_alpha,
            )
            validation = prediction_metrics(
                dataset,
                primary,
                selection_validation,
                validation_prediction,
                probe="layerwise_ridge",
                feature_kind=kind,
                layer=layer,
                evaluation_split="development_validation",
                control="observed",
                repeat=0,
            )
            domain_mean = validation.groupby("domain_id", observed=True)[
                "jsd_reduction_nats"
            ].mean()
            selection_records.append(
                {
                    "feature_kind": kind,
                    "layer": layer,
                    "validation_jsd_reduction_nats": float(domain_mean.mean()),
                    "validation_positive_domains": int((domain_mean > 0).sum()),
                    "validation_domains": int(len(domain_mean)),
                }
            )
            print(
                f"validation feature={kind} layer={layer} jsd={float(domain_mean.mean()):.6f}",
                flush=True,
            )
            del features, validation_prediction, validation

    selection = pd.DataFrame(selection_records).sort_values(
        ["validation_jsd_reduction_nats", "feature_kind", "layer"],
        ascending=[False, True, True],
        ignore_index=True,
    )
    best = selection.iloc[0]
    best_kind_index = 0 if best["feature_kind"] == "query" else 1
    best_layer_offset = layers.index(int(best["layer"]))
    best_features = _feature_matrix(store, feature_rows, best_layer_offset, best_kind_index)
    summary_tables: list[pd.DataFrame] = []
    domain_tables: list[pd.DataFrame] = []
    environment_summary_tables: list[pd.DataFrame] = []
    environment_domain_tables: list[pd.DataFrame] = []

    def record(frame: pd.DataFrame) -> None:
        summary, domains = summarize_probe_rows(frame, config)
        summary_tables.append(summary)
        domain_tables.append(domains)
        for environment in config.candidate_environments:
            selected = frame.loc[
                frame["state_kind"].eq(environment.state_kind)
                & frame["requested_corruption_ratio"].eq(environment.requested_corruption_ratio)
                & frame[environment.axis].eq(environment.value)
            ]
            if len(selected) < config.inference.minimum_environment_rows:
                continue
            environment_summary, environment_domains = summarize_probe_rows(selected, config)
            for table in (environment_summary, environment_domains):
                table["environment_id"] = environment.environment_id
                table["environment_axis"] = environment.axis
                table["environment_value"] = environment.value
                table["environment_rows"] = len(selected)
                table["environment_domains"] = selected["domain_id"].nunique()
            environment_summary_tables.append(environment_summary)
            environment_domain_tables.append(environment_domains)

    selected_prediction = ridge_predict(
        best_features[final_train],
        target[final_train],
        best_features[final_test],
        alpha=config.probes.ridge_alpha,
    )
    selected_test_rows = prediction_metrics(
        dataset,
        primary,
        final_test,
        selected_prediction,
        probe="layerwise_ridge",
        feature_kind=best["feature_kind"],
        layer=int(best["layer"]),
        target_rank=np.nan,
        evaluation_split=final_split_label,
        control="observed",
        repeat=0,
        moved_fraction=1.0,
    )
    record(selected_test_rows)
    print(
        f"selected feature={best['feature_kind']} layer={int(best['layer'])}",
        flush=True,
    )
    test_metadata = dataset.metadata.iloc[final_test].reset_index(drop=True)
    local_test_indices = np.arange(len(final_test), dtype=int)
    prediction_rng = np.random.default_rng(config.seed + 9250)

    for rank in config.probes.reduced_ranks:
        prediction = ridge_predict(
            best_features[final_train],
            target[final_train],
            best_features[final_test],
            alpha=config.probes.ridge_alpha,
            rank=rank,
        )
        record(
            prediction_metrics(
                dataset,
                primary,
                final_test,
                prediction,
                probe="reduced_rank_ridge",
                feature_kind=best["feature_kind"],
                layer=int(best["layer"]),
                target_rank=rank,
                evaluation_split=final_split_label,
                control="observed",
                repeat=0,
                moved_fraction=1.0,
            )
        )
        print(f"reduced_rank={rank}", flush=True)
        for control in config.probes.shuffle_controls:
            for repeat in range(config.probes.control_repeats):
                shuffled_prediction, moved = shuffled_target(
                    test_metadata,
                    prediction,
                    local_test_indices,
                    control,
                    prediction_rng,
                )
                record(
                    prediction_metrics(
                        dataset,
                        primary,
                        final_test,
                        shuffled_prediction,
                        probe="reduced_rank_ridge",
                        feature_kind=best["feature_kind"],
                        layer=int(best["layer"]),
                        target_rank=rank,
                        evaluation_split=final_split_label,
                        control=f"prediction_{control}",
                        repeat=repeat,
                        moved_fraction=moved,
                    )
                )

    mlp = mlp_predict(
        best_features[final_train],
        target[final_train],
        best_features[final_test],
        hidden_units=config.probes.mlp_hidden_units,
        max_iterations=config.probes.mlp_max_iterations,
        seed=config.seed,
    )
    mlp_rows = prediction_metrics(
        dataset,
        primary,
        final_test,
        mlp,
        probe="bottleneck_mlp",
        feature_kind=best["feature_kind"],
        layer=int(best["layer"]),
        target_rank=np.nan,
        evaluation_split=final_split_label,
        control="observed",
        repeat=0,
        moved_fraction=1.0,
    )
    record(mlp_rows)
    print("bottleneck_mlp=complete", flush=True)
    for control in config.probes.shuffle_controls:
        for repeat in range(config.probes.control_repeats):
            shuffled_prediction, moved = shuffled_target(
                test_metadata,
                mlp,
                local_test_indices,
                control,
                prediction_rng,
            )
            record(
                prediction_metrics(
                    dataset,
                    primary,
                    final_test,
                    shuffled_prediction,
                    probe="bottleneck_mlp",
                    feature_kind=best["feature_kind"],
                    layer=int(best["layer"]),
                    target_rank=np.nan,
                    evaluation_split=final_split_label,
                    control=f"prediction_{control}",
                    repeat=repeat,
                    moved_fraction=moved,
                )
            )
    for target_id in dataset.residuals:
        if target_id == primary:
            continue
        prediction = ridge_predict(
            best_features[final_train],
            dataset.residuals[target_id][final_train],
            best_features[final_test],
            alpha=config.probes.ridge_alpha,
        )
        record(
            prediction_metrics(
                dataset,
                target_id,
                final_test,
                prediction,
                probe="validation_selected_ridge",
                feature_kind=best["feature_kind"],
                layer=int(best["layer"]),
                target_rank=np.nan,
                evaluation_split=final_split_label,
                control="observed",
                repeat=0,
                moved_fraction=1.0,
            )
        )
        print(f"secondary_target={target_id}", flush=True)

    rng = np.random.default_rng(config.seed + 9200)
    for control in config.probes.shuffle_controls:
        for repeat in range(config.probes.control_repeats):
            shuffled, moved = shuffled_target(
                dataset.metadata,
                target,
                final_train,
                control,
                rng,
            )
            prediction = ridge_predict(
                best_features[final_train],
                shuffled,
                best_features[final_test],
                alpha=config.probes.ridge_alpha,
            )
            record(
                prediction_metrics(
                    dataset,
                    primary,
                    final_test,
                    prediction,
                    probe="layerwise_ridge",
                    feature_kind=best["feature_kind"],
                    layer=int(best["layer"]),
                    target_rank=np.nan,
                    evaluation_split=final_split_label,
                    control=control,
                    repeat=repeat,
                    moved_fraction=moved,
                )
            )
        print(f"shuffle_control={control}", flush=True)

    intercepts = {
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
    for probe, strata in intercepts.items():
        prediction = intercept_predict(
            dataset.metadata,
            target,
            final_train,
            final_test,
            strata,
        )
        record(
            prediction_metrics(
                dataset,
                primary,
                final_test,
                prediction,
                probe=probe,
                feature_kind="none",
                layer=-1,
                target_rank=np.nan,
                evaluation_split=final_split_label,
                control="observed",
                repeat=0,
                moved_fraction=1.0,
            )
        )

    summary = pd.concat(summary_tables, ignore_index=True)
    domain_estimates = pd.concat(domain_tables, ignore_index=True)
    tables = {
        "selection": selection,
        "selected_test_rows": selected_test_rows,
        "summary": summary,
        "domain_estimates": domain_estimates,
    }
    if environment_summary_tables:
        tables["environment_summary"] = pd.concat(environment_summary_tables, ignore_index=True)
        tables["environment_domain_estimates"] = pd.concat(
            environment_domain_tables, ignore_index=True
        )
    paths = {name: output / f"{name}.parquet" for name in tables}
    for name, table in tables.items():
        write_parquet(paths[name], table)
    manifest_path = output / "manifest.json"
    write_json(
        manifest_path,
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "representation_manifest": str(representation_dir / "manifest.json"),
            "selection_target": primary,
            "selected_feature_kind": str(best["feature_kind"]),
            "selected_layer": int(best["layer"]),
            "selected_validation_jsd_reduction_nats": float(best["validation_jsd_reduction_nats"]),
            "final_split": final_split_label,
            "artifacts": [table_manifest(paths[name], table) for name, table in tables.items()],
        },
    )
    paths["manifest"] = manifest_path
    return paths


def current_split_indices(dataset: ResidualDataset, seed: int) -> tuple[np.ndarray, ...]:
    """Create a development-only 12/4 selection split and keep all external domains untouched."""

    eligible = dataset.metadata.loc[
        dataset.metadata["eligible_for_training"].astype(bool), "domain_id"
    ].drop_duplicates()
    domains = eligible.to_numpy()
    rng = np.random.default_rng(seed + 41)
    rng.shuffle(domains)
    selection_validation_domains = set(domains[:4])
    selection_train_domains = set(domains[4:])
    domain = dataset.metadata["domain_id"]
    selection_train = np.flatnonzero(domain.isin(selection_train_domains).to_numpy())
    selection_validation = np.flatnonzero(domain.isin(selection_validation_domains).to_numpy())
    final_train = np.flatnonzero(dataset.metadata["eligible_for_training"].astype(bool).to_numpy())
    final_test = np.flatnonzero(~dataset.metadata["eligible_for_training"].astype(bool).to_numpy())
    return selection_train, selection_validation, final_train, final_test


def replication_split_indices(dataset: ResidualDataset) -> tuple[np.ndarray, ...]:
    """Use the preregistered train/validation/locked-test roles without resampling."""

    role = dataset.metadata["analysis_role"]
    selection_train = np.flatnonzero(role.eq("development_train").to_numpy())
    selection_validation = np.flatnonzero(role.eq("development_validation").to_numpy())
    final_train = np.flatnonzero(
        role.isin(["development_train", "development_validation"]).to_numpy()
    )
    final_test = np.flatnonzero(role.eq("locked_test").to_numpy())
    if min(map(len, (selection_train, selection_validation, final_train, final_test))) == 0:
        raise ValueError("replication dataset does not contain every preregistered split")
    return selection_train, selection_validation, final_train, final_test


def _aligned_feature_rows(keys: pd.DataFrame, metadata: pd.DataFrame) -> np.ndarray:
    columns = ["state_id", "domain_id", "position"]
    indexed = keys[columns].copy()
    indexed["feature_row"] = np.arange(len(indexed), dtype=int)
    aligned = metadata[columns].merge(indexed, on=columns, validate="one_to_one")
    if len(aligned) != len(metadata):
        raise ValueError("representation keys do not cover every residual row")
    return aligned["feature_row"].to_numpy(dtype=int)


def _feature_matrix(
    store: np.ndarray,
    feature_rows: np.ndarray,
    layer_offset: int,
    kind_index: int,
) -> np.ndarray:
    values = np.asarray(store[feature_rows, layer_offset, kind_index, :], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("representation store contains non-finite values")
    return values
