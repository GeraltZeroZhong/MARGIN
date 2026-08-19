"""Strengthened sequence-only control trained on CATH teacher actions."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from margin.attribution.metrics import normalize_log_probabilities
from margin.constants import AA_ALPHABET, AA_TO_INDEX
from margin.provenance import read_json, runtime_manifest, write_json, write_parquet
from margin.studies.action_validation.evaluation import (
    _action_rms,
    _aligned_store,
    _aligned_teacher,
    _anchor,
    _anchored_rmse,
    _global_component,
)
from margin.studies.stability.calibration import load_cath_calibration_data
from margin.studies.stability.config import StabilityStudyConfig

PROFILE_COLUMNS = [f"profile_{aa}" for aa in AA_ALPHABET]
SECONDARY_CLASSES = ["helix", "strand", "turn_or_coil"]
BURIAL_CLASSES = ["buried", "exposed", "intermediate"]


def build_strong_features(config: StabilityStudyConfig) -> dict[str, Path]:
    """Build selection and final-fit feature matrices without stability outcomes."""

    output = config.paths.storage_dir / "strong_control" / "features"
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    paths = {
        "selection_cath": output / "selection_cath.npy",
        "final_cath": output / "final_cath.npy",
        "final_panel": output / "final_panel.npy",
        "feature_names": output / "feature_names.json",
        "environment_audit": config.paths.run_dir
        / "strong_control"
        / "environment_head_audit.parquet",
        "manifest": manifest,
    }
    if all(path.exists() for path in paths.values()):
        return paths
    cath = pd.read_parquet(config.paths.cath_queries).reset_index(drop=True)
    panel = pd.read_parquet(config.paths.run_dir / "panel" / "query_rows.parquet")
    train = np.flatnonzero(
        cath["observability_split"].eq(config.calibration.training_split).to_numpy()
    )
    final = np.flatnonzero(
        cath["observability_split"].isin(config.calibration.final_training_splits).to_numpy()
    )
    selection_bundle = _feature_bundle(config, cath, panel, train, "selection")
    final_bundle = _feature_bundle(config, cath, panel, final, "final")
    if selection_bundle["feature_names"] != final_bundle["feature_names"]:
        raise RuntimeError("selection and final strong-control features disagree")
    np.save(paths["selection_cath"], selection_bundle["cath"].astype(np.float32))
    np.save(paths["final_cath"], final_bundle["cath"].astype(np.float32))
    np.save(paths["final_panel"], final_bundle["panel"].astype(np.float32))
    write_json(paths["feature_names"], selection_bundle["feature_names"])
    environment_audit = pd.concat(
        [selection_bundle["environment_audit"], final_bundle["environment_audit"]],
        ignore_index=True,
    )
    write_parquet(paths["environment_audit"], environment_audit)
    write_json(
        manifest,
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "stability_labels_used": False,
            "feature_count": len(selection_bundle["feature_names"]),
            "selection_pca_fit_split": "development_train",
            "final_pca_fit_splits": config.calibration.final_training_splits,
            "sequence_predicted_environment_training_labels": [
                "CATH_secondary_structure",
                "CATH_burial",
            ],
            "paths": {name: str(path) for name, path in paths.items() if name != "manifest"},
        },
    )
    return paths


def run_strong_control(config: StabilityStudyConfig) -> dict[str, Path]:
    """Select C+ heads on CATH validation, refit, and predict the locked panel."""

    feature_paths = build_strong_features(config)
    output = config.paths.run_dir / "strong_control"
    storage = config.paths.storage_dir / "strong_control"
    models_directory = storage / "models"
    models_directory.mkdir(parents=True, exist_ok=True)
    cath_data = load_cath_calibration_data(config)
    metadata = cath_data["metadata"]
    wild = cath_data["wild"]
    actions = cath_data["actions"]
    selection_features = np.load(feature_paths["selection_cath"], mmap_mode="r")
    final_features = np.load(feature_paths["final_cath"], mmap_mode="r")
    panel_features = np.load(feature_paths["final_panel"], mmap_mode="r")
    panel_queries = pd.read_parquet(config.paths.run_dir / "panel" / "query_rows.parquet")
    panel_wild = panel_queries["wild_type"].map(AA_TO_INDEX).to_numpy(dtype=int)
    panel_scores = pd.read_parquet(config.paths.run_dir / "teacher_scores" / "scores.parquet")
    panel_actions = {
        teacher: _anchor(_aligned_teacher(panel_scores, panel_queries, teacher), panel_wild)
        for teacher in config.calibration.teacher_ids
    }
    train = np.flatnonzero(
        metadata["observability_split"].eq(config.calibration.training_split).to_numpy()
    )
    validation = np.flatnonzero(
        metadata["observability_split"].eq(config.calibration.selection_split).to_numpy()
    )
    final = np.flatnonzero(
        metadata["observability_split"].isin(config.calibration.final_training_splits).to_numpy()
    )
    locked = np.flatnonzero(metadata["observability_split"].eq("locked_test").to_numpy())
    selection_rows = []
    summary_rows = []
    audit_rows = []
    panel_components: dict[str, dict[str, np.ndarray]] = {}
    for teacher_index, teacher in enumerate(config.calibration.teacher_ids):
        action = actions[teacher]
        g_train = _global_component(action[train], wild[train], wild[train])
        g_validation = _global_component(action[train], wild[train], wild[validation])
        target_train = _anchor(action[train] - g_train, wild[train])
        target_validation = _anchor(action[validation] - g_validation, wild[validation])
        candidates = _candidate_specs(config)
        teacher_rows = []
        for candidate_index, candidate in enumerate(candidates):
            model = _make_model(
                candidate,
                seed=config.seed + teacher_index * 100 + candidate_index,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(selection_features[train], target_train)
            prediction = _anchor(model.predict(selection_features[validation]), wild[validation])
            row = {
                "teacher_id": teacher,
                **candidate,
                "validation_anchored_action_rmse": _anchored_rmse(
                    prediction, target_validation, wild[validation]
                ),
                "validation_action_r2": _anchored_r2(
                    prediction, target_validation, wild[validation]
                ),
            }
            selection_rows.append(row)
            teacher_rows.append(row)
        selected = min(
            teacher_rows,
            key=lambda row: (
                row["validation_anchored_action_rmse"],
                row["model_family"],
                row["hyperparameter_label"],
            ),
        )
        g_final = _global_component(action[final], wild[final], wild[final])
        target_final = _anchor(action[final] - g_final, wild[final])
        model = _make_model(selected, seed=config.seed + 10_000 + teacher_index)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(final_features[final], target_final)
        joblib.dump(model, models_directory / f"{teacher}.joblib")

        g_locked = _global_component(action[final], wild[final], wild[locked])
        c_locked = _anchor(model.predict(final_features[locked]), wild[locked])
        u_locked = _anchor(action[locked] - g_locked - c_locked, wild[locked])
        audit_rows.append(
            {
                "teacher_id": teacher,
                "split": "locked_test",
                "selected_model_family": selected["model_family"],
                "selected_hyperparameter": selected["hyperparameter_label"],
                "r2_g": _anchored_r2(g_locked, action[locked], wild[locked]),
                "r2_g_plus_c_plus": _anchored_r2(g_locked + c_locked, action[locked], wild[locked]),
                "action_rms": _action_rms(action[locked], wild[locked]),
                "u_plus_rms": _action_rms(u_locked, wild[locked]),
                "u_plus_over_action_rms": _action_rms(u_locked, wild[locked])
                / _action_rms(action[locked], wild[locked]),
            }
        )
        g_panel = _global_component(action[final], wild[final], panel_wild)
        c_panel = _anchor(model.predict(panel_features), panel_wild)
        u_panel = _anchor(panel_actions[teacher] - g_panel - c_panel, panel_wild)
        panel_components[teacher] = {
            "a": panel_actions[teacher],
            "g": g_panel,
            "c_plus": c_panel,
            "u_plus": u_panel,
        }
        summary_rows.append(
            {
                "teacher_id": teacher,
                "selected_model_family": selected["model_family"],
                "selected_hyperparameter": selected["hyperparameter_label"],
                "validation_anchored_action_rmse": selected["validation_anchored_action_rmse"],
                "validation_action_r2": selected["validation_action_r2"],
                "final_training_rows": len(final),
                "feature_count": final_features.shape[1],
            }
        )

    calibration = read_json(config.paths.run_dir / "calibration" / "selection.json")
    temperatures = calibration["final_parameters"]["temperatures"]
    scaled = {
        teacher: {
            name: values / float(temperatures[teacher]) for name, values in components.items()
        }
        for teacher, components in panel_components.items()
    }
    consensus = {
        name: np.mean(
            np.stack([scaled[teacher][name] for teacher in config.calibration.teacher_ids]),
            axis=0,
        )
        for name in ("a", "g", "c_plus", "u_plus")
    }
    matrix_path = storage / "panel_strong_control_components.npz"
    values: dict[str, np.ndarray] = {
        f"consensus_{name}": value.astype(np.float32) for name, value in consensus.items()
    }
    for teacher in config.calibration.teacher_ids:
        for name, value in scaled[teacher].items():
            values[f"{teacher}_{name}"] = value.astype(np.float32)
    np.savez_compressed(matrix_path, **values)
    tables = {
        "model_selection": pd.DataFrame(selection_rows),
        "training_summary": pd.DataFrame(summary_rows),
        "locked_cath_audit": pd.DataFrame(audit_rows),
    }
    paths = {name: output / f"{name}.parquet" for name in tables}
    for name, table in tables.items():
        write_parquet(paths[name], table)
    manifest_path = output / "strong_control_manifest.json"
    write_json(
        manifest_path,
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "status": "CPLUS_SELECTED_WITH_CATH_TEACHER_ACTIONS_ONLY",
            "stability_labels_used": False,
            "feature_manifest": str(feature_paths["manifest"]),
            "component_matrices": str(matrix_path),
            "tables": {
                name: {
                    "path": str(paths[name]),
                    "rows": len(table),
                    "columns": list(table.columns),
                }
                for name, table in tables.items()
            },
        },
    )
    return {**paths, "matrices": matrix_path, "manifest": manifest_path}


def _feature_bundle(
    config: StabilityStudyConfig,
    cath: pd.DataFrame,
    panel: pd.DataFrame,
    fit_indices: np.ndarray,
    fit_role: str,
    panel_store_root: Path | None = None,
    panel_profile_path: Path | None = None,
) -> dict[str, Any]:
    if panel_store_root is None:
        panel_store_root = config.paths.storage_dir / "representations"
    if panel_profile_path is None:
        panel_profile_path = config.paths.run_dir / "strong_control" / "panel_profiles.parquet"
    cath_parts = []
    panel_parts = []
    feature_names = []
    cath_stores = {
        "carp_640M": config.paths.cath_carp640_store,
        "esm2_650M": config.paths.cath_esm2_650_store,
        "esm1b_650M": config.paths.cath_esm1b_650_store,
    }
    for model_index, model_id in enumerate(config.strong_control.representation_models):
        cath_values = _aligned_store(cath_stores[model_id], cath, "representations.npy").astype(
            np.float32
        )
        panel_values = _aligned_store(
            panel_store_root / model_id,
            panel,
            "representations.npy",
        ).astype(np.float32)
        pca = PCA(
            n_components=config.strong_control.representation_pca_components,
            svd_solver="randomized",
            iterated_power=3,
            random_state=config.seed + model_index,
        ).fit(cath_values[fit_indices])
        cath_parts.append(pca.transform(cath_values).astype(np.float32))
        panel_parts.append(pca.transform(panel_values).astype(np.float32))
        feature_names.extend(
            [
                f"{model_id}_pca_{index:02d}"
                for index in range(config.strong_control.representation_pca_components)
            ]
        )

    cath_logp = normalize_log_probabilities(
        _aligned_store(config.paths.cath_esm2_150_store, cath, "log_probabilities.npy").astype(
            float
        )
    )
    panel_logp = normalize_log_probabilities(
        _aligned_store(
            panel_store_root / "esm2_150M",
            panel,
            "log_probabilities.npy",
        ).astype(float)
    )
    cath_wild = cath["native_aa"].map(AA_TO_INDEX).to_numpy(dtype=int)
    panel_wild = panel["wild_type"].map(AA_TO_INDEX).to_numpy(dtype=int)
    cath_basic, basic_names = _basic_features(
        cath, cath_logp, cath_wild, config.strong_control.local_context_radius
    )
    panel_basic, _ = _basic_features(
        panel, panel_logp, panel_wild, config.strong_control.local_context_radius
    )
    cath_profile = _aligned_profiles(
        cath,
        config.paths.run_dir / "strong_control" / "cath_profiles.parquet",
    )
    panel_profile = _aligned_profiles(
        panel,
        panel_profile_path,
    )
    profile_names = [
        *PROFILE_COLUMNS,
        "profile_entropy",
        "log1p_homolog_observations",
        "profile_covered",
    ]
    cath_parts.extend([cath_basic, cath_profile])
    panel_parts.extend([panel_basic, panel_profile])
    feature_names.extend([*basic_names, *profile_names])
    cath_base = np.concatenate(cath_parts, axis=1).astype(np.float32)
    panel_base = np.concatenate(panel_parts, axis=1).astype(np.float32)

    environment_parts_cath = []
    environment_parts_panel = []
    audit_rows = []
    for target, classes in (
        ("secondary_structure", SECONDARY_CLASSES),
        ("burial", BURIAL_CLASSES),
    ):
        labels = cath[target].astype(str).to_numpy()
        labeled = fit_indices[np.isin(labels[fit_indices], classes)]
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(
                        C=1.0,
                        max_iter=300,
                        solver="lbfgs",
                        class_weight="balanced",
                        random_state=config.seed,
                    ),
                ),
            ]
        ).fit(cath_base[labeled], labels[labeled])
        cath_probability = _fixed_probabilities(model, cath_base, classes)
        panel_probability = _fixed_probabilities(model, panel_base, classes)
        environment_parts_cath.append(cath_probability)
        environment_parts_panel.append(panel_probability)
        feature_names.extend([f"predicted_{target}_{value}" for value in classes])
        evaluation = np.flatnonzero(~np.isin(np.arange(len(cath)), fit_indices))
        evaluation = evaluation[np.isin(labels[evaluation], classes)]
        predicted = np.asarray(classes)[np.argmax(cath_probability[evaluation], axis=1)]
        audit_rows.append(
            {
                "feature_fit_role": fit_role,
                "target": target,
                "fit_rows": len(labeled),
                "evaluation_rows": len(evaluation),
                "evaluation_accuracy": float((predicted == labels[evaluation]).mean()),
            }
        )
    cath_result = np.concatenate([cath_base, *environment_parts_cath], axis=1)
    panel_result = np.concatenate([panel_base, *environment_parts_panel], axis=1)
    return {
        "cath": cath_result,
        "panel": panel_result,
        "feature_names": feature_names,
        "environment_audit": pd.DataFrame(audit_rows),
    }


def _basic_features(
    metadata: pd.DataFrame,
    sequence_logp: np.ndarray,
    wild: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, list[str]]:
    sequence_action = _anchor(sequence_logp, wild)
    probability = softmax(sequence_logp, axis=1)
    entropy = -np.sum(probability * np.log(np.maximum(probability, 1e-12)), axis=1)[:, None]
    wild_one_hot = np.eye(len(AA_ALPHABET), dtype=np.float32)[wild]
    local = np.zeros((len(metadata), len(AA_ALPHABET)), dtype=np.float32)
    position_fraction = np.zeros((len(metadata), 1), dtype=np.float32)
    log_length = np.zeros((len(metadata), 1), dtype=np.float32)
    for row_index, row in enumerate(metadata.itertuples(index=False)):
        sequence = str(row.sequence)
        position = int(row.position)
        start = max(0, position - radius)
        stop = min(len(sequence), position + radius + 1)
        neighbors = [sequence[index] for index in range(start, stop) if index != position]
        for residue in neighbors:
            if residue in AA_TO_INDEX:
                local[row_index, AA_TO_INDEX[residue]] += 1.0
        if neighbors:
            local[row_index] /= len(neighbors)
        position_fraction[row_index, 0] = position / max(len(sequence) - 1, 1)
        log_length[row_index, 0] = np.log1p(len(sequence))
    names = [
        *[f"esm2_150M_action_{aa}" for aa in AA_ALPHABET],
        "esm2_150M_entropy",
        *[f"wild_type_{aa}" for aa in AA_ALPHABET],
        *[f"radius_{radius}_composition_{aa}" for aa in AA_ALPHABET],
        "normalized_position",
        "log1p_length",
    ]
    values = np.concatenate(
        [
            sequence_action.astype(np.float32),
            entropy.astype(np.float32),
            wild_one_hot,
            local,
            position_fraction,
            log_length,
        ],
        axis=1,
    )
    return values, names


def _aligned_profiles(metadata: pd.DataFrame, path: Path) -> np.ndarray:
    profiles = pd.read_parquet(path)
    keys = ["state_id", "domain_id", "position"]
    columns = [*PROFILE_COLUMNS, "profile_entropy", "homolog_observations", "profile_covered"]
    aligned = metadata[keys].merge(profiles[[*keys, *columns]], on=keys, validate="one_to_one")
    if len(aligned) != len(metadata):
        raise ValueError(f"profile table lacks query coverage: {path}")
    values = aligned[PROFILE_COLUMNS].to_numpy(dtype=np.float32)
    entropy = aligned[["profile_entropy"]].to_numpy(dtype=np.float32)
    observations = np.log1p(aligned[["homolog_observations"]].to_numpy(dtype=np.float32))
    covered = aligned[["profile_covered"]].to_numpy(dtype=np.float32)
    return np.concatenate([values, entropy, observations, covered], axis=1)


def _fixed_probabilities(model: Pipeline, values: np.ndarray, classes: list[str]) -> np.ndarray:
    observed = list(model.named_steps["logistic"].classes_)
    probability = model.predict_proba(values)
    return np.stack([probability[:, observed.index(value)] for value in classes], axis=1)


def _candidate_specs(config: StabilityStudyConfig) -> list[dict[str, Any]]:
    specs = [
        {
            "model_family": "ridge",
            "hyperparameter_label": f"alpha={alpha:g}",
            "alpha": float(alpha),
            "hidden_size": 0,
        }
        for alpha in config.strong_control.ridge_alphas
    ]
    specs.extend(
        {
            "model_family": "mlp",
            "hyperparameter_label": f"hidden={hidden};alpha={alpha:g}",
            "alpha": float(alpha),
            "hidden_size": int(hidden),
        }
        for hidden in config.strong_control.mlp_hidden_sizes
        for alpha in config.strong_control.mlp_alphas
    )
    return specs


def _make_model(specification: dict[str, Any], seed: int) -> Pipeline:
    if specification["model_family"] == "ridge":
        estimator = Ridge(alpha=float(specification["alpha"]), solver="lsqr")
    elif specification["model_family"] == "mlp":
        estimator = MLPRegressor(
            hidden_layer_sizes=(int(specification["hidden_size"]),),
            activation="relu",
            solver="adam",
            alpha=float(specification["alpha"]),
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=250,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=15,
            random_state=seed,
        )
    else:
        raise ValueError(f"unknown C+ model family: {specification['model_family']}")
    return Pipeline([("scale", StandardScaler()), ("model", estimator)])


def _anchored_r2(predicted: np.ndarray, observed: np.ndarray, wild: np.ndarray) -> float:
    prediction = _anchor(predicted, wild)
    target = _anchor(observed, wild)
    mask = np.ones(target.shape, dtype=bool)
    mask[np.arange(len(mask)), wild] = False
    y = target[mask]
    error = y - prediction[mask]
    denominator = np.sum((y - y.mean()) ** 2)
    return float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else float("nan")
