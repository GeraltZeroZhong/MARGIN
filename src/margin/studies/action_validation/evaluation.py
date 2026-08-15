"""Locked action-validation study structure-unique action decomposition and evaluation."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import StandardScaler

from margin.attribution.metrics import normalize_log_probabilities
from margin.constants import AA_ALPHABET, AA_TO_INDEX
from margin.provenance import (
    read_json,
    runtime_manifest,
    table_manifest,
    write_json,
    write_parquet,
)
from margin.studies.action_validation.config import ActionValidationStudyConfig
from margin.studies.counterfactuals.evaluation import stratified_domain_bootstrap
from margin.teachers.schema import logp_columns

PRIMARY_POPULATION = "megascale_dense"
REPLICATION_POPULATION = "s669_sparse_cross_platform"
SEQUENCE_METHOD = "sequence_only"
CONSENSUS = "consensus"
METRICS = (
    "spearman",
    "full_ndcg",
    "ndcg_at_10_percent",
    "stabilizing_top_10_percent_recall",
)


def evaluate_action_validation(config: ActionValidationStudyConfig) -> dict[str, Path | str | bool]:
    """Fit G/C without panel labels and evaluate the locked stability outcomes once."""

    _require_locked_inputs(config)
    output = config.paths.run_dir / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    storage = config.paths.storage_dir / "evaluation"
    storage.mkdir(parents=True, exist_ok=True)
    panel = config.paths.run_dir / "panel"
    domains = pd.read_parquet(panel / "domains.parquet")
    variants = pd.read_parquet(panel / "variants.parquet")
    residues = pd.read_parquet(panel / "residues.parquet")
    queries = pd.read_parquet(panel / "query_rows.parquet")

    panel_carp = _aligned_store(
        config.paths.storage_dir / "representations" / "carp_640M",
        queries,
        "representations.npy",
    ).astype(np.float32)
    panel_sequence_logp = _aligned_store(
        config.paths.storage_dir / "representations" / "esm2_150M",
        queries,
        "log_probabilities.npy",
    ).astype(float)
    panel_wild = queries["wild_type"].map(AA_TO_INDEX).to_numpy(dtype=int)
    panel_sequence_action = _anchor(panel_sequence_logp, panel_wild)
    panel_features = _context_features(
        queries,
        panel_carp,
        panel_sequence_logp,
        wild_type_column="wild_type",
        radius=config.decomposition.context_radius,
    )
    panel_teacher_scores = pd.read_parquet(
        config.paths.run_dir / "teacher_scores" / "scores.parquet"
    )
    panel_actions = {
        teacher: _anchor(_aligned_teacher(panel_teacher_scores, queries, teacher), panel_wild)
        for teacher in config.decomposition.teacher_ids
    }

    training = _training_artifacts(config)
    decompositions: dict[str, dict[str, np.ndarray | float]] = {}
    selection_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    for teacher in config.decomposition.teacher_ids:
        result = _fit_teacher_decomposition(
            teacher,
            training,
            panel_features,
            panel_wild,
            panel_actions[teacher],
            config,
        )
        decompositions[teacher] = result["panel"]
        selection_rows.extend(result["selection_rows"])
        training_rows.append(result["training_row"])

    scaled = {
        teacher: {
            name: np.asarray(values[name], dtype=float) * float(values["scale"])
            for name in ("a", "g", "c", "u")
        }
        for teacher, values in decompositions.items()
    }
    consensus = {
        name: np.mean(
            np.stack([scaled[teacher][name] for teacher in config.decomposition.teacher_ids]),
            axis=0,
        )
        for name in ("a", "g", "c", "u")
    }
    agreement = _teacher_agreement(queries, domains, scaled, config)
    gated_u, gate = _agreement_gated_u(consensus["u"], scaled, agreement["position"], config)
    components, query_rows, mutant = _variant_components(
        variants,
        domains,
        residues,
        queries,
        panel_sequence_action,
        scaled,
        consensus,
        gated_u,
        gate,
    )
    methods, method_registry = _method_predictions(
        components,
        scaled,
        consensus,
        panel_sequence_action,
        query_rows,
        mutant,
        gated_u,
    )
    domain_metrics = _domain_metrics(components, methods, config.inference.top_fraction)
    component_margins = _component_margins(domain_metrics, method_registry)
    component_summary = _summarize_component_margins(component_margins, config)
    u_summary = _summarize_u_margins(component_margins, config)

    shuffle = _shuffled_u_control(
        components,
        queries,
        panel_sequence_action,
        consensus,
        query_rows,
        mutant,
        domain_metrics,
        config,
    )
    subgroup = _subgroup_analysis(components, methods, method_registry, config)
    routing = _routing_diagnostic(domain_metrics, components, config)
    quality_controls = _quality_controls(
        panel_actions,
        decompositions,
        scaled,
        consensus,
        panel_sequence_action,
        methods,
        components,
    )
    gate_checks, decision = _decision(u_summary, shuffle["summary"], config)

    matrices_path = storage / "decomposition_matrices.npz"
    matrix_values: dict[str, np.ndarray] = {
        "sequence_action": panel_sequence_action.astype(np.float32),
        "consensus_g": consensus["g"].astype(np.float32),
        "consensus_c": consensus["c"].astype(np.float32),
        "consensus_u": consensus["u"].astype(np.float32),
        "consensus_a": consensus["a"].astype(np.float32),
        "agreement_gated_u": gated_u.astype(np.float32),
    }
    for teacher in config.decomposition.teacher_ids:
        for name in ("a", "g", "c", "u"):
            matrix_values[f"{teacher}_{name}"] = scaled[teacher][name].astype(np.float32)
    np.savez_compressed(matrices_path, **matrix_values)

    tables = {
        "training_selection": pd.DataFrame(selection_rows),
        "training_summary": pd.DataFrame(training_rows),
        "variant_components": components,
        "method_registry": method_registry,
        "domain_metrics": domain_metrics,
        "component_margins": component_margins,
        "component_margin_summary": component_summary,
        "u_margin_summary": u_summary,
        "shuffle_domain_metrics": shuffle["domain"],
        "shuffle_margin_summary": shuffle["summary"],
        "teacher_agreement_positions": agreement["position"],
        "teacher_agreement_domains": agreement["domain"],
        "teacher_agreement_summary": agreement["summary"],
        "subgroup_domain_margins": subgroup["domain"],
        "subgroup_margin_summary": subgroup["summary"],
        "routing_diagnostic_domains": routing["domain"],
        "routing_diagnostic_summary": routing["summary"],
        "quality_controls": quality_controls,
        "gate_checks": gate_checks,
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
            "central_estimand": "paired_structure_U_increment_after_global_G_and_sequence_C",
            "training_data": "CATH_development_teacher_actions_only",
            "panel_stability_labels_used_for_training_or_model_selection": False,
            "teachers": list(config.decomposition.teacher_ids),
            "primary_population": PRIMARY_POPULATION,
            "replication_population": REPLICATION_POPULATION,
            "replication_is_sparse": True,
            "counterfactuals_used": False,
            "artifacts": [table_manifest(paths[name], table) for name, table in tables.items()],
            "decomposition_matrices": str(matrices_path),
            "final_decision": str(decision["decision"].iloc[0]),
            "registered_route": "PIVOT_SELECTIVE_STRUCTURE_CONDITIONED",
            "selective_routing": "NOT_YET_ESTABLISHED",
            "selective_routing_authorized": False,
        },
    )
    return {
        **paths,
        "manifest": manifest_path,
        "matrices": matrices_path,
        "decision": str(decision["decision"].iloc[0]),
        "confirmed": bool(decision["unique_action_confirmed"].iloc[0]),
    }


def _require_locked_inputs(config: ActionValidationStudyConfig) -> None:
    lock_path = config.paths.run_dir / "protocol_lock.json"
    if not lock_path.exists():
        raise FileNotFoundError("action-validation study protocol lock is missing")
    lock = read_json(lock_path)
    if lock.get("status") != "FROZEN_BEFORE_ACTION_VALIDATION_PANEL_MODEL_SCORING":
        raise RuntimeError("action-validation study protocol is not in the frozen state")
    required = [
        config.paths.run_dir / "teacher_scores" / "scores.parquet",
        config.paths.storage_dir / "representations" / "carp_640M" / "manifest.json",
        config.paths.storage_dir / "representations" / "esm2_150M" / "manifest.json",
        config.paths.cath_queries,
        config.paths.cath_teacher_scores,
        config.paths.cath_mif_scores,
        config.paths.cath_carp_store / "manifest.json",
        config.paths.cath_esm2_store / "manifest.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"action-validation study scoring inputs are missing: {missing}")


def _training_artifacts(config: ActionValidationStudyConfig) -> dict[str, Any]:
    metadata = pd.read_parquet(config.paths.cath_queries).reset_index(drop=True)
    carp = _aligned_store(config.paths.cath_carp_store, metadata, "representations.npy").astype(
        np.float32
    )
    sequence_logp = _aligned_store(
        config.paths.cath_esm2_store, metadata, "log_probabilities.npy"
    ).astype(float)
    wild = metadata["native_aa"].map(AA_TO_INDEX).to_numpy(dtype=int)
    sequence_action = _anchor(sequence_logp, wild)
    features = _context_features(
        metadata,
        carp,
        sequence_logp,
        wild_type_column="native_aa",
        radius=config.decomposition.context_radius,
    )
    cached = pd.read_parquet(config.paths.cath_teacher_scores)
    plain_mif = pd.read_parquet(config.paths.cath_mif_scores)
    actions = {
        "mif": _anchor(_aligned_teacher(plain_mif, metadata, "mif"), wild),
        "esm_if1": _anchor(_aligned_teacher(cached, metadata, "esm_if1"), wild),
        "proteinmpnn": _anchor(_aligned_teacher(cached, metadata, "proteinmpnn"), wild),
    }
    return {
        "metadata": metadata,
        "features": features,
        "wild": wild,
        "sequence_action": sequence_action,
        "actions": actions,
    }


def _fit_teacher_decomposition(
    teacher: str,
    artifacts: dict[str, Any],
    panel_features: np.ndarray,
    panel_wild: np.ndarray,
    panel_action: np.ndarray,
    config: ActionValidationStudyConfig,
) -> dict[str, Any]:
    metadata = artifacts["metadata"]
    features = artifacts["features"]
    wild = artifacts["wild"]
    action = artifacts["actions"][teacher]
    train = np.flatnonzero(
        metadata["observability_split"].eq(config.decomposition.training_split).to_numpy()
    )
    validation = np.flatnonzero(
        metadata["observability_split"].eq(config.decomposition.selection_split).to_numpy()
    )
    final = np.flatnonzero(
        metadata["observability_split"].isin(config.decomposition.final_training_splits).to_numpy()
    )
    locked = np.flatnonzero(metadata["observability_split"].eq("locked_test").to_numpy())

    g_validation = _global_component(action[train], wild[train], wild[validation])
    target_train = _anchor(
        action[train] - _global_component(action[train], wild[train], wild[train]),
        wild[train],
    )
    scaler = StandardScaler().fit(features[train])
    x_train = scaler.transform(features[train])
    x_validation = scaler.transform(features[validation])
    max_rank = min(max(config.decomposition.rrr_ranks), target_train.shape[1])
    basis = PCA(n_components=max_rank, svd_solver="full").fit(target_train)
    latent_train = basis.transform(target_train)
    selection_rows = []
    candidates: list[tuple[float, int, float]] = []
    for rank in config.decomposition.rrr_ranks:
        effective_rank = min(int(rank), max_rank)
        for alpha in config.decomposition.ridge_alphas:
            ridge = Ridge(alpha=float(alpha), solver="lsqr").fit(
                x_train, latent_train[:, :effective_rank]
            )
            latent = ridge.predict(x_validation)
            prediction = basis.mean_ + latent @ basis.components_[:effective_rank]
            prediction = _anchor(prediction, wild[validation])
            observed = _anchor(action[validation] - g_validation, wild[validation])
            rmse = _anchored_rmse(prediction, observed, wild[validation])
            selection_rows.append(
                {
                    "teacher_id": teacher,
                    "rank": int(rank),
                    "effective_rank": effective_rank,
                    "ridge_alpha": float(alpha),
                    "selection_metric": config.decomposition.model_selection_metric,
                    "validation_rmse": rmse,
                    "training_rows": int(len(train)),
                    "validation_rows": int(len(validation)),
                    "stability_labels_used": False,
                }
            )
            candidates.append((rmse, int(rank), float(alpha)))
    best_rmse, best_rank, best_alpha = min(candidates, key=lambda row: (row[0], row[1], -row[2]))

    g_panel = _global_component(action[final], wild[final], panel_wild)
    g_final = _global_component(action[final], wild[final], wild[final])
    target_final = _anchor(action[final] - g_final, wild[final])
    model = _fit_rrr(features[final], target_final, best_rank, best_alpha)
    c_panel = _anchor(_predict_rrr(model, panel_features), panel_wild)
    u_panel = _anchor(panel_action - g_panel - c_panel, panel_wild)

    g_locked = _global_component(action[final], wild[final], wild[locked])
    c_locked = _anchor(_predict_rrr(model, features[locked]), wild[locked])
    u_locked = _anchor(action[locked] - g_locked - c_locked, wild[locked])
    scale = _action_rms(artifacts["sequence_action"][final], wild[final]) / max(
        _action_rms(action[final], wild[final]), 1e-12
    )
    return {
        "panel": {
            "a": panel_action,
            "g": g_panel,
            "c": c_panel,
            "u": u_panel,
            "scale": scale,
        },
        "selection_rows": selection_rows,
        "training_row": {
            "teacher_id": teacher,
            "selected_rank": best_rank,
            "selected_ridge_alpha": best_alpha,
            "validation_rmse": best_rmse,
            "final_training_rows": int(len(final)),
            "final_training_domains": int(metadata.iloc[final]["domain_id"].nunique()),
            "locked_test_rows": int(len(locked)),
            "locked_test_gc_rmse": _anchored_rmse(
                _anchor(g_locked + c_locked, wild[locked]), action[locked], wild[locked]
            ),
            "locked_test_u_rms": _action_rms(u_locked, wild[locked]),
            "teacher_action_rms": _action_rms(action[final], wild[final]),
            "sequence_action_rms": _action_rms(artifacts["sequence_action"][final], wild[final]),
            "outcome_free_teacher_scale": scale,
            "stability_labels_used": False,
        },
    }


def _fit_rrr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    rank: int,
    alpha: float,
) -> dict[str, Any]:
    scaler = StandardScaler().fit(x_train)
    train = scaler.transform(x_train)
    effective_rank = min(int(rank), y_train.shape[1])
    basis = PCA(n_components=effective_rank, svd_solver="full").fit(y_train)
    model = Ridge(alpha=float(alpha), solver="lsqr").fit(train, basis.transform(y_train))
    return {"scaler": scaler, "basis": basis, "model": model}


def _predict_rrr(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    values = model["scaler"].transform(features)
    return model["basis"].inverse_transform(model["model"].predict(values))


def _global_component(
    training_action: np.ndarray,
    training_wild: np.ndarray,
    target_wild: np.ndarray,
) -> np.ndarray:
    global_mean = np.asarray(training_action, dtype=float).mean(axis=0)
    lookup = {}
    for index, aa in enumerate(AA_ALPHABET):
        selected = np.asarray(training_action)[training_wild == index]
        lookup[aa] = selected.mean(axis=0) if len(selected) else global_mean
    result = np.stack([lookup[AA_ALPHABET[index]] for index in target_wild])
    return _anchor(result, target_wild)


def _context_features(
    metadata: pd.DataFrame,
    carp: np.ndarray,
    sequence_logp: np.ndarray,
    *,
    wild_type_column: str,
    radius: int,
) -> np.ndarray:
    wild = metadata[wild_type_column].map(AA_TO_INDEX).to_numpy(dtype=int)
    sequence_action = _anchor(sequence_logp, wild)
    probability = np.exp(normalize_log_probabilities(sequence_logp))
    entropy = -np.sum(probability * np.log(np.maximum(probability, 1e-12)), axis=1)[:, None]
    wild_one_hot = np.eye(len(AA_ALPHABET), dtype=np.float32)[wild]
    local = np.zeros((len(metadata), len(AA_ALPHABET)), dtype=np.float32)
    position_fraction = np.zeros((len(metadata), 1), dtype=np.float32)
    lengths = np.zeros((len(metadata), 1), dtype=np.float32)
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
        lengths[row_index, 0] = len(sequence)
    return np.concatenate(
        [
            np.asarray(carp, dtype=np.float32),
            sequence_action.astype(np.float32),
            entropy.astype(np.float32),
            wild_one_hot,
            local,
            position_fraction,
            lengths,
        ],
        axis=1,
    )


def _aligned_store(directory: Path, metadata: pd.DataFrame, filename: str) -> np.ndarray:
    keys = pd.read_parquet(directory / "keys.parquet").copy()
    keys["array_row"] = np.arange(len(keys), dtype=int)
    columns = ["state_id", "domain_id", "position"]
    aligned = metadata[columns].merge(keys, on=columns, validate="one_to_one")
    if len(aligned) != len(metadata):
        raise ValueError(f"representation store lacks query coverage: {directory}")
    array = np.load(directory / filename, mmap_mode="r")
    result = np.asarray(array[aligned["array_row"].to_numpy(dtype=int)])
    if not np.isfinite(result).all():
        raise ValueError(f"representation store contains non-finite values: {directory}")
    return result


def _aligned_teacher(
    scores: pd.DataFrame,
    metadata: pd.DataFrame,
    teacher: str,
) -> np.ndarray:
    selected = scores.loc[scores["teacher_id"].eq(teacher) & scores["structure_role"].eq("paired")]
    keys = ["state_id", "domain_id", "position"]
    selected = selected[[*keys, *logp_columns()]]
    if selected.duplicated(keys).any():
        raise ValueError(f"duplicate paired score rows for {teacher}")
    aligned = metadata[keys].merge(selected, on=keys, validate="one_to_one")
    if len(aligned) != len(metadata):
        raise ValueError(f"teacher {teacher} lacks query coverage: {len(aligned)}/{len(metadata)}")
    return normalize_log_probabilities(aligned[logp_columns()].to_numpy(dtype=float))


def _anchor(values: np.ndarray, wild: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values - values[np.arange(len(values)), wild][:, None]


def _action_rms(values: np.ndarray, wild: np.ndarray) -> float:
    mask = np.ones(np.asarray(values).shape, dtype=bool)
    mask[np.arange(len(mask)), wild] = False
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float)[mask] ** 2)))


def _anchored_rmse(predicted: np.ndarray, observed: np.ndarray, wild: np.ndarray) -> float:
    difference = _anchor(predicted, wild) - _anchor(observed, wild)
    mask = np.ones(difference.shape, dtype=bool)
    mask[np.arange(len(mask)), wild] = False
    return float(np.sqrt(np.mean(difference[mask] ** 2)))


def _teacher_agreement(
    queries: pd.DataFrame,
    domains: pd.DataFrame,
    scaled: dict[str, dict[str, np.ndarray]],
    config: ActionValidationStudyConfig,
) -> dict[str, pd.DataFrame]:
    pairs = list(combinations(config.decomposition.teacher_ids, 2))
    metadata = queries[["domain_id", "position"]].merge(
        domains[["domain_id", "stratum", "evaluation_population"]],
        on="domain_id",
        validate="many_to_one",
    )
    rows = metadata.copy()
    columns = []
    for left, right in pairs:
        column = f"{left}_vs_{right}_u_spearman"
        rows[column] = _rowwise_spearman(scaled[left]["u"], scaled[right]["u"])
        columns.append(column)
    rows["median_pairwise_u_spearman"] = rows[columns].median(axis=1)
    domain = (
        rows.groupby(["domain_id", "stratum", "evaluation_population"], observed=True)[
            [*columns, "median_pairwise_u_spearman"]
        ]
        .mean()
        .reset_index()
    )
    summaries = []
    for population_index, (population, frame) in enumerate(
        domain.groupby("evaluation_population", sort=True, observed=True)
    ):
        for column_index, column in enumerate([*columns, "median_pairwise_u_spearman"]):
            summaries.append(
                {
                    "evaluation_population": population,
                    "metric": column,
                    **stratified_domain_bootstrap(
                        frame,
                        column,
                        replicates=config.inference.bootstrap_replicates,
                        confidence_level=config.inference.confidence_level,
                        seed=config.seed + 500_000 + population_index * 100 + column_index,
                    ),
                }
            )
    return {"position": rows, "domain": domain, "summary": pd.DataFrame(summaries)}


def _rowwise_spearman(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_rank = rankdata(left, axis=1)
    right_rank = rankdata(right, axis=1)
    left_rank -= left_rank.mean(axis=1, keepdims=True)
    right_rank -= right_rank.mean(axis=1, keepdims=True)
    numerator = np.sum(left_rank * right_rank, axis=1)
    denominator = np.linalg.norm(left_rank, axis=1) * np.linalg.norm(right_rank, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(numerator), np.nan, dtype=float),
        where=denominator > 0,
    )


def _agreement_gated_u(
    consensus_u: np.ndarray,
    scaled: dict[str, dict[str, np.ndarray]],
    agreement_position: pd.DataFrame,
    config: ActionValidationStudyConfig,
) -> tuple[np.ndarray, np.ndarray]:
    stack = np.stack([scaled[teacher]["u"] for teacher in config.decomposition.teacher_ids])
    same_sign = (stack > 0).all(axis=0) | (stack < 0).all(axis=0)
    reliable_position = agreement_position["median_pairwise_u_spearman"].to_numpy(dtype=float) >= (
        config.inference.agreement_gate_minimum_position_spearman
    )
    gate = same_sign & reliable_position[:, None]
    return consensus_u * gate, gate


def _variant_components(
    variants: pd.DataFrame,
    domains: pd.DataFrame,
    residues: pd.DataFrame,
    queries: pd.DataFrame,
    sequence_action: np.ndarray,
    scaled: dict[str, dict[str, np.ndarray]],
    consensus: dict[str, np.ndarray],
    gated_u: np.ndarray,
    gate: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    index = queries[["state_id", "domain_id", "position"]].copy()
    index["query_row"] = np.arange(len(index), dtype=int)
    result = variants.merge(
        index.drop(columns="state_id"),
        on=["domain_id", "position"],
        validate="many_to_one",
    ).merge(
        domains[["domain_id", "evaluation_population"]],
        on="domain_id",
        validate="many_to_one",
        suffixes=("", "_domain"),
    )
    if "evaluation_population_domain" in result:
        if not result["evaluation_population"].equals(result["evaluation_population_domain"]):
            raise ValueError("variant/domain evaluation populations disagree")
        result = result.drop(columns="evaluation_population_domain")
    environment = residues[
        ["domain_id", "position", "burial", "secondary_structure", "contact_class"]
    ]
    result = result.merge(environment, on=["domain_id", "position"], validate="many_to_one")
    query_rows = result["query_row"].to_numpy(dtype=int)
    mutant = result["mutant"].map(AA_TO_INDEX).to_numpy(dtype=int)
    result["variant_row"] = np.arange(len(result), dtype=int)
    result["sequence_action"] = sequence_action[query_rows, mutant]
    for teacher, values in scaled.items():
        for name in ("a", "g", "c", "u"):
            result[f"{teacher}_{name}_action"] = values[name][query_rows, mutant]
    for name in ("a", "g", "c", "u"):
        result[f"consensus_{name}_action"] = consensus[name][query_rows, mutant]
    result["agreement_gated_u_action"] = gated_u[query_rows, mutant]
    result["agreement_gate_active"] = gate[query_rows, mutant]
    result["substitution_class"] = [
        _substitution_class(wild, mutation)
        for wild, mutation in zip(result["wild_type"], result["mutant"], strict=True)
    ]
    result["gly_pro_boundary"] = np.where(
        result["wild_type"].isin(["G", "P"]) | result["mutant"].isin(["G", "P"]),
        "involves_glycine_or_proline",
        "other_substitutions",
    )
    return result.drop(columns="query_row"), query_rows, mutant


def _method_predictions(
    components: pd.DataFrame,
    scaled: dict[str, dict[str, np.ndarray]],
    consensus: dict[str, np.ndarray],
    sequence_action: np.ndarray,
    query_rows: np.ndarray,
    mutant: np.ndarray,
    gated_u: np.ndarray,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    sequence = components["sequence_action"].to_numpy(dtype=float)
    methods = {SEQUENCE_METHOD: sequence}
    rows = [
        {
            "method": SEQUENCE_METHOD,
            "teacher_id": "sequence",
            "stage": "sequence",
            "comparator": "",
            "test_time_structure": False,
        }
    ]
    for teacher, values in [*scaled.items(), (CONSENSUS, consensus)]:
        g = values["g"][query_rows, mutant]
        c = values["c"][query_rows, mutant]
        u = values["u"][query_rows, mutant]
        names = {
            "plus_g": sequence + g,
            "plus_gc": sequence + g + c,
            "plus_gcu": sequence + g + c + u,
        }
        comparators = {
            "plus_g": SEQUENCE_METHOD,
            "plus_gc": f"{teacher}__plus_g",
            "plus_gcu": f"{teacher}__plus_gc",
        }
        for stage, prediction in names.items():
            method = f"{teacher}__{stage}"
            methods[method] = prediction
            rows.append(
                {
                    "method": method,
                    "teacher_id": teacher,
                    "stage": stage,
                    "comparator": comparators[stage],
                    "test_time_structure": stage == "plus_gcu",
                }
            )
    gated_method = "consensus__plus_gc_agreement_gated_u"
    methods[gated_method] = (
        sequence
        + consensus["g"][query_rows, mutant]
        + consensus["c"][query_rows, mutant]
        + gated_u[query_rows, mutant]
    )
    rows.append(
        {
            "method": gated_method,
            "teacher_id": CONSENSUS,
            "stage": "agreement_gated_u_diagnostic",
            "comparator": "consensus__plus_gc",
            "test_time_structure": True,
        }
    )
    return methods, pd.DataFrame(rows)


def _domain_metrics(
    components: pd.DataFrame,
    methods: dict[str, np.ndarray],
    top_fraction: float,
) -> pd.DataFrame:
    rows = []
    for domain_id, indices_value in components.groupby("domain_id", sort=True).indices.items():
        indices = np.asarray(indices_value, dtype=int)
        frame = components.iloc[indices]
        observed = frame["effect"].to_numpy(dtype=float)
        k = max(1, int(np.ceil(len(frame) * top_fraction)))
        metadata = {
            "domain_id": domain_id,
            "evaluation_population": str(frame["evaluation_population"].iloc[0]),
            "stratum": str(frame["stratum"].iloc[0]),
            "n_variants": int(len(frame)),
        }
        for method, values in methods.items():
            predicted = np.asarray(values, dtype=float)[indices]
            rows.append(
                {
                    **metadata,
                    "method": method,
                    "spearman": _spearman(predicted, observed),
                    "full_ndcg": _ndcg(predicted, observed),
                    "ndcg_at_10_percent": _ndcg(predicted, observed, k=k),
                    "stabilizing_top_10_percent_recall": _stabilizing_recall(
                        predicted, observed, k
                    ),
                }
            )
    return pd.DataFrame(rows)


def _component_margins(metrics: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    keys = ["domain_id", "evaluation_population", "stratum", "n_variants"]
    frames = []
    lookup = registry.set_index("method")
    for method in registry.loc[registry["comparator"].ne(""), "method"]:
        comparator = str(lookup.loc[method, "comparator"])
        candidate = metrics.loc[metrics["method"].eq(method), [*keys, *METRICS]]
        baseline = metrics.loc[metrics["method"].eq(comparator), [*keys, *METRICS]].rename(
            columns={metric: f"comparator_{metric}" for metric in METRICS}
        )
        merged = candidate.merge(baseline, on=keys, validate="one_to_one")
        merged["method"] = method
        merged["comparator"] = comparator
        merged["teacher_id"] = str(lookup.loc[method, "teacher_id"])
        merged["stage"] = str(lookup.loc[method, "stage"])
        for metric in METRICS:
            merged[f"{metric}_margin"] = merged[metric] - merged[f"comparator_{metric}"]
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def _summarize_u_margins(
    margins: pd.DataFrame, config: ActionValidationStudyConfig
) -> pd.DataFrame:
    selected = margins.loc[margins["stage"].eq("plus_gcu")]
    rows = []
    for teacher_index, (teacher, teacher_frame) in enumerate(
        selected.groupby("teacher_id", sort=True, observed=True)
    ):
        for population_index, (population, population_frame) in enumerate(
            teacher_frame.groupby("evaluation_population", sort=True, observed=True)
        ):
            scopes = [("all", population_frame)]
            scopes.extend(
                (str(stratum), group)
                for stratum, group in population_frame.groupby("stratum", sort=True)
            )
            for scope_index, (stratum, frame) in enumerate(scopes):
                for metric_index, metric in enumerate(METRICS):
                    column = f"{metric}_margin"
                    rows.append(
                        {
                            "teacher_id": teacher,
                            "evaluation_population": population,
                            "stratum": stratum,
                            "metric": column,
                            **stratified_domain_bootstrap(
                                frame,
                                column,
                                replicates=config.inference.bootstrap_replicates,
                                confidence_level=config.inference.confidence_level,
                                seed=(
                                    config.seed
                                    + 100_000
                                    + teacher_index * 1000
                                    + population_index * 100
                                    + scope_index * 10
                                    + metric_index
                                ),
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def _summarize_component_margins(
    margins: pd.DataFrame, config: ActionValidationStudyConfig
) -> pd.DataFrame:
    rows = []
    grouped = margins.groupby(
        ["teacher_id", "stage", "evaluation_population"],
        sort=True,
        observed=True,
    )
    for group_index, (keys, population_frame) in enumerate(grouped):
        teacher, stage, population = keys
        scopes = [("all", population_frame)]
        scopes.extend(
            (str(stratum), group)
            for stratum, group in population_frame.groupby("stratum", sort=True)
        )
        for scope_index, (stratum, frame) in enumerate(scopes):
            for metric_index, metric in enumerate(METRICS):
                column = f"{metric}_margin"
                rows.append(
                    {
                        "teacher_id": teacher,
                        "stage": stage,
                        "evaluation_population": population,
                        "stratum": stratum,
                        "metric": column,
                        **stratified_domain_bootstrap(
                            frame,
                            column,
                            replicates=config.inference.bootstrap_replicates,
                            confidence_level=config.inference.confidence_level,
                            seed=(
                                config.seed
                                + 300_000
                                + group_index * 100
                                + scope_index * 10
                                + metric_index
                            ),
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _shuffled_u_control(
    components: pd.DataFrame,
    queries: pd.DataFrame,
    sequence_action: np.ndarray,
    consensus: dict[str, np.ndarray],
    query_rows: np.ndarray,
    mutant: np.ndarray,
    observed_metrics: pd.DataFrame,
    config: ActionValidationStudyConfig,
) -> dict[str, pd.DataFrame]:
    sequence = components["sequence_action"].to_numpy(dtype=float)
    gc = sequence + consensus["g"][query_rows, mutant] + consensus["c"][query_rows, mutant]
    query_wild = queries["wild_type"].map(AA_TO_INDEX).to_numpy(dtype=int)
    group_indices = queries.groupby("domain_id", sort=True).indices
    frames = []
    for repeat in range(config.decomposition.shuffled_u_repeats):
        rng = np.random.default_rng(config.seed + 700_000 + repeat)
        shuffled = np.empty_like(consensus["u"])
        for indices_value in group_indices.values():
            indices = np.asarray(indices_value, dtype=int)
            shuffled[indices] = consensus["u"][rng.permutation(indices)]
        shuffled = _anchor(shuffled, query_wild)
        prediction = gc + shuffled[query_rows, mutant]
        metrics = _domain_metrics(
            components,
            {"consensus__plus_gc_position_shuffled_u": prediction},
            config.inference.top_fraction,
        )
        metrics["repeat"] = repeat
        frames.append(metrics)
    repeated = pd.concat(frames, ignore_index=True)
    averaged = (
        repeated.groupby(
            ["domain_id", "evaluation_population", "stratum", "n_variants"],
            observed=True,
        )[[*METRICS]]
        .mean()
        .reset_index()
    )
    actual = observed_metrics.loc[
        observed_metrics["method"].eq("consensus__plus_gcu"),
        ["domain_id", "evaluation_population", "stratum", "n_variants", *METRICS],
    ]
    margin = actual.merge(
        averaged,
        on=["domain_id", "evaluation_population", "stratum", "n_variants"],
        suffixes=("_actual", "_shuffled"),
        validate="one_to_one",
    )
    for metric in METRICS:
        margin[f"{metric}_actual_minus_shuffled"] = (
            margin[f"{metric}_actual"] - margin[f"{metric}_shuffled"]
        )
    summary_rows = []
    for population_index, (population, frame) in enumerate(
        margin.groupby("evaluation_population", sort=True, observed=True)
    ):
        for metric_index, metric in enumerate(METRICS):
            column = f"{metric}_actual_minus_shuffled"
            summary_rows.append(
                {
                    "evaluation_population": population,
                    "metric": column,
                    **stratified_domain_bootstrap(
                        frame,
                        column,
                        replicates=config.inference.bootstrap_replicates,
                        confidence_level=config.inference.confidence_level,
                        seed=config.seed + 800_000 + population_index * 100 + metric_index,
                    ),
                }
            )
    return {"domain": margin, "summary": pd.DataFrame(summary_rows)}


def _subgroup_analysis(
    components: pd.DataFrame,
    methods: dict[str, np.ndarray],
    registry: pd.DataFrame,
    config: ActionValidationStudyConfig,
) -> dict[str, pd.DataFrame]:
    dimensions = ["gly_pro_boundary", "burial", "secondary_structure", "contact_class"]
    teachers = [*config.decomposition.teacher_ids, CONSENSUS]
    rows = []
    for dimension in dimensions:
        for level, level_frame in components.groupby(dimension, sort=True, observed=True):
            domain_groups = level_frame.groupby("domain_id", sort=True).indices
            for domain_id, indices_value in domain_groups.items():
                indices = level_frame.iloc[np.asarray(indices_value, dtype=int)][
                    "variant_row"
                ].to_numpy(dtype=int)
                if len(indices) < 5:
                    continue
                observed = components.iloc[indices]["effect"].to_numpy(dtype=float)
                k = max(1, int(np.ceil(len(indices) * config.inference.top_fraction)))
                metadata = components.iloc[indices[0]]
                for teacher in teachers:
                    gc_method = f"{teacher}__plus_gc"
                    full_method = f"{teacher}__plus_gcu"
                    gc = methods[gc_method][indices]
                    full = methods[full_method][indices]
                    rows.append(
                        {
                            "dimension": dimension,
                            "level": str(level),
                            "domain_id": domain_id,
                            "evaluation_population": metadata.evaluation_population,
                            "stratum": metadata.stratum,
                            "teacher_id": teacher,
                            "n_variants": int(len(indices)),
                            "spearman_margin": _spearman(full, observed) - _spearman(gc, observed),
                            "ndcg_at_10_percent_margin": _ndcg(full, observed, k=k)
                            - _ndcg(gc, observed, k=k),
                        }
                    )
    domain = pd.DataFrame(rows)
    summaries = []
    if not domain.empty:
        grouped = domain.groupby(
            ["dimension", "level", "evaluation_population", "teacher_id"],
            sort=True,
            observed=True,
        )
        for group_index, (keys, frame) in enumerate(grouped):
            dimension, level, population, teacher = keys
            for metric_index, metric in enumerate(["spearman_margin", "ndcg_at_10_percent_margin"]):
                summaries.append(
                    {
                        "dimension": dimension,
                        "level": level,
                        "evaluation_population": population,
                        "teacher_id": teacher,
                        "metric": metric,
                        **stratified_domain_bootstrap(
                            frame,
                            metric,
                            replicates=config.inference.bootstrap_replicates,
                            confidence_level=config.inference.confidence_level,
                            seed=config.seed + 900_000 + group_index * 10 + metric_index,
                        ),
                    }
                )
    return {"domain": domain, "summary": pd.DataFrame(summaries)}


def _routing_diagnostic(
    metrics: pd.DataFrame,
    components: pd.DataFrame,
    config: ActionValidationStudyConfig,
) -> dict[str, pd.DataFrame]:
    keys = ["domain_id", "evaluation_population", "stratum", "n_variants"]
    methods = {
        "gc": "consensus__plus_gc",
        "gated": "consensus__plus_gc_agreement_gated_u",
        "full": "consensus__plus_gcu",
    }
    frames = []
    for label, method in methods.items():
        selected = metrics.loc[metrics["method"].eq(method), [*keys, *METRICS]].copy()
        selected = selected.rename(columns={metric: f"{label}_{metric}" for metric in METRICS})
        frames.append(selected)
    domain = frames[0]
    for frame in frames[1:]:
        domain = domain.merge(frame, on=keys, validate="one_to_one")
    for metric in METRICS:
        domain[f"gated_minus_gc_{metric}"] = domain[f"gated_{metric}"] - domain[f"gc_{metric}"]
        domain[f"gated_minus_full_{metric}"] = domain[f"gated_{metric}"] - domain[f"full_{metric}"]
    summary_rows = []
    for population_index, (population, frame) in enumerate(
        domain.groupby("evaluation_population", sort=True, observed=True)
    ):
        columns = [
            f"{comparison}_{metric}"
            for comparison in ("gated_minus_gc", "gated_minus_full")
            for metric in METRICS
        ]
        for column_index, column in enumerate(columns):
            summary_rows.append(
                {
                    "evaluation_population": population,
                    "metric": column,
                    **stratified_domain_bootstrap(
                        frame,
                        column,
                        replicates=config.inference.bootstrap_replicates,
                        confidence_level=config.inference.confidence_level,
                        seed=config.seed + 950_000 + population_index * 100 + column_index,
                    ),
                }
            )
        selected_variants = components.loc[
            components["evaluation_population"].eq(population), "agreement_gate_active"
        ]
        summary_rows.append(
            {
                "evaluation_population": population,
                "metric": "variant_gate_active_fraction",
                "estimate": float(selected_variants.mean()),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "positive_domain_fraction": float("nan"),
                "positive_domains": 0,
                "negative_domains": 0,
                "zero_domains": 0,
                "n_domains": int(frame["domain_id"].nunique()),
                "leave_one_domain_out_min": float("nan"),
                "leave_one_domain_out_max": float("nan"),
            }
        )
    return {"domain": domain, "summary": pd.DataFrame(summary_rows)}


def _quality_controls(
    panel_actions: dict[str, np.ndarray],
    decompositions: dict[str, dict[str, np.ndarray | float]],
    scaled: dict[str, dict[str, np.ndarray]],
    consensus: dict[str, np.ndarray],
    sequence_action: np.ndarray,
    methods: dict[str, np.ndarray],
    components: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for teacher, action in panel_actions.items():
        values = decompositions[teacher]
        reconstructed = np.asarray(values["g"]) + np.asarray(values["c"]) + np.asarray(values["u"])
        rows.append(
            {
                "check": f"{teacher}_raw_decomposition_identity",
                "estimate": float(np.max(np.abs(action - reconstructed))),
                "threshold": 1e-8,
                "passed": bool(np.max(np.abs(action - reconstructed)) <= 1e-8),
            }
        )
        scaled_reconstructed = scaled[teacher]["g"] + scaled[teacher]["c"] + scaled[teacher]["u"]
        rows.append(
            {
                "check": f"{teacher}_scaled_decomposition_identity",
                "estimate": float(np.max(np.abs(scaled[teacher]["a"] - scaled_reconstructed))),
                "threshold": 1e-8,
                "passed": bool(np.max(np.abs(scaled[teacher]["a"] - scaled_reconstructed)) <= 1e-8),
            }
        )
    consensus_error = np.max(
        np.abs(consensus["a"] - consensus["g"] - consensus["c"] - consensus["u"])
    )
    rows.append(
        {
            "check": "consensus_decomposition_identity",
            "estimate": float(consensus_error),
            "threshold": 1e-8,
            "passed": bool(consensus_error <= 1e-8),
        }
    )
    rows.append(
        {
            "check": "consensus_full_score_equals_sequence_plus_calibrated_A",
            "estimate": float(
                np.max(
                    np.abs(
                        methods["consensus__plus_gcu"]
                        - (
                            components["sequence_action"].to_numpy(dtype=float)
                            + components["consensus_a_action"].to_numpy(dtype=float)
                        )
                    )
                )
            ),
            "threshold": 1e-8,
            "passed": bool(
                np.max(
                    np.abs(
                        methods["consensus__plus_gcu"]
                        - (
                            components["sequence_action"].to_numpy(dtype=float)
                            + components["consensus_a_action"].to_numpy(dtype=float)
                        )
                    )
                )
                <= 1e-8
            ),
        }
    )
    finite = all(np.isfinite(values).all() for values in methods.values())
    rows.append(
        {
            "check": "all_variant_predictions_finite",
            "estimate": float(finite),
            "threshold": 1.0,
            "passed": finite,
        }
    )
    return pd.DataFrame(rows)


def _decision(
    summary: pd.DataFrame,
    shuffle_summary: pd.DataFrame,
    config: ActionValidationStudyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def row(teacher: str, population: str, stratum: str, metric: str) -> pd.Series:
        selected = summary.loc[
            summary["teacher_id"].eq(teacher)
            & summary["evaluation_population"].eq(population)
            & summary["stratum"].eq(stratum)
            & summary["metric"].eq(metric)
        ]
        if len(selected) != 1:
            raise ValueError(
                f"missing unique U summary row: {teacher}/{population}/{stratum}/{metric}"
            )
        return selected.iloc[0]

    dense_spearman = row(CONSENSUS, PRIMARY_POPULATION, "all", "spearman_margin")
    dense_ndcg = row(CONSENSUS, PRIMARY_POPULATION, "all", "ndcg_at_10_percent_margin")
    natural = row(CONSENSUS, PRIMARY_POPULATION, "natural", "spearman_margin")
    de_novo = row(CONSENSUS, PRIMARY_POPULATION, "de_novo", "spearman_margin")
    s669 = row(CONSENSUS, REPLICATION_POPULATION, "all", "spearman_margin")
    teacher_passes = {}
    for teacher in config.decomposition.teacher_ids:
        teacher_row = row(teacher, PRIMARY_POPULATION, "all", "spearman_margin")
        teacher_passes[teacher] = bool(teacher_row["ci_low"] > 0)
    shuffled = shuffle_summary.loc[
        shuffle_summary["evaluation_population"].eq(PRIMARY_POPULATION)
        & shuffle_summary["metric"].eq("spearman_actual_minus_shuffled")
    ]
    if len(shuffled) != 1:
        raise ValueError("missing dense shuffled-U margin summary")
    shuffled = shuffled.iloc[0]
    checks = [
        {
            "gate": "consensus_dense_spearman_ci_lower_positive",
            "estimate": float(dense_spearman["ci_low"]),
            "threshold": 0.0,
            "passed": bool(dense_spearman["ci_low"] > 0),
        },
        {
            "gate": "consensus_dense_ndcg10_ci_lower_positive",
            "estimate": float(dense_ndcg["ci_low"]),
            "threshold": 0.0,
            "passed": bool(dense_ndcg["ci_low"] > 0),
        },
        {
            "gate": "minimum_teacher_replications",
            "estimate": float(sum(teacher_passes.values())),
            "threshold": float(config.inference.minimum_teacher_replications),
            "passed": sum(teacher_passes.values()) >= config.inference.minimum_teacher_replications,
        },
        {
            "gate": "natural_dense_spearman_point_positive",
            "estimate": float(natural["estimate"]),
            "threshold": 0.0,
            "passed": bool(natural["estimate"] > 0),
        },
        {
            "gate": "de_novo_dense_spearman_point_positive",
            "estimate": float(de_novo["estimate"]),
            "threshold": 0.0,
            "passed": bool(de_novo["estimate"] > 0),
        },
        {
            "gate": "s669_spearman_point_positive",
            "estimate": float(s669["estimate"]),
            "threshold": 0.0,
            "passed": bool(s669["estimate"] > 0),
        },
        {
            "gate": "s669_minimum_finite_domains",
            "estimate": float(s669["n_domains"]),
            "threshold": float(config.inference.minimum_s669_finite_domains),
            "passed": int(s669["n_domains"]) >= config.inference.minimum_s669_finite_domains,
        },
        {
            "gate": "s669_positive_domain_fraction",
            "estimate": float(s669["positive_domain_fraction"]),
            "threshold": config.inference.minimum_s669_positive_domain_fraction,
            "passed": float(s669["positive_domain_fraction"])
            >= config.inference.minimum_s669_positive_domain_fraction,
        },
        {
            "gate": "consensus_beats_position_shuffled_u",
            "estimate": float(shuffled["ci_low"]),
            "threshold": 0.0,
            "passed": bool(shuffled["ci_low"] > 0),
        },
    ]
    gates = pd.DataFrame(checks)
    confirmed = bool(gates["passed"].all())
    partial = bool(
        dense_spearman["ci_low"] > 0 or dense_ndcg["ci_low"] > 0 or any(teacher_passes.values())
    )
    if confirmed:
        decision = "MULTI_TEACHER_STRUCTURE_UNIQUE_ACTION_CONFIRMED"
        implementation = "CALIBRATED_PAIRED_STRUCTURE_CONDITIONED_WITH_CONFIRMED_UNIQUE_ACTION"
    elif partial:
        decision = "TEACHER_SPECIFIC_STRUCTURE_INCREMENT_WITHOUT_CONSENSUS"
        implementation = "CALIBRATED_PAIRED_STRUCTURE_CONDITIONED"
    else:
        decision = "NO_STABILITY_RELEVANT_UNIQUE_STRUCTURE_ACTION_AFTER_GC"
        implementation = "CALIBRATED_PAIRED_STRUCTURE_CONDITIONED"
    project = pd.DataFrame(
        [
            {
                "decision": decision,
                "unique_action_confirmed": confirmed,
                "partial_support": partial and not confirmed,
                "teacher_spearman_pass_count": int(sum(teacher_passes.values())),
                **{
                    f"{teacher}_spearman_passed": value for teacher, value in teacher_passes.items()
                },
                "consensus_dense_spearman_margin": float(dense_spearman["estimate"]),
                "consensus_dense_spearman_ci_low": float(dense_spearman["ci_low"]),
                "consensus_dense_ndcg10_margin": float(dense_ndcg["estimate"]),
                "consensus_dense_ndcg10_ci_low": float(dense_ndcg["ci_low"]),
                "consensus_s669_spearman_margin": float(s669["estimate"]),
                "consensus_s669_positive_domain_fraction": float(s669["positive_domain_fraction"]),
                "registered_route": "PIVOT_SELECTIVE_STRUCTURE_CONDITIONED",
                "currently_supported_implementation": implementation,
                "selective_routing": "NOT_YET_ESTABLISHED",
                "counterfactual_search_reopened": False,
                "selective_routing_authorized": False,
                "historical_counterfactuals_or_mechanisms_decision_modified": False,
            }
        ]
    )
    return gates, project


def _spearman(predicted: np.ndarray, observed: np.ndarray) -> float:
    if len(predicted) < 2 or np.ptp(predicted) == 0 or np.ptp(observed) == 0:
        return float("nan")
    value = spearmanr(predicted, observed).statistic
    return float(value) if np.isfinite(value) else float("nan")


def _ndcg(predicted: np.ndarray, observed: np.ndarray, k: int | None = None) -> float:
    relevance = observed - np.min(observed)
    if np.allclose(relevance, 0):
        return float("nan")
    return float(ndcg_score(relevance[None, :], predicted[None, :], k=k))


def _stabilizing_recall(predicted: np.ndarray, observed: np.ndarray, k: int) -> float:
    stabilizing = observed > 0
    count = int(stabilizing.sum())
    if count == 0:
        return float("nan")
    predicted_top = np.argpartition(predicted, -k)[-k:]
    return float(stabilizing[predicted_top].sum() / min(k, count))


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
