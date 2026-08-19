"""Frozen mechanism study counterfactual-validity and denoising evaluation."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import StandardScaler

from margin.attribution.metrics import (
    normalize_log_probabilities,
    rowwise_jsd,
    vector_spearman,
)
from margin.constants import AA_ALPHABET, AA_TO_INDEX
from margin.provenance import (
    runtime_manifest,
    sha256_file,
    table_manifest,
    write_json,
    write_parquet,
)
from margin.studies.counterfactuals.evaluation import (
    _aligned_scores,
    _aligned_store,
    _substitution_class,
    stratified_domain_bootstrap,
)
from margin.studies.counterfactuals.mechanism import read_aaindex1
from margin.studies.generalization.audit import load_features
from margin.studies.generalization.config import load_generalization_config
from margin.studies.generalization.dms import fit_rrr_predict
from margin.studies.generalization.targets import load_generalization_residual_dataset
from margin.studies.mechanisms.config import MechanismStudyConfig
from margin.studies.observability.targets import clr

SEQUENCE_METHOD = "sequence_only"
PAIRED_CONTROLS = (
    "mif_paired_only",
    "sequence_plus_mif_paired_alpha_1",
    "sequence_plus_mif_paired_variance_matched",
    "sequence_plus_mif_paired_minus_cath_substitution_background",
)
ROUTE_B_METHOD = "route_b_carp_rank16"
ROUTE_B_CONTROLS = (
    "route_b_global_wt_mutant_matrix",
    "route_b_grantham_aaindex_blosum_linear",
    "route_b_simple_sequence_context",
    "route_b_carp_context_shuffled",
    "route_b_target_conditionally_shuffled",
    "direct_legacy_pca_rank1",
    "direct_legacy_pca_rank3",
    "direct_legacy_pca_rank5",
    "direct_legacy_pca_rank16",
    "direct_legacy_rms_shrinkage",
)
PRIMARY_LEVELS = {
    "contact_deletion": "0.05",
    "smooth_coordinate": "0.25",
    "constrained_reassignment": "0.1",
    "matched_real_structure": "descriptor_matched",
}
METRICS = (
    "spearman",
    "full_ndcg",
    "ndcg_at_10_percent",
    "stabilizing_top_10_percent_recall",
    "calibration_slope",
)
INCREMENT_METRICS = (
    "spearman_increment",
    "full_ndcg_increment",
    "ndcg_at_10_percent_increment",
    "stabilizing_top_10_percent_recall_increment",
    "calibration_error_reduction",
)


def evaluate_mechanisms(config: MechanismStudyConfig) -> dict[str, Path | str | bool]:
    """Execute the locked dense-panel audit without changing the counterfactual study decision."""

    _assert_immutable_project_state(config)
    output = config.paths.run_dir / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    storage = config.paths.storage_dir / "evaluation"
    storage.mkdir(parents=True, exist_ok=True)
    panel = config.paths.run_dir / "panel"
    domains = pd.read_parquet(panel / "domains.parquet")
    variants = pd.read_parquet(panel / "variants.parquet")
    residues = pd.read_parquet(panel / "residues.parquet")
    queries = pd.read_parquet(panel / "query_rows.parquet")
    structures = pd.read_parquet(config.paths.run_dir / "mif_requests" / "structures.parquet")
    mif_scores = pd.read_parquet(config.paths.run_dir / "mif" / "scores.parquet")
    representation_root = config.paths.storage_dir / "representations"
    sequence_logp = _aligned_store(
        representation_root / "esm2_150M", queries, "log_probabilities.npy"
    ).astype(float)
    carp_features = _aligned_store(
        representation_root / "carp_640M", queries, "representations.npy"
    ).astype(np.float32)
    paired = _aligned_scores(
        mif_scores, queries, "paired", population="mechanisms_locked_dense_panel"
    )
    role_matrices = {
        str(role): _aligned_scores(
            mif_scores,
            queries,
            str(role),
            population="mechanisms_locked_dense_panel",
        )
        for role in structures["structure_role"].astype(str).unique()
        if role != "paired"
    }
    direct_by_role = {
        role: clr(paired - values)
        for role, values in role_matrices.items()
        if role != "rigid_transform_qc"
    }
    condition_residuals, condition_roles = _condition_ensembles(structures, direct_by_role)
    legacy_direct = direct_by_role["legacy_rewired_5"]

    training = _load_training_artifacts(config, queries, sequence_logp)
    paired_background = _conditional_mean_matrix(
        training["cath_paired"][training["train"]],
        training["cath_metadata"].iloc[training["train"]]["native_aa"].astype(str).to_numpy(),
        queries["wild_type"].astype(str).to_numpy(),
    )
    background_adjusted = clr(paired - paired_background)
    variance_alpha = _variance_matched_alpha(
        sequence_logp,
        paired,
        queries["wild_type"].astype(str).to_numpy(),
    )
    route_b, route_b_training = _route_b_controls(
        config,
        training,
        queries,
        carp_features,
        sequence_logp,
        legacy_direct,
    )

    components, query_rows, wild, mutant = _variant_index(variants, domains, residues, queries)
    sequence_effect = _matrix_action(sequence_logp, query_rows, wild, mutant)
    paired_effect = _matrix_action(paired, query_rows, wild, mutant)
    methods: dict[str, np.ndarray] = {
        SEQUENCE_METHOD: sequence_effect,
        "mif_paired_only": paired_effect,
        "sequence_plus_mif_paired_alpha_1": sequence_effect + paired_effect,
        "sequence_plus_mif_paired_variance_matched": (
            sequence_effect + variance_alpha * paired_effect
        ),
        "sequence_plus_mif_paired_minus_cath_substitution_background": (
            sequence_effect + _matrix_action(background_adjusted, query_rows, wild, mutant)
        ),
        "legacy_direct_contrast": (
            sequence_effect + _matrix_action(legacy_direct, query_rows, wild, mutant)
        ),
    }
    method_rows = [
        _method_row(SEQUENCE_METHOD, "baseline", test_time_structure=False),
        _method_row("mif_paired_only", "paired_only", test_time_structure=True),
        _method_row("sequence_plus_mif_paired_alpha_1", "paired_only", test_time_structure=True),
        _method_row(
            "sequence_plus_mif_paired_variance_matched",
            "paired_only",
            test_time_structure=True,
            alpha=variance_alpha,
        ),
        _method_row(
            "sequence_plus_mif_paired_minus_cath_substitution_background",
            "paired_only",
            test_time_structure=True,
        ),
        _method_row(
            "legacy_direct_contrast",
            "legacy_ood_contrast",
            test_time_structure=True,
            family="legacy_ood_rewiring",
            level="5_swaps_per_edge",
        ),
    ]
    condition_methods: dict[tuple[str, str], str] = {}
    for (family, level), matrix in sorted(condition_residuals.items()):
        method = f"contrast__{family}__{level}"
        condition_methods[(family, level)] = method
        methods[method] = sequence_effect + _matrix_action(matrix, query_rows, wild, mutant)
        method_rows.append(
            _method_row(
                method,
                "counterfactual_subtraction",
                test_time_structure=True,
                family=family,
                level=level,
                primary_condition=PRIMARY_LEVELS.get(family) == level,
            )
        )
    for method, matrix in route_b.items():
        methods[method] = sequence_effect + _matrix_action(matrix, query_rows, wild, mutant)
        method_rows.append(
            _method_row(
                method,
                "route_b" if method == ROUTE_B_METHOD else "route_b_control",
                test_time_structure=method.startswith("direct_"),
            )
        )
    method_table = pd.DataFrame(method_rows)

    domain_metrics = _domain_metrics(components, methods, config.inference.top_fraction)
    increments = _metric_increments(domain_metrics)
    increment_summary = _summarize_increments(increments, config)
    variant_predictions = _variant_predictions(components, methods)
    distribution = _distribution_audit(
        queries,
        domains,
        paired,
        role_matrices,
        structures,
    )
    seed_reliability = _seed_reliability(
        components,
        query_rows,
        wild,
        mutant,
        direct_by_role,
        structures,
    )
    condition_validity = _condition_validity(distribution["domain"], seed_reliability, config)
    seed_metrics = _seed_performance_metrics(
        components,
        sequence_effect,
        query_rows,
        wild,
        mutant,
        direct_by_role,
        structures,
        config.inference.top_fraction,
    )
    subtraction_margins, subtraction_margin_summary = _pairwise_margin_tables(
        domain_metrics,
        candidate_methods=list(condition_methods.values()),
        comparator_methods=list(PAIRED_CONTROLS),
        metrics=["spearman", "ndcg_at_10_percent"],
        config=config,
        seed_offset=700_000,
    )
    route_b_margins, route_b_margin_summary = _pairwise_margin_tables(
        domain_metrics,
        candidate_methods=[ROUTE_B_METHOD],
        comparator_methods=list(ROUTE_B_CONTROLS),
        metrics=["spearman", "ndcg_at_10_percent"],
        config=config,
        seed_offset=800_000,
    )
    family_assessment, decisions = _audit_decisions(
        condition_validity,
        increment_summary,
        subtraction_margin_summary,
        route_b_margin_summary,
        condition_methods,
        config,
    )
    subgroup_metrics, subgroup_increments, subgroup_summary = _gly_pro_boundary(
        components,
        methods,
        config,
    )
    qc = _qc_table(
        paired,
        role_matrices["rigid_transform_qc"],
        paired_effect,
        paired_background,
        variance_alpha,
    )

    tables = {
        "method_registry": method_table,
        "route_b_training": route_b_training,
        "variant_components": components,
        "variant_predictions": variant_predictions,
        "domain_metrics": domain_metrics,
        "domain_increments": increments,
        "increment_summary": increment_summary,
        "distribution_position_rows": distribution["position"],
        "distribution_domain_rows": distribution["domain"],
        "distribution_condition_summary": distribution["condition"],
        "seed_reliability": seed_reliability,
        "condition_validity": condition_validity,
        "seed_domain_metrics": seed_metrics,
        "subtraction_margins": subtraction_margins,
        "subtraction_margin_summary": subtraction_margin_summary,
        "route_b_margins": route_b_margins,
        "route_b_margin_summary": route_b_margin_summary,
        "family_assessment": family_assessment,
        "audit_decisions": decisions,
        "gly_pro_subgroup_metrics": subgroup_metrics,
        "gly_pro_subgroup_increments": subgroup_increments,
        "gly_pro_subgroup_summary": subgroup_summary,
        "quality_controls": qc,
    }
    paths: dict[str, Path] = {}
    for name, table in tables.items():
        path = output / f"{name}.parquet"
        write_parquet(path, table)
        paths[name] = path
    matrix_path = storage / "matrices.npz"
    np.savez_compressed(
        matrix_path,
        sequence_logp=sequence_logp.astype(np.float32),
        paired_mif_logp=paired.astype(np.float32),
        legacy_direct=legacy_direct.astype(np.float32),
        route_b_carp=route_b[ROUTE_B_METHOD].astype(np.float32),
        **{
            f"condition__{family}__{level}": matrix.astype(np.float32)
            for (family, level), matrix in condition_residuals.items()
        },
    )
    paths["matrices"] = matrix_path
    manifest_path = output / "manifest.json"
    write_json(
        manifest_path,
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "protocol_lock": str(config.paths.run_dir / "protocol_lock.json"),
            "mechanisms_labels_used_for_predictor_training": False,
            "variance_matched_paired_alpha": variance_alpha,
            "condition_roles": {
                f"{family}/{level}": roles for (family, level), roles in condition_roles.items()
            },
            "immutable_counterfactuals_decision": ("RETAIN_GENERALIZATION_CLOSE_COUNTERFACTUALS"),
            "final_mechanisms_interpretation": str(decisions["mechanisms_interpretation"].iloc[0]),
            "artifacts": [table_manifest(paths[name], table) for name, table in tables.items()],
            "matrices": {"path": str(matrix_path), "sha256": sha256_file(matrix_path)},
        },
    )
    paths["manifest"] = manifest_path
    return {
        **paths,
        "mechanisms_interpretation": str(decisions["mechanisms_interpretation"].iloc[0]),
        "robust_contrast_supported": bool(decisions["robust_contrast_supported"].iloc[0]),
        "unique_subtraction_supported": bool(decisions["unique_subtraction_supported"].iloc[0]),
        "route_b_structure_recovery_supported": bool(
            decisions["route_b_structure_recovery_supported"].iloc[0]
        ),
    }


def _assert_immutable_project_state(config: MechanismStudyConfig) -> None:
    decision = pd.read_parquet(
        config.paths.counterfactual_run / "evaluation" / "project_decision.parquet"
    )
    observed = str(decision["decision"].iloc[0])
    expected = "RETAIN_GENERALIZATION_CLOSE_COUNTERFACTUALS"
    if observed != expected:
        raise RuntimeError(f"counterfactual study decision drifted: {observed}")


def _condition_ensembles(
    structures: pd.DataFrame,
    direct_by_role: dict[str, np.ndarray],
) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], list[str]]]:
    selected = structures.loc[structures["counterfactual_family"].isin(PRIMARY_LEVELS)]
    matrices: dict[tuple[str, str], np.ndarray] = {}
    roles: dict[tuple[str, str], list[str]] = {}
    for key, frame in selected.groupby(
        ["counterfactual_family", "condition_level"], sort=True, observed=True
    ):
        family, level = str(key[0]), str(key[1])
        role_list = (
            frame.drop_duplicates("structure_role")
            .sort_values("seed_index")["structure_role"]
            .astype(str)
            .tolist()
        )
        roles[(family, level)] = role_list
        matrices[(family, level)] = clr(
            np.median(np.stack([direct_by_role[role] for role in role_list]), axis=0)
        )
    return matrices, roles


def _load_training_artifacts(
    config: MechanismStudyConfig,
    test_queries: pd.DataFrame,
    test_sequence_logp: np.ndarray,
) -> dict[str, Any]:
    generalization = load_generalization_config(config.paths.generalization_config)
    cath = load_generalization_residual_dataset(generalization)
    train = np.flatnonzero(
        cath.metadata["observability_split"]
        .isin(["development_train", "development_validation"])
        .to_numpy()
    )
    cath_carp = load_features(
        generalization.paths.storage_dir / "architecture" / "carp_640M", cath.metadata
    ).astype(np.float32)
    cath_sequence_logp = _aligned_store(
        config.paths.storage_dir / "training_controls" / "esm2_150M_cath",
        cath.metadata,
        "log_probabilities.npy",
    ).astype(float)
    generalization_mif = pd.read_parquet(generalization.paths.run_dir / "mif" / "scores.parquet")
    cath_paired = _aligned_scores(generalization_mif, cath.metadata, "paired")
    return {
        "generalization": generalization,
        "cath_metadata": cath.metadata,
        "cath_target": cath.residuals["mif_paired_minus_rewired"],
        "cath_carp": cath_carp,
        "cath_sequence_logp": cath_sequence_logp,
        "cath_paired": cath_paired,
        "train": train,
        "test_queries": test_queries,
        "test_sequence_logp": test_sequence_logp,
    }


def _route_b_controls(
    config: MechanismStudyConfig,
    artifacts: dict[str, Any],
    test_queries: pd.DataFrame,
    test_carp: np.ndarray,
    test_sequence_logp: np.ndarray,
    direct_legacy: np.ndarray,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    metadata = artifacts["cath_metadata"]
    train = artifacts["train"]
    train_metadata = metadata.iloc[train].reset_index(drop=True)
    y_train = np.asarray(artifacts["cath_target"][train], dtype=float)
    x_train = np.asarray(artifacts["cath_carp"][train], dtype=np.float32)
    carp = fit_rrr_predict(
        x_train,
        y_train,
        test_carp,
        rank=config.models.rrr_rank,
        alpha=config.models.ridge_alpha,
    )
    global_matrix = _conditional_mean_matrix(
        y_train,
        train_metadata["native_aa"].astype(str).to_numpy(),
        test_queries["wild_type"].astype(str).to_numpy(),
    )
    physchem = _physicochemical_prediction(
        y_train,
        train_metadata["native_aa"].astype(str).to_numpy(),
        test_queries["wild_type"].astype(str).to_numpy(),
        config.paths.aaindex1,
        config.models.context_ridge_alpha,
    )
    context_train = _context_features(
        train_metadata,
        np.asarray(artifacts["cath_sequence_logp"][train], dtype=float),
        wild_type_column="native_aa",
    )
    context_test = _context_features(
        test_queries,
        test_sequence_logp,
        wild_type_column="wild_type",
    )
    simple_context = fit_rrr_predict(
        context_train,
        y_train,
        context_test,
        rank=config.models.rrr_rank,
        alpha=config.models.context_ridge_alpha,
    )
    shuffled_train = _shuffle_within_groups(
        x_train,
        train_metadata,
        ["domain_id"],
        np.random.default_rng(config.seed + 910_001),
    )
    shuffled_test = _shuffle_within_groups(
        test_carp,
        test_queries.reset_index(drop=True),
        ["domain_id"],
        np.random.default_rng(config.seed + 910_002),
    )
    carp_shuffled = fit_rrr_predict(
        shuffled_train,
        y_train,
        shuffled_test,
        rank=config.models.rrr_rank,
        alpha=config.models.ridge_alpha,
    )
    shuffled_target = _shuffle_within_groups(
        y_train,
        train_metadata,
        ["domain_id", "burial"],
        np.random.default_rng(config.seed + 920_001),
    )
    target_shuffled = fit_rrr_predict(
        x_train,
        shuffled_target,
        test_carp,
        rank=config.models.rrr_rank,
        alpha=config.models.ridge_alpha,
    )
    predictions: dict[str, np.ndarray] = {
        ROUTE_B_METHOD: carp,
        "route_b_global_wt_mutant_matrix": global_matrix,
        "route_b_grantham_aaindex_blosum_linear": physchem,
        "route_b_simple_sequence_context": simple_context,
        "route_b_carp_context_shuffled": carp_shuffled,
        "route_b_target_conditionally_shuffled": target_shuffled,
    }
    for rank in config.models.direct_pca_ranks:
        components = min(int(rank), y_train.shape[1] - 1, len(y_train))
        model = PCA(n_components=components, svd_solver="full").fit(y_train)
        projection = model.inverse_transform(model.transform(direct_legacy))
        predictions[f"direct_legacy_pca_rank{rank}"] = clr(projection)
    shrinkage = float(np.sqrt(np.mean(carp**2)) / max(np.sqrt(np.mean(direct_legacy**2)), 1e-12))
    predictions["direct_legacy_rms_shrinkage"] = clr(direct_legacy * shrinkage)
    rows = []
    for method, values in predictions.items():
        rows.append(
            {
                "method": method,
                "training_rows": int(len(train)),
                "training_domains": int(train_metadata["domain_id"].nunique()),
                "mechanisms_stability_labels_used": False,
                "target": "mif_paired_minus_legacy_degree_preserving_rewire",
                "prediction_rms": float(np.sqrt(np.mean(values**2))),
                "rrr_rank": config.models.rrr_rank if method.startswith("route_b_") else np.nan,
                "ridge_alpha": (
                    config.models.ridge_alpha if method.startswith("route_b_") else np.nan
                ),
                "direct_shrinkage_factor": (
                    shrinkage if method == "direct_legacy_rms_shrinkage" else np.nan
                ),
            }
        )
    return predictions, pd.DataFrame(rows)


def _conditional_mean_matrix(
    target: np.ndarray,
    training_wild_types: np.ndarray,
    test_wild_types: np.ndarray,
) -> np.ndarray:
    """Return a global WT-conditioned 20-candidate matrix without target-domain context."""

    target = np.asarray(target, dtype=float)
    global_mean = target.mean(axis=0)
    lookup = {}
    for wild_type in AA_ALPHABET:
        selected = target[training_wild_types == wild_type]
        lookup[wild_type] = selected.mean(axis=0) if len(selected) else global_mean
    result = np.stack([lookup[str(wild_type)] for wild_type in test_wild_types])
    return clr(result)


def _physicochemical_prediction(
    target: np.ndarray,
    training_wild_types: np.ndarray,
    test_wild_types: np.ndarray,
    aaindex_path: Path,
    alpha: float,
) -> np.ndarray:
    """Fit a context-free Grantham/AAindex/BLOSUM linear action model."""

    aaindex = read_aaindex1(aaindex_path)
    grantham = aaindex.loc[
        aaindex["description"].str.contains("Grantham", case=False, na=False)
    ].dropna(subset=list(AA_ALPHABET))
    if len(grantham) < 3:
        raise ValueError("AAindex1 lacks the three registered Grantham property vectors")
    properties = grantham.iloc[:3].set_index("accession")[list(AA_ALPHABET)].to_numpy(dtype=float).T
    blosum = substitution_matrices.load("BLOSUM62")
    templates = np.empty((len(AA_ALPHABET), len(AA_ALPHABET), 8), dtype=float)
    for wild_index, wild_type in enumerate(AA_ALPHABET):
        for mutant_index, mutant in enumerate(AA_ALPHABET):
            difference = properties[mutant_index] - properties[wild_index]
            templates[wild_index, mutant_index] = [
                *difference,
                *np.abs(difference),
                float(blosum[wild_type, mutant]),
                float(wild_type == mutant),
            ]
    wild_indices = np.asarray([AA_TO_INDEX[str(value)] for value in training_wild_types])
    x_train = templates[wild_indices].reshape(-1, templates.shape[-1])
    target_action = target - target[np.arange(len(target)), wild_indices, None]
    y_train = target_action.reshape(-1)
    scaler = StandardScaler().fit(x_train)
    model = Ridge(alpha=alpha, solver="lsqr").fit(scaler.transform(x_train), y_train)
    test_indices = np.asarray([AA_TO_INDEX[str(value)] for value in test_wild_types])
    prediction = model.predict(
        scaler.transform(templates[test_indices].reshape(-1, templates.shape[-1]))
    ).reshape(len(test_indices), len(AA_ALPHABET))
    return clr(prediction)


def _context_features(
    metadata: pd.DataFrame,
    sequence_logp: np.ndarray,
    *,
    wild_type_column: str,
) -> np.ndarray:
    """WT identity, ESM entropy, position, length, and local sequence composition."""

    result = np.zeros((len(metadata), 20 + 4 + 20), dtype=float)
    wild_types = metadata[wild_type_column].astype(str).to_numpy()
    result[np.arange(len(metadata)), [AA_TO_INDEX[value] for value in wild_types]] = 1.0
    probabilities = np.exp(normalize_log_probabilities(sequence_logp))
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)), axis=1)
    result[:, 20] = entropy
    for row_index, row in enumerate(metadata.itertuples(index=False)):
        sequence = str(row.sequence)
        position = int(row.position)
        length = len(sequence)
        result[row_index, 21] = position / max(length - 1, 1)
        result[row_index, 22] = min(position, length - 1 - position) / max(length - 1, 1)
        result[row_index, 23] = np.log(float(length))
        left = max(0, position - 4)
        right = min(length, position + 5)
        neighbors = sequence[left:position] + sequence[position + 1 : right]
        denominator = max(len(neighbors), 1)
        for amino_acid in neighbors:
            if amino_acid in AA_TO_INDEX:
                result[row_index, 24 + AA_TO_INDEX[amino_acid]] += 1.0 / denominator
    return result


def _shuffle_within_groups(
    values: np.ndarray,
    metadata: pd.DataFrame,
    group_columns: list[str],
    rng: np.random.Generator,
) -> np.ndarray:
    result = np.asarray(values).copy()
    grouped = metadata.reset_index(drop=True).groupby(group_columns, sort=True, observed=True)
    for indices in grouped.indices.values():
        selected = np.asarray(indices, dtype=int)
        result[selected] = values[rng.permutation(selected)]
    return result


def _variance_matched_alpha(
    sequence_logp: np.ndarray,
    paired_logp: np.ndarray,
    wild_types: np.ndarray,
) -> float:
    wild = np.asarray([AA_TO_INDEX[str(value)] for value in wild_types], dtype=int)
    sequence_action = sequence_logp - sequence_logp[np.arange(len(wild)), wild, None]
    paired_action = paired_logp - paired_logp[np.arange(len(wild)), wild, None]
    mask = np.ones_like(sequence_action, dtype=bool)
    mask[np.arange(len(wild)), wild] = False
    numerator = float(np.sqrt(np.mean(sequence_action[mask] ** 2)))
    denominator = float(np.sqrt(np.mean(paired_action[mask] ** 2)))
    if denominator == 0:
        raise ValueError("paired MIF action RMS is zero")
    return numerator / denominator


def _variant_index(
    variants: pd.DataFrame,
    domains: pd.DataFrame,
    residues: pd.DataFrame,
    queries: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    query_index = queries[["domain_id", "position"]].copy()
    query_index["query_row"] = np.arange(len(query_index), dtype=int)
    result = (
        variants.merge(query_index, on=["domain_id", "position"], validate="many_to_one")
        .merge(
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
        )
        .merge(
            domains[["domain_id", "length", "design_family", "platform"]],
            on="domain_id",
            validate="many_to_one",
        )
    )
    result = result.sort_values(["domain_id", "position", "mutant"], ignore_index=True)
    result["variant_row"] = np.arange(len(result), dtype=int)
    result["substitution_class"] = [
        _substitution_class(str(wild), str(mutant))
        for wild, mutant in zip(result["wild_type"], result["mutant"], strict=True)
    ]
    query_rows = result["query_row"].to_numpy(dtype=int)
    wild = result["wild_type"].map(AA_TO_INDEX).to_numpy(dtype=int)
    mutant = result["mutant"].map(AA_TO_INDEX).to_numpy(dtype=int)
    return result.drop(columns="query_row"), query_rows, wild, mutant


def _matrix_action(
    matrix: np.ndarray,
    query_rows: np.ndarray,
    wild: np.ndarray,
    mutant: np.ndarray,
) -> np.ndarray:
    return matrix[query_rows, mutant] - matrix[query_rows, wild]


def _method_row(
    method: str,
    category: str,
    *,
    test_time_structure: bool,
    family: str = "",
    level: str = "",
    alpha: float = 1.0,
    primary_condition: bool = False,
) -> dict[str, Any]:
    return {
        "method": method,
        "category": category,
        "test_time_structure": test_time_structure,
        "counterfactual_family": family,
        "condition_level": level,
        "alpha": alpha,
        "primary_condition": primary_condition,
    }


def _variant_predictions(components: pd.DataFrame, methods: dict[str, np.ndarray]) -> pd.DataFrame:
    identifiers = components[
        ["variant_row", "domain_id", "position", "wild_type", "mutant", "effect", "stratum"]
    ]
    frames = []
    for method, predicted in methods.items():
        frame = identifiers.copy()
        frame["method"] = method
        frame["predicted_score"] = np.asarray(predicted, dtype=float)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _domain_metrics(
    components: pd.DataFrame,
    methods: dict[str, np.ndarray],
    top_fraction: float,
) -> pd.DataFrame:
    rows = []
    domain_indices = components.groupby("domain_id", sort=True, observed=True).indices
    for domain_id, indices_value in domain_indices.items():
        indices = np.asarray(indices_value, dtype=int)
        observed = components.iloc[indices]["effect"].to_numpy(dtype=float)
        metadata = {
            "domain_id": str(domain_id),
            "stratum": str(components.iloc[indices]["stratum"].iloc[0]),
            "n_variants": int(len(indices)),
        }
        k = max(1, int(np.ceil(len(indices) * top_fraction)))
        for method, prediction in methods.items():
            predicted = np.asarray(prediction)[indices]
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
                    "calibration_slope": _calibration_slope(predicted, observed),
                }
            )
    return pd.DataFrame(rows)


def _metric_increments(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["domain_id", "stratum", "n_variants"]
    baseline = metrics.loc[metrics["method"].eq(SEQUENCE_METHOD), [*keys, *METRICS]].rename(
        columns={metric: f"baseline_{metric}" for metric in METRICS}
    )
    increments = metrics.loc[metrics["method"].ne(SEQUENCE_METHOD)].merge(
        baseline, on=keys, validate="many_to_one"
    )
    for metric in METRICS[:4]:
        increments[f"{metric}_increment"] = increments[metric] - increments[f"baseline_{metric}"]
    increments["calibration_error_reduction"] = (
        increments["baseline_calibration_slope"] - 1.0
    ).abs() - (increments["calibration_slope"] - 1.0).abs()
    return increments


def _summarize_increments(increments: pd.DataFrame, config: MechanismStudyConfig) -> pd.DataFrame:
    rows = []
    for method_index, (method, frame) in enumerate(
        increments.groupby("method", sort=True, observed=True)
    ):
        for metric_index, metric in enumerate(INCREMENT_METRICS):
            estimate = stratified_domain_bootstrap(
                frame,
                metric,
                replicates=config.inference.bootstrap_replicates,
                confidence_level=config.inference.confidence_level,
                seed=config.seed + 100_000 + method_index * 100 + metric_index,
            )
            rows.append(
                {
                    "method": method,
                    "metric": metric,
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
                    metric,
                    replicates=config.inference.bootstrap_replicates,
                    confidence_level=config.inference.confidence_level,
                    seed=(
                        config.seed
                        + 200_000
                        + method_index * 100
                        + metric_index * 10
                        + stratum_index
                    ),
                )
                rows.append(
                    {
                        "method": method,
                        "metric": metric,
                        "scope": "single_stratum",
                        "stratum": str(stratum),
                        **estimate,
                    }
                )
    return pd.DataFrame(rows)


def _distribution_audit(
    queries: pd.DataFrame,
    domains: pd.DataFrame,
    paired: np.ndarray,
    role_matrices: dict[str, np.ndarray],
    structures: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    metadata = queries[["domain_id", "position"]].merge(
        domains[["domain_id", "stratum"]], on="domain_id", validate="many_to_one"
    )
    paired_probability = np.exp(paired)
    paired_entropy = -np.sum(paired_probability * paired, axis=1)
    structure_lookup = structures.set_index("structure_role")
    frames = []
    for role, counterfactual in sorted(role_matrices.items()):
        if role == "rigid_transform_qc":
            continue
        row = structure_lookup.loc[role]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        probability = np.exp(counterfactual)
        entropy = -np.sum(probability * counterfactual, axis=1)
        frame = metadata.copy()
        frame["structure_role"] = role
        frame["counterfactual_family"] = str(row.counterfactual_family)
        frame["condition_level"] = str(row.condition_level)
        frame["seed_index"] = int(row.seed_index)
        frame["paired_counterfactual_jsd_nats"] = rowwise_jsd(paired, counterfactual)
        frame["paired_entropy_nats"] = paired_entropy
        frame["counterfactual_entropy_nats"] = entropy
        frame["entropy_shift_nats"] = entropy - paired_entropy
        frame["absolute_entropy_shift_nats"] = np.abs(entropy - paired_entropy)
        frame["counterfactual_max_probability"] = probability.max(axis=1)
        frames.append(frame)
    position = pd.concat(frames, ignore_index=True)
    value_columns = [
        "paired_counterfactual_jsd_nats",
        "paired_entropy_nats",
        "counterfactual_entropy_nats",
        "entropy_shift_nats",
        "absolute_entropy_shift_nats",
        "counterfactual_max_probability",
    ]
    domain = (
        position.groupby(
            [
                "counterfactual_family",
                "condition_level",
                "seed_index",
                "structure_role",
                "domain_id",
                "stratum",
            ],
            observed=True,
        )[value_columns]
        .mean()
        .reset_index()
    )
    condition = domain.groupby(["counterfactual_family", "condition_level"], observed=True)[
        value_columns
    ].agg(["mean", "median", "std"])
    condition.columns = ["_".join(column) for column in condition.columns]
    condition = condition.reset_index()
    return {"position": position, "domain": domain, "condition": condition}


def _seed_reliability(
    components: pd.DataFrame,
    query_rows: np.ndarray,
    wild: np.ndarray,
    mutant: np.ndarray,
    direct_by_role: dict[str, np.ndarray],
    structures: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    selected = structures.loc[structures["counterfactual_family"].isin(PRIMARY_LEVELS)]
    domain_indices = components.groupby("domain_id", sort=True, observed=True).indices
    for (family, level), frame in selected.groupby(
        ["counterfactual_family", "condition_level"], sort=True, observed=True
    ):
        role_rows = frame.drop_duplicates("structure_role").sort_values("seed_index")
        role_actions = {
            str(row.structure_role): _matrix_action(
                direct_by_role[str(row.structure_role)], query_rows, wild, mutant
            )
            for row in role_rows.itertuples(index=False)
        }
        for domain_id, indices_value in domain_indices.items():
            indices = np.asarray(indices_value, dtype=int)
            for first, second in combinations(role_rows.itertuples(index=False), 2):
                rows.append(
                    {
                        "counterfactual_family": str(family),
                        "condition_level": str(level),
                        "domain_id": str(domain_id),
                        "stratum": str(components.iloc[indices]["stratum"].iloc[0]),
                        "seed_a": int(first.seed_index),
                        "seed_b": int(second.seed_index),
                        "action_spearman": vector_spearman(
                            role_actions[str(first.structure_role)][indices],
                            role_actions[str(second.structure_role)][indices],
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _condition_validity(
    distribution_domains: pd.DataFrame,
    seed_reliability: pd.DataFrame,
    config: MechanismStudyConfig,
) -> pd.DataFrame:
    rows = []
    for (family, level), frame in distribution_domains.groupby(
        ["counterfactual_family", "condition_level"], sort=True, observed=True
    ):
        domain_medians = (
            frame.groupby(["domain_id", "stratum"], observed=True)[
                ["paired_counterfactual_jsd_nats", "absolute_entropy_shift_nats"]
            ]
            .median()
            .reset_index()
        )
        domain_pass = domain_medians["paired_counterfactual_jsd_nats"].le(
            config.inference.id_jsd_max_nats
        ) & domain_medians["absolute_entropy_shift_nats"].le(
            config.inference.id_absolute_entropy_shift_max_nats
        )
        reliability_frame = seed_reliability.loc[
            seed_reliability["counterfactual_family"].eq(str(family))
            & seed_reliability["condition_level"].eq(str(level))
        ]
        reliability = float(reliability_frame["action_spearman"].median())
        eligible_for_id = str(family) in PRIMARY_LEVELS
        median_jsd = float(frame["paired_counterfactual_jsd_nats"].median())
        median_entropy = float(frame["absolute_entropy_shift_nats"].median())
        passing_fraction = float(domain_pass.mean())
        valid = bool(
            eligible_for_id
            and median_jsd <= config.inference.id_jsd_max_nats
            and median_entropy <= config.inference.id_absolute_entropy_shift_max_nats
            and passing_fraction >= config.inference.id_minimum_domain_fraction
            and np.isfinite(reliability)
            and reliability >= config.inference.seed_reliability_minimum_spearman
        )
        rows.append(
            {
                "counterfactual_family": str(family),
                "condition_level": str(level),
                "primary_condition": PRIMARY_LEVELS.get(str(family)) == str(level),
                "eligible_for_id_claim": eligible_for_id,
                "median_jsd_nats": median_jsd,
                "median_absolute_entropy_shift_nats": median_entropy,
                "passing_domain_fraction": passing_fraction,
                "passing_domains": int(domain_pass.sum()),
                "n_domains": int(len(domain_pass)),
                "median_seed_action_spearman": reliability,
                "id_compatible": valid,
            }
        )
    return pd.DataFrame(rows)


def _seed_performance_metrics(
    components: pd.DataFrame,
    sequence_effect: np.ndarray,
    query_rows: np.ndarray,
    wild: np.ndarray,
    mutant: np.ndarray,
    direct_by_role: dict[str, np.ndarray],
    structures: pd.DataFrame,
    top_fraction: float,
) -> pd.DataFrame:
    rows = []
    selected = structures.loc[
        structures["counterfactual_family"].isin(PRIMARY_LEVELS)
    ].drop_duplicates("structure_role")
    domain_indices = components.groupby("domain_id", sort=True, observed=True).indices
    for structure in selected.sort_values(
        ["counterfactual_family", "condition_level", "seed_index"]
    ).itertuples(index=False):
        role = str(structure.structure_role)
        predicted_all = sequence_effect + _matrix_action(
            direct_by_role[role], query_rows, wild, mutant
        )
        for domain_id, indices_value in domain_indices.items():
            indices = np.asarray(indices_value, dtype=int)
            observed = components.iloc[indices]["effect"].to_numpy(dtype=float)
            predicted = predicted_all[indices]
            k = max(1, int(np.ceil(len(indices) * top_fraction)))
            rows.append(
                {
                    "counterfactual_family": str(structure.counterfactual_family),
                    "condition_level": str(structure.condition_level),
                    "seed_index": int(structure.seed_index),
                    "structure_role": role,
                    "domain_id": str(domain_id),
                    "stratum": str(components.iloc[indices]["stratum"].iloc[0]),
                    "spearman": _spearman(predicted, observed),
                    "full_ndcg": _ndcg(predicted, observed),
                    "ndcg_at_10_percent": _ndcg(predicted, observed, k=k),
                }
            )
    return pd.DataFrame(rows)


def _pairwise_margin_tables(
    domain_metrics: pd.DataFrame,
    *,
    candidate_methods: list[str],
    comparator_methods: list[str],
    metrics: list[str],
    config: MechanismStudyConfig,
    seed_offset: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["domain_id", "stratum", "n_variants"]
    rows = []
    summaries = []
    for candidate_index, candidate in enumerate(candidate_methods):
        left = domain_metrics.loc[domain_metrics["method"].eq(candidate), [*keys, *metrics]]
        for comparator_index, comparator in enumerate(comparator_methods):
            right = domain_metrics.loc[
                domain_metrics["method"].eq(comparator), [*keys, *metrics]
            ].rename(columns={metric: f"comparator_{metric}" for metric in metrics})
            merged = left.merge(right, on=keys, validate="one_to_one")
            for metric_index, metric in enumerate(metrics):
                margin = f"{metric}_margin"
                merged[margin] = merged[metric] - merged[f"comparator_{metric}"]
                for row in merged.itertuples(index=False):
                    rows.append(
                        {
                            "candidate_method": candidate,
                            "comparator_method": comparator,
                            "metric": metric,
                            "domain_id": row.domain_id,
                            "stratum": row.stratum,
                            "margin": getattr(row, margin),
                        }
                    )
                estimate = stratified_domain_bootstrap(
                    merged,
                    margin,
                    replicates=config.inference.bootstrap_replicates,
                    confidence_level=config.inference.confidence_level,
                    seed=(
                        config.seed
                        + seed_offset
                        + candidate_index * 1_000
                        + comparator_index * 10
                        + metric_index
                    ),
                )
                summaries.append(
                    {
                        "candidate_method": candidate,
                        "comparator_method": comparator,
                        "metric": metric,
                        **estimate,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def _audit_decisions(
    condition_validity: pd.DataFrame,
    increment_summary: pd.DataFrame,
    subtraction_margin_summary: pd.DataFrame,
    route_b_margin_summary: pd.DataFrame,
    condition_methods: dict[tuple[str, str], str],
    config: MechanismStudyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for family, level in PRIMARY_LEVELS.items():
        validity = condition_validity.loc[
            condition_validity["counterfactual_family"].eq(family)
            & condition_validity["condition_level"].eq(level)
        ]
        if len(validity) != 1:
            raise ValueError(f"missing primary validity row for {family}/{level}")
        method = condition_methods[(family, level)]
        spearman = _summary_row(increment_summary, method, "spearman_increment")
        ndcg = _summary_row(increment_summary, method, "ndcg_at_10_percent_increment")
        direction_supported = bool(spearman["estimate"] > 0 and ndcg["estimate"] > 0)
        margins = subtraction_margin_summary.loc[
            subtraction_margin_summary["candidate_method"].eq(method)
        ]
        expected_margins = len(PAIRED_CONTROLS) * 2
        if len(margins) != expected_margins:
            raise ValueError(f"incomplete paired-only margins for {method}")
        unique_margin_pass = bool((margins["ci_low"] > 0).all())
        id_compatible = bool(validity["id_compatible"].iloc[0])
        rows.append(
            {
                "counterfactual_family": family,
                "primary_condition_level": level,
                "method": method,
                "id_compatible": id_compatible,
                "spearman_increment": float(spearman["estimate"]),
                "spearman_ci_low": float(spearman["ci_low"]),
                "ndcg_at_10_percent_increment": float(ndcg["estimate"]),
                "ndcg_at_10_percent_ci_low": float(ndcg["ci_low"]),
                "direction_supported": direction_supported,
                "robust_family_support": id_compatible and direction_supported,
                "beats_every_paired_only_control": unique_margin_pass,
                "unique_subtraction_family_support": (
                    id_compatible and direction_supported and unique_margin_pass
                ),
            }
        )
    family = pd.DataFrame(rows)
    robust_family_count = int(family["robust_family_support"].sum())
    robust = robust_family_count >= config.inference.robust_minimum_counterfactual_families
    unique = bool(robust and family["unique_subtraction_family_support"].any())
    route_rows = route_b_margin_summary.loc[
        route_b_margin_summary["candidate_method"].eq(ROUTE_B_METHOD)
    ]
    expected_route_rows = len(ROUTE_B_CONTROLS) * 2
    if len(route_rows) != expected_route_rows:
        raise ValueError("Route B control margin table is incomplete")
    route_b_specific = bool((route_rows["ci_low"] > 0).all())
    route_b_spearman = _summary_row(increment_summary, ROUTE_B_METHOD, "spearman_increment")
    route_b_ndcg = _summary_row(increment_summary, ROUTE_B_METHOD, "ndcg_at_10_percent_increment")
    if robust and unique:
        contrast_status = "ROBUST_ID_CONTRAST_AND_UNIQUE_SUBTRACTION_SUPPORTED"
    elif robust:
        contrast_status = "ROBUST_ID_CONTRAST_WITHOUT_UNIQUE_SUBTRACTION"
    elif family["id_compatible"].any():
        contrast_status = "ID_COMPATIBLE_CONDITIONS_WITHOUT_ROBUST_CROSS_FAMILY_SIGNAL"
    else:
        contrast_status = "NO_REGISTERED_COUNTERFACTUAL_FAMILY_ESTABLISHED_AS_ID_COMPATIBLE"
    route_status = (
        "TARGET_SPECIFIC_STRUCTURE_RECOVERY_SUPPORTED"
        if route_b_specific
        else "LOW_DIMENSIONAL_STABILITY_PRIOR_NOT_STRUCTURE_RECOVERY"
    )
    decisions = pd.DataFrame(
        [
            {
                "mechanisms_interpretation": f"{contrast_status};{route_status}",
                "counterfactual_interpretation": contrast_status,
                "route_b_interpretation": route_status,
                "id_compatible_primary_families": int(family["id_compatible"].sum()),
                "robust_supporting_families": robust_family_count,
                "robust_contrast_supported": robust,
                "unique_subtraction_supported": unique,
                "route_b_structure_recovery_supported": route_b_specific,
                "route_b_spearman_increment": float(route_b_spearman["estimate"]),
                "route_b_spearman_ci_low": float(route_b_spearman["ci_low"]),
                "route_b_ndcg_at_10_percent_increment": float(route_b_ndcg["estimate"]),
                "route_b_ndcg_at_10_percent_ci_low": float(route_b_ndcg["ci_low"]),
                "registered_counterfactuals_decision": (
                    "RETAIN_GENERALIZATION_CLOSE_COUNTERFACTUALS"
                ),
                "registered_counterfactuals_decision_modified": False,
                "primary_route": "PIVOT_SELECTIVE_STRUCTURE_CONDITIONED",
                "selective_routing_authorized": False,
            }
        ]
    )
    return family, decisions


def _summary_row(summary: pd.DataFrame, method: str, metric: str) -> pd.Series:
    selected = summary.loc[
        summary["method"].eq(method)
        & summary["metric"].eq(metric)
        & summary["scope"].eq("all_stratum_preserving")
    ]
    if len(selected) != 1:
        raise ValueError(f"missing summary row for {method}/{metric}")
    return selected.iloc[0]


def _gly_pro_boundary(
    components: pd.DataFrame,
    methods: dict[str, np.ndarray],
    config: MechanismStudyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_methods = [SEQUENCE_METHOD, "legacy_direct_contrast", ROUTE_B_METHOD]
    selected_methods.extend(
        f"contrast__{family}__{level}" for family, level in PRIMARY_LEVELS.items()
    )
    subgroup = np.where(
        components["substitution_class"].eq("involves_glycine_or_proline"),
        "involves_glycine_or_proline",
        "other_substitutions",
    )
    rows = []
    grouped_components = components.assign(_subgroup=subgroup)
    for (domain_id, group_name), frame in grouped_components.groupby(
        ["domain_id", "_subgroup"], sort=True, observed=True
    ):
        indices = frame.index.to_numpy(dtype=int)
        if len(indices) < 10:
            continue
        observed = components.iloc[indices]["effect"].to_numpy(dtype=float)
        stratum = str(components.iloc[indices]["stratum"].iloc[0])
        k = max(1, int(np.ceil(len(indices) * config.inference.top_fraction)))
        for method in selected_methods:
            predicted = methods[method][indices]
            rows.append(
                {
                    "domain_id": str(domain_id),
                    "stratum": stratum,
                    "subgroup": str(group_name),
                    "method": method,
                    "n_variants": int(len(indices)),
                    "spearman": _spearman(predicted, observed),
                    "ndcg_at_10_percent": _ndcg(predicted, observed, k=k),
                }
            )
    metrics = pd.DataFrame(rows)
    keys = ["domain_id", "stratum", "subgroup", "n_variants"]
    baseline = metrics.loc[metrics["method"].eq(SEQUENCE_METHOD)].rename(
        columns={
            "spearman": "baseline_spearman",
            "ndcg_at_10_percent": "baseline_ndcg_at_10_percent",
        }
    )
    increments = metrics.loc[metrics["method"].ne(SEQUENCE_METHOD)].merge(
        baseline[[*keys, "baseline_spearman", "baseline_ndcg_at_10_percent"]],
        on=keys,
        validate="many_to_one",
    )
    increments["spearman_increment"] = increments["spearman"] - increments["baseline_spearman"]
    increments["ndcg_at_10_percent_increment"] = (
        increments["ndcg_at_10_percent"] - increments["baseline_ndcg_at_10_percent"]
    )
    summary_rows = []
    for group_index, ((subgroup_name, method), frame) in enumerate(
        increments.groupby(["subgroup", "method"], sort=True, observed=True)
    ):
        for metric_index, metric in enumerate(
            ["spearman_increment", "ndcg_at_10_percent_increment"]
        ):
            summary_rows.append(
                {
                    "subgroup": str(subgroup_name),
                    "method": str(method),
                    "metric": metric,
                    "analysis_role": "predeclared_boundary_not_a_gate",
                    **stratified_domain_bootstrap(
                        frame,
                        metric,
                        replicates=config.inference.bootstrap_replicates,
                        confidence_level=config.inference.confidence_level,
                        seed=config.seed + 900_000 + group_index * 10 + metric_index,
                    ),
                }
            )
    return metrics, increments, pd.DataFrame(summary_rows)


def _qc_table(
    paired: np.ndarray,
    rigid: np.ndarray,
    paired_effect: np.ndarray,
    paired_background: np.ndarray,
    variance_alpha: float,
) -> pd.DataFrame:
    uniform = np.full_like(paired, -np.log(len(AA_ALPHABET)))
    uniform_contrast = clr(paired - uniform)
    centered_paired = clr(paired)
    rows = [
        {
            "check": "rigid_transform_logp_invariance",
            "estimate": float(np.max(np.abs(paired - rigid))),
            "secondary_estimate": float(np.sqrt(np.mean((paired - rigid) ** 2))),
            "status": "PASS" if np.max(np.abs(paired - rigid)) < 1e-3 else "FAIL",
        },
        {
            "check": "uniform_subtraction_action_identity",
            "estimate": float(np.max(np.abs(uniform_contrast - centered_paired))),
            "secondary_estimate": float(np.sqrt(np.mean(paired_effect**2))),
            "status": (
                "PASS" if np.max(np.abs(uniform_contrast - centered_paired)) < 1e-10 else "FAIL"
            ),
        },
        {
            "check": "cath_substitution_background_rms",
            "estimate": float(np.sqrt(np.mean(paired_background**2))),
            "secondary_estimate": float("nan"),
            "status": "INFORMATIONAL",
        },
        {
            "check": "paired_variance_matched_alpha",
            "estimate": float(variance_alpha),
            "secondary_estimate": float("nan"),
            "status": "INFORMATIONAL",
        },
    ]
    return pd.DataFrame(rows)


def _spearman(predicted: np.ndarray, observed: np.ndarray) -> float:
    if len(predicted) < 2 or np.ptp(predicted) == 0 or np.ptp(observed) == 0:
        return float("nan")
    result = spearmanr(predicted, observed).statistic
    return float(result) if np.isfinite(result) else float("nan")


def _ndcg(predicted: np.ndarray, observed: np.ndarray, *, k: int | None = None) -> float:
    relevance = observed - np.min(observed)
    if np.allclose(relevance, 0):
        return float("nan")
    return float(ndcg_score(relevance[None, :], predicted[None, :], k=k))


def _stabilizing_recall(predicted: np.ndarray, observed: np.ndarray, k: int) -> float:
    stabilizing = observed > 0
    positives = int(stabilizing.sum())
    if positives == 0:
        return float("nan")
    predicted_top = np.argpartition(predicted, -k)[-k:]
    return float(stabilizing[predicted_top].sum() / min(k, positives))


def _calibration_slope(predicted: np.ndarray, observed: np.ndarray) -> float:
    centered = predicted - np.mean(predicted)
    denominator = float(np.dot(centered, centered))
    if denominator == 0:
        return float("nan")
    return float(np.dot(centered, observed - np.mean(observed)) / denominator)
