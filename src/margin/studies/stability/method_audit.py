"""Retrospective, post-lock method and cost audit for stability study.

This supplement expands the baseline matrix after the registered stability study
decision.  It is deliberately separated from the confirmatory evaluation and
cannot change any frozen gate or project decision.
"""

from __future__ import annotations

import math
import pickle
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices

from margin.attribution.metrics import normalize_log_probabilities
from margin.constants import AA_ALPHABET, AA_TO_INDEX
from margin.provenance import runtime_manifest, write_json, write_parquet
from margin.studies.action_validation.evaluation import (
    METRICS,
    _aligned_store,
    _anchor,
    _domain_metrics,
)
from margin.studies.counterfactuals.evaluation import stratified_domain_bootstrap
from margin.studies.stability.config import StabilityStudyConfig
from margin.studies.stability.prepare import EXTERNAL_POPULATION, PRIMARY_POPULATION

TEACHERS = ("mif", "esm_if1", "proteinmpnn")
SEQUENCE_BASELINE = "esm2_150M_loo"
UNSCALED_CONSENSUS = "esm2_150M_plus_unscaled_consensus"
SELECTED_CONSENSUS = "esm2_150M_plus_temperature_consensus"
CPLUS_CONTROL = "esm2_150M_plus_G_Cplus"


def audit_stability_methods(config: StabilityStudyConfig) -> dict[str, Path]:
    """Build the expanded zero-shot matrix, cost audit, and overlap audit."""

    run = config.paths.run_dir
    output = run / "method_audit"
    output.mkdir(parents=True, exist_ok=True)
    components = pd.read_parquet(run / "evaluation" / "variant_components.parquet").sort_values(
        "variant_row", ignore_index=True
    )
    if not np.array_equal(
        components["variant_row"].to_numpy(dtype=int), np.arange(len(components))
    ):
        raise ValueError("stability study variant rows are not a complete contiguous index")
    queries = pd.read_parquet(run / "panel" / "query_rows.parquet")
    domains = pd.read_parquet(run / "panel" / "domains.parquet")
    query_rows, mutants = _variant_query_indices(components, queries)
    methods, registry, subset_methods = _method_predictions(
        config, components, queries, query_rows, mutants
    )
    domain_metrics = _domain_metrics(components, methods, config.inference.top_fraction)
    domain_metrics.loc[
        domain_metrics["evaluation_population"].eq(EXTERNAL_POPULATION),
        "stabilizing_top_10_percent_recall",
    ] = np.nan
    method_summary = _method_summary(domain_metrics, registry, config)
    contrasts = _contrast_registry()
    domain_contrasts = _domain_contrasts(domain_metrics, contrasts)
    contrast_summary = _contrast_summary(domain_contrasts, config)
    subset_summary = _subset_summary(method_summary, subset_methods)
    shapley = _exact_teacher_shapley(domain_metrics, subset_methods)
    cost = _teacher_cost_audit(config)
    overlap = _supervised_split_overlap(domains)

    tables = {
        "method_registry": registry,
        "domain_metrics": domain_metrics,
        "method_summary": method_summary,
        "contrast_registry": contrasts,
        "domain_contrasts": domain_contrasts,
        "contrast_summary": contrast_summary,
        "teacher_subset_summary": subset_summary,
        "teacher_shapley": shapley,
        "teacher_cost": cost,
        "supervised_split_overlap": overlap,
    }
    paths: dict[str, Path] = {}
    for name, table in tables.items():
        path = output / f"{name}.parquet"
        write_parquet(path, table)
        paths[name] = path
    manifest = output / "manifest.json"
    write_json(
        manifest,
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": "stability.method_audit.v1",
            "analysis_role": "retrospective_postlock_supplement",
            "outcomes_were_open_before_analysis": True,
            "changes_primary_decision": False,
            "stability_decision_source": str(run / "evaluation" / "project_decision.parquet"),
            "calibration_claim": (
                "outcome-free implementation choice; not counted as a performance contribution"
            ),
            "structure_residual_interpretation": (
                "residual beyond registered G+Cplus; not pure or causal structure information"
            ),
            "external_cplus_status": (
                "descriptive only; no post-lock cross-platform Cplus confirmation claim"
            ),
            "simplified_sequence_prior_sum_mapping": {
                "method": "esm2_150M_plus_esm_if1",
                "formula_stabilizing_sign": "ESM2_sequence_log_odds + ESM_IF1_structure_log_odds",
                "scope": "simplified sequence-prior plus single-structure action sum",
                "not_claimed": (
                    "an official ensemble, importance-sampling, or free-energy protocol"
                ),
            },
            "artifacts": {
                name: {
                    "path": str(paths[name]),
                    "rows": int(len(table)),
                    "columns": list(table.columns),
                }
                for name, table in tables.items()
            },
        },
    )
    paths["manifest"] = manifest
    return paths


def _variant_query_indices(
    components: pd.DataFrame, queries: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    query_index = queries[["domain_id", "position"]].copy()
    query_index["query_row"] = np.arange(len(query_index), dtype=int)
    aligned = components[["variant_row", "domain_id", "position", "mutant"]].merge(
        query_index, on=["domain_id", "position"], validate="many_to_one"
    )
    aligned = aligned.sort_values("variant_row")
    if len(aligned) != len(components):
        raise ValueError("query rows do not cover every stability study variant")
    return (
        aligned["query_row"].to_numpy(dtype=int),
        aligned["mutant"].map(AA_TO_INDEX).to_numpy(dtype=int),
    )


def _sequence_variant_actions(
    config: StabilityStudyConfig,
    queries: pd.DataFrame,
    query_rows: np.ndarray,
    mutants: np.ndarray,
) -> dict[str, np.ndarray]:
    stores = {
        "esm2_150M_loo": config.paths.storage_dir / "representations" / "esm2_150M",
        "esm2_650M_loo": (config.paths.storage_dir / "baselines" / "sequence" / "esm2_650M_loo"),
        "esm1b_650M_loo": (config.paths.storage_dir / "baselines" / "sequence" / "esm1b_650M_loo"),
        "carp_640M_loo": (config.paths.storage_dir / "baselines" / "sequence" / "carp_640M_loo"),
    }
    wild = queries["wild_type"].map(AA_TO_INDEX).to_numpy(dtype=int)
    result = {}
    for method, store in stores.items():
        logp = normalize_log_probabilities(
            _aligned_store(store, queries, "log_probabilities.npy").astype(float)
        )
        action = _anchor(logp, wild)
        result[method] = action[query_rows, mutants]
    return result


def _method_predictions(
    config: StabilityStudyConfig,
    components: pd.DataFrame,
    queries: pd.DataFrame,
    query_rows: np.ndarray,
    mutants: np.ndarray,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[frozenset[str], str]]:
    sequence = _sequence_variant_actions(config, queries, query_rows, mutants)
    methods: dict[str, np.ndarray] = dict(sequence)
    rows: list[dict[str, Any]] = []
    for method in sequence:
        rows.append(
            _registry_row(
                method,
                "zero_shot_sequence",
                False,
                "none",
                f"strict LOO log-odds from {method.removesuffix('_loo')}",
            )
        )

    profile = _profile_action(config, queries, query_rows, mutants)
    methods["cath_homolog_profile"] = profile
    rows.append(
        _registry_row(
            "cath_homolog_profile",
            "zero_shot_evolutionary",
            False,
            "none",
            "CATH homolog-profile mutant/wild-type log-odds",
        )
    )
    blosum = _blosum_action(components)
    methods["blosum62_global"] = blosum
    rows.append(
        _registry_row(
            "blosum62_global",
            "global_substitution_matrix",
            False,
            "none",
            "BLOSUM62(wild,mutant)-BLOSUM62(wild,wild)",
        )
    )

    base = sequence[SEQUENCE_BASELINE]
    teacher_actions = {
        teacher: components[f"{teacher}_action"].to_numpy(dtype=float) for teacher in TEACHERS
    }
    structure_only = {
        "mif_action_only": teacher_actions["mif"],
        "esm_if1_action_only": teacher_actions["esm_if1"],
        "proteinmpnn_action_only": teacher_actions["proteinmpnn"],
        "unscaled_equal_action_only": components["unscaled_equal_action"].to_numpy(float),
        "temperature_consensus_action_only": components[
            "joint_temperature_native_nll_action"
        ].to_numpy(float),
    }
    for method, values in structure_only.items():
        methods[method] = values
        rows.append(
            _registry_row(
                method,
                "zero_shot_structure_action_only",
                True,
                "none",
                method.replace("_", " "),
            )
        )

    subset_methods: dict[frozenset[str], str] = {frozenset(): SEQUENCE_BASELINE}
    for size in range(1, len(TEACHERS) + 1):
        for subset in combinations(TEACHERS, size):
            key = frozenset(subset)
            label = "_".join(subset)
            method = f"esm2_150M_plus_unscaled_{label}"
            action = np.mean(np.stack([teacher_actions[name] for name in subset]), axis=0)
            methods[method] = base + action
            subset_methods[key] = method
            formula = f"ESM2-150M + mean({', '.join(subset)}) action"
            if key == frozenset(TEACHERS):
                method_alias = UNSCALED_CONSENSUS
                methods[method_alias] = methods.pop(method)
                subset_methods[key] = method_alias
                method = method_alias
            rows.append(
                _registry_row(
                    method,
                    "zero_shot_sequence_plus_structure",
                    True,
                    "none",
                    formula,
                    simplified_sequence_prior_sum=(key == frozenset({"esm_if1"})),
                )
            )

    selected_action = components["joint_temperature_native_nll_action"].to_numpy(float)
    methods[SELECTED_CONSENSUS] = base + selected_action
    rows.append(
        _registry_row(
            SELECTED_CONSENSUS,
            "zero_shot_sequence_plus_structure",
            True,
            "outcome-free CATH native-residue temperature selection",
            "ESM2-150M + temperature-scaled three-teacher action",
        )
    )
    for sequence_method in ("esm2_650M_loo", "esm1b_650M_loo", "carp_640M_loo"):
        method = f"{sequence_method.removesuffix('_loo')}_plus_temperature_consensus"
        methods[method] = sequence[sequence_method] + selected_action
        rows.append(
            _registry_row(
                method,
                "postlock_sequence_base_sensitivity",
                True,
                "outcome-free CATH native-residue temperature selection",
                f"{sequence_method} + frozen temperature-scaled action",
            )
        )

    g = components["consensus_g"].to_numpy(dtype=float)
    c_plus = components["consensus_c_plus"].to_numpy(dtype=float)
    methods["cath_teacher_G"] = g
    methods["esm2_150M_plus_G"] = base + g
    methods[CPLUS_CONTROL] = base + g + c_plus
    rows.extend(
        [
            _registry_row(
                "cath_teacher_G",
                "outcome_free_teacher_global_matrix",
                False,
                "CATH teacher actions only",
                "global wild-type-conditioned teacher action G",
            ),
            _registry_row(
                "esm2_150M_plus_G",
                "zero_shot_sequence_control",
                False,
                "CATH teacher actions only",
                "ESM2-150M + G",
            ),
            _registry_row(
                CPLUS_CONTROL,
                "registered_strong_sequence_control",
                False,
                "CATH teacher actions and representations only",
                "ESM2-150M + G + Cplus",
            ),
        ]
    )
    registry = pd.DataFrame(rows).drop_duplicates("method", keep="last")
    if set(methods) != set(registry["method"]):
        missing = sorted(set(methods).symmetric_difference(registry["method"]))
        raise ValueError(f"method registry mismatch: {missing}")
    return (
        methods,
        registry.sort_values(["method_group", "method"], ignore_index=True),
        subset_methods,
    )


def _registry_row(
    method: str,
    method_group: str,
    test_time_structure: bool,
    supervision: str,
    formula: str,
    *,
    simplified_sequence_prior_sum: bool = False,
) -> dict[str, Any]:
    return {
        "method": method,
        "method_group": method_group,
        "test_time_structure": test_time_structure,
        "stability_label_supervision": supervision,
        "formula": formula,
        "simplified_sequence_prior_sum": simplified_sequence_prior_sum,
        "analysis_role": "retrospective_postlock_supplement",
    }


def _profile_action(
    config: StabilityStudyConfig,
    queries: pd.DataFrame,
    query_rows: np.ndarray,
    mutants: np.ndarray,
) -> np.ndarray:
    profile = pd.read_parquet(config.paths.run_dir / "strong_control" / "panel_profiles.parquet")
    keys = ["state_id", "domain_id", "position"]
    aligned = queries[keys].merge(profile, on=keys, validate="one_to_one")
    if len(aligned) != len(queries):
        raise ValueError("homolog profiles do not cover all stability study queries")
    probabilities = aligned[[f"profile_{aa}" for aa in AA_ALPHABET]].to_numpy(dtype=float)
    wild = queries["wild_type"].map(AA_TO_INDEX).to_numpy(dtype=int)
    action = _anchor(np.log(np.maximum(probabilities, 1e-12)), wild)
    return action[query_rows, mutants]


def _blosum_action(components: pd.DataFrame) -> np.ndarray:
    matrix = substitution_matrices.load("BLOSUM62")
    return np.asarray(
        [
            float(matrix[wild, mutant] - matrix[wild, wild])
            for wild, mutant in zip(components["wild_type"], components["mutant"], strict=True)
        ],
        dtype=float,
    )


def _method_summary(
    metrics: pd.DataFrame, registry: pd.DataFrame, config: StabilityStudyConfig
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    lookup = registry.set_index("method")
    primary = metrics.loc[metrics["evaluation_population"].eq(PRIMARY_POPULATION)]
    for method_index, (method, frame) in enumerate(primary.groupby("method", sort=True)):
        scopes = [("all", frame)]
        scopes.extend(
            (str(stratum), group)
            for stratum, group in frame.groupby("stratum", sort=True, observed=True)
        )
        for scope_index, (stratum, scope) in enumerate(scopes):
            for metric_index, metric in enumerate(METRICS):
                result = stratified_domain_bootstrap(
                    scope,
                    metric,
                    replicates=config.inference.bootstrap_replicates,
                    confidence_level=config.inference.confidence_level,
                    seed=(
                        config.seed
                        + 500_000
                        + method_index * 1000
                        + scope_index * 10
                        + metric_index
                    ),
                )
                rows.append(
                    {
                        "method": method,
                        "method_group": str(lookup.loc[method, "method_group"]),
                        "evaluation_population": PRIMARY_POPULATION,
                        "stratum": stratum,
                        "metric": metric,
                        "interval_unit": "protein_domain",
                        **result,
                    }
                )
    external = metrics.loc[metrics["evaluation_population"].eq(EXTERNAL_POPULATION)]
    for row in external.itertuples(index=False):
        for metric in METRICS:
            rows.append(
                {
                    "method": row.method,
                    "method_group": str(lookup.loc[row.method, "method_group"]),
                    "evaluation_population": EXTERNAL_POPULATION,
                    "stratum": "external_single_protein",
                    "metric": metric,
                    "interval_unit": "point_only",
                    "estimate": float(getattr(row, metric)),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "positive_domain_fraction": float("nan"),
                    "positive_domains": 0,
                    "negative_domains": 0,
                    "zero_domains": 0,
                    "n_domains": 1,
                    "leave_one_domain_out_min": float("nan"),
                    "leave_one_domain_out_max": float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _contrast_registry() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for teacher in TEACHERS:
        single = f"esm2_150M_plus_unscaled_{teacher}"
        action_only = f"{teacher}_action_only"
        rows.extend(
            [
                {
                    "contrast": f"unscaled_consensus_vs_{teacher}",
                    "method": UNSCALED_CONSENSUS,
                    "comparator": single,
                    "role": "marginal_ensemble_value",
                },
                {
                    "contrast": f"temperature_consensus_vs_{teacher}",
                    "method": SELECTED_CONSENSUS,
                    "comparator": single,
                    "role": "robust_vs_fast_single_teacher",
                },
                {
                    "contrast": f"unscaled_action_consensus_vs_{teacher}_action",
                    "method": "unscaled_equal_action_only",
                    "comparator": action_only,
                    "role": "action_only_marginal_ensemble_value",
                },
                {
                    "contrast": f"temperature_action_consensus_vs_{teacher}_action",
                    "method": "temperature_consensus_action_only",
                    "comparator": action_only,
                    "role": "action_only_robust_vs_fast_single_teacher",
                },
            ]
        )
    rows.extend(
        [
            {
                "contrast": "temperature_vs_unscaled_consensus",
                "method": SELECTED_CONSENSUS,
                "comparator": UNSCALED_CONSENSUS,
                "role": "calibration_descriptive_not_method_contribution",
            },
            {
                "contrast": "temperature_consensus_vs_Cplus",
                "method": SELECTED_CONSENSUS,
                "comparator": CPLUS_CONTROL,
                "role": "registered_strong_control_recap",
            },
            {
                "contrast": "temperature_action_vs_unscaled_action_consensus",
                "method": "temperature_consensus_action_only",
                "comparator": "unscaled_equal_action_only",
                "role": "calibration_descriptive_not_method_contribution",
            },
            {
                "contrast": "unscaled_action_consensus_vs_registered_selected",
                "method": "unscaled_equal_action_only",
                "comparator": SELECTED_CONSENSUS,
                "role": "postlock_method_simplification",
            },
            {
                "contrast": "temperature_action_only_vs_registered_selected",
                "method": "temperature_consensus_action_only",
                "comparator": SELECTED_CONSENSUS,
                "role": "sequence_term_ablation",
            },
            {
                "contrast": "esm_if1_action_only_vs_simplified_sequence_prior_sum",
                "method": "esm_if1_action_only",
                "comparator": "esm2_150M_plus_unscaled_esm_if1",
                "role": "simplified_sequence_prior_sum_ablation",
            },
        ]
    )
    for sequence in (
        "esm2_150M_loo",
        "esm2_650M_loo",
        "esm1b_650M_loo",
        "carp_640M_loo",
        "cath_homolog_profile",
    ):
        method = (
            SELECTED_CONSENSUS
            if sequence == SEQUENCE_BASELINE
            else f"{sequence.removesuffix('_loo')}_plus_temperature_consensus"
            if sequence.endswith("_loo")
            else SELECTED_CONSENSUS
        )
        if sequence == "cath_homolog_profile":
            method = SELECTED_CONSENSUS
        rows.append(
            {
                "contrast": f"{method}_vs_{sequence}",
                "method": method,
                "comparator": sequence,
                "role": "sequence_baseline_sensitivity",
            }
        )
    return pd.DataFrame(rows).drop_duplicates("contrast", keep="first")


def _domain_contrasts(metrics: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    keys = ["domain_id", "evaluation_population", "stratum", "n_variants"]
    frames = []
    for row in registry.itertuples(index=False):
        method = metrics.loc[metrics["method"].eq(row.method), [*keys, *METRICS]]
        comparator = metrics.loc[metrics["method"].eq(row.comparator), [*keys, *METRICS]].rename(
            columns={metric: f"comparator_{metric}" for metric in METRICS}
        )
        merged = method.merge(comparator, on=keys, validate="one_to_one")
        merged["contrast"] = row.contrast
        merged["method"] = row.method
        merged["comparator"] = row.comparator
        merged["role"] = row.role
        for metric in METRICS:
            merged[f"{metric}_margin"] = merged[metric] - merged[f"comparator_{metric}"]
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def _contrast_summary(domain: pd.DataFrame, config: StabilityStudyConfig) -> pd.DataFrame:
    rows = []
    for contrast_index, (contrast, frame) in enumerate(
        domain.loc[domain["evaluation_population"].eq(PRIMARY_POPULATION)].groupby(
            "contrast", sort=True
        )
    ):
        metadata = frame.iloc[0]
        scopes = [("all", frame)]
        scopes.extend(
            (str(stratum), group)
            for stratum, group in frame.groupby("stratum", sort=True, observed=True)
        )
        for scope_index, (stratum, scope) in enumerate(scopes):
            for metric_index, metric in enumerate(METRICS):
                rows.append(
                    {
                        "contrast": contrast,
                        "method": metadata["method"],
                        "comparator": metadata["comparator"],
                        "role": metadata["role"],
                        "evaluation_population": PRIMARY_POPULATION,
                        "stratum": stratum,
                        "metric": metric,
                        "interval_unit": "protein_domain",
                        **stratified_domain_bootstrap(
                            scope,
                            f"{metric}_margin",
                            replicates=config.inference.bootstrap_replicates,
                            confidence_level=config.inference.confidence_level,
                            seed=config.seed
                            + 600_000
                            + contrast_index * 1000
                            + scope_index * 10
                            + metric_index,
                        ),
                    }
                )
    external = domain.loc[domain["evaluation_population"].eq(EXTERNAL_POPULATION)]
    for row in external.itertuples(index=False):
        for metric in METRICS:
            rows.append(
                {
                    "contrast": row.contrast,
                    "method": row.method,
                    "comparator": row.comparator,
                    "role": row.role,
                    "evaluation_population": EXTERNAL_POPULATION,
                    "stratum": "external_single_protein",
                    "metric": metric,
                    "interval_unit": "point_only_postlock_descriptive",
                    "estimate": float(getattr(row, f"{metric}_margin")),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "positive_domain_fraction": float("nan"),
                    "positive_domains": 0,
                    "negative_domains": 0,
                    "zero_domains": 0,
                    "n_domains": 1,
                    "leave_one_domain_out_min": float("nan"),
                    "leave_one_domain_out_max": float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _subset_summary(
    summary: pd.DataFrame, subset_methods: dict[frozenset[str], str]
) -> pd.DataFrame:
    reverse = {
        method: "+".join(sorted(subset)) if subset else "none"
        for subset, method in subset_methods.items()
    }
    selected = summary.loc[
        summary["method"].isin(reverse)
        & summary["evaluation_population"].eq(PRIMARY_POPULATION)
        & summary["stratum"].eq("all")
    ].copy()
    selected["teacher_subset"] = selected["method"].map(reverse)
    selected["n_teachers"] = selected["teacher_subset"].map(
        lambda value: 0 if value == "none" else len(value.split("+"))
    )
    return selected.sort_values(["metric", "n_teachers", "teacher_subset"], ignore_index=True)


def _exact_teacher_shapley(
    metrics: pd.DataFrame, subset_methods: dict[frozenset[str], str]
) -> pd.DataFrame:
    primary = metrics.loc[metrics["evaluation_population"].eq(PRIMARY_POPULATION)]
    rows = []
    n = len(TEACHERS)
    for metric in ("spearman", "ndcg_at_10_percent"):
        values = {
            subset: float(primary.loc[primary["method"].eq(method), metric].mean())
            for subset, method in subset_methods.items()
        }
        for teacher in TEACHERS:
            contribution = 0.0
            for subset in subset_methods:
                if teacher in subset:
                    continue
                augmented = frozenset((*subset, teacher))
                weight = (
                    math.factorial(len(subset))
                    * math.factorial(n - len(subset) - 1)
                    / math.factorial(n)
                )
                contribution += weight * (values[augmented] - values[subset])
            rows.append(
                {
                    "teacher_id": teacher,
                    "metric": metric,
                    "shapley_value": contribution,
                    "grand_coalition_gain": values[frozenset(TEACHERS)] - values[frozenset()],
                    "analysis_role": "outcome_opened_descriptive",
                }
            )
    return pd.DataFrame(rows)


def _teacher_cost_audit(config: StabilityStudyConfig) -> pd.DataFrame:
    request_path = config.paths.run_dir / "teacher_requests" / "requests.parquet"
    requests = pd.read_parquet(request_path)[["request_id", "domain_id", "length"]]
    foundation = config.paths
    checkpoint_paths = {
        "mif": foundation.mif_checkpoint,
        "esm_if1": foundation.esm_if1_checkpoint,
        "proteinmpnn": foundation.proteinmpnn_checkpoint,
    }
    per_teacher: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for teacher in TEACHERS:
        raw = pd.read_parquet(
            config.paths.run_dir / "teacher_scores" / "raw" / f"{teacher}.parquet"
        )
        timing = raw[["request_id", "wall_seconds", "forward_calls"]].drop_duplicates()
        timing = timing.merge(requests, on="request_id", validate="one_to_one")
        per_teacher[teacher] = timing
        checkpoint = checkpoint_paths[teacher]
        rows.append(
            {
                "method": teacher,
                "cost_role": "single_teacher_full_Lx19",
                "teachers": teacher,
                "n_domains": int(len(timing)),
                "n_residues": int(timing["length"].sum()),
                "forward_calls": int(timing["forward_calls"].sum()),
                "wall_seconds_total": float(timing["wall_seconds"].sum()),
                "wall_seconds_domain_median": float(timing["wall_seconds"].median()),
                "wall_seconds_per_residue": float(
                    timing["wall_seconds"].sum() / timing["length"].sum()
                ),
                "checkpoint_bytes": _path_bytes(checkpoint),
                "measured_scope": "adapter scoring of all 19 substitutions at every residue",
                "excluded_costs": "model load; released/predicted structure generation",
            }
        )
    merged = requests.copy()
    for teacher, frame in per_teacher.items():
        merged = merged.merge(
            frame[["request_id", "wall_seconds"]].rename(
                columns={"wall_seconds": f"wall_{teacher}"}
            ),
            on="request_id",
            validate="one_to_one",
        )
    serial = merged[[f"wall_{teacher}" for teacher in TEACHERS]].sum(axis=1)
    rows.append(
        {
            "method": "three_teacher_consensus_serial",
            "cost_role": "robust_three_teacher_full_Lx19",
            "teachers": "+".join(TEACHERS),
            "n_domains": int(len(merged)),
            "n_residues": int(merged["length"].sum()),
            "forward_calls": int(sum(row["forward_calls"] for row in rows)),
            "wall_seconds_total": float(serial.sum()),
            "wall_seconds_domain_median": float(serial.median()),
            "wall_seconds_per_residue": float(serial.sum() / merged["length"].sum()),
            "checkpoint_bytes": int(sum(row["checkpoint_bytes"] for row in rows)),
            "measured_scope": "serial sum of measured single-teacher full Lx19 scoring",
            "excluded_costs": "model load; released/predicted structure generation",
        }
    )
    return pd.DataFrame(rows)


def _supervised_split_overlap(domains: pd.DataFrame) -> pd.DataFrame:
    split_path = Path("external/repositories/ThermoMPNN/dataset_splits/mega_splits.pkl")
    if not split_path.exists():
        return pd.DataFrame(columns=["domain_id", "wt_name", "thermompnn_split", "interpretation"])
    with split_path.open("rb") as handle:
        raw = pickle.load(handle)  # noqa: S301 - pinned local upstream artifact
    split_lookup = {
        str(name): split
        for split in ("train", "val", "test")
        for name in np.asarray(raw[split]).tolist()
    }
    primary = domains.loc[domains["evaluation_population"].eq(PRIMARY_POPULATION)].copy()
    primary["thermompnn_split"] = primary["wt_name"].map(split_lookup).fillna("not_listed")
    interpretation = {
        "train": "exact label-training overlap; transductive upper bound only",
        "val": "official model-selection split; not an untouched test",
        "test": "official exact-name held-out test split",
        "not_listed": "not in official split file; independence not inferred",
    }
    primary["interpretation"] = primary["thermompnn_split"].map(interpretation)
    primary["applies_to"] = "ThermoMPNN and SPURS when using the shared Megascale split"
    return primary[
        ["domain_id", "wt_name", "stratum", "thermompnn_split", "interpretation", "applies_to"]
    ].sort_values(["thermompnn_split", "domain_id"], ignore_index=True)


def _path_bytes(path: Path) -> int:
    if path.is_file():
        return int(path.stat().st_size)
    if path.is_dir():
        return int(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()))
    return 0
