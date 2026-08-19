"""Frozen score construction and outcome opening for the FireProt confirmation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from margin.attribution.metrics import normalize_log_probabilities
from margin.constants import AA_TO_INDEX
from margin.provenance import read_json, runtime_manifest, write_json, write_parquet
from margin.studies.action_validation.evaluation import (
    _aligned_store,
    _aligned_teacher,
    _anchor,
    _global_component,
    _ndcg,
    _spearman,
)
from margin.studies.counterfactuals.evaluation import stratified_domain_bootstrap
from margin.studies.external_validation.panel import ExternalValidationConfig
from margin.studies.stability.calibration import load_cath_calibration_data
from margin.studies.stability.config import load_stability_config
from margin.studies.stability.strong_control import _feature_bundle


def build_external_validation_scores(config: ExternalValidationConfig) -> dict[str, Path]:
    """Apply frozen teachers and C+ heads without reading FireProt outcomes."""

    _require_lock(config)
    stability = load_stability_config(config.paths.stability_config)
    required = [
        config.paths.run_dir / "teacher_scores/scores.parquet",
        config.paths.run_dir / "strong_control/panel_profiles.parquet",
        config.paths.run_dir / "representations_manifest.json",
        stability.paths.run_dir / "calibration/selection.json",
    ]
    required.extend(
        stability.paths.storage_dir / "strong_control/models" / f"{teacher}.joblib"
        for teacher in stability.calibration.teacher_ids
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"cross-platform score inputs are missing: {missing}")
    output = config.paths.storage_dir / "strong_control"
    report = config.paths.run_dir / "strong_control"
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    paths = {
        "features": output / "final_panel.npy",
        "feature_names": output / "feature_names.json",
        "components": output / "components.npz",
        "environment_audit": report / "environment_head_audit.parquet",
        "manifest": report / "score_manifest.json",
    }
    cath = pd.read_parquet(stability.paths.cath_queries).reset_index(drop=True)
    queries = pd.read_parquet(config.paths.run_dir / "panel/query_rows.parquet")
    cath_data = load_cath_calibration_data(stability)
    if not cath[["state_id", "domain_id", "position"]].equals(
        cath_data["metadata"][["state_id", "domain_id", "position"]]
    ):
        raise ValueError("CATH feature and teacher metadata orders disagree")
    final = np.flatnonzero(
        cath["observability_split"].isin(stability.calibration.final_training_splits).to_numpy()
    )
    bundle = _feature_bundle(
        stability,
        cath,
        queries,
        final,
        "cross_platform_final",
        panel_store_root=config.paths.storage_dir / "representations",
        panel_profile_path=config.paths.run_dir / "strong_control/panel_profiles.parquet",
    )
    features = bundle["panel"].astype(np.float32)
    np.save(paths["features"], features)
    write_json(paths["feature_names"], bundle["feature_names"])
    write_parquet(paths["environment_audit"], bundle["environment_audit"])

    wild = queries["wild_type"].map(AA_TO_INDEX).to_numpy(dtype=int)
    scores = pd.read_parquet(config.paths.run_dir / "teacher_scores/scores.parquet")
    observed_actions = {
        teacher: _anchor(_aligned_teacher(scores, queries, teacher), wild)
        for teacher in stability.calibration.teacher_ids
    }
    temperatures = read_json(stability.paths.run_dir / "calibration/selection.json")[
        "final_parameters"
    ]["temperatures"]
    components: dict[str, np.ndarray] = {}
    for teacher in stability.calibration.teacher_ids:
        action = cath_data["actions"][teacher]
        global_component = _global_component(action[final], cath_data["wild"][final], wild)
        model = joblib.load(
            stability.paths.storage_dir / "strong_control/models" / f"{teacher}.joblib"
        )
        c_plus = _anchor(model.predict(features), wild)
        paired = observed_actions[teacher]
        u_plus = _anchor(paired - global_component - c_plus, wild)
        for name, value in {
            "a": paired,
            "g": global_component,
            "c_plus": c_plus,
            "u_plus": u_plus,
        }.items():
            components[f"{teacher}_{name}"] = value.astype(np.float32)
            components[f"temperature_{teacher}_{name}"] = (
                value / float(temperatures[teacher])
            ).astype(np.float32)
    for name in ("a", "g", "c_plus", "u_plus"):
        components[f"unscaled_consensus_{name}"] = np.mean(
            np.stack(
                [components[f"{teacher}_{name}"] for teacher in stability.calibration.teacher_ids]
            ),
            axis=0,
        ).astype(np.float32)
        components[f"temperature_consensus_{name}"] = np.mean(
            np.stack(
                [
                    components[f"temperature_{teacher}_{name}"]
                    for teacher in stability.calibration.teacher_ids
                ]
            ),
            axis=0,
        ).astype(np.float32)
    sequence_logp = normalize_log_probabilities(
        _aligned_store(
            config.paths.storage_dir / "representations/esm2_150M",
            queries,
            "log_probabilities.npy",
        ).astype(float)
    )
    components["esm2_150M_sequence_action"] = _anchor(sequence_logp, wild).astype(np.float32)
    np.savez_compressed(paths["components"], **components)
    write_json(
        paths["manifest"],
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "status": "FROZEN_CROSS_PLATFORM_SCORES_MATERIALIZED_BEFORE_OUTCOME_OPENING",
            "protocol_lock": str(config.paths.run_dir / "protocol_lock.json"),
            "stability_labels_read": False,
            "selected_domains": int(queries["domain_id"].nunique()),
            "query_rows": len(queries),
            "feature_count": int(features.shape[1]),
            "temperatures": temperatures,
            "component_names": sorted(components),
            "component_path": str(paths["components"]),
        },
    )
    return paths


def evaluate_external_validation(config: ExternalValidationConfig) -> dict[str, Path | str]:
    """Open FireProt ddG only after frozen scores exist and run the locked test."""

    _require_lock(config)
    score_manifest = config.paths.run_dir / "strong_control/score_manifest.json"
    component_path = config.paths.storage_dir / "strong_control/components.npz"
    if not score_manifest.exists() or not component_path.exists():
        raise FileNotFoundError("frozen cross-platform scores must exist before outcome opening")
    if read_json(score_manifest).get("stability_labels_read") is not False:
        raise RuntimeError("score manifest does not certify outcome-blind construction")
    panel = config.paths.run_dir / "panel"
    queries = pd.read_parquet(panel / "query_rows.parquet")
    mutation_index = pd.read_parquet(panel / "mutation_index.parquet")
    variants = _open_fireprot_labels(config.paths.source_csv, mutation_index)
    components = np.load(component_path)
    query_index = queries[["domain_id", "position"]].copy()
    query_index["query_row"] = np.arange(len(query_index), dtype=int)
    variant_components = variants.merge(
        query_index, on=["domain_id", "position"], validate="many_to_one"
    )
    query_rows = variant_components["query_row"].to_numpy(dtype=int)
    mutants = variant_components["mutant"].map(AA_TO_INDEX).to_numpy(dtype=int)
    methods = _variant_methods(components, query_rows, mutants)
    for method, values in methods.items():
        variant_components[method] = values
    variant_components = variant_components.drop(columns="query_row")
    registry = _method_registry()
    domain_metrics = _domain_metrics(variant_components, registry)
    contrasts = _contrast_registry()
    domain_contrasts = _domain_contrasts(domain_metrics, contrasts)
    summary = _contrast_summary(domain_contrasts, contrasts, config)
    gates, decision = _decision(summary, config)
    output = config.paths.run_dir / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "variants": variants,
        "variant_components": variant_components,
        "method_registry": registry,
        "domain_metrics": domain_metrics,
        "contrast_registry": contrasts,
        "domain_contrasts": domain_contrasts,
        "contrast_summary": summary,
        "gate_checks": gates,
        "decision": decision,
    }
    paths = {name: output / f"{name}.parquet" for name in tables}
    for name, table in tables.items():
        write_parquet(paths[name], table)
    manifest = output / "manifest.json"
    write_json(
        manifest,
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "status": "EXTERNAL_VALIDATION_COMPLETE",
            "protocol_lock": str(config.paths.run_dir / "protocol_lock.json"),
            "score_manifest": str(score_manifest),
            "scores_frozen_before_outcomes": True,
            "outcomes_opened_in_this_step": True,
            "effect_convention": "effect=-median(FireProt ddG); positive means stabilizing",
            "routing_evaluated": False,
            "changes_primary_decision": False,
            "decision": str(decision.iloc[0]["decision"]),
            "tables": {
                name: {"path": str(path), "rows": len(tables[name])} for name, path in paths.items()
            },
        },
    )
    return {**paths, "manifest": manifest, "decision_name": str(decision.iloc[0]["decision"])}


def _open_fireprot_labels(path: Path, mutation_index: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(
        path,
        usecols=[
            "pdb_id_corrected",
            "chain",
            "pdb_position",
            "wild_type",
            "mutation",
            "ddG",
        ],
    )
    raw["domain_id"] = (
        raw["pdb_id_corrected"].astype(str).str.upper() + ":" + raw["chain"].astype(str)
    )
    raw["position"] = pd.to_numeric(raw["pdb_position"], errors="coerce")
    raw["ddG"] = pd.to_numeric(raw["ddG"], errors="coerce")
    raw["wild_type"] = raw["wild_type"].astype(str).str.upper()
    raw["mutant"] = raw["mutation"].astype(str).str.upper()
    raw = raw.dropna(subset=["position", "ddG"])
    raw["position"] = raw["position"].astype(int)
    keys = ["domain_id", "position", "wild_type", "mutant"]
    aggregate = raw.groupby(keys, as_index=False, observed=True).agg(
        fireprot_ddg=("ddG", "median"),
        measurement_count=("ddG", "size"),
        measurement_min_ddg=("ddG", "min"),
        measurement_max_ddg=("ddG", "max"),
    )
    selected = mutation_index.merge(aggregate, on=keys, how="left", validate="one_to_one")
    if selected["fireprot_ddg"].isna().any():
        missing = selected.loc[selected["fireprot_ddg"].isna(), keys].head().to_dict("records")
        raise ValueError(f"selected FireProt substitutions lack finite ddG: {missing}")
    selected["effect"] = -selected["fireprot_ddg"]
    selected["effect_name"] = "negative_FireProt_ddG"
    selected["evaluation_population"] = "fireprot_hf_cross_platform"
    return selected.sort_values(["domain_id", "position", "mutant"], ignore_index=True)


def _variant_methods(
    components: Any, query_rows: np.ndarray, mutants: np.ndarray
) -> dict[str, np.ndarray]:
    def values(name: str) -> np.ndarray:
        return np.asarray(components[name], dtype=float)[query_rows, mutants]

    sequence = values("esm2_150M_sequence_action")
    temperature_a = values("temperature_consensus_a")
    temperature_gc = values("temperature_consensus_g") + values("temperature_consensus_c_plus")
    unscaled_a = values("unscaled_consensus_a")
    unscaled_gc = values("unscaled_consensus_g") + values("unscaled_consensus_c_plus")
    result = {
        "temperature_consensus_action": temperature_a,
        "temperature_consensus_g_plus_c_plus": temperature_gc,
        "sequence_plus_temperature_action": sequence + temperature_a,
        "sequence_plus_temperature_g_plus_c_plus": sequence + temperature_gc,
        "unscaled_consensus_action": unscaled_a,
        "unscaled_consensus_g_plus_c_plus": unscaled_gc,
        "esm2_150M_sequence_action": sequence,
    }
    for teacher in ("mif", "esm_if1", "proteinmpnn"):
        result[f"{teacher}_action"] = values(f"{teacher}_a")
        result[f"{teacher}_g_plus_c_plus"] = values(f"{teacher}_g") + values(f"{teacher}_c_plus")
    return result


def _method_registry() -> pd.DataFrame:
    rows = [
        ("temperature_consensus_action", "primary_structure_action", True),
        ("temperature_consensus_g_plus_c_plus", "primary_sequence_control", False),
        ("sequence_plus_temperature_action", "matched_sequence_sensitivity", True),
        (
            "sequence_plus_temperature_g_plus_c_plus",
            "matched_sequence_control_sensitivity",
            False,
        ),
        ("unscaled_consensus_action", "simplification_sensitivity", True),
        ("unscaled_consensus_g_plus_c_plus", "simplification_control", False),
        ("esm2_150M_sequence_action", "sequence_reference", False),
    ]
    rows.extend(
        (f"{teacher}_{suffix}", f"single_teacher_{suffix}", suffix == "action")
        for teacher in ("mif", "esm_if1", "proteinmpnn")
        for suffix in ("action", "g_plus_c_plus")
    )
    return pd.DataFrame(rows, columns=["method", "role", "test_time_structure"])


def _domain_metrics(components: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain_id, frame in components.groupby("domain_id", sort=True, observed=True):
        observed = frame["effect"].to_numpy(dtype=float)
        k = max(1, int(np.ceil(len(frame) * 0.10)))
        for method in registry["method"]:
            predicted = frame[method].to_numpy(dtype=float)
            rows.append(
                {
                    "domain_id": domain_id,
                    "method": method,
                    "n_variants": len(frame),
                    "spearman": _spearman(predicted, observed),
                    "ndcg10": _ndcg(predicted, observed, k=k),
                }
            )
    return pd.DataFrame(rows)


def _contrast_registry() -> pd.DataFrame:
    rows = [
        (
            "temperature_action_vs_g_plus_c_plus",
            "temperature_consensus_action",
            "temperature_consensus_g_plus_c_plus",
            "primary",
        ),
        (
            "sequence_plus_temperature_action_vs_g_plus_c_plus",
            "sequence_plus_temperature_action",
            "sequence_plus_temperature_g_plus_c_plus",
            "matched_sequence_sensitivity",
        ),
        (
            "unscaled_action_vs_g_plus_c_plus",
            "unscaled_consensus_action",
            "unscaled_consensus_g_plus_c_plus",
            "simplification_sensitivity",
        ),
    ]
    rows.extend(
        (
            f"{teacher}_action_vs_g_plus_c_plus",
            f"{teacher}_action",
            f"{teacher}_g_plus_c_plus",
            "single_teacher_sensitivity",
        )
        for teacher in ("mif", "esm_if1", "proteinmpnn")
    )
    return pd.DataFrame(rows, columns=["contrast", "method", "comparator", "role"])


def _domain_contrasts(metrics: pd.DataFrame, contrasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for contrast in contrasts.itertuples(index=False):
        method = metrics.loc[
            metrics["method"].eq(contrast.method),
            ["domain_id", "n_variants", "spearman", "ndcg10"],
        ]
        comparator = metrics.loc[
            metrics["method"].eq(contrast.comparator),
            ["domain_id", "n_variants", "spearman", "ndcg10"],
        ].rename(
            columns={
                "spearman": "comparator_spearman",
                "ndcg10": "comparator_ndcg10",
            }
        )
        merged = method.merge(comparator, on=["domain_id", "n_variants"], validate="one_to_one")
        merged["contrast"] = contrast.contrast
        merged["method"] = contrast.method
        merged["comparator"] = contrast.comparator
        merged["role"] = contrast.role
        merged["spearman_margin"] = merged["spearman"] - merged["comparator_spearman"]
        merged["ndcg10_margin"] = merged["ndcg10"] - merged["comparator_ndcg10"]
        merged["stratum"] = "cross_platform"
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def _contrast_summary(
    domain: pd.DataFrame, contrasts: pd.DataFrame, config: ExternalValidationConfig
) -> pd.DataFrame:
    rows = []
    for contrast_index, contrast in enumerate(contrasts.itertuples(index=False)):
        frame = domain.loc[domain["contrast"].eq(contrast.contrast)]
        for metric_index, metric in enumerate(("spearman", "ndcg10")):
            rows.append(
                {
                    "contrast": contrast.contrast,
                    "method": contrast.method,
                    "comparator": contrast.comparator,
                    "role": contrast.role,
                    "metric": metric,
                    **stratified_domain_bootstrap(
                        frame,
                        f"{metric}_margin",
                        replicates=config.inference.bootstrap_replicates,
                        confidence_level=config.inference.confidence_level,
                        seed=config.seed + contrast_index * 100 + metric_index,
                    ),
                }
            )
    return pd.DataFrame(rows)


def _decision(
    summary: pd.DataFrame, config: ExternalValidationConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = summary.loc[summary["contrast"].eq("temperature_action_vs_g_plus_c_plus")].set_index(
        "metric"
    )
    spearman = primary.loc["spearman"]
    ndcg = primary.loc["ndcg10"]
    checks = [
        {
            "gate": "minimum_evaluable_domains",
            "value": int(spearman["n_domains"]),
            "threshold": config.panel.minimum_selected_domains,
            "passed": int(spearman["n_domains"]) >= config.panel.minimum_selected_domains,
        },
        {
            "gate": "spearman_margin_ci_lower_positive",
            "value": float(spearman["ci_low"]),
            "threshold": 0.0,
            "passed": float(spearman["ci_low"]) > 0,
        },
        {
            "gate": "positive_spearman_domain_fraction",
            "value": float(spearman["positive_domain_fraction"]),
            "threshold": config.inference.minimum_positive_domain_fraction,
            "passed": float(spearman["positive_domain_fraction"])
            >= config.inference.minimum_positive_domain_fraction,
        },
        {
            "gate": "ndcg10_margin_point_positive",
            "value": float(ndcg["estimate"]),
            "threshold": 0.0,
            "passed": float(ndcg["estimate"]) > 0,
        },
    ]
    gates = pd.DataFrame(checks)
    passed = bool(gates["passed"].all())
    decision_name = (
        "CROSS_PLATFORM_STRUCTURE_ACTION_BEYOND_CPLUS_CONFIRMED"
        if passed
        else "CROSS_PLATFORM_STRUCTURE_ACTION_BEYOND_CPLUS_NOT_CONFIRMED"
    )
    decision = pd.DataFrame(
        [
            {
                "decision": decision_name,
                "passed": passed,
                "stability_decision_modified": False,
                "routing_authorized": False,
                "evaluable_domains": int(spearman["n_domains"]),
                "primary_spearman_margin": float(spearman["estimate"]),
                "primary_spearman_ci_low": float(spearman["ci_low"]),
                "primary_spearman_ci_high": float(spearman["ci_high"]),
                "primary_ndcg10_margin": float(ndcg["estimate"]),
            }
        ]
    )
    return gates, decision


def _require_lock(config: ExternalValidationConfig) -> None:
    lock = read_json(config.paths.run_dir / "protocol_lock.json")
    if lock.get("status") != config.status:
        raise RuntimeError("cross-platform protocol is not frozen")
