"""Evaluation tables for supervised stability study upper bounds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from margin.provenance import runtime_manifest, write_json, write_parquet
from margin.studies.action_validation.evaluation import METRICS, _domain_metrics
from margin.studies.counterfactuals.evaluation import stratified_domain_bootstrap
from margin.studies.stability.config import StabilityStudyConfig
from margin.studies.stability.method_audit import CPLUS_CONTROL, SELECTED_CONSENSUS
from margin.studies.stability.prepare import EXTERNAL_POPULATION, PRIMARY_POPULATION

SUPERVISED_MODELS = ("thermompnn", "spurs")
ROBUST_ZERO_SHOT = "unscaled_equal_action_only"
FAST_ZERO_SHOT = "esm_if1_action_only"


def evaluate_supervised_upper_bounds(config: StabilityStudyConfig) -> dict[str, Path]:
    """Evaluate label-supervised models without mixing them into the zero-shot table."""

    run = config.paths.run_dir
    output = run / "supervised" / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    components = pd.read_parquet(run / "evaluation" / "variant_components.parquet").sort_values(
        "variant_row", ignore_index=True
    )
    predictions: dict[str, np.ndarray] = {}
    split_tables = []
    for model in SUPERVISED_MODELS:
        path = run / "supervised" / model / "predictions.parquet"
        frame = pd.read_parquet(path).sort_values("variant_row", ignore_index=True)
        if not np.array_equal(
            frame["variant_row"].to_numpy(dtype=int),
            components["variant_row"].to_numpy(dtype=int),
        ):
            raise ValueError(f"{model} predictions do not align to stability study variants")
        predictions[model] = frame["predicted_stability"].to_numpy(dtype=float)
        split_tables.append(
            frame[["domain_id", "thermompnn_split"]].drop_duplicates().assign(method=model)
        )
    metrics = _domain_metrics(components, predictions, config.inference.top_fraction)
    split = pd.concat(split_tables, ignore_index=True)
    metrics = metrics.merge(split, on=["method", "domain_id"], validate="one_to_one")
    metrics.loc[
        metrics["evaluation_population"].eq(EXTERNAL_POPULATION),
        "stabilizing_top_10_percent_recall",
    ] = np.nan
    metrics["method_class"] = "supervised_upper_bound"
    summary = _supervised_summary(metrics, config)
    context = _heldout_context(metrics, run)
    status = _method_status(run)
    tables = {
        "domain_metrics": metrics,
        "summary": summary,
        "heldout_context": context,
        "method_status": status,
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
            "schema_version": "stability.supervised_evaluation.v1",
            "analysis_role": "supervised_upper_bounds_separate_from_zero_shot",
            "changes_primary_decision": False,
            "mixed_primary_panel_interpretation": (
                "transductive descriptive upper bound because 20/32 domains have exact "
                "ThermoMPNN/SPURS training-split overlap"
            ),
            "official_test_interpretation": (
                "only two stability study domains have exact-name official test status; estimates "
                "are descriptive and underpowered"
            ),
            "hermes": status.loc[status["method"].eq("hermes")].iloc[0].to_dict(),
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


def _supervised_summary(metrics: pd.DataFrame, config: StabilityStudyConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_index, (model, model_frame) in enumerate(metrics.groupby("method", sort=True)):
        primary = model_frame.loc[model_frame["evaluation_population"].eq(PRIMARY_POPULATION)]
        scopes = [("primary_all_mixed_overlap", primary)]
        scopes.extend(
            (str(split), frame) for split, frame in primary.groupby("thermompnn_split", sort=True)
        )
        for scope_index, (scope, frame) in enumerate(scopes):
            for metric_index, metric in enumerate(METRICS):
                rows.append(
                    {
                        "method": model,
                        "method_class": "supervised_upper_bound",
                        "scope": scope,
                        "metric": metric,
                        "interval_unit": "protein_domain",
                        **stratified_domain_bootstrap(
                            frame,
                            metric,
                            replicates=config.inference.bootstrap_replicates,
                            confidence_level=config.inference.confidence_level,
                            seed=(
                                config.seed
                                + 700_000
                                + model_index * 1000
                                + scope_index * 10
                                + metric_index
                            ),
                        ),
                    }
                )
        external = model_frame.loc[model_frame["evaluation_population"].eq(EXTERNAL_POPULATION)]
        if len(external) != 1:
            raise ValueError(f"{model} must have one external domain")
        for metric in METRICS:
            rows.append(
                {
                    "method": model,
                    "method_class": "supervised_upper_bound",
                    "scope": "external_single_protein",
                    "metric": metric,
                    "interval_unit": "point_only",
                    "estimate": float(external.iloc[0][metric]),
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


def _heldout_context(supervised: pd.DataFrame, run: Path) -> pd.DataFrame:
    zero_shot = pd.read_parquet(run / "method_audit" / "domain_metrics.parquet")
    overlap = pd.read_parquet(run / "method_audit" / "supervised_split_overlap.parquet")[
        ["domain_id", "thermompnn_split"]
    ]
    zero_shot = zero_shot.merge(overlap, on="domain_id", how="left", validate="many_to_one")
    zero_shot = zero_shot.loc[
        zero_shot["method"].isin(
            [ROBUST_ZERO_SHOT, FAST_ZERO_SHOT, SELECTED_CONSENSUS, CPLUS_CONTROL]
        )
    ].copy()
    zero_shot["method_class"] = "zero_shot_context_only"
    selected_supervised = supervised.copy()
    frames = []
    for scope, split_name in (
        ("official_test_two_domains", "test"),
        ("not_listed_six_domains_independence_not_inferred", "not_listed"),
    ):
        for frame in (selected_supervised, zero_shot):
            subset = frame.loc[frame["thermompnn_split"].eq(split_name)].copy()
            subset["scope"] = scope
            frames.append(subset)
    return pd.concat(frames, ignore_index=True)[
        [
            "scope",
            "domain_id",
            "stratum",
            "method",
            "method_class",
            "n_variants",
            *METRICS,
        ]
    ]


def _method_status(run: Path) -> pd.DataFrame:
    rows = []
    for model in SUPERVISED_MODELS:
        manifest = run / "supervised" / model / "manifest.json"
        rows.append(
            {
                "method": model,
                "status": "completed",
                "reason": "official checkpoint scored with exact variant coverage",
                "manifest": str(manifest),
                "table_role": "supervised_upper_bound",
            }
        )
    rows.append(
        {
            "method": "hermes",
            "status": "not_run_missing_licensed_preprocessor",
            "reason": (
                "Official ddG-stability weights are hermes_py only; no licensed PyRosetta "
                "installation or wheel is present. Biopython preprocessing is not "
                "weight-compatible."
            ),
            "manifest": "",
            "table_role": "documented_unavailable_supervised_upper_bound",
        }
    )
    return pd.DataFrame(rows)
