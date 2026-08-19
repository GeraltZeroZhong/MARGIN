"""Teacher-lineage, architecture-scale, and control-margin audits for generalization study."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from margin.provenance import runtime_manifest, table_manifest, write_json, write_parquet
from margin.studies.generalization.config import GeneralizationStudyConfig
from margin.studies.generalization.targets import (
    LINEAGE_TARGETS,
    load_generalization_residual_dataset,
)
from margin.studies.observability.probes import prediction_metrics, ridge_predict, shuffled_target
from margin.studies.observability.stats import domain_sensitivity_summary

METRICS = (
    "jsd_reduction_nats",
    "cross_entropy_reduction_nats",
    "residual_cosine",
    "candidate_rank_agreement",
    "top3_overlap",
)


def run_cath_audits(config: GeneralizationStudyConfig) -> dict[str, Path]:
    """Run fixed CARP lineage and five-model architecture comparisons."""

    output = config.paths.run_dir / "cath_audit"
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_generalization_residual_dataset(config)
    role = dataset.metadata["observability_split"]
    train = np.flatnonzero(role.isin(["development_train", "development_validation"]).to_numpy())
    test = np.flatnonzero(role.eq("locked_test").to_numpy())
    target_id = config.architecture.primary_target
    target = dataset.residuals[target_id]
    features = {
        model.model_id: load_features(
            config.paths.storage_dir / "architecture" / model.model_id,
            dataset.metadata,
        )
        for model in config.architecture.models
    }

    lineage_frames: list[pd.DataFrame] = []
    carp = features["carp_640M"]
    for lineage_target in (*LINEAGE_TARGETS, "mif_paired_minus_rewired"):
        prediction = ridge_predict(
            carp[train],
            dataset.residuals[lineage_target][train],
            carp[test],
            alpha=config.architecture.ridge_alpha,
            rank=config.architecture.rrr_rank,
        )
        lineage_frames.append(
            prediction_metrics(
                dataset,
                lineage_target,
                test,
                prediction,
                model_id="carp_640M",
                probe="fixed_rrr",
                target_rank=config.architecture.rrr_rank,
                evaluation_split="observability_locked_test_reused_postdecision",
                control="observed",
                repeat=0,
                moved_fraction=1.0,
            )
        )
        print(f"lineage_target={lineage_target}", flush=True)

    architecture_frames: list[pd.DataFrame] = []
    for model_index, model in enumerate(config.architecture.models):
        matrix = features[model.model_id]
        for probe, rank in (("fixed_rrr", config.architecture.rrr_rank), ("fixed_ridge", None)):
            prediction = ridge_predict(
                matrix[train],
                target[train],
                matrix[test],
                alpha=config.architecture.ridge_alpha,
                rank=rank,
            )
            architecture_frames.append(
                prediction_metrics(
                    dataset,
                    target_id,
                    test,
                    prediction,
                    model_id=model.model_id,
                    family=model.family,
                    scale_millions=model.scale_millions,
                    probe=probe,
                    target_rank=rank if rank is not None else np.nan,
                    evaluation_split="observability_locked_test_reused_postdecision",
                    control="observed",
                    repeat=0,
                    moved_fraction=1.0,
                )
            )
        rng = np.random.default_rng(config.seed + 1000 * (model_index + 1))
        for control in config.architecture.controls:
            for repeat in range(config.architecture.control_repeats):
                shuffled, moved = shuffled_target(
                    dataset.metadata,
                    target,
                    train,
                    control,
                    rng,
                )
                prediction = ridge_predict(
                    matrix[train],
                    shuffled,
                    matrix[test],
                    alpha=config.architecture.ridge_alpha,
                    rank=config.architecture.rrr_rank,
                )
                architecture_frames.append(
                    prediction_metrics(
                        dataset,
                        target_id,
                        test,
                        prediction,
                        model_id=model.model_id,
                        family=model.family,
                        scale_millions=model.scale_millions,
                        probe="fixed_rrr",
                        target_rank=config.architecture.rrr_rank,
                        evaluation_split="observability_locked_test_reused_postdecision",
                        control=control,
                        repeat=repeat,
                        moved_fraction=moved,
                    )
                )
        print(f"architecture_model={model.model_id}", flush=True)

    lineage_rows = pd.concat(lineage_frames, ignore_index=True)
    architecture_rows = pd.concat(architecture_frames, ignore_index=True)
    lineage_summary, lineage_domains = summarize(lineage_rows, config, seed_offset=100_000)
    architecture_summary, architecture_domains = summarize(
        architecture_rows, config, seed_offset=200_000
    )
    margins, margin_summary = control_margins(architecture_domains, config)
    decisions = architecture_decisions(architecture_summary, margin_summary, config)
    tables = {
        "lineage_rows": lineage_rows,
        "lineage_summary": lineage_summary,
        "lineage_domain_estimates": lineage_domains,
        "architecture_rows": architecture_rows,
        "architecture_summary": architecture_summary,
        "architecture_domain_estimates": architecture_domains,
        "control_margins": margins,
        "control_margin_summary": margin_summary,
        "architecture_decisions": decisions,
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
            "foundation_decision_modified": False,
            "evaluation_split": "observability_locked_test_reused_postdecision",
            "primary_target": target_id,
            "artifacts": [table_manifest(paths[name], table) for name, table in tables.items()],
        },
    )
    paths["manifest"] = manifest_path
    return paths


def load_features(directory: Path, metadata: pd.DataFrame) -> np.ndarray:
    """Align a compact generalization study representation store to target metadata."""

    keys = pd.read_parquet(directory / "keys.parquet")
    keys = keys.copy()
    keys["feature_row"] = np.arange(len(keys), dtype=int)
    columns = ["state_id", "domain_id", "position"]
    aligned = metadata[columns].merge(keys, on=columns, validate="one_to_one")
    if len(aligned) != len(metadata):
        raise ValueError(f"representation store does not cover target rows: {directory}")
    store = np.load(directory / "representations.npy", mmap_mode="r")
    values = np.asarray(store[aligned["feature_row"].to_numpy(dtype=int)], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"representation store contains non-finite values: {directory}")
    return values


def summarize(
    rows: pd.DataFrame,
    config: GeneralizationStudyConfig,
    *,
    seed_offset: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = [
        column
        for column in (
            "target_id",
            "model_id",
            "family",
            "scale_millions",
            "probe",
            "target_rank",
            "evaluation_split",
            "environment_route",
            "control",
            "repeat",
        )
        if column in rows
    ]
    summaries: list[pd.DataFrame] = []
    domains: list[pd.DataFrame] = []
    for group_index, (key, frame) in enumerate(rows.groupby(labels, observed=True, dropna=False)):
        values = key if isinstance(key, tuple) else (key,)
        group_labels = dict(zip(labels, values, strict=True))
        for metric_index, metric in enumerate(METRICS):
            domain, summary = domain_sensitivity_summary(
                frame,
                metric,
                confidence_level=config.inference.confidence_level,
                wild_replicates=config.inference.bootstrap_replicates,
                seed=config.seed + seed_offset + 100 * group_index + metric_index,
            )
            for name, value in group_labels.items():
                domain[name] = value
                summary[name] = value
            domain["metric"] = metric
            summary["metric"] = metric
            domains.append(domain)
            summaries.append(summary)
    return pd.concat(summaries, ignore_index=True), pd.concat(domains, ignore_index=True)


def control_margins(
    domains: pd.DataFrame,
    config: GeneralizationStudyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subtract the globally strongest registered control from each observed domain estimate."""

    selected = domains.loc[
        domains["metric"].eq("jsd_reduction_nats")
        & domains["probe"].eq("fixed_rrr")
        & domains["target_id"].eq(config.architecture.primary_target)
    ]
    margin_frames: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []
    for model_index, (model_id, frame) in enumerate(selected.groupby("model_id", observed=True)):
        observed = frame.loc[frame["control"].eq("observed"), ["domain_id", "estimate"]].rename(
            columns={"estimate": "observed_jsd_reduction_nats"}
        )
        controls = frame.loc[frame["control"].ne("observed")].copy()
        aggregate = (
            controls.groupby(["control", "repeat"], observed=True)["estimate"]
            .mean()
            .sort_values(ascending=False)
        )
        strongest_control, strongest_repeat = aggregate.index[0]
        strongest = controls.loc[
            controls["control"].eq(strongest_control) & controls["repeat"].eq(strongest_repeat),
            ["domain_id", "estimate"],
        ].rename(columns={"estimate": "strongest_control_jsd_reduction_nats"})
        margin = observed.merge(strongest, on="domain_id", validate="one_to_one")
        margin["control_unique_margin_nats"] = (
            margin["observed_jsd_reduction_nats"] - margin["strongest_control_jsd_reduction_nats"]
        )
        margin["model_id"] = model_id
        margin["strongest_control"] = strongest_control
        margin["strongest_repeat"] = int(strongest_repeat)
        domain, summary = domain_sensitivity_summary(
            margin.rename(columns={"control_unique_margin_nats": "value"}),
            "value",
            confidence_level=config.inference.confidence_level,
            wild_replicates=config.inference.bootstrap_replicates,
            seed=config.seed + 300_000 + model_index,
        )
        del domain
        summary["model_id"] = model_id
        summary["strongest_control"] = strongest_control
        summary["strongest_repeat"] = int(strongest_repeat)
        margin_frames.append(margin)
        summaries.append(summary)
    return pd.concat(margin_frames, ignore_index=True), pd.concat(summaries, ignore_index=True)


def architecture_decisions(
    summary: pd.DataFrame,
    margin_summary: pd.DataFrame,
    config: GeneralizationStudyConfig,
) -> pd.DataFrame:
    rows = []
    for model in config.architecture.models:
        observed = summary.loc[
            summary["model_id"].eq(model.model_id)
            & summary["probe"].eq("fixed_rrr")
            & summary["control"].eq("observed")
        ]
        jsd = observed.loc[observed["metric"].eq("jsd_reduction_nats")].iloc[0]
        cosine = observed.loc[observed["metric"].eq("residual_cosine")].iloc[0]
        margin = margin_summary.loc[margin_summary["model_id"].eq(model.model_id)].iloc[0]
        pass_jsd = bool(
            jsd["estimate"] >= config.inference.minimum_jsd_reduction_nats
            and (not config.inference.require_positive_ci_lower or jsd["wild_ci_low"] > 0)
        )
        pass_cosine = bool(cosine["estimate"] >= config.inference.minimum_residual_cosine)
        pass_margin = bool(
            margin["estimate"] > 0
            and (
                not config.inference.require_positive_control_margin_ci_lower
                or margin["wild_ci_low"] > 0
            )
        )
        rows.append(
            {
                "model_id": model.model_id,
                "family": model.family,
                "scale_millions": model.scale_millions,
                "jsd_reduction_nats": float(jsd["estimate"]),
                "jsd_ci_low": float(jsd["wild_ci_low"]),
                "residual_cosine": float(cosine["estimate"]),
                "control_unique_margin_nats": float(margin["estimate"]),
                "control_margin_ci_low": float(margin["wild_ci_low"]),
                "strongest_control": margin["strongest_control"],
                "pass_jsd": pass_jsd,
                "pass_cosine": pass_cosine,
                "pass_control_margin": pass_margin,
                "passed": pass_jsd and pass_cosine and pass_margin,
            }
        )
    return pd.DataFrame(rows)
