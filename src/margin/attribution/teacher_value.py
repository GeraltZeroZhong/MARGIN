"""Teacher action-value, specificity, agreement, and external-DMS audits."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from margin.attribution.metrics import (
    cluster_bootstrap_mean,
    cluster_bootstrap_statistic,
    grouped_cluster_means,
    rowwise_cosine,
    rowwise_jsd,
    rowwise_topk_overlap,
    vector_spearman,
)
from margin.config import ProjectConfig
from margin.constants import AA_ALPHABET, AA_TO_INDEX
from margin.state_sampling.bank import StateBank
from margin.teachers.cache import TeacherScoreCache
from margin.teachers.schema import logp_columns, validate_score_table


@dataclass(frozen=True)
class TeacherValueAudit:
    position_metrics: pd.DataFrame
    nll_summary: pd.DataFrame
    ranking_summary: pd.DataFrame
    environment_summary: pd.DataFrame
    specificity_positions: pd.DataFrame
    specificity_summary: pd.DataFrame
    agreement_summary: pd.DataFrame
    dms_predictions: pd.DataFrame
    dms_summary: pd.DataFrame
    dms_coverage: pd.DataFrame


def audit_teacher_value(
    cache: TeacherScoreCache,
    bank: StateBank,
    config: ProjectConfig,
    dms: pd.DataFrame | None = None,
) -> TeacherValueAudit:
    """Run every foundation audit teacher-value audit against the canonical score matrix."""

    validate_score_table(cache.scores)
    position_metrics = _position_metrics(cache.scores, bank, config)
    nll_summary = _nll_summary(position_metrics, config)
    ranking_summary = _ranking_summary(position_metrics, config)
    environment_summary = _environment_summary(position_metrics, config)
    specificity_positions = _specificity_positions(position_metrics, config)
    specificity_summary = _specificity_summary(specificity_positions, config)
    agreement_summary = _agreement_summary(position_metrics, config)
    dms_predictions, dms_summary, dms_coverage = _dms_audit(cache.scores, bank, dms, config)
    return TeacherValueAudit(
        position_metrics=position_metrics,
        nll_summary=nll_summary,
        ranking_summary=ranking_summary,
        environment_summary=environment_summary,
        specificity_positions=specificity_positions,
        specificity_summary=specificity_summary,
        agreement_summary=agreement_summary,
        dms_predictions=dms_predictions,
        dms_summary=dms_summary,
        dms_coverage=dms_coverage,
    )


def _position_metrics(scores: pd.DataFrame, bank: StateBank, config: ProjectConfig) -> pd.DataFrame:
    position_metadata = bank.positions[
        [
            "state_id",
            "domain_id",
            "position",
            "native_aa",
            "current_aa",
            "is_corrupted",
            "dataset",
            "analysis_role",
            "eligible_for_training",
            "burial",
            "secondary_structure",
            "contact_class",
            "rsa",
            "contact_degree",
            "conservation_score",
            "conservation_class",
        ]
    ]
    state_metadata = bank.states[
        [
            "state_id",
            "state_kind",
            "requested_corruption_ratio",
            "corruption_ratio",
            "edit_distance_fraction",
            "student_entropy",
            "student_top1_margin",
            "timestep",
            "scaffold_compatibility",
        ]
    ]
    metrics = scores.merge(
        position_metadata,
        on=["state_id", "domain_id", "position"],
        validate="many_to_one",
    ).merge(state_metadata, on="state_id", validate="many_to_one")
    values = metrics[logp_columns()].to_numpy(dtype=float)
    native_index = metrics["native_aa"].map(AA_TO_INDEX).to_numpy(dtype=int)
    metrics["native_logp"] = values[np.arange(len(values)), native_index]
    metrics["native_nll"] = -metrics["native_logp"]
    metrics["native_rank"] = 1 + np.sum(values > metrics["native_logp"].to_numpy()[:, None], axis=1)
    metrics["native_top1"] = metrics["native_rank"] == 1
    sequence = metrics.loc[
        metrics["teacher_id"] == config.audit.sequence_teacher_id,
        ["state_id", "domain_id", "position", "native_logp"],
    ].rename(columns={"native_logp": "sequence_native_logp"})
    if sequence.duplicated(["state_id", "domain_id", "position"]).any():
        raise ValueError("sequence teacher must have one score row per state position")
    metrics = metrics.merge(
        sequence,
        on=["state_id", "domain_id", "position"],
        how="left",
        validate="many_to_one",
    )
    if metrics["sequence_native_logp"].isna().any():
        raise ValueError("teacher cache lacks sequence baseline coverage")
    metrics["advantage_nats"] = metrics["native_logp"] - metrics["sequence_native_logp"]
    return metrics


def _nll_summary(metrics: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    group_columns = [
        "teacher_id",
        "structure_role",
        "analysis_role",
        "state_kind",
        "requested_corruption_ratio",
    ]
    result = grouped_cluster_means(
        metrics,
        group_columns,
        "native_nll",
        config.audit.cluster_column,
        config.audit.bootstrap_replicates,
        config.audit.confidence_level,
        config.seed + 100,
    )
    result.insert(len(group_columns), "metric", "native_nll")
    return result


def _ranking_summary(metrics: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    group_columns = ["teacher_id", "structure_role", "analysis_role"]
    rows: list[pd.DataFrame] = []
    for metric, source in (
        ("native_aar", "native_top1"),
        ("native_mrr", "reciprocal_rank"),
    ):
        frame = metrics.copy()
        if source == "reciprocal_rank":
            frame[source] = 1.0 / frame["native_rank"].astype(float)
        summary = grouped_cluster_means(
            frame,
            group_columns,
            source,
            config.audit.cluster_column,
            config.audit.bootstrap_replicates,
            config.audit.confidence_level,
            config.seed + 200 + len(rows) * 1000,
        )
        summary.insert(len(group_columns), "metric", metric)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def _environment_summary(metrics: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    paired = metrics.loc[
        (metrics["structure_role"] == config.audit.paired_role)
        & (metrics["teacher_id"] != config.audit.sequence_teacher_id)
    ].copy()
    dimensions = _environment_long(paired)
    return grouped_cluster_means(
        dimensions,
        [
            "teacher_id",
            "analysis_role",
            "state_kind",
            "requested_corruption_ratio",
            "environment_axis",
            "environment",
        ],
        "advantage_nats",
        config.audit.cluster_column,
        config.audit.bootstrap_replicates,
        config.audit.confidence_level,
        config.seed + 300,
    )


def _specificity_positions(metrics: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    structure = metrics.loc[metrics["teacher_id"] != config.audit.sequence_teacher_id].copy()
    paired = structure.loc[
        structure["structure_role"] == config.audit.paired_role,
        [
            "state_id",
            "domain_id",
            "position",
            "teacher_id",
            "analysis_role",
            "native_logp",
            "state_kind",
            "requested_corruption_ratio",
            "burial",
            "secondary_structure",
            "contact_class",
            "conservation_class",
        ],
    ].rename(columns={"native_logp": "paired_native_logp"})
    decoy = structure.loc[
        structure["structure_role"].isin(config.audit.decoy_roles),
        [
            "state_id",
            "domain_id",
            "position",
            "teacher_id",
            "structure_role",
            "structure_id",
            "native_logp",
        ],
    ].rename(columns={"structure_role": "decoy_role", "native_logp": "decoy_native_logp"})
    result = decoy.merge(
        paired,
        on=["state_id", "domain_id", "position", "teacher_id"],
        validate="many_to_one",
    )
    result["paired_decoy_lift_nats"] = result["paired_native_logp"] - result["decoy_native_logp"]
    return result


def _specificity_summary(specificity: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    if specificity.empty:
        return pd.DataFrame()
    long = _environment_long(specificity)
    return grouped_cluster_means(
        long,
        [
            "teacher_id",
            "decoy_role",
            "analysis_role",
            "state_kind",
            "requested_corruption_ratio",
            "environment_axis",
            "environment",
        ],
        "paired_decoy_lift_nats",
        config.audit.cluster_column,
        config.audit.bootstrap_replicates,
        config.audit.confidence_level,
        config.seed + 400,
    )


def _agreement_summary(metrics: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    paired = metrics.loc[
        (metrics["structure_role"] == config.audit.paired_role)
        & (metrics["teacher_id"] != config.audit.sequence_teacher_id)
    ]
    teacher_ids = sorted(paired["teacher_id"].unique())
    rows: list[dict[str, Any]] = []
    key = ["state_id", "domain_id", "position", "analysis_role"]
    columns = logp_columns()
    for pair_index, (left_id, right_id) in enumerate(combinations(teacher_ids, 2)):
        left = paired.loc[paired["teacher_id"] == left_id, [*key, *columns]]
        right = paired.loc[paired["teacher_id"] == right_id, [*key, *columns]]
        joined = left.merge(right, on=key, suffixes=("_left", "_right"), validate="one_to_one")
        if joined.empty:
            continue
        left_values = joined[[f"{column}_left" for column in columns]].to_numpy(dtype=float)
        right_values = joined[[f"{column}_right" for column in columns]].to_numpy(dtype=float)
        centered_left = left_values - left_values.mean(axis=1, keepdims=True)
        centered_right = right_values - right_values.mean(axis=1, keepdims=True)
        values = {
            "candidate_spearman": np.array(
                [vector_spearman(a, b) for a, b in zip(left_values, right_values, strict=True)]
            ),
            "centered_logp_cosine": rowwise_cosine(centered_left, centered_right),
            "jsd_nats": rowwise_jsd(left_values, right_values),
            "top3_overlap": rowwise_topk_overlap(left_values, right_values, 3),
        }
        for role_index, (analysis_role, indices) in enumerate(
            joined.groupby("analysis_role", observed=True).groups.items()
        ):
            positions = joined.index.get_indexer(indices)
            for metric_index, (metric, metric_values) in enumerate(values.items()):
                metric_table = joined.loc[indices, ["domain_id"]].copy()
                metric_table["value"] = metric_values[positions]
                estimate = cluster_bootstrap_mean(
                    metric_table,
                    "value",
                    config.audit.cluster_column,
                    config.audit.bootstrap_replicates,
                    config.audit.confidence_level,
                    config.seed + 500 + pair_index * 100 + role_index * 10_000 + metric_index,
                )
                rows.append(
                    {
                        "teacher_a": left_id,
                        "teacher_b": right_id,
                        "analysis_role": analysis_role,
                        "metric": metric,
                        **estimate,
                    }
                )
    return pd.DataFrame(rows)


def _dms_audit(
    scores: pd.DataFrame,
    bank: StateBank,
    dms: pd.DataFrame | None,
    config: ProjectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_columns = [
        "assay_id",
        "assay_type",
        "source",
        "domain_id",
        "position",
        "wild_type",
        "mutant",
        "effect",
        "analysis_role",
        "teacher_id",
        "predicted_effect",
    ]
    summary_columns = [
        "scope",
        "assay_id",
        "analysis_role",
        "teacher_id",
        "metric",
        "estimate",
        "ci_low",
        "ci_high",
        "n_rows",
        "n_domains",
    ]
    coverage_columns = [
        "assay_id",
        "assay_type",
        "source",
        "domain_id",
        "analysis_role",
        "teacher_id",
        "total_variants",
        "known_position_variants",
        "scored_variants",
        "coverage_fraction",
        "status",
    ]
    if dms is None or dms.empty:
        return (
            pd.DataFrame(columns=prediction_columns),
            pd.DataFrame(columns=summary_columns),
            pd.DataFrame(columns=coverage_columns),
        )
    required = {"assay_id", "domain_id", "position", "wild_type", "mutant", "effect"}
    missing = required - set(dms.columns)
    if missing:
        raise ValueError(f"DMS table is missing columns: {sorted(missing)}")
    if dms.duplicated(["assay_id", "domain_id", "position", "mutant"]).any():
        raise ValueError("DMS variants must have unique assay/domain/position/mutant keys")
    invalid = set(dms["wild_type"]) | set(dms["mutant"])
    invalid -= set(AA_ALPHABET)
    if invalid:
        raise ValueError(f"DMS variants contain noncanonical residues: {sorted(invalid)}")
    dms = dms.copy()
    if "assay_type" not in dms:
        dms["assay_type"] = "unspecified"
    if "source" not in dms:
        dms["source"] = "unspecified"
    numeric_position = pd.to_numeric(dms["position"], errors="coerce")
    if (
        numeric_position.isna().any()
        or not np.equal(numeric_position, np.floor(numeric_position)).all()
    ):
        raise ValueError("DMS positions must be finite integers")
    dms["position"] = numeric_position.astype(int)
    effect = pd.to_numeric(dms["effect"], errors="coerce")
    if not np.isfinite(effect.to_numpy(dtype=float)).all():
        raise ValueError("DMS effects must be finite numbers")
    dms["effect"] = effect.astype(float)

    position_metadata = bank.positions[
        ["domain_id", "position", "native_aa", "analysis_role"]
    ].drop_duplicates()
    if position_metadata.duplicated(["domain_id", "position"]).any():
        raise ValueError("state bank has inconsistent native metadata for a domain position")
    dms = dms.merge(
        position_metadata,
        on=["domain_id", "position"],
        how="left",
        validate="many_to_one",
    )
    known = dms["native_aa"].notna()
    mismatch = known & (dms["wild_type"] != dms["native_aa"])
    if mismatch.any():
        examples = dms.loc[
            mismatch, ["assay_id", "domain_id", "position", "wild_type", "native_aa"]
        ].head(5)
        raise ValueError(
            "DMS wild_type disagrees with the canonical state bank: "
            f"{examples.to_dict(orient='records')}"
        )
    dms["analysis_role"] = dms["analysis_role"].fillna("unknown")

    state_ratio = bank.states.set_index("state_id")["requested_corruption_ratio"]
    minimum_ratio = float(state_ratio.min())
    if not np.isclose(minimum_ratio, 0.0):
        raise ValueError("DMS audit requires a zero-corruption native_reference state")
    eligible_states = set(state_ratio.index[state_ratio == minimum_ratio])
    eligible_scores = scores.loc[
        scores["state_id"].isin(eligible_states)
        & (
            (scores["teacher_id"] == config.audit.sequence_teacher_id)
            | (scores["structure_role"] == config.audit.paired_role)
        )
    ]
    averaged = (
        eligible_scores.groupby(["domain_id", "position", "teacher_id"], observed=True)[
            logp_columns()
        ]
        .mean()
        .reset_index()
    )
    predictions: list[pd.DataFrame] = []
    coverage: list[pd.DataFrame] = []
    coverage_groups = [
        "assay_id",
        "assay_type",
        "source",
        "domain_id",
        "analysis_role",
    ]
    for teacher_id, frame in averaged.groupby("teacher_id", observed=True):
        scored_keys = frame[["domain_id", "position"]].assign(_scored=True)
        coverage_frame = dms.merge(
            scored_keys,
            on=["domain_id", "position"],
            how="left",
            validate="many_to_one",
        )
        coverage_frame["_scored"] = coverage_frame["_scored"].eq(True)
        coverage_frame["_known"] = coverage_frame["native_aa"].notna()
        teacher_coverage = (
            coverage_frame.groupby(coverage_groups, observed=True, dropna=False)
            .agg(
                total_variants=("mutant", "size"),
                known_position_variants=("_known", "sum"),
                scored_variants=("_scored", "sum"),
            )
            .reset_index()
        )
        teacher_coverage["teacher_id"] = teacher_id
        teacher_coverage["coverage_fraction"] = (
            teacher_coverage["scored_variants"] / teacher_coverage["total_variants"]
        )
        teacher_coverage["status"] = np.select(
            [
                teacher_coverage["coverage_fraction"] >= 1.0 - 1e-12,
                teacher_coverage["scored_variants"] > 0,
            ],
            ["complete", "partial"],
            default="unscored",
        )
        coverage.append(teacher_coverage[coverage_columns])
        joined = dms.merge(frame, on=["domain_id", "position"], how="inner", validate="many_to_one")
        if joined.empty:
            continue
        wild_index = joined["wild_type"].map(AA_TO_INDEX).to_numpy(dtype=int)
        mutant_index = joined["mutant"].map(AA_TO_INDEX).to_numpy(dtype=int)
        values = joined[logp_columns()].to_numpy(dtype=float)
        joined["teacher_id"] = teacher_id
        joined["predicted_effect"] = (
            values[np.arange(len(values)), mutant_index]
            - values[np.arange(len(values)), wild_index]
        )
        predictions.append(joined[prediction_columns])
    prediction_table = (
        pd.concat(predictions, ignore_index=True)
        if predictions
        else pd.DataFrame(columns=prediction_columns)
    )
    coverage_table = (
        pd.concat(coverage, ignore_index=True)
        if coverage
        else pd.DataFrame(columns=coverage_columns)
    )
    summary_rows: list[dict[str, Any]] = []
    for (assay_id, analysis_role, teacher_id), frame in prediction_table.groupby(
        ["assay_id", "analysis_role", "teacher_id"], observed=True
    ):
        if len(frame) < config.audit.dms_minimum_variants_per_assay:
            continue
        estimate = vector_spearman(frame["predicted_effect"], frame["effect"])
        summary_rows.append(
            {
                "scope": "assay",
                "assay_id": assay_id,
                "analysis_role": analysis_role,
                "teacher_id": teacher_id,
                "metric": "spearman",
                "estimate": estimate,
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "n_rows": len(frame),
                "n_domains": frame["domain_id"].nunique(),
            }
        )
    for teacher_index, ((analysis_role, teacher_id), frame) in enumerate(
        prediction_table.groupby(["analysis_role", "teacher_id"], observed=True)
    ):
        if len(frame) < config.audit.dms_minimum_variants_per_assay:
            continue

        def correlation(sample: pd.DataFrame) -> float:
            return vector_spearman(sample["predicted_effect"], sample["effect"])

        estimate = cluster_bootstrap_statistic(
            frame,
            ["predicted_effect", "effect"],
            config.audit.cluster_column,
            correlation,
            config.audit.bootstrap_replicates,
            config.audit.confidence_level,
            config.seed + 600 + teacher_index,
        )
        summary_rows.append(
            {
                "scope": "pooled",
                "assay_id": "all",
                "analysis_role": analysis_role,
                "teacher_id": teacher_id,
                "metric": "spearman",
                **estimate,
            }
        )
    return (
        prediction_table,
        pd.DataFrame(summary_rows, columns=summary_columns),
        coverage_table,
    )


def _environment_long(table: pd.DataFrame) -> pd.DataFrame:
    identifiers = [
        column
        for column in table.columns
        if column
        not in {
            "burial",
            "secondary_structure",
            "contact_class",
            "conservation_class",
        }
    ]
    return table.melt(
        id_vars=identifiers,
        value_vars=[
            "burial",
            "secondary_structure",
            "contact_class",
            "conservation_class",
        ],
        var_name="environment_axis",
        value_name="environment",
    )


def load_dms_table(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"configured DMS input does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
    raise ValueError("DMS input must be Parquet, CSV, or TSV")
