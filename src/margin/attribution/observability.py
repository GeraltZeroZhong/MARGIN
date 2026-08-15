"""CATH-grouped cross-fitted probes for sequence observability of teacher residuals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from margin.attribution.metrics import (
    grouped_cluster_means,
    rowwise_cosine,
    rowwise_jsd,
    rowwise_topk_overlap,
)
from margin.config import ProjectConfig
from margin.constants import AA_ALPHABET
from margin.data_registry.registry import RegistryTables
from margin.state_sampling.bank import StateBank
from margin.teachers.cache import TeacherScoreCache
from margin.teachers.schema import logp_columns


@dataclass(frozen=True)
class ObservabilityAudit:
    row_metrics: pd.DataFrame
    summary: pd.DataFrame
    environment_summary: pd.DataFrame
    feature_manifest: pd.DataFrame


def audit_observability(
    cache: TeacherScoreCache,
    bank: StateBank,
    registry: RegistryTables,
    config: ProjectConfig,
    embeddings: pd.DataFrame | None = None,
) -> ObservabilityAudit:
    """Cross-fit a frozen-feature residual probe under CATH-H and CATH-T holdouts."""

    features, feature_source = _feature_table(bank, config, embeddings)
    examples = _probe_examples(cache, bank, registry, config, features)
    feature_columns = [column for column in examples.columns if column.startswith("feature_")]
    row_tables: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    for level_index, group_level in enumerate(config.observability.group_levels):
        n_groups = int(examples[group_level].nunique())
        if n_groups < max(2, config.observability.minimum_groups_per_split):
            manifest_rows.append(
                {
                    "group_level": group_level,
                    "status": "insufficient_groups",
                    "reason": (
                        f"found {n_groups}; require at least "
                        f"{max(2, config.observability.minimum_groups_per_split)}"
                    ),
                    "feature_source": feature_source,
                    "feature_count": len(feature_columns),
                    "rows": len(examples),
                    "groups": n_groups,
                    "folds": 0,
                }
            )
            continue
        folds = min(config.observability.folds, n_groups)
        row_tables.append(
            _cross_fit_level(
                examples,
                feature_columns,
                group_level,
                folds,
                config,
                config.seed + 1000 * (level_index + 1),
            )
        )
        manifest_rows.append(
            {
                "group_level": group_level,
                "status": "complete",
                "reason": "",
                "feature_source": feature_source,
                "feature_count": len(feature_columns),
                "rows": len(examples),
                "groups": n_groups,
                "folds": folds,
            }
        )
    row_metrics = pd.concat(row_tables, ignore_index=True) if row_tables else _empty_rows()
    summary = _metric_summary(row_metrics, config)
    environment_summary = _environment_summary(row_metrics, config)
    return ObservabilityAudit(
        row_metrics=row_metrics,
        summary=summary,
        environment_summary=environment_summary,
        feature_manifest=pd.DataFrame(manifest_rows),
    )


def load_embeddings(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"configured embedding input does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
    raise ValueError("embedding input must be Parquet, CSV, or TSV")


def build_synthetic_sequence_features(bank: StateBank, context_radius: int) -> pd.DataFrame:
    """Make a deterministic sequence-only hidden-state fixture for software tests."""

    table = bank.positions.sort_values(["state_id", "position"]).copy()
    output = table[["state_id", "domain_id", "position"]].copy()
    for aa in AA_ALPHABET:
        output[f"feature_student_logp_{aa}"] = table[f"student_logp_{aa}"].to_numpy()
        output[f"feature_current_{aa}"] = (table["current_aa"] == aa).astype(float).to_numpy()
    output["feature_entropy"] = table["student_entropy"].to_numpy(dtype=float)
    output["feature_margin"] = table["student_top1_margin"].to_numpy(dtype=float)
    relative_position = table.groupby("state_id", observed=True)["position"].transform(
        lambda values: values / max(1, int(values.max()))
    )
    output["feature_relative_position"] = relative_position.to_numpy(dtype=float)
    output["feature_position_sin"] = np.sin(2 * np.pi * relative_position)
    output["feature_position_cos"] = np.cos(2 * np.pi * relative_position)
    for aa in AA_ALPHABET:
        indicator = (table["current_aa"] == aa).astype(float)
        output[f"feature_context_{aa}"] = (
            indicator.groupby(table["state_id"], observed=True)
            .rolling(2 * context_radius + 1, center=True, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .sort_index()
            .to_numpy()
        )
    return output


def _feature_table(
    bank: StateBank,
    config: ProjectConfig,
    embeddings: pd.DataFrame | None,
) -> tuple[pd.DataFrame, str]:
    if embeddings is None:
        if config.data_mode != "synthetic":
            raise ValueError(
                "real observability audit requires frozen student embeddings_input; "
                "sequence fixture features are synthetic-only"
            )
        return (
            build_synthetic_sequence_features(bank, config.observability.context_radius),
            "synthetic_sequence_only_fixture",
        )
    keys = ["state_id", "domain_id", "position"]
    missing_keys = set(keys) - set(embeddings.columns)
    feature_columns = [column for column in embeddings.columns if column.startswith("feature_")]
    if missing_keys or not feature_columns:
        raise ValueError(
            f"embedding table requires keys {keys} and at least one feature_* column; "
            f"missing keys={sorted(missing_keys)}"
        )
    if embeddings.duplicated(keys).any():
        raise ValueError("embedding keys must be unique")
    if not np.isfinite(embeddings[feature_columns].to_numpy(dtype=float)).all():
        raise ValueError("embedding features contain non-finite values")
    return embeddings[[*keys, *feature_columns]].copy(), "frozen_student_hidden"


def _probe_examples(
    cache: TeacherScoreCache,
    bank: StateBank,
    registry: RegistryTables,
    config: ProjectConfig,
    features: pd.DataFrame,
) -> pd.DataFrame:
    columns = logp_columns()
    keys = ["state_id", "domain_id", "position"]
    sequence = cache.scores.loc[
        cache.scores["teacher_id"] == config.audit.sequence_teacher_id,
        [*keys, *columns],
    ].rename(columns={column: f"sequence_{column}" for column in columns})
    paired = cache.scores.loc[
        (cache.scores["teacher_id"] == config.audit.primary_teacher_id)
        & (cache.scores["structure_role"] == config.audit.paired_role),
        [*keys, *columns],
    ].rename(columns={column: f"teacher_{column}" for column in columns})
    if sequence.duplicated(keys).any() or paired.duplicated(keys).any():
        raise ValueError("observability inputs require unique sequence and paired score rows")
    examples = paired.merge(sequence, on=keys, validate="one_to_one")
    examples = examples.merge(features, on=keys, validate="one_to_one")
    metadata = bank.positions[
        [
            *keys,
            "dataset",
            "analysis_role",
            "eligible_for_training",
            "burial",
            "secondary_structure",
            "contact_class",
            "conservation_class",
        ]
    ].merge(
        bank.states[
            ["state_id", "state_kind", "requested_corruption_ratio", "scaffold_compatibility"]
        ],
        on="state_id",
        validate="many_to_one",
    )
    examples = examples.merge(metadata, on=keys, validate="one_to_one")
    examples = examples.merge(
        registry.domains[["domain_id", "cath_h", "cath_t"]],
        on="domain_id",
        validate="many_to_one",
    )
    if len(examples) != len(paired):
        raise ValueError("frozen embeddings do not cover every primary paired-teacher score")
    return examples


def _cross_fit_level(
    examples: pd.DataFrame,
    feature_columns: list[str],
    group_level: str,
    folds: int,
    config: ProjectConfig,
    seed: int,
) -> pd.DataFrame:
    x = examples[feature_columns].to_numpy(dtype=float)
    sequence_logp = examples[[f"sequence_{column}" for column in logp_columns()]].to_numpy(
        dtype=float
    )
    teacher_logp = examples[[f"teacher_{column}" for column in logp_columns()]].to_numpy(
        dtype=float
    )
    target = teacher_logp - sequence_logp
    target -= target.mean(axis=1, keepdims=True)
    groups = examples[group_level].to_numpy()
    splitter = GroupKFold(n_splits=folds)
    observed = np.full_like(target, np.nan)
    fold_id = np.full(len(examples), -1, dtype=int)
    shuffled = [
        np.full_like(target, np.nan) for _ in range(config.observability.shuffled_control_repeats)
    ]
    rng = np.random.default_rng(seed)
    for fold, (train, test) in enumerate(splitter.split(x, target, groups)):
        model = _ridge_pipeline(config)
        model.fit(x[train], target[train])
        observed[test] = model.predict(x[test])
        fold_id[test] = fold
        for prediction in shuffled:
            permutation = rng.permutation(train)
            control = _ridge_pipeline(config)
            control.fit(x[train], target[permutation])
            prediction[test] = control.predict(x[test])
    tables = [
        _prediction_metrics(
            examples,
            sequence_logp,
            teacher_logp,
            target,
            observed,
            group_level,
            fold_id,
            "observed",
            0,
            config.observability.top_k,
        )
    ]
    for repeat, prediction in enumerate(shuffled):
        tables.append(
            _prediction_metrics(
                examples,
                sequence_logp,
                teacher_logp,
                target,
                prediction,
                group_level,
                fold_id,
                "shuffled",
                repeat,
                config.observability.top_k,
            )
        )
    return pd.concat(tables, ignore_index=True)


def _ridge_pipeline(config: ProjectConfig) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=config.observability.ridge_alpha)),
        ]
    )


def _prediction_metrics(
    examples: pd.DataFrame,
    sequence_logp: np.ndarray,
    teacher_logp: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    group_level: str,
    fold_id: np.ndarray,
    control: str,
    repeat: int,
    top_k: int,
) -> pd.DataFrame:
    centered_prediction = prediction - prediction.mean(axis=1, keepdims=True)
    predicted_logp = sequence_logp + centered_prediction
    baseline_jsd = rowwise_jsd(sequence_logp, teacher_logp)
    predicted_jsd = rowwise_jsd(predicted_logp, teacher_logp)
    result = examples[
        [
            "state_id",
            "domain_id",
            "position",
            "dataset",
            "analysis_role",
            "eligible_for_training",
            "state_kind",
            "requested_corruption_ratio",
            "burial",
            "secondary_structure",
            "contact_class",
            "conservation_class",
            "scaffold_compatibility",
            group_level,
        ]
    ].copy()
    result = result.rename(columns={group_level: "held_out_group"})
    result["group_level"] = group_level
    result["fold"] = fold_id
    result["control"] = control
    result["repeat"] = repeat
    result["baseline_jsd_nats"] = baseline_jsd
    result["predicted_jsd_nats"] = predicted_jsd
    result["jsd_reduction_nats"] = baseline_jsd - predicted_jsd
    result["residual_cosine"] = rowwise_cosine(centered_prediction, target)
    result["topk_overlap"] = rowwise_topk_overlap(predicted_logp, teacher_logp, top_k)
    return result


def _metric_summary(rows: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for metric in ("jsd_reduction_nats", "residual_cosine", "topk_overlap"):
        summary = grouped_cluster_means(
            rows,
            ["group_level", "control", "analysis_role"],
            metric,
            config.audit.cluster_column,
            config.audit.bootstrap_replicates,
            config.audit.confidence_level,
            config.seed + 700 + len(frames) * 1000,
        )
        summary.insert(3, "metric", metric)
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def _environment_summary(rows: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    observed = rows.loc[rows["control"] == "observed"].copy()
    if observed.empty:
        return pd.DataFrame()
    identifiers = [
        "state_id",
        "domain_id",
        "position",
        "analysis_role",
        "group_level",
        "state_kind",
        "requested_corruption_ratio",
        "jsd_reduction_nats",
        "residual_cosine",
        "topk_overlap",
    ]
    long = observed[
        [
            *identifiers,
            "burial",
            "secondary_structure",
            "contact_class",
            "conservation_class",
        ]
    ].melt(
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
    frames: list[pd.DataFrame] = []
    groups = [
        "group_level",
        "analysis_role",
        "state_kind",
        "requested_corruption_ratio",
        "environment_axis",
        "environment",
    ]
    for metric in ("jsd_reduction_nats", "residual_cosine", "topk_overlap"):
        summary = grouped_cluster_means(
            long,
            groups,
            metric,
            config.audit.cluster_column,
            config.audit.bootstrap_replicates,
            config.audit.confidence_level,
            config.seed + 800 + len(frames) * 1000,
        )
        summary.insert(len(groups), "metric", metric)
        frames.append(summary)
    result = pd.concat(frames, ignore_index=True)
    result["sufficient_rows"] = (
        result["n_rows"] >= config.observability.minimum_rows_per_environment
    )
    return result


def _empty_rows() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "state_id",
            "domain_id",
            "position",
            "analysis_role",
            "group_level",
            "control",
            "jsd_reduction_nats",
            "residual_cosine",
            "topk_overlap",
        ]
    )
