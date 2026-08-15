"""DMS tests of structural-residual value beyond the sequence baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from margin.attribution.metrics import vector_spearman
from margin.constants import AA_TO_INDEX
from margin.provenance import runtime_manifest, table_manifest, write_json, write_parquet
from margin.studies.observability.config import ObservabilityStudyConfig
from margin.studies.observability.stats import domain_sensitivity_summary
from margin.teachers.schema import logp_columns

ENVIRONMENT_COLUMNS = (
    "burial",
    "secondary_structure",
    "contact_class",
    "conservation_class",
)


def run_dms_residual_audit(config: ObservabilityStudyConfig) -> dict[str, Path]:
    """Evaluate paired and paired-minus-decoy residual increments with assay LODO fits."""

    foundation = config.paths.foundation_run
    output = config.paths.run_dir / "dms_residual"
    output.mkdir(parents=True, exist_ok=True)
    scores = pd.read_parquet(foundation / "teacher_cache" / "scores.parquet")
    states = pd.read_parquet(foundation / "state_bank" / "states.parquet")
    residues = pd.read_parquet(foundation / "registry" / "residues.parquet")
    dms = pd.read_parquet(config.paths.project_root / "data/raw/benchmarks/dms_variants.parquet")
    native_states = set(states.loc[states["state_kind"].eq("native_reference"), "state_id"])
    native_scores = scores.loc[scores["state_id"].isin(native_states)].copy()

    metadata = residues[["domain_id", "position", *ENVIRONMENT_COLUMNS]].drop_duplicates(
        ["domain_id", "position"]
    )
    variants = dms.merge(metadata, on=["domain_id", "position"], validate="many_to_one")
    sequence = _score_effects(
        native_scores.loc[native_scores["teacher_id"].eq("sequence_student")], variants
    ).rename(columns={"score_effect": "sequence_effect"})
    base_columns = [
        "assay_id",
        "domain_id",
        "position",
        "wild_type",
        "mutant",
        "effect",
        *ENVIRONMENT_COLUMNS,
        "sequence_effect",
    ]
    sequence = sequence[base_columns]

    prediction_frames: list[pd.DataFrame] = []
    for teacher_id in config.residual_targets.teacher_specific:
        teacher = native_scores.loc[native_scores["teacher_id"].eq(teacher_id)]
        paired = _score_effects(
            teacher.loc[teacher["structure_role"].eq("paired")], variants
        ).rename(columns={"score_effect": "paired_effect"})
        frame = sequence.merge(
            paired[["assay_id", "domain_id", "position", "mutant", "paired_effect"]],
            on=["assay_id", "domain_id", "position", "mutant"],
            validate="one_to_one",
        )
        frame["teacher_id"] = teacher_id
        frame["structural_residual"] = frame["paired_effect"] - frame["sequence_effect"]
        decoy_effects = []
        for decoy_role, decoy_scores in teacher.loc[
            ~teacher["structure_role"].isin(["paired", "sequence_only"])
        ].groupby("structure_role", observed=True):
            averaged = (
                decoy_scores.groupby(["domain_id", "position"], observed=True)[logp_columns()]
                .mean()
                .reset_index()
            )
            effects = _score_effects(averaged, variants)
            effects["decoy_role"] = decoy_role
            decoy_effects.append(effects)
        if decoy_effects:
            decoys = pd.concat(decoy_effects, ignore_index=True).rename(
                columns={"score_effect": "decoy_effect"}
            )
            long = frame.merge(
                decoys[
                    [
                        "assay_id",
                        "domain_id",
                        "position",
                        "mutant",
                        "decoy_role",
                        "decoy_effect",
                    ]
                ],
                on=["assay_id", "domain_id", "position", "mutant"],
                how="left",
                validate="one_to_many",
            )
            long["paired_minus_decoy_residual"] = long["paired_effect"] - long["decoy_effect"]
        else:
            long = frame.copy()
            long["decoy_role"] = "none"
            long["decoy_effect"] = np.nan
            long["paired_minus_decoy_residual"] = np.nan
        prediction_frames.append(long)

    mc_path = config.paths.run_dir / "proteinmpnn" / "mc_scores.parquet"
    if mc_path.exists():
        mc_scores = pd.read_parquet(mc_path)
        mc_effect = _score_effects(mc_scores, variants).rename(
            columns={"score_effect": "paired_effect"}
        )
        mc_frame = sequence.merge(
            mc_effect[["assay_id", "domain_id", "position", "mutant", "paired_effect"]],
            on=["assay_id", "domain_id", "position", "mutant"],
            validate="one_to_one",
        )
        mc_frame["teacher_id"] = "proteinmpnn_mc8"
        mc_frame["structural_residual"] = mc_frame["paired_effect"] - mc_frame["sequence_effect"]
        mc_frame["decoy_role"] = "not_scored"
        mc_frame["decoy_effect"] = np.nan
        mc_frame["paired_minus_decoy_residual"] = np.nan
        prediction_frames.append(mc_frame)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    raw_domain = _raw_domain_correlations(predictions)
    lodo_predictions, lodo_domains = _lodo_incremental_models(predictions)
    summary = _summarize_lodo(lodo_domains, config)
    environments = _environment_increment_summary(lodo_predictions, config)

    paths = {
        "predictions": output / "variant_components.parquet",
        "raw_domains": output / "raw_domain_correlations.parquet",
        "lodo_predictions": output / "lodo_predictions.parquet",
        "lodo_domains": output / "lodo_domain_estimates.parquet",
        "summary": output / "lodo_summary.parquet",
        "environments": output / "environment_increment_summary.parquet",
    }
    tables = {
        "predictions": predictions,
        "raw_domains": raw_domain,
        "lodo_predictions": lodo_predictions,
        "lodo_domains": lodo_domains,
        "summary": summary,
        "environments": environments,
    }
    for name, path in paths.items():
        write_parquet(path, tables[name])
    manifest_path = output / "manifest.json"
    write_json(
        manifest_path,
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "scope": "stability_only",
            "fit_design": "leave_one_assay_domain_out",
            "artifacts": [table_manifest(paths[name], table) for name, table in tables.items()],
        },
    )
    paths["manifest"] = manifest_path
    return paths


def _score_effects(scores: pd.DataFrame, variants: pd.DataFrame) -> pd.DataFrame:
    columns = logp_columns()
    if scores.duplicated(["domain_id", "position"]).any():
        raise ValueError("DMS score input must have one row per domain position")
    joined = variants.merge(
        scores[["domain_id", "position", *columns]],
        on=["domain_id", "position"],
        validate="many_to_one",
    )
    wild = joined["wild_type"].map(AA_TO_INDEX).to_numpy(dtype=int)
    mutant = joined["mutant"].map(AA_TO_INDEX).to_numpy(dtype=int)
    values = joined[columns].to_numpy(dtype=float)
    joined["score_effect"] = (
        values[np.arange(len(values)), mutant] - values[np.arange(len(values)), wild]
    )
    return joined.drop(columns=columns)


def _raw_domain_correlations(predictions: pd.DataFrame) -> pd.DataFrame:
    methods = {
        "sequence": "sequence_effect",
        "paired_teacher": "paired_effect",
        "structural_residual": "structural_residual",
        "paired_minus_decoy_residual": "paired_minus_decoy_residual",
    }
    rows = []
    group_columns = ["teacher_id", "decoy_role", "assay_id", "domain_id"]
    for key, frame in predictions.groupby(group_columns, observed=True, dropna=False):
        labels = dict(zip(group_columns, key, strict=True))
        for method, column in methods.items():
            clean = frame[[column, "effect"]].dropna()
            rows.append(
                {
                    **labels,
                    "method": method,
                    "spearman": vector_spearman(clean[column], clean["effect"]),
                    "n_variants": int(len(clean)),
                }
            )
    return pd.DataFrame(rows)


def _lodo_incremental_models(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_sets = {
        "sequence_only": ["sequence_effect"],
        "sequence_plus_structural_residual": ["sequence_effect", "structural_residual"],
        "sequence_plus_paired_decoy": ["sequence_effect", "paired_minus_decoy_residual"],
        "sequence_plus_both_residuals": [
            "sequence_effect",
            "structural_residual",
            "paired_minus_decoy_residual",
        ],
    }
    prediction_rows: list[pd.DataFrame] = []
    domain_rows: list[dict[str, object]] = []
    for (teacher_id, decoy_role), frame in predictions.groupby(
        ["teacher_id", "decoy_role"], observed=True, dropna=False
    ):
        domains = frame["domain_id"].drop_duplicates().to_list()
        for held_out in domains:
            train = frame.loc[frame["domain_id"].ne(held_out)]
            test = frame.loc[frame["domain_id"].eq(held_out)]
            baseline_prediction = None
            method_predictions: dict[str, np.ndarray] = {}
            for method, columns in feature_sets.items():
                clean_train = train.dropna(subset=[*columns, "effect"])
                clean_test = test.dropna(subset=columns)
                if clean_train.empty or clean_test.empty:
                    continue
                scaler = StandardScaler()
                x_train = scaler.fit_transform(clean_train[columns].to_numpy(dtype=float))
                x_test = scaler.transform(clean_test[columns].to_numpy(dtype=float))
                model = Ridge(alpha=1.0)
                model.fit(x_train, clean_train["effect"].to_numpy(dtype=float))
                predicted = model.predict(x_test)
                output = clean_test[
                    [
                        "assay_id",
                        "domain_id",
                        "position",
                        "mutant",
                        "effect",
                        *ENVIRONMENT_COLUMNS,
                    ]
                ].copy()
                output["teacher_id"] = teacher_id
                output["decoy_role"] = decoy_role
                output["method"] = method
                output["predicted_effect"] = predicted
                prediction_rows.append(output)
                method_predictions[method] = predicted
                estimate = vector_spearman(predicted, clean_test["effect"])
                if method == "sequence_only":
                    baseline_prediction = estimate
                domain_rows.append(
                    {
                        "teacher_id": teacher_id,
                        "decoy_role": decoy_role,
                        "assay_id": str(clean_test["assay_id"].iloc[0]),
                        "domain_id": held_out,
                        "method": method,
                        "spearman": estimate,
                        "n_variants": int(len(clean_test)),
                    }
                )
            del method_predictions
            if baseline_prediction is None:
                continue
    predictions_out = pd.concat(prediction_rows, ignore_index=True)
    domains_out = pd.DataFrame(domain_rows)
    baseline = domains_out.loc[
        domains_out["method"].eq("sequence_only"),
        [
            "teacher_id",
            "decoy_role",
            "domain_id",
            "spearman",
        ],
    ].rename(columns={"spearman": "sequence_spearman"})
    domains_out = domains_out.merge(
        baseline,
        on=["teacher_id", "decoy_role", "domain_id"],
        validate="many_to_one",
    )
    domains_out["spearman_increment"] = domains_out["spearman"] - domains_out["sequence_spearman"]
    return predictions_out, domains_out


def _summarize_lodo(domains: pd.DataFrame, config: ObservabilityStudyConfig) -> pd.DataFrame:
    rows = []
    groups = ["teacher_id", "decoy_role", "method"]
    for group_index, (key, frame) in enumerate(domains.groupby(groups, observed=True)):
        _, summary = domain_sensitivity_summary(
            frame.rename(columns={"spearman_increment": "value"}),
            "value",
            confidence_level=config.inference.confidence_level,
            wild_replicates=config.inference.bootstrap_replicates,
            seed=config.seed + 20000 + group_index,
        )
        for name, value in zip(groups, key, strict=True):
            summary[name] = value
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def _environment_increment_summary(
    predictions: pd.DataFrame, config: ObservabilityStudyConfig
) -> pd.DataFrame:
    rows = []
    keys = ["teacher_id", "decoy_role", "domain_id"]
    baseline = predictions.loc[predictions["method"].eq("sequence_only")]
    for method in predictions["method"].drop_duplicates():
        if method == "sequence_only":
            continue
        selected = predictions.loc[predictions["method"].eq(method)]
        for axis in ENVIRONMENT_COLUMNS:
            for key, frame in selected.groupby([*keys, axis], observed=True, dropna=False):
                if len(frame) < 20:
                    continue
                teacher_id, decoy_role, domain_id, environment = key
                base = baseline.loc[
                    baseline["teacher_id"].eq(teacher_id)
                    & baseline["decoy_role"].eq(decoy_role)
                    & baseline["domain_id"].eq(domain_id)
                    & baseline[axis].eq(environment)
                ]
                if len(base) != len(frame):
                    continue
                estimate = vector_spearman(frame["predicted_effect"], frame["effect"])
                base_estimate = vector_spearman(base["predicted_effect"], base["effect"])
                rows.append(
                    {
                        "teacher_id": teacher_id,
                        "decoy_role": decoy_role,
                        "domain_id": domain_id,
                        "method": method,
                        "environment_axis": axis,
                        "environment": environment,
                        "spearman_increment": estimate - base_estimate,
                        "n_variants": int(len(frame)),
                    }
                )
    domain_table = pd.DataFrame(rows)
    if domain_table.empty:
        return domain_table
    summaries = []
    group_columns = ["teacher_id", "decoy_role", "method", "environment_axis", "environment"]
    for group_index, (key, frame) in enumerate(
        domain_table.groupby(group_columns, observed=True, dropna=False)
    ):
        _, summary = domain_sensitivity_summary(
            frame.rename(columns={"spearman_increment": "value"}),
            "value",
            confidence_level=config.inference.confidence_level,
            wild_replicates=config.inference.bootstrap_replicates,
            seed=config.seed + 30000 + group_index,
        )
        for name, value in zip(group_columns, key, strict=True):
            summary[name] = value
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True)
