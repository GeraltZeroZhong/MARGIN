"""Exploratory mechanism, OOD, and mutation-stratum analyses for counterfactual study."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from margin.attribution.metrics import normalize_log_probabilities, rowwise_jsd
from margin.constants import AA_ALPHABET
from margin.provenance import runtime_manifest, table_manifest, write_json, write_parquet
from margin.studies.counterfactuals.config import CounterfactualStudyConfig
from margin.studies.counterfactuals.evaluation import (
    ROUTE_A_PRIMARY,
    ROUTE_A_REPLICATION,
    ROUTE_B_PRIMARY,
    ROUTE_B_REPLICATION,
    SEQUENCE_METHOD,
    _ndcg,
    _spearman,
    _stabilizing_topk_recall,
    stratified_domain_bootstrap,
)


def analyze_counterfactual_mechanisms(config: CounterfactualStudyConfig) -> dict[str, Path]:
    """Run analyses that are explicitly downstream of, and cannot alter, the frozen gate."""

    output = config.paths.run_dir / "mechanisms"
    output.mkdir(parents=True, exist_ok=True)
    evaluation = config.paths.run_dir / "evaluation"
    panel = config.paths.run_dir / "panel"
    components = pd.read_parquet(evaluation / "variant_components.parquet")
    domain_metrics = pd.read_parquet(evaluation / "domain_metrics.parquet")
    increments = pd.read_parquet(evaluation / "domain_increments.parquet")
    queries = pd.read_parquet(panel / "query_rows.parquet")
    residues = pd.read_parquet(panel / "residues.parquet")
    domains = pd.read_parquet(panel / "domains.parquet")
    matrices = np.load(evaluation / "residual_matrices.npz")
    sequence_logp = np.asarray(matrices["sequence_logp"], dtype=float)
    paired = np.asarray(matrices["paired_mif_logp"], dtype=float)

    ood_rows, ood_summary, intensity_smoothness = _ood_analysis(
        queries, residues, domains, paired, matrices
    )
    pca_tables = _residual_pca(queries, residues, domains, matrices)
    aaindex_tables = _aaindex_alignment(
        pca_tables["loadings"],
        config.paths.storage_dir.parent / "data" / "aaindex" / "aaindex1",
        config.seed,
    )
    action_alignment = _action_alignment(components)
    stratified_rows, stratified_summary, coverage = _stratified_analysis(
        components,
        queries,
        sequence_logp,
        domain_metrics,
        config,
    )
    heterogeneity = _heterogeneity_correlations(increments, domain_metrics)
    tables = {
        "ood_position_rows": ood_rows,
        "ood_domain_summary": ood_summary,
        "intensity_smoothness": intensity_smoothness,
        "residual_pca_variance": pca_tables["variance"],
        "residual_pca_loadings": pca_tables["loadings"],
        "residual_pca_scores": pca_tables["scores"],
        "aaindex_correlations": aaindex_tables["correlations"],
        "aaindex_latent_variance": aaindex_tables["latent_variance"],
        "aaindex_cca": aaindex_tables["cca"],
        "direct_predicted_action_alignment": action_alignment,
        "stratified_domain_rows": stratified_rows,
        "stratified_summary": stratified_summary,
        "analysis_coverage": coverage,
        "heterogeneity_correlations": heterogeneity,
    }
    paths = {}
    for name, table in tables.items():
        path = output / f"{name}.parquet"
        write_parquet(path, table)
        paths[name] = path
    manifest_path = output / "manifest.json"
    write_json(
        manifest_path,
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "analysis_role": "exploratory_post_gate_mechanism_analysis",
            "can_modify_frozen_route_decision": False,
            "conservation_analysis": "unavailable_no_MSA_conservation_on_locked_panel",
            "sequence_entropy_analysis": "reported_as_sequence_entropy_not_conservation",
            "aaindex_source": str(
                config.paths.storage_dir.parent / "data" / "aaindex" / "aaindex1"
            ),
            "artifacts": [table_manifest(paths[name], table) for name, table in tables.items()],
        },
    )
    paths["manifest"] = manifest_path
    return paths


def read_aaindex1(path: Path) -> pd.DataFrame:
    """Read complete 20-value AAindex1 records in canonical MARGIN AA order."""

    if not path.exists():
        return pd.DataFrame(columns=["accession", "description", *AA_ALPHABET])
    rows: list[dict[str, Any]] = []
    for entry in path.read_text(encoding="utf-8", errors="replace").split("//"):
        lines = [line.rstrip() for line in entry.splitlines() if line.strip()]
        accession_line = next((line for line in lines if line.startswith("H ")), None)
        index_line = next(
            (index for index, line in enumerate(lines) if line.startswith("I ")),
            None,
        )
        if accession_line is None or index_line is None:
            continue
        numeric: list[float] = []
        for line in lines[index_line + 1 :]:
            for token in line.split():
                try:
                    numeric.append(float(token))
                except ValueError:
                    numeric.append(float("nan"))
        if len(numeric) != 20:
            continue
        first = "ARNDCQEGHI"
        second = "LKMFPSTWYV"
        values = dict(zip(first, numeric[:10], strict=True))
        values.update(dict(zip(second, numeric[10:], strict=True)))
        description = " ".join(line[2:].strip() for line in lines if line.startswith("D "))
        rows.append(
            {
                "accession": accession_line[2:].strip(),
                "description": description,
                **{aa: values[aa] for aa in AA_ALPHABET},
            }
        )
    return pd.DataFrame(rows, columns=["accession", "description", *AA_ALPHABET])


def _ood_analysis(
    queries: pd.DataFrame,
    residues: pd.DataFrame,
    domains: pd.DataFrame,
    paired: np.ndarray,
    matrices: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata = (
        queries[["domain_id", "position"]]
        .merge(
            residues[["domain_id", "position", "burial", "secondary_structure", "contact_class"]],
            on=["domain_id", "position"],
            validate="one_to_one",
        )
        .merge(domains[["domain_id", "stratum", "length"]], on="domain_id", validate="many_to_one")
    )
    paired_probability = np.exp(paired)
    paired_entropy = -np.sum(paired_probability * paired, axis=1)
    roles = [
        "contact_rewired_0.5",
        "contact_rewired_1",
        "contact_rewired_2",
        "contact_rewired_5",
        "circular_permuted",
    ]
    rows = []
    for role in roles:
        residual = np.asarray(matrices[f"direct__{role}"], dtype=float)
        counterfactual = normalize_log_probabilities(paired - residual)
        probability = np.exp(counterfactual)
        top3 = np.partition(probability, -3, axis=1)[:, -3:].sum(axis=1)
        frame = metadata.copy()
        frame["counterfactual_role"] = role
        frame["rewiring_swaps_per_edge"] = (
            float(role.removeprefix("contact_rewired_"))
            if role.startswith("contact_rewired_")
            else np.nan
        )
        frame["paired_entropy_nats"] = paired_entropy
        frame["counterfactual_entropy_nats"] = -np.sum(probability * counterfactual, axis=1)
        frame["entropy_change_nats"] = (
            frame["counterfactual_entropy_nats"] - frame["paired_entropy_nats"]
        )
        frame["paired_max_probability"] = paired_probability.max(axis=1)
        frame["counterfactual_max_probability"] = probability.max(axis=1)
        frame["counterfactual_top3_mass"] = top3
        frame["paired_counterfactual_jsd_nats"] = rowwise_jsd(paired, counterfactual)
        frame["residual_l2"] = np.linalg.norm(residual, axis=1)
        frame["counterfactual_min_log_probability"] = counterfactual.min(axis=1)
        frame["extreme_low_probability_fraction"] = (counterfactual < -15.0).mean(axis=1)
        rows.append(frame)
    position_rows = pd.concat(rows, ignore_index=True)
    value_columns = [
        "paired_entropy_nats",
        "counterfactual_entropy_nats",
        "entropy_change_nats",
        "paired_max_probability",
        "counterfactual_max_probability",
        "counterfactual_top3_mass",
        "paired_counterfactual_jsd_nats",
        "residual_l2",
        "counterfactual_min_log_probability",
        "extreme_low_probability_fraction",
    ]
    domain_means = (
        position_rows.groupby(
            ["counterfactual_role", "rewiring_swaps_per_edge", "stratum", "domain_id"],
            observed=True,
            dropna=False,
        )[value_columns]
        .mean()
        .reset_index()
    )
    summary = domain_means.groupby(
        ["counterfactual_role", "rewiring_swaps_per_edge", "stratum"],
        observed=True,
        dropna=False,
    )[value_columns].agg(["mean", "median"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()
    intensity = domain_means.loc[domain_means["rewiring_swaps_per_edge"].notna()]
    smoothness_rows = []
    for domain_id, frame in intensity.groupby("domain_id", sort=True, observed=True):
        frame = frame.sort_values("rewiring_swaps_per_edge")
        for metric in ("paired_counterfactual_jsd_nats", "residual_l2", "entropy_change_nats"):
            smoothness_rows.append(
                {
                    "domain_id": domain_id,
                    "stratum": str(frame["stratum"].iloc[0]),
                    "metric": metric,
                    "spearman_with_rewiring_strength": _safe_spearman(
                        frame["rewiring_swaps_per_edge"], frame[metric]
                    ),
                }
            )
    return position_rows, summary, pd.DataFrame(smoothness_rows)


def _residual_pca(
    queries: pd.DataFrame,
    residues: pd.DataFrame,
    domains: pd.DataFrame,
    matrices: Any,
) -> dict[str, pd.DataFrame]:
    residual = np.asarray(matrices["direct__contact_rewired_5"], dtype=float)
    components = min(10, residual.shape[1] - 1, residual.shape[0])
    model = PCA(n_components=components, svd_solver="full").fit(residual)
    scores = model.transform(residual)
    variance = pd.DataFrame(
        {
            "component": [f"PC{index + 1}" for index in range(components)],
            "explained_variance_ratio": model.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(model.explained_variance_ratio_),
        }
    )
    loadings = (
        pd.DataFrame(
            model.components_.T,
            index=list(AA_ALPHABET),
            columns=[f"PC{index + 1}" for index in range(components)],
        )
        .rename_axis("amino_acid")
        .reset_index()
    )
    score_table = (
        queries[["domain_id", "position"]]
        .merge(
            residues[["domain_id", "position", "burial", "secondary_structure"]],
            on=["domain_id", "position"],
            validate="one_to_one",
        )
        .merge(domains[["domain_id", "stratum"]], on="domain_id", validate="many_to_one")
    )
    for index in range(components):
        score_table[f"PC{index + 1}"] = scores[:, index]
    return {"variance": variance, "loadings": loadings, "scores": score_table}


def _aaindex_alignment(
    loadings: pd.DataFrame,
    path: Path,
    seed: int,
) -> dict[str, pd.DataFrame]:
    aaindex = read_aaindex1(path).dropna(subset=list(AA_ALPHABET))
    if aaindex.empty:
        empty = pd.DataFrame()
        return {"correlations": empty, "latent_variance": empty, "cca": empty}
    loading_matrix = loadings.set_index("amino_acid").loc[list(AA_ALPHABET)]
    rows = []
    for entry in aaindex.itertuples(index=False):
        values = np.asarray([getattr(entry, aa) for aa in AA_ALPHABET], dtype=float)
        if np.ptp(values) == 0:
            continue
        for component in loading_matrix.columns:
            vector = loading_matrix[component].to_numpy(dtype=float)
            rows.append(
                {
                    "component": component,
                    "accession": entry.accession,
                    "description": entry.description,
                    "pearson": float(pearsonr(vector, values).statistic),
                    "spearman": float(spearmanr(vector, values).statistic),
                }
            )
    correlations = pd.DataFrame(rows)
    index_values = aaindex[list(AA_ALPHABET)].to_numpy(dtype=float).T
    index_values = StandardScaler().fit_transform(index_values)
    latent_components = min(5, index_values.shape[0] - 1, index_values.shape[1])
    index_pca = PCA(
        n_components=latent_components,
        svd_solver="randomized",
        random_state=seed,
    )
    index_latent = index_pca.fit_transform(index_values)
    residual_latent = StandardScaler().fit_transform(
        loading_matrix.iloc[:, :latent_components].to_numpy(dtype=float)
    )
    cca_components = min(3, latent_components)
    left, right = CCA(n_components=cca_components, max_iter=5_000).fit_transform(
        residual_latent, index_latent
    )
    cca = pd.DataFrame(
        {
            "canonical_component": [f"CC{index + 1}" for index in range(cca_components)],
            "in_sample_canonical_correlation": [
                float(pearsonr(left[:, index], right[:, index]).statistic)
                for index in range(cca_components)
            ],
            "interpretation": "exploratory_20_amino_acid_in_sample",
        }
    )
    latent_variance = pd.DataFrame(
        {
            "aaindex_component": [f"AAIndex_PC{index + 1}" for index in range(latent_components)],
            "explained_variance_ratio": index_pca.explained_variance_ratio_,
        }
    )
    return {
        "correlations": correlations,
        "latent_variance": latent_variance,
        "cca": cca,
    }


def _action_alignment(components: pd.DataFrame) -> pd.DataFrame:
    rows = []
    direct = "direct_contact_rewired_5_effect"
    predicted = "predicted_contact_rewired_5_effect"
    for domain_id, frame in components.groupby("domain_id", sort=True, observed=True):
        left = frame[direct].to_numpy(dtype=float)
        right = frame[predicted].to_numpy(dtype=float)
        rows.append(
            {
                "domain_id": domain_id,
                "stratum": str(frame["stratum"].iloc[0]),
                "n_variants": int(len(frame)),
                "spearman": _safe_spearman(left, right),
                "pearson": _safe_pearson(left, right),
                "candidate_pair_sign_accuracy": float((np.sign(left) == np.sign(right)).mean()),
                "direct_margin_mean": float(left.mean()),
                "predicted_margin_mean": float(right.mean()),
            }
        )
    return pd.DataFrame(rows)


def _stratified_analysis(
    components: pd.DataFrame,
    queries: pd.DataFrame,
    sequence_logp: np.ndarray,
    domain_metrics: pd.DataFrame,
    config: CounterfactualStudyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    position_entropy = -np.sum(np.exp(sequence_logp) * sequence_logp, axis=1)
    entropy_table = queries[["domain_id", "position"]].copy()
    entropy_table["sequence_entropy_nats"] = position_entropy
    entropy_table["sequence_entropy_class"] = pd.qcut(
        entropy_table["sequence_entropy_nats"].rank(method="first"),
        3,
        labels=["low", "middle", "high"],
    ).astype(str)
    enriched = components.merge(
        entropy_table,
        on=["domain_id", "position"],
        validate="many_to_one",
    )
    baseline = domain_metrics.loc[
        domain_metrics["method"].eq(SEQUENCE_METHOD), ["domain_id", "spearman"]
    ].copy()
    finite = baseline["spearman"].notna()
    baseline.loc[finite, "baseline_spearman_class"] = pd.qcut(
        baseline.loc[finite, "spearman"].rank(method="first"),
        3,
        labels=["low", "middle", "high"],
    ).astype(str)
    enriched = enriched.merge(
        baseline[["domain_id", "baseline_spearman_class"]],
        on="domain_id",
        validate="many_to_one",
    )
    dimensions = [
        "burial",
        "secondary_structure",
        "substitution_class",
        "length_class",
        "sequence_entropy_class",
        "baseline_spearman_class",
    ]
    methods = {
        ROUTE_A_PRIMARY: "direct_contact_rewired_5_effect",
        ROUTE_A_REPLICATION: "direct_circular_permuted_effect",
        ROUTE_B_PRIMARY: "predicted_contact_rewired_5_effect",
        ROUTE_B_REPLICATION: "predicted_circular_permuted_effect",
    }
    metric_rows = []
    for dimension in dimensions:
        for level, level_frame in enriched.dropna(subset=[dimension]).groupby(
            dimension, sort=True, observed=True
        ):
            for domain_id, frame in level_frame.groupby("domain_id", sort=True, observed=True):
                if len(frame) < 3:
                    continue
                observed = frame["effect"].to_numpy(dtype=float)
                sequence = frame["sequence_effect"].to_numpy(dtype=float)
                baseline_values = _ranking_metrics(
                    sequence, observed, config.inference.top_fraction
                )
                for method, column in methods.items():
                    values = _ranking_metrics(
                        sequence + frame[column].to_numpy(dtype=float),
                        observed,
                        config.inference.top_fraction,
                    )
                    metric_rows.append(
                        {
                            "dimension": dimension,
                            "level": str(level),
                            "domain_id": domain_id,
                            "stratum": str(frame["stratum"].iloc[0]),
                            "method": method,
                            "n_variants": int(len(frame)),
                            **{
                                f"{metric}_increment": values[metric] - baseline_values[metric]
                                for metric in (
                                    "spearman",
                                    "ndcg",
                                    "stabilizing_topk_recall",
                                )
                            },
                        }
                    )
    metric_table = pd.DataFrame(metric_rows)
    summary_rows = []
    if not metric_table.empty:
        for group_index, ((dimension, level, method), frame) in enumerate(
            metric_table.groupby(["dimension", "level", "method"], sort=True, observed=True)
        ):
            for metric_index, metric in enumerate(
                (
                    "spearman_increment",
                    "ndcg_increment",
                    "stabilizing_topk_recall_increment",
                )
            ):
                summary_rows.append(
                    {
                        "dimension": dimension,
                        "level": level,
                        "method": method,
                        "metric": metric,
                        **stratified_domain_bootstrap(
                            frame,
                            metric,
                            replicates=config.inference.bootstrap_replicates,
                            confidence_level=config.inference.confidence_level,
                            seed=config.seed + 700_000 + group_index * 10 + metric_index,
                        ),
                    }
                )
    coverage = pd.DataFrame(
        [
            {
                "analysis": "msa_conservation",
                "available_rows": 0,
                "total_rows": int(len(components)),
                "status": "UNAVAILABLE_NOT_IN_LOCKED_PANEL",
            },
            {
                "analysis": "sequence_entropy",
                "available_rows": int(len(components)),
                "total_rows": int(len(components)),
                "status": "AVAILABLE_NOT_A_CONSERVATION_PROXY",
            },
        ]
    )
    return metric_table, pd.DataFrame(summary_rows), coverage


def _heterogeneity_correlations(
    increments: pd.DataFrame,
    domain_metrics: pd.DataFrame,
) -> pd.DataFrame:
    del domain_metrics
    rows = []
    for method in (ROUTE_A_PRIMARY, ROUTE_A_REPLICATION, ROUTE_B_PRIMARY, ROUTE_B_REPLICATION):
        frame = increments.loc[increments["method"].eq(method)]
        for covariate in ("length", "n_variants", "baseline_spearman"):
            rows.append(
                {
                    "method": method,
                    "outcome": "spearman_increment",
                    "covariate": covariate,
                    "spearman": _safe_spearman(frame["spearman_increment"], frame[covariate]),
                    "n_domains": int(frame[["spearman_increment", covariate]].dropna().shape[0]),
                }
            )
    return pd.DataFrame(rows)


def _ranking_metrics(
    predicted: np.ndarray,
    observed: np.ndarray,
    top_fraction: float,
) -> dict[str, float]:
    return {
        "spearman": _spearman(predicted, observed),
        "ndcg": _ndcg(predicted, observed),
        "stabilizing_topk_recall": _stabilizing_topk_recall(predicted, observed, top_fraction),
    }


def _safe_spearman(left: Any, right: Any) -> float:
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(frame) < 2 or frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return float("nan")
    return float(spearmanr(frame["left"], frame["right"]).statistic)


def _safe_pearson(left: Any, right: Any) -> float:
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(frame) < 2 or frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return float("nan")
    return float(pearsonr(frame["left"], frame["right"]).statistic)
