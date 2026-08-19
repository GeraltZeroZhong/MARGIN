"""Numerically stable metrics and domain-clustered uncertainty estimates."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import rankdata


def normalize_log_probabilities(values: np.ndarray) -> np.ndarray:
    """Return row-normalized natural-log probabilities."""

    array = np.asarray(values, dtype=float)
    return array - logsumexp(array, axis=-1, keepdims=True)


def rowwise_jsd(left_logp: np.ndarray, right_logp: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence in nats for corresponding rows."""

    left = normalize_log_probabilities(left_logp)
    right = normalize_log_probabilities(right_logp)
    midpoint = np.logaddexp(left, right) - np.log(2.0)
    left_kl = np.sum(np.exp(left) * (left - midpoint), axis=1)
    right_kl = np.sum(np.exp(right) * (right - midpoint), axis=1)
    return 0.5 * (left_kl + right_kl)


def rowwise_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Cosine similarity with undefined zero-vector rows reported as NaN."""

    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    numerator = np.sum(left_array * right_array, axis=1)
    denominator = np.linalg.norm(left_array, axis=1) * np.linalg.norm(right_array, axis=1)
    result = np.full(len(left_array), np.nan, dtype=float)
    valid = denominator > 0
    result[valid] = numerator[valid] / denominator[valid]
    return result


def rowwise_topk_overlap(left: np.ndarray, right: np.ndarray, k: int) -> np.ndarray:
    """Fractional overlap between two top-k candidate sets."""

    left_top = np.argpartition(left, -k, axis=1)[:, -k:]
    right_top = np.argpartition(right, -k, axis=1)[:, -k:]
    return np.array(
        [
            len(set(a.tolist()) & set(b.tolist())) / k
            for a, b in zip(left_top, right_top, strict=True)
        ],
        dtype=float,
    )


def vector_spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman correlation without warning on constant vectors."""

    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if len(left_array) < 2 or np.ptp(left_array) == 0 or np.ptp(right_array) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(left_array), rankdata(right_array))[0, 1])


def cluster_bootstrap_statistic(
    table: pd.DataFrame,
    value_columns: str | Sequence[str],
    cluster_column: str,
    statistic: Callable[[pd.DataFrame], float],
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float | int]:
    """Estimate a statistic and CI by resampling independent domains."""

    columns = [value_columns] if isinstance(value_columns, str) else list(value_columns)
    clean = table[[cluster_column, *columns]].replace([np.inf, -np.inf], np.nan).dropna()
    clusters = clean[cluster_column].drop_duplicates().to_numpy()
    if clean.empty or not len(clusters):
        return _empty_estimate()
    estimate = float(statistic(clean))
    if len(clusters) == 1 or replicates < 2:
        return {
            "estimate": estimate,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_rows": int(len(clean)),
            "n_domains": int(len(clusters)),
        }
    frames = {cluster: clean.loc[clean[cluster_column] == cluster] for cluster in clusters}
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        resampled = pd.concat(
            [
                frames[cluster].assign(**{cluster_column: index})
                for index, cluster in enumerate(sampled)
            ],
            ignore_index=True,
        )
        bootstrap[replicate] = statistic(resampled)
    bootstrap = bootstrap[np.isfinite(bootstrap)]
    alpha = (1.0 - confidence_level) / 2.0
    low, high = (
        np.quantile(bootstrap, [alpha, 1.0 - alpha])
        if len(bootstrap)
        else (float("nan"), float("nan"))
    )
    return {
        "estimate": estimate,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_rows": int(len(clean)),
        "n_domains": int(len(clusters)),
    }


def cluster_bootstrap_mean(
    table: pd.DataFrame,
    value_column: str,
    cluster_column: str,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float | int]:
    """Equal-domain-weighted mean and cluster-bootstrap confidence interval."""

    if cluster_column not in table or value_column not in table:
        return _empty_estimate()
    clean = table[[cluster_column, value_column]].replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return _empty_estimate()
    domain_means = (
        clean.groupby(cluster_column, observed=True)[value_column].mean().to_numpy(dtype=float)
    )
    estimate = float(domain_means.mean())
    if len(domain_means) == 1 or replicates < 2:
        low = high = float("nan")
    else:
        rng = np.random.default_rng(seed)
        sampled = rng.choice(domain_means, size=(replicates, len(domain_means)), replace=True)
        bootstrap = sampled.mean(axis=1)
        alpha = (1.0 - confidence_level) / 2.0
        low, high = np.quantile(bootstrap, [alpha, 1.0 - alpha])
    return {
        "estimate": estimate,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_rows": int(len(clean)),
        "n_domains": int(len(domain_means)),
    }


def grouped_cluster_means(
    table: pd.DataFrame,
    group_columns: Sequence[str],
    value_column: str,
    cluster_column: str,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    """Apply equal-domain estimation to each declared analysis stratum."""

    rows: list[dict[str, object]] = []
    grouper: str | list[str] = list(group_columns)
    if len(group_columns) == 1:
        grouper = group_columns[0]
    for group_index, (key, frame) in enumerate(table.groupby(grouper, observed=True, dropna=False)):
        keys = (key,) if len(group_columns) == 1 else tuple(key)
        row = dict(zip(group_columns, keys, strict=True))
        row.update(
            cluster_bootstrap_mean(
                frame,
                value_column,
                cluster_column,
                replicates,
                confidence_level,
                seed + group_index,
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _empty_estimate() -> dict[str, float | int]:
    return {
        "estimate": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "n_rows": 0,
        "n_domains": 0,
    }
