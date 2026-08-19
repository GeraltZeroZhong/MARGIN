"""Identity-safe linear, reduced-rank, and nonlinear residual probes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from margin.attribution.metrics import rowwise_cosine, rowwise_jsd, rowwise_topk_overlap
from margin.studies.observability.targets import ResidualDataset, clr


def load_aligned_parquet_features(path, metadata: pd.DataFrame) -> np.ndarray:
    """Load a canonical feature table and align it to residual metadata keys."""

    frame = pd.read_parquet(path)
    keys = ["state_id", "domain_id", "position"]
    columns = [column for column in frame if column.startswith("feature_")]
    if not columns or frame.duplicated(keys).any():
        raise ValueError("feature table requires unique keys and feature_* columns")
    aligned = metadata[keys].merge(frame[[*keys, *columns]], on=keys, validate="one_to_one")
    if len(aligned) != len(metadata):
        raise ValueError("feature table does not cover every residual row")
    values = aligned[columns].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("feature matrix contains non-finite values")
    return values


def ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    alpha: float,
    rank: int | None = None,
) -> np.ndarray:
    """Fit a standardized multi-output Ridge, optionally in a training-only PCA target basis."""

    scaler = StandardScaler()
    train = scaler.fit_transform(x_train)
    test = scaler.transform(x_test)
    if rank is None:
        model = Ridge(alpha=alpha, solver="lsqr")
        model.fit(train, y_train)
        return clr(model.predict(test))
    effective_rank = min(rank, y_train.shape[1] - 1)
    basis = PCA(n_components=effective_rank, svd_solver="full", random_state=0)
    latent = basis.fit_transform(y_train)
    model = Ridge(alpha=alpha, solver="lsqr")
    model.fit(train, latent)
    return clr(basis.inverse_transform(model.predict(test)))


def mlp_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    hidden_units: int,
    max_iterations: int,
    seed: int,
) -> np.ndarray:
    """Fit the fixed two-layer bottleneck MLP with development-only early stopping."""

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    train = x_scaler.fit_transform(x_train)
    test = x_scaler.transform(x_test)
    target = y_scaler.fit_transform(y_train)
    model = MLPRegressor(
        hidden_layer_sizes=(hidden_units,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=max_iterations,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
        random_state=seed,
    )
    model.fit(train, target)
    return clr(y_scaler.inverse_transform(model.predict(test)))


def intercept_predict(
    metadata: pd.DataFrame,
    target: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    strata: Sequence[str],
) -> np.ndarray:
    """Predict training means globally or within a declared environment stratum."""

    global_mean = target[train_indices].mean(axis=0)
    if not strata:
        return np.repeat(global_mean[None, :], len(test_indices), axis=0)
    train = metadata.iloc[train_indices][list(strata)].copy()
    train["_row"] = np.arange(len(train_indices))
    test = metadata.iloc[test_indices][list(strata)].copy()
    grouped: dict[tuple[object, ...], np.ndarray] = {}
    grouper = list(strata) if len(strata) > 1 else strata[0]
    for key, frame in train.groupby(grouper, observed=True, dropna=False):
        normalized = key if isinstance(key, tuple) else (key,)
        grouped[normalized] = target[train_indices[frame["_row"].to_numpy(dtype=int)]].mean(axis=0)
    rows = []
    for values in test.itertuples(index=False, name=None):
        rows.append(grouped.get(tuple(values), global_mean))
    return clr(np.asarray(rows))


def shuffled_target(
    metadata: pd.DataFrame,
    target: np.ndarray,
    train_indices: np.ndarray,
    control: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Shuffle training targets at the registered hierarchy and report movement."""

    strata = {
        "global": [],
        "within_domain": ["domain_id"],
        "within_wild_type": ["native_aa"],
        "within_environment": [
            "burial",
            "secondary_structure",
            "contact_class",
            "conservation_class",
        ],
        "within_corruption": ["state_kind", "requested_corruption_ratio"],
        "fully_conditioned": [
            "domain_id",
            "native_aa",
            "burial",
            "secondary_structure",
            "contact_class",
            "conservation_class",
            "state_kind",
            "requested_corruption_ratio",
        ],
    }[control]
    permutation = np.arange(len(train_indices))
    if not strata:
        permutation = rng.permutation(permutation)
    else:
        frame = metadata.iloc[train_indices][strata].copy()
        frame["_row"] = np.arange(len(frame))
        grouper = strata if len(strata) > 1 else strata[0]
        for _, group in frame.groupby(grouper, observed=True, dropna=False):
            rows = group["_row"].to_numpy(dtype=int)
            permutation[rows] = rng.permutation(rows)
    moved_fraction = float(np.mean(permutation != np.arange(len(permutation))))
    return target[train_indices][permutation], moved_fraction


def prediction_metrics(
    dataset: ResidualDataset,
    target_id: str,
    indices: np.ndarray,
    prediction: np.ndarray,
    **labels: object,
) -> pd.DataFrame:
    """Evaluate a predicted CLR residual with distributional and ranking metrics."""

    sequence = dataset.sequence_logp[indices]
    teacher = dataset.teacher_logp[target_id][indices]
    target = dataset.residuals[target_id][indices]
    predicted = _normalize(sequence + clr(prediction))
    sequence = _normalize(sequence)
    teacher = _normalize(teacher)
    teacher_probability = np.exp(teacher)
    baseline_cross_entropy = -np.sum(teacher_probability * sequence, axis=1)
    predicted_cross_entropy = -np.sum(teacher_probability * predicted, axis=1)
    result = dataset.metadata.iloc[indices].reset_index(drop=True).copy()
    result["target_id"] = target_id
    for name, value in labels.items():
        result[name] = value
    result["baseline_jsd_nats"] = rowwise_jsd(sequence, teacher)
    result["predicted_jsd_nats"] = rowwise_jsd(predicted, teacher)
    result["jsd_reduction_nats"] = result["baseline_jsd_nats"] - result["predicted_jsd_nats"]
    result["cross_entropy_reduction_nats"] = baseline_cross_entropy - predicted_cross_entropy
    result["residual_cosine"] = rowwise_cosine(clr(prediction), target)
    result["candidate_rank_agreement"] = _rowwise_spearman(predicted, teacher)
    result["top3_overlap"] = rowwise_topk_overlap(predicted, teacher, 3)
    return result


def _normalize(values: np.ndarray) -> np.ndarray:
    return values - logsumexp(values, axis=1, keepdims=True)


def _rowwise_spearman(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_rank = rankdata(left, axis=1)
    right_rank = rankdata(right, axis=1)
    left_rank -= left_rank.mean(axis=1, keepdims=True)
    right_rank -= right_rank.mean(axis=1, keepdims=True)
    numerator = np.sum(left_rank * right_rank, axis=1)
    denominator = np.linalg.norm(left_rank, axis=1) * np.linalg.norm(right_rank, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(numerator), np.nan, dtype=float),
        where=denominator > 0,
    )
