"""Outcome-free teacher calibration on CATH native-residue prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, ndtri

from margin.attribution.metrics import normalize_log_probabilities
from margin.constants import AA_TO_INDEX
from margin.studies.action_validation.evaluation import (
    _action_rms,
    _aligned_store,
    _aligned_teacher,
    _anchor,
)
from margin.studies.stability.config import StabilityStudyConfig
from margin.teachers.schema import logp_columns


def load_cath_calibration_data(config: StabilityStudyConfig) -> dict[str, Any]:
    """Load the CATH sequence baseline and the three paired teacher actions."""

    metadata = pd.read_parquet(config.paths.cath_queries).reset_index(drop=True)
    sequence_logp = normalize_log_probabilities(
        _aligned_store(
            config.paths.cath_esm2_150_store,
            metadata,
            "log_probabilities.npy",
        ).astype(float)
    )
    wild = metadata["native_aa"].map(AA_TO_INDEX).to_numpy(dtype=int)
    sequence_action = _anchor(sequence_logp, wild)
    score_columns = [
        "state_id",
        "domain_id",
        "position",
        "teacher_id",
        "structure_role",
        *logp_columns(),
    ]
    cached = pd.read_parquet(
        config.paths.cath_teacher_scores,
        columns=score_columns,
        filters=[
            ("teacher_id", "in", ["esm_if1", "proteinmpnn"]),
            ("structure_role", "==", "paired"),
        ],
    )
    mif = pd.read_parquet(
        config.paths.cath_mif_scores,
        columns=score_columns,
        filters=[("teacher_id", "==", "mif"), ("structure_role", "==", "paired")],
    )
    actions = {
        "mif": _anchor(_aligned_teacher(mif, metadata, "mif"), wild),
        "esm_if1": _anchor(_aligned_teacher(cached, metadata, "esm_if1"), wild),
        "proteinmpnn": _anchor(_aligned_teacher(cached, metadata, "proteinmpnn"), wild),
    }
    return {
        "metadata": metadata,
        "sequence_logp": sequence_logp,
        "sequence_action": sequence_action,
        "wild": wild,
        "actions": actions,
    }


def select_calibration(config: StabilityStudyConfig) -> dict[str, Any]:
    """Select among four frozen schemes on CATH validation and refit the winner."""

    data = load_cath_calibration_data(config)
    metadata = data["metadata"]
    train = np.flatnonzero(
        metadata["observability_split"].eq(config.calibration.training_split).to_numpy()
    )
    validation = np.flatnonzero(
        metadata["observability_split"].eq(config.calibration.selection_split).to_numpy()
    )
    final = np.flatnonzero(
        metadata["observability_split"].isin(config.calibration.final_training_splits).to_numpy()
    )
    locked = np.flatnonzero(metadata["observability_split"].eq("locked_test").to_numpy())
    teacher_ids = config.calibration.teacher_ids
    actions = data["actions"]
    sequence_logp = data["sequence_logp"]
    wild = data["wild"]

    train_parameters = {
        "unscaled_equal": {},
        "action_rms_matched": _rms_parameters(
            actions, data["sequence_action"], wild, train, teacher_ids
        ),
        "joint_temperature_native_nll": _temperature_parameters(
            actions, sequence_logp, wild, train, teacher_ids, config
        ),
        "rowwise_rank_normalized": _rank_parameters(
            actions, sequence_logp, wild, train, teacher_ids, config
        ),
    }
    rows = []
    for scheme in config.calibration.schemes:
        action = apply_calibration(actions, scheme, train_parameters[scheme], teacher_ids)
        rows.append(
            {
                "scheme": scheme,
                "fit_split": config.calibration.training_split,
                "evaluation_split": config.calibration.selection_split,
                **native_metrics(sequence_logp[validation], action[validation], wild[validation]),
            }
        )
    validation_table = pd.DataFrame(rows).sort_values(
        [config.calibration.selection_metric, "scheme"], ignore_index=True
    )
    selected = str(validation_table.iloc[0]["scheme"])
    final_parameters_by_scheme = {
        "unscaled_equal": {},
        "action_rms_matched": _rms_parameters(
            actions, data["sequence_action"], wild, final, teacher_ids
        ),
        "joint_temperature_native_nll": _temperature_parameters(
            actions, sequence_logp, wild, final, teacher_ids, config
        ),
        "rowwise_rank_normalized": _rank_parameters(
            actions, sequence_logp, wild, final, teacher_ids, config
        ),
    }
    final_parameters = final_parameters_by_scheme[selected]
    final_action = apply_calibration(actions, selected, final_parameters, teacher_ids)
    audits = []
    for name, indices in (("development_train_plus_validation", final), ("locked_test", locked)):
        audits.append(
            {
                "scheme": selected,
                "evaluation_split": name,
                **native_metrics(sequence_logp[indices], final_action[indices], wild[indices]),
            }
        )
    return {
        "validation": validation_table,
        "selected_scheme": selected,
        "training_parameters": train_parameters,
        "final_parameters_by_scheme": final_parameters_by_scheme,
        "final_parameters": final_parameters,
        "audit": pd.DataFrame(audits),
    }


def apply_calibration(
    actions: dict[str, np.ndarray],
    scheme: str,
    parameters: dict[str, Any],
    teacher_ids: list[str],
) -> np.ndarray:
    """Return an equally aggregated teacher action under a frozen calibration."""

    if scheme == "unscaled_equal":
        calibrated = [np.asarray(actions[teacher], dtype=float) for teacher in teacher_ids]
    elif scheme == "action_rms_matched":
        calibrated = [
            np.asarray(actions[teacher], dtype=float) * float(parameters["scales"][teacher])
            for teacher in teacher_ids
        ]
    elif scheme == "joint_temperature_native_nll":
        calibrated = [
            np.asarray(actions[teacher], dtype=float) / float(parameters["temperatures"][teacher])
            for teacher in teacher_ids
        ]
    elif scheme == "rowwise_rank_normalized":
        scale = float(parameters["rank_scale"])
        calibrated = [scale * _rank_action(actions[teacher]) for teacher in teacher_ids]
    else:
        raise ValueError(f"unknown calibration scheme: {scheme}")
    return np.mean(np.stack(calibrated), axis=0)


def native_metrics(
    sequence_logp: np.ndarray, action: np.ndarray, wild: np.ndarray
) -> dict[str, float]:
    """Native NLL, top-one recovery, and reciprocal rank for calibrated logits."""

    logits = np.asarray(sequence_logp, dtype=float) + np.asarray(action, dtype=float)
    normalized = logits - logsumexp(logits, axis=1, keepdims=True)
    rows = np.arange(len(wild))
    ranks = np.argsort(np.argsort(-logits, axis=1), axis=1)[rows, wild] + 1
    return {
        "native_nll": float(-normalized[rows, wild].mean()),
        "native_aar": float((np.argmax(logits, axis=1) == wild).mean()),
        "native_mrr": float(np.mean(1.0 / ranks)),
    }


def _rms_parameters(
    actions: dict[str, np.ndarray],
    sequence_action: np.ndarray,
    wild: np.ndarray,
    indices: np.ndarray,
    teacher_ids: list[str],
) -> dict[str, Any]:
    target = _action_rms(sequence_action[indices], wild[indices])
    scales = {
        teacher: target / _action_rms(actions[teacher][indices], wild[indices])
        for teacher in teacher_ids
    }
    return {"target_rms": target, "scales": scales}


def _temperature_parameters(
    actions: dict[str, np.ndarray],
    sequence_logp: np.ndarray,
    wild: np.ndarray,
    indices: np.ndarray,
    teacher_ids: list[str],
    config: StabilityStudyConfig,
) -> dict[str, Any]:
    stacked = np.stack([actions[teacher][indices] for teacher in teacher_ids])
    baseline = sequence_logp[indices]
    native = wild[indices]

    def objective(log_temperature: np.ndarray) -> float:
        temperatures = np.exp(log_temperature)[:, None, None]
        action = np.mean(stacked / temperatures, axis=0)
        return native_metrics(baseline, action, native)["native_nll"]

    lower = np.log(config.calibration.temperature_minimum)
    upper = np.log(config.calibration.temperature_maximum)
    result = minimize(
        objective,
        np.zeros(len(teacher_ids), dtype=float),
        method="L-BFGS-B",
        bounds=[(lower, upper)] * len(teacher_ids),
    )
    if not result.success:
        raise RuntimeError(f"joint temperature calibration failed: {result.message}")
    temperatures = np.exp(result.x)
    return {
        "temperatures": {
            teacher: float(temperatures[index]) for index, teacher in enumerate(teacher_ids)
        },
        "fit_native_nll": float(result.fun),
    }


def _rank_parameters(
    actions: dict[str, np.ndarray],
    sequence_logp: np.ndarray,
    wild: np.ndarray,
    indices: np.ndarray,
    teacher_ids: list[str],
    config: StabilityStudyConfig,
) -> dict[str, Any]:
    rank_mean = np.mean(
        np.stack([_rank_action(actions[teacher][indices]) for teacher in teacher_ids]),
        axis=0,
    )

    def objective(value: np.ndarray) -> float:
        return native_metrics(sequence_logp[indices], rank_mean * float(value[0]), wild[indices])[
            "native_nll"
        ]

    result = minimize(
        objective,
        np.ones(1, dtype=float),
        method="L-BFGS-B",
        bounds=[
            (
                config.calibration.rank_scale_minimum,
                config.calibration.rank_scale_maximum,
            )
        ],
    )
    if not result.success:
        raise RuntimeError(f"rank calibration failed: {result.message}")
    return {"rank_scale": float(result.x[0]), "fit_native_nll": float(result.fun)}


def _rank_action(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(np.asarray(values, dtype=float), axis=1), axis=1)
    quantiles = (order + 0.5) / values.shape[1]
    return ndtri(quantiles)
