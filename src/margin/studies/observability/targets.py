"""Canonical observability study residual targets in the compositional tangent space."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from margin.attribution.metrics import normalize_log_probabilities
from margin.constants import AA_TO_INDEX
from margin.studies.observability.config import ObservabilityStudyConfig
from margin.teachers.schema import logp_columns


@dataclass(frozen=True)
class ResidualDataset:
    metadata: pd.DataFrame
    sequence_logp: np.ndarray
    teacher_logp: dict[str, np.ndarray]
    residuals: dict[str, np.ndarray]
    temperatures: pd.DataFrame


def load_foundation_residual_dataset(config: ObservabilityStudyConfig) -> ResidualDataset:
    """Load frozen foundation audit tables and construct teacher-specific CLR targets."""

    root = config.paths.foundation_run
    positions = pd.read_parquet(root / "state_bank" / "positions.parquet")
    states = pd.read_parquet(root / "state_bank" / "states.parquet")
    domains = pd.read_parquet(root / "registry" / "domains.parquet")
    scores = pd.read_parquet(root / "teacher_cache" / "scores.parquet")
    return build_residual_dataset(positions, states, domains, scores, config)


def load_replication_residual_dataset(
    config: ObservabilityStudyConfig, replication_run: Path | None = None
) -> ResidualDataset:
    """Load the frozen observability study replication registry, states, and teacher cache."""

    root = (
        replication_run.resolve()
        if replication_run is not None
        else config.paths.run_dir / "replication"
    )
    positions = pd.read_parquet(root / "state_bank" / "positions.parquet")
    states = pd.read_parquet(root / "state_bank" / "states.parquet")
    domains = pd.read_parquet(root / "registry" / "domains.parquet")
    scores = pd.read_parquet(root / "teacher_cache" / "scores.parquet")
    return build_residual_dataset(positions, states, domains, scores, config)


def build_residual_dataset(
    positions: pd.DataFrame,
    states: pd.DataFrame,
    domains: pd.DataFrame,
    scores: pd.DataFrame,
    config: ObservabilityStudyConfig,
) -> ResidualDataset:
    """Align metadata and score matrices, then define all registered residual targets."""

    keys = ["state_id", "domain_id", "position"]
    position_columns = [
        *keys,
        "native_aa",
        "current_aa",
        "burial",
        "secondary_structure",
        "contact_class",
        "conservation_class",
        "analysis_role",
        "eligible_for_training",
    ]
    state_columns = [
        "state_id",
        "state_kind",
        "requested_corruption_ratio",
        "scaffold_compatibility",
    ]
    domain_columns = ["domain_id", "cath_h", "cath_t"]
    if "observability_split" in domains:
        domain_columns.append("observability_split")
    metadata = (
        positions[position_columns]
        .merge(states[state_columns], on="state_id", validate="many_to_one")
        .merge(domains[domain_columns], on="domain_id", validate="many_to_one")
        .sort_values(keys, ignore_index=True)
    )
    if "observability_split" in metadata:
        metadata["analysis_role"] = metadata["observability_split"]
    if metadata.duplicated(keys).any():
        raise ValueError("observability study metadata keys are not unique")

    columns = logp_columns()
    sequence_id = "sequence_student"
    sequence_frame = scores.loc[
        scores["teacher_id"].eq(sequence_id), [*keys, *columns]
    ].sort_values(keys, ignore_index=True)
    if sequence_frame.duplicated(keys).any():
        raise ValueError("sequence score keys are not unique")
    aligned = metadata[keys].merge(sequence_frame, on=keys, validate="one_to_one")
    if len(aligned) != len(metadata):
        raise ValueError("sequence scores do not cover every observability study row")
    sequence_logp = normalize_log_probabilities(aligned[columns].to_numpy(dtype=float))

    requested = list(config.residual_targets.teacher_specific)
    paired = scores.loc[
        scores["teacher_id"].isin(requested) & scores["structure_role"].eq("paired")
    ]
    teacher_logp: dict[str, np.ndarray] = {}
    residuals: dict[str, np.ndarray] = {}
    temperature_rows: list[dict[str, float | int | str]] = []
    calibrated: dict[str, np.ndarray] = {}
    calibration_mask = (
        metadata["eligible_for_training"].astype(bool).to_numpy()
        & metadata["state_kind"].eq(config.residual_targets.calibrate_on_state_kind).to_numpy()
    )
    native_index = metadata["native_aa"].map(AA_TO_INDEX).to_numpy(dtype=int)

    for teacher_id in requested:
        frame = paired.loc[paired["teacher_id"].eq(teacher_id), [*keys, *columns]].sort_values(
            keys, ignore_index=True
        )
        merged = metadata[keys].merge(frame, on=keys, validate="one_to_one")
        if len(merged) != len(metadata):
            raise ValueError(f"paired scores for {teacher_id} do not cover every row")
        values = normalize_log_probabilities(merged[columns].to_numpy(dtype=float))
        teacher_logp[teacher_id] = values
        residuals[teacher_id] = clr(values - sequence_logp)
        temperature, before, after = fit_temperature(
            values[calibration_mask],
            native_index[calibration_mask],
            config.residual_targets.calibration_temperature_bounds,
        )
        calibrated[teacher_id] = normalize_log_probabilities(values / temperature)
        temperature_rows.append(
            {
                "teacher_id": teacher_id,
                "temperature": temperature,
                "native_nll_before": before,
                "native_nll_after": after,
                "calibration_rows": int(calibration_mask.sum()),
            }
        )

    members = config.residual_targets.consensus_members
    missing = set(members) - set(teacher_logp)
    if missing:
        raise ValueError(f"consensus members lack paired scores: {sorted(missing)}")
    equal_residual = np.mean([residuals[member] for member in members], axis=0)
    equal_logp = normalize_log_probabilities(sequence_logp + equal_residual)
    teacher_logp["consensus_equal"] = equal_logp
    residuals["consensus_equal"] = clr(equal_logp - sequence_logp)

    calibrated_residual = np.mean(
        [clr(calibrated[member] - sequence_logp) for member in members], axis=0
    )
    calibrated_logp = normalize_log_probabilities(sequence_logp + calibrated_residual)
    teacher_logp["consensus_calibrated"] = calibrated_logp
    residuals["consensus_calibrated"] = clr(calibrated_logp - sequence_logp)

    return ResidualDataset(
        metadata=metadata,
        sequence_logp=sequence_logp,
        teacher_logp=teacher_logp,
        residuals=residuals,
        temperatures=pd.DataFrame(temperature_rows),
    )


def clr(values: np.ndarray) -> np.ndarray:
    """Project 20-part log contrasts into their 19-dimensional tangent space."""

    array = np.asarray(values, dtype=float)
    return array - array.mean(axis=1, keepdims=True)


def helmert_basis(parts: int = 20) -> np.ndarray:
    """Return an orthonormal basis for the zero-sum compositional tangent space."""

    if parts < 2:
        raise ValueError("an ILR basis requires at least two parts")
    basis = np.zeros((parts - 1, parts), dtype=float)
    for row in range(parts - 1):
        count = row + 1
        basis[row, :count] = 1.0 / np.sqrt(count * (count + 1))
        basis[row, count] = -count / np.sqrt(count * (count + 1))
    return basis


def ilr(values: np.ndarray) -> np.ndarray:
    """Map CLR contrasts to 19 orthonormal isometric log-ratio coordinates."""

    array = clr(values)
    return array @ helmert_basis(array.shape[1]).T


def inverse_ilr(coordinates: np.ndarray) -> np.ndarray:
    """Map orthonormal ILR coordinates back to zero-sum CLR contrasts."""

    array = np.asarray(coordinates, dtype=float)
    return array @ helmert_basis(array.shape[1] + 1)


def fit_temperature(
    logp: np.ndarray,
    native_index: np.ndarray,
    bounds: tuple[float, float],
) -> tuple[float, float, float]:
    """Fit one scalar temperature using only declared development rows."""

    if len(logp) == 0:
        raise ValueError("temperature calibration requires at least one row")

    def objective(temperature: float) -> float:
        normalized = normalize_log_probabilities(logp / temperature)
        return float(-normalized[np.arange(len(normalized)), native_index].mean())

    before = objective(1.0)
    result = minimize_scalar(objective, bounds=bounds, method="bounded")
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError("teacher temperature calibration did not converge")
    return float(result.x), before, float(result.fun)
