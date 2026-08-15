"""Canonical score-table schema shared by heterogeneous protein teachers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from margin.constants import AA_ALPHABET

SCORE_KEY_COLUMNS = (
    "state_id",
    "domain_id",
    "position",
    "teacher_id",
    "teacher_role",
    "structure_role",
    "structure_id",
)

SCORE_METADATA_COLUMNS = (
    "input_score_type",
    "conditioning",
    "model_name",
    "model_revision",
    "device",
    "wall_seconds",
    "forward_calls",
    "data_mode",
)


def logp_columns(prefix: str = "logp") -> list[str]:
    return [f"{prefix}_{aa}" for aa in AA_ALPHABET]


def validate_score_table(table: pd.DataFrame) -> None:
    required = {*SCORE_KEY_COLUMNS, *SCORE_METADATA_COLUMNS, *logp_columns()}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"teacher score table is missing columns: {sorted(missing)}")
    if table.duplicated(list(SCORE_KEY_COLUMNS)).any():
        raise ValueError("teacher score cache contains duplicate scientific keys")
    scores = table[logp_columns()].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("teacher scores contain non-finite values")
    if not np.allclose(logsumexp(scores, axis=1), 0.0, atol=1e-5):
        raise ValueError("canonical teacher scores must be normalized log probabilities")
    if (table["position"].to_numpy(dtype=int) < 0).any():
        raise ValueError("teacher score positions must be non-negative")
    forward_calls = table["forward_calls"].to_numpy(dtype=float)
    if (
        not np.isfinite(forward_calls).all()
        or (forward_calls < 0).any()
        or not np.equal(forward_calls, np.floor(forward_calls)).all()
    ):
        raise ValueError("teacher forward_calls must be non-negative finite integers")
