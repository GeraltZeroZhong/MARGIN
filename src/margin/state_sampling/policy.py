"""Sequence-only policy contract used to distinguish true rollout from static corruption."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from margin.constants import AA_ALPHABET, AA_TO_INDEX


class SequencePolicy(ABC):
    """A frozen sequence model that returns one score vector per current token."""

    policy_id: str
    model_revision: str
    state_conditioned: bool

    @abstractmethod
    def log_probabilities(self, domain_id: str, tokens: Sequence[str]) -> np.ndarray:
        """Return normalized log probabilities with shape ``[length, 20]``."""


class ImportedNativePolicy(SequencePolicy):
    """Static per-domain logits, valid for offline corruption but not rollout claims."""

    state_conditioned = False

    def __init__(
        self,
        table: pd.DataFrame,
        policy_id: str,
        model_revision: str,
    ) -> None:
        self.policy_id = policy_id
        self.model_revision = model_revision
        score_columns = [f"score_{aa}" for aa in AA_ALPHABET]
        missing = {"domain_id", "position", *score_columns} - set(table.columns)
        if missing:
            raise ValueError(f"imported policy table is missing columns: {sorted(missing)}")
        self._scores = {
            domain_id: frame.sort_values("position")[score_columns].to_numpy(dtype=float)
            for domain_id, frame in table.groupby("domain_id", sort=False, observed=True)
        }

    def log_probabilities(self, domain_id: str, tokens: Sequence[str]) -> np.ndarray:
        scores = self._scores.get(domain_id)
        if scores is None:
            raise KeyError(f"no imported policy scores for {domain_id}")
        if len(scores) != len(tokens):
            raise ValueError(f"policy length mismatch for {domain_id}")
        return _normalize(scores)


class SyntheticSequencePolicy(SequencePolicy):
    """Deterministic state-dependent fixture policy; never valid as scientific evidence."""

    state_conditioned = True

    def __init__(
        self,
        references: dict[str, str],
        policy_id: str = "synthetic-sequence",
        model_revision: str = "fixture-v1",
    ) -> None:
        self.policy_id = policy_id
        self.model_revision = model_revision
        self._references = references

    def log_probabilities(self, domain_id: str, tokens: Sequence[str]) -> np.ndarray:
        reference = self._references[domain_id]
        if len(reference) != len(tokens):
            raise ValueError(f"synthetic policy length mismatch for {domain_id}")
        scores = np.full((len(tokens), len(AA_ALPHABET)), -2.0, dtype=float)
        for position, native in enumerate(reference):
            native_index = AA_TO_INDEX[native]
            scores[position, native_index] = 2.2
            left = tokens[position - 1] if position > 0 else "X"
            right = tokens[position + 1] if position + 1 < len(tokens) else "X"
            context_code = sum(
                ord(character) for character in f"{domain_id}:{position}:{left}:{right}"
            )
            alternative = context_code % len(AA_ALPHABET)
            if alternative == native_index:
                alternative = (alternative + 7) % len(AA_ALPHABET)
            scores[position, alternative] = 1.4 if context_code % 5 else 2.6
            if tokens[position] == "X":
                scores[position, native_index] -= 0.35
            elif tokens[position] != native:
                scores[position, AA_TO_INDEX[tokens[position]]] += 0.45
        return _normalize(scores)


def load_policy_from_factory(factory_path: str, **kwargs: Any) -> SequencePolicy:
    """Load the user-owned frozen student through an explicit ``module:function`` contract."""

    if ":" not in factory_path:
        raise ValueError("student policy factory must use module:function syntax")
    module_name, function_name = factory_path.split(":", maxsplit=1)
    module = import_module(module_name)
    factory = getattr(module, function_name, None)
    if not callable(factory):
        raise ValueError(f"student policy factory is not callable: {factory_path}")
    policy = factory(**kwargs)
    required = ["policy_id", "model_revision", "state_conditioned", "log_probabilities"]
    missing = [name for name in required if not hasattr(policy, name)]
    if missing:
        raise TypeError(f"student policy factory result lacks attributes: {missing}")
    return policy


def load_imported_native_policy(
    path: Path, policy_id: str, model_revision: str
) -> ImportedNativePolicy:
    if not path.exists():
        raise FileNotFoundError(f"student policy score input does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        table = pd.read_parquet(path)
    elif path.suffix.lower() in {".csv", ".tsv"}:
        table = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
    else:
        raise ValueError("student policy scores must be Parquet, CSV, or TSV")
    return ImportedNativePolicy(table, policy_id, model_revision)


def validate_policy_output(log_probabilities: np.ndarray, length: int) -> None:
    if log_probabilities.shape != (length, len(AA_ALPHABET)):
        raise ValueError(
            f"policy output must have shape {(length, len(AA_ALPHABET))}, "
            f"got {log_probabilities.shape}"
        )
    if not np.isfinite(log_probabilities).all():
        raise ValueError("policy output contains non-finite values")
    normalization = logsumexp(log_probabilities, axis=1)
    if not np.allclose(normalization, 0.0, atol=1e-5):
        raise ValueError("policy output must be normalized log probabilities")


def _normalize(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    return scores - logsumexp(scores, axis=-1, keepdims=True)
