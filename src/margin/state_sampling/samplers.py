"""Reproducible offline and on-policy corruption operators."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from Bio.Align import substitution_matrices
from scipy.special import softmax

from margin.config import StateBankConfig
from margin.constants import AA_ALPHABET, AA_TO_INDEX
from margin.state_sampling.policy import SequencePolicy, validate_policy_output


@dataclass(frozen=True)
class SampledState:
    tokens: tuple[str, ...]
    timestep: int
    policy_calls: int


def sample_state(
    kind: str,
    reference: str,
    residue_features: dict[str, np.ndarray],
    corruption_ratio: float,
    policy: SequencePolicy,
    domain_id: str,
    rng: np.random.Generator,
    config: StateBankConfig,
) -> SampledState | None:
    count = min(
        len(reference),
        max(config.minimum_positions, int(round(len(reference) * corruption_ratio))),
    )
    if kind == "random_mask":
        positions = _positions(rng, len(reference), count)
        return SampledState(_mask(reference, positions, config.mask_token), 0, 0)
    if kind == "random_substitution":
        positions = _positions(rng, len(reference), count)
        return SampledState(_uniform_substitute(reference, positions, rng), 0, 0)
    if kind == "blosum_substitution":
        positions = _positions(rng, len(reference), count)
        return SampledState(
            _blosum_substitute(reference, positions, rng, config.blosum_temperature), 0, 0
        )
    if kind == "model_aware_offline":
        positions = _positions(rng, len(reference), count)
        log_probabilities = policy.log_probabilities(domain_id, tuple(reference))
        validate_policy_output(log_probabilities, len(reference))
        return SampledState(
            _policy_substitute(
                reference,
                positions,
                log_probabilities,
                rng,
                config.model_aware_temperature,
            ),
            0,
            1,
        )
    if kind == "on_policy_rollout":
        if not policy.state_conditioned:
            raise ValueError("on_policy_rollout requires a state-conditioned sequence policy")
        return _rollout(reference, count, policy, domain_id, rng, config)
    if kind == "span_mask":
        positions = _span_positions(rng, len(reference), count, config.span_mean_length)
        return SampledState(_mask(reference, positions, config.mask_token), 0, 0)
    if kind in {"core_targeted", "surface_targeted"}:
        target = "buried" if kind == "core_targeted" else "exposed"
        candidates = np.flatnonzero(residue_features["burial"] == target)
        if len(candidates) < count:
            return None
        positions = np.sort(rng.choice(candidates, size=count, replace=False))
        if config.targeted_operation == "mask":
            tokens = _mask(reference, positions, config.mask_token)
        else:
            tokens = _uniform_substitute(reference, positions, rng)
        return SampledState(tokens, 0, 0)
    raise ValueError(f"unknown state kind: {kind}")


def _rollout(
    reference: str,
    count: int,
    policy: SequencePolicy,
    domain_id: str,
    rng: np.random.Generator,
    config: StateBankConfig,
) -> SampledState:
    positions = _positions(rng, len(reference), count)
    current = list(_mask(reference, positions, config.mask_token))
    snapshots: list[tuple[tuple[str, ...], int]] = [(tuple(current), 0)]
    calls = 0
    for timestep in range(1, config.rollout_steps + 1):
        unresolved = np.array(
            [position for position in positions if current[position] == config.mask_token],
            dtype=int,
        )
        if not len(unresolved):
            break
        log_probabilities = policy.log_probabilities(domain_id, tuple(current))
        validate_policy_output(log_probabilities, len(reference))
        calls += 1
        entropy = -(np.exp(log_probabilities[unresolved]) * log_probabilities[unresolved]).sum(
            axis=1
        )
        selected = int(unresolved[np.argmax(entropy)])
        probability = softmax(log_probabilities[selected] / config.rollout_temperature)
        current[selected] = str(rng.choice(list(AA_ALPHABET), p=probability))
        snapshots.append((tuple(current), timestep))
    snapshot_index = int(rng.integers(0, len(snapshots)))
    tokens, timestep = snapshots[snapshot_index]
    return SampledState(tokens=tokens, timestep=timestep, policy_calls=calls)


def _positions(rng: np.random.Generator, length: int, count: int) -> np.ndarray:
    return np.sort(rng.choice(length, size=count, replace=False))


def _mask(reference: str, positions: Sequence[int], mask_token: str) -> tuple[str, ...]:
    tokens = list(reference)
    for position in positions:
        tokens[int(position)] = mask_token
    return tuple(tokens)


def _uniform_substitute(
    reference: str, positions: Sequence[int], rng: np.random.Generator
) -> tuple[str, ...]:
    tokens = list(reference)
    for position_value in positions:
        position = int(position_value)
        alternatives = [aa for aa in AA_ALPHABET if aa != reference[position]]
        tokens[position] = str(rng.choice(alternatives))
    return tuple(tokens)


def _blosum_substitute(
    reference: str,
    positions: Sequence[int],
    rng: np.random.Generator,
    temperature: float,
) -> tuple[str, ...]:
    matrix = substitution_matrices.load("BLOSUM62")
    tokens = list(reference)
    for position_value in positions:
        position = int(position_value)
        native = reference[position]
        scores = np.array([float(matrix[native, aa]) for aa in AA_ALPHABET])
        scores[AA_TO_INDEX[native]] = -np.inf
        probability = softmax(scores / temperature)
        tokens[position] = str(rng.choice(list(AA_ALPHABET), p=probability))
    return tuple(tokens)


def _policy_substitute(
    reference: str,
    positions: Sequence[int],
    log_probabilities: np.ndarray,
    rng: np.random.Generator,
    temperature: float,
) -> tuple[str, ...]:
    tokens = list(reference)
    for position_value in positions:
        position = int(position_value)
        scores = log_probabilities[position].copy() / temperature
        scores[AA_TO_INDEX[reference[position]]] = -np.inf
        probability = softmax(scores)
        tokens[position] = str(rng.choice(list(AA_ALPHABET), p=probability))
    return tuple(tokens)


def _span_positions(
    rng: np.random.Generator, length: int, count: int, mean_length: float
) -> np.ndarray:
    selected: set[int] = set()
    while len(selected) < count:
        span = max(1, int(rng.poisson(mean_length)))
        start = int(rng.integers(0, length))
        selected.update(range(start, min(length, start + span)))
    return np.array(sorted(selected)[:count], dtype=int)
