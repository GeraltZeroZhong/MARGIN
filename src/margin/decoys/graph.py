"""Degree-preserving contact-graph rewiring."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

Edge = tuple[int, int]


def contact_edges(
    ca_coordinates: np.ndarray,
    cutoff: float,
    minimum_sequence_separation: int = 1,
) -> set[Edge]:
    """Construct the canonical undirected CA contact graph."""

    finite = np.isfinite(ca_coordinates).all(axis=1)
    safe = np.where(finite[:, None], ca_coordinates, 0.0)
    distance = np.linalg.norm(safe[:, None, :] - safe[None, :, :], axis=-1)
    positions = np.arange(len(ca_coordinates))
    separation = np.abs(positions[:, None] - positions[None, :])
    rows, columns = np.where(
        np.triu(
            (distance <= cutoff)
            & (separation >= minimum_sequence_separation)
            & finite[:, None]
            & finite[None, :],
            1,
        )
    )
    return {(int(row), int(column)) for row, column in zip(rows, columns, strict=True)}


def degree_preserving_rewire(
    edges: Iterable[Edge],
    node_count: int,
    requested_swaps: int,
    max_attempts_per_swap: int,
    rng: np.random.Generator,
) -> tuple[set[Edge], int]:
    """Perform undirected double-edge swaps while preserving every node degree."""

    current = {_ordered(edge) for edge in edges}
    if len(current) < 2 or requested_swaps == 0:
        return current, 0
    completed = 0
    attempts = 0
    maximum_attempts = requested_swaps * max_attempts_per_swap
    while completed < requested_swaps and attempts < maximum_attempts:
        attempts += 1
        edge_list = tuple(current)
        first_index, second_index = rng.choice(len(edge_list), size=2, replace=False)
        (a, b), (c, d) = edge_list[int(first_index)], edge_list[int(second_index)]
        if len({a, b, c, d}) < 4:
            continue
        if rng.random() < 0.5:
            proposed = {_ordered((a, d)), _ordered((c, b))}
        else:
            proposed = {_ordered((a, c)), _ordered((b, d))}
        if any(left == right for left, right in proposed):
            continue
        if len(proposed) != 2 or proposed & current:
            continue
        current.remove(_ordered((a, b)))
        current.remove(_ordered((c, d)))
        current.update(proposed)
        completed += 1
    if any(node < 0 or node >= node_count for edge in current for node in edge):
        raise ValueError("rewired edge references an out-of-range node")
    return current, completed


def degrees(edges: Iterable[Edge], node_count: int) -> np.ndarray:
    result = np.zeros(node_count, dtype=int)
    for left, right in edges:
        result[left] += 1
        result[right] += 1
    return result


def _ordered(edge: Edge) -> Edge:
    left, right = edge
    return (left, right) if left < right else (right, left)
