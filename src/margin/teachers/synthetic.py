"""Deterministic heterogeneous teacher fixture for end-to-end software validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from margin.config import ProjectConfig, TeacherSpec
from margin.constants import AA_ALPHABET, AA_TO_INDEX
from margin.data_registry.registry import RegistryTables
from margin.decoys.generate import DecoyArtifacts
from margin.state_sampling.bank import StateBank
from margin.teachers.cache import sequence_policy_scores
from margin.teachers.schema import logp_columns, validate_score_table


def build_synthetic_teacher_scores(
    bank: StateBank,
    registry: RegistryTables,
    decoys: DecoyArtifacts,
    config: ProjectConfig,
) -> list[pd.DataFrame]:
    """Create score tables with known paired advantage and weaker decoy signal."""

    if config.data_mode != "synthetic":
        raise ValueError("synthetic teacher scores are forbidden in real-data mode")
    enabled = [teacher for teacher in config.teacher_cache.teachers if teacher.enabled]
    sequence_teachers = [teacher for teacher in enabled if teacher.role == "sequence"]
    if len(sequence_teachers) != 1:
        raise ValueError("synthetic fixture requires exactly one sequence teacher")
    tables = [sequence_policy_scores(bank, sequence_teachers[0], config)]
    state_metadata = bank.states.set_index("state_id")
    position_table = bank.positions.copy()
    decoys_by_target = {
        domain_id: frame
        for domain_id, frame in decoys.decoys.groupby("target_domain_id", sort=False, observed=True)
    }
    sequences = registry.domains.set_index("domain_id")["sequence"].to_dict()
    for teacher_index, teacher in enumerate([item for item in enabled if item.role != "sequence"]):
        scale = 1.0 - 0.12 * teacher_index
        tables.append(
            _structure_teacher_table(
                teacher,
                scale,
                position_table,
                state_metadata,
                decoys_by_target,
                sequences,
                config,
            )
        )
    return tables


def _structure_teacher_table(
    teacher: TeacherSpec,
    scale: float,
    positions: pd.DataFrame,
    states: pd.DataFrame,
    decoys_by_target: dict[str, pd.DataFrame],
    sequences: dict[str, str],
    config: ProjectConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base_columns = [f"student_logp_{aa}" for aa in AA_ALPHABET]
    for position in positions.itertuples(index=False):
        state = states.loc[position.state_id]
        base = np.array([getattr(position, column) for column in base_columns], dtype=float)
        native_index = AA_TO_INDEX[position.native_aa]
        advantage = _paired_advantage(position, state) * scale
        paired = base.copy()
        paired[native_index] += advantage
        paired[(native_index + 3 + int(position.position) % 5) % len(AA_ALPHABET)] += 0.12 * scale
        rows.append(
            _score_row(
                position,
                teacher,
                "paired",
                position.domain_id,
                paired,
                config,
            )
        )
        for decoy in decoys_by_target[position.domain_id].itertuples(index=False):
            decoy_scores = base.copy()
            if decoy.decoy_type in {"permuted", "shuffled_residue"} and decoy.mapping:
                source_position = int(decoy.mapping[int(position.position)])
                source_aa = sequences[decoy.source_domain_id][source_position]
                decoy_index = AA_TO_INDEX[source_aa]
            else:
                code = sum(ord(character) for character in str(decoy.decoy_id))
                decoy_index = (native_index + 1 + code % 17) % len(AA_ALPHABET)
            if decoy_index == native_index:
                decoy_index = (decoy_index + 7) % len(AA_ALPHABET)
            decoy_scores[native_index] += 0.10 * advantage
            decoy_scores[decoy_index] += 0.35 * scale
            rows.append(
                _score_row(
                    position,
                    teacher,
                    decoy.decoy_type,
                    decoy.decoy_id,
                    decoy_scores,
                    config,
                )
            )
    table = pd.DataFrame(rows)
    validate_score_table(table)
    return table


def _paired_advantage(position: Any, state: pd.Series) -> float:
    environment = 0.25
    if position.burial == "buried":
        environment += 0.85
    if position.secondary_structure in {"helix", "strand"}:
        environment += 0.45
    if position.contact_class == "high_contact":
        environment += 0.45
    if position.burial == "exposed":
        environment -= 0.15
    reliability = float(state["scaffold_compatibility"])
    return max(0.05, environment * reliability)


def _score_row(
    position: Any,
    teacher: TeacherSpec,
    structure_role: str,
    structure_id: str,
    scores: np.ndarray,
    config: ProjectConfig,
) -> dict[str, Any]:
    clipped = np.clip(scores, -config.teacher_cache.score_clip, config.teacher_cache.score_clip)
    normalized = clipped - logsumexp(clipped)
    conditioning = {
        "mifst": "leave_one_out_masked_bidirectional_structure_sequence",
        "proteinmpnn": "full_sequence_backbone_conditional",
        "esm_if1": "autoregressive_prefix_backbone_conditional",
    }.get(teacher.teacher_id, "synthetic_structure_conditional")
    if teacher.teacher_id == "mifst" and structure_role == "contact_rewired":
        conditioning = "leave_one_out_masked_sequence_rewired_contact_graph"
    row: dict[str, Any] = {
        "state_id": position.state_id,
        "domain_id": position.domain_id,
        "position": int(position.position),
        "teacher_id": teacher.teacher_id,
        "teacher_role": teacher.role,
        "structure_role": structure_role,
        "structure_id": structure_id,
        "input_score_type": teacher.score_type,
        "conditioning": conditioning,
        "model_name": teacher.model_name,
        "model_revision": teacher.model_revision,
        "device": "synthetic",
        "wall_seconds": 0.0,
        "forward_calls": 1,
        "data_mode": config.data_mode,
    }
    row.update({column: float(normalized[index]) for index, column in enumerate(logp_columns())})
    return row
