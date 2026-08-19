"""Build the state × position bank used by every teacher and audit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from margin.config import ProjectConfig
from margin.constants import AA_ALPHABET, AA_TO_INDEX
from margin.data_registry.registry import RegistryTables
from margin.provenance import (
    runtime_manifest,
    table_manifest,
    write_json,
    write_parquet,
)
from margin.state_sampling.policy import SequencePolicy, validate_policy_output
from margin.state_sampling.samplers import sample_state

STATE_COLUMNS = (
    "state_id",
    "domain_id",
    "dataset",
    "analysis_role",
    "eligible_for_training",
    "state_kind",
    "replicate",
    "reference_sequence",
    "state_sequence",
    "requested_corruption_ratio",
    "corruption_ratio",
    "edit_distance",
    "edit_distance_fraction",
    "mask_count",
    "student_entropy",
    "student_top1_margin",
    "error_positions",
    "timestep",
    "scaffold_compatibility",
    "modified_core_fraction",
    "sequence_policy_calls",
)

POSITION_COLUMNS = (
    "state_id",
    "domain_id",
    "dataset",
    "analysis_role",
    "eligible_for_training",
    "position",
    "native_aa",
    "current_aa",
    "is_corrupted",
    "is_masked",
    "burial",
    "secondary_structure",
    "contact_class",
    "rsa",
    "contact_degree",
    "conservation_score",
    "conservation_class",
    "student_entropy",
    "student_top1_margin",
    "student_top1_aa",
    "student_logp_native",
    *tuple(f"student_logp_{aa}" for aa in AA_ALPHABET),
)


@dataclass(frozen=True)
class StateBank:
    states: pd.DataFrame
    positions: pd.DataFrame


def build_state_bank(
    registry: RegistryTables,
    policy: SequencePolicy,
    config: ProjectConfig,
) -> tuple[StateBank, pd.DataFrame]:
    """Generate all declared corruption families from a single frozen policy."""

    rng = np.random.default_rng(config.seed)
    state_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    residues_by_domain = {
        domain_id: frame.sort_values("position").reset_index(drop=True)
        for domain_id, frame in registry.residues.groupby("domain_id", sort=False, observed=True)
    }
    for domain in registry.domains.sort_values("domain_id").itertuples(index=False):
        residue_table = residues_by_domain[domain.domain_id]
        residue_features = {
            "burial": residue_table["burial"].to_numpy(dtype=str),
        }
        reference_state, reference_positions = _describe_state(
            _state_id(domain.domain_id, "native_reference", 0.0, 0),
            domain.domain_id,
            domain.dataset,
            domain.analysis_role,
            bool(domain.eligible_for_training),
            domain.sequence,
            tuple(domain.sequence),
            "native_reference",
            0.0,
            0,
            0,
            0,
            residue_table,
            policy,
            config,
        )
        state_rows.append(reference_state)
        position_rows.extend(reference_positions)
        for kind in config.state_bank.kinds:
            for ratio in config.state_bank.corruption_levels:
                for replicate in range(config.state_bank.samples_per_domain_kind_level):
                    sampled = sample_state(
                        kind,
                        domain.sequence,
                        residue_features,
                        ratio,
                        policy,
                        domain.domain_id,
                        rng,
                        config.state_bank,
                    )
                    if sampled is None:
                        skipped.append(
                            {
                                "domain_id": domain.domain_id,
                                "state_kind": kind,
                                "requested_corruption_ratio": ratio,
                                "replicate": replicate,
                                "reason": "insufficient_target_environment_positions",
                            }
                        )
                        continue
                    state_id = _state_id(domain.domain_id, kind, ratio, replicate)
                    state_row, per_position = _describe_state(
                        state_id,
                        domain.domain_id,
                        domain.dataset,
                        domain.analysis_role,
                        bool(domain.eligible_for_training),
                        domain.sequence,
                        sampled.tokens,
                        kind,
                        ratio,
                        replicate,
                        sampled.timestep,
                        sampled.policy_calls,
                        residue_table,
                        policy,
                        config,
                    )
                    state_rows.append(state_row)
                    position_rows.extend(per_position)
    bank = StateBank(
        states=pd.DataFrame(state_rows, columns=STATE_COLUMNS),
        positions=pd.DataFrame(position_rows, columns=POSITION_COLUMNS),
    )
    validate_state_bank(bank)
    return bank, pd.DataFrame(skipped)


def write_state_bank(
    directory: Path,
    bank: StateBank,
    policy: SequencePolicy,
    config: ProjectConfig,
    skipped: pd.DataFrame | None = None,
) -> dict[str, Any]:
    validate_state_bank(bank)
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "states.parquet"
    position_path = directory / "positions.parquet"
    write_parquet(state_path, bank.states)
    write_parquet(position_path, bank.positions)
    manifest: dict[str, Any] = {
        **runtime_manifest(config.paths.project_root),
        "schema_version": config.schema_version,
        "data_mode": config.data_mode,
        "seed": config.seed,
        "parameters": config.state_bank.model_dump(mode="json"),
        "automatic_reference_state": {
            "state_kind": "native_reference",
            "requested_corruption_ratio": 0.0,
            "states_per_domain": 1,
        },
        "policy": {
            "policy_id": policy.policy_id,
            "model_revision": policy.model_revision,
            "state_conditioned": policy.state_conditioned,
        },
        "states": table_manifest(state_path, bank.states),
        "positions": table_manifest(position_path, bank.positions),
    }
    if skipped is not None:
        skipped_path = directory / "skipped_states.parquet"
        write_parquet(skipped_path, skipped)
        manifest["skipped"] = table_manifest(skipped_path, skipped)
    write_json(directory / "manifest.json", manifest)
    return manifest


def load_state_bank(directory: Path) -> StateBank:
    bank = StateBank(
        states=pd.read_parquet(directory / "states.parquet"),
        positions=pd.read_parquet(directory / "positions.parquet"),
    )
    validate_state_bank(bank)
    return bank


def validate_state_bank(bank: StateBank) -> None:
    missing_states = set(STATE_COLUMNS) - set(bank.states.columns)
    missing_positions = set(POSITION_COLUMNS) - set(bank.positions.columns)
    if missing_states or missing_positions:
        raise ValueError(
            f"state bank schema mismatch: states={sorted(missing_states)}, "
            f"positions={sorted(missing_positions)}"
        )
    if bank.states["state_id"].duplicated().any():
        raise ValueError("state_id values must be unique")
    if bank.positions.duplicated(["state_id", "position"]).any():
        raise ValueError("state position keys must be unique")
    expected = bank.states.set_index("state_id")["reference_sequence"].str.len()
    observed = bank.positions.groupby("state_id", observed=True).size().reindex(expected.index)
    if not observed.equals(expected.astype(int)):
        raise ValueError("every state must contain one row per sequence position")


def _describe_state(
    state_id: str,
    domain_id: str,
    dataset: str,
    analysis_role: str,
    eligible_for_training: bool,
    reference: str,
    tokens: tuple[str, ...],
    kind: str,
    requested_ratio: float,
    replicate: int,
    timestep: int,
    policy_calls: int,
    residue_table: pd.DataFrame,
    policy: SequencePolicy,
    config: ProjectConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    log_probabilities = policy.log_probabilities(domain_id, tokens)
    validate_policy_output(log_probabilities, len(reference))
    probability = np.exp(log_probabilities)
    entropy = -(probability * log_probabilities).sum(axis=1)
    sorted_probability = np.sort(probability, axis=1)
    margin = sorted_probability[:, -1] - sorted_probability[:, -2]
    top1_index = np.argmax(log_probabilities, axis=1)
    corrupted = np.array([token != native for token, native in zip(tokens, reference, strict=True)])
    error_positions = np.flatnonzero(corrupted).astype(int).tolist()
    mask_count = sum(token == config.state_bank.mask_token for token in tokens)
    core = residue_table["burial"].to_numpy(dtype=str) == "buried"
    modified_core_fraction = float((corrupted & core).sum() / max(1, core.sum()))
    edit_fraction = float(corrupted.mean())
    scaffold = float(
        np.exp(-config.state_bank.scaffold_edit_decay * edit_fraction)
        * np.exp(-config.state_bank.scaffold_core_penalty * modified_core_fraction)
    )
    state = {
        "state_id": state_id,
        "domain_id": domain_id,
        "dataset": dataset,
        "analysis_role": analysis_role,
        "eligible_for_training": eligible_for_training,
        "state_kind": kind,
        "replicate": replicate,
        "reference_sequence": reference,
        "state_sequence": "".join(tokens),
        "requested_corruption_ratio": requested_ratio,
        "corruption_ratio": edit_fraction,
        "edit_distance": len(error_positions),
        "edit_distance_fraction": edit_fraction,
        "mask_count": mask_count,
        "student_entropy": float(entropy.mean()),
        "student_top1_margin": float(margin.mean()),
        "error_positions": error_positions,
        "timestep": timestep,
        "scaffold_compatibility": scaffold,
        "modified_core_fraction": modified_core_fraction,
        "sequence_policy_calls": policy_calls + 1,
    }
    positions: list[dict[str, Any]] = []
    for position, (native, current) in enumerate(zip(reference, tokens, strict=True)):
        residue = residue_table.iloc[position]
        position_row = {
            "state_id": state_id,
            "domain_id": domain_id,
            "dataset": dataset,
            "analysis_role": analysis_role,
            "eligible_for_training": eligible_for_training,
            "position": position,
            "native_aa": native,
            "current_aa": current,
            "is_corrupted": bool(corrupted[position]),
            "is_masked": current == config.state_bank.mask_token,
            "burial": residue["burial"],
            "secondary_structure": residue["secondary_structure"],
            "contact_class": residue["contact_class"],
            "rsa": float(residue["rsa"]),
            "contact_degree": int(residue["contact_degree"]),
            "conservation_score": float(residue["conservation_score"]),
            "conservation_class": residue["conservation_class"],
            "student_entropy": float(entropy[position]),
            "student_top1_margin": float(margin[position]),
            "student_top1_aa": AA_ALPHABET[int(top1_index[position])],
            "student_logp_native": float(log_probabilities[position, AA_TO_INDEX[native]]),
        }
        for aa_index, aa in enumerate(AA_ALPHABET):
            position_row[f"student_logp_{aa}"] = float(log_probabilities[position, aa_index])
        positions.append(position_row)
    return state, positions


def _state_id(domain_id: str, kind: str, ratio: float, replicate: int) -> str:
    ratio_code = int(round(ratio * 1000))
    return f"{domain_id}:{kind}:c{ratio_code:03d}:r{replicate:03d}"
