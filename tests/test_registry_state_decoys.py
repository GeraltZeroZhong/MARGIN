from __future__ import annotations

import numpy as np

from margin.config import ProjectConfig
from margin.data_registry.leakage import audit_benchmark_leakage
from margin.data_registry.registry import RegistryTables
from margin.decoys.generate import build_decoys
from margin.decoys.graph import contact_edges, degrees
from margin.fixtures import build_synthetic_leakage_inputs
from margin.state_sampling.bank import build_state_bank
from margin.state_sampling.policy import SyntheticSequencePolicy


def test_leakage_catches_exact_sequence_and_homology(
    synthetic_config: ProjectConfig, synthetic_registry: RegistryTables
) -> None:
    benchmarks, homology = build_synthetic_leakage_inputs(synthetic_registry)
    audit = audit_benchmark_leakage(
        synthetic_registry.domains, benchmarks, homology, synthetic_config
    )
    reasons = set(audit.relations["reason"])
    assert {"exact_pdb_chain", "exact_sequence", "sequence_identity"} <= reasons
    assert audit.summary["excluded_domains"] == 4


def test_state_bank_is_reproducible_and_covers_all_declared_kinds(
    synthetic_config: ProjectConfig, synthetic_registry: RegistryTables
) -> None:
    references = synthetic_registry.domains.set_index("domain_id")["sequence"].to_dict()
    policy = SyntheticSequencePolicy(references)
    first, _ = build_state_bank(synthetic_registry, policy, synthetic_config)
    second, _ = build_state_bank(synthetic_registry, policy, synthetic_config)
    assert set(first.states["state_kind"]) == {
        *synthetic_config.state_bank.kinds,
        "native_reference",
    }
    assert set(first.states["requested_corruption_ratio"]) == {
        *synthetic_config.state_bank.corruption_levels,
        0.0,
    }
    native = first.states.loc[first.states["state_kind"] == "native_reference"]
    assert len(native) == synthetic_registry.domains["domain_id"].nunique()
    assert (native["edit_distance"] == 0).all()
    assert first.states.equals(second.states)
    assert first.positions.equals(second.positions)
    rollout = first.states.loc[first.states["state_kind"] == "on_policy_rollout"]
    assert (rollout["sequence_policy_calls"] > 1).all()


def test_decoy_declarations_and_rewired_degree_sequence(
    synthetic_config: ProjectConfig, synthetic_registry: RegistryTables
) -> None:
    artifacts = build_decoys(synthetic_registry, synthetic_config)
    assert set(artifacts.decoys["decoy_type"]) == {
        "matched_cath",
        "permuted",
        "contact_rewired",
        "shuffled_residue",
    }
    for row in artifacts.decoys.loc[artifacts.decoys["decoy_type"] == "contact_rewired"].itertuples(
        index=False
    ):
        residues = synthetic_registry.residues.loc[
            synthetic_registry.residues["domain_id"] == row.target_domain_id
        ].sort_values("position")
        ca = residues[["ca_x", "ca_y", "ca_z"]].to_numpy(dtype=float)
        original = contact_edges(
            ca,
            synthetic_config.registry.contact_distance_angstrom,
            synthetic_config.registry.contact_minimum_sequence_separation,
        )
        rewired_frame = artifacts.edges.loc[artifacts.edges["decoy_id"] == row.decoy_id]
        rewired = set(
            zip(
                rewired_frame["source"].astype(int),
                rewired_frame["target"].astype(int),
                strict=True,
            )
        )
        np.testing.assert_array_equal(
            degrees(original, int(row.target_length)),
            degrees(rewired, int(row.target_length)),
        )
