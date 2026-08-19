from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from margin.studies.observability.config import load_observability_config
from margin.studies.observability.layerwise import replication_split_indices
from margin.studies.observability.probes import shuffled_target
from margin.studies.observability.report import (
    evaluate_probe_summary,
    select_observability_decision,
)
from margin.studies.observability.targets import (
    ResidualDataset,
    clr,
    fit_temperature,
    ilr,
    inverse_ilr,
)
from margin.teachers.runner_cache import (
    completed_request_ids,
    finalize_parts,
    part_directory,
    write_request_part,
)


def test_clr_and_temperature_calibration() -> None:
    values = np.array([[1.0, 2.0, 4.0], [-2.0, 3.0, 1.0]])
    transformed = clr(values)
    assert np.allclose(transformed.mean(axis=1), 0.0)

    logp = np.array([[0.0, -2.0, -3.0], [-2.0, 0.0, -3.0], [-1.0, -2.0, 0.0]])
    temperature, before, after = fit_temperature(logp, np.array([0, 1, 2]), (0.25, 4.0))
    assert 0.25 <= temperature <= 4.0
    assert after <= before + 1e-10

    round_trip = inverse_ilr(ilr(values))
    assert np.allclose(round_trip, transformed)


def test_hierarchical_shuffle_stays_inside_domain() -> None:
    metadata = pd.DataFrame(
        {
            "domain_id": ["a"] * 8 + ["b"] * 8,
            "native_aa": ["A", "C"] * 8,
        }
    )
    target = np.arange(16 * 3, dtype=float).reshape(16, 3)
    shuffled, moved = shuffled_target(
        metadata,
        target,
        np.arange(16),
        "within_domain",
        np.random.default_rng(7),
    )
    assert moved > 0
    assert {tuple(row) for row in shuffled[:8]} == {tuple(row) for row in target[:8]}
    assert {tuple(row) for row in shuffled[8:]} == {tuple(row) for row in target[8:]}


def test_replication_split_indices_use_locked_roles() -> None:
    metadata = pd.DataFrame(
        {
            "analysis_role": [
                "development_train",
                "development_validation",
                "locked_test",
                "development_train",
            ]
        }
    )
    zeros = np.zeros((4, 20))
    dataset = ResidualDataset(metadata, zeros, {}, {}, pd.DataFrame())
    selection_train, validation, final_train, test = replication_split_indices(dataset)
    assert selection_train.tolist() == [0, 3]
    assert validation.tolist() == [1]
    assert final_train.tolist() == [0, 1, 3]
    assert test.tolist() == [2]


def test_teacher_runner_parts_resume_and_finalize(tmp_path) -> None:
    output = tmp_path / "raw.parquet"
    directory = part_directory(output, "fixed-run")
    write_request_part(
        directory,
        0,
        [{"request_id": "r0", "position": 0, "score_A": 1.0}],
    )
    write_request_part(
        directory,
        1,
        [{"request_id": "r1", "position": 0, "score_A": 2.0}],
    )
    assert completed_request_ids(directory) == {"r0", "r1"}
    finalize_parts(output, directory, ["r0", "r1"])
    table = pd.read_parquet(output).sort_values("request_id", ignore_index=True)
    assert table["score_A"].tolist() == [1.0, 2.0]


def test_lora_injection_keeps_base_frozen_and_state_is_small() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    from margin.studies.observability.lora import (
        adapter_state,
        inject_esm2_lora,
        load_adapter_state,
    )

    class SelfAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.query = nn.Linear(5, 5)
            self.value = nn.Linear(5, 5)

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention = nn.Module()
            self.attention.self = SelfAttention()

    class FakeEsm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.layer = nn.ModuleList([Block(), Block()])

    class FakeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.esm = FakeEsm()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FakeModel().to(device=device, dtype=torch.float64)
    original = model.esm.encoder.layer[1].attention.self.query
    sample = torch.randn(3, 5, device=device, dtype=torch.float64)
    expected = original(sample).detach()
    adapters = inject_esm2_lora(model, [1], rank=2, alpha=4.0)
    observed = model.esm.encoder.layer[1].attention.self.query(sample)
    assert torch.allclose(observed, expected)
    assert all(not parameter.requires_grad for parameter in original.parameters())
    assert all(
        module.lora_a.weight.device == original.weight.device for module in adapters.values()
    )
    assert all(module.lora_a.weight.dtype == original.weight.dtype for module in adapters.values())
    state = adapter_state(adapters)
    for module in adapters.values():
        nn.init.ones_(module.lora_b.weight)
    load_adapter_state(adapters, state)
    assert all(torch.count_nonzero(module.lora_b.weight) == 0 for module in adapters.values())


def test_observability_success_rule_requires_every_control() -> None:
    config = load_observability_config(Path("configs/observability.yaml"))
    common = {
        "target_id": "mifst",
        "probe": "layerwise_ridge",
        "feature_kind": "query",
        "layer": 12,
        "target_rank": np.nan,
        "evaluation_split": "locked_test",
        "wild_ci_low": 0.011,
        "wild_ci_high": 0.031,
        "positive_domains": 35,
        "negative_domains": 13,
        "zero_domains": 0,
        "n_domains": 48,
        "n_rows": 100,
    }
    rows = [
        {
            **common,
            "metric": "jsd_reduction_nats",
            "estimate": 0.02,
            "control": "observed",
            "repeat": 0,
        },
        {
            **common,
            "metric": "residual_cosine",
            "estimate": 0.20,
            "control": "observed",
            "repeat": 0,
        },
        {
            **common,
            "metric": "cross_entropy_reduction_nats",
            "estimate": 0.03,
            "control": "observed",
            "repeat": 0,
        },
        {
            **common,
            "metric": "candidate_rank_agreement",
            "estimate": 0.50,
            "control": "observed",
            "repeat": 0,
        },
    ]
    for control in config.probes.shuffle_controls:
        for repeat in range(config.probes.control_repeats):
            rows.append(
                {
                    **common,
                    "metric": "jsd_reduction_nats",
                    "estimate": 0.005,
                    "control": control,
                    "repeat": repeat,
                }
            )
    result = evaluate_probe_summary(
        pd.DataFrame(rows),
        config,
        source_id="test",
        model_id="esm2_150m",
        allowed_probes={"layerwise_ridge"},
        decision_eligible=True,
        scope="global",
    )
    assert result.iloc[0]["status"] == "PASS"

    incomplete = pd.DataFrame(rows[:-1])
    result = evaluate_probe_summary(
        incomplete,
        config,
        source_id="test",
        model_id="esm2_150m",
        allowed_probes={"layerwise_ridge"},
        decision_eligible=True,
        scope="global",
    )
    assert result.iloc[0]["status"] == "INCOMPLETE"


def test_observability_decision_keeps_environment_success_selective() -> None:
    probes = pd.DataFrame(
        {
            "decision_eligible": [True],
            "status": ["FAIL"],
            "model_id": ["esm2_150m"],
            "probe": ["layerwise_ridge"],
            "route_id": ["ridge"],
        }
    )
    environments = pd.DataFrame({"status": ["PASS"], "route_id": ["core30"]})
    decision, _ = select_observability_decision(probes, environments)
    assert decision == "SELECTIVE_HYBRID"
