from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from margin.studies.generalization.config import load_generalization_config
from margin.studies.generalization.dms import (
    _calibration_slope,
    _ndcg,
    _stabilizing_topk_recall,
    fit_rrr_predict,
)
from margin.studies.generalization.report import _architecture_table_rows


def test_generalization_config_locks_required_architecture_matrix() -> None:
    config = load_generalization_config(Path("configs/generalization.yaml"))
    assert {model.model_id for model in config.architecture.models} == {
        "carp_76M",
        "carp_640M",
        "esm2_150M",
        "esm2_650M",
        "esm1b_650M",
    }
    assert config.architecture.primary_target == "consensus_leave_mifst_out"
    assert config.dms.primary_alpha == 1.0


def test_rank_reduced_predictor_returns_zero_sum_residuals() -> None:
    rng = np.random.default_rng(7)
    x_train = rng.normal(size=(80, 12))
    y_train = rng.normal(size=(80, 20))
    y_train -= y_train.mean(axis=1, keepdims=True)
    prediction = fit_rrr_predict(
        x_train,
        y_train,
        rng.normal(size=(11, 12)),
        rank=4,
        alpha=10.0,
    )
    assert prediction.shape == (11, 20)
    np.testing.assert_allclose(prediction.sum(axis=1), 0.0, atol=1e-10)


def test_dms_ranking_and_calibration_metrics_have_registered_direction() -> None:
    observed = np.array([-2.0, -1.0, 0.5, 1.0, 2.0])
    perfect = observed.copy()
    reversed_score = observed[::-1]
    assert _ndcg(perfect, observed) > _ndcg(reversed_score, observed)
    assert _stabilizing_topk_recall(perfect, observed, 0.4) == 1.0
    assert np.isclose(_calibration_slope(perfect, observed), 1.0)


def test_architecture_report_formats_boolean_decision_once() -> None:
    frame = pd.DataFrame(
        [
            {
                "model_id": "carp_640M",
                "family": "CARP",
                "jsd_reduction_nats": 0.0123,
                "control_unique_margin_nats": 0.0045,
                "passed": True,
            }
        ]
    )
    assert _architecture_table_rows(frame).endswith("| PASS |")
