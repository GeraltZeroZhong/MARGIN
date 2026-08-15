from pathlib import Path

import pandas as pd

from margin.studies.structure_sensitivity.evaluation import _confidence_bin
from margin.studies.structure_sensitivity.panel import load_structure_sensitivity_config


def test_confidence_bins_follow_frozen_thresholds() -> None:
    config = load_structure_sensitivity_config(Path("configs/structure_sensitivity.yaml"))
    values = pd.Series([69.9, 70.0, 89.9, 90.0, float("nan")])
    assert _confidence_bin(values, config).tolist() == [
        "low",
        "medium",
        "medium",
        "high",
        "not_applicable",
    ]
