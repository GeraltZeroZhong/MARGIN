import numpy as np

from margin.studies.external_validation.evaluation import _variant_methods


def test_variant_methods_use_matched_action_and_g_cplus() -> None:
    shape = (2, 20)
    matrices = {
        "esm2_150M_sequence_action": np.ones(shape),
        "temperature_consensus_a": np.full(shape, 2.0),
        "temperature_consensus_g": np.full(shape, 3.0),
        "temperature_consensus_c_plus": np.full(shape, 5.0),
        "unscaled_consensus_a": np.full(shape, 7.0),
        "unscaled_consensus_g": np.full(shape, 11.0),
        "unscaled_consensus_c_plus": np.full(shape, 13.0),
    }
    for teacher_index, teacher in enumerate(("mif", "esm_if1", "proteinmpnn")):
        matrices[f"{teacher}_a"] = np.full(shape, teacher_index + 1.0)
        matrices[f"{teacher}_g"] = np.full(shape, teacher_index + 2.0)
        matrices[f"{teacher}_c_plus"] = np.full(shape, teacher_index + 3.0)
    methods = _variant_methods(matrices, np.array([0, 1]), np.array([2, 3]))
    assert np.all(methods["temperature_consensus_action"] == 2.0)
    assert np.all(methods["temperature_consensus_g_plus_c_plus"] == 8.0)
    assert np.all(methods["sequence_plus_temperature_action"] == 3.0)
    assert np.all(methods["sequence_plus_temperature_g_plus_c_plus"] == 9.0)
