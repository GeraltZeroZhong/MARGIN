from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from margin.attribution.teacher_value import _dms_audit
from margin.config import ProjectConfig
from margin.constants import AA_ALPHABET
from margin.data_registry.registry import RegistryTables
from margin.decoys.generate import build_decoys
from margin.state_sampling.bank import build_state_bank
from margin.state_sampling.policy import SyntheticSequencePolicy
from margin.teachers.cache import merge_score_tables, write_teacher_cache
from margin.teachers.requests import export_teacher_requests
from margin.teachers.schema import logp_columns
from margin.teachers.synthetic import build_synthetic_teacher_scores


def test_synthetic_teacher_matrix_is_normalized_and_fully_covered(
    tmp_path,
    synthetic_config: ProjectConfig,
    synthetic_registry: RegistryTables,
) -> None:
    references = synthetic_registry.domains.set_index("domain_id")["sequence"].to_dict()
    policy = SyntheticSequencePolicy(references)
    bank, _ = build_state_bank(synthetic_registry, policy, synthetic_config)
    decoys = build_decoys(synthetic_registry, synthetic_config)
    requests = export_teacher_requests(
        tmp_path / "requests", bank, synthetic_registry, decoys, synthetic_config
    )
    cache = merge_score_tables(
        build_synthetic_teacher_scores(bank, synthetic_registry, decoys, synthetic_config)
    )
    values = cache.scores[logp_columns()].to_numpy(dtype=float)
    np.testing.assert_allclose(logsumexp(values, axis=1), 0.0, atol=1e-8)
    manifest = write_teacher_cache(
        tmp_path / "cache",
        cache,
        synthetic_config,
        requests=requests.requests,
    )
    assert manifest["cache_compatibility_key"]
    coverage = pd.read_parquet(tmp_path / "cache" / "coverage_audit.parquet")
    assert coverage["required"].all()
    np.testing.assert_allclose(coverage["coverage_fraction"], 1.0)
    stored = json.loads((tmp_path / "cache" / "manifest.json").read_text())
    assert stored["shards"]


def test_dms_audit_records_unknown_positions_and_rejects_native_mismatch(
    synthetic_config: ProjectConfig,
    synthetic_registry: RegistryTables,
) -> None:
    references = synthetic_registry.domains.set_index("domain_id")["sequence"].to_dict()
    bank, _ = build_state_bank(
        synthetic_registry, SyntheticSequencePolicy(references), synthetic_config
    )
    decoys = build_decoys(synthetic_registry, synthetic_config)
    cache = merge_score_tables(
        build_synthetic_teacher_scores(bank, synthetic_registry, decoys, synthetic_config)
    )
    first = bank.positions.iloc[0]
    mutant = next(amino_acid for amino_acid in AA_ALPHABET if amino_acid != first.native_aa)
    dms = pd.DataFrame(
        [
            {
                "assay_id": "known",
                "domain_id": first.domain_id,
                "position": int(first.position),
                "wild_type": first.native_aa,
                "mutant": mutant,
                "effect": 1.0,
            },
            {
                "assay_id": "unknown",
                "domain_id": "NOT_IN_BANK",
                "position": 0,
                "wild_type": "A",
                "mutant": "C",
                "effect": 0.0,
            },
        ]
    )
    predictions, _, coverage = _dms_audit(cache.scores, bank, dms, synthetic_config)
    assert predictions["assay_id"].eq("known").all()
    unknown = coverage.loc[coverage["assay_id"] == "unknown"]
    assert set(unknown["analysis_role"]) == {"unknown"}
    assert (unknown["status"] == "unscored").all()
    assert (unknown["scored_variants"] == 0).all()

    mismatched = dms.iloc[[0]].copy()
    mismatched["wild_type"] = mutant
    with pytest.raises(ValueError, match="wild_type disagrees"):
        _dms_audit(cache.scores, bank, mismatched, synthetic_config)
