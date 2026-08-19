from pathlib import Path

import pandas as pd

from margin.studies.external_validation.panel import (
    _scan_candidates,
    load_external_validation_config,
)


def test_cross_platform_protocol_is_frozen() -> None:
    config = load_external_validation_config(Path("configs/external_validation.yaml"))
    assert config.status == "FROZEN_BEFORE_CROSS_PLATFORM_MODEL_SCORING"
    assert config.panel.minimum_selected_domains == 8
    assert config.inference.routing_allowed is False
    assert config.inference.changes_primary_decision is False


def test_candidate_scan_does_not_require_outcomes() -> None:
    config = load_external_validation_config(Path("configs/external_validation.yaml"))
    rows = []
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    for position in range(10):
        for offset in range(2 if position < 5 else 1):
            mutant = sequence[(position + offset + 1) % len(sequence)]
            rows.append(
                {
                    "experiment_id": f"x{position}_{offset}",
                    "protein_name": "example",
                    "uniprot_id": "P00000",
                    "pdb_id_corrected": "1ABC",
                    "chain": "A",
                    "pdb_position": position,
                    "wild_type": sequence[position],
                    "mutation": mutant,
                    "pdb_sequence": sequence,
                }
            )
    scan, mutations, exclusions = _scan_candidates(pd.DataFrame(rows), config)
    assert len(mutations) == 15
    assert bool(scan.loc[0, "metadata_eligible"])
    assert exclusions == []
    assert "ddG" not in scan.columns
