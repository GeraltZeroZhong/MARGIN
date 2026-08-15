from __future__ import annotations

from pathlib import Path

import numpy as np

from margin.config import ProjectConfig
from margin.data_registry.conservation import attach_conservation
from margin.data_registry.homology import parse_mmseqs_hits
from margin.data_registry.registry import RegistryTables


def test_mmseqs_parser_normalizes_percent_and_keeps_best_alignment(tmp_path: Path) -> None:
    raw = tmp_path / "hits.tsv"
    raw.write_text(
        "D1\tB1\t42.0\t90.0\t80.0\nD1\tB1\t35.0\t100.0\t100.0\nD2\tB2\t0.50\t0.70\t0.60\n"
    )
    hits = parse_mmseqs_hits(raw)
    assert len(hits) == 2
    first = hits.loc[(hits["domain_id"] == "D1") & (hits["benchmark_id"] == "B1")].iloc[0]
    assert first["sequence_identity"] == 0.42
    assert first["query_coverage"] == 0.90


def test_conservation_attachment_requires_complete_bounded_scores(
    synthetic_config: ProjectConfig, synthetic_registry: RegistryTables
) -> None:
    source = synthetic_registry.residues[["domain_id", "position"]].copy()
    source["conservation_score"] = np.where(source["position"] % 2 == 0, 0.9, 0.2)
    attached = attach_conservation(synthetic_registry, source, synthetic_config)
    assert set(attached.residues["conservation_class"]) == {"conserved", "variable"}
    incomplete = source.iloc[:-1]
    try:
        attach_conservation(synthetic_registry, incomplete, synthetic_config)
    except ValueError as error:
        assert "lacks 1" in str(error)
    else:
        raise AssertionError("incomplete conservation input was accepted")
