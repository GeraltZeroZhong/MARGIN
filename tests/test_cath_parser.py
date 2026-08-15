from __future__ import annotations

from pathlib import Path

import pytest

from margin.data_registry.cath import read_cath_fasta
from margin.preprocessing.structure import _sequence_mapping


def test_official_cath_fasta_header_extracts_domain_id(tmp_path: Path) -> None:
    path = tmp_path / "cath.fa"
    path.write_text(">cath|4_4_0|4a4jA00/2-70 example\nACDEFG\n>1abcB01/4-9\nHIKLMN\n")
    assert read_cath_fasta(path) == {"4a4jA00": "ACDEFG", "1abcB01": "HIKLMN"}


def test_duplicate_cath_domain_header_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.fa"
    path.write_text(">cath|4_4_0|4a4jA00/2-70\nACDEFG\n>cath|4_4_0|4a4jA00/2-70\nACDEFG\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_cath_fasta(path)


def test_structure_mapping_never_assigns_coordinates_across_a_residue_mismatch() -> None:
    assert _sequence_mapping("ACDE", "AXDE") == {0: 0, 2: 2, 3: 3}
