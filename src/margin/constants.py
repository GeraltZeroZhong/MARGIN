"""Shared scientific constants and canonical column names."""

from __future__ import annotations

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: index for index, aa in enumerate(AA_ALPHABET)}
INDEX_TO_AA = dict(enumerate(AA_ALPHABET))
MASK_TOKEN = "X"
SCHEMA_VERSION = "foundation.v1"

BACKBONE_ATOMS = ("N", "CA", "C", "O")
STRUCTURAL_ENVIRONMENTS = (
    "buried",
    "intermediate",
    "exposed",
    "helix",
    "strand",
    "turn_or_coil",
    "high_contact",
    "low_contact",
)


def aa_score_columns(prefix: str = "score") -> list[str]:
    """Return canonical score columns in the one allowed amino-acid order."""

    return [f"{prefix}_{aa}" for aa in AA_ALPHABET]
