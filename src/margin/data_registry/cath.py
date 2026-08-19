"""CATH v4.4 classification and sequence import."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from Bio import SeqIO


def read_cath_domain_list(path: Path) -> pd.DataFrame:
    """Parse the official whitespace-delimited CATH domain-list format."""

    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 12:
                raise ValueError(f"invalid CATH domain-list row at line {line_number}")
            domain_id = fields[0]
            rows.append(
                {
                    "domain_id": domain_id,
                    "pdb_id": domain_id[:4].lower(),
                    "chain_id": domain_id[4],
                    "cath_c": fields[1],
                    "cath_a": ".".join(fields[1:3]),
                    "cath_t": ".".join(fields[1:4]),
                    "cath_h": ".".join(fields[1:5]),
                    "cath_s35": ".".join(fields[1:6]),
                    "domain_length": int(fields[10]),
                    "resolution_angstrom": _parse_resolution(fields[11]),
                }
            )
    return pd.DataFrame(rows)


def read_cath_fasta(path: Path) -> dict[str, str]:
    """Read official ``cath|version|domain/range`` and plain-ID FASTA headers."""

    sequences: dict[str, str] = {}
    for record in SeqIO.parse(path, "fasta"):
        fields = record.id.split("|")
        identifier = (
            fields[2].split("/", maxsplit=1)[0]
            if len(fields) >= 3 and fields[0].lower() == "cath"
            else fields[0].split("/", maxsplit=1)[0]
        )
        if identifier in sequences:
            raise ValueError(f"duplicate CATH FASTA domain ID: {identifier}")
        sequences[identifier] = str(record.seq).upper()
    return sequences


def locate_domain_structure(directory: Path, domain_id: str) -> Path | None:
    candidates = (
        directory / domain_id,
        directory / f"{domain_id}.pdb",
        directory / f"{domain_id}.ent",
        directory / f"{domain_id}.cif",
        directory / f"{domain_id}.pdb.gz",
        directory / domain_id[:2] / f"{domain_id}.pdb",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _parse_resolution(value: str) -> float:
    resolution = float(value)
    return float("nan") if resolution >= 999 else resolution
