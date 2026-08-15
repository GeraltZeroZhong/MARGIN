"""Extract identity-safe residue features from experimental backbones."""

from __future__ import annotations

import gzip
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB.Polypeptide import protein_letters_3to1

from margin.config import RegistryConfig
from margin.constants import BACKBONE_ATOMS


@dataclass(frozen=True)
class ParsedResidue:
    sequence_letter: str
    chain_id: str
    residue_id: tuple[str, int, str]
    residue_id_text: str
    coordinates: dict[str, np.ndarray]


def preprocess_domain_structure(
    domain_id: str,
    sequence: str,
    path: Path,
    chain_id: str,
    config: RegistryConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a sequence-aligned, backbone-only residue table and domain summary."""

    model, selected_chain, observed = _parse_structure(path, chain_id)
    observed_sequence = "".join(residue.sequence_letter for residue in observed)
    mapping = _sequence_mapping(sequence, observed_sequence)
    dssp = _dssp_annotations(model, path, config)

    rows: list[dict[str, Any]] = []
    for position, residue_letter in enumerate(sequence):
        observed_index = mapping.get(position)
        if observed_index is None:
            rows.append(_missing_row(domain_id, position, residue_letter))
            continue
        residue = observed[observed_index]
        annotation = dssp.get((residue.chain_id, residue.residue_id))
        if annotation is None and config.require_dssp:
            raise ValueError(f"DSSP annotation is missing for {domain_id} position {position}")
        dssp_code, rsa = annotation if annotation is not None else ("C", np.nan)
        row = {
            "domain_id": domain_id,
            "position": position,
            "residue": residue_letter,
            "pdb_residue_id": residue.residue_id_text,
            "is_resolved": True,
            "has_complete_backbone": all(atom in residue.coordinates for atom in BACKBONE_ATOMS),
            "dssp": dssp_code,
            "secondary_structure": _secondary_structure(dssp_code),
            "rsa": float(rsa),
        }
        for atom in BACKBONE_ATOMS:
            coordinate = residue.coordinates.get(atom, np.full(3, np.nan))
            for axis, value in zip("xyz", coordinate, strict=True):
                row[f"{atom.lower()}_{axis}"] = float(value)
        rows.append(row)

    table = pd.DataFrame(rows)
    ca = table[[f"ca_{axis}" for axis in "xyz"]].to_numpy(dtype=float)
    table["contact_degree"] = _contact_degree(
        ca,
        cutoff=config.contact_distance_angstrom,
        minimum_sequence_separation=config.contact_minimum_sequence_separation,
    )
    table["burial"] = table["rsa"].map(lambda value: _burial(value, config))
    table["contact_class"] = np.where(
        table["contact_degree"] >= config.high_contact_degree_min,
        "high_contact",
        "low_contact",
    )
    table["conservation_score"] = np.nan
    table["conservation_class"] = "unavailable"
    missing_count = int((~table["has_complete_backbone"]).sum())
    summary = {
        "selected_chain": selected_chain,
        "missing_residue_count": missing_count,
        "missing_fraction": missing_count / len(sequence),
        "helix_fraction": float((table["secondary_structure"] == "helix").mean()),
        "strand_fraction": float((table["secondary_structure"] == "strand").mean()),
    }
    return table, summary


def _parse_structure(path: Path, requested_chain: str) -> tuple[Any, str, list[ParsedResidue]]:
    suffixes = path.suffixes
    structure_suffix = suffixes[-2] if suffixes and suffixes[-1] == ".gz" else path.suffix
    parser = (
        MMCIFParser(QUIET=True) if structure_suffix in {".cif", ".mmcif"} else PDBParser(QUIET=True)
    )
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            structure = parser.get_structure(path.stem, handle)
    else:
        structure = parser.get_structure(path.stem, str(path))
    model = next(structure.get_models())
    chains = list(model.get_chains())
    selected = next((chain for chain in chains if chain.id == requested_chain), None)
    if selected is None and len(chains) == 1:
        selected = chains[0]
    if selected is None:
        available = [chain.id for chain in chains]
        raise ValueError(f"chain {requested_chain!r} not found; available chains are {available}")

    residues: list[ParsedResidue] = []
    for residue in selected.get_residues():
        hetero, number, insertion = residue.id
        if hetero.strip():
            continue
        letter = protein_letters_3to1.get(residue.resname.upper(), "X")
        if letter == "X":
            continue
        coordinates = {
            atom: residue[atom].coord.astype(float) for atom in BACKBONE_ATOMS if atom in residue
        }
        residues.append(
            ParsedResidue(
                sequence_letter=letter,
                chain_id=selected.id,
                residue_id=residue.id,
                residue_id_text=f"{number}{insertion.strip()}",
                coordinates=coordinates,
            )
        )
    if not residues:
        raise ValueError(f"no standard residues found in {path}")
    return model, selected.id, residues


def _sequence_mapping(reference: str, observed: str) -> dict[int, int]:
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -3.0
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(reference, observed)[0]
    return {
        int(reference_index): int(observed_index)
        for reference_index, observed_index in zip(*alignment.indices, strict=True)
        if reference_index >= 0
        and observed_index >= 0
        and reference[int(reference_index)] == observed[int(observed_index)]
    }


def _dssp_annotations(
    model: Any, path: Path, config: RegistryConfig
) -> dict[tuple[str, tuple[str, int, str]], tuple[str, float]]:
    executable = shutil.which(config.dssp_executable)
    if executable is None:
        if config.require_dssp:
            raise RuntimeError(f"DSSP executable not found: {config.dssp_executable}")
        return {}
    if path.suffix == ".gz":
        suffix = path.suffixes[-2] if len(path.suffixes) >= 2 else ".pdb"
        with tempfile.NamedTemporaryFile(suffix=suffix) as extracted:
            with gzip.open(path, "rb") as compressed:
                shutil.copyfileobj(compressed, extracted)
            extracted.flush()
            annotations = _run_dssp(model, extracted.name, executable)
            return _format_dssp(annotations)
    if path.suffix.lower() == ".pdb" and not _has_pdb_header(path):
        # Public model archives used by counterfactual study contain valid ATOM records but
        # omit HEADER. DSSP 4.x then attempts mmCIF parsing despite the .pdb
        # suffix, so supply the format marker in a temporary view.
        with tempfile.NamedTemporaryFile(suffix=".pdb") as named_pdb:
            named_pdb.write(b"HEADER    PROTEIN STRUCTURE\n")
            with path.open("rb") as source:
                shutil.copyfileobj(source, named_pdb)
            named_pdb.flush()
            annotations = _run_dssp(model, named_pdb.name, executable)
            return _format_dssp(annotations)
    if path.suffix.lower() not in {".pdb", ".cif", ".mmcif"}:
        # CATH's non-redundant PDB archives use bare domain identifiers as
        # filenames.  Bio.PDB parses their PDB records correctly, but DSSP
        # infers the input format exclusively from the filename suffix.
        with tempfile.NamedTemporaryFile(suffix=".pdb") as named_pdb:
            named_pdb.write(b"HEADER    CATH DOMAIN\n")
            with path.open("rb") as source:
                shutil.copyfileobj(source, named_pdb)
            named_pdb.flush()
            annotations = _run_dssp(model, named_pdb.name, executable)
            return _format_dssp(annotations)
    annotations = _run_dssp(model, str(path), executable)
    return _format_dssp(annotations)


def _has_pdb_header(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(6) == b"HEADER"


def _run_dssp(model: Any, path: str, executable: str) -> Any:
    """Normalize Biopython's generic DSSP failure into a per-domain runtime error."""

    try:
        return DSSP(model, path, dssp=executable)
    except Exception as error:  # Bio.PDB.DSSP raises base Exception for tool failures.
        raise RuntimeError(f"DSSP failed for {path}: {error}") from error


def _format_dssp(annotations: Any) -> dict[tuple[str, tuple[str, int, str]], tuple[str, float]]:
    result: dict[tuple[str, tuple[str, int, str]], tuple[str, float]] = {}
    # ``DSSP`` inherits ``AbstractResiduePropertyMap``. Iterating over recent
    # Biopython releases yields property values, whereas indexing requires the
    # explicit ``(chain_id, residue_id)`` keys.
    for key in annotations.keys():  # noqa: SIM118 - DSSP iteration yields values, not keys
        values = annotations[key]
        code = values[2] if values[2] != "-" else "C"
        result[(key[0], key[1])] = (code, float(values[3]))
    return result


def _contact_degree(
    ca_coordinates: np.ndarray, cutoff: float, minimum_sequence_separation: int
) -> np.ndarray:
    finite = np.isfinite(ca_coordinates).all(axis=1)
    safe = np.where(finite[:, None], ca_coordinates, 0.0)
    distance = np.linalg.norm(safe[:, None, :] - safe[None, :, :], axis=-1)
    index = np.arange(len(ca_coordinates))
    separated = np.abs(index[:, None] - index[None, :]) >= minimum_sequence_separation
    contacts = (distance <= cutoff) & separated & finite[:, None] & finite[None, :]
    return contacts.sum(axis=1).astype(int)


def _missing_row(domain_id: str, position: int, residue: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "domain_id": domain_id,
        "position": position,
        "residue": residue,
        "pdb_residue_id": "",
        "is_resolved": False,
        "has_complete_backbone": False,
        "dssp": "",
        "secondary_structure": "missing",
        "rsa": np.nan,
        "burial": "missing",
        "contact_degree": 0,
        "contact_class": "missing",
        "conservation_score": np.nan,
        "conservation_class": "unavailable",
    }
    for atom in BACKBONE_ATOMS:
        for axis in "xyz":
            row[f"{atom.lower()}_{axis}"] = np.nan
    return row


def _secondary_structure(code: str) -> str:
    if code in {"H", "G", "I"}:
        return "helix"
    if code in {"E", "B"}:
        return "strand"
    return "turn_or_coil"


def _burial(rsa: float, config: RegistryConfig) -> str:
    if not np.isfinite(rsa):
        return "missing"
    if rsa <= config.buried_rsa_max:
        return "buried"
    if rsa >= config.exposed_rsa_min:
        return "exposed"
    return "intermediate"
