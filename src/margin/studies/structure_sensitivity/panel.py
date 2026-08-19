"""Matched experimental/predicted-backbone preparation for structure-sensitivity study."""

from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1
from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from margin.constants import BACKBONE_ATOMS
from margin.provenance import (
    runtime_manifest,
    sha256_file,
    table_manifest,
    write_json,
    write_parquet,
)
from margin.studies.external_validation.panel import load_external_validation_config

POPULATION = "structure_sensitivity_fireprot_matched_structures"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class StructureSensitivityPaths(StrictModel):
    project_root: Path
    run_dir: Path
    storage_dir: Path
    external_validation_protocol: Path
    foundation_config: Path
    stability_config: Path


class StructureSensitivityPanel(StrictModel):
    minimum_matched_domains_per_predictor: PositiveInt
    require_complete_backbone: bool
    alphafold_api: str
    esmfold_api: str
    perturbation_ca_rmsd_angstrom: list[float]
    perturbation_smoothing_window: PositiveInt
    confidence_low_upper: float = Field(gt=0, le=100)
    confidence_high_lower: float = Field(gt=0, le=100)


class StructureSensitivityInference(StrictModel):
    bootstrap_replicates: PositiveInt
    confidence_level: float = Field(gt=0, lt=1)
    top_fraction: float = Field(gt=0, le=1)
    comparison_roles: list[str]
    confirmatory: bool
    routing_allowed: bool
    changes_primary_decision: bool


class StructureSensitivityConfig(StrictModel):
    schema_version: Literal["structure_sensitivity.v1"]
    status: Literal["FROZEN_AFTER_OUTCOME_OPENING_BEFORE_ALTERNATE_STRUCTURE_SCORING"]
    seed: int
    paths: StructureSensitivityPaths
    panel: StructureSensitivityPanel
    inference: StructureSensitivityInference


def load_structure_sensitivity_config(path: Path) -> StructureSensitivityConfig:
    path = path.resolve()
    with path.open(encoding="utf-8") as handle:
        config = StructureSensitivityConfig.model_validate(yaml.safe_load(handle))
    root = _resolve(path.parent, config.paths.project_root)
    config.paths.project_root = root
    for name in StructureSensitivityPaths.model_fields:
        if name != "project_root":
            setattr(config.paths, name, _resolve(root, getattr(config.paths, name)))
    return config


def prepare_structure_sensitivity_panel(
    config: StructureSensitivityConfig, *, force: bool = False
) -> dict[str, Path]:
    """Build matched structure requests before any alternate-structure scoring."""

    cross = load_external_validation_config(config.paths.external_validation_protocol)
    cross_manifest = cross.paths.run_dir / "evaluation/manifest.json"
    if not cross_manifest.exists():
        raise FileNotFoundError("completed cross-platform evaluation is required")
    output = config.paths.run_dir / "panel"
    request_dir = config.paths.run_dir / "teacher_requests"
    structure_root = config.paths.storage_dir / "structures"
    input_root = config.paths.storage_dir / "teacher_inputs"
    for directory in (output, request_dir, structure_root, input_root):
        directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "structures": output / "structures.parquet",
        "confidence": output / "residue_confidence.parquet",
        "requests": request_dir / "requests.parquet",
        "request_structures": request_dir / "structures.parquet",
        "exclusions": output / "exclusions.parquet",
        "manifest": output / "manifest.json",
        "lock": config.paths.run_dir / "protocol_lock.json",
    }
    if not force and all(path.exists() for path in paths.values()):
        return paths
    domains = pd.read_parquet(cross.paths.run_dir / "panel/domains.parquet")
    experimental_requests = pd.read_parquet(
        cross.paths.run_dir / "teacher_requests/requests.parquet"
    ).set_index("domain_id")
    structure_rows = []
    confidence_rows = []
    request_rows = []
    exclusions = []
    for domain_index, domain in enumerate(domains.sort_values("domain_id").itertuples(index=False)):
        experimental_path = Path(experimental_requests.loc[domain.domain_id, "input_path"])
        experimental = np.load(experimental_path)["coordinates"].astype(np.float32)
        _append_structure(
            domain,
            "experimental",
            experimental,
            np.full(len(domain.sequence), np.nan),
            experimental_path,
            0.0,
            structure_rows,
            confidence_rows,
            request_rows,
        )
        for amplitude in config.panel.perturbation_ca_rmsd_angstrom:
            role = f"perturbed_{str(amplitude).replace('.', 'p')}"
            perturbed = smooth_perturbation(
                experimental,
                amplitude,
                config.panel.perturbation_smoothing_window,
                seed=config.seed + domain_index * 100 + int(round(amplitude * 10)),
            )
            path = input_root / f"{_safe(domain.domain_id)}__{role}.npz"
            np.savez_compressed(path, coordinates=perturbed)
            _append_structure(
                domain,
                role,
                perturbed,
                np.full(len(domain.sequence), np.nan),
                path,
                float(amplitude),
                structure_rows,
                confidence_rows,
                request_rows,
            )
        for role in ("alphafold", "esmfold"):
            try:
                prediction_path = _fetch_prediction(
                    role,
                    str(domain.uniprot_id),
                    str(domain.sequence),
                    structure_root,
                    config,
                )
                predicted_sequence, predicted, confidence = _coordinates(prediction_path)
                start = _unique_subsequence(predicted_sequence, str(domain.sequence))
                stop = start + len(domain.sequence)
                predicted = predicted[start:stop]
                confidence = confidence[start:stop]
                if config.panel.require_complete_backbone and not np.isfinite(predicted).all():
                    raise ValueError("predicted crop lacks complete backbone")
                path = input_root / f"{_safe(domain.domain_id)}__{role}.npz"
                np.savez_compressed(path, coordinates=predicted.astype(np.float32))
                rmsd = kabsch_rmsd(experimental[:, 1], predicted[:, 1])
                _append_structure(
                    domain,
                    role,
                    predicted,
                    confidence,
                    path,
                    rmsd,
                    structure_rows,
                    confidence_rows,
                    request_rows,
                    source_path=prediction_path,
                )
            except (OSError, RuntimeError, ValueError) as error:
                exclusions.append(
                    {
                        "domain_id": domain.domain_id,
                        "structure_role": role,
                        "reason": str(error),
                    }
                )
    structures = pd.DataFrame(structure_rows).sort_values(
        ["structure_role", "domain_id"], ignore_index=True
    )
    confidence = pd.DataFrame(confidence_rows).sort_values(
        ["structure_role", "domain_id", "position"], ignore_index=True
    )
    requests = pd.DataFrame(request_rows).sort_values(
        ["structure_role", "domain_id"], ignore_index=True
    )
    role_counts = structures.groupby("structure_role")["domain_id"].nunique()
    predictor_summary_eligible = {
        role: int(role_counts.get(role, 0)) >= config.panel.minimum_matched_domains_per_predictor
        for role in ("alphafold", "esmfold")
    }
    request_structures = structures[
        ["structure_role", "structure_id", "domain_id", "input_path", "sha256"]
    ].rename(columns={"domain_id": "target_domain_id"})
    request_structures["input_kind"] = "coordinates"
    request_structures["analysis_population"] = POPULATION
    exclusion_table = pd.DataFrame(exclusions, columns=["domain_id", "structure_role", "reason"])
    tables = {
        "structures": structures,
        "confidence": confidence,
        "requests": requests,
        "request_structures": request_structures,
        "exclusions": exclusion_table,
    }
    for name, table in tables.items():
        write_parquet(paths[name], table)
    lock = {
        **runtime_manifest(config.paths.project_root),
        "schema_version": config.schema_version,
        "status": config.status,
        "outcomes_were_open_before_protocol": True,
        "alternate_structure_scores_existed_before_protocol": False,
        "confirmatory": False,
        "routing_allowed": False,
        "changes_primary_decision": False,
        "roles": {str(key): int(value) for key, value in role_counts.items()},
        "predictor_summary_eligible": predictor_summary_eligible,
    }
    write_json(paths["lock"], lock)
    write_json(
        paths["manifest"],
        {
            **lock,
            "cross_platform_evaluation": str(cross_manifest),
            "tables": [table_manifest(paths[name], table) for name, table in tables.items()],
        },
    )
    return paths


def smooth_perturbation(
    coordinates: np.ndarray, target_rmsd: float, window: int, seed: int
) -> np.ndarray:
    """Apply a smooth residue translation with an exact CA displacement RMS."""

    values = np.asarray(coordinates, dtype=np.float64)
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(len(values), 3))
    kernel = np.ones(window, dtype=float) / window
    smooth = np.stack(
        [np.convolve(noise[:, axis], kernel, mode="same") for axis in range(3)], axis=1
    )
    smooth -= smooth.mean(axis=0, keepdims=True)
    current = float(np.sqrt(np.mean(np.sum(smooth**2, axis=1))))
    if current == 0:
        raise ValueError("degenerate perturbation")
    displacement = smooth * (float(target_rmsd) / current)
    return (values + displacement[:, None, :]).astype(np.float32)


def kabsch_rmsd(reference: np.ndarray, mobile: np.ndarray) -> float:
    left = np.asarray(reference, dtype=float) - np.asarray(reference, dtype=float).mean(axis=0)
    right = np.asarray(mobile, dtype=float) - np.asarray(mobile, dtype=float).mean(axis=0)
    covariance = right.T @ left
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    aligned = right @ rotation
    return float(np.sqrt(np.mean(np.sum((left - aligned) ** 2, axis=1))))


def _append_structure(
    domain,
    role: str,
    coordinates: np.ndarray,
    confidence: np.ndarray,
    input_path: Path,
    ca_rmsd: float,
    structure_rows: list[dict],
    confidence_rows: list[dict],
    request_rows: list[dict],
    *,
    source_path: Path | None = None,
) -> None:
    if coordinates.shape != (len(domain.sequence), 4, 3):
        raise ValueError(f"coordinate shape mismatch for {domain.domain_id}/{role}")
    structure_id = f"{domain.domain_id}|{role}"
    finite_confidence = confidence[np.isfinite(confidence)]
    structure_rows.append(
        {
            "domain_id": domain.domain_id,
            "uniprot_id": domain.uniprot_id,
            "structure_role": role,
            "structure_id": structure_id,
            "input_path": str(input_path.resolve()),
            "sha256": sha256_file(input_path),
            "source_path": str((source_path or input_path).resolve()),
            "length": len(domain.sequence),
            "ca_rmsd_to_experimental": float(ca_rmsd),
            "mean_confidence": (
                float(finite_confidence.mean()) if len(finite_confidence) else float("nan")
            ),
            "minimum_confidence": (
                float(finite_confidence.min()) if len(finite_confidence) else float("nan")
            ),
            "analysis_population": POPULATION,
        }
    )
    confidence_rows.extend(
        {
            "domain_id": domain.domain_id,
            "structure_role": role,
            "position": position,
            "confidence": float(value),
        }
        for position, value in enumerate(confidence)
    )
    request_rows.append(
        {
            "request_id": f"{domain.domain_id}|{role}|{structure_id}",
            "state_id": domain.domain_id,
            "domain_id": domain.domain_id,
            "state_sequence": domain.sequence,
            "structure_role": role,
            "structure_id": structure_id,
            "input_kind": "coordinates",
            "input_path": str(input_path.resolve()),
            "length": len(domain.sequence),
        }
    )


def _fetch_prediction(
    role: str,
    accession: str,
    sequence: str,
    root: Path,
    config: StructureSensitivityConfig,
) -> Path:
    directory = root / role
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{accession}.pdb"
    if path.exists() and path.stat().st_size:
        return path
    service_audit = directory / "service_audit.json"
    if role == "esmfold" and service_audit.exists():
        audit = json.loads(service_audit.read_text(encoding="utf-8"))
        if accession in set(audit.get("unavailable_accessions", [])):
            raise OSError(f"ESMFold service audit marked {accession} unavailable after retries")
    if role == "alphafold":
        api = config.panel.alphafold_api.format(accession=accession)
        metadata = json.loads(_request(api).decode())
        entries = [
            entry
            for entry in metadata
            if str(entry.get("uniprotAccession")) == accession
            and str(entry.get("entryId", "")).startswith(f"AF-{accession}-F1")
        ]
        if not entries:
            raise OSError(f"AlphaFold DB has no canonical model for {accession}")
        payload = _request(str(entries[0]["pdbUrl"]))
    elif role == "esmfold":
        payload = _request(config.panel.esmfold_api, data=sequence.encode())
    else:
        raise ValueError(f"unsupported predicted structure role: {role}")
    if b"ATOM  " not in payload[:20000]:
        raise OSError(f"{role} response for {accession} is not a PDB file")
    temporary = path.with_suffix(".pdb.part")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return path


def _request(url: str, data: bytes | None = None) -> bytes:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(
                url,
                data=data,
                method="POST" if data is not None else "GET",
                headers={"User-Agent": "MARGIN/structure-sensitivity"},
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.read()
        except (OSError, TimeoutError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(1 + attempt)
    raise OSError(f"request failed for {url}: {last_error}")


def _coordinates(path: Path) -> tuple[str, np.ndarray, np.ndarray]:
    model = next(PDBParser(QUIET=True).get_structure(path.stem, str(path)).get_models())
    chains = list(model.get_chains())
    chain = next((value for value in chains if value.id == "A"), chains[0] if chains else None)
    if chain is None:
        raise ValueError(f"predicted structure has no chain: {path}")
    sequence = []
    coordinates = []
    confidence = []
    for residue in chain.get_residues():
        if residue.id[0].strip():
            continue
        letter = protein_letters_3to1.get(residue.resname.upper())
        if letter is None:
            continue
        sequence.append(letter)
        if all(atom in residue for atom in BACKBONE_ATOMS):
            coordinates.append(
                np.stack([residue[atom].coord.astype(float) for atom in BACKBONE_ATOMS])
            )
        else:
            coordinates.append(np.full((4, 3), np.nan))
        confidence.append(float(residue["CA"].bfactor) if "CA" in residue else np.nan)
    if not sequence:
        raise ValueError(f"predicted structure contains no canonical residues: {path}")
    confidence_values = np.asarray(confidence)
    finite = confidence_values[np.isfinite(confidence_values)]
    if len(finite) and float(finite.max()) <= 1.5:
        confidence_values = confidence_values * 100.0
    return "".join(sequence), np.asarray(coordinates), confidence_values


def _unique_subsequence(full: str, query: str) -> int:
    first = full.find(query)
    if first < 0:
        raise ValueError("target chain is not an exact predicted-sequence substring")
    if full.find(query, first + 1) >= 0:
        raise ValueError("target chain occurs more than once in predicted sequence")
    return first


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value)


def _resolve(base: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (base / value).resolve()
