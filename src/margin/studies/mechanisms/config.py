"""Validated configuration for the frozen mechanism study mechanism audit."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class MechanismPaths(StrictModel):
    project_root: Path = Path("..")
    run_dir: Path = Path("runs/mechanisms")
    storage_dir: Path = Path("data/workspaces/mechanisms")
    foundation_config: Path = Path("configs/foundation.yaml")
    generalization_config: Path = Path("configs/generalization.yaml")
    generalization_run: Path = Path("runs/generalization")
    counterfactual_run: Path = Path("runs/counterfactuals")
    observability_replication_run: Path = Path("runs/observability/replication")
    megascale_archive: Path = Path("data/external/tsuboyama/7844779/Processed_K50_dG_datasets.zip")
    megascale_structures: Path = Path("data/external/tsuboyama/7992926/AlphaFold_model_PDBs")
    aaindex1: Path = Path("data/external/aaindex/aaindex1")
    mmseqs_executable: Path = Path("tools/mmseqs/bin/mmseqs")
    sequence_models_repository: Path = Path("external/repositories/protein-sequence-models")
    mif_checkpoint: Path = Path("models/protein_sequence_models/mif.pt")


class PanelConfig(StrictModel):
    megascale_member: str = "Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv"
    minimum_single_variants: PositiveInt = 500
    minimum_unique_positions: PositiveInt = 30
    natural_domains: PositiveInt = 16
    de_novo_domains_per_family: PositiveInt = 2
    de_novo_families: list[str] = [
        "EA",
        "GG",
        "XX",
        "EHEE",
        "EEHEE",
        "HEEH",
        "HHH",
        "trRosetta_hallucination",
    ]
    near_duplicate_identity: float = Field(0.80, ge=0, le=1)
    near_duplicate_minimum_coverage: float = Field(0.90, ge=0, le=1)
    homology_evalue: float = Field(1e-3, gt=0)
    homology_sensitivity: float = Field(7.5, gt=0)
    homology_threads: PositiveInt = 8
    require_complete_backbone: bool = True
    matched_real_decoys: PositiveInt = 3
    matched_real_maximum_source_length_excess: PositiveInt = 20

    @model_validator(mode="after")
    def validate_panel(self) -> PanelConfig:
        if len(self.de_novo_families) != len(set(self.de_novo_families)):
            raise ValueError("de novo families must be unique")
        return self


class CounterfactualConfig(StrictModel):
    seeds: list[int] = [0, 1, 2, 3, 4]
    contact_deletion_fractions: list[float] = [0.05, 0.10, 0.20]
    coordinate_rmsd_angstrom: list[float] = [0.25, 0.50, 1.00]
    coordinate_low_frequency_modes: PositiveInt = 3
    maximum_adjacent_ca_distance_change_angstrom: float = Field(0.25, gt=0)
    constrained_reassignment_fraction: float = Field(0.10, gt=0, lt=1)
    constrained_max_attempts_per_swap: PositiveInt = 500
    legacy_rewiring_swaps_per_edge: float = Field(5.0, gt=0)
    legacy_rewire_max_attempts_per_swap: PositiveInt = 25
    contact_distance_angstrom: float = Field(8.0, gt=0)
    contact_minimum_sequence_separation: PositiveInt = 3

    @model_validator(mode="after")
    def validate_counterfactuals(self) -> CounterfactualConfig:
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("counterfactual seeds must be non-empty and unique")
        if self.contact_deletion_fractions != sorted(set(self.contact_deletion_fractions)):
            raise ValueError("contact deletion fractions must be sorted and unique")
        if self.coordinate_rmsd_angstrom != sorted(set(self.coordinate_rmsd_angstrom)):
            raise ValueError("coordinate RMSD levels must be sorted and unique")
        return self


class ModelConfig(StrictModel):
    sequence_model_id: Literal["esm2_150M"] = "esm2_150M"
    predictor_model_id: Literal["carp_640M"] = "carp_640M"
    rrr_rank: PositiveInt = 16
    ridge_alpha: float = Field(10.0, ge=0)
    context_ridge_alpha: float = Field(10.0, ge=0)
    direct_pca_ranks: list[PositiveInt] = [1, 3, 5, 16]
    mif_batch_size: PositiveInt = 2


class InferenceConfig(StrictModel):
    confidence_level: float = Field(0.95, gt=0, lt=1)
    bootstrap_replicates: PositiveInt = 5000
    top_fraction: float = Field(0.10, gt=0, le=1)
    relevance_transform: Literal["domain_shifted_nonnegative"] = "domain_shifted_nonnegative"
    id_jsd_max_nats: float = Field(0.10, gt=0)
    id_absolute_entropy_shift_max_nats: float = Field(0.25, gt=0)
    id_minimum_domain_fraction: float = Field(0.80, gt=0.5, le=1)
    seed_reliability_minimum_spearman: float = Field(0.50, ge=-1, le=1)
    robust_minimum_counterfactual_families: PositiveInt = 2
    require_margin_ci_lower_positive: bool = True


class MechanismStudyConfig(StrictModel):
    schema_version: Literal["mechanisms.v1"] = "mechanisms.v1"
    seed: int = 20260817
    paths: MechanismPaths
    panel: PanelConfig = PanelConfig()
    counterfactuals: CounterfactualConfig = CounterfactualConfig()
    models: ModelConfig = ModelConfig()
    inference: InferenceConfig = InferenceConfig()


def load_mechanism_config(path: Path) -> MechanismStudyConfig:
    """Load mechanism study YAML and resolve filesystem paths relative to the project."""

    path = path.resolve()
    with path.open(encoding="utf-8") as handle:
        config = MechanismStudyConfig.model_validate(yaml.safe_load(handle))
    config.paths.project_root = _resolve(path.parent, config.paths.project_root)
    for field_name in MechanismPaths.model_fields:
        if field_name == "project_root":
            continue
        value = getattr(config.paths, field_name)
        setattr(config.paths, field_name, _resolve(config.paths.project_root, value))
    return config


def _resolve(base: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (base / value).resolve()
