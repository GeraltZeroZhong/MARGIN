"""Validated configuration for the independently locked counterfactual study study."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class CounterfactualPaths(StrictModel):
    project_root: Path = Path("..")
    run_dir: Path = Path("runs/counterfactuals")
    storage_dir: Path = Path("data/workspaces/counterfactuals")
    foundation_config: Path = Path("configs/foundation.yaml")
    generalization_config: Path = Path("configs/generalization.yaml")
    generalization_run: Path = Path("runs/generalization")
    observability_replication_run: Path = Path("runs/observability/replication")
    current_assays: Path = Path("runs/generalization/dms/assays.parquet")
    cath_domain_list: Path = Path("data/external/cath/v4_4_0/cath-domain-list-v4_4_0.txt")
    cath_fasta: Path = Path("data/external/cath/v4_4_0/cath-domain-seqs-v4_4_0.fa")
    s669_root: Path = Path("external/repositories/proteinmpnn_ddg/paper/datasets/s669")
    s669_repository: Path = Path("external/repositories/proteinmpnn_ddg")
    megascale_archive: Path = Path("data/external/tsuboyama/7844779/Processed_K50_dG_datasets.zip")
    megascale_structures: Path = Path("data/external/tsuboyama/7992926/AlphaFold_model_PDBs")
    tsuboyama_reference_structures: Path = Path(
        "external/repositories/proteinmpnn_ddg/paper/datasets/tsuboyama/pdb"
    )
    current52_supplemental_structures: Path = Path(
        "data/external/counterfactuals/current52_structures"
    )
    mmseqs_executable: Path = Path("tools/mmseqs/bin/mmseqs")
    foldseek_executable: Path = Path("tools/foldseek/bin/foldseek")
    sequence_models_repository: Path = Path("external/repositories/protein-sequence-models")
    mif_checkpoint: Path = Path("models/protein_sequence_models/mif.pt")


class PanelConfig(StrictModel):
    s669_minimum_variants_per_domain: PositiveInt = 3
    megascale_member: str = "Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv"
    megascale_minimum_single_variants: PositiveInt = 500
    megascale_domains_per_family: PositiveInt = 2
    megascale_design_families: list[str] = [
        "EA",
        "GG",
        "XX",
        "EHEE",
        "EEHEE",
        "HEEH",
        "HHH",
        "trRosetta_hallucination",
    ]
    identity_threshold: float = Field(0.30, ge=0, le=1)
    minimum_bidirectional_coverage: float = Field(0.80, ge=0, le=1)
    homology_evalue: float = Field(1e-3, gt=0)
    homology_sensitivity: float = Field(7.5, gt=0)
    homology_threads: PositiveInt = 8
    structural_tmscore_threshold: float = Field(0.50, ge=0, le=1)
    structural_minimum_bidirectional_coverage: float = Field(0.80, ge=0, le=1)
    foldseek_threads: PositiveInt = 8
    require_s669_cath_assignment: bool = True
    require_complete_backbone: bool = True

    @model_validator(mode="after")
    def validate_families(self) -> PanelConfig:
        if len(self.megascale_design_families) != len(set(self.megascale_design_families)):
            raise ValueError("Megascale design families must be unique")
        return self


class CounterfactualConfig(StrictModel):
    primary_role: Literal["contact_rewired_5"] = "contact_rewired_5"
    replication_role: Literal["circular_permuted"] = "circular_permuted"
    rewiring_swaps_per_edge: list[float] = [0.5, 1.0, 2.0, 5.0]
    rewire_max_attempts_per_swap: PositiveInt = 25
    contact_distance_angstrom: float = Field(8.0, gt=0)
    contact_minimum_sequence_separation: PositiveInt = 3
    circular_minimum_displacement_fraction: float = Field(0.50, gt=0, le=0.50)
    matched_random_repeats: PositiveInt = 20

    @model_validator(mode="after")
    def validate_rewiring(self) -> CounterfactualConfig:
        if self.rewiring_swaps_per_edge != sorted(set(self.rewiring_swaps_per_edge)):
            raise ValueError("rewiring strengths must be sorted and unique")
        if 5.0 not in self.rewiring_swaps_per_edge:
            raise ValueError("the frozen primary 5-swaps-per-edge condition is required")
        return self


class ModelConfig(StrictModel):
    sequence_model_id: Literal["esm2_150M"] = "esm2_150M"
    predictor_model_id: Literal["carp_640M"] = "carp_640M"
    rrr_rank: PositiveInt = 16
    ridge_alpha: float = Field(10.0, ge=0)
    residual_alpha: float = 1.0
    mif_batch_size: PositiveInt = 2


class InferenceConfig(StrictModel):
    confidence_level: float = Field(0.95, gt=0, lt=1)
    bootstrap_replicates: PositiveInt = 5000
    top_fraction: float = Field(0.10, gt=0, le=1)
    minimum_positive_domain_fraction: float = Field(0.51, gt=0.50, le=1)
    require_spearman_ci_lower_positive: bool = True
    require_ndcg_ci_lower_positive: bool = True
    require_topk_point_positive: bool = True
    require_replication_counterfactual_point_positive: bool = True
    require_random_control_margin_ci_lower_positive: bool = True
    require_both_strata_point_positive: bool = True


class CounterfactualStudyConfig(StrictModel):
    schema_version: Literal["counterfactuals.v1"] = "counterfactuals.v1"
    seed: int = 20260816
    paths: CounterfactualPaths
    panel: PanelConfig = PanelConfig()
    counterfactuals: CounterfactualConfig = CounterfactualConfig()
    models: ModelConfig = ModelConfig()
    inference: InferenceConfig = InferenceConfig()


def load_counterfactual_config(path: Path) -> CounterfactualStudyConfig:
    """Load counterfactual study YAML and resolve all filesystem paths."""

    path = path.resolve()
    with path.open(encoding="utf-8") as handle:
        config = CounterfactualStudyConfig.model_validate(yaml.safe_load(handle))
    config.paths.project_root = _resolve(path.parent, config.paths.project_root)
    for field_name in CounterfactualPaths.model_fields:
        if field_name == "project_root":
            continue
        value = getattr(config.paths, field_name)
        setattr(config.paths, field_name, _resolve(config.paths.project_root, value))
    return config


def _resolve(base: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (base / value).resolve()
