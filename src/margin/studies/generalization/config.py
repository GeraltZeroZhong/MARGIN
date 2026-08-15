"""Validated configuration for the generalization study audit."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class GeneralizationPaths(StrictModel):
    project_root: Path = Path("..")
    run_dir: Path = Path("runs/generalization")
    storage_dir: Path = Path("data/workspaces/generalization")
    observability_config: Path = Path("configs/observability.yaml")
    observability_replication_config: Path = Path("configs/observability_replication.yaml")
    observability_run: Path = Path("runs/observability")
    observability_replication_run: Path = Path("runs/observability/replication")
    observability_carp_representations: Path = Path("runs/observability/cache/replication_carp")
    observability_esm2_representations: Path = Path("runs/observability/cache/replication_layers")
    protein_gym_metadata: Path = Path("data/external/proteingym_DMS_substitutions.csv")
    protein_gym_substitutions: Path = Path("data/external/DMS_ProteinGym_substitutions_v1.3.zip")
    mmseqs_executable: Path = Path("tools/mmseqs/bin/mmseqs")
    sequence_models_repository: Path = Path("external/repositories/protein-sequence-models")
    carp_76m_checkpoint: Path = Path("models/protein_sequence_models/carp_76M.pt")
    carp_640m_checkpoint: Path = Path("runs/model_cache/carp_640M.pt")
    mif_checkpoint: Path = Path("models/protein_sequence_models/mif.pt")
    esm2_150m_model: Path = Path("models/esm2_t30_150M_UR50D")
    esm2_650m_checkpoint: Path = Path("models/esm2_t33_650M_UR50D.pt")
    esm1b_650m_checkpoint: Path = Path("models/esm1b_t33_650M_UR50S.pt")


class ArchitectureModel(StrictModel):
    model_id: str
    family: Literal["CARP", "ESM2", "ESM1b"]
    scale_millions: PositiveInt
    loader: Literal["carp", "hf_esm", "fair_esm"]
    checkpoint_path_key: str
    layer: int = Field(ge=0)
    batch_size: PositiveInt
    reuse_observability_store: Literal["carp", "esm2"] | None = None


class ArchitectureConfig(StrictModel):
    positions_per_domain: PositiveInt = 64
    state_kind: Literal["native_reference"] = "native_reference"
    feature_kind: Literal["strict_loo_final_query"] = "strict_loo_final_query"
    primary_target: Literal["consensus_leave_mifst_out"] = "consensus_leave_mifst_out"
    rrr_rank: PositiveInt = 16
    ridge_alpha: float = Field(10.0, ge=0)
    control_repeats: PositiveInt = 5
    controls: list[
        Literal[
            "global",
            "within_domain",
            "within_wild_type",
            "within_environment",
            "within_corruption",
            "fully_conditioned",
        ]
    ]
    models: list[ArchitectureModel]

    @model_validator(mode="after")
    def validate_models(self) -> ArchitectureConfig:
        identifiers = [model.model_id for model in self.models]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("architecture model identifiers must be unique")
        required = {"carp_76M", "carp_640M", "esm2_150M", "esm2_650M", "esm1b_650M"}
        if set(identifiers) != required:
            raise ValueError(f"architecture matrix must contain exactly {sorted(required)}")
        return self


class DMSConfig(StrictModel):
    source_pattern: str = "_Tsuboyama_2023_"
    minimum_single_variants: PositiveInt = 100
    identity_threshold: float = Field(0.30, ge=0, le=1)
    minimum_bidirectional_coverage: float = Field(0.80, ge=0, le=1)
    homology_evalue: float = Field(1e-3, gt=0)
    homology_sensitivity: float = Field(7.5, gt=0)
    homology_threads: PositiveInt = 8
    primary_alpha: float = 1.0
    sensitivity_alphas: list[float] = [0.25, 0.5, 2.0]
    top_fraction: float = Field(0.10, gt=0, le=1)
    minimum_mean_spearman_increment: float = Field(0.02, ge=-2, le=2)
    minimum_positive_assay_fraction: float = Field(0.60, ge=0, le=1)
    require_positive_cluster_ci_lower: bool = True
    require_positive_mean_ndcg_increment: bool = True


class InferenceConfig(StrictModel):
    confidence_level: float = Field(0.95, gt=0, lt=1)
    bootstrap_replicates: PositiveInt = 2000
    minimum_jsd_reduction_nats: float = Field(0.01, ge=0)
    minimum_residual_cosine: float = Field(0.10, ge=-1, le=1)
    require_positive_ci_lower: bool = True
    require_positive_control_margin_ci_lower: bool = True


class EnvironmentConfig(StrictModel):
    gate_training_split: Literal["development_train"] = "development_train"
    gate_selection_split: Literal["development_validation"] = "development_validation"
    evaluation_split: Literal["observability_locked_test_reused_postdecision"] = (
        "observability_locked_test_reused_postdecision"
    )
    minimum_rows_per_route: PositiveInt = 30


class GeneralizationStudyConfig(StrictModel):
    schema_version: Literal["generalization.v1"] = "generalization.v1"
    seed: int = 20260815
    paths: GeneralizationPaths
    architecture: ArchitectureConfig
    dms: DMSConfig = DMSConfig()
    inference: InferenceConfig = InferenceConfig()
    environments: EnvironmentConfig = EnvironmentConfig()


def load_generalization_config(path: Path) -> GeneralizationStudyConfig:
    """Load generalization study YAML and resolve every path relative to the project root."""

    path = path.resolve()
    with path.open(encoding="utf-8") as handle:
        config = GeneralizationStudyConfig.model_validate(yaml.safe_load(handle))
    config.paths.project_root = _resolve(path.parent, config.paths.project_root)
    for field_name in GeneralizationPaths.model_fields:
        if field_name == "project_root":
            continue
        value = getattr(config.paths, field_name)
        setattr(config.paths, field_name, _resolve(config.paths.project_root, value))
    return config


def _resolve(base: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (base / value).resolve()
