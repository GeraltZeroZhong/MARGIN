"""Validated configuration for the observability study."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ObservabilityPaths(StrictModel):
    project_root: Path = Path("..")
    foundation_run: Path = Path("runs/foundation")
    run_dir: Path = Path("runs/observability")
    storage_dir: Path = Path("data/workspaces/observability")
    cath_domain_list: Path
    cath_fasta: Path
    cath_s40_list: Path
    cath_structures_dir: Path
    benchmark_registry: Path
    benchmark_sequences: Path
    foundation_config: Path = Path("configs/foundation.yaml")
    replication_config: Path = Path("configs/observability_replication.yaml")
    esm2_model: Path = Path("models/esm2_t30_150M_UR50D")
    carp_model: Path = Path("runs/model_cache/carp_640M.pt")
    carp_repository: Path = Path("external/repositories/protein-sequence-models")


class ReplicationConfig(StrictModel):
    total_domains: PositiveInt = 240
    train_domains: PositiveInt = 144
    validation_domains: PositiveInt = 48
    locked_test_domains: PositiveInt = 48
    candidate_pool_size: PositiveInt = 600
    minimum_length: PositiveInt = 40
    maximum_length: PositiveInt = 300
    maximum_resolution_angstrom: float = Field(3.0, gt=0)
    maximum_missing_fraction: float = Field(0.05, ge=0, lt=1)
    exclude_foundation_cath_h: bool = True
    exclude_foundation_cath_t: bool = True
    one_domain_per_cath_t: bool = True
    benchmark_identity_threshold: float = Field(0.30, ge=0, le=1)
    homology_evalue: float = Field(1e-3, gt=0)
    homology_sensitivity: float = Field(7.5, gt=0)
    homology_threads: PositiveInt = 8
    state_kinds: list[Literal["random_substitution", "core_targeted"]] = [
        "random_substitution",
        "core_targeted",
    ]
    corruption_levels: list[float] = [0.05, 0.15, 0.30]

    @model_validator(mode="after")
    def validate_counts(self) -> ReplicationConfig:
        split_total = self.train_domains + self.validation_domains + self.locked_test_domains
        if split_total != self.total_domains:
            raise ValueError("replication split sizes must sum to total_domains")
        if self.candidate_pool_size < self.total_domains:
            raise ValueError("candidate_pool_size must be at least total_domains")
        if self.minimum_length > self.maximum_length:
            raise ValueError("minimum_length must not exceed maximum_length")
        if self.corruption_levels != sorted(set(self.corruption_levels)):
            raise ValueError("corruption_levels must be sorted and unique")
        if any(level <= 0 or level >= 1 for level in self.corruption_levels):
            raise ValueError("corruption_levels must lie strictly between zero and one")
        return self


class ResidualTargetConfig(StrictModel):
    primary: str = "mifst"
    teacher_specific: list[str] = ["mifst", "esm_if1", "proteinmpnn"]
    consensus_members: list[str] = ["mifst", "esm_if1"]
    calibrate_on_state_kind: str = "native_reference"
    calibration_temperature_bounds: tuple[float, float] = (0.25, 4.0)


class ProbeConfig(StrictModel):
    ridge_alpha: float = Field(10.0, ge=0)
    reduced_ranks: list[PositiveInt] = [2, 4, 8, 16]
    mlp_hidden_units: PositiveInt = 64
    mlp_max_iterations: PositiveInt = 100
    local_window_radius: PositiveInt = 4
    top_k: PositiveInt = 3
    selected_layers: list[int] = list(range(31))
    control_repeats: PositiveInt = 5
    lora_rank: PositiveInt = 8
    lora_alpha: float = Field(16.0, gt=0)
    lora_target_layers: list[int] = [26, 27, 28, 29]
    lora_train_positions_per_state: PositiveInt = 16
    lora_validation_positions_per_state: PositiveInt = 8
    lora_epochs: PositiveInt = 3
    lora_batch_size: PositiveInt = 4
    lora_learning_rate: float = Field(5e-4, gt=0)
    alternate_model_id: str = "carp_640M"
    alternate_model_layer: int = 56
    alternate_model_batch_size: PositiveInt = 4
    shuffle_controls: list[
        Literal[
            "global",
            "within_domain",
            "within_wild_type",
            "within_environment",
            "within_corruption",
            "fully_conditioned",
        ]
    ] = [
        "global",
        "within_domain",
        "within_wild_type",
        "within_environment",
        "within_corruption",
        "fully_conditioned",
    ]


class InferenceConfig(StrictModel):
    confidence_level: float = Field(0.95, gt=0, lt=1)
    bootstrap_replicates: PositiveInt = 2000
    minimum_environment_rows: PositiveInt = 30
    minimum_jsd_reduction_nats: float = Field(0.01, ge=0)
    minimum_residual_cosine: float = Field(0.10, ge=-1, le=1)
    require_positive_ci_lower_bound: bool = True


class CandidateEnvironment(StrictModel):
    environment_id: str
    state_kind: Literal["random_substitution", "core_targeted"]
    requested_corruption_ratio: float = Field(gt=0, lt=1)
    axis: Literal["burial", "secondary_structure", "contact_class", "conservation_class"]
    value: str


class ObservabilityStudyConfig(StrictModel):
    schema_version: Literal["observability.v1"] = "observability.v1"
    seed: int = 20260814
    paths: ObservabilityPaths
    replication: ReplicationConfig = ReplicationConfig()
    residual_targets: ResidualTargetConfig = ResidualTargetConfig()
    probes: ProbeConfig = ProbeConfig()
    inference: InferenceConfig = InferenceConfig()
    candidate_environments: list[CandidateEnvironment] = []


def load_observability_config(path: Path) -> ObservabilityStudyConfig:
    """Load and resolve a fixed observability study YAML configuration."""

    path = path.resolve()
    with path.open(encoding="utf-8") as handle:
        config = ObservabilityStudyConfig.model_validate(yaml.safe_load(handle))
    config.paths.project_root = _resolve(path.parent, config.paths.project_root)
    for field_name in (
        "foundation_run",
        "run_dir",
        "storage_dir",
        "cath_domain_list",
        "cath_fasta",
        "cath_s40_list",
        "cath_structures_dir",
        "benchmark_registry",
        "benchmark_sequences",
        "foundation_config",
        "replication_config",
        "esm2_model",
        "carp_model",
        "carp_repository",
    ):
        value = getattr(config.paths, field_name)
        setattr(config.paths, field_name, _resolve(config.paths.project_root, value))
    return config


def _resolve(base: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (base / value).resolve()
