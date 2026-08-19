"""Validated configuration for every foundation audit algorithm and decision threshold."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator, model_validator

from margin.constants import AA_ALPHABET, SCHEMA_VERSION


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PathsConfig(StrictModel):
    project_root: Path = Path("..")
    run_dir: Path = Path("runs/foundation")
    domain_input: Path | None = None
    audit_domain_input: Path | None = None
    cath_domain_list: Path | None = None
    cath_fasta: Path | None = None
    structures_dir: Path | None = None
    benchmark_input: Path | None = None
    homology_hits_input: Path | None = None
    conservation_input: Path | None = None
    dms_input: Path | None = None
    embeddings_input: Path | None = None

    @property
    def registry_dir(self) -> Path:
        return self.run_dir / "registry"

    @property
    def state_bank_dir(self) -> Path:
        return self.run_dir / "state_bank"

    @property
    def decoy_dir(self) -> Path:
        return self.run_dir / "decoys"

    @property
    def teacher_cache_dir(self) -> Path:
        return self.run_dir / "teacher_cache"

    @property
    def audit_dir(self) -> Path:
        return self.run_dir / "audit"

    @property
    def report_dir(self) -> Path:
        return self.run_dir / "reports"

    @property
    def source_data_dir(self) -> Path:
        return self.run_dir / "source_data"

    @property
    def figure_dir(self) -> Path:
        return self.run_dir / "figures"


class RegistryConfig(StrictModel):
    source_name: str = "CATH"
    source_version: str = "4.4.0"
    min_length: PositiveInt = 40
    max_length: PositiveInt = 300
    max_resolution_angstrom: float = Field(3.0, gt=0)
    max_missing_fraction: float = Field(0.05, ge=0, lt=1)
    contact_distance_angstrom: float = Field(8.0, gt=0)
    contact_minimum_sequence_separation: PositiveInt = 3
    buried_rsa_max: float = Field(0.20, ge=0, le=1)
    exposed_rsa_min: float = Field(0.50, ge=0, le=1)
    high_contact_degree_min: PositiveInt = 8
    conserved_score_min: float = Field(0.80, ge=0, le=1)
    variable_score_max: float = Field(0.30, ge=0, le=1)
    allowed_amino_acids: str = AA_ALPHABET
    benchmark_identity_threshold: float = Field(0.30, ge=0, le=1)
    exclude_exact_benchmark_ids: bool = True
    exclude_cath_h: bool = True
    exclude_cath_t_for_topology_ood: bool = True
    require_experimental_structure: bool = True
    require_dssp: bool = True
    dssp_executable: str = "mkdssp"

    @model_validator(mode="after")
    def validate_ranges(self) -> RegistryConfig:
        if self.min_length > self.max_length:
            raise ValueError("registry.min_length must not exceed max_length")
        if self.buried_rsa_max >= self.exposed_rsa_min:
            raise ValueError("buried_rsa_max must be lower than exposed_rsa_min")
        if self.variable_score_max >= self.conserved_score_min:
            raise ValueError("variable_score_max must be lower than conserved_score_min")
        if set(self.allowed_amino_acids) != set(AA_ALPHABET):
            raise ValueError("allowed_amino_acids must contain the canonical 20 residues")
        return self


StateKind = Literal[
    "random_mask",
    "random_substitution",
    "blosum_substitution",
    "model_aware_offline",
    "on_policy_rollout",
    "span_mask",
    "core_targeted",
    "surface_targeted",
]


class StateBankConfig(StrictModel):
    kinds: list[StateKind] = [
        "random_mask",
        "random_substitution",
        "blosum_substitution",
        "model_aware_offline",
        "on_policy_rollout",
        "span_mask",
        "core_targeted",
        "surface_targeted",
    ]
    corruption_levels: list[float] = [0.05, 0.15, 0.30]
    samples_per_domain_kind_level: PositiveInt = 1
    mask_token: str = "X"
    blosum_temperature: float = Field(1.0, gt=0)
    span_mean_length: float = Field(4.0, gt=0)
    rollout_steps: PositiveInt = 4
    rollout_temperature: float = Field(1.0, gt=0)
    model_aware_temperature: float = Field(1.0, gt=0)
    policy_context_radius: PositiveInt = 4
    scaffold_edit_decay: float = Field(3.0, gt=0)
    scaffold_core_penalty: float = Field(2.0, ge=0)
    minimum_positions: PositiveInt = 1
    targeted_operation: Literal["mask", "substitution"] = "mask"

    @field_validator("corruption_levels")
    @classmethod
    def validate_corruption_levels(cls, values: list[float]) -> list[float]:
        if not values or any(value <= 0 or value >= 1 for value in values):
            raise ValueError("corruption_levels must be non-empty and strictly between 0 and 1")
        if values != sorted(set(values)):
            raise ValueError("corruption_levels must be sorted and unique")
        return values


class StudentPolicyConfig(StrictModel):
    adapter: Literal["synthetic", "python_factory", "imported_native"]
    policy_id: str
    model_revision: str
    factory: str | None = None
    scores_input: Path | None = None
    model_path: Path | None = None
    weights_sha256: str | None = None
    batch_size: PositiveInt = 16
    device: str = "auto"

    @model_validator(mode="after")
    def validate_adapter_inputs(self) -> StudentPolicyConfig:
        if self.adapter == "python_factory" and not self.factory:
            raise ValueError("student_policy.factory is required for python_factory")
        if self.adapter == "imported_native" and self.scores_input is None:
            raise ValueError("student_policy.scores_input is required for imported_native")
        return self


class HomologyConfig(StrictModel):
    executable: str = "mmseqs"
    sensitivity: float = Field(7.5, gt=0)
    evalue: float = Field(1e-3, gt=0)
    threads: PositiveInt = 8


class DecoyConfig(StrictModel):
    matched_decoys_per_domain: PositiveInt = 1
    require_exact_length: bool = True
    maximum_length_difference: int = Field(0, ge=0)
    maximum_helix_fraction_difference: float = Field(0.10, ge=0, le=1)
    maximum_strand_fraction_difference: float = Field(0.10, ge=0, le=1)
    require_different_cath_h: bool = True
    contact_rewire_swaps_per_edge: float = Field(5.0, gt=0)
    contact_rewire_max_attempts_per_swap: PositiveInt = 25
    permutation_minimum_displacement_fraction: float = Field(0.50, ge=0, le=1)


class TeacherSpec(StrictModel):
    teacher_id: str
    adapter: Literal[
        "canonical_parquet",
        "mifst",
        "proteinmpnn",
        "esm_if1_candidates",
        "synthetic",
    ]
    role: Literal["sequence", "primary_structure", "audit_structure"]
    model_name: str
    model_revision: str
    conda_env: str | None = None
    repository: Path | None = None
    repository_revision: str | None = None
    weights: Path | None = None
    auxiliary_weights: Path | None = None
    enabled: bool = True
    score_type: Literal["log_probability", "candidate_score"] = "log_probability"
    batch_size: PositiveInt = 1
    order_repeats: PositiveInt = 1


class TeacherCacheConfig(StrictModel):
    alphabet: str = AA_ALPHABET
    score_clip: float = Field(20.0, gt=0)
    normalization_temperature: float = Field(1.0, gt=0)
    shard_rows: PositiveInt = 100_000
    teachers: list[TeacherSpec]

    @model_validator(mode="after")
    def validate_teachers(self) -> TeacherCacheConfig:
        identifiers = [teacher.teacher_id for teacher in self.teachers]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("teacher_id values must be unique")
        if self.alphabet != AA_ALPHABET:
            raise ValueError(f"teacher cache alphabet must be {AA_ALPHABET}")
        return self


class ObservabilityConfig(StrictModel):
    group_levels: list[Literal["cath_h", "cath_t"]] = ["cath_h", "cath_t"]
    folds: PositiveInt = 5
    ridge_alpha: float = Field(10.0, ge=0)
    context_radius: PositiveInt = 4
    top_k: PositiveInt = 3
    shuffled_control_repeats: PositiveInt = 5
    minimum_groups_per_split: PositiveInt = 3
    minimum_rows_per_environment: PositiveInt = 30

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, value: int) -> int:
        if value > len(AA_ALPHABET):
            raise ValueError("top_k cannot exceed the amino-acid alphabet size")
        return value


class OnPolicyConfig(StrictModel):
    reference_offline_kinds: list[StateKind] = [
        "random_mask",
        "random_substitution",
        "span_mask",
    ]
    model_aware_kind: StateKind = "model_aware_offline"
    on_policy_kind: StateKind = "on_policy_rollout"
    match_columns: list[str] = [
        "corruption_ratio",
        "edit_distance_fraction",
        "mask_count",
        "student_entropy",
        "student_top1_margin",
    ]
    match_caliper: float = Field(1.0, gt=0)
    maximum_standardized_mean_difference: float = Field(0.10, ge=0)
    high_confidence_margin: float = Field(0.50, ge=0)
    minimum_match_fraction: float = Field(0.70, ge=0, le=1)


class AuditSettings(StrictModel):
    bootstrap_replicates: PositiveInt = 500
    confidence_level: float = Field(0.95, gt=0, lt=1)
    cluster_column: str = "domain_id"
    decision_analysis_role: Literal["external_benchmark", "all"] = "external_benchmark"
    dms_minimum_variants_per_assay: PositiveInt = 20
    primary_teacher_id: str = "mifst"
    sequence_teacher_id: str = "sequence_student"
    paired_role: str = "paired"
    decoy_roles: list[str] = [
        "matched_cath",
        "permuted",
        "contact_rewired",
        "shuffled_residue",
    ]


class DecisionConfig(StrictModel):
    allow_real_decision: bool = True
    minimum_environment_advantage_nats: float = Field(0.02, ge=0)
    minimum_paired_decoy_lift_nats: float = Field(0.02, ge=0)
    minimum_dms_spearman: float = Field(0.05, ge=-1, le=1)
    minimum_observability_jsd_reduction: float = Field(0.01, ge=0)
    minimum_observability_cosine: float = Field(0.10, ge=-1, le=1)
    minimum_teacher_action_valid_radius: float = Field(0.15, ge=0, le=1)
    minimum_directionally_consistent_structure_teachers: PositiveInt = 2
    minimum_on_policy_advantage_nats: float = Field(0.01, ge=0)
    require_positive_ci_lower_bound: bool = True


class PlotConfig(StrictModel):
    journal: Literal["nature", "generic"] = "nature"
    width_inches: float = Field(7.2, gt=0)
    height_inches: float = Field(3.8, gt=0)
    dpi: PositiveInt = 300
    formats: list[Literal["png", "pdf", "svg"]] = ["png", "pdf", "svg"]
    minimum_marker_area: float = Field(30.0, gt=0)
    maximum_marker_area: float = Field(220.0, gt=0)


class ProjectConfig(StrictModel):
    schema_version: str = SCHEMA_VERSION
    project_name: str = "MARGIN"
    data_mode: Literal["real", "synthetic"] = "real"
    seed: int = 20260814
    paths: PathsConfig
    registry: RegistryConfig = RegistryConfig()
    state_bank: StateBankConfig = StateBankConfig()
    student_policy: StudentPolicyConfig
    homology: HomologyConfig = HomologyConfig()
    decoys: DecoyConfig = DecoyConfig()
    teacher_cache: TeacherCacheConfig
    observability: ObservabilityConfig = ObservabilityConfig()
    on_policy: OnPolicyConfig = OnPolicyConfig()
    audit: AuditSettings = AuditSettings()
    decision: DecisionConfig = DecisionConfig()
    plot: PlotConfig = PlotConfig()

    @model_validator(mode="after")
    def validate_mode(self) -> ProjectConfig:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        if self.data_mode == "synthetic" and self.decision.allow_real_decision:
            raise ValueError("synthetic configurations must set decision.allow_real_decision=false")
        if self.data_mode == "synthetic" and self.student_policy.adapter != "synthetic":
            raise ValueError("synthetic data mode requires the synthetic student policy")
        if (
            "on_policy_rollout" in self.state_bank.kinds
            and self.student_policy.adapter == "imported_native"
        ):
            raise ValueError(
                "imported_native policy scores are not state-conditioned and cannot generate "
                "on_policy_rollout states"
            )
        return self


def load_config(path: Path) -> ProjectConfig:
    """Load YAML, validate it, and resolve all filesystem locations deterministically."""

    path = path.resolve()
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    config = ProjectConfig.model_validate(payload)
    config.paths.project_root = _resolve(path.parent, config.paths.project_root)
    config.paths.run_dir = _resolve(config.paths.project_root, config.paths.run_dir)
    for field_name in (
        "domain_input",
        "audit_domain_input",
        "cath_domain_list",
        "cath_fasta",
        "structures_dir",
        "benchmark_input",
        "homology_hits_input",
        "conservation_input",
        "dms_input",
        "embeddings_input",
    ):
        value = getattr(config.paths, field_name)
        if value is not None:
            setattr(config.paths, field_name, _resolve(config.paths.project_root, value))
    for teacher in config.teacher_cache.teachers:
        if teacher.repository is not None:
            teacher.repository = _resolve(config.paths.project_root, teacher.repository)
        if teacher.weights is not None:
            teacher.weights = _resolve(config.paths.project_root, teacher.weights)
        if teacher.auxiliary_weights is not None:
            teacher.auxiliary_weights = _resolve(
                config.paths.project_root, teacher.auxiliary_weights
            )
    if config.student_policy.scores_input is not None:
        config.student_policy.scores_input = _resolve(
            config.paths.project_root, config.student_policy.scores_input
        )
    if config.student_policy.model_path is not None:
        config.student_policy.model_path = _resolve(
            config.paths.project_root, config.student_policy.model_path
        )
    return config


def _resolve(base: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (base / value).resolve()
