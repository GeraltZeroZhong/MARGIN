"""Validated configuration for structure-conditioned stability study."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class StabilityPaths(StrictModel):
    project_root: Path = Path("..")
    run_dir: Path = Path("runs/stability")
    storage_dir: Path = Path("data/workspaces/stability")
    foundation_config: Path = Path("configs/observability_replication.yaml")
    generalization_config: Path = Path("configs/generalization.yaml")
    observability_replication_run: Path = Path("runs/observability/replication")
    generalization_run: Path = Path("runs/generalization")
    counterfactual_run: Path = Path("runs/counterfactuals")
    mechanism_run: Path = Path("runs/mechanisms")
    action_validation_run: Path = Path("runs/action_validation")
    megascale_archive: Path = Path("data/external/tsuboyama/7844779/Processed_K50_dG_datasets.zip")
    megascale_structures: Path = Path("data/external/tsuboyama/7992926/AlphaFold_model_PDBs")
    protein_gym_metadata: Path = Path("data/external/proteingym_DMS_substitutions.csv")
    protein_gym_substitutions: Path = Path("data/external/DMS_ProteinGym_substitutions_v1.3.zip")
    protein_gym_structures: Path = Path("data/external/proteingym/v1.3/structures")
    protein_gym_msa_archive: Path = Path("data/external/DMS_msa_files_v1.3.zip")
    mmseqs_executable: Path = Path("tools/mmseqs/bin/mmseqs")
    sequence_models_repository: Path = Path("external/repositories/protein-sequence-models")
    mif_checkpoint: Path = Path("models/protein_sequence_models/mif.pt")
    proteinmpnn_repository: Path = Path("external/repositories/ProteinMPNN")
    proteinmpnn_checkpoint: Path = Path(
        "external/repositories/ProteinMPNN/vanilla_model_weights/v_48_020.pt"
    )
    esm_if1_repository: Path = Path("external/repositories/fair-esm")
    esm_if1_checkpoint: Path = Path("models/esm_if1_gvp4_t16_142M_UR50.pt")
    esm1b_hf_checkpoint: Path = Path("models/esm1b_t33_650M_UR50S_hf")
    cath_queries: Path = Path("runs/generalization/architecture/query_rows.parquet")
    cath_teacher_scores: Path = Path("runs/observability/replication/teacher_cache/scores.parquet")
    cath_mif_scores: Path = Path("runs/generalization/mif/scores.parquet")
    cath_carp640_store: Path = Path("data/workspaces/generalization/architecture/carp_640M")
    cath_esm2_150_store: Path = Path("data/workspaces/mechanisms/training_controls/esm2_150M_cath")
    cath_esm2_650_store: Path = Path("data/workspaces/generalization/architecture/esm2_650M")
    cath_esm1b_650_store: Path = Path("data/workspaces/generalization/architecture/esm1b_650M")
    cath_conservation_alignments: Path = Path(
        "runs/observability/replication/conservation/alignments.tsv"
    )
    cath_fasta: Path = Path("data/external/cath/v4_4_0/cath-domain-seqs-v4_4_0.fa")


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
    external_assay_id: Literal["ESTA_BACSU_Nutschel_2020"] = "ESTA_BACSU_Nutschel_2020"
    external_structure_file: Literal["ESTA_BACSU.pdb"] = "ESTA_BACSU.pdb"
    external_minimum_variants: PositiveInt = 2000
    external_minimum_positions: PositiveInt = 150
    near_duplicate_identity: float = Field(0.80, ge=0, le=1)
    near_duplicate_minimum_coverage: float = Field(0.90, ge=0, le=1)
    homology_evalue: float = Field(1e-3, gt=0)
    homology_sensitivity: float = Field(7.5, gt=0)
    homology_threads: PositiveInt = 8
    require_complete_backbone: bool = True

    @model_validator(mode="after")
    def validate_panel(self) -> PanelConfig:
        if len(self.de_novo_families) != len(set(self.de_novo_families)):
            raise ValueError("de novo families must be unique")
        return self


class CalibrationConfig(StrictModel):
    teacher_ids: list[Literal["mif", "esm_if1", "proteinmpnn"]] = [
        "mif",
        "esm_if1",
        "proteinmpnn",
    ]
    sequence_model_id: Literal["esm2_150M"] = "esm2_150M"
    schemes: list[
        Literal[
            "unscaled_equal",
            "action_rms_matched",
            "joint_temperature_native_nll",
            "rowwise_rank_normalized",
        ]
    ] = [
        "unscaled_equal",
        "action_rms_matched",
        "joint_temperature_native_nll",
        "rowwise_rank_normalized",
    ]
    selection_metric: Literal["native_nll"] = "native_nll"
    training_split: Literal["development_train"] = "development_train"
    selection_split: Literal["development_validation"] = "development_validation"
    final_training_splits: list[Literal["development_train", "development_validation"]] = [
        "development_train",
        "development_validation",
    ]
    temperature_minimum: float = Field(0.25, gt=0)
    temperature_maximum: float = Field(4.0, gt=0)
    rank_scale_minimum: float = Field(0.25, gt=0)
    rank_scale_maximum: float = Field(4.0, gt=0)
    proteinmpnn_order_repeats: PositiveInt = 8
    mif_batch_size: PositiveInt = 4

    @model_validator(mode="after")
    def validate_calibration(self) -> CalibrationConfig:
        if len(self.teacher_ids) != 3 or len(set(self.teacher_ids)) != 3:
            raise ValueError("exactly three unique paired teachers are required")
        if len(self.schemes) != 4 or len(set(self.schemes)) != 4:
            raise ValueError("all four unique registered calibration schemes are required")
        if self.temperature_minimum >= self.temperature_maximum:
            raise ValueError("temperature bounds are reversed")
        if self.rank_scale_minimum >= self.rank_scale_maximum:
            raise ValueError("rank-scale bounds are reversed")
        return self


class StrongControlConfig(StrictModel):
    enabled: bool = True
    representation_models: list[Literal["carp_640M", "esm2_650M", "esm1b_650M"]] = [
        "carp_640M",
        "esm2_650M",
        "esm1b_650M",
    ]
    representation_pca_components: PositiveInt = 48
    local_context_radius: PositiveInt = 8
    profile_pseudocount: float = Field(0.5, gt=0)
    profile_minimum_identity: float = Field(0.20, ge=0, le=1)
    profile_maximum_identity: float = Field(0.80, ge=0, le=1)
    profile_minimum_query_coverage: float = Field(0.70, ge=0, le=1)
    ridge_alphas: list[float] = [1.0, 10.0, 100.0]
    mlp_hidden_sizes: list[PositiveInt] = [64, 128]
    mlp_alphas: list[float] = [0.001, 0.01]
    model_selection_metric: Literal["anchored_action_rmse"] = "anchored_action_rmse"

    @model_validator(mode="after")
    def validate_strong_control(self) -> StrongControlConfig:
        if self.profile_minimum_identity >= self.profile_maximum_identity:
            raise ValueError("profile identity bounds are reversed")
        if self.ridge_alphas != sorted(set(self.ridge_alphas)):
            raise ValueError("ridge alphas must be sorted and unique")
        return self


class InferenceConfig(StrictModel):
    confidence_level: float = Field(0.95, gt=0, lt=1)
    bootstrap_replicates: PositiveInt = 5000
    external_position_bootstrap_replicates: PositiveInt = 5000
    top_fraction: float = Field(0.10, gt=0, le=1)
    minimum_teacher_replications: PositiveInt = 2
    require_primary_spearman_ci_lower_positive: bool = True
    require_primary_ndcg_ci_lower_positive: bool = True
    require_both_megascale_strata_point_positive: bool = True
    require_external_spearman_ci_lower_positive: bool = True
    require_external_ndcg_point_positive: bool = True
    require_strong_control_spearman_ci_lower_positive: bool = True


class StabilityStudyConfig(StrictModel):
    schema_version: Literal["stability.v1"] = "stability.v1"
    seed: int = 20260819
    paths: StabilityPaths
    panel: PanelConfig = PanelConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    strong_control: StrongControlConfig = StrongControlConfig()
    inference: InferenceConfig = InferenceConfig()


def load_stability_config(path: Path) -> StabilityStudyConfig:
    """Load stability study YAML and resolve filesystem paths relative to the project."""

    path = path.resolve()
    with path.open(encoding="utf-8") as handle:
        config = StabilityStudyConfig.model_validate(yaml.safe_load(handle))
    config.paths.project_root = _resolve(path.parent, config.paths.project_root)
    for field_name in StabilityPaths.model_fields:
        if field_name == "project_root":
            continue
        value = getattr(config.paths, field_name)
        setattr(config.paths, field_name, _resolve(config.paths.project_root, value))
    return config


def _resolve(base: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (base / value).resolve()
