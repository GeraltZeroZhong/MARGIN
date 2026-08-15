"""Validated configuration for the locked structure-unique action study."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ActionValidationPaths(StrictModel):
    project_root: Path = Path("..")
    run_dir: Path = Path("runs/action_validation")
    storage_dir: Path = Path("data/workspaces/action_validation")
    foundation_config: Path = Path("configs/observability_replication.yaml")
    generalization_config: Path = Path("configs/generalization.yaml")
    observability_replication_run: Path = Path("runs/observability/replication")
    generalization_run: Path = Path("runs/generalization")
    counterfactual_run: Path = Path("runs/counterfactuals")
    mechanism_run: Path = Path("runs/mechanisms")
    megascale_archive: Path = Path("data/external/tsuboyama/7844779/Processed_K50_dG_datasets.zip")
    megascale_structures: Path = Path("data/external/tsuboyama/7992926/AlphaFold_model_PDBs")
    s669_root: Path = Path("external/repositories/proteinmpnn_ddg/paper/datasets/s669")
    mmseqs_executable: Path = Path("tools/mmseqs/bin/mmseqs")
    sequence_models_repository: Path = Path("external/repositories/protein-sequence-models")
    mif_checkpoint: Path = Path("models/protein_sequence_models/mif.pt")
    proteinmpnn_repository: Path = Path("external/repositories/ProteinMPNN")
    proteinmpnn_checkpoint: Path = Path(
        "external/repositories/ProteinMPNN/vanilla_model_weights/v_48_020.pt"
    )
    esm_if1_repository: Path = Path("external/repositories/fair-esm")
    esm_if1_checkpoint: Path = Path("models/esm_if1_gvp4_t16_142M_UR50.pt")
    cath_queries: Path = Path("runs/generalization/architecture/query_rows.parquet")
    cath_teacher_scores: Path = Path("runs/observability/replication/teacher_cache/scores.parquet")
    cath_mif_scores: Path = Path("runs/generalization/mif/scores.parquet")
    cath_carp_store: Path = Path("data/workspaces/generalization/architecture/carp_640M")
    cath_esm2_store: Path = Path("data/workspaces/mechanisms/training_controls/esm2_150M_cath")


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
    s669_minimum_variants: PositiveInt = 5
    s669_maximum_domains: PositiveInt = 8
    s669_minimum_selected_domains: PositiveInt = 4
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
        if self.s669_minimum_selected_domains > self.s669_maximum_domains:
            raise ValueError("S669 minimum cannot exceed its maximum")
        return self


class DecompositionConfig(StrictModel):
    teacher_ids: list[Literal["mif", "esm_if1", "proteinmpnn"]] = [
        "mif",
        "esm_if1",
        "proteinmpnn",
    ]
    context_model_id: Literal["carp_640M"] = "carp_640M"
    sequence_model_id: Literal["esm2_150M"] = "esm2_150M"
    training_split: Literal["development_train"] = "development_train"
    selection_split: Literal["development_validation"] = "development_validation"
    final_training_splits: list[Literal["development_train", "development_validation"]] = [
        "development_train",
        "development_validation",
    ]
    context_radius: PositiveInt = 4
    rrr_ranks: list[PositiveInt] = [4, 8, 16]
    ridge_alphas: list[float] = [1.0, 10.0, 100.0]
    model_selection_metric: Literal["anchored_action_rmse"] = "anchored_action_rmse"
    variance_match_on_final_training: bool = True
    proteinmpnn_order_repeats: PositiveInt = 8
    mif_batch_size: PositiveInt = 4
    shuffled_u_repeats: PositiveInt = 20
    consensus_weighting: Literal["equal_after_action_rms_matching"] = (
        "equal_after_action_rms_matching"
    )

    @model_validator(mode="after")
    def validate_decomposition(self) -> DecompositionConfig:
        if len(self.teacher_ids) != 3 or len(set(self.teacher_ids)) != 3:
            raise ValueError("exactly three unique registered teachers are required")
        if self.rrr_ranks != sorted(set(self.rrr_ranks)):
            raise ValueError("RRR ranks must be sorted and unique")
        if self.ridge_alphas != sorted(set(self.ridge_alphas)):
            raise ValueError("Ridge alphas must be sorted and unique")
        if any(alpha < 0 for alpha in self.ridge_alphas):
            raise ValueError("Ridge alphas must be nonnegative")
        return self


class InferenceConfig(StrictModel):
    confidence_level: float = Field(0.95, gt=0, lt=1)
    bootstrap_replicates: PositiveInt = 5000
    top_fraction: float = Field(0.10, gt=0, le=1)
    minimum_positive_domain_fraction: float = Field(0.51, gt=0.50, le=1)
    minimum_teacher_replications: PositiveInt = 2
    minimum_s669_finite_domains: PositiveInt = 4
    minimum_s669_positive_domain_fraction: float = Field(0.50, ge=0.50, le=1)
    agreement_gate_minimum_position_spearman: float = Field(0.30, ge=-1, le=1)
    require_consensus_spearman_ci_lower_positive: bool = True
    require_consensus_ndcg_ci_lower_positive: bool = True
    require_two_teacher_spearman_ci_lower_positive: bool = True
    require_both_megascale_strata_point_positive: bool = True
    require_s669_spearman_point_positive: bool = True
    require_shuffled_u_margin_ci_lower_positive: bool = True


class ActionValidationStudyConfig(StrictModel):
    schema_version: Literal["action_validation.v1"] = "action_validation.v1"
    seed: int = 20260818
    paths: ActionValidationPaths
    panel: PanelConfig = PanelConfig()
    decomposition: DecompositionConfig = DecompositionConfig()
    inference: InferenceConfig = InferenceConfig()


def load_action_validation_config(path: Path) -> ActionValidationStudyConfig:
    """Load action-validation study YAML and resolve filesystem paths relative to the project."""

    path = path.resolve()
    with path.open(encoding="utf-8") as handle:
        config = ActionValidationStudyConfig.model_validate(yaml.safe_load(handle))
    config.paths.project_root = _resolve(path.parent, config.paths.project_root)
    for field_name in ActionValidationPaths.model_fields:
        if field_name == "project_root":
            continue
        value = getattr(config.paths, field_name)
        setattr(config.paths, field_name, _resolve(config.paths.project_root, value))
    return config


def _resolve(base: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (base / value).resolve()
