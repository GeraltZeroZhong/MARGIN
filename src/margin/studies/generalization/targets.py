"""Teacher-lineage targets for the native generalization study CATH audit."""

from __future__ import annotations

import numpy as np
import pandas as pd

from margin.attribution.metrics import normalize_log_probabilities
from margin.studies.generalization.config import GeneralizationStudyConfig
from margin.studies.observability.targets import ResidualDataset, clr
from margin.teachers.schema import logp_columns

LINEAGE_TARGETS = (
    "mif",
    "mifst",
    "esm_if1",
    "proteinmpnn_mc8",
    "consensus_if1_mpnn",
    "consensus_all_teachers",
    "consensus_leave_mifst_out",
)


def load_generalization_residual_dataset(config: GeneralizationStudyConfig) -> ResidualDataset:
    """Align native query rows with sequence and structure-teacher distributions."""

    root = config.paths.observability_replication_run
    queries = pd.read_parquet(config.paths.run_dir / "architecture" / "query_rows.parquet")
    domains = pd.read_parquet(root / "registry" / "domains.parquet")
    metadata = queries.merge(
        domains[["domain_id", "cath_h", "cath_t"]],
        on="domain_id",
        validate="many_to_one",
    ).sort_values(["state_id", "domain_id", "position"], ignore_index=True)
    metadata["analysis_role"] = metadata["observability_split"]
    metadata["eligible_for_training"] = metadata["observability_split"].ne("locked_test")
    metadata["state_kind"] = "native_reference"
    metadata["requested_corruption_ratio"] = 0.0
    keys = ["state_id", "domain_id", "position"]
    scores = pd.read_parquet(root / "teacher_cache" / "scores.parquet")
    sequence = _aligned(scores, metadata, "sequence_student", "sequence_only", keys)
    teacher_logp: dict[str, np.ndarray] = {}
    residuals: dict[str, np.ndarray] = {}
    mapping = {
        "mifst": "mifst",
        "esm_if1": "esm_if1",
        "proteinmpnn_mc8": "proteinmpnn",
    }
    for target_id, teacher_id in mapping.items():
        values = _aligned(scores, metadata, teacher_id, "paired", keys)
        teacher_logp[target_id] = values
        residuals[target_id] = clr(values - sequence)

    mif_scores = pd.read_parquet(config.paths.run_dir / "mif" / "scores.parquet")
    paired_mif = _aligned(mif_scores, metadata, "mif", "paired", keys)
    rewired_mif = _aligned(mif_scores, metadata, "mif", "contact_rewired", keys)
    teacher_logp["mif"] = paired_mif
    residuals["mif"] = clr(paired_mif - sequence)
    residuals["mif_paired_minus_rewired"] = clr(paired_mif - rewired_mif)
    teacher_logp["mif_paired_minus_rewired"] = normalize_log_probabilities(
        sequence + residuals["mif_paired_minus_rewired"]
    )

    _add_consensus(
        teacher_logp,
        residuals,
        sequence,
        "consensus_if1_mpnn",
        ["esm_if1", "proteinmpnn_mc8"],
    )
    _add_consensus(
        teacher_logp,
        residuals,
        sequence,
        "consensus_all_teachers",
        ["mif", "mifst", "esm_if1", "proteinmpnn_mc8"],
    )
    _add_consensus(
        teacher_logp,
        residuals,
        sequence,
        "consensus_leave_mifst_out",
        ["mif", "esm_if1", "proteinmpnn_mc8"],
    )
    return ResidualDataset(
        metadata=metadata,
        sequence_logp=sequence,
        teacher_logp=teacher_logp,
        residuals=residuals,
        temperatures=pd.DataFrame(
            columns=[
                "teacher_id",
                "temperature",
                "native_nll_before",
                "native_nll_after",
                "calibration_rows",
            ]
        ),
    )


def _aligned(
    scores: pd.DataFrame,
    metadata: pd.DataFrame,
    teacher_id: str,
    structure_role: str,
    keys: list[str],
) -> np.ndarray:
    selected = scores.loc[
        scores["teacher_id"].eq(teacher_id) & scores["structure_role"].eq(structure_role),
        [*keys, *logp_columns()],
    ]
    if selected.duplicated(keys).any():
        selected = selected.groupby(keys, observed=True)[logp_columns()].mean().reset_index()
    aligned = metadata[keys].merge(selected, on=keys, validate="one_to_one")
    if len(aligned) != len(metadata):
        raise ValueError(
            f"{teacher_id}/{structure_role} lacks generalization study query coverage: "
            f"{len(aligned)}/{len(metadata)}"
        )
    return normalize_log_probabilities(aligned[logp_columns()].to_numpy(dtype=float))


def _add_consensus(
    teacher_logp: dict[str, np.ndarray],
    residuals: dict[str, np.ndarray],
    sequence: np.ndarray,
    target_id: str,
    members: list[str],
) -> None:
    residual = np.mean([residuals[member] for member in members], axis=0)
    values = normalize_log_probabilities(sequence + residual)
    teacher_logp[target_id] = values
    residuals[target_id] = clr(values - sequence)
