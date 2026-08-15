"""Attach precomputed per-residue conservation to the canonical registry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from margin.config import ProjectConfig
from margin.data_registry.registry import RegistryTables


def load_conservation(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"conservation input does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
    raise ValueError("conservation input must be Parquet, CSV, or TSV")


def attach_conservation(
    registry: RegistryTables,
    conservation: pd.DataFrame,
    config: ProjectConfig,
) -> RegistryTables:
    """Require one bounded conservation score for every candidate residue."""

    required = {"domain_id", "position", "conservation_score"}
    missing = required - set(conservation.columns)
    if missing:
        raise ValueError(f"conservation table is missing columns: {sorted(missing)}")
    keys = ["domain_id", "position"]
    if conservation.duplicated(keys).any():
        raise ValueError("conservation domain/position keys must be unique")
    values = conservation["conservation_score"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("conservation_score must be finite and in [0, 1]")
    source = conservation[[*keys, "conservation_score"]].copy()
    residues = registry.residues.drop(
        columns=["conservation_score", "conservation_class"], errors="ignore"
    ).merge(source, on=keys, how="left", validate="one_to_one")
    if residues["conservation_score"].isna().any():
        missing_count = int(residues["conservation_score"].isna().sum())
        raise ValueError(f"conservation input lacks {missing_count} canonical residue rows")
    residues["conservation_class"] = np.select(
        [
            residues["conservation_score"] >= config.registry.conserved_score_min,
            residues["conservation_score"] <= config.registry.variable_score_max,
        ],
        ["conserved", "variable"],
        default="intermediate",
    )
    return RegistryTables(registry.domains.copy(), residues)
