"""Position-specificity control for the retained action under the final C+ model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from margin.constants import AA_TO_INDEX
from margin.provenance import read_json, runtime_manifest, write_csv, write_json, write_parquet
from margin.studies.action_validation.evaluation import METRICS, _anchor, _domain_metrics
from margin.studies.counterfactuals.evaluation import stratified_domain_bootstrap
from margin.studies.stability.config import StabilityStudyConfig

DEFAULT_POPULATION = "megascale_stability_dense"
ACTUAL_METHOD = "sequence_plus_final_g_cplus_uplus"
SHUFFLED_METHOD = "sequence_plus_final_g_cplus_position_shuffled_uplus"


def evaluate_position_specificity(
    queries: pd.DataFrame,
    variants: pd.DataFrame,
    component_matrices: Path,
    *,
    population: str = DEFAULT_POPULATION,
    shuffle_repeats: int = 20,
    bootstrap_replicates: int = 5000,
    confidence_level: float = 0.95,
    top_fraction: float = 0.10,
    seed: int = 20260819,
) -> dict[str, pd.DataFrame]:
    """Compare correctly positioned U+ with within-domain position permutations."""

    if shuffle_repeats < 1 or bootstrap_replicates < 1:
        raise ValueError("resampling counts must be positive")
    queries = queries.reset_index(drop=True).copy()
    if queries.duplicated(["domain_id", "position"]).any():
        raise ValueError("query positions must be unique within domain")
    selected = variants.loc[variants["evaluation_population"].eq(population)].copy()
    selected = selected.reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"evaluation population is empty: {population}")

    matrices = _load_component_matrices(component_matrices, len(queries))
    lookup = queries[["domain_id", "position"]].copy()
    lookup["query_row"] = np.arange(len(lookup), dtype=int)
    if "query_row" in selected:
        selected = selected.drop(columns="query_row")
    selected = selected.merge(
        lookup,
        on=["domain_id", "position"],
        how="left",
        validate="many_to_one",
    )
    if selected["query_row"].isna().any():
        raise ValueError("variant table contains positions absent from the query table")

    query_rows = selected["query_row"].to_numpy(dtype=int)
    mutant = selected["mutant"].map(AA_TO_INDEX)
    if mutant.isna().any():
        raise ValueError("variant table contains a non-canonical mutant residue")
    mutant_indices = mutant.to_numpy(dtype=int)
    sequence = selected["sequence_action"].to_numpy(dtype=float)
    g = matrices["consensus_g"]
    c_plus = matrices["consensus_c_plus"]
    u_plus = matrices["consensus_u_plus"]
    action = matrices["consensus_a"]
    _verify_component_identity(action, g, c_plus, u_plus)

    actual = sequence + g[query_rows, mutant_indices]
    actual += c_plus[query_rows, mutant_indices] + u_plus[query_rows, mutant_indices]
    if "joint_temperature_native_nll_action" in selected:
        recorded = selected["joint_temperature_native_nll_action"].to_numpy(dtype=float)
        if not np.allclose(actual - sequence, recorded, rtol=1e-5, atol=2e-6):
            raise ValueError("component matrices and recorded paired actions are misaligned")

    actual_metrics = _domain_metrics(selected, {ACTUAL_METHOD: actual}, top_fraction)
    domain_indices = {
        str(domain_id): np.asarray(indices, dtype=int)
        for domain_id, indices in queries.groupby("domain_id", sort=True).indices.items()
        if domain_id in set(selected["domain_id"])
    }
    wild = queries["wild_type"].map(AA_TO_INDEX)
    if wild.isna().any():
        raise ValueError("query table contains a non-canonical wild-type residue")
    wild_indices = wild.to_numpy(dtype=int)

    repeat_frames = []
    for repeat in range(shuffle_repeats):
        rng = np.random.default_rng(seed + 700_000 + repeat)
        shuffled = u_plus.copy()
        for indices in domain_indices.values():
            shuffled[indices] = u_plus[rng.permutation(indices)]
        shuffled = _anchor(shuffled, wild_indices)
        prediction = sequence + g[query_rows, mutant_indices]
        prediction += c_plus[query_rows, mutant_indices]
        prediction += shuffled[query_rows, mutant_indices]
        metrics = _domain_metrics(selected, {SHUFFLED_METHOD: prediction}, top_fraction)
        metrics["repeat"] = repeat
        repeat_frames.append(metrics)

    repeated = pd.concat(repeat_frames, ignore_index=True)
    keys = ["domain_id", "evaluation_population", "stratum", "n_variants"]
    shuffled_mean = repeated.groupby(keys, observed=True)[list(METRICS)].mean().reset_index()
    actual_domain = actual_metrics[[*keys, *METRICS]]
    domain = actual_domain.merge(
        shuffled_mean,
        on=keys,
        suffixes=("_actual", "_shuffled"),
        validate="one_to_one",
    )
    for metric in METRICS:
        domain[f"{metric}_margin"] = (
            domain[f"{metric}_actual"] - domain[f"{metric}_shuffled"]
        )
    domain["shuffle_repeats"] = shuffle_repeats

    summary_rows: list[dict[str, Any]] = []
    scopes = [("all", domain)]
    scopes.extend(
        (str(stratum), frame)
        for stratum, frame in domain.groupby("stratum", sort=True, observed=True)
    )
    for scope_index, (scope, frame) in enumerate(scopes):
        for metric_index, metric in enumerate(METRICS):
            column = f"{metric}_margin"
            estimate = stratified_domain_bootstrap(
                frame,
                column,
                replicates=bootstrap_replicates,
                confidence_level=confidence_level,
                seed=seed + 910_000 + scope_index * 100 + metric_index,
            )
            summary_rows.append(
                {
                    "evaluation_population": population,
                    "scope": scope,
                    "metric": column,
                    "actual_method": ACTUAL_METHOD,
                    "control_method": SHUFFLED_METHOD,
                    "shuffle_repeats": shuffle_repeats,
                    **estimate,
                }
            )
    return {
        "position_shuffle_repeats": repeated,
        "position_shuffle_domains": domain,
        "position_shuffle_summary": pd.DataFrame(summary_rows),
    }


def run_position_specificity_audit(
    config: StabilityStudyConfig,
    *,
    run_dir: Path | None = None,
    component_matrices: Path | None = None,
    output_dir: Path | None = None,
    population: str = DEFAULT_POPULATION,
) -> dict[str, Path]:
    """Read final C+ artifacts, run the control and write reusable result tables."""

    run_dir = config.paths.run_dir if run_dir is None else run_dir.resolve()
    if component_matrices is None:
        manifest = read_json(run_dir / "strong_control/strong_control_manifest.json")
        component_matrices = Path(str(manifest["component_matrices"]))
    output_dir = run_dir / "position_specificity" if output_dir is None else output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = evaluate_position_specificity(
        pd.read_parquet(run_dir / "panel/query_rows.parquet"),
        pd.read_parquet(run_dir / "evaluation/variant_components.parquet"),
        component_matrices,
        population=population,
        shuffle_repeats=config.inference.position_shuffle_repeats,
        bootstrap_replicates=config.inference.bootstrap_replicates,
        confidence_level=config.inference.confidence_level,
        top_fraction=config.inference.top_fraction,
        seed=config.seed,
    )
    paths: dict[str, Path] = {}
    for name, table in tables.items():
        parquet = output_dir / f"{name}.parquet"
        csv = output_dir / f"{name}.csv"
        write_parquet(parquet, table)
        write_csv(csv, table)
        paths[name] = parquet
        paths[f"{name}_csv"] = csv
    manifest_path = output_dir / "manifest.json"
    write_json(
        manifest_path,
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": "stability.position_specificity.v1",
            "analysis_role": "final_c_plus_position_specificity_control",
            "evaluation_population": population,
            "component_matrices": str(component_matrices),
            "shuffle_repeats": config.inference.position_shuffle_repeats,
            "bootstrap_replicates": config.inference.bootstrap_replicates,
            "tables": {name: str(path) for name, path in paths.items()},
        },
    )
    paths["manifest"] = manifest_path
    return paths


def _load_component_matrices(path: Path, query_rows: int) -> dict[str, np.ndarray]:
    required = ("consensus_a", "consensus_g", "consensus_c_plus", "consensus_u_plus")
    with np.load(path) as archive:
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError(f"component matrix archive lacks: {', '.join(missing)}")
        matrices = {name: np.asarray(archive[name], dtype=float) for name in required}
    expected_shape = (query_rows, len(AA_TO_INDEX))
    if any(matrix.shape != expected_shape for matrix in matrices.values()):
        raise ValueError(f"component matrices must all have shape {expected_shape}")
    return matrices


def _verify_component_identity(
    action: np.ndarray,
    g: np.ndarray,
    c_plus: np.ndarray,
    u_plus: np.ndarray,
) -> None:
    if not np.allclose(action, g + c_plus + u_plus, rtol=1e-5, atol=2e-6):
        raise ValueError("final component matrices violate A = G + C+ + U+")
