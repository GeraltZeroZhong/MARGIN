"""Post-lock explanatory audits for matched structure-sensitivity study structures.

These analyses are descriptive.  Outcomes were already open when the audit
specification was recorded, so none of the tables produced here can alter a
registered gate or authorize routing.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from margin.attribution.metrics import normalize_log_probabilities
from margin.constants import AA_TO_INDEX
from margin.provenance import read_json
from margin.studies.action_validation.evaluation import _anchor, _ndcg, _spearman
from margin.studies.counterfactuals.evaluation import stratified_domain_bootstrap
from margin.studies.external_validation.panel import load_external_validation_config
from margin.studies.stability.config import load_stability_config
from margin.studies.structure_sensitivity.panel import load_structure_sensitivity_config
from margin.teachers.schema import logp_columns

TEACHERS = ("mif", "esm_if1", "proteinmpnn")
CONSENSUS = "registered_temperature_consensus"


def build_structure_audit_tables(
    project_root: Path, specification: Mapping[str, Any]
) -> dict[str, pd.DataFrame]:
    """Build per-teacher, geometry, and teacher-distribution diagnostics."""

    structure_sensitivity = load_structure_sensitivity_config(
        project_root / specification["paths"]["structure_sensitivity_protocol"]
    )
    cross = load_external_validation_config(
        project_root / specification["paths"]["external_validation_protocol"]
    )
    stability = load_stability_config(project_root / specification["paths"]["stability_config"])
    queries = pd.read_parquet(cross.paths.run_dir / "panel/query_rows.parquet")
    variants = pd.read_parquet(cross.paths.run_dir / "evaluation/variants.parquet")
    structures = pd.read_parquet(structure_sensitivity.paths.run_dir / "panel/structures.parquet")
    confidence = pd.read_parquet(
        structure_sensitivity.paths.run_dir / "panel/residue_confidence.parquet"
    )
    requests = pd.read_parquet(
        structure_sensitivity.paths.run_dir / "teacher_requests/requests.parquet"
    )
    scores = pd.read_parquet(structure_sensitivity.paths.run_dir / "teacher_scores/scores.parquet")
    temperatures = read_json(stability.paths.run_dir / "calibration/selection.json")[
        "final_parameters"
    ]["temperatures"]
    action_matrices, logp_matrices = _teacher_matrices(scores, requests, temperatures)
    eligible_roles = _eligible_roles(
        structures, structure_sensitivity.paths.run_dir / "protocol_lock.json"
    )
    teacher_domain = _teacher_domain_metrics(
        variants,
        queries,
        structures,
        action_matrices,
        temperatures,
        cross.paths.storage_dir / "strong_control/components.npz",
    )
    teacher_summary = _summarize_teacher_metrics(
        teacher_domain,
        eligible_roles,
        specification,
    )
    teacher_deltas, teacher_delta_summary = _teacher_paired_deltas(
        teacher_domain,
        eligible_roles,
        specification,
    )
    geometry_query, action_shift_query = _query_geometry_and_action_shift(
        queries,
        structures,
        confidence,
        action_matrices,
        specification["structure_geometry_audit"],
    )
    geometry_domain, geometry_summary = _summarize_geometry(
        geometry_query,
        eligible_roles,
        specification,
    )
    action_geometry_domain, action_geometry_summary = _geometry_action_correlations(
        geometry_query,
        action_shift_query,
        eligible_roles,
        specification,
    )
    distribution_domain, distribution_summary = _teacher_distribution_shift(
        logp_matrices,
        requests,
        eligible_roles,
        specification,
    )
    backbone_domain, backbone_summary = _backbone_geometry_diagnostics(
        structures,
        eligible_roles,
        specification,
    )
    return {
        "structure_sensitivity_teacher_domain_metrics": teacher_domain,
        "structure_sensitivity_teacher_summary": teacher_summary,
        "structure_sensitivity_teacher_paired_deltas": teacher_deltas,
        "structure_sensitivity_teacher_delta_summary": teacher_delta_summary,
        "structure_sensitivity_query_geometry": geometry_query,
        "structure_sensitivity_query_action_shift": action_shift_query,
        "structure_sensitivity_geometry_domain_summary": geometry_domain,
        "structure_sensitivity_geometry_summary": geometry_summary,
        "structure_sensitivity_geometry_action_domain_correlations": action_geometry_domain,
        "structure_sensitivity_geometry_action_correlation_summary": action_geometry_summary,
        "structure_sensitivity_teacher_distribution_domain": distribution_domain,
        "structure_sensitivity_teacher_distribution_summary": distribution_summary,
        "structure_sensitivity_backbone_geometry_domain": backbone_domain,
        "structure_sensitivity_backbone_geometry_summary": backbone_summary,
    }


def _teacher_matrices(
    scores: pd.DataFrame,
    requests: pd.DataFrame,
    temperatures: Mapping[str, float],
) -> tuple[dict[tuple[str, str, str], np.ndarray], dict[tuple[str, str, str], np.ndarray]]:
    request_lookup = requests.set_index(["structure_role", "domain_id"])["state_sequence"]
    actions: dict[tuple[str, str, str], np.ndarray] = {}
    log_probabilities: dict[tuple[str, str, str], np.ndarray] = {}
    for (teacher, role, domain_id), frame in scores.groupby(
        ["teacher_id", "structure_role", "domain_id"], sort=True, observed=True
    ):
        frame = frame.sort_values("position")
        sequence = str(request_lookup.loc[(role, domain_id)])
        positions = frame["position"].to_numpy(dtype=int)
        if not np.array_equal(positions, np.arange(len(sequence), dtype=int)):
            raise ValueError(
                "incomplete structure-sensitivity study score rows for "
                f"{teacher}/{role}/{domain_id}"
            )
        logp = normalize_log_probabilities(frame[logp_columns()].to_numpy(dtype=float))
        wild = np.asarray([AA_TO_INDEX[residue] for residue in sequence], dtype=int)
        key = (str(teacher), str(role), str(domain_id))
        log_probabilities[key] = logp
        actions[key] = _anchor(logp, wild)
    roles_and_domains = sorted({(role, domain) for _, role, domain in actions})
    for role, domain_id in roles_and_domains:
        if not all((teacher, role, domain_id) in actions for teacher in TEACHERS):
            raise ValueError(f"teacher coverage is incomplete for {role}/{domain_id}")
        actions[(CONSENSUS, role, domain_id)] = np.mean(
            np.stack(
                [
                    actions[(teacher, role, domain_id)] / float(temperatures[teacher])
                    for teacher in TEACHERS
                ]
            ),
            axis=0,
        )
    return actions, log_probabilities


def _eligible_roles(structures: pd.DataFrame, lock_path: Path) -> dict[str, bool]:
    lock = read_json(lock_path)
    predictor = lock["predictor_summary_eligible"]
    roles = set(structures["structure_role"])
    return {role: bool(predictor.get(role, True)) for role in roles}


def _teacher_domain_metrics(
    variants: pd.DataFrame,
    queries: pd.DataFrame,
    structures: pd.DataFrame,
    actions: Mapping[tuple[str, str, str], np.ndarray],
    temperatures: Mapping[str, float],
    component_path: Path,
) -> pd.DataFrame:
    del temperatures  # Per-teacher positive scaling does not change rank metrics.
    query_index = queries[["domain_id", "position"]].copy()
    query_index["query_row"] = np.arange(len(query_index), dtype=int)
    indexed = variants.merge(query_index, on=["domain_id", "position"], validate="many_to_one")
    frozen = np.load(component_path)
    comparators = {
        teacher: np.asarray(frozen[f"{teacher}_g"], dtype=float)
        + np.asarray(frozen[f"{teacher}_c_plus"], dtype=float)
        for teacher in TEACHERS
    }
    comparators[CONSENSUS] = np.asarray(
        frozen["temperature_consensus_g"], dtype=float
    ) + np.asarray(frozen["temperature_consensus_c_plus"], dtype=float)
    available = structures.groupby("structure_role", observed=True)["domain_id"].agg(set)
    rows: list[dict[str, Any]] = []
    for role, domains in available.items():
        for domain_id in sorted(domains):
            selected = indexed.loc[indexed["domain_id"].eq(domain_id)]
            positions = selected["position"].to_numpy(dtype=int)
            query_rows = selected["query_row"].to_numpy(dtype=int)
            mutants = selected["mutant"].map(AA_TO_INDEX).to_numpy(dtype=int)
            observed = selected["effect"].to_numpy(dtype=float)
            k = max(1, int(np.ceil(0.10 * len(selected))))
            for teacher in (*TEACHERS, CONSENSUS):
                action = actions[(teacher, role, domain_id)][positions, mutants]
                control = comparators[teacher][query_rows, mutants]
                action_spearman = _spearman(action, observed)
                control_spearman = _spearman(control, observed)
                action_ndcg = _ndcg(action, observed, k=k)
                control_ndcg = _ndcg(control, observed, k=k)
                rows.append(
                    {
                        "teacher_id": teacher,
                        "structure_role": role,
                        "domain_id": domain_id,
                        "n_variants": len(selected),
                        "action_spearman": action_spearman,
                        "g_plus_c_plus_spearman": control_spearman,
                        "spearman_margin": action_spearman - control_spearman,
                        "action_ndcg10": action_ndcg,
                        "g_plus_c_plus_ndcg10": control_ndcg,
                        "ndcg10_margin": action_ndcg - control_ndcg,
                        "stratum": "structure_sensitivity_matched_structure",
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["teacher_id", "structure_role", "domain_id"], ignore_index=True
    )


def _summarize_teacher_metrics(
    domain: pd.DataFrame,
    eligible_roles: Mapping[str, bool],
    specification: Mapping[str, Any],
) -> pd.DataFrame:
    metrics = (
        "action_spearman",
        "g_plus_c_plus_spearman",
        "spearman_margin",
        "action_ndcg10",
        "g_plus_c_plus_ndcg10",
        "ndcg10_margin",
    )
    rows = []
    seed = int(specification["seed"])
    for group_index, ((teacher, role), frame) in enumerate(
        domain.groupby(["teacher_id", "structure_role"], sort=True, observed=True)
    ):
        if not eligible_roles.get(str(role), False):
            continue
        for metric_index, metric in enumerate(metrics):
            rows.append(
                {
                    "teacher_id": teacher,
                    "structure_role": role,
                    "metric": metric,
                    **_bootstrap(
                        frame, metric, specification, seed + group_index * 100 + metric_index
                    ),
                }
            )
    return pd.DataFrame(rows)


def _teacher_paired_deltas(
    domain: pd.DataFrame,
    eligible_roles: Mapping[str, bool],
    specification: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ("action_spearman", "spearman_margin", "action_ndcg10", "ndcg10_margin")
    experimental = domain.loc[
        domain["structure_role"].eq("experimental"), ["teacher_id", "domain_id", *metrics]
    ].rename(columns={metric: f"experimental_{metric}" for metric in metrics})
    paired = domain.merge(experimental, on=["teacher_id", "domain_id"], validate="many_to_one")
    for metric in metrics:
        paired[f"{metric}_delta_vs_experimental"] = (
            paired[metric] - paired[f"experimental_{metric}"]
        )
    delta_metrics = tuple(f"{metric}_delta_vs_experimental" for metric in metrics)
    rows = []
    seed = int(specification["seed"]) + 10_000
    for group_index, ((teacher, role), frame) in enumerate(
        paired.groupby(["teacher_id", "structure_role"], sort=True, observed=True)
    ):
        if not eligible_roles.get(str(role), False):
            continue
        for metric_index, metric in enumerate(delta_metrics):
            rows.append(
                {
                    "teacher_id": teacher,
                    "structure_role": role,
                    "metric": metric,
                    **_bootstrap(
                        frame, metric, specification, seed + group_index * 100 + metric_index
                    ),
                }
            )
    return paired, pd.DataFrame(rows)


def _query_geometry_and_action_shift(
    queries: pd.DataFrame,
    structures: pd.DataFrame,
    confidence: pd.DataFrame,
    actions: Mapping[tuple[str, str, str], np.ndarray],
    settings: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    confidence_lookup = confidence.set_index(["structure_role", "domain_id", "position"])[
        "confidence"
    ]
    structure_lookup = structures.set_index(["structure_role", "domain_id"])
    roles = sorted(structures["structure_role"].unique())
    geometry_rows: list[dict[str, Any]] = []
    shift_rows: list[dict[str, Any]] = []
    local_cutoff = float(settings["local_backbone_neighborhood_angstrom"])
    contact_cutoff = float(settings["contact_cutoff_angstrom"])
    minimum_separation = int(settings["contact_minimum_sequence_separation"])
    lddt_cutoff = float(settings["ca_lddt_neighborhood_angstrom"])
    lddt_thresholds = np.asarray(settings["ca_lddt_thresholds_angstrom"], dtype=float)
    for domain_id, query_frame in queries.groupby("domain_id", sort=True, observed=True):
        reference_path = Path(structure_lookup.loc[("experimental", domain_id), "input_path"])
        reference = np.load(reference_path)["coordinates"].astype(float)
        reference_ca = reference[:, 1]
        for role in roles:
            if (role, domain_id) not in structure_lookup.index:
                continue
            mobile_path = Path(structure_lookup.loc[(role, domain_id), "input_path"])
            mobile = np.load(mobile_path)["coordinates"].astype(float)
            aligned = _align_to_reference(reference, mobile)
            global_rmsd = float(
                np.sqrt(np.mean(np.sum((reference_ca - aligned[:, 1]) ** 2, axis=1)))
            )
            for query in query_frame.itertuples(index=False):
                position = int(query.position)
                distances = np.linalg.norm(reference_ca - reference_ca[position], axis=1)
                local = distances <= local_cutoff
                local_backbone_rmsd = float(
                    np.sqrt(np.mean(np.sum((reference[local] - aligned[local]) ** 2, axis=2)))
                )
                lddt_neighbors = (distances > 0) & (distances <= lddt_cutoff)
                if lddt_neighbors.any():
                    mobile_distances = np.linalg.norm(aligned[:, 1] - aligned[position, 1], axis=1)
                    errors = np.abs(mobile_distances[lddt_neighbors] - distances[lddt_neighbors])
                    ca_lddt = float(np.mean(errors[:, None] < lddt_thresholds[None, :]))
                else:
                    ca_lddt = float("nan")
                indices = np.arange(len(reference))
                contacts = (distances <= contact_cutoff) & (
                    np.abs(indices - position) >= minimum_separation
                )
                if contacts.any():
                    mobile_distances = np.linalg.norm(aligned[:, 1] - aligned[position, 1], axis=1)
                    contact_retention = float((mobile_distances[contacts] <= contact_cutoff).mean())
                else:
                    contact_retention = float("nan")
                try:
                    residue_confidence = float(confidence_lookup.loc[(role, domain_id, position)])
                except KeyError:
                    residue_confidence = float("nan")
                geometry_rows.append(
                    {
                        "structure_role": role,
                        "domain_id": domain_id,
                        "position": position,
                        "global_ca_rmsd_angstrom": global_rmsd,
                        "local_backbone_rmsd_10a": local_backbone_rmsd,
                        "ca_lddt15": ca_lddt,
                        "ca_contact_retention_8a": contact_retention,
                        "local_frame_angular_error_degrees": _frame_angle_degrees(
                            reference[position], aligned[position]
                        ),
                        "ca_displacement_angstrom": float(
                            np.linalg.norm(reference_ca[position] - aligned[position, 1])
                        ),
                        "local_confidence": residue_confidence,
                        "local_residue_count": int(local.sum()),
                    }
                )
                wild = AA_TO_INDEX[str(query.wild_type)]
                keep = np.arange(20) != wild
                for teacher in (*TEACHERS, CONSENSUS):
                    reference_action = actions[(teacher, "experimental", domain_id)][position]
                    role_action = actions[(teacher, role, domain_id)][position]
                    difference = role_action[keep] - reference_action[keep]
                    shift_rows.append(
                        {
                            "teacher_id": teacher,
                            "structure_role": role,
                            "domain_id": domain_id,
                            "position": position,
                            "action_rmse_vs_experimental": float(np.sqrt(np.mean(difference**2))),
                            "action_mae_vs_experimental": float(np.mean(np.abs(difference))),
                            "action_rank_concordance_experimental": _spearman(
                                role_action[keep], reference_action[keep]
                            ),
                        }
                    )
    return (
        pd.DataFrame(geometry_rows).sort_values(
            ["structure_role", "domain_id", "position"], ignore_index=True
        ),
        pd.DataFrame(shift_rows).sort_values(
            ["teacher_id", "structure_role", "domain_id", "position"], ignore_index=True
        ),
    )


def _summarize_geometry(
    query: pd.DataFrame,
    eligible_roles: Mapping[str, bool],
    specification: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = (
        "global_ca_rmsd_angstrom",
        "local_backbone_rmsd_10a",
        "ca_lddt15",
        "ca_contact_retention_8a",
        "local_frame_angular_error_degrees",
        "ca_displacement_angstrom",
        "local_confidence",
    )
    domain_rows = []
    for (role, domain_id), frame in query.groupby(
        ["structure_role", "domain_id"], sort=True, observed=True
    ):
        for metric in metrics:
            values = frame[metric].replace([np.inf, -np.inf], np.nan).dropna()
            domain_rows.append(
                {
                    "structure_role": role,
                    "domain_id": domain_id,
                    "metric": metric,
                    "domain_mean": float(values.mean()) if len(values) else float("nan"),
                    "domain_median": float(values.median()) if len(values) else float("nan"),
                    "positions": int(len(values)),
                    "stratum": "structure_sensitivity_geometry",
                }
            )
    domain = pd.DataFrame(domain_rows)
    rows = []
    seed = int(specification["seed"]) + 20_000
    for group_index, ((role, metric), frame) in enumerate(
        domain.groupby(["structure_role", "metric"], sort=True, observed=True)
    ):
        if not eligible_roles.get(str(role), False):
            continue
        query_values = query.loc[query["structure_role"].eq(role), metric].dropna()
        result = _bootstrap(
            frame.rename(columns={"domain_mean": "value"}),
            "value",
            specification,
            seed + group_index,
        )
        rows.append(
            {
                "structure_role": role,
                "metric": metric,
                **result,
                "pooled_position_p50": float(query_values.quantile(0.50))
                if len(query_values)
                else float("nan"),
                "pooled_position_p90": float(query_values.quantile(0.90))
                if len(query_values)
                else float("nan"),
                "positions": int(len(query_values)),
            }
        )
    return domain, pd.DataFrame(rows)


def _geometry_action_correlations(
    geometry: pd.DataFrame,
    action_shift: pd.DataFrame,
    eligible_roles: Mapping[str, bool],
    specification: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = (
        "local_backbone_rmsd_10a",
        "ca_lddt15",
        "ca_contact_retention_8a",
        "local_frame_angular_error_degrees",
        "ca_displacement_angstrom",
        "local_confidence",
    )
    minimum = int(
        specification["structure_geometry_audit"]["minimum_positions_for_within_domain_correlation"]
    )
    merged = action_shift.merge(
        geometry,
        on=["structure_role", "domain_id", "position"],
        validate="many_to_one",
    )
    rows = []
    for (teacher, role, domain_id), frame in merged.groupby(
        ["teacher_id", "structure_role", "domain_id"], sort=True, observed=True
    ):
        for metric in metrics:
            clean = frame[[metric, "action_rmse_vs_experimental"]].dropna()
            value = (
                _spearman(
                    clean[metric].to_numpy(dtype=float),
                    clean["action_rmse_vs_experimental"].to_numpy(dtype=float),
                )
                if len(clean) >= minimum
                else float("nan")
            )
            rows.append(
                {
                    "teacher_id": teacher,
                    "structure_role": role,
                    "domain_id": domain_id,
                    "geometry_metric": metric,
                    "positions": len(clean),
                    "spearman_with_action_rmse": value,
                    "stratum": "within_domain_query_positions",
                }
            )
    domain = pd.DataFrame(rows)
    summary_rows = []
    seed = int(specification["seed"]) + 30_000
    for group_index, ((teacher, role, metric), frame) in enumerate(
        domain.groupby(
            ["teacher_id", "structure_role", "geometry_metric"], sort=True, observed=True
        )
    ):
        if not eligible_roles.get(str(role), False) or role == "experimental":
            continue
        summary_rows.append(
            {
                "teacher_id": teacher,
                "structure_role": role,
                "geometry_metric": metric,
                **_bootstrap(
                    frame.rename(columns={"spearman_with_action_rmse": "value"}),
                    "value",
                    specification,
                    seed + group_index,
                ),
            }
        )
    return domain, pd.DataFrame(summary_rows)


def _teacher_distribution_shift(
    logp: Mapping[tuple[str, str, str], np.ndarray],
    requests: pd.DataFrame,
    eligible_roles: Mapping[str, bool],
    specification: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sequence_lookup = requests.set_index(["structure_role", "domain_id"])["state_sequence"]
    rows = []
    for (teacher, role, domain_id), values in sorted(logp.items()):
        reference = logp[(teacher, "experimental", domain_id)]
        p = np.exp(values)
        q = np.exp(reference)
        midpoint = 0.5 * (p + q)
        jsd = 0.5 * np.sum(p * (values - np.log(midpoint)), axis=1) + 0.5 * np.sum(
            q * (reference - np.log(midpoint)), axis=1
        )
        entropy = -np.sum(p * values, axis=1)
        reference_entropy = -np.sum(q * reference, axis=1)
        sequence = str(sequence_lookup.loc[(role, domain_id)])
        wild = np.asarray([AA_TO_INDEX[residue] for residue in sequence], dtype=int)
        index = np.arange(len(wild), dtype=int)
        native_nll = float(-values[index, wild].mean())
        reference_nll = float(-reference[index, wild].mean())
        native_aar = float((np.argmax(values, axis=1) == wild).mean())
        reference_aar = float((np.argmax(reference, axis=1) == wild).mean())
        rows.append(
            {
                "teacher_id": teacher,
                "structure_role": role,
                "domain_id": domain_id,
                "residues": len(values),
                "jsd_mean_vs_experimental": float(jsd.mean()),
                "entropy_mean": float(entropy.mean()),
                "entropy_delta_vs_experimental": float(entropy.mean() - reference_entropy.mean()),
                "native_nll": native_nll,
                "native_nll_delta_vs_experimental": native_nll - reference_nll,
                "native_aar": native_aar,
                "native_aar_delta_vs_experimental": native_aar - reference_aar,
                "stratum": "teacher_input_distribution",
            }
        )
    domain = pd.DataFrame(rows)
    metrics = (
        "jsd_mean_vs_experimental",
        "entropy_delta_vs_experimental",
        "native_nll",
        "native_nll_delta_vs_experimental",
        "native_aar",
        "native_aar_delta_vs_experimental",
    )
    summary_rows = []
    seed = int(specification["seed"]) + 40_000
    for group_index, ((teacher, role), frame) in enumerate(
        domain.groupby(["teacher_id", "structure_role"], sort=True, observed=True)
    ):
        if not eligible_roles.get(str(role), False):
            continue
        for metric_index, metric in enumerate(metrics):
            summary_rows.append(
                {
                    "teacher_id": teacher,
                    "structure_role": role,
                    "metric": metric,
                    **_bootstrap(
                        frame,
                        metric,
                        specification,
                        seed + group_index * 100 + metric_index,
                    ),
                }
            )
    return domain, pd.DataFrame(summary_rows)


def _backbone_geometry_diagnostics(
    structures: pd.DataFrame,
    eligible_roles: Mapping[str, bool],
    specification: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    break_threshold = float(
        specification["structure_geometry_audit"]["chain_break_cn_distance_angstrom"]
    )
    lookup = structures.set_index(["structure_role", "domain_id"])
    rows = []
    for domain_id in sorted(structures["domain_id"].unique()):
        reference = np.load(Path(lookup.loc[("experimental", domain_id), "input_path"]))[
            "coordinates"
        ].astype(float)
        reference_values = _backbone_arrays(reference)
        roles = structures.loc[structures["domain_id"].eq(domain_id), "structure_role"]
        for role in sorted(roles):
            coordinates = np.load(Path(lookup.loc[(role, domain_id), "input_path"]))[
                "coordinates"
            ].astype(float)
            values = _backbone_arrays(coordinates)
            bond_delta = np.concatenate(
                [np.abs(values[name] - reference_values[name]) for name in ("n_ca", "ca_c", "c_o")]
            )
            omega_delta = _circular_degrees(values["omega"], reference_values["omega"])
            reference_sign = np.sign(reference_values["pseudochirality"])
            role_sign = np.sign(values["pseudochirality"])
            rows.append(
                {
                    "structure_role": role,
                    "domain_id": domain_id,
                    "within_residue_bond_mae_vs_experimental_angstrom": float(bond_delta.mean()),
                    "n_ca_c_angle_mae_vs_experimental_degrees": float(
                        np.mean(np.abs(values["n_ca_c_angle"] - reference_values["n_ca_c_angle"]))
                    ),
                    "peptide_cn_mean_angstrom": float(values["peptide_cn"].mean()),
                    "peptide_cn_max_angstrom": float(values["peptide_cn"].max()),
                    "peptide_cn_mae_vs_experimental_angstrom": float(
                        np.mean(np.abs(values["peptide_cn"] - reference_values["peptide_cn"]))
                    ),
                    "chain_break_fraction_cn_gt_2a": float(
                        (values["peptide_cn"] > break_threshold).mean()
                    ),
                    "omega_mae_vs_experimental_degrees": float(omega_delta.mean()),
                    "backbone_pseudochirality_flip_fraction": float(
                        (reference_sign != role_sign).mean()
                    ),
                    "stratum": "backbone_geometry",
                }
            )
    domain = pd.DataFrame(rows)
    metrics = tuple(
        column
        for column in domain.columns
        if column not in {"structure_role", "domain_id", "stratum"}
    )
    summary_rows = []
    seed = int(specification["seed"]) + 50_000
    for group_index, (role, frame) in enumerate(
        domain.groupby("structure_role", sort=True, observed=True)
    ):
        if not eligible_roles.get(str(role), False):
            continue
        for metric_index, metric in enumerate(metrics):
            summary_rows.append(
                {
                    "structure_role": role,
                    "metric": metric,
                    **_bootstrap(
                        frame,
                        metric,
                        specification,
                        seed + group_index * 100 + metric_index,
                    ),
                }
            )
    return domain, pd.DataFrame(summary_rows)


def _align_to_reference(reference: np.ndarray, mobile: np.ndarray) -> np.ndarray:
    """Rigidly align a mobile backbone to a reference using all C-alpha atoms."""

    reference_ca = np.asarray(reference[:, 1], dtype=float)
    mobile_ca = np.asarray(mobile[:, 1], dtype=float)
    reference_center = reference_ca.mean(axis=0)
    mobile_center = mobile_ca.mean(axis=0)
    left = reference_ca - reference_center
    right = mobile_ca - mobile_center
    u, _, vt = np.linalg.svd(right.T @ left)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    return (np.asarray(mobile, dtype=float) - mobile_center) @ rotation + reference_center


def _frame_angle_degrees(reference: np.ndarray, mobile: np.ndarray) -> float:
    def frame(values: np.ndarray) -> np.ndarray:
        n, ca, c = values[:3]
        x = c - ca
        x /= np.linalg.norm(x)
        y = n - ca
        y -= np.dot(y, x) * x
        y /= np.linalg.norm(y)
        z = np.cross(x, y)
        return np.stack([x, y, z], axis=1)

    relative = frame(reference).T @ frame(mobile)
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _backbone_arrays(coordinates: np.ndarray) -> dict[str, np.ndarray]:
    n = coordinates[:, 0]
    ca = coordinates[:, 1]
    c = coordinates[:, 2]
    o = coordinates[:, 3]
    return {
        "n_ca": np.linalg.norm(n - ca, axis=1),
        "ca_c": np.linalg.norm(ca - c, axis=1),
        "c_o": np.linalg.norm(c - o, axis=1),
        "n_ca_c_angle": _angles(n, ca, c),
        "peptide_cn": np.linalg.norm(c[:-1] - n[1:], axis=1),
        "omega": _dihedrals(ca[:-1], c[:-1], n[1:], ca[1:]),
        "pseudochirality": np.einsum("ij,ij->i", np.cross(n - ca, c - ca), o - ca),
    }


def _angles(left: np.ndarray, center: np.ndarray, right: np.ndarray) -> np.ndarray:
    first = left - center
    second = right - center
    cosine = np.einsum("ij,ij->i", first, second) / (
        np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    )
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _dihedrals(
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
    fourth: np.ndarray,
) -> np.ndarray:
    b0 = -(second - first)
    b1 = third - second
    b2 = fourth - third
    b1 /= np.linalg.norm(b1, axis=1)[:, None]
    v = b0 - np.einsum("ij,ij->i", b0, b1)[:, None] * b1
    w = b2 - np.einsum("ij,ij->i", b2, b1)[:, None] * b1
    x = np.einsum("ij,ij->i", v, w)
    y = np.einsum("ij,ij->i", np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))


def _circular_degrees(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.abs((left - right + 180.0) % 360.0 - 180.0)


def _bootstrap(
    frame: pd.DataFrame,
    metric: str,
    specification: Mapping[str, Any],
    seed: int,
) -> dict[str, float | int]:
    selected = frame.copy()
    if "stratum" not in selected:
        selected["stratum"] = "postlock_audit"
    return stratified_domain_bootstrap(
        selected,
        metric,
        replicates=int(specification["inference"]["bootstrap_replicates"]),
        confidence_level=float(specification["inference"]["confidence_level"]),
        seed=seed,
    )
