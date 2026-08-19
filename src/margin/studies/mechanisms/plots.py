"""Publication-ready figures for the mechanism study audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from margin.provenance import (
    runtime_manifest,
    sha256_file,
    table_manifest,
    write_csv,
    write_json,
    write_parquet,
)
from margin.studies.mechanisms.config import MechanismStudyConfig

PALETTE = {
    "contact_deletion": "#0072B2",
    "smooth_coordinate": "#009E73",
    "constrained_reassignment": "#E69F00",
    "matched_real_structure": "#CC79A7",
    "legacy_ood_rewiring": "#D55E00",
    "paired_only": "#56B4E9",
    "route_b": "#000000",
    "route_b_control": "#999999",
}
CONDITION_ORDER = [
    ("contact_deletion", "0.05"),
    ("contact_deletion", "0.1"),
    ("contact_deletion", "0.2"),
    ("smooth_coordinate", "0.25"),
    ("smooth_coordinate", "0.5"),
    ("smooth_coordinate", "1"),
    ("constrained_reassignment", "0.1"),
    ("matched_real_structure", "descriptor_matched"),
    ("legacy_ood_rewiring", "5_swaps_per_edge"),
]
CONDITION_LABELS = {
    ("contact_deletion", "0.05"): "delete 5%",
    ("contact_deletion", "0.1"): "delete 10%",
    ("contact_deletion", "0.2"): "delete 20%",
    ("smooth_coordinate", "0.25"): "coordinate 0.25 Å",
    ("smooth_coordinate", "0.5"): "coordinate 0.50 Å",
    ("smooth_coordinate", "1"): "coordinate 1.00 Å",
    ("constrained_reassignment", "0.1"): "constrained 10%",
    ("matched_real_structure", "descriptor_matched"): "matched real",
    ("legacy_ood_rewiring", "5_swaps_per_edge"): "legacy rewire",
}
METHOD_LABELS = {
    "sequence_plus_mif_paired_alpha_1": "seq + paired (α=1)",
    "sequence_plus_mif_paired_variance_matched": "seq + paired (RMS match)",
    "legacy_direct_contrast": "legacy contrast",
    "contrast__contact_deletion__0.05": "delete 5% contrast",
    "contrast__smooth_coordinate__0.25": "coordinate 0.25 Å",
    "contrast__constrained_reassignment__0.1": "constrained contrast",
    "contrast__matched_real_structure__descriptor_matched": "matched-real contrast",
    "route_b_carp_rank16": "CARP-B rank 16",
    "route_b_global_wt_mutant_matrix": "global WT→mutant",
    "route_b_simple_sequence_context": "simple seq context",
}


def build_mechanism_figures(config: MechanismStudyConfig) -> dict[str, Path]:
    """Build main and mechanism figures with reusable source-data tables."""

    evaluation = config.paths.run_dir / "evaluation"
    output = config.paths.run_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    distribution = pd.read_parquet(evaluation / "distribution_domain_rows.parquet")
    validity = pd.read_parquet(evaluation / "condition_validity.parquet")
    increments = pd.read_parquet(evaluation / "increment_summary.parquet")
    registry = pd.read_parquet(evaluation / "method_registry.parquet")
    route_margins = pd.read_parquet(evaluation / "route_b_margin_summary.parquet")
    subgroup = pd.read_parquet(evaluation / "gly_pro_subgroup_summary.parquet")

    condition_source = _condition_source(distribution, validity, increments)
    performance_source = _performance_source(increments, registry)
    main_source = pd.concat(
        [
            condition_source.assign(source_panel="a_b"),
            performance_source.assign(source_panel="c_d"),
        ],
        ignore_index=True,
        sort=False,
    )
    main_paths = _plot_main(condition_source, performance_source, output)

    mechanism_source = _mechanism_source(condition_source, route_margins, subgroup)
    mechanism_paths = _plot_mechanism(condition_source, route_margins, subgroup, output)
    main_source_path = output / "mechanisms_main_source_data.parquet"
    main_csv_path = output / "mechanisms_main_source_data.csv"
    mechanism_source_path = output / "mechanisms_mechanism_source_data.parquet"
    mechanism_csv_path = output / "mechanisms_mechanism_source_data.csv"
    write_parquet(main_source_path, main_source)
    write_csv(main_csv_path, main_source)
    write_parquet(mechanism_source_path, mechanism_source)
    write_csv(mechanism_csv_path, mechanism_source)
    paths = {
        **main_paths,
        **mechanism_paths,
        "main_source_data": main_source_path,
        "main_source_csv": main_csv_path,
        "mechanism_source_data": mechanism_source_path,
        "mechanism_source_csv": mechanism_csv_path,
    }
    manifest_path = output / "manifest.json"
    write_json(
        manifest_path,
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "style": "colorblind_safe_redundant_markers_vector_exports",
            "artifacts": [
                {"name": name, "path": str(path), "sha256": sha256_file(path)}
                for name, path in paths.items()
            ],
            "source_tables": [
                table_manifest(main_source_path, main_source),
                table_manifest(mechanism_source_path, mechanism_source),
            ],
        },
    )
    paths["manifest"] = manifest_path
    return paths


def _condition_source(
    distribution: pd.DataFrame,
    validity: pd.DataFrame,
    increments: pd.DataFrame,
) -> pd.DataFrame:
    grouped = distribution.groupby(
        ["counterfactual_family", "condition_level"], sort=True, observed=True
    )[["paired_counterfactual_jsd_nats", "absolute_entropy_shift_nats"]].agg(
        ["median", lambda values: values.quantile(0.25), lambda values: values.quantile(0.75)]
    )
    grouped.columns = [
        "jsd_median",
        "jsd_q25",
        "jsd_q75",
        "entropy_median",
        "entropy_q25",
        "entropy_q75",
    ]
    source = grouped.reset_index().merge(
        validity,
        on=["counterfactual_family", "condition_level"],
        how="left",
        validate="one_to_one",
    )
    metric_rows = increments.loc[
        increments["scope"].eq("all_stratum_preserving")
        & increments["metric"].isin(["spearman_increment", "ndcg_at_10_percent_increment"])
    ][["method", "metric", "estimate", "ci_low", "ci_high"]]
    metric_wide = metric_rows.pivot(index="method", columns="metric").reset_index()
    metric_wide.columns = [
        "method" if column == ("method", "") or column == "method" else f"{column[1]}_{column[0]}"
        for column in metric_wide.columns
    ]
    source["method"] = source.apply(
        lambda row: (
            "legacy_direct_contrast"
            if row["counterfactual_family"] == "legacy_ood_rewiring"
            else f"contrast__{row['counterfactual_family']}__{row['condition_level']}"
        ),
        axis=1,
    )
    source = source.merge(metric_wide, on="method", how="left", validate="one_to_one")
    source["condition_label"] = [
        CONDITION_LABELS.get((family, level), f"{family}/{level}")
        for family, level in zip(
            source["counterfactual_family"], source["condition_level"], strict=True
        )
    ]
    return source


def _performance_source(increments: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    methods = list(METHOD_LABELS)
    source = increments.loc[
        increments["method"].isin(methods)
        & increments["scope"].eq("all_stratum_preserving")
        & increments["metric"].isin(["spearman_increment", "ndcg_at_10_percent_increment"])
    ][["method", "metric", "estimate", "ci_low", "ci_high"]].merge(
        registry[["method", "category"]], on="method", validate="many_to_one"
    )
    source["method_label"] = source["method"].map(METHOD_LABELS)
    return source


def _mechanism_source(
    condition_source: pd.DataFrame,
    route_margins: pd.DataFrame,
    subgroup: pd.DataFrame,
) -> pd.DataFrame:
    condition = condition_source.assign(source_panel="a_b")
    route = route_margins.assign(source_panel="c")
    boundary = subgroup.loc[
        subgroup["method"].isin(
            [
                "legacy_direct_contrast",
                "contrast__matched_real_structure__descriptor_matched",
                "route_b_carp_rank16",
            ]
        )
        & subgroup["metric"].eq("spearman_increment")
    ].assign(source_panel="d")
    return pd.concat([condition, route, boundary], ignore_index=True, sort=False)


def _plot_main(
    conditions: pd.DataFrame, performance: pd.DataFrame, output: Path
) -> dict[str, Path]:
    _set_style()
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), constrained_layout=True)
    ordered = _ordered_conditions(conditions)
    y = np.arange(len(ordered))
    for axis, value, lower, upper, threshold, title, xlabel in (
        (
            axes[0, 0],
            "jsd_median",
            "jsd_q25",
            "jsd_q75",
            0.10,
            "Counterfactual distribution shift",
            "Paired–counterfactual JSD (nats)",
        ),
        (
            axes[0, 1],
            "entropy_median",
            "entropy_q25",
            "entropy_q75",
            0.25,
            "Absolute entropy shift",
            "|Δ entropy| (nats)",
        ),
    ):
        for index, row in ordered.iterrows():
            center = float(row[value])
            axis.errorbar(
                center,
                y[index],
                xerr=np.asarray([[center - row[lower]], [row[upper] - center]]),
                fmt="o" if bool(row.get("id_compatible", False)) else "s",
                color=PALETTE.get(row["counterfactual_family"], "#666666"),
                markersize=4.5,
                capsize=2,
                linewidth=1,
            )
        axis.axvline(threshold, color="#444444", linestyle="--", linewidth=1)
        axis.set_yticks(y, ordered["condition_label"])
        axis.invert_yaxis()
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.grid(axis="x", color="#dddddd", linewidth=0.6)
    _performance_forest(
        axes[1, 0], performance, "spearman_increment", "Δ Spearman", "Ranking increment"
    )
    _performance_forest(
        axes[1, 1],
        performance,
        "ndcg_at_10_percent_increment",
        "Δ NDCG@10%",
        "Top-ranking increment",
    )
    _panel_labels(axes.ravel())
    return _save_figure(figure, output, "mechanisms_main")


def _performance_forest(
    axis: plt.Axes,
    source: pd.DataFrame,
    metric: str,
    xlabel: str,
    title: str,
) -> None:
    selected = source.loc[source["metric"].eq(metric)].copy()
    order = list(METHOD_LABELS)
    selected["ordinal"] = selected["method"].map(
        {method: index for index, method in enumerate(order)}
    )
    selected = selected.sort_values("ordinal").reset_index(drop=True)
    y = np.arange(len(selected))
    for index, row in selected.iterrows():
        axis.errorbar(
            row["estimate"],
            y[index],
            xerr=np.asarray(
                [[row["estimate"] - row["ci_low"]], [row["ci_high"] - row["estimate"]]]
            ),
            fmt="o",
            color=PALETTE.get(str(row["category"]), "#666666"),
            capsize=2,
            markersize=4.5,
            linewidth=1,
        )
    axis.axvline(0, color="#444444", linewidth=0.8)
    axis.set_yticks(y, selected["method_label"])
    axis.invert_yaxis()
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    axis.grid(axis="x", color="#dddddd", linewidth=0.6)


def _plot_mechanism(
    conditions: pd.DataFrame,
    route_margins: pd.DataFrame,
    subgroup: pd.DataFrame,
    output: Path,
) -> dict[str, Path]:
    _set_style()
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), constrained_layout=True)
    ordered = _ordered_conditions(conditions)
    short_labels = {
        "delete 5%": ("del 5%", 5, -5),
        "delete 10%": ("del 10%", 5, -12),
        "delete 20%": ("del 20%", 5, 5),
        "coordinate 0.25 Å": ("coord 0.25 Å", 5, 8),
        "coordinate 0.50 Å": ("coord 0.50 Å", 5, 6),
        "coordinate 1.00 Å": ("coord 1.00 Å", 5, 6),
        "constrained 10%": ("constrained 10%", 5, -10),
        "matched real": ("matched real", 5, 5),
        "legacy rewire": ("legacy rewire", 5, 5),
    }
    for _, row in ordered.iterrows():
        color = PALETTE.get(row["counterfactual_family"], "#666666")
        axes[0, 0].scatter(row["jsd_median"], row["spearman_increment_estimate"], color=color, s=28)
        label, offset_x, offset_y = short_labels[row["condition_label"]]
        axes[0, 0].annotate(
            label,
            (row["jsd_median"], row["spearman_increment_estimate"]),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            fontsize=6.5,
        )
        axes[0, 1].scatter(
            row["median_seed_action_spearman"],
            row["spearman_increment_estimate"],
            color=color,
            s=28,
        )
    axes[0, 0].axvline(0.10, color="#444444", linestyle="--", linewidth=1)
    axes[0, 0].axhline(0, color="#444444", linewidth=0.8)
    axes[0, 0].set(
        xlabel="Median JSD (nats)",
        ylabel="Δ Spearman",
        title="Signal grows with distribution shift",
    )
    axes[0, 1].axvline(0.50, color="#444444", linestyle="--", linewidth=1)
    axes[0, 1].axhline(0, color="#444444", linewidth=0.8)
    axes[0, 1].set(
        xlabel="Median seed action Spearman",
        ylabel="Δ Spearman",
        title="Seed reliability versus utility",
    )
    _route_margin_forest(axes[1, 0], route_margins, "spearman", "CARP-B Δ Spearman margin")
    _boundary_plot(axes[1, 1], subgroup)
    _panel_labels(axes.ravel())
    return _save_figure(figure, output, "mechanisms_mechanism")


def _route_margin_forest(axis: plt.Axes, source: pd.DataFrame, metric: str, title: str) -> None:
    labels = {
        "route_b_global_wt_mutant_matrix": "global WT→mutant",
        "route_b_grantham_aaindex_blosum_linear": "physicochemical linear",
        "route_b_simple_sequence_context": "simple seq context",
        "route_b_carp_context_shuffled": "CARP context shuffled",
        "route_b_target_conditionally_shuffled": "target shuffled",
        "direct_legacy_pca_rank1": "direct PCA r1",
        "direct_legacy_pca_rank3": "direct PCA r3",
        "direct_legacy_pca_rank5": "direct PCA r5",
        "direct_legacy_pca_rank16": "direct PCA r16",
        "direct_legacy_rms_shrinkage": "direct RMS shrink",
    }
    selected = source.loc[source["metric"].eq(metric)].copy()
    selected["label"] = selected["comparator_method"].map(labels)
    selected["ordinal"] = selected["comparator_method"].map(
        {method: index for index, method in enumerate(labels)}
    )
    selected = selected.sort_values("ordinal").reset_index(drop=True)
    y = np.arange(len(selected))
    for index, row in selected.iterrows():
        axis.errorbar(
            row["estimate"],
            y[index],
            xerr=np.asarray(
                [[row["estimate"] - row["ci_low"]], [row["ci_high"] - row["estimate"]]]
            ),
            fmt="o",
            color="#0072B2" if row["estimate"] > 0 else "#D55E00",
            capsize=2,
            markersize=4.5,
            linewidth=1,
        )
    axis.axvline(0, color="#444444", linewidth=0.8)
    axis.set_yticks(y, selected["label"])
    axis.invert_yaxis()
    axis.set_xlabel("CARP-B minus control")
    axis.set_title(title)
    axis.grid(axis="x", color="#dddddd", linewidth=0.6)


def _boundary_plot(axis: plt.Axes, source: pd.DataFrame) -> None:
    methods = [
        "legacy_direct_contrast",
        "contrast__matched_real_structure__descriptor_matched",
        "route_b_carp_rank16",
    ]
    labels = ["legacy", "matched real", "CARP-B"]
    selected = source.loc[
        source["method"].isin(methods) & source["metric"].eq("spearman_increment")
    ]
    offsets = {
        "involves_glycine_or_proline": -0.12,
        "other_substitutions": 0.12,
    }
    colors = {
        "involves_glycine_or_proline": "#D55E00",
        "other_substitutions": "#0072B2",
    }
    markers = {"involves_glycine_or_proline": "s", "other_substitutions": "o"}
    for method_index, method in enumerate(methods):
        for subgroup_name in offsets:
            row = selected.loc[
                selected["method"].eq(method) & selected["subgroup"].eq(subgroup_name)
            ].iloc[0]
            x = method_index + offsets[subgroup_name]
            axis.errorbar(
                x,
                row["estimate"],
                yerr=np.asarray(
                    [[row["estimate"] - row["ci_low"]], [row["ci_high"] - row["estimate"]]]
                ),
                fmt=markers[subgroup_name],
                color=colors[subgroup_name],
                capsize=2,
                markersize=4.5,
                linewidth=1,
                label=(
                    (
                        "Gly/Pro"
                        if subgroup_name == "involves_glycine_or_proline"
                        else "other substitutions"
                    )
                    if method_index == 0
                    else None
                ),
            )
    axis.axhline(0, color="#444444", linewidth=0.8)
    axis.set_xticks(np.arange(len(methods)), labels)
    axis.set_ylabel("Δ Spearman")
    axis.set_title("Predeclared Gly/Pro boundary")
    axis.legend(frameon=False, fontsize=7, loc="best")
    axis.grid(axis="y", color="#dddddd", linewidth=0.6)


def _ordered_conditions(source: pd.DataFrame) -> pd.DataFrame:
    order = {key: index for index, key in enumerate(CONDITION_ORDER)}
    result = source.copy()
    result["ordinal"] = [
        order.get((family, level), len(order))
        for family, level in zip(
            result["counterfactual_family"], result["condition_level"], strict=True
        )
    ]
    return result.sort_values("ordinal").reset_index(drop=True)


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _panel_labels(axes: Any) -> None:
    for label, axis in zip("abcd", axes, strict=True):
        axis.text(
            -0.12,
            1.07,
            label,
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )


def _save_figure(figure: plt.Figure, output: Path, stem: str) -> dict[str, Path]:
    paths = {}
    for extension in ("png", "pdf", "svg"):
        path = output / f"{stem}.{extension}"
        figure.savefig(path, dpi=300 if extension == "png" else None, bbox_inches="tight")
        paths[f"{stem}_{extension}"] = path
    plt.close(figure)
    return paths
