"""Publication-quality static figures for the counterfactual study report."""

from __future__ import annotations

from pathlib import Path
from string import ascii_uppercase
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from margin.provenance import (
    runtime_manifest,
    sha256_file,
    table_manifest,
    write_json,
    write_parquet,
)
from margin.studies.counterfactuals.config import CounterfactualStudyConfig
from margin.studies.counterfactuals.evaluation import (
    ROUTE_A_PRIMARY,
    ROUTE_A_REPLICATION,
    ROUTE_B_PRIMARY,
    ROUTE_B_REPLICATION,
    stratified_domain_bootstrap,
)

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GRAY = "#777777"


def build_counterfactual_figures(config: CounterfactualStudyConfig) -> dict[str, Path]:
    """Create the main decision and exploratory mechanism figures."""

    _configure_style()
    evaluation = config.paths.run_dir / "evaluation"
    mechanisms = config.paths.run_dir / "mechanisms"
    output = config.paths.run_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.read_parquet(evaluation / "increment_summary.parquet")
    increments = pd.read_parquet(evaluation / "domain_increments.parquet")
    margins = pd.read_parquet(evaluation / "random_control_margins.parquet")
    main_source = _main_figure(summary, increments, margins, output)
    mechanism_source = _mechanism_figure(config, summary, mechanisms, output)
    source_paths = {
        "main_source_data": output / "counterfactuals_main_source_data.parquet",
        "mechanism_source_data": output / "counterfactuals_mechanism_source_data.parquet",
    }
    write_parquet(source_paths["main_source_data"], main_source)
    write_parquet(source_paths["mechanism_source_data"], mechanism_source)
    paths: dict[str, Path] = {**source_paths}
    for stem in ("counterfactuals_main", "counterfactuals_mechanism"):
        for suffix in ("pdf", "svg", "png"):
            paths[f"{stem}_{suffix}"] = output / f"{stem}.{suffix}"
    manifest_path = output / "manifest.json"
    write_json(
        manifest_path,
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "style": "double-column_publication_colorblind_safe",
            "uncertainty": "95% stratum-preserving domain bootstrap confidence intervals",
            "source_tables": [
                table_manifest(source_paths["main_source_data"], main_source),
                table_manifest(source_paths["mechanism_source_data"], mechanism_source),
            ],
            "figures": [
                {"path": str(path), "sha256": sha256_file(path)}
                for key, path in paths.items()
                if key.endswith(("_pdf", "_svg", "_png"))
            ],
        },
    )
    paths["manifest"] = manifest_path
    return paths


def _main_figure(
    summary: pd.DataFrame,
    increments: pd.DataFrame,
    margins: pd.DataFrame,
    output: Path,
) -> pd.DataFrame:
    methods = [ROUTE_A_PRIMARY, ROUTE_A_REPLICATION, ROUTE_B_PRIMARY, ROUTE_B_REPLICATION]
    labels = {
        ROUTE_A_PRIMARY: "A · rewired 5",
        ROUTE_A_REPLICATION: "A · circular",
        ROUTE_B_PRIMARY: "B · rewired 5",
        ROUTE_B_REPLICATION: "B · circular",
    }
    all_rows = summary.loc[summary["method"].isin(methods) & summary["stratum"].eq("all")].copy()
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.4), constrained_layout=True)
    _forest(
        axes[0, 0],
        all_rows,
        "spearman_increment",
        methods,
        labels,
        "Spearman increment",
    )
    _forest(
        axes[0, 1],
        all_rows,
        "ndcg_increment",
        methods,
        labels,
        "NDCG increment",
    )
    a = increments.loc[
        increments["method"].eq(ROUTE_A_PRIMARY),
        ["domain_id", "stratum", "spearman_increment"],
    ].rename(columns={"spearman_increment": "route_a_spearman_increment"})
    b = increments.loc[
        increments["method"].eq(ROUTE_B_PRIMARY),
        ["domain_id", "spearman_increment"],
    ].rename(columns={"spearman_increment": "route_b_spearman_increment"})
    scatter = a.merge(b, on="domain_id", validate="one_to_one")
    _domain_scatter(
        axes[1, 0],
        scatter,
        "route_a_spearman_increment",
        "route_b_spearman_increment",
        "Route A increment",
        "Route B increment",
    )
    control = margins[
        [
            "domain_id",
            "stratum",
            "spearman_increment",
            "random_mean_spearman_increment",
        ]
    ].copy()
    _domain_scatter(
        axes[1, 1],
        control,
        "random_mean_spearman_increment",
        "spearman_increment",
        "Matched-random increment",
        "Direct Route A increment",
    )
    for index, axis in enumerate(axes.flat):
        axis.text(
            -0.16,
            1.08,
            ascii_uppercase[index],
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )
    _save(fig, output / "counterfactuals_main")
    all_rows["figure_panel"] = all_rows["metric"].map(
        {"spearman_increment": "A", "ndcg_increment": "B"}
    )
    scatter_source = scatter.assign(figure_panel="C", method="route_a_vs_route_b")
    control_source = control.assign(figure_panel="D", method="direct_vs_matched_random")
    return pd.concat([all_rows, scatter_source, control_source], ignore_index=True, sort=False)


def _mechanism_figure(
    config: CounterfactualStudyConfig,
    summary: pd.DataFrame,
    mechanisms: Path,
    output: Path,
) -> pd.DataFrame:
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.7), constrained_layout=True)
    intensity = summary.loc[
        summary["method"].str.startswith("direct_mif_paired_minus_contact_rewired_")
        & summary["metric"].eq("spearman_increment")
        & summary["stratum"].eq("all")
    ].copy()
    intensity["rewiring_strength"] = intensity["method"].str.rsplit("_", n=1).str[-1].astype(float)
    intensity = intensity.sort_values("rewiring_strength")
    axes[0, 0].errorbar(
        intensity["rewiring_strength"],
        intensity["estimate"],
        yerr=np.vstack(
            [
                intensity["estimate"] - intensity["ci_low"],
                intensity["ci_high"] - intensity["estimate"],
            ]
        ),
        color=BLUE,
        marker="o",
        markersize=4,
        capsize=3,
    )
    axes[0, 0].axhline(0, color=GRAY, linewidth=0.8, linestyle="--")
    axes[0, 0].axvline(5, color=ORANGE, linewidth=0.8, linestyle=":")
    axes[0, 0].set_xlabel("Requested edge swaps per contact")
    axes[0, 0].set_ylabel("Spearman increment")
    axes[0, 0].set_title("Rewiring-strength sensitivity")

    ood = pd.read_parquet(mechanisms / "ood_position_rows.parquet")
    ood = ood.loc[ood["rewiring_swaps_per_edge"].notna()]
    ood_domain = (
        ood.groupby(
            ["counterfactual_role", "rewiring_swaps_per_edge", "domain_id", "stratum"],
            observed=True,
        )[["paired_counterfactual_jsd_nats", "entropy_change_nats"]]
        .mean()
        .reset_index()
    )
    ood_rows = []
    for strength, frame in ood_domain.groupby("rewiring_swaps_per_edge", sort=True):
        for metric_index, metric in enumerate(
            ("paired_counterfactual_jsd_nats", "entropy_change_nats")
        ):
            ood_rows.append(
                {
                    "rewiring_strength": float(strength),
                    "metric": metric,
                    **stratified_domain_bootstrap(
                        frame,
                        metric,
                        replicates=config.inference.bootstrap_replicates,
                        confidence_level=config.inference.confidence_level,
                        seed=config.seed + 900_000 + int(strength * 10) + metric_index,
                    ),
                }
            )
    ood_source = pd.DataFrame(ood_rows)
    for metric, color, marker, label in (
        ("paired_counterfactual_jsd_nats", GREEN, "o", "JSD"),
        ("entropy_change_nats", PURPLE, "s", "Entropy change"),
    ):
        frame = ood_source.loc[ood_source["metric"].eq(metric)]
        axes[0, 1].errorbar(
            frame["rewiring_strength"],
            frame["estimate"],
            yerr=np.vstack(
                [
                    frame["estimate"] - frame["ci_low"],
                    frame["ci_high"] - frame["estimate"],
                ]
            ),
            color=color,
            marker=marker,
            markersize=4,
            capsize=2,
            label=label,
        )
    axes[0, 1].set_xlabel("Requested edge swaps per contact")
    axes[0, 1].set_ylabel("Distribution shift (nats)")
    axes[0, 1].set_title("Counterfactual distribution shift")
    axes[0, 1].legend(frameon=False)

    pca = pd.read_parquet(mechanisms / "residual_pca_variance.parquet")
    x = np.arange(1, len(pca) + 1)
    axes[1, 0].bar(x, pca["explained_variance_ratio"], color=BLUE, alpha=0.8)
    axes[1, 0].plot(
        x,
        pca["cumulative_explained_variance_ratio"],
        color=ORANGE,
        marker="o",
        markersize=3,
        label="Cumulative",
    )
    axes[1, 0].set_xlabel("Residual principal component")
    axes[1, 0].set_ylabel("Explained variance fraction")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title("20-AA residual dimensionality")
    axes[1, 0].legend(frameon=False)

    strata = pd.read_parquet(mechanisms / "stratified_summary.parquet")
    strata = strata.loc[
        strata["dimension"].eq("substitution_class")
        & strata["metric"].eq("spearman_increment")
        & strata["method"].isin([ROUTE_A_PRIMARY, ROUTE_B_PRIMARY])
    ].copy()
    levels = sorted(strata["level"].unique())
    y = np.arange(len(levels), dtype=float)
    for offset, method, color, marker, label in (
        (-0.12, ROUTE_A_PRIMARY, BLUE, "o", "Route A"),
        (0.12, ROUTE_B_PRIMARY, ORANGE, "s", "Route B"),
    ):
        frame = strata.set_index(["method", "level"]).loc[method].reindex(levels)
        axes[1, 1].errorbar(
            frame["estimate"],
            y + offset,
            xerr=np.vstack(
                [
                    frame["estimate"] - frame["ci_low"],
                    frame["ci_high"] - frame["estimate"],
                ]
            ),
            color=color,
            marker=marker,
            linestyle="none",
            markersize=4,
            capsize=2,
            label=label,
        )
    axes[1, 1].axvline(0, color=GRAY, linewidth=0.8, linestyle="--")
    axes[1, 1].set_yticks(y)
    axes[1, 1].set_yticklabels([_short_substitution_label(value) for value in levels])
    axes[1, 1].set_xlabel("Spearman increment")
    axes[1, 1].set_title("Mutation-class heterogeneity")
    axes[1, 1].legend(frameon=False)
    for index, axis in enumerate(axes.flat):
        axis.text(
            -0.16,
            1.08,
            ascii_uppercase[index],
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )
    _save(fig, output / "counterfactuals_mechanism")
    intensity["figure_panel"] = "A"
    ood_source["figure_panel"] = "B"
    pca["figure_panel"] = "C"
    strata["figure_panel"] = "D"
    return pd.concat([intensity, ood_source, pca, strata], ignore_index=True, sort=False)


def _forest(
    axis: Any,
    rows: pd.DataFrame,
    metric: str,
    methods: list[str],
    labels: dict[str, str],
    xlabel: str,
) -> None:
    frame = rows.loc[rows["metric"].eq(metric)].set_index("method").reindex(methods)
    y = np.arange(len(methods))[::-1]
    for index, method in enumerate(methods):
        item = frame.loc[method]
        route_a = method.startswith("direct_")
        primary = method.endswith("contact_rewired_5")
        axis.errorbar(
            item["estimate"],
            y[index],
            xerr=np.asarray(
                [[item["estimate"] - item["ci_low"]], [item["ci_high"] - item["estimate"]]]
            ),
            color=BLUE if route_a else ORANGE,
            marker="o" if primary else "s",
            fillstyle="full" if primary else "none",
            linestyle="none",
            markersize=5,
            capsize=3,
        )
    axis.axvline(0, color=GRAY, linewidth=0.8, linestyle="--")
    axis.set_yticks(y)
    axis.set_yticklabels([labels[method] for method in methods])
    axis.set_xlabel(xlabel)
    axis.set_title("Domain-equal increment (95% CI)")


def _domain_scatter(
    axis: Any,
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    xlabel: str,
    ylabel: str,
) -> None:
    for stratum, color, marker in (
        ("natural", GREEN, "o"),
        ("de_novo", PURPLE, "s"),
    ):
        selected = frame.loc[frame["stratum"].eq(stratum)]
        axis.scatter(
            selected[x_column],
            selected[y_column],
            s=18,
            color=color,
            marker=marker,
            alpha=0.75,
            edgecolors="none",
            label=stratum.replace("_", " "),
        )
    finite = frame[[x_column, y_column]].to_numpy(dtype=float)
    bound = float(np.nanmax(np.abs(finite))) * 1.08
    axis.plot([-bound, bound], [-bound, bound], color=GRAY, linewidth=0.8, linestyle=":")
    axis.axhline(0, color=GRAY, linewidth=0.6, linestyle="--")
    axis.axvline(0, color=GRAY, linewidth=0.6, linestyle="--")
    axis.set_xlim(-bound, bound)
    axis.set_ylim(-bound, bound)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.legend(frameon=False)


def _short_substitution_label(value: str) -> str:
    return {
        "charge_reversal": "Charge reversal",
        "hydrophobic_to_nonhydrophobic": "Hydrophobic → other",
        "involves_glycine_or_proline": "Involves Gly/Pro",
        "nonhydrophobic_to_hydrophobic": "Other → hydrophobic",
        "other_polar_or_charged": "Other polar/charged",
        "within_hydrophobic": "Within hydrophobic",
    }.get(value, value.replace("_", " "))


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _save(figure: Any, path: Path) -> None:
    for suffix in ("pdf", "svg", "png"):
        figure.savefig(
            path.with_suffix(f".{suffix}"),
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
    plt.close(figure)
