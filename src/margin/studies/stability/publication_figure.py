"""Publication figure for the stability study post-lock submission audit."""

from __future__ import annotations

from pathlib import Path
from string import ascii_lowercase

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_publication_audit_figure(
    tables: dict[str, pd.DataFrame], figure_dir: Path
) -> dict[str, Path]:
    """Render the external confirmation and structure-boundary evidence."""

    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        extension: figure_dir / f"figure4_submission_audit.{extension}"
        for extension in ("pdf", "svg", "png")
    }
    _set_style()
    figure = plt.figure(figsize=(7.2, 6.4))
    axes = [
        figure.add_axes([0.17, 0.10, 0.23, 0.82]),
        figure.add_axes([0.61, 0.59, 0.36, 0.33]),
        figure.add_axes([0.51, 0.10, 0.19, 0.34]),
        figure.add_axes([0.78, 0.10, 0.19, 0.34]),
    ]
    _fireprot_domains(axes[0], tables["fireprot_domain_results"])
    _fireprot_methods(axes[1], tables["fireprot_method_summary"])
    _fast_robust(axes[2], tables["fast_robust_domain_contrasts"], tables["fast_robust_summary"])
    _geometry_performance(
        axes[3],
        tables["structure_sensitivity_geometry_summary"],
        tables["structure_sensitivity_teacher_delta_summary"],
    )
    for label, axis in zip(ascii_lowercase[: len(axes)], axes, strict=True):
        axis.text(
            -0.13,
            1.05,
            label,
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )
    for extension, path in paths.items():
        figure.savefig(
            path,
            dpi=450 if extension == "png" else None,
            facecolor="white",
        )
    plt.close(figure)
    return paths


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _fireprot_domains(axis, table: pd.DataFrame) -> None:
    frame = table.sort_values("spearman_margin").reset_index(drop=True)
    colors = np.where(frame["spearman_margin"].ge(0), "#0072B2", "#D55E00")
    axis.axvline(0, color="#555555", linewidth=0.8, linestyle="--", zorder=1)
    axis.scatter(
        frame["spearman_margin"],
        np.arange(len(frame)),
        c=colors,
        s=22,
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )
    axis.set_yticks(
        np.arange(len(frame)),
        [f"{row.domain_id}  ({row.n_variants})" for row in frame.itertuples(index=False)],
    )
    axis.set_xlabel(r"$\Delta$Spearman: action − G+C+")
    axis.set_title("Per-protein FireProt confirmation", loc="left", fontweight="bold")
    axis.text(
        0.02,
        0.99,
        "17/18 positive; labels show variant count",
        transform=axis.transAxes,
        va="top",
        fontsize=6.2,
        color="#555555",
    )
    _clean(axis, grid_axis="x")


def _fireprot_methods(axis, table: pd.DataFrame) -> None:
    order = [
        ("temperature_consensus_g_plus_c_plus", "G+C+ control", "#CC79A7", "s"),
        ("esm_if1_action", "ESM-IF1", "#E69F00", "^"),
        ("mif_action", "MIF", "#0072B2", "D"),
        ("proteinmpnn_action", "ProteinMPNN", "#009E73", "v"),
        ("temperature_consensus_action", "Temperature consensus", "#000000", "o"),
        ("unscaled_consensus_action", "Unscaled consensus", "#D55E00", "o"),
    ]
    selected = table.loc[table["metric"].eq("spearman")].set_index("method")
    for y, (method, _label, color, marker) in enumerate(order):
        row = selected.loc[method]
        axis.errorbar(
            row["estimate"],
            y,
            xerr=[[row["estimate"] - row["ci_low"]], [row["ci_high"] - row["estimate"]]],
            fmt=marker,
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=5,
            capsize=2,
            elinewidth=1,
        )
    axis.set_yticks(np.arange(len(order)), [label for _, label, _, _ in order])
    axis.set_xlim(0.24, 0.62)
    axis.set_xlabel("Equal-protein Spearman (95% bootstrap CI)")
    axis.set_title("FireProt method performance", loc="left", fontweight="bold")
    _clean(axis, grid_axis="x")


def _fast_robust(axis, domain: pd.DataFrame, summary: pd.DataFrame) -> None:
    panels = ["Megascale-32", "FireProt-18"]
    colors = ["#0072B2", "#D55E00"]
    rng = np.random.default_rng(20260825)
    for x, (panel, color) in enumerate(zip(panels, colors, strict=True)):
        values = domain.loc[domain["panel"].eq(panel), "spearman_margin"].to_numpy(float)
        jitter = rng.uniform(-0.13, 0.13, size=len(values))
        axis.scatter(
            np.full(len(values), x) + jitter,
            values,
            s=9,
            facecolor="none",
            edgecolor=color,
            linewidth=0.7,
            alpha=0.75,
            zorder=2,
        )
        row = summary.loc[
            summary["panel"].eq(panel)
            & summary["scope"].eq("all")
            & summary["metric"].eq("spearman_margin")
        ].iloc[0]
        axis.errorbar(
            x,
            row["estimate"],
            yerr=[[row["estimate"] - row["ci_low"]], [row["ci_high"] - row["estimate"]]],
            fmt="o",
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=5.5,
            capsize=3,
            elinewidth=1.3,
            zorder=4,
        )
    axis.axhline(0, color="#555555", linewidth=0.8, linestyle="--", zorder=1)
    axis.set_xticks([0, 1], ["Megascale\n(n=32)", "FireProt\n(n=18)"])
    axis.set_ylabel(r"$\Delta$Spearman: robust − fast")
    axis.set_title("Robust-tier gain", loc="left", fontweight="bold")
    _clean(axis, grid_axis="y")


def _geometry_performance(axis, geometry: pd.DataFrame, deltas: pd.DataFrame) -> None:
    roles = [
        ("alphafold", "AlphaFold DB", "#56B4E9", "o"),
        ("perturbed_0p5", "+0.5 Å", "#E69F00", "s"),
        ("perturbed_1p0", "+1.0 Å", "#D55E00", "^"),
    ]
    geometry = geometry.loc[geometry["metric"].eq("local_backbone_rmsd_10a")].set_index(
        "structure_role"
    )
    delta = deltas.loc[
        deltas["teacher_id"].eq("registered_temperature_consensus")
        & deltas["metric"].eq("action_spearman_delta_vs_experimental")
    ].set_index("structure_role")
    for role, label, color, marker in roles:
        x = geometry.loc[role]
        y = delta.loc[role]
        axis.errorbar(
            x["estimate"],
            y["estimate"],
            xerr=[[x["estimate"] - x["ci_low"]], [x["ci_high"] - x["estimate"]]],
            yerr=[[y["estimate"] - y["ci_low"]], [y["ci_high"] - y["estimate"]]],
            fmt=marker,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=6,
            capsize=2,
            elinewidth=1,
            label=label,
        )
    axis.axhline(0, color="#555555", linewidth=0.8, linestyle="--")
    axis.set_xlabel("Local backbone RMSD (Å)")
    axis.set_ylabel(r"$\Delta$Spearman vs experimental")
    axis.set_title("Structure sensitivity", loc="left", fontweight="bold")
    axis.legend(frameon=False, loc="lower left", handletextpad=0.4)
    _clean(axis, grid_axis="both")


def _clean(axis, *, grid_axis: str) -> None:
    axis.grid(axis=grid_axis, color="#E2E2E2", linewidth=0.55, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
