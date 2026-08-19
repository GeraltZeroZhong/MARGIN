"""Publication-ready figures for the action-validation study G/C/U study."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from margin.provenance import write_csv, write_text
from margin.studies.action_validation.config import ActionValidationStudyConfig

COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#6B7280",
    "light_gray": "#D1D5DB",
    "black": "#111827",
}
TEACHER_LABELS = {
    "mif": "Plain MIF",
    "esm_if1": "ESM-IF1",
    "proteinmpnn": "ProteinMPNN",
    "consensus": "Consensus",
}
TEACHER_COLORS = {
    "mif": COLORS["orange"],
    "esm_if1": COLORS["green"],
    "proteinmpnn": COLORS["purple"],
    "consensus": COLORS["blue"],
}


def build_action_validation_figures(config: ActionValidationStudyConfig) -> dict[str, Path]:
    """Build two Nature-width multi-panel figures and their source data."""

    evaluation = config.paths.run_dir / "evaluation"
    output = config.paths.run_dir / "figures"
    source = output / "source_data"
    output.mkdir(parents=True, exist_ok=True)
    source.mkdir(parents=True, exist_ok=True)
    summary = pd.read_parquet(evaluation / "u_margin_summary.parquet")
    components = pd.read_parquet(evaluation / "component_margin_summary.parquet")
    metrics = pd.read_parquet(evaluation / "domain_metrics.parquet")
    agreement = pd.read_parquet(evaluation / "teacher_agreement_summary.parquet")
    subgroup = pd.read_parquet(evaluation / "subgroup_margin_summary.parquet")
    routing = pd.read_parquet(evaluation / "routing_diagnostic_summary.parquet")
    margins = pd.read_parquet(evaluation / "component_margins.parquet")
    _apply_style()

    figure1_data = _figure1(summary, components, metrics, agreement, output)
    figure2_data = _figure2(subgroup, routing, margins, output)
    figure1_source = source / "action_validation_figure1_source_data.csv"
    figure2_source = source / "action_validation_figure2_source_data.csv"
    write_csv(figure1_source, figure1_data)
    write_csv(figure2_source, figure2_data)
    captions = output / "figure_captions.md"
    write_text(captions, _captions())
    return {
        "figure1_png": output / "action_validation_figure1_unique_action.png",
        "figure1_pdf": output / "action_validation_figure1_unique_action.pdf",
        "figure1_svg": output / "action_validation_figure1_unique_action.svg",
        "figure1_source": figure1_source,
        "figure2_png": output / "action_validation_figure2_boundaries_routing.png",
        "figure2_pdf": output / "action_validation_figure2_boundaries_routing.pdf",
        "figure2_svg": output / "action_validation_figure2_boundaries_routing.svg",
        "figure2_source": figure2_source,
        "captions": captions,
    }


def _figure1(
    summary: pd.DataFrame,
    components: pd.DataFrame,
    metrics: pd.DataFrame,
    agreement: pd.DataFrame,
    output: Path,
) -> pd.DataFrame:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.flat
    source_frames = []

    selected = summary.loc[
        summary["metric"].eq("spearman_margin") & summary["stratum"].eq("all")
    ].copy()
    teachers = ["consensus", "proteinmpnn", "esm_if1", "mif"]
    y = np.arange(len(teachers))
    for offset, (population, label, marker, color) in enumerate(
        [
            ("megascale_dense", "Megascale dense", "o", COLORS["blue"]),
            (
                "s669_sparse_cross_platform",
                "S669 sparse",
                "s",
                COLORS["vermillion"],
            ),
        ]
    ):
        frame = selected.loc[selected["evaluation_population"].eq(population)].set_index(
            "teacher_id"
        )
        values = frame.loc[teachers]
        positions = y + (-0.11 if offset == 0 else 0.11)
        ax_a.errorbar(
            values["estimate"],
            positions,
            xerr=np.vstack(
                [
                    values["estimate"] - values["ci_low"],
                    values["ci_high"] - values["estimate"],
                ]
            ),
            fmt=marker,
            color=color,
            markerfacecolor=color if offset == 0 else "white",
            markeredgecolor=color,
            capsize=2,
            label=label,
        )
        source_frames.append(values.reset_index().assign(panel="a", series=label))
    ax_a.axvline(0, color=COLORS["gray"], lw=0.8, ls="--")
    ax_a.set_yticks(y, [TEACHER_LABELS[item] for item in teachers])
    ax_a.set_xlabel("U Spearman margin")
    ax_a.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.55, -0.15),
        ncol=2,
    )

    stage_labels = {"plus_g": "+G", "plus_gc": "+C", "plus_gcu": "+U"}
    stage_order = ["plus_g", "plus_gc", "plus_gcu"]
    frame = (
        components.loc[
            components["teacher_id"].eq("consensus")
            & components["evaluation_population"].eq("megascale_dense")
            & components["stratum"].eq("all")
            & components["metric"].eq("spearman_margin")
            & components["stage"].isin(stage_order)
        ]
        .set_index("stage")
        .loc[stage_order]
    )
    stage_colors = [COLORS["orange"], COLORS["green"], COLORS["blue"]]
    for index, stage in enumerate(stage_order):
        row = frame.loc[stage]
        ax_b.errorbar(
            row["estimate"],
            index,
            xerr=np.asarray(
                [[row["estimate"] - row["ci_low"]], [row["ci_high"] - row["estimate"]]]
            ),
            fmt="o",
            color=stage_colors[index],
            capsize=2,
        )
    ax_b.axvline(0, color=COLORS["gray"], lw=0.8, ls="--")
    ax_b.set_yticks(np.arange(3), [stage_labels[item] for item in stage_order])
    ax_b.set_xlabel("Sequential Spearman margin")
    source_frames.append(frame.reset_index().assign(panel="b", series="dense_consensus"))

    agreement_metrics = [
        "mif_vs_esm_if1_u_spearman",
        "mif_vs_proteinmpnn_u_spearman",
        "esm_if1_vs_proteinmpnn_u_spearman",
        "median_pairwise_u_spearman",
    ]
    agreement_labels = [
        "MIF / ESM-IF1",
        "MIF / ProteinMPNN",
        "ESM-IF1 / ProteinMPNN",
        "Median pair",
    ]
    dense_agreement = (
        agreement.loc[agreement["evaluation_population"].eq("megascale_dense")]
        .set_index("metric")
        .loc[agreement_metrics]
    )
    for index, metric in enumerate(agreement_metrics):
        row = dense_agreement.loc[metric]
        ax_c.errorbar(
            row["estimate"],
            index,
            xerr=np.asarray(
                [[row["estimate"] - row["ci_low"]], [row["ci_high"] - row["estimate"]]]
            ),
            fmt="D" if metric == "median_pairwise_u_spearman" else "o",
            color=COLORS["blue"] if metric == "median_pairwise_u_spearman" else COLORS["gray"],
            capsize=2,
        )
    ax_c.axvline(0, color=COLORS["gray"], lw=0.8, ls="--")
    ax_c.set_yticks(np.arange(4), agreement_labels)
    ax_c.set_xlabel("Cross-teacher U Spearman")
    source_frames.append(dense_agreement.reset_index().assign(panel="c", series="megascale_dense"))

    method_order = [
        "sequence_only",
        "consensus__plus_g",
        "consensus__plus_gc",
        "consensus__plus_gcu",
    ]
    method_labels = ["Sequence", "+G", "+G+C", "+G+C+U"]
    dense_metrics = metrics.loc[
        metrics["evaluation_population"].eq("megascale_dense")
        & metrics["method"].isin(method_order)
    ]
    values = [
        dense_metrics.loc[dense_metrics["method"].eq(method), "spearman"].to_numpy()
        for method in method_order
    ]
    box = ax_d.boxplot(
        values,
        positions=np.arange(4),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": COLORS["black"], "linewidth": 1.1},
        whiskerprops={"color": COLORS["gray"]},
        capprops={"color": COLORS["gray"]},
    )
    for patch, color in zip(box["boxes"], stage_colors[:1] + stage_colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)
        patch.set_edgecolor(color)
    rng = np.random.default_rng(20260818)
    for index, data in enumerate(values):
        jitter = rng.uniform(-0.12, 0.12, len(data))
        ax_d.scatter(
            np.full(len(data), index) + jitter,
            data,
            s=9,
            color=COLORS["black"],
            alpha=0.38,
            linewidth=0,
        )
    ax_d.axhline(0, color=COLORS["gray"], lw=0.8, ls="--")
    ax_d.set_xticks(np.arange(4), method_labels, rotation=18, ha="right")
    ax_d.set_ylabel("Domain Spearman")
    source_frames.append(
        dense_metrics.loc[dense_metrics["method"].isin(method_order)].assign(
            panel="d", series="domain_values"
        )
    )

    _panel_labels([ax_a, ax_b, ax_c, ax_d])
    _save(fig, output / "action_validation_figure1_unique_action")
    plt.close(fig)
    return pd.concat(source_frames, ignore_index=True, sort=False)


def _figure2(
    subgroup: pd.DataFrame,
    routing: pd.DataFrame,
    margins: pd.DataFrame,
    output: Path,
) -> pd.DataFrame:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.3), constrained_layout=True)
    ax_a, ax_b, ax_c = axes
    source_frames = []

    selected_levels = [
        ("gly_pro_boundary", "involves_glycine_or_proline", "Gly/Pro"),
        ("gly_pro_boundary", "other_substitutions", "Other"),
        ("burial", "buried", "Buried"),
        ("burial", "intermediate", "Intermediate"),
        ("burial", "exposed", "Exposed"),
        ("contact_class", "high_contact", "High contact"),
        ("contact_class", "low_contact", "Low contact"),
        ("secondary_structure", "helix", "Helix"),
        ("secondary_structure", "strand", "Strand"),
        ("secondary_structure", "turn_or_coil", "Turn/coil"),
    ]
    dense_subgroup = subgroup.loc[
        subgroup["teacher_id"].eq("consensus")
        & subgroup["evaluation_population"].eq("megascale_dense")
        & subgroup["metric"].eq("spearman_margin")
    ].set_index(["dimension", "level"])
    labels = []
    subgroup_rows = []
    for index, (dimension, level, label) in enumerate(selected_levels):
        row = dense_subgroup.loc[(dimension, level)]
        labels.append(label)
        subgroup_rows.append(row)
        ax_a.errorbar(
            row["estimate"],
            index,
            xerr=np.asarray(
                [[row["estimate"] - row["ci_low"]], [row["ci_high"] - row["estimate"]]]
            ),
            fmt="o",
            color=COLORS["blue"],
            capsize=2,
        )
    ax_a.axvline(0, color=COLORS["gray"], lw=0.8, ls="--")
    ax_a.set_yticks(np.arange(len(labels)), labels)
    ax_a.set_xlabel("Consensus U Spearman margin")
    subgroup_source = pd.DataFrame(subgroup_rows).reset_index(drop=True)
    subgroup_source["dimension"] = [item[0] for item in selected_levels]
    subgroup_source["level"] = [item[1] for item in selected_levels]
    source_frames.append(subgroup_source.assign(panel="a", series="dense_subgroups"))

    route_metrics = ["gated_minus_gc_spearman", "gated_minus_full_spearman"]
    route_labels = ["Gated U vs G+C", "Gated U vs full U"]
    route_markers = {"megascale_dense": "o", "s669_sparse_cross_platform": "s"}
    route_colors = {
        "megascale_dense": COLORS["blue"],
        "s669_sparse_cross_platform": COLORS["vermillion"],
    }
    for population_index, population in enumerate(route_markers):
        frame = (
            routing.loc[
                routing["evaluation_population"].eq(population)
                & routing["metric"].isin(route_metrics)
            ]
            .set_index("metric")
            .loc[route_metrics]
        )
        positions = np.arange(2) + (-0.10 if population_index == 0 else 0.10)
        ax_b.errorbar(
            frame["estimate"],
            positions,
            xerr=np.vstack(
                [
                    frame["estimate"] - frame["ci_low"],
                    frame["ci_high"] - frame["estimate"],
                ]
            ),
            fmt=route_markers[population],
            color=route_colors[population],
            markerfacecolor=route_colors[population] if population_index == 0 else "white",
            capsize=2,
            label="Megascale dense" if population_index == 0 else "S669 sparse",
        )
        source_frames.append(frame.reset_index().assign(panel="b", series=population))
    ax_b.axvline(0, color=COLORS["gray"], lw=0.8, ls="--")
    ax_b.set_yticks(np.arange(2), route_labels)
    ax_b.set_xlabel("Spearman difference")
    ax_b.legend(frameon=False, loc="center right")

    domain = margins.loc[
        margins["teacher_id"].eq("consensus") & margins["stage"].eq("plus_gcu")
    ].copy()
    domain = domain.sort_values(
        ["evaluation_population", "stratum", "spearman_margin"], kind="stable"
    ).reset_index(drop=True)
    color_lookup = {
        "natural": COLORS["blue"],
        "de_novo": COLORS["orange"],
        "s669_natural": COLORS["purple"],
    }
    marker_lookup = {"natural": "o", "de_novo": "^", "s669_natural": "s"}
    for stratum, frame in domain.groupby("stratum", sort=False):
        ax_c.scatter(
            frame.index,
            frame["spearman_margin"],
            s=18,
            marker=marker_lookup[stratum],
            color=color_lookup[stratum],
            label={
                "natural": "natural",
                "de_novo": "de novo",
                "s669_natural": "S669 natural",
            }[stratum],
        )
    ax_c.axhline(0, color=COLORS["gray"], lw=0.8, ls="--")
    boundary = int(domain["evaluation_population"].eq("megascale_dense").sum()) - 0.5
    ax_c.axvline(boundary, color=COLORS["light_gray"], lw=0.8)
    ax_c.set_xlabel("Domains (ordered within stratum)")
    ax_c.set_ylabel("Consensus U Spearman margin")
    ax_c.set_xticks([])
    ax_c.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        columnspacing=0.8,
    )
    source_frames.append(domain.assign(panel="c", series="domain_u_margin"))

    _panel_labels([ax_a, ax_b, ax_c])
    _save(fig, output / "action_validation_figure2_boundaries_routing")
    plt.close(fig)
    return pd.concat(source_frames, ignore_index=True, sort=False)


def _apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.7,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _panel_labels(axes: list[plt.Axes]) -> None:
    labels = "abcdefghijklmnopqrstuvwxyz"[: len(axes)]
    for label, axis in zip(labels, axes, strict=True):
        axis.text(
            -0.16,
            1.05,
            label,
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
            va="top",
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)


def _save(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")


def _captions() -> str:
    return """# action-validation study figure captions

## Figure 1 | Paired structure retains multi-teacher unique action value

**a,** Domain-equal Spearman margin of the structure-unique component U over the G+C
baseline for three inverse-folding teachers and their outcome-free calibrated consensus.
Points are means and bars are stratified domain-bootstrap 95% confidence intervals;
Megascale has 32 dense domains and S669 has 8 sparse domains. **b,** Sequential consensus
increments from global substitution G, sequence context C, and structure-unique U on the
dense panel. **c,** Cross-teacher candidate-action Spearman agreement after G/C removal.
**d,** Per-domain Spearman distributions across the four registered score stages; points
show individual domains and boxes show the median and interquartile range.

## Figure 2 | Boundary analyses and the unresolved routing problem

**a,** Dense-panel consensus U Spearman margins within predeclared mutation and structural
environments. Points are equal-domain means and bars are 95% bootstrap intervals. **b,** A
fixed teacher-agreement gate improves over G+C but underperforms using full U; the S669
intervals are sparse directional estimates. **c,** Domainwise consensus U margins across
the dense natural, dense de novo, and sparse S669 strata. The vertical divider separates
Megascale from S669. No boundary or routing diagnostic enters the primary confirmation
gate.
"""
