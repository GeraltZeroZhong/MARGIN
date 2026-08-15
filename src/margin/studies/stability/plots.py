"""Publication-ready stability study figures and figure source data."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from margin.provenance import write_csv
from margin.studies.counterfactuals.evaluation import stratified_domain_bootstrap
from margin.studies.stability.config import StabilityStudyConfig
from margin.studies.stability.evaluation import CPLUS_METHOD, SELECTED_METHOD, SEQUENCE_METHOD
from margin.studies.stability.prepare import EXTERNAL_POPULATION, PRIMARY_POPULATION

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#6B7280"
LIGHT_GRAY = "#D1D5DB"


def build_stability_figures(config: StabilityStudyConfig) -> dict[str, Path]:
    """Create two multi-panel figures as SVG, PDF, and 300-dpi PNG."""

    _style()
    evaluation = config.paths.run_dir / "evaluation"
    figures = config.paths.run_dir / "figures"
    source = figures / "source_data"
    figures.mkdir(parents=True, exist_ok=True)
    source.mkdir(parents=True, exist_ok=True)
    calibration = pd.read_parquet(
        config.paths.run_dir / "calibration" / "scheme_validation.parquet"
    )
    metrics = pd.read_parquet(evaluation / "domain_metrics.parquet")
    summary = pd.read_parquet(evaluation / "contrast_summary.parquet")
    cath_audit = pd.read_parquet(
        config.paths.run_dir / "strong_control" / "locked_cath_audit.parquet"
    )
    subgroup = pd.read_parquet(evaluation / "subgroup_domain_metrics.parquet")
    source_tables = _source_tables(calibration, metrics, summary, cath_audit, subgroup, config)
    for name, table in source_tables.items():
        write_csv(source / f"{name}.csv", table)
    figure1 = figures / "figure1_paired_action"
    _figure1(source_tables, figure1)
    figure2 = figures / "figure2_strong_control"
    _figure2(source_tables, figure2)
    return {
        "figure1_png": figure1.with_suffix(".png"),
        "figure1_pdf": figure1.with_suffix(".pdf"),
        "figure1_svg": figure1.with_suffix(".svg"),
        "figure2_png": figure2.with_suffix(".png"),
        "figure2_pdf": figure2.with_suffix(".pdf"),
        "figure2_svg": figure2.with_suffix(".svg"),
        "source_data": source,
    }


def _source_tables(
    calibration: pd.DataFrame,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    cath_audit: pd.DataFrame,
    subgroup: pd.DataFrame,
    config: StabilityStudyConfig,
) -> dict[str, pd.DataFrame]:
    primary = metrics.loc[metrics["evaluation_population"].eq(PRIMARY_POPULATION)]
    paired = (
        primary.loc[primary["method"].isin([SEQUENCE_METHOD, SELECTED_METHOD])]
        .pivot(
            index=["domain_id", "stratum"],
            columns="method",
            values="spearman",
        )
        .reset_index()
        .rename(
            columns={
                SEQUENCE_METHOD: "sequence_spearman",
                SELECTED_METHOD: "selected_spearman",
            }
        )
    )
    contrast_order = [
        "mif_vs_sequence",
        "esm_if1_vs_sequence",
        "proteinmpnn_vs_sequence",
        "unscaled_consensus_vs_sequence",
        "selected_consensus_vs_sequence",
        "rms_consensus_vs_sequence",
        "rank_consensus_vs_sequence",
    ]
    teacher_margins = summary.loc[
        summary["evaluation_population"].eq(PRIMARY_POPULATION)
        & summary["stratum"].eq("all")
        & summary["metric"].eq("spearman")
        & summary["contrast"].isin(contrast_order),
        ["contrast", "estimate", "ci_low", "ci_high"],
    ].copy()
    teacher_margins["order"] = teacher_margins["contrast"].map(
        {value: index for index, value in enumerate(contrast_order)}
    )
    teacher_margins = teacher_margins.sort_values("order").drop(columns="order")
    external = summary.loc[
        summary["evaluation_population"].eq(EXTERNAL_POPULATION)
        & summary["contrast"].eq("selected_consensus_vs_sequence")
        & summary["metric"].isin(["spearman", "ndcg_at_10_percent"]),
        ["metric", "estimate", "ci_low", "ci_high", "interval_unit"],
    ].copy()
    cplus = summary.loc[
        summary["evaluation_population"].eq(PRIMARY_POPULATION)
        & summary["contrast"].eq("selected_consensus_vs_Cplus")
        & summary["metric"].isin(["spearman", "ndcg_at_10_percent"]),
        ["stratum", "metric", "estimate", "ci_low", "ci_high"],
    ].copy()
    subgroup_margin = _subgroup_margins(subgroup, config)
    return {
        "calibration_validation": calibration,
        "primary_domain_spearman": paired,
        "primary_spearman_margins": teacher_margins,
        "external_selected_margins": external,
        "locked_cath_control": cath_audit,
        "cplus_primary_margins": cplus,
        "subgroup_cplus_spearman_margins": subgroup_margin,
    }


def _subgroup_margins(subgroup: pd.DataFrame, config: StabilityStudyConfig) -> pd.DataFrame:
    selected = subgroup.loc[
        subgroup["evaluation_population"].eq(PRIMARY_POPULATION)
        & subgroup["method"].isin([SELECTED_METHOD, CPLUS_METHOD])
    ]
    keys = ["dimension", "level", "domain_id", "stratum"]
    pivot = selected.pivot(index=keys, columns="method", values="spearman").reset_index()
    pivot = pivot.dropna(subset=[SELECTED_METHOD, CPLUS_METHOD])
    pivot["spearman_margin"] = pivot[SELECTED_METHOD] - pivot[CPLUS_METHOD]
    rows = []
    for group_index, ((dimension, level), frame) in enumerate(
        pivot.groupby(["dimension", "level"], sort=True, observed=True)
    ):
        rows.append(
            {
                "dimension": dimension,
                "level": level,
                **stratified_domain_bootstrap(
                    frame,
                    "spearman_margin",
                    replicates=config.inference.bootstrap_replicates,
                    confidence_level=config.inference.confidence_level,
                    seed=config.seed + 400_000 + group_index,
                ),
            }
        )
    return pd.DataFrame(rows)


def _figure1(tables: dict[str, pd.DataFrame], stem: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(7.09, 5.6), constrained_layout=True)
    calibration = tables["calibration_validation"].copy()
    labels = {
        "joint_temperature_native_nll": "Temperature",
        "unscaled_equal": "Unscaled",
        "action_rms_matched": "RMS",
        "rowwise_rank_normalized": "Rank",
    }
    calibration["label"] = calibration["scheme"].map(labels)
    calibration = calibration.sort_values("native_nll")
    colors = [BLUE if value == "Temperature" else LIGHT_GRAY for value in calibration["label"]]
    axes[0, 0].bar(calibration["label"], calibration["native_nll"], color=colors)
    axes[0, 0].set_ylabel("CATH validation native NLL")
    axes[0, 0].set_ylim(1.40, 1.70)
    axes[0, 0].tick_params(axis="x", rotation=25)
    _panel_label(axes[0, 0], "a")

    paired = tables["primary_domain_spearman"]
    for stratum, color in (("natural", BLUE), ("de_novo", ORANGE)):
        frame = paired.loc[paired["stratum"].eq(stratum)]
        for row in frame.itertuples(index=False):
            axes[0, 1].plot(
                [0, 1],
                [row.sequence_spearman, row.selected_spearman],
                color=color,
                alpha=0.35,
                linewidth=0.8,
            )
        axes[0, 1].scatter(
            np.zeros(len(frame)),
            frame["sequence_spearman"],
            s=12,
            color=color,
            label=stratum.replace("_", " "),
            zorder=3,
        )
        axes[0, 1].scatter(
            np.ones(len(frame)), frame["selected_spearman"], s=12, color=color, zorder=3
        )
    axes[0, 1].set_xticks([0, 1], ["Sequence", "Paired action"])
    axes[0, 1].set_ylabel("Domain Spearman")
    axes[0, 1].legend(frameon=False, fontsize=7, loc="lower right")
    _panel_label(axes[0, 1], "b")

    margins = tables["primary_spearman_margins"]
    display = {
        "mif_vs_sequence": "MIF",
        "esm_if1_vs_sequence": "ESM-IF1",
        "proteinmpnn_vs_sequence": "ProteinMPNN",
        "unscaled_consensus_vs_sequence": "Unscaled consensus",
        "selected_consensus_vs_sequence": "Temperature consensus",
        "rms_consensus_vs_sequence": "RMS consensus",
        "rank_consensus_vs_sequence": "Rank consensus",
    }
    y = np.arange(len(margins))[::-1]
    colors = [
        GREEN if value == "selected_consensus_vs_sequence" else GRAY
        for value in margins["contrast"]
    ]
    axes[1, 0].errorbar(
        margins["estimate"],
        y,
        xerr=np.vstack(
            [margins["estimate"] - margins["ci_low"], margins["ci_high"] - margins["estimate"]]
        ),
        fmt="none",
        ecolor=GRAY,
        elinewidth=1.2,
        capsize=2,
    )
    axes[1, 0].scatter(margins["estimate"], y, c=colors, s=22, zorder=3)
    axes[1, 0].axvline(0, color="#111827", linewidth=0.7)
    axes[1, 0].set_yticks(y, [display[value] for value in margins["contrast"]])
    axes[1, 0].set_xlabel("Spearman increment over sequence")
    _panel_label(axes[1, 0], "c")

    external = tables["external_selected_margins"]
    x = np.arange(len(external))
    bars = axes[1, 1].bar(x, external["estimate"], color=[BLUE, ORANGE])
    spearman = external["metric"].eq("spearman")
    axes[1, 1].errorbar(
        x[spearman],
        external.loc[spearman, "estimate"],
        yerr=np.vstack(
            [
                external.loc[spearman, "estimate"] - external.loc[spearman, "ci_low"],
                external.loc[spearman, "ci_high"] - external.loc[spearman, "estimate"],
            ]
        ),
        fmt="none",
        ecolor="#111827",
        capsize=3,
        linewidth=1,
    )
    axes[1, 1].set_xticks(x, ["Spearman", "NDCG@10%"])
    axes[1, 1].set_ylabel("ESTA increment over sequence")
    axes[1, 1].set_ylim(0, max(0.32, float(external["estimate"].max()) * 1.25))
    for bar, value in zip(bars, external["estimate"], strict=True):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    _panel_label(axes[1, 1], "d")
    _save(figure, stem)


def _figure2(tables: dict[str, pd.DataFrame], stem: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(7.09, 5.8), constrained_layout=True)
    audit = tables["locked_cath_control"]
    teachers = audit["teacher_id"].tolist()
    teacher_labels = {
        "mif": "MIF",
        "esm_if1": "ESM-IF1",
        "proteinmpnn": "ProteinMPNN",
    }
    x = np.arange(len(audit))
    width = 0.34
    axes[0, 0].bar(x - width / 2, audit["r2_g"], width, label="G", color=LIGHT_GRAY)
    axes[0, 0].bar(
        x + width / 2,
        audit["r2_g_plus_c_plus"],
        width,
        label="G + C+",
        color=BLUE,
    )
    axes[0, 0].set_xticks(x, [teacher_labels[value] for value in teachers])
    axes[0, 0].set_ylabel("Locked CATH action $R^2$")
    axes[0, 0].legend(frameon=False, fontsize=7)
    _panel_label(axes[0, 0], "a")

    bars = axes[0, 1].bar(x, audit["u_plus_over_action_rms"], color=PURPLE)
    axes[0, 1].set_xticks(x, [teacher_labels[value] for value in teachers])
    axes[0, 1].set_ylabel("RMS($U^+$) / RMS($A$)")
    axes[0, 1].set_ylim(0, 0.7)
    for bar, value in zip(bars, audit["u_plus_over_action_rms"], strict=True):
        axes[0, 1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.2f}",
            ha="center",
            fontsize=7,
        )
    _panel_label(axes[0, 1], "b")

    cplus = tables["cplus_primary_margins"].copy()
    stratum_order = ["all", "natural", "de_novo"]
    for metric, offset, color, label in (
        ("spearman", -0.12, GREEN, "Spearman"),
        ("ndcg_at_10_percent", 0.12, ORANGE, "NDCG@10%"),
    ):
        frame = cplus.loc[cplus["metric"].eq(metric)].set_index("stratum").loc[stratum_order]
        positions = np.arange(len(stratum_order)) + offset
        axes[1, 0].errorbar(
            positions,
            frame["estimate"],
            yerr=np.vstack(
                [frame["estimate"] - frame["ci_low"], frame["ci_high"] - frame["estimate"]]
            ),
            fmt="o",
            color=color,
            capsize=2,
            markersize=4,
            label=label,
        )
    axes[1, 0].axhline(0, color="#111827", linewidth=0.7)
    axes[1, 0].set_xticks(np.arange(3), ["All", "Natural", "De novo"])
    axes[1, 0].set_ylabel("Full action minus C+ increment")
    axes[1, 0].legend(frameon=False, fontsize=7)
    _panel_label(axes[1, 0], "c")

    subgroup = tables["subgroup_cplus_spearman_margins"].copy()
    subgroup["label"] = (
        subgroup["dimension"].str.replace("_", " ") + ": " + subgroup["level"].str.replace("_", " ")
    )
    subgroup = subgroup.sort_values(["dimension", "estimate"])
    y = np.arange(len(subgroup))[::-1]
    axes[1, 1].errorbar(
        subgroup["estimate"],
        y,
        xerr=np.vstack(
            [subgroup["estimate"] - subgroup["ci_low"], subgroup["ci_high"] - subgroup["estimate"]]
        ),
        fmt="o",
        color=VERMILLION,
        ecolor=GRAY,
        capsize=2,
        markersize=3.5,
    )
    axes[1, 1].axvline(0, color="#111827", linewidth=0.7)
    axes[1, 1].set_yticks(y, subgroup["label"], fontsize=6.5)
    axes[1, 1].set_xlabel("Subgroup Spearman increment over C+")
    _panel_label(axes[1, 1], "d")
    _save(figure, stem)


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(-0.16, 1.08, label, transform=axis.transAxes, fontweight="bold", fontsize=9)


def _save(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)
