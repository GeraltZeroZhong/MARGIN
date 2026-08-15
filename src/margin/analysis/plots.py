"""Generate publication-ready figures exclusively from persisted source-data tables."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_matplotlib_cache = Path(tempfile.gettempdir()) / "margin-matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from margin.config import ProjectConfig  # noqa: E402

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "sky": "#56B4E9",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
    "gray": "#777777",
}


def make_audit_figures(config: ProjectConfig) -> list[Path]:
    """Render the distillability map and audit overview from source data."""

    _set_style()
    output: list[Path] = []
    map_table = pd.read_parquet(
        config.paths.source_data_dir / "figure_1_distillability_map.parquet"
    )
    map_table = _decision_scope(map_table, config)
    output.extend(_plot_distillability_map(map_table, config))
    output.extend(_plot_audit_overview(config))
    return output


def _plot_distillability_map(table: pd.DataFrame, config: ProjectConfig) -> list[Path]:
    figure, axis = plt.subplots(
        figsize=(config.plot.width_inches, config.plot.height_inches), constrained_layout=True
    )
    if table.empty:
        _empty_axis(axis, "Distillability map", "No complete observability evidence")
    else:
        clean = table.dropna(
            subset=[
                "teacher_advantage_nats",
                "observability_jsd_reduction_nats",
                "matched_decoy_lift_nats",
                "scaffold_reliability",
            ]
        ).copy()
        if clean.empty:
            _empty_axis(axis, "Distillability map", "No rows contain all four gate dimensions")
        else:
            sizes = _marker_sizes(clean["scaffold_reliability"], config)
            scatter = axis.scatter(
                clean["teacher_advantage_nats"],
                clean["observability_jsd_reduction_nats"],
                c=clean["matched_decoy_lift_nats"],
                s=sizes,
                cmap="cividis",
                edgecolor="black",
                linewidth=0.35,
                alpha=0.82,
            )
            axis.axvline(
                config.decision.minimum_environment_advantage_nats,
                color=OKABE_ITO["gray"],
                linestyle="--",
                linewidth=0.8,
            )
            axis.axhline(
                config.decision.minimum_observability_jsd_reduction,
                color=OKABE_ITO["gray"],
                linestyle="--",
                linewidth=0.8,
            )
            colorbar = figure.colorbar(scatter, ax=axis, pad=0.02)
            colorbar.set_label("Paired–matched-decoy lift (nats)")
            high_value_observable = clean.loc[
                (
                    clean["teacher_advantage_nats"]
                    >= config.decision.minimum_environment_advantage_nats
                )
                & (
                    clean["observability_jsd_reduction_nats"]
                    >= config.decision.minimum_observability_jsd_reduction
                )
            ]
            representative = pd.concat(
                [
                    high_value_observable.nlargest(1, "teacher_advantage_nats"),
                    high_value_observable.nlargest(1, "observability_jsd_reduction_nats"),
                ]
            ).drop_duplicates()
            offsets = [(-10, -28), (10, -28)]
            alignments = ["right", "left"]
            for index, row in enumerate(representative.itertuples(index=False)):
                axis.annotate(
                    f"{row.environment.replace('_', ' ')} · "
                    f"{row.state_kind.replace('_', ' ')} · "
                    f"{row.requested_corruption_ratio:.0%}",
                    (row.teacher_advantage_nats, row.observability_jsd_reduction_nats),
                    xytext=offsets[index],
                    textcoords="offset points",
                    ha=alignments[index],
                    va="top",
                    fontsize=6,
                    arrowprops={"arrowstyle": "-", "color": "0.35", "linewidth": 0.5},
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1},
                )
            axis.set_title("foundation audit distillability map")
            axis.set_xlabel("Teacher action value (native-NLL reduction, nats)")
            axis.set_ylabel("Sequence observability (JSD reduction, nats)")
            axis.text(
                0.99,
                0.02,
                "Marker area: scaffold reliability",
                ha="right",
                va="bottom",
                transform=axis.transAxes,
                fontsize=7,
                color=OKABE_ITO["gray"],
            )
    return _save(figure, config.paths.figure_dir / "figure_1_distillability_map", config)


def _plot_audit_overview(config: ProjectConfig) -> list[Path]:
    paired = pd.read_parquet(config.paths.source_data_dir / "figure_2_paired_decoy.parquet")
    observable = pd.read_parquet(config.paths.source_data_dir / "figure_3_observability.parquet")
    on_policy = pd.read_parquet(config.paths.source_data_dir / "figure_4_on_policy.parquet")
    gate = pd.read_parquet(config.paths.source_data_dir / "decision_criteria.parquet")
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(config.plot.width_inches, config.plot.height_inches * 1.35),
        constrained_layout=True,
    )
    _paired_panel(axes[0, 0], paired, config)
    _observability_panel(axes[0, 1], observable, config)
    _on_policy_panel(axes[1, 0], on_policy, config)
    _gate_panel(axes[1, 1], gate)
    for label, axis in zip("abcd", axes.flat, strict=True):
        axis.text(
            -0.13,
            1.06,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=10,
            va="top",
        )
    return _save(figure, config.paths.figure_dir / "figure_2_audit_overview", config)


def _paired_panel(axis: plt.Axes, table: pd.DataFrame, config: ProjectConfig) -> None:
    table = _decision_scope(table, config)
    selected = table.loc[
        (table.get("teacher_id", pd.Series(dtype=str)) == config.audit.primary_teacher_id)
        & (table.get("decoy_role", pd.Series(dtype=str)) == "matched_cath")
        & (table.get("environment_axis", pd.Series(dtype=str)) == "burial")
    ]
    state_kind = _representative_state_kind(selected)
    if state_kind is not None:
        selected = selected.loc[selected["state_kind"] == state_kind]
    if selected.empty:
        _empty_axis(axis, "Paired specificity", "No matched-decoy estimate")
        return
    colors = [OKABE_ITO["blue"], OKABE_ITO["orange"], OKABE_ITO["green"]]
    for index, environment in enumerate(sorted(selected["environment"].unique())):
        frame = selected.loc[selected["environment"] == environment].sort_values(
            "requested_corruption_ratio"
        )
        x = frame["requested_corruption_ratio"].to_numpy(dtype=float)
        y = frame["estimate"].to_numpy(dtype=float)
        error = np.vstack(
            [
                y - frame["ci_low"].to_numpy(dtype=float),
                frame["ci_high"].to_numpy(dtype=float) - y,
            ]
        )
        axis.errorbar(
            x,
            y,
            yerr=error,
            marker="o",
            color=colors[index % len(colors)],
            capsize=2,
            label=environment,
        )
    axis.axhline(config.decision.minimum_paired_decoy_lift_nats, color="0.55", linestyle="--")
    axis.set_title(f"Paired vs matched decoy ({_display_name(state_kind)})")
    axis.set_xlabel("Requested corruption")
    axis.set_ylabel("Native-logp lift (nats)")
    axis.legend(frameon=False, fontsize=7)


def _observability_panel(axis: plt.Axes, table: pd.DataFrame, config: ProjectConfig) -> None:
    table = _decision_scope(table, config)
    selected = table.loc[
        (table.get("metric", pd.Series(dtype=str)) == "jsd_reduction_nats")
        & (table.get("environment_axis", pd.Series(dtype=str)) == "burial")
        & (table.get("group_level", pd.Series(dtype=str)) == "cath_h")
    ]
    state_kind = _representative_state_kind(selected)
    if state_kind is not None:
        selected = selected.loc[selected["state_kind"] == state_kind]
    if selected.empty:
        _empty_axis(axis, "Residual observability", "No CATH-H cross-fit estimate")
        return
    colors = [OKABE_ITO["blue"], OKABE_ITO["orange"], OKABE_ITO["green"]]
    for index, environment in enumerate(sorted(selected["environment"].unique())):
        frame = (
            selected.loc[selected["environment"] == environment]
            .groupby("requested_corruption_ratio", observed=True)["estimate"]
            .mean()
            .sort_index()
        )
        axis.plot(
            frame.index,
            frame.values,
            marker="o",
            label=environment,
            color=colors[index % len(colors)],
        )
    axis.axhline(0, color="0.55", linewidth=0.8)
    axis.set_title(f"CATH-H observability ({_display_name(state_kind)})")
    axis.set_xlabel("Requested corruption")
    axis.set_ylabel("JSD reduction (nats)")
    axis.legend(frameon=False, fontsize=7)


def _on_policy_panel(axis: plt.Axes, table: pd.DataFrame, config: ProjectConfig) -> None:
    table = _decision_scope(table, config)
    selected = table.loc[
        table.get("metric", pd.Series(dtype=str)) == "teacher_advantage_difference_nats"
    ]
    if selected.empty:
        _empty_axis(axis, "On-policy necessity", "No matched comparison")
        return
    labels = selected["comparison"].str.replace("on_policy_vs_", "", regex=False)
    y = selected["estimate"].to_numpy(dtype=float)
    error = np.vstack(
        [
            y - selected["ci_low"].to_numpy(dtype=float),
            selected["ci_high"].to_numpy(dtype=float) - y,
        ]
    )
    positions = np.arange(len(selected))
    axis.errorbar(
        positions,
        y,
        yerr=error,
        fmt="o",
        color=OKABE_ITO["purple"],
        capsize=2,
    )
    axis.axhline(config.decision.minimum_on_policy_advantage_nats, color="0.55", linestyle="--")
    axis.set_xticks(positions, labels, rotation=15, ha="right")
    axis.set_title("Matched on-policy increment")
    axis.set_ylabel("Advantage difference (nats)")


def _gate_panel(axis: plt.Axes, table: pd.DataFrame) -> None:
    if table.empty:
        _empty_axis(axis, "foundation decision", "No criteria")
        return
    status_order = {"PASS": 1.0, "INCOMPLETE": 0.0, "FAIL": -1.0}
    color = {
        "PASS": OKABE_ITO["blue"],
        "INCOMPLETE": OKABE_ITO["gray"],
        "FAIL": OKABE_ITO["orange"],
    }
    shown = table.tail(9).copy()
    positions = np.arange(len(shown))
    values = shown["status"].map(status_order).to_numpy(dtype=float)
    axis.scatter(
        values, positions, c=shown["status"].map(color), s=32, edgecolor="black", linewidth=0.3
    )
    axis.set_yticks(positions, shown["criterion"].str.replace("_", " "), fontsize=6)
    axis.set_xticks([-1, 0, 1], ["Fail", "Incomplete", "Pass"])
    axis.set_xlim(-1.25, 1.25)
    axis.set_title("Fixed foundation decision criteria")
    axis.grid(axis="y", visible=False)


def _marker_sizes(values: pd.Series, config: ProjectConfig) -> np.ndarray:
    array = values.to_numpy(dtype=float)
    low, high = np.nanmin(array), np.nanmax(array)
    if np.isclose(low, high):
        return np.full(
            len(array), (config.plot.minimum_marker_area + config.plot.maximum_marker_area) / 2
        )
    scaled = (array - low) / (high - low)
    return config.plot.minimum_marker_area + scaled * (
        config.plot.maximum_marker_area - config.plot.minimum_marker_area
    )


def _decision_scope(table: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    if table.empty or config.audit.decision_analysis_role == "all":
        return table
    if "analysis_role" not in table.columns:
        return table.iloc[0:0]
    return table.loc[table["analysis_role"] == config.audit.decision_analysis_role]


def _representative_state_kind(table: pd.DataFrame) -> str | None:
    if table.empty or "state_kind" not in table:
        return None
    available = set(table["state_kind"].astype(str))
    return "random_mask" if "random_mask" in available else sorted(available)[0]


def _display_name(value: str | None) -> str:
    return value.replace("_", " ") if value is not None else "all states"


def _empty_axis(axis: plt.Axes, title: str, message: str) -> None:
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
    axis.set_axis_off()


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(figure: plt.Figure, stem: Path, config: ProjectConfig) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for file_format in config.plot.formats:
        path = stem.with_suffix(f".{file_format}")
        figure.savefig(
            path,
            dpi=config.plot.dpi if file_format == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
        paths.append(path)
    plt.close(figure)
    return paths
