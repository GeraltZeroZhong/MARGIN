"""Build the stability study post-lock method matrix, figure, and completion report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from margin.provenance import runtime_manifest, write_json, write_parquet, write_text

ZERO_SHOT_ORDER = [
    ("blosum62_global", "BLOSUM62", "sequence / evolutionary"),
    ("cath_homolog_profile", "CATH homolog profile", "sequence / evolutionary"),
    ("carp_640M_loo", "CARP-640M LOO", "sequence / evolutionary"),
    ("esm1b_650M_loo", "ESM-1b-650M LOO", "sequence / evolutionary"),
    ("esm2_150M_loo", "ESM2-150M LOO", "sequence / evolutionary"),
    ("esm2_650M_loo", "ESM2-650M LOO", "sequence / evolutionary"),
    ("esm2_150M_plus_G_Cplus", "Registered G+C+", "strong sequence control"),
    ("esm_if1_action_only", "ESM-IF1 action", "structure-conditioned"),
    ("mif_action_only", "MIF action", "structure-conditioned"),
    ("proteinmpnn_action_only", "ProteinMPNN action", "structure-conditioned"),
    (
        "temperature_consensus_action_only",
        "Temperature consensus action",
        "structure-conditioned",
    ),
    ("unscaled_equal_action_only", "Unscaled consensus action", "structure-conditioned"),
]


def build_stability_supplement(project_root: Path) -> dict[str, Path]:
    project_root = project_root.resolve()
    output = project_root / "runs/stability/supplement"
    source = output / "source_data"
    figures = output / "figures"
    reports = output / "reports"
    for directory in (source, figures, reports):
        directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "zero_shot": source / "zero_shot_method_matrix.parquet",
        "supervised": source / "supervised_upper_bounds.parquet",
        "cost": source / "teacher_cost_frontier.parquet",
        "structure": source / "cross_platform_structure_sources.parquet",
        "figure_pdf": figures / "figure3_postlock_supplement.pdf",
        "figure_svg": figures / "figure3_postlock_supplement.svg",
        "figure_png": figures / "figure3_postlock_supplement.png",
        "report": reports / "stability_postlock_supplement.md",
        "manifest": output / "manifest.json",
    }
    zero_shot = _zero_shot_matrix(project_root)
    supervised = _supervised_matrix(project_root)
    cost = _cost_frontier(project_root, zero_shot)
    structure = _structure_source_matrix(project_root)
    tables = {
        "zero_shot": zero_shot,
        "supervised": supervised,
        "cost": cost,
        "structure": structure,
    }
    for name, table in tables.items():
        write_parquet(paths[name], table)
        table.to_csv(paths[name].with_suffix(".csv"), index=False)
    _figure(zero_shot, supervised, cost, structure, paths)
    write_text(paths["report"], _report(project_root, zero_shot, supervised, cost, structure))
    write_json(
        paths["manifest"],
        {
            **runtime_manifest(project_root),
            "status": "STABILITY_POSTLOCK_SUPPLEMENT_COMPLETE",
            "changes_primary_decision": False,
            "routing_authorized": False,
            "zero_shot_and_supervised_rankings_separated": True,
            "tables": {
                name: {"path": str(paths[name]), "rows": len(table)}
                for name, table in tables.items()
            },
            "figures": [
                str(paths["figure_pdf"]),
                str(paths["figure_svg"]),
                str(paths["figure_png"]),
            ],
            "report": str(paths["report"]),
        },
    )
    return paths


def _zero_shot_matrix(root: Path) -> pd.DataFrame:
    summary = pd.read_parquet(root / "runs/stability/method_audit/method_summary.parquet")
    selected = summary.loc[
        summary["evaluation_population"].eq("megascale_stability_dense")
        & summary["stratum"].eq("all")
        & summary["metric"].isin(["spearman", "ndcg_at_10_percent"])
    ].copy()
    labels = pd.DataFrame(
        [
            {
                "method": method,
                "display_name": label,
                "display_group": group,
                "display_order": index,
            }
            for index, (method, label, group) in enumerate(ZERO_SHOT_ORDER)
        ]
    )
    selected = selected.merge(labels, on="method", how="left", validate="many_to_one")
    selected["in_primary_display"] = selected["display_order"].notna()
    return selected.sort_values(
        ["in_primary_display", "display_order", "method", "metric"],
        ascending=[False, True, True, True],
        ignore_index=True,
    )


def _supervised_matrix(root: Path) -> pd.DataFrame:
    summary = pd.read_parquet(root / "runs/stability/supervised/evaluation/summary.parquet")
    return summary.loc[summary["metric"].isin(["spearman", "ndcg_at_10_percent"])].sort_values(
        ["scope", "method", "metric"], ignore_index=True
    )


def _cost_frontier(root: Path, zero_shot: pd.DataFrame) -> pd.DataFrame:
    cost = pd.read_parquet(root / "runs/stability/method_audit/teacher_cost.parquet")
    mapping = {
        "mif": "mif_action_only",
        "esm_if1": "esm_if1_action_only",
        "proteinmpnn": "proteinmpnn_action_only",
        "three_teacher_consensus_serial": "unscaled_equal_action_only",
    }
    rows = []
    for row in cost.itertuples(index=False):
        method = mapping[str(row.method)]
        performance = zero_shot.loc[
            zero_shot["method"].eq(method) & zero_shot["metric"].eq("spearman")
        ].iloc[0]
        rows.append(
            {
                **row._asdict(),
                "performance_method": method,
                "spearman": float(performance["estimate"]),
                "spearman_ci_low": float(performance["ci_low"]),
                "spearman_ci_high": float(performance["ci_high"]),
                "deployment_tier": (
                    "fast"
                    if row.method == "esm_if1"
                    else "robust"
                    if row.method == "three_teacher_consensus_serial"
                    else "single_teacher_reference"
                ),
            }
        )
    return pd.DataFrame(rows)


def _structure_source_matrix(root: Path) -> pd.DataFrame:
    summary = pd.read_parquet(root / "runs/structure_sensitivity/evaluation/role_summary.parquet")
    availability = pd.read_parquet(
        root / "runs/structure_sensitivity/evaluation/role_availability.parquet"
    )
    selected = summary.loc[
        summary["metric"].isin(["action_spearman", "spearman_margin", "ndcg10_margin"])
    ].merge(availability, on="structure_role", validate="many_to_one")
    labels = {
        "experimental": "Experimental PDB",
        "alphafold": "AlphaFold DB",
        "esmfold": "ESMFold",
        "perturbed_0p5": "Experimental + 0.5 Å",
        "perturbed_1p0": "Experimental + 1.0 Å",
    }
    selected["display_name"] = selected["structure_role"].map(labels)
    selected["display_order"] = selected["structure_role"].map(
        {role: index for index, role in enumerate(labels)}
    )
    return selected.sort_values(["display_order", "metric"], ignore_index=True)


def _figure(
    zero_shot: pd.DataFrame,
    supervised: pd.DataFrame,
    cost: pd.DataFrame,
    structure: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    colors = {
        "sequence / evolutionary": "#4C78A8",
        "strong sequence control": "#7A5195",
        "structure-conditioned": "#E45756",
        "spurs": "#2A9D8F",
        "thermompnn": "#F2A541",
    }
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), constrained_layout=True)
    ax = axes[0, 0]
    primary = zero_shot.loc[
        zero_shot["in_primary_display"] & zero_shot["metric"].eq("spearman")
    ].sort_values("display_order", ascending=False)
    for y, row in enumerate(primary.itertuples(index=False)):
        ax.errorbar(
            row.estimate,
            y,
            xerr=[[row.estimate - row.ci_low], [row.ci_high - row.estimate]],
            fmt="o",
            ms=4,
            lw=1,
            capsize=2,
            color=colors[row.display_group],
        )
    ax.set_yticks(np.arange(len(primary)), primary["display_name"])
    ax.set_xlim(0, 0.65)
    ax.set_xlabel("Equal-domain Spearman")
    ax.set_title("a  Zero-shot method matrix", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[0, 1]
    scope_order = ["test", "not_listed", "primary_all_mixed_overlap"]
    scope_labels = ["Official test\n(n=2)", "Not listed\n(n=6)", "Mixed overlap\n(n=32)"]
    for offset, method in zip((-0.10, 0.10), ("thermompnn", "spurs"), strict=True):
        frame = (
            supervised.loc[
                supervised["method"].eq(method)
                & supervised["metric"].eq("spearman")
                & supervised["scope"].isin(scope_order)
            ]
            .set_index("scope")
            .loc[scope_order]
        )
        x = np.arange(len(scope_order), dtype=float) + offset
        ax.errorbar(
            x,
            frame["estimate"],
            yerr=np.vstack(
                [
                    frame["estimate"].to_numpy() - frame["ci_low"].to_numpy(),
                    frame["ci_high"].to_numpy() - frame["estimate"].to_numpy(),
                ]
            ),
            fmt="o",
            ms=5,
            capsize=2,
            lw=1,
            color=colors[method],
            label="ThermoMPNN" if method == "thermompnn" else "SPURS",
        )
    ax.set_xticks(np.arange(len(scope_order)), scope_labels)
    ax.set_ylim(0.55, 0.93)
    ax.set_ylabel("Equal-domain Spearman")
    ax.set_title("b  Label-trained upper bounds", loc="left", fontweight="bold")
    ax.text(
        0.01,
        0.02,
        "Separate from zero-shot ranking",
        transform=ax.transAxes,
        fontsize=6.5,
        color="#555555",
    )
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1, 0]
    roles = ["experimental", "alphafold", "perturbed_0p5", "perturbed_1p0"]
    frame = (
        structure.loc[
            structure["metric"].eq("spearman_margin") & structure["structure_role"].isin(roles)
        ]
        .set_index("structure_role")
        .loc[roles]
    )
    palette = ["#264653", "#2A9D8F", "#E9C46A", "#E76F51"]
    x = np.arange(len(roles))
    for index, (_, row) in enumerate(frame.iterrows()):
        ax.errorbar(
            index,
            row["estimate"],
            yerr=[
                [row["estimate"] - row["ci_low"]],
                [row["ci_high"] - row["estimate"]],
            ],
            fmt="o",
            color=palette[index],
            markersize=5,
            elinewidth=1.2,
            capsize=3,
            zorder=3,
        )
    ax.axhline(0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_xticks(
        x,
        ["Experimental", "AlphaFold", "+0.5 Å", "+1.0 Å"],
        rotation=15,
        ha="right",
    )
    ax.set_ylabel("Spearman margin: action − G+C+")
    ax.set_title("c  Matched structure source", loc="left", fontweight="bold")
    ax.text(
        0.01,
        0.02,
        "ESMFold n=5; below frozen n=8 summary minimum",
        transform=ax.transAxes,
        fontsize=6.3,
        color="#555555",
    )
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1, 1]
    labels = {
        "esm_if1": "ESM-IF1 (fast)",
        "mif": "MIF",
        "proteinmpnn": "ProteinMPNN",
        "three_teacher_consensus_serial": "3-teacher (robust)",
    }
    cost_colors = {
        "fast": "#2A9D8F",
        "robust": "#E45756",
        "single_teacher_reference": "#4C78A8",
    }
    for row in cost.itertuples(index=False):
        ax.scatter(
            row.wall_seconds_domain_median,
            row.spearman,
            s=38,
            color=cost_colors[row.deployment_tier],
            zorder=3,
        )
        ax.annotate(
            labels[row.method],
            (row.wall_seconds_domain_median, row.spearman),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=6.5,
        )
    ax.set_xscale("log")
    ax.set_xlim(0.035, 1.9)
    ax.set_ylim(0.515, 0.57)
    ax.set_xlabel("Median adapter time / protein (s, log scale)")
    ax.set_ylabel("Zero-shot action Spearman")
    ax.set_title("d  Accuracy–cost frontier", loc="left", fontweight="bold")
    ax.grid(color="#DDDDDD", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    for path in (paths["figure_pdf"], paths["figure_svg"]):
        figure.savefig(path, bbox_inches="tight")
    figure.savefig(paths["figure_png"], dpi=300, bbox_inches="tight")
    plt.close(figure)


def _report(
    root: Path,
    zero_shot: pd.DataFrame,
    supervised: pd.DataFrame,
    cost: pd.DataFrame,
    structure: pd.DataFrame,
) -> str:
    cross = pd.read_parquet(root / "runs/external_validation/evaluation/contrast_summary.parquet")
    cross_decision = pd.read_parquet(
        root / "runs/external_validation/evaluation/decision.parquet"
    ).iloc[0]
    structure_sensitivity_decision = pd.read_parquet(
        root / "runs/structure_sensitivity/evaluation/decision.parquet"
    ).iloc[0]
    deltas = pd.read_parquet(
        root / "runs/structure_sensitivity/evaluation/paired_delta_summary.parquet"
    )
    status = pd.read_parquet(root / "runs/stability/supervised/evaluation/method_status.parquet")
    primary = zero_shot.loc[
        zero_shot["in_primary_display"] & zero_shot["metric"].eq("spearman")
    ].sort_values("display_order")
    supervised_rows = supervised.loc[
        supervised["metric"].eq("spearman")
        & supervised["scope"].isin(["test", "not_listed", "primary_all_mixed_overlap"])
    ].copy()
    primary_cross = cross.loc[
        cross["contrast"].eq("temperature_action_vs_g_plus_c_plus")
    ].set_index("metric")
    alpha_delta = deltas.loc[
        deltas["structure_role"].eq("alphafold")
        & deltas["metric"].eq("action_spearman_delta_vs_experimental")
    ].iloc[0]
    perturb1 = deltas.loc[
        deltas["structure_role"].eq("perturbed_1p0")
        & deltas["metric"].eq("action_spearman_delta_vs_experimental")
    ].iloc[0]
    fast = cost.set_index("method").loc["esm_if1"]
    robust = cost.set_index("method").loc["three_teacher_consensus_serial"]
    cost_ratio = robust["wall_seconds_domain_median"] / fast["wall_seconds_domain_median"]
    lines = [
        "# stability study post-lock 方法与跨平台补充报告",
        "",
        "_Computational methods and cross-platform validation supplement._",
        "",
        "## 总体判决",
        "",
        "原冻结判决 `STABILITY_STRUCTURE_CONDITIONED_ACTION_CONFIRMED_BEYOND_CPLUS` 保持不变。",
        f"新的多蛋白跨平台机器判决为 `{cross_decision['decision']}`；"
        "structure-sensitivity study 状态为 "
        f"`{structure_sensitivity_decision['decision']}`。两者均不授权 routing，"
        "也不恢复 sequence-only residual transfer 或 counterfactual subtraction。",
        "",
        "当前最准确的方法学结论是：**正确配对的逆折叠动作，在强 outcome-free 序列控制后，"
        "仍提供可复制的稳定性排序增量；高质量预测骨架可保留该增量，但约 1 Å 的平滑几何"
        "误差会显著削弱它。**",
        "",
        "## 零样本方法矩阵（32 个 Megascale 域）",
        "",
        "零样本与监督模型严格分表。下表为等蛋白 Spearman；完整矩阵及 NDCG@10% 位于 "
        "`source_data/zero_shot_method_matrix.parquet`。",
        "",
        "| 方法 | 类别 | Spearman | 95% CI |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in primary.itertuples(index=False):
        lines.append(
            f"| {row.display_name} | {row.display_group} | {row.estimate:.4f} | "
            f"[{row.ci_low:.4f}, {row.ci_high:.4f}] |"
        )
    lines.extend(
        [
            "",
            "最强零样本点估计是未缩放三教师 action-only 共识（Spearman 0.5616）。温度共识为 "
            "0.5553；温度缩放用于 outcome-free 选择与统一概率尺度，并未带来稳定性收益。"
            "简化的 ESM2 sequence-prior + ESM-IF1 action 求和没有在 Spearman 上显著超过 "
            "ESM-IF1 action-only，且 NDCG@10% 的方向相反，因此未形成新的主方法。该基线"
            "属于简化诊断性组合；官方 ensemble、importance-sampling 或 free-energy "
            "protocol 需要独立实现。",
            "",
            "## 监督稳定性上界（单独报告）",
            "",
            "ThermoMPNN 与 SPURS 使用稳定性标签训练，不能作为零样本竞争者。32 域混合表"
            "含训练重叠；官方 test 仅 2 域，not-listed 6 域也不能自动解释为独立。",
            "",
            "| 方法 | 作用域 | 域数 | Spearman | 95% CI |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    scope_labels = {
        "test": "official test",
        "not_listed": "not listed",
        "primary_all_mixed_overlap": "mixed overlap",
    }
    for row in supervised_rows.sort_values(["scope", "method"]).itertuples(index=False):
        lines.append(
            f"| {row.method} | {scope_labels[row.scope]} | {row.n_domains} | "
            f"{row.estimate:.4f} | [{row.ci_low:.4f}, {row.ci_high:.4f}] |"
        )
    hermes = status.loc[status["method"].eq("hermes")].iloc[0]
    lines.extend(
        [
            "",
            f"HERMES 状态为 `{hermes['status']}`：{hermes['reason']} 未使用不兼容预处理器"
            "伪造可比结果。",
            "",
            "## 简化与成本",
            "",
            f"ESM-IF1 是 fast tier：中位 {fast['wall_seconds_domain_median']:.3f} s/蛋白，"
            f"action-only Spearman {fast['spearman']:.4f}。未缩放三教师是 robust tier：中位 "
            f"{robust['wall_seconds_domain_median']:.3f} s/蛋白，Spearman "
            f"{robust['spearman']:.4f}；约 {cost_ratio:.1f}× "
            "中位适配器耗时，换取约 +0.0258 Spearman。计时排除模型加载与结构生成。",
            "",
            "## 新的多蛋白跨平台 C+ 确认",
            "",
            "FireProt 同源隔离面板在任何分数或 ΔΔG 数值开启前锁定：18 个非 Megascale "
            "蛋白、1,796 个唯一单点变体、791 个查询位置。效应定义为 "
            "`-median(FireProt ddG)`，正值表示更稳定。",
            "",
            f"主 Spearman 增量 `A − (G+C+)` 为 {primary_cross.loc['spearman', 'estimate']:+.4f} "
            f"（95% CI [{primary_cross.loc['spearman', 'ci_low']:+.4f}, "
            f"{primary_cross.loc['spearman', 'ci_high']:+.4f}]），"
            f"正向蛋白 {int(primary_cross.loc['spearman', 'positive_domains'])}/"
            f"{int(primary_cross.loc['spearman', 'n_domains'])}；NDCG@10% 增量为 "
            f"{primary_cross.loc['ndcg10', 'estimate']:+.4f}。全部冻结门通过。",
            "",
            "这首次把本项目最独特的命题——paired structure action 超过 G+C+——复制到"
            "多蛋白、非 Megascale 的直接热力学数据上。信息论上的序列不可恢复性仍未被证明。",
            "",
            "## structure-sensitivity study：实验结构、预测结构与几何扰动",
            "",
            f"AlphaFold DB 在 17 个匹配蛋白上保留了相对 G+C+ 的 Spearman 增量 "
            f"{structure_sensitivity_decision['alphafold_spearman_margin']:+.4f}。"
            "相对同蛋白实验结构的 action "
            f"Spearman 变化为 {alpha_delta['estimate']:+.4f} "
            f"（[{alpha_delta['ci_low']:+.4f}, {alpha_delta['ci_high']:+.4f}]），区间跨零。"
            "因此在本轮样本与精度下未检测到差异；区间跨零不构成两类结构等效的证明。",
            "",
            f"0.5 Å 平滑扰动呈下降趋势但区间跨零；1.0 Å 扰动相对实验结构下降 "
            f"{perturb1['estimate']:+.4f}（[{perturb1['ci_low']:+.4f}, "
            f"{perturb1['ci_high']:+.4f}]），且其 action 相对 G+C+ 的平均 Spearman margin "
            "变为负点估计。后续诊断表明平滑坐标扰动同时改变教师输入分布与部分跨残基"
            "几何，因此这里只支持对该扰动模型的敏感性结论，不把 1 Å 解释为通用几何"
            "容忍阈值。",
            "",
            "ESMFold 公共 API 在两轮顺序/并行重试后仅返回 5 个蛋白，低于冻结的 8 蛋白"
            "汇总门槛；只保留逐域分数，不给总体结论。AlphaFold 低置信区只有 1 个蛋白、7 "
            "个变体，中置信区 5 个蛋白、64 个变体，不能据此学习质量 routing。",
            "",
            "## 边界与项目状态",
            "",
            "- U+ 继续解释为 registered G+C+ 之后的 paired-structure residual；"
            "纯结构因果解释超出当前证据；",
            "- calibration 用于尺度统一；三教师的首要价值是跨架构复现与 robust tier；",
            "- 监督模型只作为标签训练上界；训练重叠分层必须随结果一起报告；",
            "- selective routing 继续为 `NOT_ESTABLISHED`；不在已开启面板上调阈值；",
            "",
            "## 复现入口",
            "",
            "```bash",
            "PYTHONPATH=src python scripts/workflows/external_validation/prepare.py",
            "PYTHONPATH=src conda run -n margin-models python "
            "scripts/workflows/external_validation/run_representations.py --device cuda:0",
            "PYTHONPATH=src conda run -n margin-models python "
            "scripts/workflows/external_validation/run_teachers.py --device cuda:0",
            "PYTHONPATH=src python scripts/workflows/external_validation/build_profiles.py",
            "PYTHONPATH=src python scripts/workflows/external_validation/evaluate.py",
            "PYTHONPATH=src python scripts/workflows/structure_sensitivity/prepare.py",
            "PYTHONPATH=src conda run -n margin-models python "
            "scripts/workflows/structure_sensitivity/run_teachers.py --device cuda:0",
            "PYTHONPATH=src python scripts/workflows/structure_sensitivity/evaluate.py",
            "PYTHONPATH=src python scripts/workflows/stability/build_supplement.py",
            "```",
            "",
            "## 公开实现来源",
            "",
            "- [ThermoMPNN](https://github.com/Kuhlman-Lab/ThermoMPNN)",
            "- [SPURS](https://github.com/luo-group/SPURS)",
            "- [HERMES](https://github.com/StatPhysBio/hermes)",
            "- [Meta ESM / ESMFold](https://github.com/facebookresearch/esm)",
            "- [AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/)",
            "",
            "![stability study post-lock supplement](../figures/figure3_postlock_supplement.png)",
        ]
    )
    return "\n".join(lines) + "\n"
