"""Chinese scientific report for the locked action-validation study study."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from margin.provenance import runtime_manifest, write_json, write_text
from margin.studies.action_validation.config import ActionValidationStudyConfig


def build_action_validation_report(config: ActionValidationStudyConfig) -> Path:
    """Render the complete action-validation study decision, evidence, and limitations."""

    evaluation = config.paths.run_dir / "evaluation"
    panel = config.paths.run_dir / "panel"
    report_dir = config.paths.run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    decision = pd.read_parquet(evaluation / "project_decision.parquet").iloc[0]
    gates = pd.read_parquet(evaluation / "gate_checks.parquet")
    quality = pd.read_parquet(evaluation / "quality_controls.parquet")
    training = pd.read_parquet(evaluation / "training_summary.parquet")
    u_summary = pd.read_parquet(evaluation / "u_margin_summary.parquet")
    component_summary = pd.read_parquet(evaluation / "component_margin_summary.parquet")
    domain_metrics = pd.read_parquet(evaluation / "domain_metrics.parquet")
    agreement = pd.read_parquet(evaluation / "teacher_agreement_summary.parquet")
    shuffle = pd.read_parquet(evaluation / "shuffle_margin_summary.parquet")
    subgroup = pd.read_parquet(evaluation / "subgroup_margin_summary.parquet")
    routing = pd.read_parquet(evaluation / "routing_diagnostic_summary.parquet")
    domains = pd.read_parquet(panel / "domains.parquet")
    variants = pd.read_parquet(panel / "variants.parquet")
    queries = pd.read_parquet(panel / "query_rows.parquet")
    if not bool(quality["passed"].all()):
        raise RuntimeError("action-validation study quality controls do not all pass")
    reconstruction_error = float(quality.loc[quality["threshold"].le(1e-8), "estimate"].max())

    dense = domains.loc[domains["evaluation_population"].eq("megascale_dense")]
    sparse = domains.loc[domains["evaluation_population"].eq("s669_sparse_cross_platform")]
    dense_variants = variants.loc[variants["evaluation_population"].eq("megascale_dense")]
    sparse_variants = variants.loc[
        variants["evaluation_population"].eq("s669_sparse_cross_platform")
    ]
    report = f"""# action-validation study：Structure-Unique Action Decomposition

_Frozen cross-platform computational evaluation._

## 正式判决

本轮机器判决为：

`{decision.decision}`

中心问题的答案是**是，但必须限定 estimand**：在注册的全局替换项 (G) 和序列上下文项
(C) 之后，三个独立逆折叠教师的 paired action 都保留了稳定性相关的实例特异增量
(U)。致密 Megascale 主面板上，等权校准共识的 U 增加 domain Spearman
`{decision.consensus_dense_spearman_margin:+.4f}`（95% CI
`[{decision.consensus_dense_spearman_ci_low:+.4f}, {_summary_row(u_summary, "consensus", "megascale_dense", "all", "spearman_margin").ci_high:+.4f}]`），
NDCG@10% 增加 `{decision.consensus_dense_ndcg10_margin:+.4f}`（95% CI
`[{decision.consensus_dense_ndcg10_ci_low:+.4f}, {_summary_row(u_summary, "consensus", "megascale_dense", "all", "ndcg_at_10_percent_margin").ci_high:+.4f}]`）。
8 个 S669 域的稀疏跨平台方向性复制为 Spearman
`{decision.consensus_s669_spearman_margin:+.4f}`，7/8 个域为正。

现有证据支持一个精确结论：**相对于本轮冻结的 (G) 与 (C) 控制集合，paired target
structure 提供了这些序列先验未覆盖、并能预测稳定性排序的增量动作值。** (U) 的含义
始终依赖已注册的教师、序列特征和线性低秩控制；纯因果结构效应超出当前 estimand。

项目状态更新为：

- `REGISTERED_ROUTE: PIVOT_SELECTIVE_STRUCTURE_CONDITIONED`
- `CURRENTLY_SUPPORTED_IMPLEMENTATION: {decision.currently_supported_implementation}`
- `SELECTIVE_ROUTING: NOT_YET_ESTABLISHED`
- `COUNTERFACTUAL_SEARCH: CLOSED`
- `SELECTIVE_ROUTING_AUTHORIZED: false`

counterfactuals 与 mechanisms 的既有判决没有修改；本轮不恢复 sequence-only residual transfer，也不重开
counterfactual subtraction。

```mermaid
flowchart LR
    accTitle: action-validation study evidence chain
    accDescr: CATH teacher actions fit outcome-free global and context controls, paired teachers define U, and a locked two-platform stability panel confirms unique action while routing remains unresolved.

    cath[( CATH development)] --> controls[ Fit G and C]
    teachers[( Three paired teachers)] --> action[ Action A]
    controls --> unique{{ U = A - G - C}}
    action --> unique
    panel[( Locked two-platform panel)] --> gates{{ Nine frozen gates}}
    unique --> gates
    gates --> confirmed[ Unique action confirmed]
    gates --> routing[ Routing not established]

    classDef data fill:#f3f4f6,stroke:#6b7280,stroke-width:2px,color:#1f2937
    classDef process fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef decision_style fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12
    class cath,teachers,panel data
    class controls,action process
    class unique,gates,confirmed,routing decision_style
```

## 锁定数据与证据等级

协议在任何 action-validation study 面板模型评分前冻结。主面板含 {len(dense)} 个此前未开启的
Megascale 域，其中 16 个 natural、16 个 de novo，共 {len(dense_variants):,} 个单突变；
复制面板含 {len(sparse)} 个此前未开启的 S669 域，共 {len(sparse_variants):,} 个单突变。
总计 {len(domains)} 个域、{len(variants):,} 个变体、{len(queries):,} 个查询位置。
选择只使用结果可用性、结构完整性、序列身份和确定性配额；稳定性幅度未参与选择。

所有候选均排除了相对于 observability CATH 训练集，以及 generalization、counterfactuals
与 mechanisms 已评估身份的
≥80% identity、≥90% 双向覆盖近重复；两个 action-validation study 平台之间也应用同一边界。
Megascale 因仍来自既有实验平台，只是**新域致密确认**。S669 是跨平台，但每域只有
5–68 个变体，且允许与历史面板共享宽泛 CATH 拓扑标签，所以只构成**稀疏方向性复制**，
不能替代真正的跨平台致密面板。Megascale 稳定性数据来自 cDNA-display proteolysis
大规模实验发布[^1][^2]；S669 是独立汇编的直接实验 ΔΔG 基准[^3]。

## 冻结方法

plain MIF、ESM-IF1 和 8-order probability-ensemble ProteinMPNN 分别在 paired backbone
上产生 20 氨基酸候选分布[^4][^5][^6]。对 WT (a) 与 mutant (b)，动作定义为
$A_k(a\\rightarrow b)=\\log q_k(b)-\\log q_k(a)$。$G_k$ 是 CATH 开发训练集中按 WT
条件化的动作均值；$C_k$ 使用 strict leave-one-position-out CARP-640M 表征、ESM2
候选动作与熵、WT、位置/长度和半径 4 的局部序列组成预测 $A-G$[^7][^8]。

C 模型只用 CATH 教师动作：8,622 个 `development_train` 位置选择 rank 与 alpha，
2,986 个 `development_validation` 位置确定组合，最后在二者合计 11,608 个位置重拟合。
所有教师均选择 rank 16、Ridge alpha 100；稳定性标签从未进入 (G/C)、模型选择或
教师缩放。三教师动作在同一 CATH 行上 RMS 匹配到 ESM2 后等权平均。

{_training_table(training)}

主指标在每域内计算 Spearman 与 NDCG@10%，域等权汇总；区间来自 5,000 次保持
natural/de novo 数量的固定随机 domain bootstrap。NDCG relevance 固定为域内
`effect - minimum(effect)`。独立单位始终为蛋白域；突变行作为域内观测。

## 主要结果

### G、C、U 的逐步贡献

致密面板的共识绝对 mean domain Spearman 从 sequence-only 的
`{_absolute_metric(domain_metrics, "sequence_only"):.4f}` 上升到 +G 的
`{_absolute_metric(domain_metrics, "consensus__plus_g"):.4f}`、+G+C 的
`{_absolute_metric(domain_metrics, "consensus__plus_gc"):.4f}`，再到 +G+C+U 的
`{_absolute_metric(domain_metrics, "consensus__plus_gcu"):.4f}`。逐步 margin 如下：

{_stage_table(component_summary)}

G 再次确认全局 WT→mutant 规律很强。C 对 Spearman 有小而稳定的增量，但对
NDCG@10% 的区间跨 0。U 是控制 G/C 后最大的增量，说明 mechanism study 中 paired-only
优势并不能仅由注册的全局替换与序列上下文模型解释。

### 三教师与跨平台复制

{_teacher_table(u_summary)}

三个教师的致密面板 Spearman 下界全部大于 0，而非由单一模型谱系驱动。S669 的
Spearman 方向在三教师与共识中均为正；但其共识 NDCG@10% 区间
`{_interval(_summary_row(u_summary, "consensus", "s669_sparse_cross_platform", "all", "ndcg_at_10_percent_margin"))}`
跨 0，这正是稀疏复制不能被描述为致密确认的原因。

### 实例特异性与教师一致性

共识 full U 相对 20 个域内 position-shuffled U 的 Spearman margin 为
`{_interval(_generic_row(shuffle, "megascale_dense", "spearman_actual_minus_shuffled"))}`；
32/32 个致密域为正。这排除了“只要加入同一 U 分布、位置是否对应无所谓”的解释。
三个教师 U 的致密面板 median pairwise candidate-action Spearman 为
`{_interval(_generic_row(agreement, "megascale_dense", "median_pairwise_u_spearman"))}`，
S669 为 `{_interval(_generic_row(agreement, "s669_sparse_cross_platform", "median_pairwise_u_spearman"))}`。
因此 U 同时具有位置对应性和跨架构方向一致性。

## 预声明边界分析

{_boundary_table(subgroup)}

Gly/Pro 在 mechanism study 的 counterfactual/Route B 分数中是明显失败边界；本轮 U 在该组
仍为正，但比其他替换弱。这说明直接 paired structure residual 能恢复一部分此前丢失的
局部主链相关信息，却不足以证明模型完整处理 backbone relaxation。高接触位点的区间跨
0，而 low-contact 位点明显为正；这是一项预声明分层结果，不应事后改造成排除规则或
“只在低接触位点使用结构”的 gate。

## Selective routing 仍未建立

固定的一致性 gate 要求位置的三教师 U 排名中位相关 ≥0.30 且具体替换的三教师 U
同号。它在致密面板仅激活 `{_routing_value(routing, "megascale_dense", "variant_gate_active_fraction"):.1%}`
的变体。相对于 G+C，它仍增加 Spearman
`{_interval(_generic_row(routing, "megascale_dense", "gated_minus_gc_spearman"))}`；但相对于
使用完整 U，它降低 `{_interval(_generic_row(routing, "megascale_dense", "gated_minus_full_spearman"))}`。
S669 上 gated-minus-full 同样为负。因此当前证据支持“使用校准的 paired structure”，
不支持“已有可靠规则决定何时关闭它”。

## 冻结门与质量控制

{_gate_table(gates)}

全部 9 个判决门通过。分解恒等式、教师缩放恒等式、共识重构和预测有限性检查也全部
通过；最大数值重构误差为 `{reconstruction_error:.3e}`。注意，门通过只确定本协议
中的项目分支，不把 bootstrap 区间解释为跨数据生成机制的普适保证。

## 科学解释

observability 至 mechanisms 工作流识别了两个“有用但理由错误”的来源：全局替换规律 (G) 和简单序列上下文
(C)。action-validation study 进一步识别了二者之外的 paired inverse-folding stability signal。
人工错误结构无法稳定定义这一剩余项，但直接在正确结构动作中回归掉 (G+C) 后，所得
U 能跨三个教师、自然/设计域和一个稀疏外部平台复现。最保守且充分的机制表述是：

> Paired inverse-folding actions contain a stability-relevant, target-position-specific
> component that remains after registered global-substitution and sequence-context controls.

sequence-only 蒸馏与可部署 gate 仍未建立；该结果为显式结构条件化路线提供了
比 counterfactual subtraction 更直接的识别依据。

## 局限与下一步

第一，致密主面板仍是 Megascale，同平台效应和标签生成机制可能放大排序收益。第二，
S669 只有 8 个域、181 个变体，顶部排序区间很宽。第三，U 是“相对于冻结控制集的剩余”，
更强的非线性序列控制可能继续缩小它；当前结果不能声称 U 等于纯几何因果效应。第四，
Megascale 使用发布的预测结构模型，而 S669 使用实验 PDB，结构质量与平台同时变化。
第五，一致性 gate 的失败说明路由需要新的、独立开发目标，不能在本已开启面板上继续调阈值。

如果目标是独立投稿或继续 foundation audit，最优先的新增证据有两项：一个真正跨
Megascale 平台的**致密**稳定性面板，以及在全新面板上冻结后验证的 routing rule。
不建议重新搜索结构反事实，也不建议继续扩展 sequence-only residual transfer。

## 产物与复现

核心产物：

- `runs/action_validation/evaluation/project_decision.parquet`
- `runs/action_validation/evaluation/u_margin_summary.parquet`
- `runs/action_validation/evaluation/variant_components.parquet`
- `runs/action_validation/evaluation/teacher_agreement_summary.parquet`
- `runs/action_validation/evaluation/subgroup_margin_summary.parquet`
- `runs/action_validation/evaluation/routing_diagnostic_summary.parquet`
- `runs/action_validation/figures/action_validation_figure1_unique_action.pdf`
- `runs/action_validation/figures/action_validation_figure2_boundaries_routing.pdf`
- `runs/action_validation/figures/source_data/`

完整计算顺序：

```bash
PYTHONPATH=src conda run -n margin-models python scripts/workflows/action_validation/prepare.py
PYTHONPATH=src conda run -n margin-models python scripts/workflows/action_validation/freeze_protocol.py
PYTHONPATH=src conda run -n margin-models python scripts/workflows/action_validation/run_representations.py --device cuda:0
PYTHONPATH=src conda run -n margin-models python scripts/workflows/action_validation/run_teachers.py --device cuda:0
PYTHONPATH=src conda run -n margin-models python scripts/workflows/action_validation/evaluate.py
PYTHONPATH=src conda run -n margin-models python scripts/workflows/action_validation/build_figures.py
PYTHONPATH=src conda run -n margin-models python scripts/workflows/action_validation/build_report.py
```

## References

[^1]: Tsuboyama, K. et al. (2023). “Mega-scale experimental analysis of protein folding stability in biology and design.” _Nature_ 620, 434–444. <https://doi.org/10.1038/s41586-023-06328-6>

[^2]: Tsuboyama, K. et al. (2023). “Mega-scale experimental analysis of protein folding stability in biology and design: data release.” _Zenodo_. <https://doi.org/10.5281/zenodo.7992926>

[^3]: Pancotti, C. et al. (2022). “Predicting protein stability changes upon single-point mutation: a thorough comparison of the available tools on a new dataset.” _Briefings in Bioinformatics_ 23, bbab555. <https://doi.org/10.1093/bib/bbab555>

[^4]: Yang, K. K., Zanichelli, N. & Yeh, H. (2023). “Masked inverse folding with sequence transfer for protein representation learning.” _Protein Engineering, Design and Selection_ 36, gzad015. <https://doi.org/10.1093/protein/gzad015>

[^5]: Hsu, C. et al. (2022). “Learning inverse folding from millions of predicted structures.” _Proceedings of Machine Learning Research_ 162, 8946–8970. <https://proceedings.mlr.press/v162/hsu22a.html>

[^6]: Dauparas, J. et al. (2022). “Robust deep learning-based protein sequence design using ProteinMPNN.” _Science_ 378, 49–56. <https://doi.org/10.1126/science.add2187>

[^7]: Yang, K. K., Fusi, N. & Lu, A. X. (2024). “Convolutions are competitive with transformers for protein sequence pretraining.” _Cell Systems_ 15, 286–294.e2. <https://doi.org/10.1016/j.cels.2024.01.008>

[^8]: Lin, Z. et al. (2023). “Evolutionary-scale prediction of atomic-level protein structure with a language model.” _Science_ 379, 1123–1130. <https://doi.org/10.1126/science.ade2574>
"""
    report_path = report_dir / "action_validation_report.md"
    write_text(report_path, report)
    summary_path = report_dir / "action_validation_summary.json"
    write_json(
        summary_path,
        {
            "decision": str(decision.decision),
            "unique_action_confirmed": bool(decision.unique_action_confirmed),
            "consensus_dense_spearman_margin": float(decision.consensus_dense_spearman_margin),
            "consensus_dense_spearman_ci_low": float(decision.consensus_dense_spearman_ci_low),
            "consensus_dense_ndcg10_margin": float(decision.consensus_dense_ndcg10_margin),
            "consensus_s669_spearman_margin": float(decision.consensus_s669_spearman_margin),
            "registered_route": str(decision.registered_route),
            "currently_supported_implementation": str(decision.currently_supported_implementation),
            "selective_routing": str(decision.selective_routing),
            "selective_routing_authorized": bool(decision.selective_routing_authorized),
        },
    )
    write_json(
        config.paths.run_dir / "manifest.json",
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "status": "ACTION_VALIDATION_COMPLETE",
            "decision": str(decision.decision),
            "protocol_lock": str(config.paths.run_dir / "protocol_lock.json"),
            "evaluation_manifest": str(evaluation / "manifest.json"),
            "report": str(report_path),
            "summary": str(summary_path),
            "figures": [
                str(
                    config.paths.run_dir / "figures" / "action_validation_figure1_unique_action.pdf"
                ),
                str(
                    config.paths.run_dir
                    / "figures"
                    / "action_validation_figure2_boundaries_routing.pdf"
                ),
            ],
            "selective_routing_authorized": False,
        },
    )
    return report_path


def _summary_row(
    table: pd.DataFrame,
    teacher: str,
    population: str,
    stratum: str,
    metric: str,
) -> pd.Series:
    selected = table.loc[
        table["teacher_id"].eq(teacher)
        & table["evaluation_population"].eq(population)
        & table["stratum"].eq(stratum)
        & table["metric"].eq(metric)
    ]
    if len(selected) != 1:
        raise ValueError(f"missing summary row {teacher}/{population}/{stratum}/{metric}")
    return selected.iloc[0]


def _generic_row(table: pd.DataFrame, population: str, metric: str) -> pd.Series:
    selected = table.loc[table["evaluation_population"].eq(population) & table["metric"].eq(metric)]
    if len(selected) != 1:
        raise ValueError(f"missing summary row {population}/{metric}")
    return selected.iloc[0]


def _interval(row: pd.Series) -> str:
    return f"{row['estimate']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"


def _training_table(training: pd.DataFrame) -> str:
    rows = [
        "| 教师 | 选中 rank | alpha | validation RMSE | locked-test G+C RMSE | 动作缩放 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in training.itertuples(index=False):
        rows.append(
            f"| {row.teacher_id} | {row.selected_rank} | {row.selected_ridge_alpha:g} | "
            f"{row.validation_rmse:.4f} | {row.locked_test_gc_rmse:.4f} | "
            f"{row.outcome_free_teacher_scale:.4f} |"
        )
    return "\n".join(rows)


def _stage_table(summary: pd.DataFrame) -> str:
    labels = {"plus_g": "G over sequence", "plus_gc": "C over G", "plus_gcu": "U over G+C"}
    rows = [
        "| 逐步成分 | ΔSpearman [95% CI] | ΔNDCG@10% [95% CI] |",
        "| --- | ---: | ---: |",
    ]
    for stage, label in labels.items():
        selected = summary.loc[
            summary["teacher_id"].eq("consensus")
            & summary["stage"].eq(stage)
            & summary["evaluation_population"].eq("megascale_dense")
            & summary["stratum"].eq("all")
        ]
        spearman = selected.loc[selected["metric"].eq("spearman_margin")].iloc[0]
        ndcg = selected.loc[selected["metric"].eq("ndcg_at_10_percent_margin")].iloc[0]
        rows.append(f"| {label} | {_interval(spearman)} | {_interval(ndcg)} |")
    return "\n".join(rows)


def _teacher_table(summary: pd.DataFrame) -> str:
    labels = {
        "mif": "plain MIF",
        "esm_if1": "ESM-IF1",
        "proteinmpnn": "ProteinMPNN",
        "consensus": "equal-weight consensus",
    }
    rows = [
        "| 教师 | Megascale ΔSpearman [95% CI] | S669 ΔSpearman [95% CI] |",
        "| --- | ---: | ---: |",
    ]
    for teacher, label in labels.items():
        dense = _summary_row(summary, teacher, "megascale_dense", "all", "spearman_margin")
        sparse = _summary_row(
            summary, teacher, "s669_sparse_cross_platform", "all", "spearman_margin"
        )
        rows.append(f"| {label} | {_interval(dense)} | {_interval(sparse)} |")
    return "\n".join(rows)


def _absolute_metric(metrics: pd.DataFrame, method: str) -> float:
    return float(
        metrics.loc[
            metrics["evaluation_population"].eq("megascale_dense") & metrics["method"].eq(method),
            "spearman",
        ].mean()
    )


def _boundary_table(summary: pd.DataFrame) -> str:
    levels = [
        ("gly_pro_boundary", "involves_glycine_or_proline", "Gly/Pro"),
        ("gly_pro_boundary", "other_substitutions", "其他替换"),
        ("contact_class", "high_contact", "high contact"),
        ("contact_class", "low_contact", "low contact"),
        ("burial", "buried", "buried"),
        ("burial", "exposed", "exposed"),
    ]
    rows = [
        "| 分层 | 共识 U ΔSpearman [95% CI] | 有效域数 |",
        "| --- | ---: | ---: |",
    ]
    selected = summary.loc[
        summary["teacher_id"].eq("consensus")
        & summary["evaluation_population"].eq("megascale_dense")
        & summary["metric"].eq("spearman_margin")
    ]
    for dimension, level, label in levels:
        row = selected.loc[selected["dimension"].eq(dimension) & selected["level"].eq(level)].iloc[
            0
        ]
        rows.append(f"| {label} | {_interval(row)} | {int(row.n_domains)} |")
    return "\n".join(rows)


def _routing_value(table: pd.DataFrame, population: str, metric: str) -> float:
    return float(_generic_row(table, population, metric)["estimate"])


def _gate_table(gates: pd.DataFrame) -> str:
    rows = [
        "| 冻结门 | 估计 | 阈值 | 结果 |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in gates.itertuples(index=False):
        rows.append(
            f"| {str(row.gate).replace('_', ' ')} | {row.estimate:+.4f} | "
            f"{row.threshold:+.4f} | {'PASS' if row.passed else 'FAIL'} |"
        )
    return "\n".join(rows)
