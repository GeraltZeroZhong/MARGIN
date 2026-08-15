"""Build the source-grounded mechanism study technical report."""

# Markdown prose and tables are intentionally kept on long source lines.
# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path

import pandas as pd

from margin.provenance import runtime_manifest, write_json
from margin.studies.mechanisms.config import MechanismStudyConfig
from margin.studies.mechanisms.evaluation import PAIRED_CONTROLS, ROUTE_B_METHOD

PRIMARY_METHODS = (
    ("contact_deletion", "delete 5%", "contrast__contact_deletion__0.05"),
    ("smooth_coordinate", "coordinate 0.25 Å", "contrast__smooth_coordinate__0.25"),
    (
        "constrained_reassignment",
        "constrained reassignment 10%",
        "contrast__constrained_reassignment__0.1",
    ),
    (
        "matched_real_structure",
        "descriptor-matched real backbone",
        "contrast__matched_real_structure__descriptor_matched",
    ),
)

METHOD_LABELS = {
    "sequence_only": "ESM2 sequence-only",
    "mif_paired_only": "MIF paired only",
    "sequence_plus_mif_paired_alpha_1": "sequence + paired MIF (α=1)",
    "sequence_plus_mif_paired_variance_matched": "sequence + paired MIF (variance matched)",
    "sequence_plus_mif_paired_minus_cath_substitution_background": "sequence + paired MIF − CATH substitution background",
    "legacy_direct_contrast": "legacy degree-preserving rewire",
    "contrast__contact_deletion__0.05": "delete 5% contrast",
    "contrast__contact_deletion__0.1": "delete 10% contrast",
    "contrast__contact_deletion__0.2": "delete 20% contrast",
    "contrast__smooth_coordinate__0.25": "coordinate 0.25 Å contrast",
    "contrast__smooth_coordinate__0.5": "coordinate 0.50 Å contrast",
    "contrast__smooth_coordinate__1": "coordinate 1.00 Å contrast",
    "contrast__constrained_reassignment__0.1": "constrained reassignment contrast",
    "contrast__matched_real_structure__descriptor_matched": "matched-real contrast",
    "route_b_carp_rank16": "CARP-B rank 16",
    "route_b_global_wt_mutant_matrix": "global WT→mutant matrix",
    "route_b_grantham_aaindex_blosum_linear": "Grantham/AAindex/BLOSUM linear",
    "route_b_simple_sequence_context": "simple WT/entropy/context",
    "route_b_carp_context_shuffled": "CARP context shuffled",
    "route_b_target_conditionally_shuffled": "target conditionally shuffled",
    "direct_legacy_pca_rank1": "direct residual PCA rank 1",
    "direct_legacy_pca_rank3": "direct residual PCA rank 3",
    "direct_legacy_pca_rank5": "direct residual PCA rank 5",
    "direct_legacy_pca_rank16": "direct residual PCA rank 16",
    "direct_legacy_rms_shrinkage": "direct residual RMS shrinkage",
}


def build_mechanism_report(config: MechanismStudyConfig) -> dict[str, Path]:
    """Render the frozen evidence, mechanism audit, and immutable decision."""

    evaluation = config.paths.run_dir / "evaluation"
    panel = config.paths.run_dir / "panel"
    output = config.paths.run_dir / "reports"
    output.mkdir(parents=True, exist_ok=True)

    domains = pd.read_parquet(panel / "domains.parquet")
    variants = pd.read_parquet(panel / "variants.parquet")
    queries = pd.read_parquet(panel / "query_rows.parquet")
    structures = pd.read_parquet(config.paths.run_dir / "mif_requests" / "structures.parquet")
    scores = pd.read_parquet(config.paths.run_dir / "mif" / "scores.parquet")
    validity = pd.read_parquet(evaluation / "condition_validity.parquet")
    summary = pd.read_parquet(evaluation / "increment_summary.parquet")
    route_margins = pd.read_parquet(evaluation / "route_b_margin_summary.parquet")
    subgroup = pd.read_parquet(evaluation / "gly_pro_subgroup_summary.parquet")
    decisions = pd.read_parquet(evaluation / "audit_decisions.parquet").iloc[0]
    qc = pd.read_parquet(evaluation / "quality_controls.parquet").set_index("check")
    training = pd.read_parquet(evaluation / "route_b_training.parquet")

    natural_domains = domains.loc[domains["stratum"].eq("natural")]
    de_novo_domains = domains.loc[domains["stratum"].eq("de_novo")]
    natural_variants = variants.loc[variants["stratum"].eq("natural")]
    de_novo_variants = variants.loc[variants["stratum"].eq("de_novo")]
    route_b_spearman = _summary_row(summary, ROUTE_B_METHOD, "spearman_increment")
    route_b_ndcg = _summary_row(summary, ROUTE_B_METHOD, "ndcg_at_10_percent_increment")
    legacy_spearman = _summary_row(summary, "legacy_direct_contrast", "spearman_increment")
    legacy_ndcg = _summary_row(summary, "legacy_direct_contrast", "ndcg_at_10_percent_increment")
    smooth_half = _summary_row(summary, "contrast__smooth_coordinate__0.5", "spearman_increment")
    variance_alpha = float(qc.loc["paired_variance_matched_alpha", "estimate"])
    training_row = training.loc[training["method"].eq(ROUTE_B_METHOD)].iloc[0]

    report = f"""# mechanism study：分布内反事实与去噪机制审计

_MARGIN computational study · 2026-08-15_

---

## 摘要

mechanism study 已完成。它不重开 counterfactual study，也不把本轮面板冒充新的外部确认；它专门审计两个仍未解决的机制问题：较温和、较接近原生条件分布的结构反事实能否给出跨家族稳定的增量，以及 counterfactual study 的 CARP Route B 是否真的识别了结构残差，而不只是低维稳定性先验。

本轮在评分前冻结 32 个 Megascale 域（16 natural、16 de novo，后者覆盖 8 个设计家族），共 {len(variants):,} 个单突变、{len(queries):,} 个查询位置。对每个域生成 contact deletion、低频平滑坐标扰动、受约束 contact reassignment、descriptor-matched real backbone、历史强重连和刚体不变性对照，共 {len(structures):,} 个 MIF 请求、{len(scores):,} 行评分。Megascale 数据及其发布结构来自公开数据集[^1][^2]。

结论是一个清晰的机制收缩：**确实找到了分布位移很小的反事实，但没有任何冻结家族同时通过 seed-to-seed 动作可靠性门槛。** delete 5%、delete 10% 与 coordinate 0.25 Å 的 JSD/熵条件温和；其 seed 动作 Spearman 中位数却只有 0.127、0.236 与 0.213，均低于冻结的 0.50。因而不能把历史强重连信号重新命名为已建立的 in-distribution structure contrast。

反事实减法的效应随条件偏移增强，但所有主条件都未优于完整 paired-only 对照矩阵。Route B 本身仍为正：ΔSpearman `{route_b_spearman["estimate"]:+.4f}`（95% CI `[{route_b_spearman["ci_low"]:+.4f}, {route_b_spearman["ci_high"]:+.4f}]`），ΔNDCG@10% `{route_b_ndcg["estimate"]:+.4f}`（`[{route_b_ndcg["ci_low"]:+.4f}, {route_b_ndcg["ci_high"]:+.4f}]`）；但 global WT→mutant、简单序列上下文与 direct residual PCA/shrinkage 控制更强。因此最窄、证据相称的解释为：**Route B 可用作低维稳定性先验；现有证据尚未识别结构恢复。**

机器判决为 `{decisions["mechanisms_interpretation"]}`。counterfactual study 决策保持 `{decisions["registered_counterfactuals_decision"]}`，主路线保持 `{decisions["primary_route"]}`，`SELECTIVE_ROUTING = NOT_AUTHORIZED`。

**关键词：** in-distribution counterfactual；paired-only calibration；denoising audit；CARP；MIF；dense NDCG

## 冻结问题与边界

counterfactual study 的正式判决在 mechanism study 开始前已成为不可修改输入：Route A 只因确认性 NDCG 下界失败，Route B 单独通过，但 A 是打开 sequence-control branch 的前置条件。mechanism study 允许解释这次分叉，不允许改写它。

本轮固定回答四个问题：

1. 是否存在同时满足低 JSD、低熵偏移、至少 80% 域通过和 seed 动作 Spearman ≥0.50 的反事实条件？
2. 至少两个已通过分布条件的反事实家族，是否都在 Spearman 与 NDCG@10% 上产生正方向？
3. 反事实减法是否在两个指标上严格优于每一个 paired-only 校准对照？
4. CARP rank-16 是否严格优于 global substitution、物化性质、简单序列上下文、两类 shuffle、direct PCA rank 1/3/5/16 与 RMS shrinkage？

涉及 Gly/Pro 的替换被预声明为边界分析，但不能创建新 gate。全部工作限定为计算审计。

```mermaid
flowchart LR
    accTitle: mechanism study frozen mechanism audit
    accDescr: Mild counterfactuals are tested for distribution and seed reliability, subtraction is compared with paired-only calibration, and Route B is compared with identification controls while the counterfactual study decision remains immutable.

    lock[( Frozen counterfactual study decision)] --> panel[( Dense 32-domain panel)]
    panel --> cf[ Four plausible CF families]
    panel --> paired[ Paired-only controls]
    panel --> routeb[ Route B identification controls]
    cf --> validity{{ID + seed reliable?}}
    validity -->|No| no_cf[ No registered ID family]
    paired --> unique{{Subtraction uniquely better?}}
    unique -->|No| no_unique[ Unique value not established]
    routeb --> recovery{{Beats every control?}}
    recovery -->|No| prior[ Low-dimensional stability prior]
    no_cf --> retain([ Retain counterfactual study / no sequence-control branch])
    no_unique --> retain
    prior --> retain

    classDef data fill:#f3f4f6,stroke:#6b7280,stroke-width:2px,color:#1f2937
    classDef process fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef decision_style fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12
    classDef stop fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#7f1d1d
    class lock,panel data
    class cf,paired,routeb process
    class validity,unique,recovery decision_style
    class no_cf,no_unique,prior,retain stop
```

## 数据、反事实与推断

### 密集的同平台机制面板

| 分层 | 域 | 单突变 | 查询位置 | 选择边界 |
| --- | ---: | ---: | ---: | --- |
| natural | {len(natural_domains)} | {len(natural_variants):,} | {natural_variants[["domain_id", "position"]].drop_duplicates().shape[0]:,} | ≥500 变体、≥30 位置 |
| de novo | {len(de_novo_domains)} | {len(de_novo_variants):,} | {de_novo_variants[["domain_id", "position"]].drop_duplicates().shape[0]:,} | 每个冻结设计家族 2 个域 |
| 总计 | {len(domains)} | {len(variants):,} | {len(queries):,} | 结局幅度不参与域选择 |

所有序列与发布结构链一致，N/CA/C/O backbone 完整，变体 WT 与锁定序列一致。面板与 observability study/C/D 执行 exact/near-duplicate 排除（identity ≥80%、双向 coverage ≥90%）。该面板来自项目已经使用过的 Megascale 测量平台，故定位为 **in-distribution mechanism audit**；独立外部复现需要新的测量平台。

### 反事实与教师

- **Contact deletion：** 仅删除原生接触的 5%、10% 或 20%，保留其余几何特征；每级 5 seeds。
- **Smooth coordinate：** 三个低频平移模态，目标 Cα RMSD 0.25/0.50/1.00 Å，并约束相邻 Cα 距离变化 ≤0.25 Å；每级 5 seeds。
- **Constrained reassignment：** 在拓扑匹配非边之间重分配 10% 接触特征，保持局部操作约束；5 seeds。
- **Matched real：** 从未进入目标面板的真实发布 backbone 中，以长度和结构描述符匹配，每域 3 个 exact-length 连续片段。
- **Legacy OOD reference：** counterfactual study 的 5 swaps/contact degree-preserving 重连，只作历史参照。

所有条件都用同一 plain MIF 教师进行 leave-one-position-out 候选评分；MIF 的公开方法将蛋白骨架表示为几何图并据此重建遮蔽序列[^3]。刚体变换 QC 的最大绝对 log-probability 差为 `{float(qc.loc["rigid_transform_logp_invariance", "estimate"]):.3e}`，通过实现不变性检查；uniform subtraction 的动作差为 `{float(qc.loc["uniform_subtraction_action_identity", "estimate"]):.3e}`，验证了常数背景在 mutant-minus-WT 中严格抵消。

### 模型与统计

序列基线为冻结 ESM2-150M masked log-probability action[^4]。所有方法都按域先计算 Spearman、full NDCG、NDCG@10%、top-10% stabilizing recall 与 calibration slope，再让 32 个域等权汇总；95% 区间来自 5,000 次固定 natural/de novo 数量的分层 domain bootstrap。NDCG 同时报告 full 与 10%，避免 counterfactual study 小面板顶部估计不稳定。

CARP Route B 使用冻结 CARP-640M 表征；CARP 是卷积式蛋白序列预训练模型[^5]。其 rank-16 reduced-rank Ridge 仅在既有 CATH 开发集的 {int(training_row["training_domains"])} 个域、{int(training_row["training_rows"]):,} 个位置上拟合，未读取任何 mechanism study 稳定性标签。AAindex 控制使用公开的氨基酸性质数据库[^6]。

## 反事实有效性结果

{_validity_table(validity)}

delete 5%、delete 10% 与 coordinate 0.25 Å 表明部分新反事实仍位于冻结分布阈值内：三者的总体 JSD、绝对熵偏移和域覆盖均满足阈值。失败项是独立的 seed 可靠性。换言之，条件分布可以温和，但从这个反事实抽样得到的 mutant-vs-WT residual action 仍不稳定；这足以拒绝一个可重复的结构 estimand，教师在这些结构上的有效性仍需独立判断。

坐标扰动呈受控剂量关系：0.25/0.50/1.00 Å 的 JSD 为 0.018/0.060/0.128，seed 相关随强度上升；历史重连的 JSD 0.321、熵偏移 0.951，明显处于饱和 OOD 区。descriptor-matched real backbone 虽然来自真实结构，但目标序列与 donor backbone 并不天然配对，JSD 0.364，因此“真实坐标”不能自动推出“分布内条件”。

### 排序增量随分布偏移增强

{_counterfactual_performance_table(summary)}

最温和条件的效应很小；随删除比例或坐标 RMSD 增加，Spearman 增量上升。coordinate 0.50 Å 达到 ΔSpearman `{smooth_half["estimate"]:+.4f}`（`[{smooth_half["ci_low"]:+.4f}, {smooth_half["ci_high"]:+.4f}]`），但其熵偏移 0.263 超过 0.25、仅 11/32 域通过且 seed 相关 0.393，不能后验升级为主条件。历史 OOD 重连在密集面板上的 ΔSpearman `{legacy_spearman["estimate"]:+.4f}`、ΔNDCG@10% `{legacy_ndcg["estimate"]:+.4f}`，区间均为正；这解决了“历史 NDCG 是否只是稀疏估计噪声”的问题，却同时显示强信号集中在偏离最大的条件，不能恢复结构机制主张。

![mechanism study main results](../figures/mechanisms_main.png)

_图 1｜A、B：各反事实的 paired-vs-counterfactual JSD 与绝对熵偏移，虚线为冻结阈值。C、D：主要反事实、paired-only 与 Route B 方法相对 sequence-only 的域等权 Spearman 和 NDCG@10% 增量及 95% 分层 bootstrap CI。图源数据随 PNG/PDF/SVG 一并导出。_

## Paired-only 与减法唯一性

{_paired_performance_table(summary)}

paired MIF 本身和三种不读取 mechanism study 结局的校准都给出强增量。所有四个冻结主反事实在 Spearman 或 NDCG@10% 上至少输给一个 paired-only 对照，且没有任何主家族已通过 ID/seed 门槛。因此 `unique_subtraction_supported = {bool(decisions["unique_subtraction_supported"])}`。

最强的主反事实 matched real 相对 `sequence + paired MIF (α=1)` 的 Spearman margin 是 {_margin_text(config.paths.run_dir / "evaluation" / "subtraction_margin_summary.parquet", "contrast__matched_real_structure__descriptor_matched", "sequence_plus_mif_paired_alpha_1", "spearman")}，NDCG@10% margin 是 {_margin_text(config.paths.run_dir / "evaluation" / "subtraction_margin_summary.parquet", "contrast__matched_real_structure__descriptor_matched", "sequence_plus_mif_paired_alpha_1", "ndcg_at_10_percent")}。相对 variance-matched paired 的 Spearman margin接近零但区间跨零。最合理的解释是 paired MIF 提供了与稳定性相关的强校准方向，而减去未被验证为稳定 estimand 的 counterfactual 往往删除有用成分；本轮不能声称 subtraction 有独立机制价值。variance-matched α 在结果揭示前由预测尺度确定为 `{variance_alpha:.4f}`。

## Route B 识别审计

{_route_b_performance_table(summary)}

CARP-B 包含有效信号：它相对 sequence-only 的 Spearman 和 NDCG@10% 区间都为正；相对 CARP-context shuffled 的 Spearman margin 为 {_route_margin_text(route_margins, "route_b_carp_context_shuffled", "spearman")}，相对 target-conditionally-shuffled 为 {_route_margin_text(route_margins, "route_b_target_conditionally_shuffled", "spearman")}。这支持真实的上下文对齐信息。

但识别标准要求同时赢过每一个冻结控制。CARP-B 相对 global WT→mutant matrix 的 Spearman margin 为 {_route_margin_text(route_margins, "route_b_global_wt_mutant_matrix", "spearman")}，相对 simple sequence context 为 {_route_margin_text(route_margins, "route_b_simple_sequence_context", "spearman")}，相对 direct PCA rank 16 为 {_route_margin_text(route_margins, "direct_legacy_pca_rank16", "spearman")}；均显著为负。其 NDCG@10% 也没有在 shuffled controls 上建立严格 margin。因而 `route_b_structure_recovery_supported = {bool(decisions["route_b_structure_recovery_supported"])}`。

这组结果把 counterfactual study 的“Route B PASS”解释得更精确：它是正向、可复用的序列预测器，但主要信号可被 global substitution、简单上下文和对 direct residual 的低秩/缩放处理解释。把它称为 `LOW_DIMENSIONAL_STABILITY_PRIOR_NOT_STRUCTURE_RECOVERY` 比“序列恢复了结构教师残差”更符合控制结果。

![mechanism study mechanism audit](../figures/mechanisms_mechanism.png)

_图 2｜A：Spearman 增量随 JSD 增长。B：各条件 seed 动作可靠性与冻结 0.50 阈值。C：CARP-B 相对 Route B controls 的逐域 margin。D：涉及 Gly/Pro 与其他替换的预声明边界。_

## Gly/Pro 预声明边界

{_boundary_table(subgroup)}

历史 direct contrast 和 matched-real contrast 在其他替换上很强，但涉及 Gly/Pro 时 Spearman 增量转负；CARP-B 也从其他替换的正增量降到接近零。NDCG@10% 的若干 Gly/Pro 区间仍跨零，因此这里只报告稳定的异质性边界，不把 Gly/Pro 建成路由规则或新 gate。

## 限制与可解释范围

- 面板在域层面与早期确认集去重且足够密集，但与 counterfactual study 的 de novo 子集共享 Megascale 测量体系；它解决机制分辨率，不提供跨平台外部确认。
- natural/de novo 是来源与设计分层，不等于覆盖一般蛋白家族。16 个 de novo 域由 8 个冻结设计家族均衡抽取，仍不能外推到所有设计分布。
- matched-real 使用真实 backbone，却没有目标序列与 donor backbone 的天然配对；它的高 seed 相关和强预测增量不能抵消其大 JSD/熵偏移。
- 本轮估计针对固定 plain MIF、ESM2-150M、CARP-640M 和 `ddG_ML` 方向；不建立跨教师或跨表型的一般因果结构结论。
- seed reliability 门槛直接约束 residual action，并同时检查教师输出分布。温和 JSD 加低 seed 相关构成实质阴性；数据完整性检查已通过。
- Gly/Pro 是预声明描述性边界；它不修改项目门控。

## 冻结判决

最终 mechanism study 解释为：**`{decisions["mechanisms_interpretation"]}`**。

1. **没有冻结反事实家族被建立为 ID-compatible。** 分布温和条件存在，但 seed 动作可靠性不足；robust cross-family contrast 为 FAIL。
2. **反事实减法的唯一价值未建立。** paired-only 校准更强，且主条件没有通过 ID 前提。
3. **Route B 保留为正预测器，但机制降格。** 结果支持低维稳定性先验，不支持 target-specific structure recovery。
4. **Gly/Pro 负向边界重复出现，但不创建 gate。**
5. **counterfactual study 不变。** `registered_counterfactuals_decision_modified = {bool(decisions["registered_counterfactuals_decision_modified"])}`；`selective_routing_authorized = {bool(decisions["selective_routing_authorized"])}`；主路线继续为 `{decisions["primary_route"]}`。

因此，mechanism study 关闭的是“用较合理反事实即可挽救通用结构残差减法、或把 CARP-B 解释为结构恢复”的机制叙事；它不关闭结构条件化主路线，也不抹去 paired MIF 与 Route B 的真实预测信号。

## 复现与产物

固定执行顺序如下；冻结脚本只允许在正式 mechanism study 模型评分出现前运行：

```bash
PYTHONPATH=src conda run -n margin-models python scripts/workflows/mechanisms/prepare.py
PYTHONPATH=src conda run -n margin-models python scripts/workflows/mechanisms/freeze_protocol.py
PYTHONPATH=src conda run -n margin-models python scripts/workflows/mechanisms/prepare_counterfactuals.py
PYTHONPATH=src conda run -n margin-models python scripts/workflows/mechanisms/run_mif.py --device cuda:0
PYTHONPATH=src conda run -n margin-models python scripts/workflows/mechanisms/run_representations.py --device cuda:0
PYTHONPATH=src conda run -n margin-models python scripts/workflows/mechanisms/derive_cath_esm2_logp.py --device cuda:0
PYTHONPATH=src conda run -n margin-models python scripts/workflows/mechanisms/evaluate.py
PYTHONPATH=src conda run -n margin-models python scripts/workflows/mechanisms/build_figures.py
PYTHONPATH=src conda run -n margin-models python scripts/workflows/mechanisms/build_report.py
```

关键紧凑产物位于 `runs/mechanisms/`：协议锁、panel、MIF score、25-method/32-domain 指标、5,000-bootstrap 汇总、control margins、Gly/Pro 边界、图源数据和本报告。大型模型、表征和矩阵保存在 `{config.paths.storage_dir}`。

## 参考文献

[^1]: Tsuboyama, K. et al. (2023). “Mega-scale experimental analysis of protein folding stability in biology and design.” _Nature_ 620, 434–444. <https://doi.org/10.1038/s41586-023-06328-6>

[^2]: Tsuboyama, K. et al. (2023). “Mega-scale experimental analysis of protein folding stability in biology and design: data release.” _Zenodo_. <https://doi.org/10.5281/zenodo.7992926>

[^3]: Yang, K. K., Zanichelli, N. & Yeh, H. (2023). “Masked inverse folding with sequence transfer for protein representation learning.” _Protein Engineering, Design and Selection_ 36, gzad015. <https://doi.org/10.1093/protein/gzad015>

[^4]: Lin, Z. et al. (2023). “Evolutionary-scale prediction of atomic-level protein structure with a language model.” _Science_ 379, 1123–1130. <https://doi.org/10.1126/science.ade2574>

[^5]: Yang, K. K., Fusi, N. & Lu, A. X. (2024). “Convolutions are competitive with transformers for protein sequence pretraining.” _Cell Systems_ 15, 286–294.e2. <https://doi.org/10.1016/j.cels.2024.01.008>

[^6]: Kawashima, S., Ogata, H. & Kanehisa, M. (1999). “AAindex: Amino Acid Index Database.” _Nucleic Acids Research_ 27, 368–369. <https://doi.org/10.1093/nar/27.1.368>

---

_最后更新：2026-08-15_
"""
    report_path = output / "mechanisms_report.md"
    report_path.write_text(report, encoding="utf-8")
    manifest_path = output / "manifest.json"
    write_json(
        manifest_path,
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "mechanisms_interpretation": str(decisions["mechanisms_interpretation"]),
            "registered_counterfactuals_decision": str(
                decisions["registered_counterfactuals_decision"]
            ),
            "selective_routing_authorized": bool(decisions["selective_routing_authorized"]),
            "report": str(report_path),
            "source_tables": [
                str(evaluation / "audit_decisions.parquet"),
                str(evaluation / "condition_validity.parquet"),
                str(evaluation / "increment_summary.parquet"),
                str(evaluation / "subtraction_margin_summary.parquet"),
                str(evaluation / "route_b_margin_summary.parquet"),
                str(evaluation / "gly_pro_subgroup_summary.parquet"),
            ],
        },
    )
    return {"report": report_path, "manifest": manifest_path}


def _summary_row(summary: pd.DataFrame, method: str, metric: str) -> pd.Series:
    selected = summary.loc[
        summary["method"].eq(method)
        & summary["metric"].eq(metric)
        & summary["scope"].eq("all_stratum_preserving")
    ]
    if len(selected) != 1:
        raise ValueError(f"missing mechanism study summary row for {method}/{metric}")
    return selected.iloc[0]


def _interval(row: pd.Series) -> str:
    return f"{row['estimate']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"


def _validity_table(validity: pd.DataFrame) -> str:
    labels = {
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
    order = {key: index for index, key in enumerate(labels)}
    selected = validity.assign(
        _order=[
            order[(str(row.counterfactual_family), str(row.condition_level))]
            for row in validity.itertuples(index=False)
        ]
    ).sort_values("_order")
    rows = [
        "| 条件 | median JSD | median abs(ΔH) | 通过域 | seed action ρ | ID-compatible |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in selected.itertuples(index=False):
        key = (str(row.counterfactual_family), str(row.condition_level))
        reliability = (
            "NA"
            if pd.isna(row.median_seed_action_spearman)
            else f"{row.median_seed_action_spearman:.3f}"
        )
        rows.append(
            f"| {labels[key]} | {row.median_jsd_nats:.3f} | "
            f"{row.median_absolute_entropy_shift_nats:.3f} | "
            f"{int(row.passing_domains)}/{int(row.n_domains)} | {reliability} | "
            f"{'PASS' if bool(row.id_compatible) else 'FAIL'} |"
        )
    return "\n".join(rows)


def _counterfactual_performance_table(summary: pd.DataFrame) -> str:
    methods = [
        "contrast__contact_deletion__0.05",
        "contrast__contact_deletion__0.1",
        "contrast__contact_deletion__0.2",
        "contrast__smooth_coordinate__0.25",
        "contrast__smooth_coordinate__0.5",
        "contrast__smooth_coordinate__1",
        "contrast__constrained_reassignment__0.1",
        "contrast__matched_real_structure__descriptor_matched",
        "legacy_direct_contrast",
    ]
    return _performance_table(summary, methods)


def _paired_performance_table(summary: pd.DataFrame) -> str:
    return _performance_table(summary, list(PAIRED_CONTROLS))


def _route_b_performance_table(summary: pd.DataFrame) -> str:
    methods = [
        ROUTE_B_METHOD,
        "route_b_global_wt_mutant_matrix",
        "route_b_grantham_aaindex_blosum_linear",
        "route_b_simple_sequence_context",
        "route_b_carp_context_shuffled",
        "route_b_target_conditionally_shuffled",
        "direct_legacy_pca_rank1",
        "direct_legacy_pca_rank3",
        "direct_legacy_pca_rank5",
        "direct_legacy_pca_rank16",
        "direct_legacy_rms_shrinkage",
    ]
    return _performance_table(summary, methods)


def _performance_table(summary: pd.DataFrame, methods: list[str]) -> str:
    rows = [
        "| 方法 | ΔSpearman [95% CI] | Δfull NDCG [95% CI] | ΔNDCG@10% [95% CI] |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method in methods:
        spearman = _summary_row(summary, method, "spearman_increment")
        full_ndcg = _summary_row(summary, method, "full_ndcg_increment")
        top_ndcg = _summary_row(summary, method, "ndcg_at_10_percent_increment")
        rows.append(
            f"| {METHOD_LABELS[method]} | {_interval(spearman)} | "
            f"{_interval(full_ndcg)} | {_interval(top_ndcg)} |"
        )
    return "\n".join(rows)


def _margin_text(path: Path, candidate: str, comparator: str, metric: str) -> str:
    margins = pd.read_parquet(path)
    row = margins.loc[
        margins["candidate_method"].eq(candidate)
        & margins["comparator_method"].eq(comparator)
        & margins["metric"].eq(metric)
    ]
    if len(row) != 1:
        raise ValueError(f"missing subtraction margin {candidate}/{comparator}/{metric}")
    return _interval(row.iloc[0])


def _route_margin_text(margins: pd.DataFrame, comparator: str, metric: str) -> str:
    row = margins.loc[
        margins["candidate_method"].eq(ROUTE_B_METHOD)
        & margins["comparator_method"].eq(comparator)
        & margins["metric"].eq(metric)
    ]
    if len(row) != 1:
        raise ValueError(f"missing Route B margin {comparator}/{metric}")
    return _interval(row.iloc[0])


def _boundary_table(subgroup: pd.DataFrame) -> str:
    methods = [
        "legacy_direct_contrast",
        "contrast__matched_real_structure__descriptor_matched",
        ROUTE_B_METHOD,
    ]
    rows = [
        "| 方法 | Gly/Pro ΔSpearman [95% CI] | 其他替换 ΔSpearman [95% CI] |",
        "| --- | ---: | ---: |",
    ]
    for method in methods:
        selected = subgroup.loc[
            subgroup["method"].eq(method) & subgroup["metric"].eq("spearman_increment")
        ].set_index("subgroup")
        rows.append(
            f"| {METHOD_LABELS[method]} | "
            f"{_interval(selected.loc['involves_glycine_or_proline'])} | "
            f"{_interval(selected.loc['other_substitutions'])} |"
        )
    return "\n".join(rows)
