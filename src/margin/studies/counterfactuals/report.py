"""Build the source-grounded counterfactual study technical report."""

# Markdown tables and citation entries are intentionally kept on single source lines.
# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path

import pandas as pd

from margin.provenance import runtime_manifest, sha256_file, write_json
from margin.studies.counterfactuals.config import CounterfactualStudyConfig
from margin.studies.counterfactuals.evaluation import (
    ROUTE_A_PRIMARY,
    ROUTE_A_REPLICATION,
    ROUTE_B_PRIMARY,
    ROUTE_B_REPLICATION,
)


def build_counterfactual_report(config: CounterfactualStudyConfig) -> dict[str, Path]:
    """Render the frozen decision, evidence, and exploratory analyses as Markdown."""

    evaluation = config.paths.run_dir / "evaluation"
    mechanisms = config.paths.run_dir / "mechanisms"
    panel = config.paths.run_dir / "panel"
    output = config.paths.run_dir / "reports"
    output.mkdir(parents=True, exist_ok=True)
    decisions = pd.read_parquet(evaluation / "route_decisions.parquet").set_index("route")
    project = pd.read_parquet(evaluation / "project_decision.parquet").iloc[0]
    summary = pd.read_parquet(evaluation / "increment_summary.parquet")
    random_summary = pd.read_parquet(evaluation / "random_control_summary.parquet")
    domains = pd.read_parquet(panel / "domains.parquet")
    variants = pd.read_parquet(panel / "variants.parquet")
    training = pd.read_parquet(evaluation / "route_b_training.parquet")
    pca = pd.read_parquet(mechanisms / "residual_pca_variance.parquet")
    action = pd.read_parquet(mechanisms / "direct_predicted_action_alignment.parquet")
    ood = pd.read_parquet(mechanisms / "ood_position_rows.parquet")
    aaindex = pd.read_parquet(mechanisms / "aaindex_correlations.parquet")
    strata = pd.read_parquet(mechanisms / "stratified_summary.parquet")
    coverage = pd.read_parquet(mechanisms / "analysis_coverage.parquet")
    route_a = decisions.loc["route_a_direct_structure"]
    route_b = decisions.loc["route_b_sequence_predicted"]
    intensity = _intensity_table(summary)
    replication = _replication_table(summary)
    ood_table = _ood_table(ood)
    action_table = _action_table(action)
    aaindex_table = _aaindex_table(aaindex)
    mutation_table = _mutation_table(strata)
    pca3 = float(pca.iloc[2]["cumulative_explained_variance_ratio"])
    pca5 = float(pca.iloc[4]["cumulative_explained_variance_ratio"])
    natural = domains.loc[domains["stratum"].eq("natural")]
    de_novo = domains.loc[domains["stratum"].eq("de_novo")]
    natural_variants = variants.loc[variants["stratum"].eq("natural")]
    de_novo_variants = variants.loc[variants["stratum"].eq("de_novo")]
    control = random_summary.loc[random_summary["metric"].eq("spearman_increment_margin")].iloc[0]
    report = f"""# counterfactual study：反事实结构残差的独立验证

_MARGIN computational study · 2026-08-15_

---

## 摘要

counterfactual study 在任何锁定面板模型评分之前，冻结了两条互不混称的路线：测试时使用结构的
`direct_mif_paired_minus_counterfactual`（Route A），以及测试时只使用序列的
`carp_predicted_mif_paired_minus_counterfactual`（Route B）。确认面板包含 50 个与此前
52-assay 面板无冻结边界内序列或结构重叠的域，共 13,609 个单突变；34 个天然 S669 域
与 16 个 de novo Megascale 域均按域等权进入推断。S669 来自独立、人工核验的稳定性
变化基准[^1]；Megascale 数据与结构模型来自公开的 2023 数据发布[^2][^3]。

结果形成了一个非对称分叉。Route A 的 Spearman 增量为
`{route_a["spearman_increment"]:+.4f}`（95% CI
`[{route_a["spearman_ci_low"]:+.4f}, {route_a["spearman_ci_high"]:+.4f}]`），top-10%
稳定化召回增量为 `{route_a["topk_increment"]:+.4f}`，且显著优于 matched-random
对照；但 NDCG 增量 `{route_a["ndcg_increment"]:+.4f}` 的 95% CI 下界为
`{route_a["ndcg_ci_low"]:+.4f}`，因此 Route A **FAIL**。Route B 的 Spearman、NDCG、
top-k、域多数、天然/de novo 分层和 circular replication 条件全部满足，单独为
**PASS**。冻结分支规定只要 A 失败就不得扩展新路线，因此最终判决是
`{project["decision"]}`：不启动 sequence-control branch，保留 generalization study 的
`PIVOT_SELECTIVE_STRUCTURE_CONDITIONED`，通用 sequence-only 迁移仍保持关闭。

**关键词：** counterfactual residual；protein stability；MIF；CARP；domain bootstrap；
confirmatory validation

---

## 问题与冻结假设

generalization study 已表明，跨模型的“结构教师减序列学生”残差会混入模型谱系、氨基酸先验与校准
差异。counterfactual study 改为同一个 plain MIF 教师内部的 paired-minus-counterfactual 差分。
MIF 是以骨架结构为条件、重建被遮蔽序列的结构图模型；MIF-ST 另行加入序列迁移，而本轮
确认性教师明确使用不含该分支的 plain MIF。[^6]

本轮回答三个问题：

1. **Route A：** 直接的 MIF paired-minus-counterfactual 动作是否在新稳定性域上改善
   ESM2-150M 序列基线？
2. **Route B：** 只用 CARP-640M 序列表征预测该反事实残差，是否能在同一新面板重复？
3. **稳健性：** 正结果是否同时跨天然/de novo 域、两种反事实和 matched-random 对照？

范围限定为计算评估。

## 方法

### 预冻结与独立面板

协议锁文件状态为 `FROZEN_BEFORE_LOCKED_PANEL_MODEL_SCORING`。面板选择不使用模型分数，
也不汇总稳定性结局。

| 分层 | 来源 | 域 | 单突变 | 结构来源 |
| --- | --- | ---: | ---: | --- |
| 天然 | S669 | {len(natural)} | {len(natural_variants):,} | 实验 PDB |
| de novo | Megascale 2023 | {len(de_novo)} | {len(de_novo_variants):,} | 发布的模型结构归档 |
| 总计 | 两者 | {len(domains)} | {len(variants):,} | 混合 |

天然域排除与此前 52 assays 的 CATH-H/T 重叠，并执行 30% identity、双向 80% coverage
序列边界。de novo 域没有官方 CATH 标签，因此使用 Foldseek 对此前 52 个结构做直接
结构检索；双向 TM-score ≥0.50 且双向 coverage ≥0.80 时排除。Foldseek 以 3Di 结构字母
进行快速结构检索。[^5] CATH 分类固定在 4.4.0。[^4]

### 两条路线

对位置 `i` 的 WT `a` 与 mutant `b`，序列基线为 ESM2-150M 严格 leave-one-position-out
对数概率差。ESM2 是单序列蛋白语言模型；本轮使用固定的 150M、30-layer 版本。[^7]

- **Route A：** `s_seq + [r_MIF,paired(b)-r_MIF,paired(a)] -
  [r_MIF,counterfactual(b)-r_MIF,counterfactual(a)]`
- **Route B：** `s_seq + r_CARP,predicted(b)-r_CARP,predicted(a)`

主反事实是每条原始 contact 请求 5 次 degree-preserving double-edge swap；独立复现是
最大可行循环 residue-to-geometry permutation。Route B 的 rank-16 reduced-rank Ridge
只在既有 CATH native query 上拟合：{int(training["training_domains"].iloc[0])} 个开发域、
{int(training["training_rows"].iloc[0]):,} 个位置，`ridge_alpha=10`，不使用任何 counterfactual study
稳定性标签。

### 指标与推断

每个蛋白域贡献一个等权估计。固定指标为 Spearman、全排序 NDCG、稳定化 top-10%
召回与 calibration slope；报告 hybrid-minus-sequence-only 增量。区间使用 5,000 次、
固定天然/de novo 域数量的 percentile bootstrap。matched-random 对照在同一域和 burial
类别内置换完整 20-AA 残差向量，共 20 次。未定义的域级 top-k 指标仅从该指标中排除。

```mermaid
flowchart LR
    accTitle: counterfactual study frozen decision path
    accDescr: Route A fails its NDCG confidence-bound requirement while Route B passes; the frozen branch therefore retains the generalization study decision and does not open sequence-control branch.

    panel[( Frozen 50-domain panel)] --> route_a[ Route A direct residual]
    panel --> route_b[ Route B predicted residual]
    route_a --> gate_a{{ A passed?}}
    route_b --> gate_b{{ B passed?}}
    gate_a -->|No| retain([ Retain generalization study decision])
    gate_a -->|Yes| gate_b
    gate_b -->|Yes| open([ Open narrow sequence-control branch])
    gate_b -->|No| structure([ Structure-conditioned only])

    classDef data fill:#f3f4f6,stroke:#6b7280,stroke-width:2px,color:#1f2937
    classDef process fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef decision_style fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12
    classDef stop fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#7f1d1d

    class panel data
    class route_a,route_b process
    class gate_a,gate_b decision_style
    class retain stop
```

## 结果

### Route A 只在冻结 NDCG 下界上失败

| 条件 | 观察值 | 判定 |
| --- | ---: | --- |
| Spearman 增量 CI 下界 > 0 | {route_a["spearman_ci_low"]:+.4f} | {_pass(route_a["pass_spearman_ci"])} |
| NDCG 增量 CI 下界 > 0 | {route_a["ndcg_ci_low"]:+.4f} | {_pass(route_a["pass_ndcg_ci"])} |
| top-k 增量 > 0 | {route_a["topk_increment"]:+.4f} | {_pass(route_a["pass_topk_point"])} |
| 正 Spearman 域比例 ≥ 0.51 | {route_a["positive_domain_fraction"]:.2%} | {_pass(route_a["pass_majority_domains"])} |
| 天然 / de novo 点估计均 > 0 | {route_a["natural_spearman_increment"]:+.4f} / {route_a["de_novo_spearman_increment"]:+.4f} | {_pass(route_a["pass_natural_point"] and route_a["pass_de_novo_point"])} |
| circular 三指标均 > 0 | 见下表 | {_pass(route_a["pass_replication_spearman"] and route_a["pass_replication_ndcg"] and route_a["pass_replication_topk"])} |
| 优于 matched-random，CI 下界 > 0 | {float(control["estimate"]):+.4f} [{float(control["ci_low"]):+.4f}, {float(control["ci_high"]):+.4f}] | {_pass(route_a["pass_random_control_margin"])} |

Route A 对整体排序的改善很大且域级稳健，但 NDCG 的天然域不确定性使确认性下界跨零。
整体排序信号保持为正；顶部排序证据未达到事先要求的强度。

### Route B 单独通过，但不能越过 A 前置条件

| 条件 | 观察值 | 判定 |
| --- | ---: | --- |
| Spearman 增量 | {route_b["spearman_increment"]:+.4f} [{route_b["spearman_ci_low"]:+.4f}, {route_b["spearman_ci_high"]:+.4f}] | {_pass(route_b["pass_spearman_ci"])} |
| NDCG 增量 | {route_b["ndcg_increment"]:+.4f} [{route_b["ndcg_ci_low"]:+.4f}, {route_b["ndcg_ci_high"]:+.4f}] | {_pass(route_b["pass_ndcg_ci"])} |
| top-k 增量 | {route_b["topk_increment"]:+.4f} | {_pass(route_b["pass_topk_point"])} |
| 正 Spearman 域比例 | {route_b["positive_domain_fraction"]:.2%} | {_pass(route_b["pass_majority_domains"])} |
| 天然 / de novo Spearman 增量 | {route_b["natural_spearman_increment"]:+.4f} / {route_b["de_novo_spearman_increment"]:+.4f} | {_pass(route_b["pass_natural_point"] and route_b["pass_de_novo_point"])} |
| circular replication | 三个点估计均为正 | {_pass(route_b["pass_replication_spearman"] and route_b["pass_replication_ndcg"] and route_b["pass_replication_topk"])} |

{replication}

![counterfactual study main results](../figures/counterfactuals_main.png)

_图 1｜A、B：四个明确命名的方法相对 ESM2 序列基线的域等权增量与 95% 分层
bootstrap CI。C：主 Route A 与 Route B 的逐域 Spearman 增量。D：直接 Route A 与
同域同 burial matched-random 对照。绿色圆点为天然域，紫色方点为 de novo 域。_

### 冻结强度与探索性强度

{intensity}

`0.5` swaps/contact 的探索性条件同时得到正的 Spearman、NDCG 和 top-k CI/点估计，
而冻结主条件 `5.0` 的 NDCG 下界跨零。这个结果只能提出“较温和反事实可能更合适”的
新假设，不能在本轮把 `0.5` 追认为主条件，也不能改写 Route A 判决。

## 机制、OOD 与异质性

### 反事实在最低重连强度已经接近饱和

{ood_table}

从 `0.5` 到 `5.0`，JSD 与熵变化几乎不随请求 swap 数单调增加；域级强度相关的平均
Spearman 接近零。这说明 degree-preserving 图虽然保持节点度，却在最低强度已让 MIF
进入明显不同的条件分布。`5.0` 的弱 NDCG 可能来自过强或已饱和的反事实，但这是后验
机制解释仅作为后续假设，确认性判决保持不变。

### 存在低维动作成分，但不等于序列恢复了结构

主直接残差的 PC1 解释 `{float(pca.iloc[0]["explained_variance_ratio"]):.1%}` 方差，前三个
PC 累计 `{pca3:.1%}`，前五个累计 `{pca5:.1%}`。Route A 与 Route B 的 mutant-vs-WT
残差动作在域内具有中等一致性：

{action_table}

AAindex1 是公开的 20-AA 数值性质集合。[^8] 载荷对齐显示：

{aaindex_table}

这些是 20 个氨基酸上的探索性相关，且 CCA 为 in-sample 描述，不能被解释为独立机制
验证。它们只支持“稳定性相关动作可能集中在少数 physicochemical directions”这一较窄
假设。

![counterfactual study mechanism analyses](../figures/counterfactuals_mechanism.png)

_图 2｜A：直接残差对重连强度的敏感性；橙色虚线为冻结主强度 5。B：MIF paired 与
反事实分布的 JSD 和熵变化，误差线为 95% 域 bootstrap CI。C：20-AA 残差 PCA。
D：按突变类别分层的主 A/B Spearman 增量。_

### 突变分层

{mutation_table}

收益在 ESM2 基线较弱和序列熵较高的位置更强；涉及 Gly/Pro 的突变是明确的负向例外。
buried、exposed 与 intermediate 并未形成“只在 core 有效”的单一路由。锁定面板没有 MSA
conservation，因此不把 ESM2 entropy 冒充 conservation；覆盖表记录为
`{coverage.loc[coverage["analysis"].eq("msa_conservation"), "status"].iloc[0]}`。

### 局限

- 天然分层有 34 个域但只有 {len(natural_variants):,} 个突变，单域 NDCG 方差较大；
  de novo 分层的 {len(de_novo_variants):,} 个突变不会因行数多而获得更大域权重。
- 16 个 de novo 域来自同一 Megascale 测量体系；即使天然 S669 提供另一平台，也不能把
  结果外推到一般长度、功能或所有结构类型。
- de novo 结构没有官方 CATH-H/T，Foldseek 阈值是透明的操作化替代，而非 CATH 标签。
- circular replication 的点估计为正，但其多项 CI 跨零；协议只要求 replication 点估计，
  因此报告 PASS 不等于其单独已确认。
- Route B 可能对过强的直接残差做了低秩正则化或去噪。A 失败、B 通过不能单独证明序列
  “恢复了结构”；它只说明固定 CARP 映射在这一面板上产生了可用排序分量。

## 冻结判决

最终判决为：**`{project["decision"]}`**。

1. **不启动 sequence-control branch。** 冻结分支把 Route A 作为反事实 estimand 本身有效的前置条件；
   B 的孤立 PASS 不能绕过 A 的 NDCG FAIL。
2. **保留 `PIVOT_SELECTIVE_STRUCTURE_CONDITIONED`。** 本轮不修改 generalization study 的历史判决，
   也不恢复通用 CARP/ESM sequence-only residual transfer。
3. **将 `0.5` 重连仅登记为新假设。** 若未来继续，必须用新协议、新面板和更
   in-distribution 的真实结构/局部扰动反事实重新确认；不得复用本轮 50 域作确认集。
4. **保留完整的正结果语义。** Route A 在 Spearman、top-k、两分层、replication 和
   random control 上均为正；Route B 完整通过。项目停止扩展的原因是严格遵守预注册
   NDCG 下界，同时完整保留其他指标中的正信号。

## 复现与产物

执行顺序固定如下；`freeze_counterfactuals_protocol.py` 只允许在正式评分文件出现前运行：

```bash
PYTHONPATH=src conda run -n margin-models python scripts/workflows/counterfactuals/prepare.py
PYTHONPATH=src conda run -n margin-models python scripts/workflows/counterfactuals/freeze_protocol.py
PYTHONPATH=src conda run -n margin-models python scripts/workflows/counterfactuals/prepare_requests.py
PYTHONPATH=src conda run -n margin-models python scripts/workflows/counterfactuals/run_mif.py --device cuda:0
PYTHONPATH=src conda run -n margin-models python scripts/workflows/counterfactuals/run_representations.py --device cuda:0
PYTHONPATH=src conda run -n margin-models python scripts/workflows/counterfactuals/evaluate.py
PYTHONPATH=src conda run -n margin-models python scripts/workflows/counterfactuals/analyze_mechanisms.py
PYTHONPATH=src conda run -n margin-models python scripts/workflows/counterfactuals/build_figures.py
PYTHONPATH=src conda run -n margin-models python scripts/workflows/counterfactuals/build_report.py
```

关键机器可读产物包括协议锁、面板表、65,094 行 MIF score、域级指标与增量、随机对照、
机制表、图源数据和最终报告。所有大型模型、数据归档与 counterfactual study 表征保存在 D 盘；运行
目录只保存紧凑结果与溯源信息。

## 参考文献

[^1]: Pancotti, C. et al. (2022). “Predicting protein stability changes upon single-point mutation: a thorough comparison of the available tools on a new dataset.” _Briefings in Bioinformatics_ 23, bbab555. <https://doi.org/10.1093/bib/bbab555>

[^2]: Tsuboyama, K. et al. (2023). “Mega-scale experimental analysis of protein folding stability in biology and design.” _Nature_ 620, 434–444. <https://doi.org/10.1038/s41586-023-06328-6>

[^3]: Tsuboyama, K. et al. (2023). “Mega-scale experimental analysis of protein folding stability in biology and design: data release.” _Zenodo_. <https://doi.org/10.5281/zenodo.7992926>

[^4]: CATH. “CATH release 4.4.0.” <https://download.cathdb.info/cath/releases/all-releases/v4_4_0/>

[^5]: van Kempen, M. et al. (2024). “Fast and accurate protein structure search with Foldseek.” _Nature Biotechnology_ 42, 243–246. <https://doi.org/10.1038/s41587-023-01773-0>

[^6]: Yang, K. K., Zanichelli, N. & Yeh, H. (2023). “Masked inverse folding with sequence transfer for protein representation learning.” _Protein Engineering, Design and Selection_ 36, gzad015. <https://doi.org/10.1093/protein/gzad015>

[^7]: Lin, Z. et al. (2023). “Evolutionary-scale prediction of atomic-level protein structure with a language model.” _Science_ 379, 1123–1130. <https://doi.org/10.1126/science.ade2574>

[^8]: Kawashima, S., Ogata, H. & Kanehisa, M. (1999). “AAindex: Amino Acid Index Database.” _Nucleic Acids Research_ 27, 368–369. <https://doi.org/10.1093/nar/27.1.368>

---

_最后更新：2026-08-15_
"""
    report_path = output / "counterfactuals_report.md"
    report_path.write_text(report, encoding="utf-8")
    manifest_path = output / "manifest.json"
    write_json(
        manifest_path,
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "decision": str(project["decision"]),
            "route_a_passed": bool(project["route_a_passed"]),
            "route_b_passed": bool(project["route_b_passed"]),
            "report": {"path": str(report_path), "sha256": sha256_file(report_path)},
        },
    )
    return {"report": report_path, "manifest": manifest_path}


def _intensity_table(summary: pd.DataFrame) -> str:
    selected = summary.loc[
        summary["method"].str.startswith("direct_mif_paired_minus_contact_rewired_")
        & summary["stratum"].eq("all")
        & summary["metric"].isin(
            ["spearman_increment", "ndcg_increment", "stabilizing_topk_recall_increment"]
        )
    ].copy()
    selected["strength"] = selected["method"].str.rsplit("_", n=1).str[-1].astype(float)
    pivot = selected.pivot(index="strength", columns="metric", values=["estimate", "ci_low"])
    rows = [
        "| swaps/contact | ΔSpearman [CI low] | ΔNDCG [CI low] | Δtop-k |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for strength, values in pivot.sort_index().iterrows():
        rows.append(
            "| "
            f"{strength:g} | {values[('estimate', 'spearman_increment')]:+.4f} "
            f"[{values[('ci_low', 'spearman_increment')]:+.4f}] | "
            f"{values[('estimate', 'ndcg_increment')]:+.4f} "
            f"[{values[('ci_low', 'ndcg_increment')]:+.4f}] | "
            f"{values[('estimate', 'stabilizing_topk_recall_increment')]:+.4f} |"
        )
    return "\n".join(rows)


def _replication_table(summary: pd.DataFrame) -> str:
    rows = [
        "| Circular replication | ΔSpearman | ΔNDCG | Δtop-k |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method, label in (
        (ROUTE_A_REPLICATION, "Route A direct"),
        (ROUTE_B_REPLICATION, "Route B CARP-predicted"),
    ):
        frame = summary.loc[summary["method"].eq(method) & summary["stratum"].eq("all")].set_index(
            "metric"
        )
        rows.append(
            f"| {label} | {frame.loc['spearman_increment', 'estimate']:+.4f} | "
            f"{frame.loc['ndcg_increment', 'estimate']:+.4f} | "
            f"{frame.loc['stabilizing_topk_recall_increment', 'estimate']:+.4f} |"
        )
    return "\n".join(rows)


def _ood_table(ood: pd.DataFrame) -> str:
    values = (
        ood.loc[ood["rewiring_swaps_per_edge"].notna()]
        .groupby(["rewiring_swaps_per_edge", "domain_id", "stratum"], observed=True)[
            ["paired_counterfactual_jsd_nats", "entropy_change_nats"]
        ]
        .mean()
        .reset_index()
        .groupby("rewiring_swaps_per_edge", observed=True)[
            ["paired_counterfactual_jsd_nats", "entropy_change_nats"]
        ]
        .mean()
    )
    rows = [
        "| swaps/contact | paired-vs-CF JSD (nats) | CF entropy change (nats) |",
        "| ---: | ---: | ---: |",
    ]
    for strength, row in values.iterrows():
        rows.append(
            f"| {strength:g} | {row['paired_counterfactual_jsd_nats']:.4f} | "
            f"{row['entropy_change_nats']:+.4f} |"
        )
    return "\n".join(rows)


def _action_table(action: pd.DataFrame) -> str:
    values = action.groupby("stratum", observed=True)[
        ["spearman", "pearson", "candidate_pair_sign_accuracy"]
    ].mean()
    rows = [
        "| 分层 | direct-vs-predicted Spearman | Pearson | 动作符号一致率 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for stratum, row in values.iterrows():
        rows.append(
            f"| {stratum} | {row['spearman']:.4f} | {row['pearson']:.4f} | "
            f"{row['candidate_pair_sign_accuracy']:.2%} |"
        )
    return "\n".join(rows)


def _aaindex_table(aaindex: pd.DataFrame) -> str:
    selected = aaindex.loc[aaindex["component"].isin(["PC1", "PC2", "PC3"])].copy()
    selected["absolute"] = selected["spearman"].abs()
    selected = (
        selected.sort_values(["component", "absolute"], ascending=[True, False])
        .groupby("component", observed=True)
        .head(2)
    )
    rows = [
        "| Residual PC | AAindex accession | 描述 | Spearman |",
        "| --- | --- | --- | ---: |",
    ]
    for row in selected.itertuples(index=False):
        description = str(row.description).replace("|", "/")
        rows.append(f"| {row.component} | {row.accession} | {description} | {row.spearman:+.3f} |")
    return "\n".join(rows)


def _mutation_table(strata: pd.DataFrame) -> str:
    selected = strata.loc[
        strata["dimension"].eq("substitution_class")
        & strata["metric"].eq("spearman_increment")
        & strata["method"].isin([ROUTE_A_PRIMARY, ROUTE_B_PRIMARY])
    ]
    pivot = selected.pivot(index="level", columns="method", values=["estimate", "ci_low"])
    rows = [
        "| 突变类别 | Route A Δρ [CI low] | Route B Δρ [CI low] |",
        "| --- | ---: | ---: |",
    ]
    for level, row in pivot.sort_index().iterrows():
        label = str(level).replace("_", " ")
        rows.append(
            f"| {label} | {row[('estimate', ROUTE_A_PRIMARY)]:+.4f} "
            f"[{row[('ci_low', ROUTE_A_PRIMARY)]:+.4f}] | "
            f"{row[('estimate', ROUTE_B_PRIMARY)]:+.4f} "
            f"[{row[('ci_low', ROUTE_B_PRIMARY)]:+.4f}] |"
        )
    return "\n".join(rows)


def _pass(value: object) -> str:
    return "PASS" if bool(value) else "FAIL"
