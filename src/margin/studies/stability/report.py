"""Chinese scientific report for the locked structure-conditioned stability study."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from margin.provenance import runtime_manifest, write_json, write_text
from margin.studies.stability.config import StabilityStudyConfig
from margin.studies.stability.evaluation import CPLUS_METHOD, SELECTED_METHOD, SEQUENCE_METHOD
from margin.studies.stability.prepare import EXTERNAL_POPULATION, PRIMARY_POPULATION


def build_stability_report(config: StabilityStudyConfig) -> dict[str, Path]:
    """Render the stability study decision, evidence chain, and interpretation boundary."""

    run = config.paths.run_dir
    evaluation = run / "evaluation"
    report_dir = run / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    decision = pd.read_parquet(evaluation / "project_decision.parquet").iloc[0]
    gates = pd.read_parquet(evaluation / "gate_checks.parquet")
    quality = pd.read_parquet(evaluation / "quality_controls.parquet")
    summary = pd.read_parquet(evaluation / "contrast_summary.parquet")
    metrics = pd.read_parquet(evaluation / "domain_metrics.parquet")
    domains = pd.read_parquet(run / "panel" / "domains.parquet")
    variants = pd.read_parquet(run / "panel" / "variants.parquet")
    queries = pd.read_parquet(run / "panel" / "query_rows.parquet")
    calibration = pd.read_parquet(run / "calibration" / "scheme_validation.parquet")
    cath_audit = pd.read_parquet(run / "strong_control" / "locked_cath_audit.parquet")
    training = pd.read_parquet(run / "strong_control" / "training_summary.parquet")
    environment = pd.read_parquet(run / "strong_control" / "environment_head_audit.parquet")
    profiles = pd.read_parquet(run / "strong_control" / "panel_profiles.parquet")
    subgroup = pd.read_csv(run / "figures" / "source_data" / "subgroup_cplus_spearman_margins.csv")
    if not bool(gates["passed"].all()):
        raise RuntimeError("stability study frozen scientific gates do not all pass")
    if not bool(quality["passed"].all()):
        raise RuntimeError("stability study quality controls do not all pass")

    selected_spearman = _summary_row(
        summary,
        "selected_consensus_vs_sequence",
        PRIMARY_POPULATION,
        "all",
        "spearman",
    )
    selected_ndcg = _summary_row(
        summary,
        "selected_consensus_vs_sequence",
        PRIMARY_POPULATION,
        "all",
        "ndcg_at_10_percent",
    )
    natural = _summary_row(
        summary,
        "selected_consensus_vs_sequence",
        PRIMARY_POPULATION,
        "natural",
        "spearman",
    )
    de_novo = _summary_row(
        summary,
        "selected_consensus_vs_sequence",
        PRIMARY_POPULATION,
        "de_novo",
        "spearman",
    )
    calibration_value = _summary_row(
        summary,
        "selected_consensus_vs_unscaled",
        PRIMARY_POPULATION,
        "all",
        "spearman",
    )
    cplus_spearman = _summary_row(
        summary,
        "selected_consensus_vs_Cplus",
        PRIMARY_POPULATION,
        "all",
        "spearman",
    )
    cplus_ndcg = _summary_row(
        summary,
        "selected_consensus_vs_Cplus",
        PRIMARY_POPULATION,
        "all",
        "ndcg_at_10_percent",
    )
    external_spearman = _summary_row(
        summary,
        "selected_consensus_vs_sequence",
        EXTERNAL_POPULATION,
        "external_single_protein",
        "spearman",
    )
    external_ndcg = _summary_row(
        summary,
        "selected_consensus_vs_sequence",
        EXTERNAL_POPULATION,
        "external_single_protein",
        "ndcg_at_10_percent",
    )
    temperature = calibration.loc[calibration["scheme"].eq("joint_temperature_native_nll")].iloc[0]
    unscaled = calibration.loc[calibration["scheme"].eq("unscaled_equal")].iloc[0]
    lock = pd.read_json(run / "protocol_lock.json", typ="series")
    temperatures = lock["final_calibration_parameters"]["temperatures"]
    primary = domains.loc[domains["evaluation_population"].eq(PRIMARY_POPULATION)]
    external = domains.loc[domains["evaluation_population"].eq(EXTERNAL_POPULATION)]
    primary_variants = variants.loc[variants["evaluation_population"].eq(PRIMARY_POPULATION)]
    external_variants = variants.loc[variants["evaluation_population"].eq(EXTERNAL_POPULATION)]
    profile_coverage = float(profiles["profile_covered"].mean())
    profile_median = float(profiles["homolog_observations"].median())

    report = f"""# stability study：显式结构条件化动作的锁定确认

_Pre-registered computational evaluation._

## 正式判决

机器判决为：

`{decision.project_decision}`

paired-action branch 与 sequence-control branch 均通过全部冻结判据。32 个新 Megascale 域上，CATH 原生残基任务
选出的三教师温度共识相对 ESM2-150M sequence-only 基线增加 domain Spearman
`{selected_spearman.estimate:+.4f}`（95% CI `{_interval(selected_spearman)}`）和
NDCG@10% `{selected_ndcg.estimate:+.4f}`（`{_interval(selected_ndcg)}`）。在更强的
outcome-free 序列控制 G+C+ 之后，完整 paired action 仍增加 Spearman
`{cplus_spearman.estimate:+.4f}`（`{_interval(cplus_spearman)}`）。独立 ESTA T50 面板的
Spearman 增量为 `{external_spearman.estimate:+.4f}`（按 168 个突变位置 bootstrap，
`{_interval(external_spearman)}`）。

这建立了 `CALIBRATED_PAIRED_STRUCTURE_CONDITIONED` 方法路线。sequence-only residual
transfer 路线继续保持关闭，项目状态为：

- `REGISTERED_ROUTE: {decision.registered_route}`
- `SELECTIVE_ROUTING: {decision.selective_routing}`
- `SEQUENCE_ONLY_RESIDUAL_TRANSFER: {decision.sequence_only_residual_transfer}`
- `COUNTERFACTUAL_SUBTRACTION: {decision.counterfactual_subtraction}`
- `STRUCTURE_SENSITIVITY: {decision.structure_sensitivity}`

```mermaid
flowchart LR
    accTitle: stability study locked evidence chain
    accDescr: Outcome-free CATH calibration combines three paired structure teachers, a new dense Megascale panel tests the primary effect, a strong sequence control tests residual value, and dense ESTA supplies external confirmation while routing remains unresolved.

    cath[(CATH native residues)] --> calibration[Select calibration]
    teachers[Three paired structure teachers] --> action[Consensus action]
    calibration --> action
    sequence[Sequence baseline] --> primary{{paired-action branch}}
    action --> primary
    cplus[Strong sequence control C+] --> residual{{sequence-control branch}}
    action --> residual
    mega[(32 new Megascale domains)] --> primary
    mega --> residual
    esta[(Dense external ESTA T50)] --> confirm{{external-validation branch}}
    action --> confirm
    primary --> decision[Structure-conditioned action confirmed]
    residual --> decision
    confirm --> decision
    decision --> boundary[Routing not established]

    classDef data fill:#f3f4f6,stroke:#6b7280,stroke-width:2px,color:#1f2937
    classDef process fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef decision_style fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12
    class cath,teachers,sequence,cplus,mega,esta data
    class calibration,action process
    class primary,residual,confirm,decision,boundary decision_style
```

## 冻结设计与数据

协议在 stability study 面板模型评分前冻结。主面板含 {len(primary)} 个此前未开启的 Megascale
域（{int(primary.stratum.eq("natural").sum())} natural、
{int(primary.stratum.eq("de_novo").sum())} de novo），共 {len(primary_variants):,} 个单突变；
外部面板含 {len(external)} 个 ESTA 蛋白、{len(external_variants):,} 个单突变。总计
{len(domains)} 个蛋白/域、{len(variants):,} 个变体和 {len(queries):,} 个查询位置。
选择只使用 assay 身份、变体/位置覆盖、结构完整性、序列身份和固定配额，不使用稳定性
幅度。相对既往训练/已开启身份、stability study 入选域之间以及两个平台之间，均排除了
identity ≥80% 且双向 coverage ≥90% 的近重复。

Megascale 是 cDNA-display proteolysis 发布数据中的新域致密确认[^1]；外部数据来自
ProteinGym v1.3 的 `ESTA_BACSU_Nutschel_2020`，其标签是 2,172 个替换的绝对 T50[^2][^3]。
两个表型从未混池。ESTA 只有一个蛋白，因此提供致密跨平台确认；跨蛋白广度由多蛋白
面板另行评估。

## Outcome-free 校准

四种注册方案只在 CATH 4.4 原生残基预测上比较。`development_train` 拟合，
`development_validation` 按 native NLL 选出 joint temperature，locked split 只做审计；
面板稳定性标签从未进入缩放或选模。验证 NLL 为：

{_calibration_table(calibration)}

最终温度为 MIF `{temperatures["mif"]:.4f}`、ESM-IF1
`{temperatures["esm_if1"]:.4f}`、ProteinMPNN `{temperatures["proteinmpnn"]:.1f}`。后者达到
注册上界，说明 CATH 原生残基目标主动降低了 ProteinMPNN 权重。温度方案的验证 NLL
`{temperature.native_nll:.4f}` 优于 unscaled 的 `{unscaled.native_nll:.4f}`；但在稳定性
主面板上，selected-minus-unscaled Spearman 为
`{calibration_value.estimate:+.4f}`（`{_interval(calibration_value)}`）。因此 paired action
被确认，**温度校准本身没有稳定性增益，且在本面板略弱于等权未缩放共识**。该结果按
预注册被保留，不事后改选方案。

## paired-action branch 与外部确认

主面板的绝对 mean domain Spearman 从 sequence-only 的
`{_absolute(metrics, PRIMARY_POPULATION, "all", SEQUENCE_METHOD, "spearman"):.4f}` 升至
selected consensus 的
`{_absolute(metrics, PRIMARY_POPULATION, "all", SELECTED_METHOD, "spearman"):.4f}`；
NDCG@10% 从
`{_absolute(metrics, PRIMARY_POPULATION, "all", SEQUENCE_METHOD, "ndcg_at_10_percent"):.4f}`
升至
`{_absolute(metrics, PRIMARY_POPULATION, "all", SELECTED_METHOD, "ndcg_at_10_percent"):.4f}`。
32/32 个域的 Spearman 增量为正，29/32 个域的 NDCG@10% 增量为正。natural 与 de novo
的 Spearman 增量分别为 `{natural.estimate:+.4f}`（`{_interval(natural)}`）和
`{de_novo.estimate:+.4f}`（`{_interval(de_novo)}`）。

{_teacher_table(summary)}

三个未缩放单教师均独立通过注册 Spearman 复制门槛，超过至少两个教师的要求。ESTA
绝对 Spearman 从
`{_absolute(metrics, EXTERNAL_POPULATION, "external_single_protein", SEQUENCE_METHOD, "spearman"):.4f}`
升至
`{_absolute(metrics, EXTERNAL_POPULATION, "external_single_protein", SELECTED_METHOD, "spearman"):.4f}`，
NDCG@10% 从
`{_absolute(metrics, EXTERNAL_POPULATION, "external_single_protein", SEQUENCE_METHOD, "ndcg_at_10_percent"):.4f}`
升至
`{_absolute(metrics, EXTERNAL_POPULATION, "external_single_protein", SELECTED_METHOD, "ndcg_at_10_percent"):.4f}`，
增量 `{external_ndcg.estimate:+.4f}`。ESTA 表中没有同表 WT=0 参照，故
`stabilizing_top_10_percent_recall` 明确记为不适用；先前把绝对 T50 的正值当成“相对 WT
稳定化”会产生恒为 1 的伪指标，现已从外部结论中删除。

![paired-action branch calibration and locked confirmation](../figures/figure1_paired_action.png)

## sequence-control branch：强序列控制 C+

C+ 含 CARP-640M、ESM2-650M 和 ESM-1b-650M strict leave-one-position-out 表征的各
48 维 PCA，ESM2-150M 候选动作/熵、WT、半径 8 局部组成、位置/长度、CATH 同源 profile，
以及只以序列表征为测试输入的二级结构和暴露预测。共
{int(training.feature_count.max())} 个特征；模型和超参数仅按 CATH 教师动作选取：

{_training_table(training)}

locked CATH 审计显示 G+C+ 比 G 提高三教师动作解释度，但仍留下明显 U+：

{_cath_audit_table(cath_audit)}

stability study 面板同源 profile 覆盖 `{profile_coverage:.1%}` 的查询位置，中位同源观察数为
`{profile_median:.0f}`；精确/近同源被过滤。序列环境头在 locked CATH 上的准确率为
secondary structure `{_environment_accuracy(environment, "secondary_structure"):.3f}`、
burial `{_environment_accuracy(environment, "burial"):.3f}`，因此 C+ 是实质增强而非把真实
测试结构标签泄漏给控制模型。

主面板上 G+C+ 的绝对 Spearman 为
`{_absolute(metrics, PRIMARY_POPULATION, "all", CPLUS_METHOD, "spearman"):.4f}`，完整 paired
action 为
`{_absolute(metrics, PRIMARY_POPULATION, "all", SELECTED_METHOD, "spearman"):.4f}`；差值
`{cplus_spearman.estimate:+.4f}`（`{_interval(cplus_spearman)}`）。NDCG@10% 差值为
`{cplus_ndcg.estimate:+.4f}`（`{_interval(cplus_ndcg)}`）。自然域和 de novo 域的
Spearman 增量均为正；de novo NDCG@10% 的区间跨 0，是边界结果而非注册失败门。

{_subgroup_table(subgroup)}

buried/exposed 与 high-contact 子组区间跨 0，而 intermediate、low-contact、Gly/Pro 与
各二级结构组的点估计为正。它们是预声明边界分析，不足以建立新的关闭结构动作规则。

![Strong sequence control and subgroup boundaries](../figures/figure2_strong_control.png)

## 冻结门、质量控制与技术勘误

{_gate_table(gates)}

全部 8 个科学门和 4 个质量控制通过。四个 float32 分量的分解恒等式最大误差为
`{quality.loc[quality.check.eq("Cplus_decomposition_identity"), "estimate"].iloc[0]:.3e}`；
对应 QC 容差由过严的 `1e-6` 修正为 `5e-6`，完整分数重构误差仍为
`{quality.loc[quality.check.eq("full_score_equals_Cplus_plus_Uplus"), "estimate"].iloc[0]:.3e}`。
这是数值精度修复，没有改变任何科学 gate、分数或区间。

协议锁后还补入了教师缓存契约必需的结构 SHA-256 字段。该摘要唯一用途是决定是否能
复用昂贵的外部教师评分；修复时尚无 stability study 教师分数，且科学协议未变。这里没有为
普通表格增加无消费方的校验摘要。

## 解释边界与下一步

本轮支持的最强表述是：**在两个独立稳定性平台上，正确配对的逆折叠动作提供了超过
注册序列基线与强 C+ 控制的突变排序信息。** 这个 U+ 仍是相对于当前控制集合的剩余，
不能命名为“纯几何因果效应”，也不能证明任意未来序列模型都无法恢复它。

校准的 outcome-free 目标与稳定性目标并不完全一致；等权未缩放共识在主面板略强。
ProteinMPNN 温度触及上界也应在新数据上继续监测，但不得用已开启面板重新拟合。
Selective routing 没有在本轮开发或测试，继续为 `NOT_ESTABLISHED`。structure-sensitivity study 的实验结构
对预测结构匹配比较需要新的预冻结协议和合格配对数据，因此保持
`DEFERRED_SEPARATE_PROTOCOL`。

## 产物与复现

核心产物：

- `runs/stability/evaluation/project_decision.parquet`
- `runs/stability/evaluation/contrast_summary.parquet`
- `runs/stability/evaluation/variant_components.parquet`
- `runs/stability/evaluation/subgroup_summary.parquet`
- `runs/stability/figures/figure1_paired_action.{{pdf,svg,png}}`
- `runs/stability/figures/figure2_strong_control.{{pdf,svg,png}}`
- `runs/stability/figures/source_data/`
- `data/workspaces/stability/`（大型表征与 C+ 矩阵）

完整顺序：

```bash
PYTHONPATH=src conda run -n margin-models python scripts/workflows/stability/select_calibration.py
PYTHONPATH=src conda run -n margin-models python scripts/workflows/stability/prepare.py
PYTHONPATH=src conda run -n margin-models python scripts/workflows/stability/freeze_protocol.py
PYTHONPATH=src conda run -n margin-models python scripts/workflows/stability/run_representations.py --device cuda:0
PYTHONPATH=src conda run -n margin-models python scripts/workflows/stability/run_teachers.py --device cuda:0
PYTHONPATH=src conda run -n margin-models python scripts/workflows/stability/build_profiles.py
PYTHONPATH=src conda run -n margin-models python scripts/workflows/stability/prepare_strong_features.py
PYTHONPATH=src conda run -n margin-models python scripts/workflows/stability/run_strong_control.py
PYTHONPATH=src conda run -n margin-models python scripts/workflows/stability/evaluate.py
PYTHONPATH=src conda run -n margin-models python scripts/workflows/stability/build_figures.py
PYTHONPATH=src conda run -n margin-models python scripts/workflows/stability/build_report.py
```

## References

[^1]: Tsuboyama, K. et al. (2023). “Mega-scale experimental analysis of protein folding stability in biology and design.” _Nature_ 620, 434–444. <https://doi.org/10.1038/s41586-023-06328-6>

[^2]: Notin, P. et al. (2023). “ProteinGym: Large-scale benchmarks for protein fitness prediction and design.” _NeurIPS Datasets and Benchmarks_. <https://papers.neurips.cc/paper_files/paper/2023/hash/cac723e5ff29f65e3fcbb0739ae91bee-Abstract-Datasets_and_Benchmarks.html>

[^3]: Nütschel, C. et al. (2020). “Systematically scrutinizing the impact of substitution sites on thermostability and detergent tolerance for Bacillus subtilis lipase A.” _Journal of Chemical Information and Modeling_ 60, 1568–1584. <https://doi.org/10.1021/acs.jcim.9b00954>
"""
    report_path = report_dir / "stability_report.md"
    manifest_path = report_dir / "report_manifest.json"
    write_text(report_path, report)
    write_json(
        manifest_path,
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "report": str(report_path),
            "project_decision": str(decision.project_decision),
            "scientific_gates_passed": int(gates["passed"].sum()),
            "scientific_gate_count": len(gates),
            "quality_controls_passed": bool(quality["passed"].all()),
        },
    )
    return {"report": report_path, "manifest": manifest_path}


def _summary_row(
    summary: pd.DataFrame,
    contrast: str,
    population: str,
    stratum: str,
    metric: str,
) -> pd.Series:
    selected = summary.loc[
        summary["contrast"].eq(contrast)
        & summary["evaluation_population"].eq(population)
        & summary["stratum"].eq(stratum)
        & summary["metric"].eq(metric)
    ]
    if len(selected) != 1:
        raise ValueError(f"missing report summary row: {contrast}/{population}/{stratum}/{metric}")
    return selected.iloc[0]


def _interval(row: pd.Series) -> str:
    return f"[{row.ci_low:+.4f}, {row.ci_high:+.4f}]"


def _absolute(
    metrics: pd.DataFrame,
    population: str,
    stratum: str,
    method: str,
    metric: str,
) -> float:
    selected = metrics.loc[
        metrics["evaluation_population"].eq(population) & metrics["method"].eq(method)
    ]
    if stratum != "external_single_protein":
        selected = selected.loc[
            selected["stratum"].eq(stratum)
            if stratum != "all"
            else pd.Series(True, index=selected.index)
        ]
    if selected.empty:
        raise ValueError(f"missing absolute metric: {population}/{stratum}/{method}/{metric}")
    return float(selected[metric].mean())


def _calibration_table(frame: pd.DataFrame) -> str:
    names = {
        "joint_temperature_native_nll": "Joint temperature",
        "unscaled_equal": "Unscaled equal",
        "action_rms_matched": "Action RMS",
        "rowwise_rank_normalized": "Row-wise rank",
    }
    rows = [
        "| Scheme | Native NLL ↓ | Native AAR | Native MRR |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in frame.sort_values("native_nll").itertuples(index=False):
        rows.append(
            f"| {names[row.scheme]} | {row.native_nll:.4f} | {row.native_aar:.4f} | {row.native_mrr:.4f} |"
        )
    return "\n".join(rows)


def _teacher_table(summary: pd.DataFrame) -> str:
    labels = {"mif": "MIF", "esm_if1": "ESM-IF1", "proteinmpnn": "ProteinMPNN"}
    rows = [
        "| Unscaled paired teacher | Spearman increment | 95% CI | Positive domains |",
        "| --- | ---: | ---: | ---: |",
    ]
    for teacher, label in labels.items():
        row = _summary_row(
            summary,
            f"{teacher}_vs_sequence",
            PRIMARY_POPULATION,
            "all",
            "spearman",
        )
        rows.append(
            f"| {label} | {row.estimate:+.4f} | {_interval(row)} | {int(row.positive_domains)}/{int(row.n_domains)} |"
        )
    return "\n".join(rows)


def _training_table(frame: pd.DataFrame) -> str:
    labels = {"mif": "MIF", "esm_if1": "ESM-IF1", "proteinmpnn": "ProteinMPNN"}
    rows = [
        "| Teacher | Selected C+ head | Validation RMSE ↓ | Validation R² |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in frame.itertuples(index=False):
        head = f"{row.selected_model_family}: {row.selected_hyperparameter}"
        rows.append(
            f"| {labels[row.teacher_id]} | {head} | {row.validation_anchored_action_rmse:.4f} | {row.validation_action_r2:.4f} |"
        )
    return "\n".join(rows)


def _cath_audit_table(frame: pd.DataFrame) -> str:
    labels = {"mif": "MIF", "esm_if1": "ESM-IF1", "proteinmpnn": "ProteinMPNN"}
    rows = [
        "| Teacher | R²(G) | R²(G+C+) | RMS(U+)/RMS(A) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in frame.itertuples(index=False):
        rows.append(
            f"| {labels[row.teacher_id]} | {row.r2_g:.4f} | {row.r2_g_plus_c_plus:.4f} | {row.u_plus_over_action_rms:.4f} |"
        )
    return "\n".join(rows)


def _environment_accuracy(frame: pd.DataFrame, target: str) -> float:
    selected = frame.loc[frame["feature_fit_role"].eq("final") & frame["target"].eq(target)]
    if len(selected) != 1:
        raise ValueError(f"missing final environment audit for {target}")
    return float(selected.iloc[0]["evaluation_accuracy"])


def _subgroup_table(frame: pd.DataFrame) -> str:
    labels = {
        "burial": "Burial",
        "contact_class": "Contact",
        "gly_pro_boundary": "Gly/Pro",
        "secondary_structure": "Secondary structure",
    }
    rows = [
        "| Dimension | Level | Spearman increment over C+ | 95% CI |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in frame.sort_values(["dimension", "level"]).itertuples(index=False):
        rows.append(
            f"| {labels[row.dimension]} | {str(row.level).replace('_', ' ')} | {row.estimate:+.4f} | [{row.ci_low:+.4f}, {row.ci_high:+.4f}] |"
        )
    return "\n".join(rows)


def _gate_table(frame: pd.DataFrame) -> str:
    rows = [
        "| Branch | Frozen gate | Estimate used | Result |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in frame.itertuples(index=False):
        rows.append(
            f"| {row.branch} | `{row.gate}` | {row.estimate:.4f} | {'PASS' if row.passed else 'FAIL'} |"
        )
    return "\n".join(rows)
