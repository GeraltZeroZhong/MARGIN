"""Export the curated, path-clean publication bundle from frozen run artifacts."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class FigureExport:
    source_stem: Path
    public_stem: str
    supports: str
    description: str


@dataclass(frozen=True)
class TableExport:
    source: Path
    public: Path
    supports: str
    description: str


FIGURES: tuple[FigureExport, ...] = (
    FigureExport(
        Path("figures/figure1_paired_action"),
        "figure_1_paired_action",
        "Figure 1",
        "Calibration, paired-action performance, teacher margins, and ESTA validation.",
    ),
    FigureExport(
        Path("figures/figure2_strong_control"),
        "figure_2_sequence_controls",
        "Figure 2",
        "Strong sequence controls and residual paired-action increments.",
    ),
    FigureExport(
        Path("supplement/figures/figure3_postlock_supplement"),
        "figure_3_external_validation",
        "Figure 3",
        "Method matrix, supervised references, structure sources, and compute frontier.",
    ),
    FigureExport(
        Path("postlock_submission_audit/figures/figure4_submission_audit"),
        "figure_4_robustness_and_structure_sensitivity",
        "Figure 4",
        "FireProt confirmation, deployment tiers, and structure sensitivity.",
    ),
)


def _table(
    source: str,
    public: str,
    supports: str,
    description: str,
    *,
    area: str,
) -> TableExport:
    roots = {
        "main": Path("figures/source_data"),
        "supplement": Path("supplement/source_data"),
        "audit": Path("postlock_submission_audit/source_data"),
    }
    return TableExport(roots[area] / f"{source}.csv", Path(public), supports, description)


TABLES: tuple[TableExport, ...] = (
    _table(
        "calibration_validation",
        "figure_1/calibration_validation.csv",
        "Figure 1a",
        "Outcome-free calibration comparison on the validation split.",
        area="main",
    ),
    _table(
        "primary_domain_spearman",
        "figure_1/primary_domain_spearman.csv",
        "Figure 1b",
        "Per-domain sequence and selected paired-action Spearman correlations.",
        area="main",
    ),
    _table(
        "primary_spearman_margins",
        "figure_1/primary_spearman_margins.csv",
        "Figure 1c",
        "Equal-domain Spearman increments over the sequence baseline.",
        area="main",
    ),
    _table(
        "external_selected_margins",
        "figure_1/external_selected_margins.csv",
        "Figure 1d",
        "External ESTA increments for the selected paired-action score.",
        area="main",
    ),
    _table(
        "locked_cath_control",
        "figure_2/locked_cath_control.csv",
        "Figure 2a,b",
        "Locked CATH predictability and unexplained-action magnitude.",
        area="main",
    ),
    _table(
        "cplus_primary_margins",
        "figure_2/cplus_primary_margins.csv",
        "Figure 2c",
        "Paired-action increments over the strong sequence control.",
        area="main",
    ),
    _table(
        "subgroup_cplus_spearman_margins",
        "figure_2/subgroup_cplus_spearman_margins.csv",
        "Figure 2d",
        "Subgroup sensitivity of paired-action Spearman increments.",
        area="main",
    ),
    _table(
        "zero_shot_method_matrix",
        "figure_3/zero_shot_method_matrix.csv",
        "Figure 3a",
        "Unified zero-shot method comparison across evaluation panels.",
        area="supplement",
    ),
    _table(
        "supervised_upper_bounds",
        "figure_3/supervised_upper_bounds.csv",
        "Figure 3b",
        "Label-trained reference methods, separated from zero-shot claims.",
        area="supplement",
    ),
    _table(
        "cross_platform_structure_sources",
        "figure_3/structure_sources.csv",
        "Figure 3c",
        "Matched experimental, predicted, and perturbed structure-source results.",
        area="supplement",
    ),
    _table(
        "teacher_cost_frontier",
        "figure_3/teacher_cost_frontier.csv",
        "Figure 3d",
        "Measured adapter time and zero-shot performance by deployment tier.",
        area="supplement",
    ),
    _table(
        "fireprot_domain_results",
        "figure_4/fireprot_domain_results.csv",
        "Figure 4a",
        "Per-protein FireProt performance and action-control margins.",
        area="audit",
    ),
    _table(
        "fireprot_method_summary",
        "figure_4/fireprot_method_summary.csv",
        "Figure 4b",
        "Equal-protein FireProt method performance with bootstrap intervals.",
        area="audit",
    ),
    _table(
        "fast_robust_domain_contrasts",
        "figure_4/fast_robust_domain_contrasts.csv",
        "Figure 4c",
        "Per-domain robust-versus-fast deployment contrasts.",
        area="audit",
    ),
    _table(
        "fast_robust_summary",
        "figure_4/fast_robust_summary.csv",
        "Figure 4c",
        "Bootstrap summaries for robust-versus-fast deployment contrasts.",
        area="audit",
    ),
    _table(
        "structure_sensitivity_geometry_summary",
        "figure_4/structure_sensitivity_geometry_summary.csv",
        "Figure 4d",
        "Aggregate matched-backbone geometry differences by structure source.",
        area="audit",
    ),
    _table(
        "structure_sensitivity_teacher_delta_summary",
        "figure_4/structure_sensitivity_teacher_delta_summary.csv",
        "Figure 4d",
        "Performance changes relative to experimental structures.",
        area="audit",
    ),
    _table(
        "fast_robust_runtime_domain",
        "supplementary/fast_robust_runtime_domain.csv",
        "Supplementary data",
        "Per-domain measured adapter runtime for the fast and robust tiers.",
        area="audit",
    ),
    _table(
        "fast_robust_runtime_summary",
        "supplementary/fast_robust_runtime_summary.csv",
        "Supplementary data",
        "Aggregate runtime summaries for the fast and robust tiers.",
        area="audit",
    ),
    _table(
        "fireprot_domain_metadata",
        "supplementary/fireprot_domain_metadata.csv",
        "Supplementary data",
        "Protein, measurement, and experimental-structure metadata for FireProt.",
        area="audit",
    ),
    _table(
        "fireprot_failure_case",
        "supplementary/fireprot_failure_case.csv",
        "Supplementary data",
        "Detailed descriptive audit of the negative FireProt domain.",
        area="audit",
    ),
    _table(
        "fireprot_measurement_summary",
        "supplementary/fireprot_measurement_summary.csv",
        "Supplementary data",
        "Aggregate repeated-measurement audit without redistributing raw records.",
        area="audit",
    ),
    _table(
        "fireprot_provenance",
        "supplementary/fireprot_provenance.csv",
        "Supplementary data",
        "Upstream dataset, selection, endpoint, and aggregation provenance.",
        area="audit",
    ),
    _table(
        "fireprot_structure_audit",
        "supplementary/fireprot_structure_audit.csv",
        "Supplementary data",
        "Experimental structure-quality and composition audit.",
        area="audit",
    ),
    _table(
        "fireprot_subset_summary",
        "supplementary/fireprot_subset_summary.csv",
        "Supplementary data",
        "Sensitivity to curated and repeated-measurement subsets.",
        area="audit",
    ),
    _table(
        "structure_sensitivity_backbone_geometry_summary",
        "supplementary/structure_backbone_geometry_summary.csv",
        "Supplementary data",
        "Backbone-geometry quality summaries by structure source.",
        area="audit",
    ),
    _table(
        "structure_sensitivity_geometry_action_correlation_summary",
        "supplementary/structure_geometry_action_correlations.csv",
        "Supplementary data",
        "Geometry-action correlation summaries by teacher and structure source.",
        area="audit",
    ),
    _table(
        "structure_sensitivity_geometry_domain_summary",
        "supplementary/structure_geometry_domain_summary.csv",
        "Supplementary data",
        "Per-domain geometry summaries used in the sensitivity audit.",
        area="audit",
    ),
    _table(
        "structure_sensitivity_teacher_distribution_summary",
        "supplementary/structure_teacher_distribution_summary.csv",
        "Supplementary data",
        "Teacher-distribution shifts by structure source.",
        area="audit",
    ),
    _table(
        "structure_sensitivity_teacher_summary",
        "supplementary/structure_teacher_performance_summary.csv",
        "Supplementary data",
        "Teacher performance summaries by structure source.",
        area="audit",
    ),
    _table(
        "score_nomenclature",
        "supplementary/score_nomenclature.csv",
        "Supplementary data",
        "Canonical score identifiers, formulas, roles, and registration status.",
        area="audit",
    ),
)


LEGACY_PATH_MARKERS: tuple[tuple[str, str], ...] = (
    ("protocols/external_validation_cplus_fixed.yaml", "configs/external_validation.yaml"),
    ("baselines/repos/", "external/repositories/"),
    ("external_validation/", "data/workspaces/external_validation/"),
)


def export_publication_bundle(
    project_root: Path,
    output_dir: Path,
    *,
    frozen_run_root: Path | None = None,
) -> tuple[Path, Path]:
    """Export curated CSV source data, figures, manifest, and data dictionary."""

    project_root = project_root.resolve()
    output_dir = output_dir if output_dir.is_absolute() else project_root / output_dir
    source_root = frozen_run_root or project_root / "runs" / "stability"
    if not source_root.is_absolute():
        source_root = project_root / source_root
    figure_dir = output_dir / "figures"
    source_data_dir = output_dir / "source_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    source_data_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    dictionary_rows: list[dict[str, object]] = []

    for item in FIGURES:
        for extension in ("pdf", "svg", "png"):
            source = (source_root / item.source_stem).with_suffix(f".{extension}")
            target = figure_dir / f"{item.public_stem}.{extension}"
            _require_file(source)
            shutil.copy2(source, target)
            manifest_rows.append(
                _manifest_row(
                    target,
                    output_dir,
                    kind="figure",
                    supports=item.supports,
                    description=item.description,
                    source=source.relative_to(project_root),
                )
            )

    for item in TABLES:
        source = source_root / item.source
        target = source_data_dir / item.public
        _require_file(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        table = _sanitize_paths(pd.read_csv(source), project_root)
        table.to_csv(target, index=False)
        _assert_no_local_paths(target)
        manifest_rows.append(
            _manifest_row(
                target,
                output_dir,
                kind="source_data",
                supports=item.supports,
                description=item.description,
                source=source.relative_to(project_root),
                rows=len(table),
                columns=len(table.columns),
            )
        )
        dictionary_rows.extend(_dictionary_rows(target, output_dir, table))

    manifest_path = output_dir / "manifest.csv"
    dictionary_path = output_dir / "data_dictionary.csv"
    pd.DataFrame(manifest_rows).sort_values("path").to_csv(manifest_path, index=False)
    pd.DataFrame(dictionary_rows).sort_values(["dataset", "column_order"]).to_csv(
        dictionary_path,
        index=False,
    )
    return manifest_path, dictionary_path


def _sanitize_paths(table: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    result = table.copy()
    for column in result.select_dtypes(include=["object", "string"]).columns:
        result[column] = result[column].map(lambda value: _sanitize_value(value, project_root))
    return result


def _sanitize_value(value: object, project_root: Path) -> object:
    if not isinstance(value, str):
        return value
    value = value.replace(f"{project_root.as_posix()}/", "")
    for marker, target in LEGACY_PATH_MARKERS:
        if marker in value:
            prefix, suffix = value.split(marker, maxsplit=1)
            if prefix.startswith("/") or not prefix:
                value = f"{target}{suffix}"
    return value


def _manifest_row(
    path: Path,
    output_dir: Path,
    *,
    kind: str,
    supports: str,
    description: str,
    source: Path,
    rows: int | None = None,
    columns: int | None = None,
) -> dict[str, object]:
    return {
        "path": path.relative_to(output_dir).as_posix(),
        "kind": kind,
        "supports": supports,
        "description": description,
        "rows": rows,
        "columns": columns,
        "format": path.suffix.lstrip("."),
        "frozen_source": source.as_posix(),
        "access": "included_in_publication_bundle",
    }


def _dictionary_rows(path: Path, output_dir: Path, table: pd.DataFrame) -> list[dict[str, object]]:
    dataset = path.relative_to(output_dir).as_posix()
    rows = []
    for order, column in enumerate(table.columns):
        rows.append(
            {
                "dataset": dataset,
                "column_order": order,
                "column": column,
                "dtype": str(table[column].dtype),
                "description": _describe_column(column),
                "unit": _column_unit(column),
                "missing_values": int(table[column].isna().sum()),
            }
        )
    return rows


def _describe_column(column: str) -> str:
    definitions = {
        "domain_id": "Stable protein-domain identifier within the analysis panel.",
        "protein_name": "Protein name supplied by the upstream dataset.",
        "pdb_id": "Protein Data Bank structure identifier.",
        "uniprot_id": "UniProt accession supplied by the upstream dataset.",
        "method": "Canonical method identifier.",
        "metric": "Evaluation metric named by the row.",
        "estimate": "Equal-domain point estimate for the named metric.",
        "ci_low": "Lower endpoint of the reported confidence interval.",
        "ci_high": "Upper endpoint of the reported confidence interval.",
        "stratum": "Predefined natural or de novo analysis stratum.",
        "structure_role": "Experimental, predicted, or perturbed structure source.",
        "teacher_id": "Inverse-folding teacher or registered consensus identifier.",
        "panel": "Evaluation panel identifier.",
        "scope": "Predefined subset used for the row's estimate.",
        "n_domains": "Number of independent protein domains in the estimate.",
        "n_variants": "Number of single-amino-acid substitutions in the estimate.",
        "positive_domain_fraction": "Fraction of domain-level effects greater than zero.",
        "leave_one_domain_out_min": "Minimum estimate after omitting one domain at a time.",
        "leave_one_domain_out_max": "Maximum estimate after omitting one domain at a time.",
        "structure_path": "Repository-relative logical path to the selected structure file.",
    }
    if column in definitions:
        return definitions[column]
    if column.endswith("_spearman"):
        return "Spearman rank correlation for the method named by the column."
    if column.endswith("_ndcg10") or "ndcg" in column:
        return "Normalized discounted cumulative gain at the top 10% of variants."
    if column.endswith("_margin"):
        return "Difference between the primary method and its named comparator."
    if "rmsd" in column:
        return "Root-mean-square structural deviation defined by the dataset context."
    if column.endswith("_seconds") or "wall_seconds" in column:
        return "Measured wall-clock runtime for the named scope."
    if column.endswith("_fraction"):
        return "Fraction for the population or condition named by the column."
    if column.startswith("n_") or column.endswith("_count") or column.endswith("_rows"):
        return "Count for the entity named by the column."
    return column.replace("_", " ").capitalize() + "."


def _column_unit(column: str) -> str:
    if "angstrom" in column or "rmsd" in column or column.endswith("_mae"):
        return "angstrom where the column denotes distance"
    if "degrees" in column or "angle" in column or "omega" in column:
        return "degree"
    if "seconds" in column:
        return "second"
    if "ddg" in column:
        return "kcal/mol (upstream convention)"
    if column.endswith("_fraction") or column.startswith("r2_"):
        return "dimensionless"
    return "not_applicable_or_defined_by_metric"


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"frozen publication source is missing: {path}")


def _assert_no_local_paths(path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    if re.search(r"/(?:home|mnt)/", content):
        raise ValueError(f"local absolute path leaked into publication data: {path}")
