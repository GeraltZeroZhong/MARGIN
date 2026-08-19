"""Outcome-blind preparation of the independently locked counterfactual study panel."""

from __future__ import annotations

import re
import subprocess
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1

from margin.config import load_config
from margin.constants import AA_ALPHABET, BACKBONE_ATOMS
from margin.data_registry.cath import read_cath_domain_list
from margin.preprocessing.structure import preprocess_domain_structure
from margin.provenance import runtime_manifest, table_manifest, write_json, write_parquet
from margin.studies.counterfactuals.config import CounterfactualStudyConfig

SINGLE_MUTATION = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")
DESIGN_NAME = re.compile(r"^(?:EA\||GG\||XX\||EHEE_|EEHEE_|HHH_|HEEH_|r\d+_.*TrROS_Hall)")


def prepare_counterfactual_panel(config: CounterfactualStudyConfig) -> dict[str, Path]:
    """Select independent natural/design domains without computing outcome summaries."""

    output = config.paths.run_dir / "panel"
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "domains": output / "domains.parquet",
        "variants": output / "variants.parquet",
        "residues": output / "residues.parquet",
        "queries": output / "query_rows.parquet",
        "exclusions": output / "exclusions.parquet",
        "homology_hits": output / "homology_hits.parquet",
        "structure_hits": output / "design_structure_hits.parquet",
        "manifest": output / "manifest.json",
    }
    if all(path.exists() for path in paths.values()):
        return paths
    _require_inputs(config)

    cath = read_cath_domain_list(config.paths.cath_domain_list)
    current = pd.read_parquet(config.paths.current_assays)
    training = pd.read_parquet(
        config.paths.observability_replication_run / "registry" / "domains.parquet"
    )
    s669_rows, s669_candidates, s669_exclusions = _scan_s669(config, cath, current)
    design_scan, design_exclusions = _scan_megascale_designs(config)
    candidates = pd.concat(
        [
            s669_candidates[["domain_id", "sequence", "structure_path"]],
            design_scan.loc[
                design_scan["metadata_eligible"],
                ["domain_id", "sequence", "structure_path"],
            ],
        ],
        ignore_index=True,
    )
    homology = _run_homology(candidates, current, training, paths["homology_hits"], config)
    strict_homology = set(
        homology.loc[
            homology["sequence_identity"].ge(config.panel.identity_threshold)
            & homology[["query_coverage", "target_coverage"]]
            .min(axis=1)
            .ge(config.panel.minimum_bidirectional_coverage),
            "domain_id",
        ]
    )
    for domain_id in sorted(strict_homology):
        source = str(homology.loc[homology["domain_id"].eq(domain_id), "target_source"].iloc[0])
        reason = (
            "current52_sequence_homology"
            if source == "generalization_52"
            else "observability_homology"
        )
        if domain_id.startswith("s669:"):
            s669_exclusions.append({"domain_id": domain_id, "reason": reason})
        else:
            design_exclusions.append({"domain_id": domain_id, "reason": reason})

    structure_hits = _run_design_structure_search(
        design_scan.loc[design_scan["metadata_eligible"]],
        current,
        paths["structure_hits"],
        config,
    )
    strong_structure = set(
        structure_hits.loc[
            structure_hits[["query_tmscore", "target_tmscore"]]
            .min(axis=1)
            .ge(config.panel.structural_tmscore_threshold)
            & structure_hits[["query_coverage", "target_coverage"]]
            .min(axis=1)
            .ge(config.panel.structural_minimum_bidirectional_coverage),
            "domain_id",
        ]
    )
    design_exclusions.extend(
        {"domain_id": domain_id, "reason": "current52_structure_topology_overlap"}
        for domain_id in sorted(strong_structure)
    )

    s669_excluded = {row["domain_id"] for row in s669_exclusions}
    selected_s669 = s669_candidates.loc[~s669_candidates["domain_id"].isin(s669_excluded)].copy()
    design_excluded = {row["domain_id"] for row in design_exclusions}
    eligible_design = design_scan.loc[
        design_scan["metadata_eligible"] & ~design_scan["domain_id"].isin(design_excluded)
    ].copy()
    selected_design = (
        eligible_design.sort_values(["design_family", "wt_name"], kind="stable")
        .groupby("design_family", sort=False, observed=True)
        .head(config.panel.megascale_domains_per_family)
        .reset_index(drop=True)
    )
    selected_design_ids = set(selected_design["domain_id"])
    for row in eligible_design.loc[
        ~eligible_design["domain_id"].isin(selected_design_ids)
    ].itertuples(index=False):
        design_exclusions.append(
            {"domain_id": row.domain_id, "reason": "deterministic_family_quota"}
        )

    s669_variants = _materialize_s669_variants(s669_rows, selected_s669)
    design_variants = _materialize_megascale_variants(config, selected_design)
    domains = pd.concat(
        [_s669_domain_table(selected_s669), _design_domain_table(selected_design)],
        ignore_index=True,
    ).sort_values("domain_id", ignore_index=True)
    variants = pd.concat([s669_variants, design_variants], ignore_index=True).sort_values(
        ["domain_id", "position", "mutant"], ignore_index=True
    )
    observed_counts = variants.groupby("domain_id", observed=True).size()
    domains["variant_count"] = domains["domain_id"].map(observed_counts).astype(int)
    residues = _materialize_residue_table(domains, config)
    query_rows = (
        variants[["domain_id", "position", "wild_type"]]
        .drop_duplicates(["domain_id", "position"])
        .merge(domains[["domain_id", "sequence"]], on="domain_id", validate="many_to_one")
        .assign(state_id=lambda frame: frame["domain_id"])[
            ["state_id", "domain_id", "position", "wild_type", "sequence"]
        ]
        .sort_values(["state_id", "position"], ignore_index=True)
    )
    exclusions = pd.DataFrame(
        [*s669_exclusions, *design_exclusions], columns=["domain_id", "reason"]
    ).drop_duplicates(ignore_index=True)
    for key, table in (
        ("domains", domains),
        ("variants", variants),
        ("residues", residues),
        ("queries", query_rows),
        ("exclusions", exclusions),
    ):
        write_parquet(paths[key], table)
    write_json(
        paths["manifest"],
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "status": "PREPARED_WITHOUT_MODEL_SCORING_OR_OUTCOME_SUMMARIES",
            "selection_uses_outcome_magnitudes": False,
            "effect_conventions": {
                "s669": "effect=-ddG_experimental; positive means stabilizing",
                "megascale_de_novo": "effect=ddG_ML; positive means stabilizing",
            },
            "selected_domains": int(len(domains)),
            "selected_domains_by_stratum": {
                str(key): int(value) for key, value in domains["stratum"].value_counts().items()
            },
            "selected_variants": int(len(variants)),
            "source_revisions": {
                "proteinmpnn_ddg": _git_revision(config.paths.s669_repository),
                "s669": "Pancotti_et_al_2022_copy_in_proteinmpnn_ddg",
                "megascale": "Zenodo_7992926_April_2023_release",
                "cath": "4.4.0",
            },
            "tables": [
                table_manifest(paths[name], table)
                for name, table in (
                    ("domains", domains),
                    ("variants", variants),
                    ("residues", residues),
                    ("queries", query_rows),
                    ("exclusions", exclusions),
                    ("homology_hits", homology),
                    ("structure_hits", structure_hits),
                )
            ],
        },
    )
    return paths


def _require_inputs(config: CounterfactualStudyConfig) -> None:
    required = [
        config.paths.current_assays,
        config.paths.cath_domain_list,
        config.paths.cath_fasta,
        config.paths.s669_root / "s669_ddg_experimental.csv",
        config.paths.s669_root / "pdb",
        config.paths.megascale_archive,
        config.paths.megascale_structures,
        config.paths.mmseqs_executable,
        config.paths.foldseek_executable,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"counterfactual study inputs are missing: {missing}")


def _scan_s669(
    config: CounterfactualStudyConfig,
    cath: pd.DataFrame,
    current: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, str]]]:
    path = config.paths.s669_root / "s669_ddg_experimental.csv"
    columns = ["pdb_id", "chain", "pre", "pos", "post", "ddG_experimental"]
    rows = pd.read_csv(path, usecols=columns)
    rows["chain"] = rows["chain"].astype(str)
    current_pdb = set(current["assay_id"].str.rsplit("_", n=1).str[-1].str.lower())
    current_cath = cath.loc[cath["pdb_id"].isin(current_pdb)]
    current_h = set(current_cath["cath_h"])
    current_t = set(current_cath["cath_t"])
    candidates: list[dict[str, object]] = []
    exclusions: list[dict[str, str]] = []
    for (pdb_id, requested_chain), frame in rows.groupby(["pdb_id", "chain"], sort=True):
        domain_id = f"s669:{pdb_id}:{requested_chain}"
        path = config.paths.s669_root / "pdb" / f"{pdb_id}.pdb"
        parsed = _read_chain(path, requested_chain)
        reasons: list[str] = []
        if len(frame) < config.panel.s669_minimum_variants_per_domain:
            reasons.append("too_few_domain_variants")
        if config.panel.require_complete_backbone and not parsed["complete_backbone"]:
            reasons.append("incomplete_backbone")
        by_number = parsed["pdb_number_to_position"]
        mutation_mismatch = any(
            int(row.pos) not in by_number
            or parsed["sequence"][by_number[int(row.pos)]] != str(row.pre)
            for row in frame.itertuples(index=False)
        )
        if mutation_mismatch:
            reasons.append("mutation_structure_mismatch")
        classifications = cath.loc[
            cath["pdb_id"].eq(str(pdb_id).lower()) & cath["chain_id"].eq(str(parsed["chain_id"]))
        ]
        if classifications.empty and config.panel.require_s669_cath_assignment:
            reasons.append("cath_assignment_unavailable")
        if classifications["cath_h"].isin(current_h).any():
            reasons.append("current52_cath_h_overlap")
        if classifications["cath_t"].isin(current_t).any():
            reasons.append("current52_cath_t_overlap")
        for reason in reasons:
            exclusions.append({"domain_id": domain_id, "reason": reason})
        candidates.append(
            {
                "domain_id": domain_id,
                "pdb_id": str(pdb_id),
                "chain_id": str(parsed["chain_id"]),
                "requested_chain": str(requested_chain),
                "sequence": parsed["sequence"],
                "length": len(parsed["sequence"]),
                "structure_path": str(path.resolve()),
                "candidate_variant_count": int(len(frame)),
                "pdb_number_to_position": by_number,
                "cath_h": ";".join(sorted(set(classifications["cath_h"]))),
                "cath_t": ";".join(sorted(set(classifications["cath_t"]))),
                "metadata_eligible": not reasons,
            }
        )
    table = pd.DataFrame(candidates)
    return rows, table.loc[table["metadata_eligible"]].reset_index(drop=True), exclusions


def _scan_megascale_designs(
    config: CounterfactualStudyConfig,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    counts: Counter[str] = Counter()
    sequences: dict[str, str] = {}
    clusters: dict[str, str] = {}
    usecols = ["WT_name", "WT_cluster", "aa_seq", "mut_type", "ddG_ML"]
    with (
        zipfile.ZipFile(config.paths.megascale_archive) as archive,
        archive.open(config.panel.megascale_member) as handle,
    ):
        chunks = pd.read_csv(handle, usecols=usecols, chunksize=200_000, low_memory=False)
        for chunk in chunks:
            selected = chunk.loc[chunk["WT_name"].astype(str).str.match(DESIGN_NAME)].copy()
            wild = selected.loc[
                selected["mut_type"].eq("wt"), ["WT_name", "WT_cluster", "aa_seq"]
            ].drop_duplicates()
            for row in wild.itertuples(index=False):
                name = str(row.WT_name)
                sequence = str(row.aa_seq)
                prior = sequences.get(name)
                if prior is not None and prior != sequence:
                    raise ValueError(f"Megascale WT has inconsistent sequences: {name}")
                sequences[name] = sequence
                clusters[name] = str(row.WT_cluster)
            mutation = selected["mut_type"].astype(str).str.match(SINGLE_MUTATION)
            usable = mutation & pd.to_numeric(selected["ddG_ML"], errors="coerce").notna()
            counts.update(selected.loc[usable, "WT_name"].astype(str))
    rows: list[dict[str, object]] = []
    exclusions: list[dict[str, str]] = []
    allowed_families = set(config.panel.megascale_design_families)
    for name, sequence in sorted(sequences.items()):
        family = _design_family(name)
        domain_id = f"megascale:{name}"
        filename = name.replace("|", ":")
        if not filename.endswith(".pdb"):
            filename += ".pdb"
        path = config.paths.megascale_structures / filename
        reasons: list[str] = []
        if family not in allowed_families:
            reasons.append("design_family_not_registered")
        if counts[name] < config.panel.megascale_minimum_single_variants:
            reasons.append("too_few_usable_single_variants")
        if not path.is_file():
            reasons.append("structure_unavailable")
            parsed = None
        else:
            parsed = _read_chain(path, "A")
            if parsed["sequence"] != sequence:
                reasons.append("sequence_structure_mismatch")
            if config.panel.require_complete_backbone and not parsed["complete_backbone"]:
                reasons.append("incomplete_backbone")
        for reason in reasons:
            exclusions.append({"domain_id": domain_id, "reason": reason})
        rows.append(
            {
                "domain_id": domain_id,
                "wt_name": name,
                "design_family": family,
                "design_cluster": clusters.get(name, ""),
                "sequence": sequence,
                "length": len(sequence),
                "chain_id": str(parsed["chain_id"]) if parsed else "",
                "structure_path": str(path.resolve()),
                "candidate_variant_count": int(counts[name]),
                "metadata_eligible": not reasons,
            }
        )
    return pd.DataFrame(rows), exclusions


def _run_homology(
    candidates: pd.DataFrame,
    current: pd.DataFrame,
    training: pd.DataFrame,
    output_path: Path,
    config: CounterfactualStudyConfig,
) -> pd.DataFrame:
    query_path = output_path.with_suffix(".queries.fasta")
    target_path = output_path.with_suffix(".targets.fasta")
    query_path.write_text(_fasta(candidates["domain_id"], candidates["sequence"]), encoding="utf-8")
    targets = pd.concat(
        [
            current[["assay_id", "sequence"]]
            .rename(columns={"assay_id": "target_id"})
            .assign(target_source="generalization_52"),
            training[["domain_id", "sequence"]]
            .rename(columns={"domain_id": "target_id"})
            .assign(target_source="observability_cath"),
        ],
        ignore_index=True,
    )
    encoded_ids = [f"target_{index:04d}" for index in range(len(targets))]
    target_path.write_text(_fasta(pd.Series(encoded_ids), targets["sequence"]), encoding="utf-8")
    target_lookup = pd.DataFrame(
        {
            "encoded_target": encoded_ids,
            "target_id": targets["target_id"].astype(str),
            "target_source": targets["target_source"].astype(str),
        }
    )
    with tempfile.TemporaryDirectory(
        prefix="counterfactuals-mmseqs-", dir=output_path.parent
    ) as name:
        temporary = Path(name)
        raw = temporary / "hits.tsv"
        command = [
            str(config.paths.mmseqs_executable),
            "easy-search",
            str(query_path),
            str(target_path),
            str(raw),
            str(temporary / "work"),
            "--format-output",
            "query,target,fident,qcov,tcov",
            "-s",
            str(config.panel.homology_sensitivity),
            "-e",
            str(config.panel.homology_evalue),
            "--max-seqs",
            str(len(targets)),
            "--threads",
            str(config.panel.homology_threads),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"MMseqs2 failed: {completed.stderr[-2000:]}")
        if raw.stat().st_size:
            hits = pd.read_csv(
                raw,
                sep="\t",
                names=["domain_id", "encoded_target", "fident", "qcov", "tcov"],
            )
        else:
            hits = pd.DataFrame(columns=["domain_id", "encoded_target", "fident", "qcov", "tcov"])
    for column in ("fident", "qcov", "tcov"):
        hits[column] = pd.to_numeric(hits[column], errors="coerce")
        if len(hits) and hits[column].max() > 1:
            hits[column] /= 100.0
    hits = hits.merge(target_lookup, on="encoded_target", how="left", validate="many_to_one")
    result = hits.rename(
        columns={
            "fident": "sequence_identity",
            "qcov": "query_coverage",
            "tcov": "target_coverage",
        }
    )[
        [
            "domain_id",
            "target_id",
            "target_source",
            "sequence_identity",
            "query_coverage",
            "target_coverage",
        ]
    ]
    write_parquet(output_path, result)
    return result


def _run_design_structure_search(
    designs: pd.DataFrame,
    current: pd.DataFrame,
    output_path: Path,
    config: CounterfactualStudyConfig,
) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(
        prefix="counterfactuals-foldseek-", dir=output_path.parent
    ) as name:
        temporary = Path(name)
        query_dir = temporary / "queries"
        target_dir = temporary / "targets"
        query_dir.mkdir()
        target_dir.mkdir()
        query_lookup: dict[str, str] = {}
        for index, row in enumerate(designs.sort_values("domain_id").itertuples(index=False)):
            key = f"query_{index:04d}"
            (query_dir / f"{key}.pdb").symlink_to(Path(row.structure_path))
            query_lookup[key] = str(row.domain_id)
        target_lookup: dict[str, str] = {}
        pdb_ids = sorted(set(current["assay_id"].str.rsplit("_", n=1).str[-1]))
        for index, pdb_id in enumerate(pdb_ids):
            source = config.paths.tsuboyama_reference_structures / f"{pdb_id}.pdb"
            if not source.is_file():
                source = config.paths.current52_supplemental_structures / f"{pdb_id}.pdb"
            if not source.is_file():
                raise FileNotFoundError(
                    f"current generalization study structure is unavailable: {pdb_id}"
                )
            key = f"target_{index:04d}"
            (target_dir / f"{key}.pdb").symlink_to(source)
            target_lookup[key] = pdb_id
        raw = temporary / "hits.tsv"
        command = [
            str(config.paths.foldseek_executable),
            "easy-search",
            str(query_dir),
            str(target_dir),
            str(raw),
            str(temporary / "work"),
            "--format-output",
            "query,target,fident,alntmscore,qtmscore,ttmscore,qcov,tcov,evalue",
            "--threads",
            str(config.panel.foldseek_threads),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"Foldseek failed: {completed.stderr[-2000:]}")
        if raw.stat().st_size:
            hits = pd.read_csv(
                raw,
                sep="\t",
                names=[
                    "query",
                    "target",
                    "sequence_identity",
                    "alignment_tmscore",
                    "query_tmscore",
                    "target_tmscore",
                    "query_coverage",
                    "target_coverage",
                    "evalue",
                ],
            )
        else:
            hits = pd.DataFrame(
                columns=[
                    "query",
                    "target",
                    "sequence_identity",
                    "alignment_tmscore",
                    "query_tmscore",
                    "target_tmscore",
                    "query_coverage",
                    "target_coverage",
                    "evalue",
                ]
            )
    hits["domain_id"] = hits["query"].map(query_lookup)
    hits["current52_pdb_id"] = hits["target"].map(target_lookup)
    result = hits[
        [
            "domain_id",
            "current52_pdb_id",
            "sequence_identity",
            "alignment_tmscore",
            "query_tmscore",
            "target_tmscore",
            "query_coverage",
            "target_coverage",
            "evalue",
        ]
    ].copy()
    write_parquet(output_path, result)
    return result


def _materialize_s669_variants(
    rows: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    lookup = selected.set_index(["pdb_id", "requested_chain"])
    result: list[dict[str, object]] = []
    for row in rows.itertuples(index=False):
        key = (str(row.pdb_id), str(row.chain))
        if key not in lookup.index:
            continue
        domain = lookup.loc[key]
        position = int(domain.pdb_number_to_position[int(row.pos)])
        result.append(
            {
                "domain_id": domain.domain_id,
                "position": position,
                "pdb_position": int(row.pos),
                "wild_type": str(row.pre),
                "mutant": str(row.post),
                "effect": -float(row.ddG_experimental),
                "source": "S669",
                "stratum": "natural",
            }
        )
    return pd.DataFrame(result)


def _materialize_megascale_variants(
    config: CounterfactualStudyConfig,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    selected_names = set(selected["wt_name"])
    sequence_lookup = selected.set_index("wt_name")["sequence"].to_dict()
    domain_lookup = selected.set_index("wt_name")["domain_id"].to_dict()
    frames: list[pd.DataFrame] = []
    usecols = ["WT_name", "mut_type", "ddG_ML"]
    with (
        zipfile.ZipFile(config.paths.megascale_archive) as archive,
        archive.open(config.panel.megascale_member) as handle,
    ):
        chunks = pd.read_csv(handle, usecols=usecols, chunksize=200_000, low_memory=False)
        for chunk in chunks:
            chunk = chunk.loc[chunk["WT_name"].astype(str).isin(selected_names)].copy()
            parsed = chunk["mut_type"].astype(str).str.extract(SINGLE_MUTATION)
            numeric = pd.to_numeric(chunk["ddG_ML"], errors="coerce")
            keep = parsed.notna().all(axis=1) & numeric.notna()
            if not keep.any():
                continue
            parsed = parsed.loc[keep]
            names = chunk.loc[keep, "WT_name"].astype(str)
            frame = pd.DataFrame(
                {
                    "domain_id": names.map(domain_lookup).to_numpy(),
                    "position": parsed[1].astype(int).to_numpy() - 1,
                    "pdb_position": parsed[1].astype(int).to_numpy(),
                    "wild_type": parsed[0].to_numpy(),
                    "mutant": parsed[2].to_numpy(),
                    "effect": numeric.loc[keep].astype(float).to_numpy(),
                    "source": "Megascale_2023",
                    "stratum": "de_novo",
                    "wt_name": names.to_numpy(),
                }
            )
            expected = np.array(
                [
                    sequence_lookup[name][position]
                    for name, position in zip(names, frame["position"], strict=True)
                ]
            )
            frame = frame.loc[frame["wild_type"].to_numpy() == expected].drop(columns="wt_name")
            frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["domain_id", "position", "wild_type", "mutant"]).any():
        result = result.groupby(
            [
                "domain_id",
                "position",
                "pdb_position",
                "wild_type",
                "mutant",
                "source",
                "stratum",
            ],
            as_index=False,
            observed=True,
        )["effect"].median()
    return result


def _s669_domain_table(selected: pd.DataFrame) -> pd.DataFrame:
    result = selected[
        [
            "domain_id",
            "sequence",
            "length",
            "structure_path",
            "chain_id",
            "pdb_id",
            "cath_h",
            "cath_t",
        ]
    ].copy()
    result["source"] = "S669"
    result["stratum"] = "natural"
    result["platform"] = "direct_experimental_ddG"
    result["structure_kind"] = "experimental_PDB"
    result["design_family"] = "not_applicable"
    result["evaluation_role"] = "counterfactuals_locked_confirmatory"
    return result


def _design_domain_table(selected: pd.DataFrame) -> pd.DataFrame:
    result = selected[
        [
            "domain_id",
            "sequence",
            "length",
            "structure_path",
            "chain_id",
            "design_family",
        ]
    ].copy()
    result["pdb_id"] = ""
    result["cath_h"] = "unassigned_de_novo"
    result["cath_t"] = "unassigned_de_novo"
    result["source"] = "Megascale_2023"
    result["stratum"] = "de_novo"
    result["platform"] = "cDNA_display_proteolysis"
    result["structure_kind"] = "AlphaFold_model_archive"
    result["evaluation_role"] = "counterfactuals_locked_confirmatory"
    return result


def _materialize_residue_table(
    domains: pd.DataFrame,
    config: CounterfactualStudyConfig,
) -> pd.DataFrame:
    foundation = load_config(config.paths.foundation_config)
    frames: list[pd.DataFrame] = []
    for domain in domains.itertuples(index=False):
        table, _ = preprocess_domain_structure(
            str(domain.domain_id),
            str(domain.sequence),
            Path(domain.structure_path),
            str(domain.chain_id),
            foundation.registry,
        )
        table["source"] = domain.source
        table["stratum"] = domain.stratum
        frames.append(table)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["domain_id", "position"], ignore_index=True
    )


def _read_chain(path: Path, requested_chain: str) -> dict[str, object]:
    model = next(PDBParser(QUIET=True).get_structure(path.stem, str(path)).get_models())
    chains = list(model.get_chains())
    chain = next((item for item in chains if item.id == requested_chain), None)
    if chain is None and len(chains) == 1:
        chain = chains[0]
    if chain is None:
        raise ValueError(f"chain {requested_chain!r} is unavailable in {path}")
    residues = []
    number_to_position: dict[int, int] = {}
    complete = True
    for residue in chain.get_residues():
        if residue.id[0].strip():
            continue
        letter = protein_letters_3to1.get(residue.resname.upper())
        if letter is None:
            continue
        position = len(residues)
        residues.append(letter)
        number = int(residue.id[1])
        if number in number_to_position:
            raise ValueError(f"ambiguous residue number {number} in {path} chain {chain.id}")
        number_to_position[number] = position
        complete &= all(atom in residue for atom in BACKBONE_ATOMS)
    if not residues or set(residues) - set(AA_ALPHABET):
        raise ValueError(f"no canonical protein chain found in {path}")
    return {
        "sequence": "".join(residues),
        "chain_id": chain.id,
        "pdb_number_to_position": number_to_position,
        "complete_backbone": bool(complete),
    }


def _design_family(name: str) -> str:
    if re.match(r"^r\d+_.*TrROS_Hall", name):
        return "trRosetta_hallucination"
    return re.split(r"[|_:]", name, maxsplit=1)[0]


def _fasta(identifiers: pd.Series, sequences: pd.Series) -> str:
    return "".join(
        f">{identifier}\n{sequence}\n"
        for identifier, sequence in zip(identifiers, sequences, strict=True)
    )


def _git_revision(path: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=False, capture_output=True, text=True
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"
