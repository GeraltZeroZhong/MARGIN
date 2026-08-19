"""Sequence-only homolog profiles for the strengthened stability study control."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from margin.constants import AA_ALPHABET, AA_TO_INDEX
from margin.provenance import write_parquet, write_text
from margin.studies.stability.config import StabilityStudyConfig

ALIGNMENT_COLUMNS = ["query", "target", "qstart", "qaln", "taln", "evalue"]


def build_profiles(config: StabilityStudyConfig) -> dict[str, pd.DataFrame]:
    """Build CATH-training and stability study panel residue-frequency profiles."""

    output = config.paths.run_dir / "strong_control"
    output.mkdir(parents=True, exist_ok=True)
    cath_path = output / "cath_profiles.parquet"
    panel_path = output / "panel_profiles.parquet"
    alignment_path = output / "panel_alignments.tsv"
    cath_queries = pd.read_parquet(config.paths.cath_queries)
    panel_queries = pd.read_parquet(config.paths.run_dir / "panel" / "query_rows.parquet")
    if cath_path.exists() and panel_path.exists():
        return {
            "cath": pd.read_parquet(cath_path),
            "panel": pd.read_parquet(panel_path),
        }
    cath_alignments = _read_alignments(config.paths.cath_conservation_alignments)
    cath_profiles = profile_queries(cath_queries, cath_alignments, config)
    write_parquet(cath_path, cath_profiles)
    if not alignment_path.exists():
        _search_panel(panel_queries, alignment_path, config)
    panel_alignments = _read_alignments(alignment_path)
    panel_profiles = profile_queries(panel_queries, panel_alignments, config)
    write_parquet(panel_path, panel_profiles)
    return {"cath": cath_profiles, "panel": panel_profiles}


def profile_queries(
    queries: pd.DataFrame,
    alignments: pd.DataFrame,
    config: StabilityStudyConfig,
) -> pd.DataFrame:
    """Convert filtered alignments into pseudocount-smoothed amino-acid profiles."""

    key_columns = ["state_id", "domain_id", "position"]
    rows = []
    for domain_id, query_frame in queries.groupby("domain_id", sort=True, observed=True):
        sequences = query_frame["sequence"].drop_duplicates()
        if len(sequences) != 1:
            raise ValueError(f"profile query {domain_id} has inconsistent sequences")
        sequence = str(sequences.iloc[0])
        counts = np.zeros((len(sequence), len(AA_ALPHABET)), dtype=np.float64)
        observations = np.zeros(len(sequence), dtype=np.int64)
        accepted_hits = 0
        for hit in alignments.loc[alignments["query"].eq(domain_id)].itertuples(index=False):
            metrics = _alignment_metrics(str(hit.qaln), str(hit.taln), len(sequence))
            if metrics["identity"] < config.strong_control.profile_minimum_identity:
                continue
            if metrics["identity"] >= config.strong_control.profile_maximum_identity:
                continue
            if metrics["query_coverage"] < config.strong_control.profile_minimum_query_coverage:
                continue
            accepted_hits += 1
            position = int(hit.qstart) - 2
            for query_aa, target_aa in zip(str(hit.qaln), str(hit.taln), strict=True):
                if query_aa == "-":
                    continue
                position += 1
                target_aa = target_aa.upper()
                if 0 <= position < len(sequence) and target_aa in AA_TO_INDEX:
                    counts[position, AA_TO_INDEX[target_aa]] += 1.0
                    observations[position] += 1
        pseudocount = config.strong_control.profile_pseudocount
        frequency = (counts + pseudocount) / (
            observations[:, None] + pseudocount * len(AA_ALPHABET)
        )
        entropy = -np.sum(frequency * np.log(frequency), axis=1)
        for query in query_frame.itertuples(index=False):
            position = int(query.position)
            row = {
                "state_id": query.state_id,
                "domain_id": domain_id,
                "position": position,
                "homolog_observations": int(observations[position]),
                "accepted_homolog_hits": int(accepted_hits),
                "profile_entropy": float(entropy[position]),
                "profile_covered": bool(observations[position] > 0),
            }
            row.update(
                {
                    f"profile_{aa}": float(frequency[position, index])
                    for index, aa in enumerate(AA_ALPHABET)
                }
            )
            rows.append(row)
    result = pd.DataFrame(rows).sort_values(key_columns, ignore_index=True)
    if result.duplicated(key_columns).any() or len(result) != len(queries):
        raise ValueError("profile output does not preserve one row per query")
    return result


def _alignment_metrics(qaln: str, taln: str, query_length: int) -> dict[str, float]:
    aligned_query = 0
    valid_pairs = 0
    matches = 0
    for query_aa, target_aa in zip(qaln, taln, strict=True):
        if query_aa != "-":
            aligned_query += 1
        if query_aa.upper() in AA_TO_INDEX and target_aa.upper() in AA_TO_INDEX:
            valid_pairs += 1
            matches += int(query_aa.upper() == target_aa.upper())
    return {
        "identity": matches / max(valid_pairs, 1),
        "query_coverage": aligned_query / max(query_length, 1),
    }


def _search_panel(queries: pd.DataFrame, output_path: Path, config: StabilityStudyConfig) -> None:
    fasta = output_path.with_suffix(".queries.fasta")
    domains = queries[["domain_id", "sequence"]].drop_duplicates("domain_id")
    write_text(
        fasta,
        "".join(f">{row.domain_id}\n{row.sequence}\n" for row in domains.itertuples(index=False)),
    )
    with tempfile.TemporaryDirectory(prefix="stability-profile-", dir=output_path.parent) as name:
        command = [
            str(config.paths.mmseqs_executable),
            "easy-search",
            str(fasta),
            str(config.paths.cath_fasta),
            str(output_path),
            str(Path(name) / "work"),
            "-s",
            str(config.panel.homology_sensitivity),
            "-e",
            str(config.panel.homology_evalue),
            "--threads",
            str(config.panel.homology_threads),
            "--max-seqs",
            "500",
            "--format-output",
            "query,target,qstart,qaln,taln,evalue",
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"stability study profile search failed: {completed.stderr[-2000:]}")


def _read_alignments(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", names=ALIGNMENT_COLUMNS)
