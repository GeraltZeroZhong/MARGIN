#!/usr/bin/env python
"""Score the locked action-validation study paired structures with three registered teachers."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from margin.config import TeacherSpec, load_config
from margin.provenance import read_json, runtime_manifest, table_manifest, write_json, write_parquet
from margin.studies.action_validation.config import load_action_validation_config
from margin.teachers.external import run_external_teacher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/action_validation.yaml"))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--teacher",
        choices=["all", "mif", "esm_if1", "proteinmpnn"],
        default="all",
    )
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args()
    config = load_action_validation_config(arguments.config)
    lock = read_json(config.paths.run_dir / "protocol_lock.json")
    if lock.get("status") != "FROZEN_BEFORE_ACTION_VALIDATION_PANEL_MODEL_SCORING":
        raise RuntimeError("action-validation study protocol lock is missing")
    foundation = load_config(config.paths.foundation_config)
    foundation.seed = config.seed
    teachers = _teachers(config, foundation)
    requested = list(teachers) if arguments.teacher == "all" else [arguments.teacher]
    request_path = config.paths.run_dir / "teacher_requests" / "requests.parquet"
    output = config.paths.run_dir / "teacher_scores"
    raw_directory = output / "raw"
    raw_directory.mkdir(parents=True, exist_ok=True)
    for teacher_id in requested:
        teacher = teachers[teacher_id]
        raw_path = raw_directory / f"{teacher_id}.parquet"
        canonical = run_external_teacher(
            teacher,
            request_path,
            raw_path,
            foundation,
            device=arguments.device,
            limit=arguments.limit,
        )
        write_parquet(raw_directory / f"{teacher_id}.canonical.parquet", canonical)
        print(f"complete_teacher={teacher_id} rows={len(canonical)}", flush=True)
    if arguments.limit is not None:
        return
    canonical_paths = [raw_directory / f"{teacher_id}.canonical.parquet" for teacher_id in teachers]
    if not all(path.exists() for path in canonical_paths):
        print("combined_scores=pending_other_teachers")
        return
    scores = pd.concat([pd.read_parquet(path) for path in canonical_paths], ignore_index=True)
    structures = pd.read_parquet(request_path.parent / "structures.parquet")
    scores = scores.merge(
        structures[["structure_id", "analysis_population"]],
        on="structure_id",
        validate="many_to_one",
    ).sort_values(["teacher_id", "domain_id", "position"], ignore_index=True)
    score_path = output / "scores.parquet"
    write_parquet(score_path, scores)
    coverage = (
        scores.groupby(["teacher_id", "analysis_population"], observed=True)
        .agg(rows=("position", "size"), domains=("domain_id", "nunique"))
        .reset_index()
    )
    coverage_path = output / "coverage.parquet"
    write_parquet(coverage_path, coverage)
    manifest_path = output / "manifest.json"
    write_json(
        manifest_path,
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "teachers": [teacher.model_dump(mode="json") for teacher in teachers.values()],
            "paired_structures_only": True,
            "counterfactuals_scored": False,
            "panel_stability_labels_read": False,
            "scores": table_manifest(score_path, scores),
            "coverage": table_manifest(coverage_path, coverage),
        },
    )
    print(f"combined_scores={score_path}")


def _teachers(config, foundation) -> dict[str, TeacherSpec]:
    configured = {teacher.teacher_id: teacher for teacher in foundation.teacher_cache.teachers}
    proteinmpnn = configured["proteinmpnn"].model_copy(
        update={
            "repository": config.paths.proteinmpnn_repository,
            "weights": config.paths.proteinmpnn_checkpoint,
            "order_repeats": config.decomposition.proteinmpnn_order_repeats,
        }
    )
    esm_if1 = configured["esm_if1"].model_copy(
        update={
            "repository": config.paths.esm_if1_repository,
            "weights": config.paths.esm_if1_checkpoint,
        }
    )
    mif = TeacherSpec(
        teacher_id="mif",
        adapter="mifst",
        role="audit_structure",
        model_name="mif",
        model_revision="zenodo-6573779-mif",
        conda_env="margin-models",
        repository=config.paths.sequence_models_repository,
        repository_revision="af695772c4a1c056d930c95ec7e6428aa042f5cd",
        weights=config.paths.mif_checkpoint,
        score_type="log_probability",
        batch_size=config.decomposition.mif_batch_size,
    )
    return {"mif": mif, "esm_if1": esm_if1, "proteinmpnn": proteinmpnn}


if __name__ == "__main__":
    main()
