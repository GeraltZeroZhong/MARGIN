#!/usr/bin/env python
"""Score the frozen cross-platform experimental structures with three teachers."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from margin.config import TeacherSpec, load_config
from margin.provenance import read_json, runtime_manifest, write_json, write_parquet
from margin.studies.external_validation.panel import load_external_validation_config
from margin.studies.stability.config import load_stability_config
from margin.teachers.external import run_external_teacher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/external_validation.yaml"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--teacher", choices=["all", "mif", "esm_if1", "proteinmpnn"], default="all"
    )
    arguments = parser.parse_args()
    config = load_external_validation_config(arguments.protocol)
    lock = read_json(config.paths.run_dir / "protocol_lock.json")
    if lock.get("status") != config.status:
        raise RuntimeError("cross-platform protocol lock is missing")
    stability = load_stability_config(config.paths.stability_config)
    foundation = load_config(config.paths.foundation_config)
    foundation.seed = config.seed
    teachers = _teachers(stability, foundation)
    requested = list(teachers) if arguments.teacher == "all" else [arguments.teacher]
    request_path = config.paths.run_dir / "teacher_requests" / "requests.parquet"
    output = config.paths.run_dir / "teacher_scores"
    raw_directory = output / "raw"
    raw_directory.mkdir(parents=True, exist_ok=True)
    for teacher_id in requested:
        canonical = run_external_teacher(
            teachers[teacher_id],
            request_path,
            raw_directory / f"{teacher_id}.parquet",
            foundation,
            device=arguments.device,
        )
        write_parquet(raw_directory / f"{teacher_id}.canonical.parquet", canonical)
        print(f"complete_teacher={teacher_id} rows={len(canonical)}", flush=True)
    canonical_paths = [raw_directory / f"{teacher}.canonical.parquet" for teacher in teachers]
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
    write_parquet(output / "scores.parquet", scores)
    coverage = (
        scores.groupby(["teacher_id", "analysis_population"], observed=True)
        .agg(rows=("position", "size"), domains=("domain_id", "nunique"))
        .reset_index()
    )
    write_parquet(output / "coverage.parquet", coverage)
    write_json(
        output / "manifest.json",
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "teachers": [teacher.model_dump(mode="json") for teacher in teachers.values()],
            "paired_experimental_structures_only": True,
            "panel_stability_labels_read": False,
            "score_rows": len(scores),
            "score_path": str(output / "scores.parquet"),
        },
    )
    print(f"combined_scores={output / 'scores.parquet'}")


def _teachers(stability, foundation) -> dict[str, TeacherSpec]:
    configured = {teacher.teacher_id: teacher for teacher in foundation.teacher_cache.teachers}
    proteinmpnn = configured["proteinmpnn"].model_copy(
        update={
            "repository": stability.paths.proteinmpnn_repository,
            "weights": stability.paths.proteinmpnn_checkpoint,
            "order_repeats": stability.calibration.proteinmpnn_order_repeats,
        }
    )
    esm_if1 = configured["esm_if1"].model_copy(
        update={
            "repository": stability.paths.esm_if1_repository,
            "weights": stability.paths.esm_if1_checkpoint,
        }
    )
    mif = TeacherSpec(
        teacher_id="mif",
        adapter="mifst",
        role="audit_structure",
        model_name="mif",
        model_revision="zenodo-6573779-mif",
        conda_env="margin-models",
        repository=stability.paths.sequence_models_repository,
        repository_revision="af695772c4a1c056d930c95ec7e6428aa042f5cd",
        weights=stability.paths.mif_checkpoint,
        score_type="log_probability",
        batch_size=stability.calibration.mif_batch_size,
    )
    return {"mif": mif, "esm_if1": esm_if1, "proteinmpnn": proteinmpnn}


if __name__ == "__main__":
    main()
