#!/usr/bin/env python
"""Score the frozen counterfactual study paired and counterfactual requests with plain MIF."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from margin.config import TeacherSpec, load_config
from margin.provenance import runtime_manifest, table_manifest, write_json, write_parquet
from margin.studies.counterfactuals.config import load_counterfactual_config
from margin.studies.generalization.config import load_generalization_config
from margin.teachers.external import run_external_teacher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/counterfactuals.yaml"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args()
    config = load_counterfactual_config(arguments.config)
    generalization = load_generalization_config(config.paths.generalization_config)
    foundation = load_config(generalization.paths.observability_replication_config)
    foundation.seed = config.seed
    request_path = config.paths.run_dir / "mif_requests" / "requests.parquet"
    output_name = "mif" if arguments.limit is None else f"mif_smoke_{arguments.limit}"
    output = config.paths.run_dir / output_name
    output.mkdir(parents=True, exist_ok=True)
    score_path = output / "scores.parquet"
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and score_path.exists():
        print(f"complete={manifest_path}")
        return
    teacher = TeacherSpec(
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
        batch_size=config.models.mif_batch_size,
    )
    canonical = run_external_teacher(
        teacher,
        request_path,
        output / "raw.parquet",
        foundation,
        device=arguments.device,
        limit=arguments.limit,
    )
    structures = pd.read_parquet(request_path.parent / "structures.parquet")
    canonical = canonical.merge(
        structures[["structure_id", "analysis_population"]],
        on="structure_id",
        validate="many_to_one",
    )
    write_parquet(score_path, canonical)
    write_json(
        manifest_path,
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "teacher_id": "mif",
            "model_revision": teacher.model_revision,
            "batch_size": teacher.batch_size,
            "limit": arguments.limit,
            "locked_panel_outcomes_read": False,
            "conditioning_roles": sorted(canonical["structure_role"].unique().tolist()),
            "analysis_populations": sorted(canonical["analysis_population"].unique().tolist()),
            "scores": table_manifest(score_path, canonical),
        },
    )
    print(f"complete={manifest_path}")


if __name__ == "__main__":
    main()
