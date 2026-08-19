"""Command-line entry points for audits, adapters, and workflow discovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from margin.config import load_config
from margin.data_registry.homology import build_mmseqs_homology
from margin.data_registry.registry import load_registry
from margin.doctor import diagnose
from margin.pipeline import (
    build_candidates_stage,
    build_state_bank_stage,
    prepare_registry_stage,
    run_foundation_audit,
)
from margin.teachers.external import run_external_teacher
from margin.workflows import WORKFLOWS, get_workflow


def main() -> None:
    parser = _parser()
    arguments = parser.parse_args()
    if arguments.command == "list-workflows":
        for workflow in WORKFLOWS:
            print(f"{workflow.name:24} {workflow.purpose}")
        return
    if arguments.command == "describe-workflow":
        try:
            workflow = get_workflow(arguments.name)
        except KeyError as error:
            raise SystemExit(str(error)) from error
        print(f"name={workflow.name}")
        print(f"purpose={workflow.purpose}")
        print(f"package={workflow.package}")
        print(f"config={workflow.config}")
        print(f"scripts={workflow.scripts}")
        return
    if arguments.command == "validate-config":
        config = load_config(arguments.config)
        print(config.model_dump_json(indent=2))
        return
    if arguments.command == "doctor":
        table = diagnose(load_config(arguments.config))
        if arguments.json:
            print(json.dumps(table.to_dict(orient="records"), indent=2))
        else:
            print(table.to_string(index=False))
        raise SystemExit(1 if (table["status"] == "FAIL").any() else 0)
    if arguments.command == "run":
        result = run_foundation_audit(arguments.config, device=arguments.device)
        print(f"decision={result.decision.decision}")
        print(f"report={result.report_path}")
        print(f"manifest={result.run_manifest_path}")
        return
    if arguments.command == "build-candidates":
        directory = build_candidates_stage(arguments.config)
        print(f"candidate_registry={directory}")
        return
    if arguments.command == "prepare-registry":
        registry, leakage = prepare_registry_stage(arguments.config)
        print(f"registry_domains={len(registry.domains)}")
        print(f"eligible_training_domains={leakage.summary['eligible_domains']}")
        return
    if arguments.command == "build-state-bank":
        directory = build_state_bank_stage(arguments.config, arguments.registry)
        print(f"state_bank={directory}")
        return
    if arguments.command == "score-teacher":
        config = load_config(arguments.config)
        teachers = {teacher.teacher_id: teacher for teacher in config.teacher_cache.teachers}
        if arguments.teacher_id not in teachers:
            raise SystemExit(f"unknown teacher_id: {arguments.teacher_id}")
        table = run_external_teacher(
            teachers[arguments.teacher_id],
            arguments.requests,
            arguments.output,
            config,
            device=arguments.device,
            limit=arguments.limit,
        )
        canonical = arguments.output.with_name(f"{arguments.output.stem}.canonical.parquet")
        table.to_parquet(canonical, index=False)
        print(f"canonical_scores={canonical}")
        return
    if arguments.command == "build-homology":
        config = load_config(arguments.config)
        registry = load_registry(arguments.registry)
        if arguments.benchmarks.suffix.lower() == ".parquet":
            import pandas as pd

            benchmarks = pd.read_parquet(arguments.benchmarks)
        else:
            import pandas as pd

            benchmarks = pd.read_csv(
                arguments.benchmarks,
                sep="\t" if arguments.benchmarks.suffix.lower() == ".tsv" else ",",
            )
        hits, _ = build_mmseqs_homology(registry.domains, benchmarks, arguments.output, config)
        print(f"homology_hits={arguments.output} rows={len(hits)}")
        return
    parser.error("a command is required")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="margin")
    commands = parser.add_subparsers(dest="command")
    validate = commands.add_parser("validate-config", help="validate and resolve a YAML config")
    validate.add_argument("--config", type=Path, required=True)
    doctor = commands.add_parser(
        "doctor", help="check configured files, environments, and adapters"
    )
    doctor.add_argument("--config", type=Path, required=True)
    doctor.add_argument("--json", action="store_true")
    commands.add_parser("list-workflows", help="list workflows by scientific purpose")
    describe = commands.add_parser(
        "describe-workflow", help="show the package, config, and scripts for one workflow"
    )
    describe.add_argument("name")
    run = commands.add_parser("run", help="execute the complete foundation audit")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--device", default="auto")
    candidates = commands.add_parser(
        "build-candidates", help="build the filtered candidate registry for homology search"
    )
    candidates.add_argument("--config", type=Path, required=True)
    registry = commands.add_parser(
        "prepare-registry", help="apply leakage exclusions and assemble the unified registry"
    )
    registry.add_argument("--config", type=Path, required=True)
    states = commands.add_parser(
        "build-state-bank", help="materialize states before exporting frozen embeddings"
    )
    states.add_argument("--config", type=Path, required=True)
    states.add_argument("--registry", type=Path)
    score = commands.add_parser("score-teacher", help="run one isolated external teacher")
    score.add_argument("--config", type=Path, required=True)
    score.add_argument("--teacher-id", required=True)
    score.add_argument("--requests", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--device", default="auto")
    score.add_argument("--limit", type=int)
    homology = commands.add_parser(
        "build-homology", help="build a reproducible MMseqs2 benchmark-hit table"
    )
    homology.add_argument("--config", type=Path, required=True)
    homology.add_argument("--registry", type=Path, required=True)
    homology.add_argument("--benchmarks", type=Path, required=True)
    homology.add_argument("--output", type=Path, required=True)
    return parser


if __name__ == "__main__":
    main()
