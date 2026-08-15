"""Preflight checks for the exact inputs needed by a configured foundation audit run."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from margin.config import ProjectConfig


def diagnose(config: ProjectConfig) -> pd.DataFrame:
    """Report actionable prerequisites without modifying the environment."""

    rows: list[dict[str, str]] = []
    _record(rows, "python_package", "PASS", "configuration validated")
    if config.registry.require_dssp:
        _record(
            rows,
            "dssp",
            "PASS" if shutil.which(config.registry.dssp_executable) else "FAIL",
            config.registry.dssp_executable,
        )
    if config.data_mode == "real":
        if config.paths.domain_input is not None:
            _path_check(rows, "canonical_registry", config.paths.domain_input)
        else:
            _path_check(rows, "cath_domain_list", config.paths.cath_domain_list)
            _path_check(rows, "cath_fasta", config.paths.cath_fasta)
            _path_check(rows, "cath_structures", config.paths.structures_dir)
        _path_check(rows, "external_audit_registry", config.paths.audit_domain_input)
        _path_check(rows, "benchmark_registry", config.paths.benchmark_input)
        _path_check(rows, "homology_hits", config.paths.homology_hits_input)
        _path_check(rows, "residue_conservation", config.paths.conservation_input)
        _path_check(rows, "dms_variants", config.paths.dms_input)
        _path_check(rows, "student_embeddings", config.paths.embeddings_input)
        _record(
            rows,
            "homology_engine",
            "PASS" if shutil.which(config.homology.executable) else "FAIL",
            config.homology.executable,
        )
    policy = config.student_policy
    if policy.adapter == "python_factory":
        module_name = str(policy.factory).split(":", maxsplit=1)[0]
        _record(
            rows,
            "student_policy_factory",
            "PASS" if importlib.util.find_spec(module_name) is not None else "FAIL",
            str(policy.factory),
        )
    elif policy.scores_input is not None:
        _path_check(rows, "student_policy_scores", policy.scores_input)
    conda_envs = _conda_environments()
    for teacher in config.teacher_cache.teachers:
        if not teacher.enabled or teacher.role == "sequence":
            continue
        if teacher.adapter == "synthetic":
            _record(
                rows,
                f"teacher_adapter:{teacher.teacher_id}",
                "PASS",
                "in-process synthetic fixture",
            )
            continue
        _record(
            rows,
            f"teacher_env:{teacher.teacher_id}",
            "PASS" if teacher.conda_env in conda_envs else "FAIL",
            str(teacher.conda_env),
        )
        if teacher.repository is not None:
            _path_check(rows, f"teacher_repo:{teacher.teacher_id}", teacher.repository)
            if teacher.repository_revision is not None and teacher.repository.exists():
                completed = subprocess.run(
                    ["git", "-C", str(teacher.repository), "rev-parse", "HEAD"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                observed = completed.stdout.strip()
                _record(
                    rows,
                    f"teacher_repo_revision:{teacher.teacher_id}",
                    "PASS" if observed == teacher.repository_revision else "FAIL",
                    f"expected={teacher.repository_revision} observed={observed}",
                )
        if teacher.weights is not None:
            _path_check(rows, f"teacher_weights:{teacher.teacher_id}", teacher.weights)
        if teacher.auxiliary_weights is not None:
            _path_check(
                rows,
                f"teacher_auxiliary_weights:{teacher.teacher_id}",
                teacher.auxiliary_weights,
            )
    return pd.DataFrame(rows, columns=["check", "status", "detail"])


def _path_check(rows: list[dict[str, str]], name: str, path: Path | None) -> None:
    _record(
        rows,
        name,
        "PASS" if path is not None and path.exists() else "FAIL",
        str(path),
    )


def _record(rows: list[dict[str, str]], name: str, status: str, detail: str) -> None:
    rows.append({"check": name, "status": status, "detail": detail})


def _conda_environments() -> set[str]:
    try:
        completed = subprocess.run(
            ["conda", "env", "list", "--json"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return set()
    import json

    return {Path(path).name for path in json.loads(completed.stdout).get("envs", [])}
