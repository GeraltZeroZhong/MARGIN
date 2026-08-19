"""Run optional GPU teachers in isolated Conda environments and import their scores."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from margin.config import ProjectConfig, TeacherSpec
from margin.provenance import canonical_json_hash, read_json, sha256_file, write_json
from margin.teachers.cache import ADAPTER_INPUT_KINDS, canonicalize_external_scores

RUNNERS = {
    "mifst": "models/run_mifst.py",
    "proteinmpnn": "models/run_proteinmpnn.py",
    "esm_if1_candidates": "models/run_esm_if1.py",
}


def run_external_teacher(
    teacher: TeacherSpec,
    request_table_path: Path,
    output_path: Path,
    config: ProjectConfig,
    *,
    device: str = "auto",
    limit: int | None = None,
) -> pd.DataFrame:
    """Execute one official model adapter, then return canonical normalized scores."""

    runner_name = RUNNERS.get(teacher.adapter)
    if runner_name is None:
        raise ValueError(f"teacher adapter {teacher.adapter!r} is not an external runner")
    if not teacher.conda_env:
        raise ValueError(f"teacher {teacher.teacher_id} must declare conda_env")
    runner = config.paths.project_root / "scripts" / runner_name
    cache_key = _raw_cache_key(
        teacher,
        request_table_path,
        runner,
        config,
        device=device,
        limit=limit,
    )
    command = _build_command(
        teacher,
        request_table_path,
        output_path,
        config,
        runner,
        device=device,
        limit=limit,
        run_key=cache_key,
    )
    raw = _load_reusable_raw_output(output_path, request_table_path, teacher, cache_key, limit)
    if raw is not None:
        requests = pd.read_parquet(request_table_path)
        return canonicalize_external_scores(raw, requests, teacher, config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    python_paths = [str(config.paths.project_root / "src")]
    if teacher.repository is not None:
        python_paths.append(str(teacher.repository))
    existing = environment.get("PYTHONPATH")
    if existing:
        python_paths.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    completed = subprocess.run(
        command,
        cwd=config.paths.project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    log_path = output_path.with_suffix(".log")
    log_path.write_text(
        f"COMMAND: {' '.join(command)}\n\nSTDOUT\n{completed.stdout}\n\nSTDERR\n{completed.stderr}",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"teacher {teacher.teacher_id} failed with exit code {completed.returncode}; "
            f"see {log_path}"
        )
    raw = pd.read_parquet(output_path)
    requests = pd.read_parquet(request_table_path)
    _validate_raw_coverage(raw, requests, teacher, limit)
    canonical = canonicalize_external_scores(raw, requests, teacher, config)
    _write_raw_cache_manifest(output_path, raw, cache_key)
    return canonical


def _build_command(
    teacher: TeacherSpec,
    request_table_path: Path,
    output_path: Path,
    config: ProjectConfig,
    runner: Path,
    *,
    device: str,
    limit: int | None,
    run_key: str,
) -> list[str]:
    command = [
        "conda",
        "run",
        "-n",
        teacher.conda_env,
        "python",
        str(runner),
        "--requests",
        str(request_table_path),
        "--output",
        str(output_path),
        "--model",
        teacher.model_name,
        "--device",
        device,
        "--seed",
        str(config.seed),
        "--run-key",
        run_key,
    ]
    if teacher.repository is not None:
        command.extend(["--repository", str(teacher.repository)])
    if teacher.weights is not None:
        command.extend(["--weights", str(teacher.weights)])
    if teacher.auxiliary_weights is not None:
        command.extend(["--auxiliary-weights", str(teacher.auxiliary_weights)])
    if teacher.adapter == "mifst":
        command.extend(["--batch-size", str(teacher.batch_size)])
    if teacher.adapter == "proteinmpnn":
        command.extend(["--order-repeats", str(teacher.order_repeats)])
    if limit is not None:
        command.extend(["--limit", str(limit)])
    return command


def _raw_cache_key(
    teacher: TeacherSpec,
    request_table_path: Path,
    runner: Path,
    config: ProjectConfig,
    *,
    device: str,
    limit: int | None,
) -> str:
    artifacts = []
    for role, path in (
        ("weights", teacher.weights),
        ("auxiliary_weights", teacher.auxiliary_weights),
    ):
        if path is not None:
            artifacts.append({"role": role, "sha256": sha256_file(path)})
    payload = {
        "schema_version": config.schema_version,
        "teacher": teacher.model_dump(mode="json"),
        "request_fingerprint": _request_fingerprint(request_table_path),
        "runner_sha256": sha256_file(runner),
        "model_artifacts": artifacts,
        "seed": config.seed,
        "device": device,
        "limit": limit,
    }
    return canonical_json_hash(payload)


def _request_fingerprint(request_table_path: Path) -> str:
    requests = pd.read_parquet(request_table_path)
    request_columns = [
        "request_id",
        "state_id",
        "domain_id",
        "state_sequence",
        "structure_role",
        "structure_id",
        "input_kind",
        "length",
    ]
    structures = pd.read_parquet(request_table_path.parent / "structures.parquet")
    structure_columns = [
        "structure_role",
        "structure_id",
        "target_domain_id",
        "input_kind",
        "sha256",
    ]
    payload = {
        "requests": requests.sort_values("request_id")[request_columns].to_dict(orient="records"),
        "structures": structures.sort_values(
            ["structure_role", "structure_id", "target_domain_id"]
        )[structure_columns].to_dict(orient="records"),
    }
    return canonical_json_hash(payload)


def _load_reusable_raw_output(
    output_path: Path,
    request_table_path: Path,
    teacher: TeacherSpec,
    cache_key: str,
    limit: int | None,
) -> pd.DataFrame | None:
    manifest_path = output_path.with_suffix(".manifest.json")
    if not output_path.exists() or not manifest_path.exists():
        return None
    manifest = read_json(manifest_path)
    if manifest.get("cache_key") != cache_key:
        return None
    output = manifest.get("output", {})
    if output.get("sha256") != sha256_file(output_path):
        return None
    raw = pd.read_parquet(output_path)
    if output.get("rows") != len(raw) or output.get("columns") != list(raw.columns):
        return None
    requests = pd.read_parquet(request_table_path)
    _validate_raw_coverage(raw, requests, teacher, limit)
    return raw


def _validate_raw_coverage(
    raw: pd.DataFrame,
    requests: pd.DataFrame,
    teacher: TeacherSpec,
    limit: int | None,
) -> None:
    supported = ADAPTER_INPUT_KINDS.get(teacher.adapter, set())
    expected = requests.loc[requests["input_kind"].isin(supported), ["request_id", "length"]]
    if limit is not None:
        expected = expected.head(limit)
    expected = expected.set_index("request_id")["length"].astype(int).sort_index()
    if raw.duplicated(["request_id", "position"]).any():
        raise ValueError(f"teacher {teacher.teacher_id} raw output contains duplicate positions")
    observed_ids = set(raw["request_id"].astype(str))
    if observed_ids != set(expected.index.astype(str)):
        raise ValueError(f"teacher {teacher.teacher_id} raw output has incomplete request coverage")
    positions = raw.assign(position=raw["position"].astype(int)).groupby("request_id")["position"]
    observed = pd.DataFrame(
        {"rows": positions.size(), "minimum": positions.min(), "maximum": positions.max()}
    ).sort_index()
    if not (
        observed["rows"].equals(expected)
        and observed["minimum"].eq(0).all()
        and observed["maximum"].equals(expected - 1)
    ):
        raise ValueError(
            f"teacher {teacher.teacher_id} raw output has incomplete position coverage"
        )


def _write_raw_cache_manifest(output_path: Path, raw: pd.DataFrame, cache_key: str) -> None:
    output: dict[str, Any] = {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "rows": int(len(raw)),
        "columns": list(raw.columns),
    }
    write_json(
        output_path.with_suffix(".manifest.json"),
        {"cache_key": cache_key, "output": output},
    )
