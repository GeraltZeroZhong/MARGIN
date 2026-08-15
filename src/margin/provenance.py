"""Reproducibility helpers used by registries, caches, and reports."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    """Hash a data artifact for the provenance and cache manifests required by the protocol."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    """Hash canonical JSON when the digest is used to validate cache compatibility."""

    payload = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision(project_root: Path) -> str:
    """Return the current commit, or ``uncommitted`` before the first repository commit."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "uncommitted"
    return completed.stdout.strip()


def runtime_manifest(project_root: Path) -> dict[str, Any]:
    """Capture the minimal runtime context needed to interpret an experiment."""

    return {
        "created_at": utc_now(),
        "code_revision": git_revision(project_root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def write_json(path: Path, value: Any) -> None:
    """Atomically write JSON so an interrupted run cannot leave a valid-looking partial manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    _atomic_write_text(path, payload)


def write_text(path: Path, payload: str) -> None:
    """Atomically write a UTF-8 text artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(path, payload)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_parquet(path: Path, table: pd.DataFrame) -> None:
    """Atomically write a typed table in the canonical foundation audit interchange format."""

    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        table.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(path: Path, table: pd.DataFrame) -> None:
    """Atomically write a human-readable source-data table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        table.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def table_manifest(path: Path, table: pd.DataFrame) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": int(len(table)),
        "columns": list(table.columns),
    }


def _atomic_write_text(path: Path, payload: str) -> None:
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if hasattr(value, "item"):
        return value.item()
    return value
