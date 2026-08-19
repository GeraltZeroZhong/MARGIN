from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from margin.config import ProjectConfig, TeacherSpec
from margin.constants import AA_ALPHABET
from margin.teachers import external


def _request_tables(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    request_path = directory / "requests.parquet"
    pd.DataFrame(
        [
            {
                "request_id": "request-1",
                "state_id": "state-1",
                "domain_id": "domain-1",
                "state_sequence": "AC",
                "structure_role": "paired",
                "structure_id": "structure-1",
                "input_kind": "coordinates",
                "input_path": str(directory / "structure.npz"),
                "length": 2,
            }
        ]
    ).to_parquet(request_path, index=False)
    pd.DataFrame(
        [
            {
                "structure_role": "paired",
                "structure_id": "structure-1",
                "target_domain_id": "domain-1",
                "input_kind": "coordinates",
                "input_path": str(directory / "structure.npz"),
                "sha256": "structure-content-digest",
            }
        ]
    ).to_parquet(directory / "structures.parquet", index=False)
    return request_path


def _raw_scores() -> pd.DataFrame:
    rows = []
    for position in range(2):
        row = {
            "request_id": "request-1",
            "position": position,
            "conditioning": "full_sequence_backbone_conditional",
            "device": "cpu",
            "wall_seconds": 0.1,
            "forward_calls": 1,
        }
        row.update({f"score_{amino_acid}": 0.0 for amino_acid in AA_ALPHABET})
        rows.append(row)
    return pd.DataFrame(rows)


def test_external_teacher_reuses_exact_verified_raw_cache(
    tmp_path: Path,
    synthetic_config: ProjectConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_tables(tmp_path / "requests")
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"fixed-test-weights")
    teacher = TeacherSpec(
        teacher_id="cached_proteinmpnn",
        adapter="proteinmpnn",
        role="audit_structure",
        model_name="test-model",
        model_revision="test-revision",
        conda_env="test-env",
        weights=weights,
    )
    runner = synthetic_config.paths.project_root / "scripts" / external.RUNNERS[teacher.adapter]
    output = tmp_path / "raw.parquet"
    raw = _raw_scores()
    raw.to_parquet(output, index=False)
    cache_key = external._raw_cache_key(
        teacher,
        request_path,
        runner,
        synthetic_config,
        device="cpu",
        limit=None,
    )
    external._write_raw_cache_manifest(output, raw, cache_key)

    def unexpected_subprocess(*args: object, **kwargs: object) -> None:
        raise AssertionError("a compatible raw cache must not launch the teacher subprocess")

    monkeypatch.setattr(external.subprocess, "run", unexpected_subprocess)
    canonical = external.run_external_teacher(
        teacher,
        request_path,
        output,
        synthetic_config,
        device="cpu",
    )
    assert len(canonical) == 2
    assert canonical["teacher_id"].eq(teacher.teacher_id).all()


def test_raw_cache_coverage_rejects_missing_position(tmp_path: Path) -> None:
    request_path = _request_tables(tmp_path)
    requests = pd.read_parquet(request_path)
    teacher = TeacherSpec(
        teacher_id="proteinmpnn",
        adapter="proteinmpnn",
        role="audit_structure",
        model_name="test-model",
        model_revision="test-revision",
        conda_env="test-env",
    )
    with pytest.raises(ValueError, match="incomplete position coverage"):
        external._validate_raw_coverage(_raw_scores().iloc[[0]], requests, teacher, None)
