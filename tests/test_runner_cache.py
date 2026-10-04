from __future__ import annotations

import pandas as pd

from margin.teachers.runner_cache import (
    completed_request_ids,
    finalize_parts,
    part_directory,
    write_request_part,
)


def test_teacher_runner_parts_resume_and_finalize(tmp_path) -> None:
    output = tmp_path / "raw.parquet"
    directory = part_directory(output, "fixed-run")
    write_request_part(
        directory,
        0,
        [{"request_id": "r0", "position": 0, "score_A": 1.0}],
    )
    write_request_part(
        directory,
        1,
        [{"request_id": "r1", "position": 0, "score_A": 2.0}],
    )
    assert completed_request_ids(directory) == {"r0", "r1"}
    finalize_parts(output, directory, ["r0", "r1"])
    table = pd.read_parquet(output).sort_values("request_id", ignore_index=True)
    assert table["score_A"].tolist() == [1.0, 2.0]
