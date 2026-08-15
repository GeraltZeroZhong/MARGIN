"""Small per-request checkpoints shared by isolated long-running teacher runners."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def part_directory(output: Path, run_key: str) -> Path:
    """Select a cache namespace tied to the external adapter compatibility key."""

    directory = output.parent / f"{output.stem}.parts" / run_key
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def completed_request_ids(directory: Path) -> set[str]:
    """Read request IDs from completed atomic part files."""

    completed: set[str] = set()
    for path in sorted(directory.glob("part-*.parquet")):
        values = pd.read_parquet(path, columns=["request_id"])["request_id"].astype(str)
        completed.update(values.unique())
    return completed


def write_request_part(directory: Path, ordinal: int, rows: list[dict[str, object]]) -> None:
    """Atomically checkpoint one fully scored teacher request."""

    path = directory / f"part-{ordinal:07d}.parquet"
    temporary = directory / f"part-{ordinal:07d}.tmp.parquet"
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    temporary.replace(path)


def finalize_parts(output: Path, directory: Path, expected_ids: list[str]) -> None:
    """Join checkpoints only after every declared request has complete output."""

    paths = sorted(directory.glob("part-*.parquet"))
    if not paths:
        raise ValueError("teacher runner produced no request parts")
    frames = [pd.read_parquet(path) for path in paths]
    table = pd.concat(frames, ignore_index=True)
    observed = set(table["request_id"].astype(str))
    if observed != set(expected_ids):
        missing = sorted(set(expected_ids) - observed)
        extra = sorted(observed - set(expected_ids))
        raise ValueError(
            f"teacher request checkpoints are incomplete: missing={missing[:3]} extra={extra[:3]}"
        )
    temporary = output.with_suffix(".tmp.parquet")
    table.to_parquet(temporary, index=False)
    temporary.replace(output)
