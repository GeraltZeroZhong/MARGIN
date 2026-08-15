#!/usr/bin/env python
"""Run the finite post-lock stability study submission audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.stability.publication_audit import (
    load_publication_audit_specification,
    run_publication_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--specification",
        type=Path,
        default=Path("configs/publication_audit.yaml"),
    )
    arguments = parser.parse_args()
    result = run_publication_audit(load_publication_audit_specification(arguments.specification))
    print("status=POSTLOCK_SUBMISSION_AUDIT_COMPLETE_NO_GATE_CHANGE")
    print(f"report={result['report']}")
    print(f"manifest={result['manifest']}")


if __name__ == "__main__":
    main()
