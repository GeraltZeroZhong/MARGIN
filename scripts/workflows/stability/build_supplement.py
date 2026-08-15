#!/usr/bin/env python
"""Build the stability study post-lock tables, figure, and report."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.studies.stability.supplement import build_stability_supplement


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    arguments = parser.parse_args()
    paths = build_stability_supplement(arguments.project_root)
    print(f"report={paths['report']}")
    print(f"figure={paths['figure_pdf']}")


if __name__ == "__main__":
    main()
