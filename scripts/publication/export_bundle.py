#!/usr/bin/env python
"""Export curated figures and source data for repository publication."""

from __future__ import annotations

import argparse
from pathlib import Path

from margin.publication import export_publication_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("publication"))
    parser.add_argument("--frozen-run-root", type=Path)
    arguments = parser.parse_args()
    manifest, dictionary = export_publication_bundle(
        arguments.project_root,
        arguments.output,
        frozen_run_root=arguments.frozen_run_root,
    )
    print(f"manifest={manifest}")
    print(f"data_dictionary={dictionary}")


if __name__ == "__main__":
    main()
