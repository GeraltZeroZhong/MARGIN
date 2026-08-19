#!/usr/bin/env python
"""Concurrently retry missing ESMFold structures before structure-sensitivity study preparation."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from margin.provenance import runtime_manifest, write_json
from margin.studies.external_validation.panel import load_external_validation_config
from margin.studies.structure_sensitivity.panel import (
    _fetch_prediction,
    load_structure_sensitivity_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/structure_sensitivity.yaml"),
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--finalize-current-cache", action="store_true")
    arguments = parser.parse_args()
    config = load_structure_sensitivity_config(arguments.protocol)
    cross = load_external_validation_config(config.paths.external_validation_protocol)
    domains = pd.read_parquet(cross.paths.run_dir / "panel/domains.parquet")
    root = config.paths.storage_dir / "structures"
    missing = [
        row
        for row in domains.sort_values("domain_id").itertuples(index=False)
        if not (root / "esmfold" / f"{row.uniprot_id}.pdb").exists()
    ]
    if arguments.finalize_current_cache:
        _write_audit(config, domains, root)
        return
    with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        futures = {
            executor.submit(
                _fetch_prediction,
                "esmfold",
                str(row.uniprot_id),
                str(row.sequence),
                root,
                config,
            ): row
            for row in missing
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                path = future.result()
                print(f"complete={row.domain_id} path={path}", flush=True)
            except (OSError, RuntimeError, ValueError) as error:
                print(f"failed={row.domain_id} reason={error}", flush=True)
    _write_audit(config, domains, root)


def _write_audit(config, domains: pd.DataFrame, root: Path) -> None:
    directory = root / "esmfold"
    available = [
        str(row.uniprot_id)
        for row in domains.itertuples(index=False)
        if (directory / f"{row.uniprot_id}.pdb").exists()
    ]
    unavailable = sorted(set(domains["uniprot_id"].astype(str)) - set(available))
    write_json(
        directory / "service_audit.json",
        {
            **runtime_manifest(config.paths.project_root),
            "status": "ESMFOLD_PUBLIC_API_RETRIES_COMPLETE",
            "available_accessions": sorted(available),
            "unavailable_accessions": unavailable,
            "available_count": len(available),
            "unavailable_count": len(unavailable),
            "scientific_threshold_changed": False,
        },
    )
    print(
        f"service_audit={directory / 'service_audit.json'} "
        f"available={len(available)} unavailable={len(unavailable)}"
    )


if __name__ == "__main__":
    main()
