#!/usr/bin/env python
"""Derive CATH ESM2 candidate log-probabilities from cached masked hidden states."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmForMaskedLM

from margin.constants import AA_ALPHABET
from margin.provenance import runtime_manifest, sha256_file, write_json
from margin.studies.generalization.config import load_generalization_config
from margin.studies.mechanisms.config import load_mechanism_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/mechanisms.yaml"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1024)
    arguments = parser.parse_args()
    config = load_mechanism_config(arguments.config)
    generalization = load_generalization_config(config.paths.generalization_config)
    source = generalization.paths.storage_dir / "architecture" / "esm2_150M"
    output = config.paths.storage_dir / "training_controls" / "esm2_150M_cath"
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "log_probabilities.npy"
    key_path = output / "keys.parquet"
    manifest_path = output / "manifest.json"
    if result_path.exists() and key_path.exists() and manifest_path.exists():
        print(f"complete={manifest_path}")
        return

    device = torch.device(arguments.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        generalization.paths.esm2_150m_model, local_files_only=True
    )
    model = (
        EsmForMaskedLM.from_pretrained(
            generalization.paths.esm2_150m_model,
            local_files_only=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        .eval()
        .to(device)
    )
    aa_indices = torch.tensor(tokenizer.convert_tokens_to_ids(list(AA_ALPHABET)), device=device)
    features = np.load(source / "representations.npy", mmap_mode="r")
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".log_probabilities.", suffix=".npy", dir=output
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        result = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.float16, shape=(len(features), len(AA_ALPHABET))
        )
        with torch.no_grad():
            for start in range(0, len(features), arguments.batch_size):
                stop = min(len(features), start + arguments.batch_size)
                hidden = torch.as_tensor(
                    np.asarray(features[start:stop], dtype=np.float32), device=device
                ).to(model.dtype)
                logits = model.lm_head(hidden).index_select(-1, aa_indices)
                result[start:stop] = logits.log_softmax(dim=-1).float().cpu().numpy()
        result.flush()
        del result
        os.replace(temporary, result_path)
    finally:
        temporary.unlink(missing_ok=True)
    keys = pd.read_parquet(source / "keys.parquet")
    keys.to_parquet(key_path, index=False)
    write_json(
        manifest_path,
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "derivation": "cached_strict_loo_final_hidden_state_through_pinned_esm2_lm_head",
            "mechanisms_stability_labels_used": False,
            "source_representations": {
                "path": str(source / "representations.npy"),
                "sha256": sha256_file(source / "representations.npy"),
            },
            "model": str(generalization.paths.esm2_150m_model),
            "rows": int(len(features)),
            "log_probabilities": {
                "path": str(result_path),
                "sha256": sha256_file(result_path),
            },
        },
    )
    print(f"complete={manifest_path}")


if __name__ == "__main__":
    main()
