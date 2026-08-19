"""Frozen Hugging Face ESM2 policy with identity-safe per-position features."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from margin.config import ProjectConfig
from margin.constants import AA_ALPHABET
from margin.data_registry.registry import RegistryTables
from margin.provenance import (
    runtime_manifest,
    sha256_file,
    table_manifest,
    write_json,
    write_parquet,
)
from margin.state_sampling.bank import StateBank
from margin.state_sampling.policy import SequencePolicy


class Esm2SequencePolicy(SequencePolicy):
    """Use a frozen ESM2 masked LM as a state-conditioned sequence policy.

    Every queried position is replaced by the model mask token before either its
    candidate distribution or hidden representation is read. Existing ``X``
    corruption tokens elsewhere in the state are also represented as masks.
    """

    state_conditioned = True

    def __init__(
        self,
        *,
        model_path: Path,
        policy_id: str,
        model_revision: str,
        expected_weights_sha256: str,
        batch_size: int,
        device: str,
    ) -> None:
        import torch
        from transformers import AutoTokenizer, EsmForMaskedLM

        weights_path = model_path / "model.safetensors"
        if not weights_path.is_file():
            raise FileNotFoundError(f"frozen ESM2 weights are missing: {weights_path}")
        observed_sha256 = sha256_file(weights_path)
        if observed_sha256 != expected_weights_sha256:
            raise ValueError(
                "frozen ESM2 weight hash mismatch: "
                f"expected={expected_weights_sha256} observed={observed_sha256}"
            )
        resolved_device = "cuda:0" if device == "auto" and torch.cuda.is_available() else device
        if resolved_device == "auto":
            resolved_device = "cpu"
        self.policy_id = policy_id
        self.model_revision = model_revision
        self.model_path = model_path.resolve()
        self.checkpoint_sha256 = observed_sha256
        self.batch_size = batch_size
        self.device = torch.device(resolved_device)
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self._model = (
            EsmForMaskedLM.from_pretrained(
                model_path,
                local_files_only=True,
            )
            .eval()
            .to(self.device)
        )
        self.representation_dim = int(self._model.config.hidden_size)
        self._aa_token_ids = torch.tensor(
            self._tokenizer.convert_tokens_to_ids(list(AA_ALPHABET)),
            dtype=torch.long,
            device=self.device,
        )
        self._score_cache: dict[tuple[str, str], np.ndarray] = {}
        self._embedding_cache: dict[tuple[str, str], np.ndarray] = {}

    def log_probabilities(self, domain_id: str, tokens: Sequence[str]) -> np.ndarray:
        key = (domain_id, "".join(tokens))
        if key not in self._score_cache:
            self._score_state(key, tokens)
        return self._score_cache[key]

    def position_embeddings(self, domain_id: str, tokens: Sequence[str]) -> np.ndarray:
        """Return leave-one-position-out final-layer representations."""

        key = (domain_id, "".join(tokens))
        if key not in self._embedding_cache:
            self._score_state(key, tokens)
        return self._embedding_cache[key]

    def export_embeddings(
        self,
        bank: StateBank,
        output_path: Path,
        config: ProjectConfig,
    ) -> Path:
        """Persist one frozen feature vector for every canonical state-position."""

        feature_columns = [f"feature_esm2_{index:04d}" for index in range(self.representation_dim)]
        frames: list[pd.DataFrame] = []
        for state in bank.states.itertuples(index=False):
            values = self.position_embeddings(state.domain_id, tuple(state.state_sequence))
            frame = pd.DataFrame(values, columns=feature_columns)
            frame.insert(0, "position", np.arange(len(values), dtype=int))
            frame.insert(0, "domain_id", state.domain_id)
            frame.insert(0, "state_id", state.state_id)
            frames.append(frame)
        table = pd.concat(frames, ignore_index=True)
        if not np.isfinite(table[feature_columns].to_numpy(dtype=float)).all():
            raise ValueError("frozen ESM2 embeddings contain non-finite values")
        write_parquet(output_path, table)
        state_manifest = config.paths.state_bank_dir / "manifest.json"
        manifest: dict[str, Any] = {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "policy_id": self.policy_id,
            "model_revision": self.model_revision,
            "model_path": str(self.model_path),
            "weights_sha256": self.checkpoint_sha256,
            "conditioning": "strict_leave_one_position_out",
            "representation_layer": int(self._model.config.num_hidden_layers),
            "representation_dim": self.representation_dim,
            "state_bank_manifest": (
                {"path": str(state_manifest), "sha256": sha256_file(state_manifest)}
                if state_manifest.exists()
                else None
            ),
            "embeddings": table_manifest(output_path, table),
        }
        write_json(output_path.with_suffix(".manifest.json"), manifest)
        return output_path

    def _score_state(self, key: tuple[str, str], tokens: Sequence[str]) -> None:
        torch = self._torch
        length = len(tokens)
        log_probabilities = np.empty((length, len(AA_ALPHABET)), dtype=np.float32)
        embeddings = np.empty((length, self.representation_dim), dtype=np.float32)
        base_tokens = [self._tokenizer.mask_token if token == "X" else token for token in tokens]
        with torch.inference_mode():
            for start in range(0, length, self.batch_size):
                positions = list(range(start, min(length, start + self.batch_size)))
                masked_sequences: list[str] = []
                for position in positions:
                    masked = base_tokens.copy()
                    masked[position] = self._tokenizer.mask_token
                    masked_sequences.append("".join(masked))
                encoded = self._tokenizer(
                    masked_sequences,
                    add_special_tokens=True,
                    padding=True,
                    return_tensors="pt",
                )
                encoded = {name: value.to(self.device) for name, value in encoded.items()}
                output = self._model(
                    **encoded,
                    output_hidden_states=True,
                    return_dict=True,
                )
                batch_index = torch.arange(len(positions), device=self.device)
                token_index = torch.tensor(positions, device=self.device) + 1
                candidate_logits = output.logits[batch_index, token_index]
                candidate_logits = candidate_logits.index_select(-1, self._aa_token_ids)
                selected_logp = candidate_logits.log_softmax(dim=-1)
                selected_hidden = output.hidden_states[-1][batch_index, token_index]
                log_probabilities[positions] = selected_logp.float().cpu().numpy()
                embeddings[positions] = selected_hidden.float().cpu().numpy()
        self._score_cache[key] = log_probabilities
        self._embedding_cache[key] = embeddings


def create_policy(config: ProjectConfig, registry: RegistryTables) -> Esm2SequencePolicy:
    """Factory used by ``student_policy.factory`` in the real configuration."""

    del registry
    specification = config.student_policy
    if specification.model_path is None or specification.weights_sha256 is None:
        raise ValueError("ESM2 policy requires student_policy.model_path and weights_sha256")
    return Esm2SequencePolicy(
        model_path=specification.model_path,
        policy_id=specification.policy_id,
        model_revision=specification.model_revision,
        expected_weights_sha256=specification.weights_sha256,
        batch_size=specification.batch_size,
        device=specification.device,
    )
