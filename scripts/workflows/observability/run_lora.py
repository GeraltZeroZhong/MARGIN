#!/usr/bin/env python
"""Train and evaluate the frozen-ESM2 LoRA residual probe on observability study."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from torch import nn
from transformers import AutoTokenizer, EsmForMaskedLM

from margin.provenance import runtime_manifest, table_manifest, write_json, write_parquet
from margin.studies.observability.config import load_observability_config
from margin.studies.observability.current import summarize_probe_rows
from margin.studies.observability.lora import adapter_state, inject_esm2_lora, load_adapter_state
from margin.studies.observability.probes import prediction_metrics, shuffled_target
from margin.studies.observability.targets import load_replication_residual_dataset


def main() -> None:
    arguments = parse_arguments()
    config = load_observability_config(arguments.config)
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        print(f"complete={manifest_path}")
        return
    dataset = load_replication_residual_dataset(config, arguments.replication_run)
    root = (
        arguments.replication_run.resolve()
        if arguments.replication_run is not None
        else config.paths.run_dir / "replication"
    )
    states = pd.read_parquet(root / "state_bank" / "states.parquet")
    sequence_by_state = states.set_index("state_id")["state_sequence"].astype(str).to_dict()
    train_examples = _sample_examples(
        dataset.metadata,
        "development_train",
        config.probes.lora_train_positions_per_state,
        config.seed + 501,
    )
    validation_examples = _sample_examples(
        dataset.metadata,
        "development_validation",
        config.probes.lora_validation_positions_per_state,
        config.seed + 502,
    )
    target_id = config.residual_targets.primary
    target = dataset.residuals[target_id]
    device = torch.device(arguments.device if arguments.device != "auto" else "cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(config.paths.esm2_model, local_files_only=True)
    model = EsmForMaskedLM.from_pretrained(config.paths.esm2_model, local_files_only=True).to(
        device
    )
    model.eval()
    adapters = inject_esm2_lora(
        model,
        config.probes.lora_target_layers,
        config.probes.lora_rank,
        config.probes.lora_alpha,
    )
    head = nn.Linear(int(model.config.hidden_size), target.shape[1]).to(device)
    parameters = [
        parameter
        for module in [*adapters.values(), head]
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(parameters, lr=config.probes.lora_learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    best_loss = float("inf")
    best_epoch = 0
    checkpoint_path = output / "best_adapter.pt"
    rng = np.random.default_rng(config.seed + 503)
    for epoch in range(1, config.probes.lora_epochs + 1):
        order = rng.permutation(len(train_examples))
        train_loss = _train_epoch(
            model,
            head,
            adapters,
            tokenizer,
            train_examples.iloc[order],
            sequence_by_state,
            target,
            optimizer,
            scaler,
            config.probes.lora_batch_size,
            device,
        )
        validation_prediction = _predict(
            model,
            head,
            tokenizer,
            validation_examples,
            sequence_by_state,
            config.probes.lora_batch_size,
            device,
        )
        validation_loss = mean_squared_error(
            target[validation_examples["row_index"].to_numpy(dtype=int)],
            validation_prediction,
        )
        history.append(
            {
                "epoch": epoch,
                "train_mse": train_loss,
                "validation_mse": float(validation_loss),
            }
        )
        write_parquet(output / "training_history.parquet", pd.DataFrame(history))
        print(
            f"epoch={epoch} train_mse={train_loss:.6f} validation_mse={validation_loss:.6f}",
            flush=True,
        )
        if validation_loss < best_loss:
            best_loss = float(validation_loss)
            best_epoch = epoch
            torch.save(
                {
                    "adapters": adapter_state(adapters),
                    "head": {
                        name: value.detach().cpu() for name, value in head.state_dict().items()
                    },
                    "epoch": epoch,
                    "validation_mse": best_loss,
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    load_adapter_state(adapters, checkpoint["adapters"])
    head.load_state_dict(checkpoint["head"])
    test_indices = np.flatnonzero(dataset.metadata["analysis_role"].eq("locked_test").to_numpy())
    test_examples = pd.DataFrame(
        {
            "row_index": test_indices,
            "state_id": dataset.metadata.iloc[test_indices]["state_id"].to_numpy(),
            "position": dataset.metadata.iloc[test_indices]["position"].to_numpy(dtype=int),
        }
    )
    test_prediction = _predict(
        model,
        head,
        tokenizer,
        test_examples,
        sequence_by_state,
        config.probes.lora_batch_size,
        device,
    )
    rows = prediction_metrics(
        dataset,
        target_id,
        test_indices,
        test_prediction,
        probe="esm2_lora",
        feature_kind="query",
        layer=min(config.probes.lora_target_layers),
        target_rank=config.probes.lora_rank,
        evaluation_split="locked_test",
        control="observed",
        repeat=0,
        moved_fraction=1.0,
    )
    summary_frames = []
    domain_frames = []
    environment_summary_frames = []
    environment_domain_frames = []

    def record(frame: pd.DataFrame) -> None:
        summary, domains = summarize_probe_rows(frame, config)
        summary_frames.append(summary)
        domain_frames.append(domains)
        for environment in config.candidate_environments:
            selected = frame.loc[
                frame["state_kind"].eq(environment.state_kind)
                & frame["requested_corruption_ratio"].eq(environment.requested_corruption_ratio)
                & frame[environment.axis].eq(environment.value)
            ]
            if len(selected) < config.inference.minimum_environment_rows:
                continue
            environment_summary, environment_domains = summarize_probe_rows(selected, config)
            for table in (environment_summary, environment_domains):
                table["environment_id"] = environment.environment_id
                table["environment_axis"] = environment.axis
                table["environment_value"] = environment.value
                table["environment_rows"] = len(selected)
                table["environment_domains"] = selected["domain_id"].nunique()
            environment_summary_frames.append(environment_summary)
            environment_domain_frames.append(environment_domains)

    record(rows)
    test_metadata = dataset.metadata.iloc[test_indices].reset_index(drop=True)
    local_test_indices = np.arange(len(test_indices), dtype=int)
    control_rng = np.random.default_rng(config.seed + 504)
    for control in config.probes.shuffle_controls:
        for repeat in range(config.probes.control_repeats):
            shuffled_prediction, moved = shuffled_target(
                test_metadata,
                test_prediction,
                local_test_indices,
                control,
                control_rng,
            )
            control_rows = prediction_metrics(
                dataset,
                target_id,
                test_indices,
                shuffled_prediction,
                probe="esm2_lora",
                feature_kind="query",
                layer=min(config.probes.lora_target_layers),
                target_rank=config.probes.lora_rank,
                evaluation_split="locked_test",
                control=f"prediction_{control}",
                repeat=repeat,
                moved_fraction=moved,
            )
            record(control_rows)
    summary = pd.concat(summary_frames, ignore_index=True)
    domains = pd.concat(domain_frames, ignore_index=True)
    artifacts = {
        "training_history": pd.DataFrame(history),
        "test_rows": rows,
        "summary": summary,
        "domain_estimates": domains,
    }
    if environment_summary_frames:
        artifacts["environment_summary"] = pd.concat(environment_summary_frames, ignore_index=True)
        artifacts["environment_domain_estimates"] = pd.concat(
            environment_domain_frames, ignore_index=True
        )
    paths = {name: output / f"{name}.parquet" for name in artifacts}
    for name, table in artifacts.items():
        write_parquet(paths[name], table)
    write_json(
        manifest_path,
        {
            **runtime_manifest(config.paths.project_root),
            "schema_version": config.schema_version,
            "probe": "frozen_esm2_lora",
            "conditioning": "strict_leave_one_position_out",
            "target": target_id,
            "rank": config.probes.lora_rank,
            "alpha": config.probes.lora_alpha,
            "target_layers": config.probes.lora_target_layers,
            "train_examples": len(train_examples),
            "validation_examples": len(validation_examples),
            "best_epoch": best_epoch,
            "best_validation_mse": best_loss,
            "checkpoint": {"path": str(checkpoint_path), "bytes": checkpoint_path.stat().st_size},
            "artifacts": [table_manifest(paths[name], table) for name, table in artifacts.items()],
        },
    )
    print(f"manifest={manifest_path}")


def _sample_examples(
    metadata: pd.DataFrame, role: str, positions_per_state: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    selected = []
    eligible = metadata.loc[metadata["analysis_role"].eq(role)].copy()
    eligible["row_index"] = eligible.index
    for _, frame in eligible.groupby("state_id", observed=True, sort=True):
        rows = frame["row_index"].to_numpy(dtype=int)
        count = min(positions_per_state, len(rows))
        selected.extend(np.sort(rng.choice(rows, size=count, replace=False)).tolist())
    indices = np.asarray(selected, dtype=int)
    return pd.DataFrame(
        {
            "row_index": indices,
            "state_id": metadata.iloc[indices]["state_id"].to_numpy(),
            "position": metadata.iloc[indices]["position"].to_numpy(dtype=int),
        }
    )


def _train_epoch(
    model,
    head,
    adapters,
    tokenizer,
    examples,
    sequence_by_state,
    target,
    optimizer,
    scaler,
    batch_size,
    device,
) -> float:
    model.eval()
    head.train()
    for adapter in adapters.values():
        adapter.train()
    total = 0.0
    count = 0
    for start in range(0, len(examples), batch_size):
        batch = examples.iloc[start : start + batch_size]
        optimizer.zero_grad(set_to_none=True)
        encoded, positions = _encode_batch(batch, sequence_by_state, tokenizer, device)
        expected = torch.as_tensor(
            target[batch["row_index"].to_numpy(dtype=int)],
            dtype=torch.float32,
            device=device,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            output = model.esm(**encoded, return_dict=True).last_hidden_state
            batch_index = torch.arange(len(batch), device=device)
            predicted = head(output[batch_index, positions])
            loss = torch.mean((predicted.float() - expected) ** 2)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [parameter for group in optimizer.param_groups for parameter in group["params"]], 1.0
        )
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach()) * len(batch)
        count += len(batch)
        if count % 1000 < batch_size:
            print(f"train_examples={count}/{len(examples)}", flush=True)
    return total / count


def _predict(
    model,
    head,
    tokenizer,
    examples,
    sequence_by_state,
    batch_size,
    device,
) -> np.ndarray:
    model.eval()
    head.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch = examples.iloc[start : start + batch_size]
            encoded, positions = _encode_batch(batch, sequence_by_state, tokenizer, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model.esm(**encoded, return_dict=True).last_hidden_state
                batch_index = torch.arange(len(batch), device=device)
                values = head(output[batch_index, positions])
            predictions.append(values.float().cpu().numpy())
    return np.concatenate(predictions, axis=0)


def _encode_batch(batch, sequence_by_state, tokenizer, device):
    sequences = []
    positions = batch["position"].to_numpy(dtype=int)
    for state_id, position in zip(batch["state_id"], positions, strict=True):
        tokens = [
            tokenizer.mask_token if token == "X" else token for token in sequence_by_state[state_id]
        ]
        tokens[position] = tokenizer.mask_token
        sequences.append("".join(tokens))
    encoded = tokenizer(sequences, add_special_tokens=True, padding=True, return_tensors="pt")
    encoded = {name: value.to(device) for name, value in encoded.items()}
    return encoded, torch.as_tensor(positions + 1, dtype=torch.long, device=device)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/observability.yaml"))
    parser.add_argument("--replication-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    main()
