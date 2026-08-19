"""End-to-end foundation audit over the reusable scientific modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from margin.analysis.plots import make_audit_figures
from margin.analysis.report import write_foundation_report, write_run_index
from margin.attribution.decision import FoundationDecision, evaluate_foundation_decision
from margin.attribution.distillability import build_distillability_map
from margin.attribution.io import write_audit_bundle
from margin.attribution.observability import audit_observability, load_embeddings
from margin.attribution.on_policy import audit_on_policy
from margin.attribution.teacher_value import audit_teacher_value, load_dms_table
from margin.config import ProjectConfig, load_config
from margin.data_registry.conservation import attach_conservation, load_conservation
from margin.data_registry.leakage import (
    LeakageAudit,
    audit_benchmark_leakage,
    write_leakage_audit,
)
from margin.data_registry.registry import (
    RegistryTables,
    build_cath_registry,
    load_registry,
    registry_from_canonical_input,
    write_registry,
)
from margin.decoys.generate import build_decoys, write_decoys
from margin.fixtures import (
    build_synthetic_dms,
    build_synthetic_leakage_inputs,
    build_synthetic_registry,
)
from margin.provenance import (
    read_json,
    runtime_manifest,
    sha256_file,
    write_json,
    write_parquet,
    write_text,
)
from margin.state_sampling.bank import (
    StateBank,
    build_state_bank,
    load_state_bank,
    write_state_bank,
)
from margin.state_sampling.policy import (
    SequencePolicy,
    SyntheticSequencePolicy,
    load_imported_native_policy,
    load_policy_from_factory,
)
from margin.teachers.cache import (
    merge_score_tables,
    sequence_policy_scores,
    write_teacher_cache,
)
from margin.teachers.external import run_external_teacher
from margin.teachers.requests import export_teacher_requests
from margin.teachers.synthetic import build_synthetic_teacher_scores


@dataclass(frozen=True)
class AuditRun:
    config: ProjectConfig
    decision: FoundationDecision
    report_path: Path
    run_manifest_path: Path


def build_candidates_stage(config_path: Path) -> Path:
    """Build the quality-filtered candidate registry before homology search."""

    config = load_config(config_path)
    _prepare_run_directory(config, config_path)
    candidates, exclusions, input_files = _load_candidate_registry(config)
    directory = config.paths.registry_dir / "candidates"
    write_registry(directory, candidates, config, exclusions, input_files)
    return directory


def prepare_registry_stage(config_path: Path) -> tuple[RegistryTables, LeakageAudit]:
    """Build candidates, apply leakage exclusions, and add audit domains."""

    config = load_config(config_path)
    _prepare_run_directory(config, config_path)
    return _prepare_registry(config)


def build_state_bank_stage(config_path: Path, registry_directory: Path | None = None) -> Path:
    """Materialize states so frozen student embeddings can be exported."""

    config = load_config(config_path)
    _prepare_run_directory(config, config_path)
    registry = load_registry(registry_directory or config.paths.registry_dir)
    policy = _build_policy(config, registry)
    bank, skipped = build_state_bank(registry, policy, config)
    write_state_bank(config.paths.state_bank_dir, bank, policy, config, skipped)
    _export_policy_embeddings(policy, bank, config)
    return config.paths.state_bank_dir


def run_foundation_audit(config_path: Path, *, device: str = "auto") -> AuditRun:
    """Execute the foundation audit and return its guarded decision."""

    config = load_config(config_path)
    _prepare_run_directory(config, config_path)
    registry, leakage = _prepare_registry(config)
    bank = _load_reusable_state_bank(config, registry)
    if bank is None:
        policy = _build_policy(config, registry)
        bank, skipped_states = build_state_bank(registry, policy, config)
        write_state_bank(config.paths.state_bank_dir, bank, policy, config, skipped_states)
        _export_policy_embeddings(policy, bank, config)
        del policy
    decoys = build_decoys(
        registry,
        config,
        target_domain_ids=set(registry.domains["domain_id"]),
        source_domain_ids=set(
            registry.domains.loc[registry.domains["eligible_for_training"], "domain_id"]
        ),
    )
    write_decoys(config.paths.decoy_dir, decoys, config)
    requests = export_teacher_requests(
        config.paths.teacher_cache_dir / "requests", bank, registry, decoys, config
    )
    score_tables = _score_teachers(config, registry, bank, decoys, requests.requests, device)
    cache = merge_score_tables(score_tables)
    upstream = [
        config.paths.registry_dir / "manifest.json",
        config.paths.state_bank_dir / "manifest.json",
        config.paths.decoy_dir / "manifest.json",
        config.paths.teacher_cache_dir / "requests" / "manifest.json",
    ]
    write_teacher_cache(
        config.paths.teacher_cache_dir,
        cache,
        config,
        upstream,
        requests=requests.requests,
    )

    dms = _load_or_create_dms(config, registry, cache)
    embeddings = load_embeddings(config.paths.embeddings_input)
    teacher_audit = audit_teacher_value(cache, bank, config, dms)
    observability = audit_observability(cache, bank, registry, config, embeddings)
    on_policy = audit_on_policy(teacher_audit.position_metrics, bank, cache, config)
    distillability = build_distillability_map(teacher_audit, observability, bank, config)
    decision_result = evaluate_foundation_decision(
        teacher_audit, observability, on_policy, distillability, config
    )
    audit_manifest = write_audit_bundle(
        teacher_audit,
        observability,
        on_policy,
        distillability,
        decision_result,
        config,
        [*upstream, config.paths.teacher_cache_dir / "manifest.json"],
    )
    del audit_manifest
    figure_paths = make_audit_figures(config)
    report_path = write_foundation_report(
        decision_result,
        leakage,
        bank,
        decoys,
        cache,
        teacher_audit,
        observability,
        on_policy,
        config,
    )
    primary_artifacts = [
        config.paths.registry_dir / "manifest.json",
        config.paths.registry_dir / "leakage" / "leakage_manifest.json",
        config.paths.state_bank_dir / "manifest.json",
        config.paths.decoy_dir / "manifest.json",
        config.paths.teacher_cache_dir / "manifest.json",
        config.paths.audit_dir / "manifest.json",
        config.paths.audit_dir / "audit_result_table.parquet",
        report_path,
        *figure_paths,
    ]
    index_path = write_run_index(config, primary_artifacts)
    run_manifest_path = _write_run_manifest(
        config, decision_result, [*primary_artifacts, index_path]
    )
    return AuditRun(config, decision_result, report_path, run_manifest_path)


def _prepare_run_directory(config: ProjectConfig, config_path: Path) -> None:
    config.paths.run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    write_text(config.paths.run_dir / "config.resolved.yaml", snapshot)
    write_text(
        config.paths.run_dir / "config.original.yaml", config_path.read_text(encoding="utf-8")
    )


def _prepare_registry(config: ProjectConfig) -> tuple[RegistryTables, LeakageAudit]:
    candidates, preprocessing_exclusions, input_files = _load_candidate_registry(config)
    if config.data_mode == "synthetic":
        benchmarks, homology = build_synthetic_leakage_inputs(candidates)
        fixture_directory = config.paths.run_dir / "fixtures"
        write_parquet(fixture_directory / "benchmarks.parquet", benchmarks)
        write_parquet(fixture_directory / "homology_hits.parquet", homology)
        input_files.extend(
            [
                fixture_directory / "benchmarks.parquet",
                fixture_directory / "homology_hits.parquet",
            ]
        )
    else:
        benchmarks = _read_required_table(config.paths.benchmark_input, "benchmark_input")
        homology = _read_required_table(config.paths.homology_hits_input, "homology_hits_input")
        input_files.extend(
            path
            for path in [
                config.paths.benchmark_input,
                config.paths.homology_hits_input,
            ]
            if path is not None
        )
    write_registry(
        config.paths.registry_dir / "candidates",
        candidates,
        config,
        preprocessing_exclusions,
        input_files,
    )
    leakage = audit_benchmark_leakage(candidates.domains, benchmarks, homology, config)
    write_leakage_audit(config.paths.registry_dir / "leakage", leakage, config, input_files)
    eligible_ids = set(
        leakage.eligibility.loc[leakage.eligibility["eligible_for_training"], "domain_id"]
    )
    training_domains = candidates.domains.loc[
        candidates.domains["domain_id"].isin(eligible_ids)
    ].copy()
    training_domains["analysis_role"] = "training_candidate"
    training_domains["eligible_for_training"] = True
    training = RegistryTables(
        training_domains.reset_index(drop=True),
        candidates.residues.loc[candidates.residues["domain_id"].isin(eligible_ids)].reset_index(
            drop=True
        ),
    )
    if training.domains.empty:
        raise ValueError("leakage exclusions removed every candidate domain")
    if config.data_mode == "synthetic":
        audit_ids = set(benchmarks["domain_id"].dropna().astype(str))
        audit_domains = candidates.domains.loc[
            candidates.domains["domain_id"].isin(audit_ids)
        ].copy()
        audit_domains["dataset"] = "synthetic-control"
        audit_domains["analysis_role"] = "external_benchmark"
        audit_domains["eligible_for_training"] = False
        audit_registry = RegistryTables(
            audit_domains.reset_index(drop=True),
            candidates.residues.loc[candidates.residues["domain_id"].isin(audit_ids)].reset_index(
                drop=True
            ),
        )
    else:
        if config.paths.audit_domain_input is None:
            raise FileNotFoundError("real foundation audit requires paths.audit_domain_input")
        audit_registry = registry_from_canonical_input(config.paths.audit_domain_input)
        audit_manifest = config.paths.audit_domain_input / "manifest.json"
        if audit_manifest.exists():
            input_files.append(audit_manifest)
        if not (
            (audit_registry.domains["analysis_role"] == "external_benchmark").all()
            and (~audit_registry.domains["eligible_for_training"].astype(bool)).all()
        ):
            raise ValueError(
                "audit_domain_input must mark every domain external_benchmark and ineligible"
            )
        declared_audit_ids = set(benchmarks["domain_id"].dropna().astype(str))
        unknown_audit = set(audit_registry.domains["domain_id"]) - declared_audit_ids
        if unknown_audit:
            raise ValueError(
                f"audit registry domains lack benchmark metadata: {sorted(unknown_audit)[:5]}"
            )
    overlap = set(training.domains["domain_id"]) & set(audit_registry.domains["domain_id"])
    if overlap:
        raise ValueError(f"training and audit registries overlap: {sorted(overlap)[:5]}")
    registry = RegistryTables(
        pd.concat([training.domains, audit_registry.domains], ignore_index=True),
        pd.concat([training.residues, audit_registry.residues], ignore_index=True),
    )
    write_registry(config.paths.registry_dir, registry, config, input_files=input_files)
    return registry, leakage


def _load_candidate_registry(
    config: ProjectConfig,
) -> tuple[RegistryTables, pd.DataFrame, list[Path]]:
    input_files: list[Path] = []
    if config.data_mode == "synthetic":
        return (
            build_synthetic_registry(config),
            pd.DataFrame(columns=["domain_id", "stage", "reason"]),
            input_files,
        )
    if config.paths.domain_input is not None:
        candidates = registry_from_canonical_input(config.paths.domain_input)
        candidates, exclusions = _filter_canonical_candidates(candidates, config)
        canonical_manifest = config.paths.domain_input / "manifest.json"
        if canonical_manifest.exists():
            input_files.append(canonical_manifest)
    else:
        candidates, exclusions = build_cath_registry(config)
        input_files.extend(
            path
            for path in [config.paths.cath_domain_list, config.paths.cath_fasta]
            if path is not None
        )
    if config.paths.conservation_input is None:
        raise FileNotFoundError("real foundation audit requires paths.conservation_input")
    candidates = attach_conservation(
        candidates,
        load_conservation(config.paths.conservation_input),
        config,
    )
    input_files.append(config.paths.conservation_input)
    return candidates, exclusions, input_files


def _build_policy(config: ProjectConfig, registry: RegistryTables) -> SequencePolicy:
    specification = config.student_policy
    if specification.adapter == "synthetic":
        references = registry.domains.set_index("domain_id")["sequence"].to_dict()
        return SyntheticSequencePolicy(
            references,
            policy_id=specification.policy_id,
            model_revision=specification.model_revision,
        )
    if specification.adapter == "imported_native":
        assert specification.scores_input is not None
        return load_imported_native_policy(
            specification.scores_input,
            specification.policy_id,
            specification.model_revision,
        )
    assert specification.factory is not None
    policy = load_policy_from_factory(specification.factory, config=config, registry=registry)
    if policy.policy_id != specification.policy_id:
        raise ValueError(
            f"student policy ID mismatch: config={specification.policy_id}, "
            f"factory={policy.policy_id}"
        )
    if policy.model_revision != specification.model_revision:
        raise ValueError("student policy model_revision does not match the frozen configuration")
    if "on_policy_rollout" in config.state_bank.kinds and not policy.state_conditioned:
        raise ValueError("configured on-policy states require a state-conditioned student policy")
    return policy


def _export_policy_embeddings(
    policy: SequencePolicy,
    bank: Any,
    config: ProjectConfig,
) -> None:
    """Let a frozen policy export the same identity-safe features used for scoring."""

    exporter = getattr(policy, "export_embeddings", None)
    if callable(exporter) and config.paths.embeddings_input is not None:
        exporter(bank, config.paths.embeddings_input, config)


def _load_reusable_state_bank(
    config: ProjectConfig,
    registry: RegistryTables,
) -> StateBank | None:
    """Reuse a staged bank only when its policy, parameters, and residue inputs still match."""

    manifest_path = config.paths.state_bank_dir / "manifest.json"
    state_path = config.paths.state_bank_dir / "states.parquet"
    position_path = config.paths.state_bank_dir / "positions.parquet"
    embeddings_path = config.paths.embeddings_input
    if not (
        manifest_path.is_file()
        and state_path.is_file()
        and position_path.is_file()
        and embeddings_path is not None
        and embeddings_path.is_file()
    ):
        return None
    manifest = read_json(manifest_path)
    policy = manifest.get("policy", {})
    specification = config.student_policy
    if (
        manifest.get("data_mode") != config.data_mode
        or manifest.get("seed") != config.seed
        or manifest.get("parameters") != config.state_bank.model_dump(mode="json")
        or policy.get("policy_id") != specification.policy_id
        or policy.get("model_revision") != specification.model_revision
        or not policy.get("state_conditioned")
    ):
        return None
    bank = load_state_bank(config.paths.state_bank_dir)
    expected_sequences = registry.domains.set_index("domain_id")["sequence"].to_dict()
    observed_sequences = (
        bank.states[["domain_id", "reference_sequence"]]
        .drop_duplicates()
        .set_index("domain_id")["reference_sequence"]
        .to_dict()
    )
    if observed_sequences != expected_sequences:
        return None

    native_ids = set(bank.states.loc[bank.states["state_kind"] == "native_reference", "state_id"])
    observed = bank.positions.loc[bank.positions["state_id"].isin(native_ids)].sort_values(
        ["domain_id", "position"]
    )
    expected = registry.residues.sort_values(["domain_id", "position"])
    categorical_pairs = (
        ("native_aa", "residue"),
        ("burial", "burial"),
        ("secondary_structure", "secondary_structure"),
        ("contact_class", "contact_class"),
        ("conservation_class", "conservation_class"),
    )
    if len(observed) != len(expected) or any(
        not observed[left].reset_index(drop=True).equals(expected[right].reset_index(drop=True))
        for left, right in categorical_pairs
    ):
        return None
    for column in ("rsa", "contact_degree", "conservation_score"):
        if not np.allclose(
            observed[column].to_numpy(dtype=float),
            expected[column].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            return None
    return bank


def _filter_canonical_candidates(
    registry: RegistryTables, config: ProjectConfig
) -> tuple[RegistryTables, pd.DataFrame]:
    """Apply the configured CATH-pool quality limits to an imported registry."""

    excluded: list[dict[str, str]] = []
    keep: list[str] = []
    for domain in registry.domains.itertuples(index=False):
        reason: str | None = None
        if not config.registry.min_length <= int(domain.length) <= config.registry.max_length:
            reason = "length_out_of_range"
        elif (
            pd.notna(domain.resolution_angstrom)
            and float(domain.resolution_angstrom) > config.registry.max_resolution_angstrom
        ):
            reason = "resolution_above_limit"
        elif float(domain.missing_fraction) > config.registry.max_missing_fraction:
            reason = "too_many_missing_backbone_residues"
        elif config.registry.require_experimental_structure and not bool(domain.is_experimental):
            reason = "nonexperimental_structure"
        if reason is None:
            keep.append(str(domain.domain_id))
        else:
            excluded.append(
                {"domain_id": str(domain.domain_id), "stage": "metadata", "reason": reason}
            )
    keep_ids = set(keep)
    filtered = RegistryTables(
        registry.domains.loc[registry.domains["domain_id"].isin(keep_ids)].reset_index(drop=True),
        registry.residues.loc[registry.residues["domain_id"].isin(keep_ids)].reset_index(drop=True),
    )
    return filtered, pd.DataFrame(excluded, columns=["domain_id", "stage", "reason"])


def _score_teachers(
    config: ProjectConfig,
    registry: RegistryTables,
    bank: Any,
    decoys: Any,
    request_table: pd.DataFrame,
    device: str,
) -> list[pd.DataFrame]:
    if config.data_mode == "synthetic":
        return build_synthetic_teacher_scores(bank, registry, decoys, config)
    enabled = [teacher for teacher in config.teacher_cache.teachers if teacher.enabled]
    sequence = [teacher for teacher in enabled if teacher.role == "sequence"]
    if len(sequence) != 1:
        raise ValueError("real teacher matrix requires exactly one sequence teacher")
    tables = [sequence_policy_scores(bank, sequence[0], config)]
    request_path = config.paths.teacher_cache_dir / "requests" / "requests.parquet"
    del request_table
    for teacher in enabled:
        if teacher.role == "sequence":
            continue
        raw_output = config.paths.teacher_cache_dir / "raw" / f"{teacher.teacher_id}.parquet"
        tables.append(
            run_external_teacher(
                teacher,
                request_path,
                raw_output,
                config,
                device=device,
            )
        )
    return tables


def _load_or_create_dms(
    config: ProjectConfig,
    registry: RegistryTables,
    cache: Any,
) -> pd.DataFrame | None:
    if config.data_mode == "synthetic":
        dms = build_synthetic_dms(cache, registry, config)
        write_parquet(config.paths.run_dir / "fixtures" / "dms_variants.parquet", dms)
        return dms
    return load_dms_table(config.paths.dms_input)


def _read_required_table(path: Path | None, label: str) -> pd.DataFrame:
    if path is None or not path.exists():
        raise FileNotFoundError(f"real foundation audit requires {label}: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
    raise ValueError(f"{label} must be Parquet, CSV, or TSV")


def _write_run_manifest(
    config: ProjectConfig, decision_result: FoundationDecision, artifacts: list[Path]
) -> Path:
    path = config.paths.run_dir / "manifest.json"
    payload = {
        **runtime_manifest(config.paths.project_root),
        "schema_version": config.schema_version,
        "project": config.project_name,
        "data_mode": config.data_mode,
        "seed": config.seed,
        "decision": decision_result.decision,
        "config_snapshot": {
            "path": str(config.paths.run_dir / "config.resolved.yaml"),
            "sha256": sha256_file(config.paths.run_dir / "config.resolved.yaml"),
        },
        "artifacts": [
            {"path": str(artifact), "sha256": sha256_file(artifact)}
            for artifact in artifacts
            if artifact.exists() and artifact.is_file()
        ],
    }
    write_json(path, payload)
    return path
