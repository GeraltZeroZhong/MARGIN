from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from margin.pipeline import run_foundation_audit


def test_complete_synthetic_pipeline_emits_all_audit_deliverables(tmp_path: Path) -> None:
    payload = yaml.safe_load(Path("configs/synthetic.yaml").read_text())
    payload["paths"]["project_root"] = str(Path.cwd())
    payload["paths"]["run_dir"] = str(tmp_path / "run")
    payload["audit"]["bootstrap_replicates"] = 20
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    result = run_foundation_audit(config_path)
    run = tmp_path / "run"
    assert result.decision.decision == "SYNTHETIC_ONLY"
    required = [
        run / "registry/manifest.json",
        run / "registry/leakage/leakage_manifest.json",
        run / "state_bank/manifest.json",
        run / "teacher_cache/manifest.json",
        run / "decoys/manifest.json",
        run / "audit/distillability_map.parquet",
        run / "audit/dms_coverage.parquet",
        run / "audit/paired_decoy_summary.parquet",
        run / "audit/on_policy_effect_summary.parquet",
        run / "reports/foundation_report.md",
        run / "audit/audit_result_table.parquet",
        run / "source_data/figure_1_distillability_map.csv",
        run / "figures/figure_1_distillability_map.pdf",
        run / "manifest.json",
    ]
    assert all(path.exists() and path.stat().st_size > 0 for path in required)
    criteria = pd.read_parquet(run / "audit/decision_criteria.parquet")
    assert set(criteria["status"]) <= {"PASS", "FAIL", "INCOMPLETE"}
    assert criteria["criterion"].nunique() == 9
    domains = pd.read_parquet(run / "registry/domains.parquet")
    assert domains.groupby("analysis_role")["domain_id"].nunique().to_dict() == {
        "external_benchmark": 3,
        "training_candidate": 5,
    }
    assert not domains.loc[
        domains["analysis_role"] == "external_benchmark", "eligible_for_training"
    ].any()
    dms_coverage = pd.read_parquet(run / "audit/dms_coverage.parquet")
    assert set(dms_coverage["analysis_role"]) == {"external_benchmark"}
    assert (dms_coverage["status"] == "complete").all()
    on_policy = pd.read_parquet(run / "audit/on_policy_effect_summary.parquet")
    assert set(on_policy["analysis_role"]) == {
        "external_benchmark",
        "training_candidate",
    }
