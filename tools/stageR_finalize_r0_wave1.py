#!/usr/bin/env python3
"""Finalize Wave-1 reports from its already-written read-only metric tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from tools.stageR_execute_r0_wave1 import (
    EVIDENCE_LEVEL,
    RESULTS,
    SEEDS,
    checkpoint_sha_locks,
    git,
    markdown_report,
    now,
    sha256,
    verify_freeze,
    write_csv,
    write_json,
)


REQUIRED_PARTIAL = {
    "r0_wave1_environment.json", "r0_d1_family_results.csv", "r0_kernel_bandwidth_audit.csv",
    "r0_latent_geometry_metrics.csv", "r0_measurement_null_calibration.csv",
    "r0_measurement_readout_metrics.csv", "r0_projection_rank_selection.csv",
    "r0_semantic_probe_metrics.csv", "r0_temporal_content_window_descriptive_metrics.csv",
    "r0_temporal_contract_audit.csv", "r0_temporal_orthogonal_experiment_metrics.csv",
}
FINAL_PRODUCTS = {
    "r0_wave1_finalization_freeze_verification.json", "r0_wave1_execution_attempt_log.json",
    "r0_wave1_hypothesis_results.json", "R0_D0_Temporal_Decision_Report_v1.md",
    "R0_D1_Information_Geometry_Decision_Report_v1.md", "R0_D3_Measurement_Readout_Decision_Report_v1.md",
    "R0_Wave1_Cross_Module_Diagnosis_v1.md", "R0_Wave1_Training_Implication_Report_v1.md",
    "r0_wave1_protocol_deviation_log.csv", "r0_wave1_execution_manifest.json", "r0_wave1_command_ledger.json",
}


def rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_true(value: str | bool | None) -> bool:
    return value is True or (isinstance(value, str) and value.strip().lower() == "true")


def study_supported(source: list[dict[str, str]], study: str) -> bool:
    for candidate in ("A", "B", "C"):
        reps = {
            row["representation"]
            for row in source
            if row.get("study") == study
            and row.get("representation", "").startswith(candidate + "_")
            and as_true(row.get("gate_absolute_effect_ge_0_10"))
        }
        if len(reps) >= 2:
            return True
    return False


def main() -> None:
    present = {path.name for path in RESULTS.iterdir()} if RESULTS.exists() else set()
    if present != REQUIRED_PARTIAL:
        raise RuntimeError(f"Refusing finalization: expected exact partial metric set, got {sorted(present)}")
    if any((RESULTS / name).exists() for name in FINAL_PRODUCTS):
        raise RuntimeError("Refusing to overwrite an existing final product")
    verification = verify_freeze()
    locks = checkpoint_sha_locks()
    pooling_rows = rows("r0_temporal_orthogonal_experiment_metrics.csv")
    d0_rows = rows("r0_temporal_contract_audit.csv")
    family_rows = rows("r0_d1_family_results.csv")

    d0_pool = "SUPPORTED" if study_supported(pooling_rows, "D0-C_SAME_HIDDEN_SEQUENCE_POOLING_STUDY") else "NOT_SUPPORTED"
    d0_mask = "SUPPORTED" if study_supported(d0_rows, "D0-D_MASK_PADDING_SENSITIVITY") else "NOT_SUPPORTED"
    family = {(row["representation"], row["semantic_family"]): row["result"] for row in family_rows}
    expected = {(f"{candidate}_seed{seed}", domain) for candidate in ("A", "B", "C") for seed in SEEDS for domain in ("longitudinal", "lateral", "interaction")}
    if not expected.issubset(family):
        raise RuntimeError("D1 family table lacks a required candidate/seed/domain result")
    learned = {(candidate, domain): sum(family[(f"{candidate}_seed{seed}", domain)] == "SUPPORTED" for seed in SEEDS) >= 2 for candidate in ("A", "B", "C") for domain in ("longitudinal", "lateral", "interaction")}
    d1_module = "SUPPORTED" if sum(any(learned[(candidate, domain)] for candidate in ("A", "B", "C")) for domain in ("longitudinal", "lateral", "interaction")) >= 2 else "NOT_SUPPORTED"
    lateral = "SUPPORTED" if any(learned[(candidate, "lateral")] for candidate in ("A", "B", "C")) else "NOT_SUPPORTED"
    diagnosis = "CASE_C_NOT_ESTABLISHED" if d0_pool != "SUPPORTED" else ("CASE_C_TEMPORAL_CONTRIBUTION_SUPPORTED" if lateral == "SUPPORTED" else "CASE_A_REPRESENTATION_INFORMATION_LOSS_FAVORED")
    status = {
        "D0_LENGTH_EFFECT": "NOT_EVALUABLE", "D0_POSITION_RETENTION_ASSOCIATION": "NOT_EVALUABLE", "D0_POOLING_EFFECT": d0_pool, "D0_MASK_PADDING_SENSITIVITY": d0_mask,
        "D1_KNOWN_SEMANTIC_INFORMATION_PRESENT": d1_module, "D1_CROSS_DOMAIN_SEMANTIC_TRANSFER": "NOT_EVALUABLE", "D1_GEOMETRY_DEGENERACY": "INCONCLUSIVE",
        "D3_FULL64_SIGNAL_DILUTION": "INCONCLUSIVE", "D3_PROJECTED_READOUT_GAIN": "INCONCLUSIVE", "D3_NULL_CALIBRATION_PRESERVED": "INCONCLUSIVE",
    }

    write_json(RESULTS / "r0_wave1_finalization_freeze_verification.json", verification)
    write_json(RESULTS / "r0_wave1_execution_attempt_log.json", {
        "execution_id": "R0_WAVE1_D0_D1_D3_V1", "status": "METRICS_COMPLETE_REPORTING_FINALIZED",
        "attempts": [
            {"attempt": 1, "result": "STOPPED_BEFORE_MODEL_LOAD", "reason": "local PyTorch 1.9 does not accept the later weights_only load argument"},
            {"attempt": 2, "result": "METRICS_COMPLETE", "reason": "report aggregation key error after all metric tables had been written"},
            {"attempt": 3, "result": "REPORTING_FINALIZATION", "reason": "reports derived read-only from attempt-2 metric tables"},
        ],
        "scientific_protocol_deviation": False, "metric_tables_overwritten": False, "embeddings_recomputed": False,
    })
    write_json(RESULTS / "r0_wave1_hypothesis_results.json", {
        "execution_status": "COMPLETE_WITH_EXPLICIT_NOT_EVALUABLE_AND_INCONCLUSIVE_RESULTS", "evidence_level": EVIDENCE_LEVEL,
        "hypothesis_results": status,
        "limitations": ["R0_AUDIT_HOLDOUT=NOT_AVAILABLE", "D0-A controlled content equivalence unavailable", "D0-B frozen matched-natural ledger unavailable", "D3 independent null calibration series unavailable", "D3 non-lateral domain contrasts have no frozen Wave-1 readout contract"],
        "next_action": "Do not alter frozen protocol; address only under a separately authorized future wave.",
    })
    markdown_report(RESULTS / "R0_D0_Temporal_Decision_Report_v1.md", "R0 D0 Temporal Decision Report v1", f"Evidence level: `{EVIDENCE_LEVEL}`.\n\n- D0-A: `NOT_EVALUABLE`; no same-event-content controlled T80/T150 construction is available.\n- D0-B: `NOT_EVALUABLE`; the required frozen matched-natural position ledger is unavailable.\n- D0-C same-hidden pooling: `{d0_pool}` from the fixed last/mean/max comparisons across A/B/C seeds.\n- D0-D mask/padding diagnostic: `{d0_mask}`; all altered views are `DIAGNOSTIC_NOT_HISTORICAL`.\n\nHistorical reference remains T150 + final hidden + original mask/padding behavior. Content-window rows are descriptive only.")
    markdown_report(RESULTS / "R0_D1_Information_Geometry_Decision_Report_v1.md", "R0 D1 Information & Geometry Decision Report v1", f"Evidence level: `{EVIDENCE_LEVEL}`.\n\nD1 known semantic information result: `{d1_module}`. The nine frozen CORE targets used five-fold scenario-grouped held-out linear probes and 5,000 scenario-cluster bootstrap replicates. Geometry is `INCONCLUSIVE` because no frozen numerical geometry-degeneracy gate exists; it was not used alone to determine semantic support. See `r0_d1_family_results.csv` and the target-level table.")
    markdown_report(RESULTS / "R0_D3_Measurement_Readout_Decision_Report_v1.md", "R0 D3 Measurement Readout Decision Report v1", "Evidence level: `DEVELOPMENT_DIAGNOSTIC_EVIDENCE`.\n\nPure-lateral historical development comparisons use the frozen RBF kernel and treatment-label-blind Waymo reference bandwidth, with 49,999 same-scenario pair-label swaps. `D3_PROJECTED_READOUT_GAIN`, `D3_FULL64_SIGNAL_DILUTION`, and `D3_NULL_CALIBRATION_PRESERVED` are `INCONCLUSIVE`: R0_AUDIT_HOLDOUT is unavailable and there is no independent null calibration series. Longitudinal, following, and interaction Wave-1 contrasts are explicitly `NOT_EVALUABLE`, not substituted with new planner rollouts or outcome-selected assets.")
    markdown_report(RESULTS / "R0_Wave1_Cross_Module_Diagnosis_v1.md", "R0 Wave1 Cross-Module Diagnosis v1", f"Evidence level: `{EVIDENCE_LEVEL}`.\n\nD1 lateral semantic result: `{lateral}`. D0 pooling result: `{d0_pool}`. D3 results remain `INCONCLUSIVE` because the audit holdout and independent null-calibration series are unavailable. Cross-module diagnosis: `{diagnosis}`. This is not a unique causal conclusion; multiple mechanisms may be supported and no post-hoc threshold was introduced.")
    markdown_report(RESULTS / "R0_Wave1_Training_Implication_Report_v1.md", "R0 Wave1 Training Implication Report v1", "RBR-A/B/C remain `NOT_AUTHORIZED`. Wave 1 does not modify the frozen training authorization manifest. Even after these development diagnostics, R0 scientific decision records for all required modules, candidate-specific activation gates, and exact R4 source acquisition choice remain incomplete.")
    write_csv(RESULTS / "r0_wave1_protocol_deviation_log.csv", [], ["deviation_id", "timestamp_utc", "detected_after_outcome_access", "description", "affected_protocol_section", "affects_primary", "evidence_downgrade", "mitigation", "scientific_owner_disposition", "closed_timestamp_utc"])
    metric_shas = {path.name: sha256(path) for path in RESULTS.iterdir() if path.is_file()}
    write_json(RESULTS / "r0_wave1_execution_manifest.json", {"execution_id": "R0_WAVE1_D0_D1_D3_V1", "completed_at_utc": now(), "freeze_verification": verification, "checkpoint_sha256": locks, "input_metric_table_sha256": metric_shas, "hypothesis_results": status, "protocol_deviation_count": 0, "evidence_level": EVIDENCE_LEVEL, "training_authorization_modified": False, "embeddings_committed": False})
    write_json(RESULTS / "r0_wave1_command_ledger.json", {"execution_id": "R0_WAVE1_D0_D1_D3_V1", "command_id": "R0_WAVE1_FINALIZE_003", "timestamp_utc": now(), "operator": "Codex", "command": "tools/stageR_finalize_r0_wave1.py", "git_commit": git("rev-parse", "HEAD"), "input_artifact_sha256": locks, "output_artifact_sha256": {path.name: sha256(path) for path in RESULTS.iterdir() if path.is_file()}, "exit_code": 0, "seed": 2026082601, "environment_record_id": "r0_wave1_environment.json", "protocol_deviation_id": "NONE"})
    print(json.dumps({"status": "R0_WAVE1_COMPLETE", "hypothesis_results": status, "result_dir": str(RESULTS)}, indent=2))


if __name__ == "__main__":
    main()
