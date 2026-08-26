#!/usr/bin/env python3
"""Apply the frozen MIXED-rule to completed Wave 1.1 D0 result tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from tools import stageR_execute_r0_wave1_1_d0_retention as d0


RESULTS = d0.RESULTS


def read_rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def report(path: Path, title: str, lines: list[str]) -> None:
    path.write_text(f"# {title}\n\n" + "\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    required = {
        "r0_d0_retention_probe_metrics_wave1_1.csv", "r0_d0_retention_family_summary_wave1_1.csv",
        "r0_d0_retention_decision_matrix_wave1_1.csv", "r0_wave1_1_hypothesis_results.json",
        "R0_Wave1_1_D0_Retention_Completion_Report_v1.md", "R0_Wave1_Cross_Module_Diagnosis_v1.1.md",
        "r0_wave1_1_execution_manifest.json", "r0_wave1_1_command_ledger.json",
        "r0_wave1_1_execution_completeness_assessment.json",
    }
    if not all((RESULTS / name).exists() for name in required):
        raise RuntimeError("Wave 1.1 D0 metric/report set is incomplete")
    finalization = RESULTS / "r0_wave1_1_decision_finalization_log.json"
    if finalization.exists():
        raise RuntimeError("Refusing to overwrite Wave 1.1 decision finalization")
    family_rows = read_rows("r0_d0_retention_family_summary_wave1_1.csv")
    pooling_status, pooling_reason = d0.formal_status(family_rows, "D0-C_SAME_HIDDEN_SEQUENCE_POOLING_STUDY")
    mask_status, mask_reason = d0.formal_status(family_rows, "D0-D_MASK_PADDING_SENSITIVITY")
    if pooling_status != "MIXED" or mask_status != "MIXED":
        raise RuntimeError(f"Unexpected finalization precondition: pooling={pooling_status}, mask={mask_status}")
    decision_rows: list[dict[str, object]] = []
    for hypothesis, study, status, reason in (
        ("D0_POOLING_EFFECT", "D0-C_SAME_HIDDEN_SEQUENCE_POOLING_STUDY", pooling_status, pooling_reason),
        ("D0_MASK_PADDING_SENSITIVITY", "D0-D_MASK_PADDING_SENSITIVITY", mask_status, mask_reason),
    ):
        local = [row for row in family_rows if row["study"] == study]
        decision_rows.append({
            "hypothesis_id": hypothesis, "study": study, "hypothesis_status": status, "decision_reason": reason,
            "geometry_shift_family_seed_rows": sum(row["family_interpretation"] == "COORDINATE_GEOMETRY_SHIFT_FAVORED" for row in local),
            "information_loss_family_seed_rows": sum(row["family_interpretation"] == "INFORMATION_RETENTION_LOSS_FAVORED" for row in local),
            "no_material_loss_family_seed_rows": sum(row["family_interpretation"] == "NO_MATERIAL_RETENTION_LOSS_SUPPORTED" for row in local),
            "mixed_family_seed_rows": sum(row["family_interpretation"] == "MIXED_OR_UNRESOLVED" for row in local),
            "frozen_primary_rule": "absolute paired standardized retention/readout difference >=0.10 with 95% CI excluding 0 and >=2/3 fixed-seed direction consistency; MIXED when predeclared strata materially conflict",
            "evidence_level": d0.wave1.EVIDENCE_LEVEL,
        })
    write_csv(RESULTS / "r0_d0_retention_decision_matrix_wave1_1.csv", decision_rows, list(decision_rows[0]))
    d3_note = d0.d3_direction_note()
    hypothesis = {
        "execution_status": "COMPLETE_EXECUTION_COMPLETENESS_CORRECTION",
        "evidence_level": d0.wave1.EVIDENCE_LEVEL,
        "supersedes": {"D0_POOLING_EFFECT": "Wave 1 result only", "D0_MASK_PADDING_SENSITIVITY": "Wave 1 result only"},
        "hypothesis_results": {"D0_POOLING_EFFECT": pooling_status, "D0_MASK_PADDING_SENSITIVITY": mask_status},
        "embedding_geometry_effect": "SUPPORTED",
        "semantic_retention_effect": "MIXED",
        "case_c": "CASE_C_TEMPORAL_CONTRIBUTION_MIXED_NOT_GENERALIZED",
        "D1_KNOWN_SEMANTIC_INFORMATION_PRESENT": "SUPPORTED_UNCHANGED_FROM_WAVE1",
        "D3_FORMAL_HYPOTHESES": "INCONCLUSIVE_UNCHANGED_FROM_WAVE1",
        "no_training_authorization_change": True,
        "no_d2_d4_or_new_planner_rollout": True,
    }
    write_json(RESULTS / "r0_wave1_1_hypothesis_results.json", hypothesis)
    by_study = {study: [row for row in family_rows if row["study"] == study] for study in d0.D0_VIEWS}
    report(RESULTS / "R0_Wave1_1_D0_Retention_Completion_Report_v1.md", "R0 Wave 1.1 D0 Retention Completion Report v1", [
        f"Evidence level: `{d0.wave1.EVIDENCE_LEVEL}`.", "",
        "Wave 1.1 completes the frozen D0 readout contract without altering Protocol v1.0. It uses only historical Stage7L rows, fixed A/B/C checkpoints and seeds 3407/3408/3409, the frozen nine CORE semantic targets, five-fold scenario-grouped splits, the frozen ridge/logistic grid, and 5,000 log-cluster bootstrap replicates.", "",
        f"- D0-C pooling formal status: `{pooling_status}` — {pooling_reason}.",
        f"- D0-D mask/padding formal status: `{mask_status}` — {mask_reason}.",
        f"- D0-C family-seed matrix counts: geometry shift={sum(row['family_interpretation'] == 'COORDINATE_GEOMETRY_SHIFT_FAVORED' for row in by_study['D0-C_SAME_HIDDEN_SEQUENCE_POOLING_STUDY'])}; information loss={sum(row['family_interpretation'] == 'INFORMATION_RETENTION_LOSS_FAVORED' for row in by_study['D0-C_SAME_HIDDEN_SEQUENCE_POOLING_STUDY'])}; no-material-loss={sum(row['family_interpretation'] == 'NO_MATERIAL_RETENTION_LOSS_SUPPORTED' for row in by_study['D0-C_SAME_HIDDEN_SEQUENCE_POOLING_STUDY'])}; mixed={sum(row['family_interpretation'] == 'MIXED_OR_UNRESOLVED' for row in by_study['D0-C_SAME_HIDDEN_SEQUENCE_POOLING_STUDY'])}.",
        f"- D0-D family-seed matrix counts: geometry shift={sum(row['family_interpretation'] == 'COORDINATE_GEOMETRY_SHIFT_FAVORED' for row in by_study['D0-D_MASK_PADDING_SENSITIVITY'])}; information loss={sum(row['family_interpretation'] == 'INFORMATION_RETENTION_LOSS_FAVORED' for row in by_study['D0-D_MASK_PADDING_SENSITIVITY'])}; no-material-loss={sum(row['family_interpretation'] == 'NO_MATERIAL_RETENTION_LOSS_SUPPORTED' for row in by_study['D0-D_MASK_PADDING_SENSITIVITY'])}; mixed={sum(row['family_interpretation'] == 'MIXED_OR_UNRESOLVED' for row in by_study['D0-D_MASK_PADDING_SENSITIVITY'])}.",
        "",
        "The seed-consistent `max × interaction` retention-loss cells are recorded, but they conflict with other prespecified pooling views/families that are geometry-only or show no material loss. The frozen decision table therefore makes the module-level D0-C conclusion `MIXED`, not `SUPPORTED`.",
        "",
        "`last` is the historical reference. `mean`, `max`, `final_valid`, and `masked_mean` are all `DIAGNOSTIC_NOT_HISTORICAL`; none rewrites historical Stage7L inference. The categorical lane-change proxy has only 18 negative rows across the 400 fixed rows, and three fixed scenario folds have no negative test row; its absolute BA is interpreted with that fixed-support limitation, while frozen/refit comparison uses the identical rows and folds.",
        "",
        "Wave 1 used embedding displacement alone to label D0 as SUPPORTED. That omission is logged as a protocol deviation affecting the old D0 primary conclusion, with this Wave 1.1 execution as its completeness correction. No training authorization is created.",
    ])
    report(RESULTS / "R0_Wave1_Cross_Module_Diagnosis_v1.1.md", "R0 Wave 1 Cross-Module Diagnosis v1.1", [
        f"Evidence level: `{d0.wave1.EVIDENCE_LEVEL}`.", "",
        f"D0 pooling: `{pooling_status}`; D0 mask/padding: `{mask_status}`. Embedding geometry sensitivity is `SUPPORTED`, whereas semantic retention/readout evidence is `MIXED`. Therefore `CASE_C_TEMPORAL_CONTRIBUTION_SUPPORTED` is not retained as a general conclusion; the corrected status is `CASE_C_TEMPORAL_CONTRIBUTION_MIXED_NOT_GENERALIZED`.", "",
        "D1 is preserved: `KNOWN_SEMANTIC_INFORMATION_PRESENT = SUPPORTED` from Wave 1. D3 formal hypotheses remain `INCONCLUSIVE`; no primary D3 result is changed.",
        f"Development-direction note only: in the existing pure-lateral table, R_linear_task has a higher ratio_to_null_q95 than R_full64 in {d3_note['linear_task_higher_count']}/{d3_note['representation_count']} representations and R_fixed_semantic in {d3_note['fixed_semantic_higher_count']}/{d3_note['representation_count']}. This descriptive direction does not upgrade any D3 status.", "",
        "Corrected scientific diagnosis: pooling geometry sensitivity is supported, and semantic retention loss appears in a predeclared subset but is not consistent across all predeclared D0 pooling strata. No D2/D4 execution, RBR training, or new planner rollout was performed.",
    ])
    finalization_value = {
        "finalized_at_utc": d0.now(), "basis": "frozen r0_decision_table_v1.0 MIXED rule for material conflict across predeclared strata",
        "prior_intermediate_status": {"D0_POOLING_EFFECT": "SUPPORTED", "D0_MASK_PADDING_SENSITIVITY": "MIXED", "case_c": "CASE_C_TEMPORAL_CONTRIBUTION_SUPPORTED"},
        "final_status": hypothesis["hypothesis_results"] | {"case_c": hypothesis["case_c"]},
        "thresholds_or_targets_changed": False, "outcomes_recomputed": False,
    }
    write_json(finalization, finalization_value)
    manifest_path = RESULTS / "r0_wave1_1_execution_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["hypothesis_results"] = hypothesis
    manifest["decision_finalization"] = finalization_value
    manifest["result_files"] = sorted(set(manifest["result_files"]) | {finalization.name})
    write_json(manifest_path, manifest)
    ledger_path = RESULTS / "r0_wave1_1_command_ledger.json"
    write_json(ledger_path, {
        "execution_id": "R0_WAVE1_1_D0_RETENTION_COMPLETION_V1", "command_id": "R0_WAVE1_1_D0_FINALIZE_002", "timestamp_utc": d0.now(),
        "command": "python -m tools.stageR_finalize_r0_wave1_1_d0_decision", "git_commit": d0.WAVE1_COMMIT,
        "input_artifact_sha256": {"family_summary": d0.sha256(RESULTS / "r0_d0_retention_family_summary_wave1_1.csv"), "probe_metrics": d0.sha256(RESULTS / "r0_d0_retention_probe_metrics_wave1_1.csv")},
        "output_artifact_sha256": {path.name: d0.sha256(path) for path in RESULTS.iterdir() if path.name.startswith("r0_wave1_1_") or path.name.startswith("R0_Wave1")},
        "exit_code": 0, "thresholds_or_targets_changed": False, "outcomes_recomputed": False,
    })
    print(json.dumps({"status": "R0_WAVE1_1_DECISION_FINALIZED", "pooling": pooling_status, "mask_padding": mask_status, "case_c": hypothesis["case_c"]}, indent=2))


if __name__ == "__main__":
    main()
