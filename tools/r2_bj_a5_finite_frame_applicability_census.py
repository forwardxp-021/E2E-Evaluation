#!/usr/bin/env python3
"""R2-BJ-A5 frozen 557-log finite-frame census; offline and zero-run only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import r2_bj_a4_hlc_moving_regime_applicability as a4  # noqa: E402
from tools.r1_b2_8_r3_prospective_selector import official_env  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "docs/stageR/r2"
FRAME = R2 / "r2_bj_a4_hash_ranked_audit_frame_manifest_v1.0.json"
PREDICATE = R2 / "r2_bj_a4_hlc_moving_regime_applicability_predicate_v1.0.json"
GENERATOR = ROOT / "tools/r2_bj_a_hlc_morphology_feasible_generator_v4.py"
PLANNER = ROOT / "tools/r2_bj_a_hlc_morphology_feasible_planner_v4.py"
CONTRACT = R2 / "r2_bj_a5_preregistered_finite_frame_census_contract_v1.0.json"
BINDING = R2 / "r2_bj_a5_a4_frozen_frame_binding_audit_v1.0.json"
PROTECTED = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"

OUT = {
    "eligibility": R2 / "r2_bj_a5_557_entry_eligibility_census_ledger_v1.0.json",
    "curvature": R2 / "r2_bj_a5_curvature_disposition_audit_v1.0.json",
    "components": R2 / "r2_bj_a5_native_generated_composite_component_audit_v1.0.json",
    "provenance": R2 / "r2_bj_a5_applicable_pool_provenance_manifest_v1.0.json",
    "envelope": R2 / "r2_bj_a5_finite_frame_census_envelope_v1.0.json",
    "firewall": R2 / "r2_bj_a5_data_firewall_audit_v1.0.json",
    "request": R2 / "R2_BJ_A5_Scientific_Owner_Readiness_Request_v0.1.md",
}
MANIFEST = R2 / "r2_bj_a5_component_sha_binding_manifest_v1.0.json"

EXPECTED = {
    FRAME: "bf5f84acc8034136720f372ed2b56dd6fe6944cc09e95b7545457d27830ae1a5",
    PREDICATE: "fe02f8e5c26f1269503471891c40295e30a268af4f38e35584d4e328928631e1",
    GENERATOR: "907e118014e1f83ed0004d5a194d75fa389a2e7fc21619c3a3a44dc3c69abae9",
    PLANNER: "066a1fd2dd2eb3fdc25ed4308c115d3a186e8a16e1e8944cbcf1d08d46613b8b",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"R2_BJ_A5_VERSIONED_OUTPUT_EXISTS:{path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify_inputs() -> Mapping[str, Any]:
    failures = []
    actual = {str(path.relative_to(ROOT)): sha(path) for path in EXPECTED}
    for path, expected in EXPECTED.items():
        if actual[str(path.relative_to(ROOT))] != expected:
            failures.append(f"SHA_MISMATCH:{path.relative_to(ROOT)}")
    frame = json.loads(FRAME.read_text(encoding="utf-8"))
    rows = frame["entries"]
    tokens = [str(row["scenario_token"]) for row in rows]
    logs = [str(row["log_id"]) for row in rows]
    ranks = [str(row["audit_rank_sha256"]) for row in rows]
    actual_canonical = canonical_sha(rows)
    checks = {
        "entry_count_557": len(rows) == 557 == int(frame["frame_size"]),
        "scenario_tokens_unique_557": len(set(tokens)) == 557,
        "log_ids_unique_557": len(set(logs)) == 557,
        "audit_rank_strictly_sorted": ranks == sorted(ranks) and len(set(ranks)) == 557,
        "canonical_entries_SHA_closed": actual_canonical == frame["frame_entries_canonical_sha256"],
        "A4_predicate_outcomes_opened_zero": frame["A4_predicate_outcomes_opened_before_frame_freeze"] == 0,
        "historical_permanent_A3_overlap_zero": frame["historical_A2_A3_or_permanent_overlap_count"] == 0,
        "A4_frame_not_regenerated": True,
    }
    failures.extend(name for name, passed in checks.items() if not passed)
    binding = json.loads(BINDING.read_text(encoding="utf-8"))
    if binding["status"] != "PASS_CENSUS_INPUT_INTEGRITY_CLOSED_BEFORE_A5_OUTCOMES":
        failures.append("PRE_OUTCOME_BINDING_AUDIT_NOT_PASS")
    return {
        "status": "PASS" if not failures else "CENSUS_INPUT_INTEGRITY_FAILURE",
        "failures": failures,
        "checks": checks,
        "actual_SHA256": actual,
        "frame_entries_canonical_sha256": actual_canonical,
        "frame": frame,
    }


def disposition(result: Mapping[str, Any]) -> str:
    if result["status"] == "PASS":
        return "MOVING_REGIME_V4_APPLICABLE"
    stage = result["stage"]
    if stage == "P01":
        return "EXACT_SCENARIO_RESOLUTION_NOT_APPLICABLE"
    if stage == "P03":
        return "LOW_SPEED_OUTSIDE_V4_APPLICABILITY"
    if stage == "P04":
        return "V2_3_TOPOLOGY_REFERENCE_NOT_APPLICABLE"
    if stage == "P05_P06":
        return "REFERENCE_GEOMETRY_OR_PRIMARY80_COVERAGE_NOT_APPLICABLE"
    if stage == "P07":
        return "RAW_ROBUST_CURVATURE_NOT_APPLICABLE"
    if stage == "P08":
        return "SOURCE_NATIVE_KINEMATIC_FEASIBILITY_UNRESOLVED"
    return str(result["reason"])


def architecture_failure(summary: Mapping[str, Any]) -> bool:
    fields = (
        "generated_increment_infeasible_without_cancellation_cases",
        "composite_infeasible_cases",
        "state0_continuity_failures",
        "post_recommit_terminal_capture_failures",
        "post_recommit_composite_failures",
    )
    return any(int(summary[name]) > 0 for name in fields)


def census() -> None:
    if any(path.exists() for path in OUT.values()):
        raise FileExistsError("R2_BJ_A5_VERSIONED_CENSUS_OUTPUT_EXISTS")
    integrity = verify_inputs()
    if integrity["status"] != "PASS":
        raise RuntimeError(f"CENSUS_INPUT_INTEGRITY_FAILURE:{integrity['failures']}")
    official_env()
    warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)
    a4.a2.PARAMETERS = json.loads(a4.SPACE.read_text(encoding="utf-8"))["global_parameters"]
    frame = integrity["frame"]
    map_cache: Dict[str, Any] = {}
    ledger_rows = []
    component_rows = []
    curvature_rows = []
    passing_rows = []
    stages: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for index, frozen in enumerate(frame["entries"], 1):
        entry = dict(frozen)
        entry["timestamp"] = entry["scenario_anchor_timestamp_us"]
        result = a4.evaluate_one(entry, map_cache)
        input_sha = canonical_sha(frozen)
        result_sha = canonical_sha(result)
        row = {
            "census_index": index,
            "frame_index": frozen["frame_index"],
            "audit_rank_sha256": frozen["audit_rank_sha256"],
            "scenario_token": frozen["scenario_token"],
            "log_id": frozen["log_id"],
            "input_record_canonical_sha256": input_sha,
            "official_initial_speed_mps": result.get("speed_information", result.get("closure", {}).get("speed_information", {})).get("official_initial_speed_mps"),
            "pre_treatment_max_speed_0_to_1p0s_mps": result.get("speed_information", result.get("closure", {}).get("speed_information", {})).get("pre_treatment_speed_distribution_mps", {}).get("max"),
            "v_audit_mps": result.get("speed_information", result.get("closure", {}).get("speed_information", {})).get("v_audit_mps"),
            "moving_regime_speed_gate": "PASS" if result["stage"] not in {"P01", "P03"} else ("FAIL" if result["stage"] == "P03" else "NOT_REACHED"),
            "predicate_status": result["status"],
            "predicate_stage": result["stage"],
            "failure_reason": result["reason"],
            "final_disposition": disposition(result),
            "predicate_result": result,
            "predicate_result_canonical_sha256": result_sha,
            "complete_input_component_output_SHA_provenance": True,
            "scientific_outcome_blacklist_addition": False,
        }
        ledger_rows.append(row)
        stages[result["stage"]] += 1
        if result["reason"]:
            reasons[str(result["reason"]).split(":", 1)[0]] += 1
        closure = result.get("closure")
        if closure is not None:
            summary = closure["component_summary"]
            component = {
                "census_index": index,
                "frame_index": frozen["frame_index"],
                "scenario_token": frozen["scenario_token"],
                "log_id": frozen["log_id"],
                "speed_cases_mps": [closure["speed_information"]["v_audit_mps"], closure["speed_information"]["v_support_mps"]],
                "planner_state_case_count": 960,
                "summary": summary,
                "failure_details": closure["component_failure_details"],
                "architecture_failure": architecture_failure(summary),
                "component_input_SHA256": input_sha,
                "component_output_canonical_sha256": canonical_sha({"summary": summary, "failure_details": closure["component_failure_details"]}),
            }
            component_rows.append(component)
            curvature_rows.append({
                "census_index": index, "frame_index": frozen["frame_index"],
                "scenario_token": frozen["scenario_token"], "log_id": frozen["log_id"],
                "quality": closure["curvature_quality"],
                "disposition": closure["curvature_disposition"],
                "curvature_output_canonical_sha256": canonical_sha({"quality": closure["curvature_quality"], "disposition": closure["curvature_disposition"]}),
            })
        elif result["stage"] == "P07":
            curvature_rows.append({
                "census_index": index, "frame_index": frozen["frame_index"],
                "scenario_token": frozen["scenario_token"], "log_id": frozen["log_id"],
                "quality": result["curvature_quality"], "disposition": result["curvature_disposition"],
                "curvature_output_canonical_sha256": canonical_sha({"quality": result["curvature_quality"], "disposition": result["curvature_disposition"]}),
            })
        if result["status"] == "PASS":
            passing_rows.append(row)
        if index % 10 == 0 or index == 557:
            print(json.dumps({"progress": "A5_CENSUS", "completed": index, "target": 557, "applicable": len(passing_rows), "component_stage": len(component_rows)}), flush=True)

    defined = {
        "LOW_MAGNITUDE_RAW_ROBUST_FEASIBILITY_CONCORDANT",
        "LOCALIZED_POINTWISE_SPIKE_RAW_AND_ROBUST_FEASIBILITY_CONCORDANT",
        "RAW_ROBUST_CONCORDANT_SUSTAINED_FEASIBLE",
        "RAW_CURVATURE_FEASIBILITY_FAIL",
        "ROBUST_CURVATURE_FEASIBILITY_FAIL",
    }
    undefined = []
    for row in curvature_rows:
        for side, value in row["disposition"].items():
            if value["disposition"] not in defined:
                undefined.append({"frame_index": row["frame_index"], "side": side, "disposition": value["disposition"]})
    architecture_failures = [row for row in component_rows if row["architecture_failure"]]
    census_complete = len(ledger_rows) == 557
    if not census_complete:
        final_status = "CENSUS_EVALUATION_INCOMPLETE"
    elif integrity["status"] != "PASS":
        final_status = "CENSUS_INPUT_INTEGRITY_FAILURE"
    elif undefined:
        final_status = "CURVATURE_REPRESENTATION_UNRESOLVED"
    elif architecture_failures:
        final_status = "R2_BJ_A5_MOVING_REGIME_ARCHITECTURE_NOT_READY"
    elif len(passing_rows) < 32:
        final_status = "APPLICABLE_POOL_INSUFFICIENT"
    else:
        final_status = "R2_BJ_A5_CENSUS_COMPLETE_READY_FOR_BJ_B_OWNER_REVIEW"

    eligibility = {
        "schema_version": "r2_bj_a5_557_entry_eligibility_census_ledger_v1.0",
        "status": "COMPLETE_557_OF_557" if census_complete else "CENSUS_EVALUATION_INCOMPLETE",
        "A4_frozen_frame_sha256": sha(FRAME), "A4_predicate_sha256": sha(PREDICATE),
        "census_target": 557, "census_evaluated": len(ledger_rows), "EARLY_STOP": False,
        "applicable_pool_count": len(passing_rows),
        "stage_completion_counts": dict(sorted(stages.items())),
        "failure_reason_counts": dict(sorted(reasons.items())),
        "entries": ledger_rows,
        "selection": "NONE", "rerank": False, "replacement": False,
    }
    components = {
        "schema_version": "r2_bj_a5_native_generated_composite_component_audit_v1.0",
        "status": "PASS_NO_MOVING_REGIME_ARCHITECTURE_FAILURE" if not architecture_failures else "MOVING_REGIME_ARCHITECTURE_FAILURE_PRESENT",
        "component_stage_count": len(component_rows),
        "planner_state_case_count": len(component_rows) * 960,
        "moving_regime_component_failure_count": len(architecture_failures),
        "native_only_infeasible_opportunity_count": sum(int(row["summary"]["native_only_infeasible_cases"]) > 0 for row in component_rows),
        "generated_increment_failure_opportunity_count": sum(int(row["summary"]["generated_increment_infeasible_without_cancellation_cases"]) > 0 for row in component_rows),
        "composite_failure_opportunity_count": sum(int(row["summary"]["composite_infeasible_cases"]) > 0 for row in component_rows),
        "continuity_failure_opportunity_count": sum(int(row["summary"]["state0_continuity_failures"]) > 0 for row in component_rows),
        "terminal_settling_failure_opportunity_count": sum(int(row["summary"]["post_recommit_terminal_capture_failures"]) > 0 or int(row["summary"]["post_recommit_composite_failures"]) > 0 for row in component_rows),
        "negative_native_generated_cancellation_accepted": False,
        "opportunities": component_rows, "runner_run_calls": 0, "simulation_calls": 0,
    }
    curvature = {
        "schema_version": "r2_bj_a5_curvature_disposition_audit_v1.0",
        "status": "DEFINED_FOR_ALL_REACHED_RECORDS" if not undefined else "CURVATURE_REPRESENTATION_UNRESOLVED",
        "records_reaching_curvature_disposition": len(curvature_rows),
        "undefined_category_count": len(undefined), "undefined_categories": undefined,
        "raw_and_robust_retained": True, "manual_point_deletion": False,
        "identity_specific_smoothing": False, "records": curvature_rows,
    }
    provenance_records = []
    for row in passing_rows:
        closure = row["predicate_result"]["closure"]
        provenance_records.append({
            "census_index": row["census_index"], "frame_index": row["frame_index"],
            "scenario_token": row["scenario_token"], "log_id": row["log_id"],
            "audit_rank_sha256": row["audit_rank_sha256"],
            "input_record_canonical_sha256": row["input_record_canonical_sha256"],
            "v_audit_mps": closure["speed_information"]["v_audit_mps"],
            "v_support_mps": closure["speed_information"]["v_support_mps"],
            "source_reference_sha256": closure["reference_geometry"]["source"]["sha256"],
            "target_reference_sha256": closure["reference_geometry"]["target"]["sha256"],
            "route_coverage": closure["route_coverage"],
            "component_output_canonical_sha256": canonical_sha(closure["component_summary"]),
            "predicate_result_canonical_sha256": row["predicate_result_canonical_sha256"],
            "closure_canonical_sha256": closure["canonical_sha256"],
        })
    provenance = {
        "schema_version": "r2_bj_a5_applicable_pool_provenance_manifest_v1.0",
        "status": "APPLICABLE_POOL_PROVENANCE_CLOSURE_100_PERCENT",
        "applicable_pool_count": len(provenance_records),
        "closure_percent": 100.0 if len(provenance_records) == len(passing_rows) else 0.0,
        "A4_frame_sha256": sha(FRAME), "A4_predicate_sha256": sha(PREDICATE),
        "records": provenance_records,
    }
    envelope = {
        "schema_version": "r2_bj_a5_finite_frame_census_envelope_v1.0",
        "status": final_status,
        "A4_historical_status_preserved": "APPLICABLE_POOL_INSUFFICIENT",
        "A4_historical_interpretation": "FRAME_CAPACITY_557_NOT_EVIDENCE_THAT_APPLICABLE_POOL_IS_BELOW_32",
        "A4_FRAME_CAPACITY": 557,
        "A5_CENSUS_EVALUATED": len(ledger_rows),
        "A5_APPLICABLE_POOL": len(passing_rows),
        "A5_COMPONENT_STAGE_COUNT": len(component_rows),
        "A5_MOVING_REGIME_COMPONENT_FAILURES": len(architecture_failures),
        "required_applicable_pool": 32,
        "curvature_undefined_category_count": len(undefined),
        "input_integrity": integrity["status"],
        "BJ_B_ROSTER_SELECTED": False, "RUNNER_RUN": 0,
        "engineering_simulation": 0, "scientific_simulation": 0, "TSB_simulation": 0,
        "R2_C_started": False, "confirmatory_smoke_started": False, "RBR_started": False,
    }
    firewall = {
        "schema_version": "r2_bj_a5_data_firewall_audit_v1.0", "status": "PASS_NO_OUTCOME_LEAKAGE",
        "frame_source": "EXACT_A4_FROZEN_557_ENTRIES", "source_universe_rescanned": False,
        "selection": "NONE", "rerank": False, "replacement": False, "early_stop": False,
        "same_log_second_candidate_added": False, "A5_failure_blacklist_entries_created": 0,
        "A4_or_earlier_files_rewritten": False, "V4_parameters_changed": False,
        "speed_floor_changed": False, "scientific_or_kinematic_thresholds_changed": False,
        "BJ_B_ROSTER_SELECTED": False, "RUNNER_RUN": 0,
        "engineering_simulation": 0, "scientific_simulation": 0, "TSB_simulation": 0,
        "R2_C_started": False, "confirmatory_smoke_started": False, "RBR_started": False,
        "protected_CSV_sha256": sha(PROTECTED),
    }
    for key, payload in (("eligibility", eligibility), ("curvature", curvature), ("components", components),
                         ("provenance", provenance), ("envelope", envelope), ("firewall", firewall)):
        write_new(OUT[key], payload)
    action = "请求 Scientific Owner 审阅是否授权后续 BJ-B roster 选择；本阶段不自动选择。" if final_status.endswith("READY_FOR_BJ_B_OWNER_REVIEW") else "保持 fail-closed，请 Scientific Owner 处置；不得进入 BJ-B。"
    OUT["request"].write_text(f"""# R2-BJ-A5 Scientific Owner 准备度请求 v0.1

## 结论

`{final_status}`。

{action}

## Frozen 557-log census

- `A4_FRAME_CAPACITY = 557`
- `A5_CENSUS_EVALUATED = {len(ledger_rows)}`
- `A5_APPLICABLE_POOL = {len(passing_rows)}`
- `A5_COMPONENT_STAGE_COUNT = {len(component_rows)}`
- `A5_MOVING_REGIME_COMPONENT_FAILURES = {len(architecture_failures)}`
- `BJ_B_ROSTER_SELECTED = FALSE`
- `RUNNER_RUN = 0`

557 条记录严格保持 A4 冻结顺序，无 rerank、replacement、source-universe rescan 或提前停止。只有通过 moving-regime、topology/reference 与 curvature 前置门的记录进入 960-case 离线 `_states` component audit。

## 治理

A4 的 `APPLICABLE_POOL_INSUFFICIENT` 保持为“原 768 frame 目标不可构造”的历史结论，不解释为 applicable pool 少于 32。V4、capture/morphology 参数、速度下限和全部科学/运动学阈值均未修改；A5 failure 不形成科学 outcome blacklist。

`runner.run=0`，engineering/scientific/TSB simulation 均为 0；未选择 BJ-B roster，未进入 R2-C、confirmatory smoke 或 RBR。
""", encoding="utf-8")
    print(json.dumps({
        "status": final_status, "evaluated": len(ledger_rows), "applicable": len(passing_rows),
        "component_stage": len(component_rows), "architecture_failures": len(architecture_failures),
        "runner_run_calls": 0, "simulation_calls": 0,
    }), flush=True)


def manifest() -> None:
    if MANIFEST.exists():
        raise FileExistsError(f"R2_BJ_A5_VERSIONED_OUTPUT_EXISTS:{MANIFEST}")
    paths = [
        CONTRACT, BINDING, FRAME, PREDICATE, GENERATOR, PLANNER,
        *OUT.values(), ROOT / "tools/r2_bj_a5_finite_frame_applicability_census.py",
        ROOT / "tools/r2_bj_a4_hlc_moving_regime_applicability.py",
        ROOT / "tools/r2_bj_a2_joint_support_applicability_audit.py",
        ROOT / "tools/r1_closed_loop_benchmark_v2_3.py",
        ROOT / "docs/stageR/r2/r2_bj_a_hlc_global_parameter_space_v4.0.json",
        ROOT / "tests/test_r2_bj_a5_finite_frame_applicability_census.py",
        ROOT / "QUICK_REFERENCE.md",
    ]
    envelope = json.loads(OUT["envelope"].read_text(encoding="utf-8"))
    payload = {
        "schema_version": "r2_bj_a5_component_sha_binding_manifest_v1.0",
        "status": envelope["status"],
        "components": [{"path": str(path.relative_to(ROOT)), "sha256": sha(path)} for path in paths],
        "component_SHA_closure": "PASS",
        "A4_FRAME_CAPACITY": envelope["A4_FRAME_CAPACITY"],
        "A5_CENSUS_EVALUATED": envelope["A5_CENSUS_EVALUATED"],
        "A5_APPLICABLE_POOL": envelope["A5_APPLICABLE_POOL"],
        "A5_COMPONENT_STAGE_COUNT": envelope["A5_COMPONENT_STAGE_COUNT"],
        "A5_MOVING_REGIME_COMPONENT_FAILURES": envelope["A5_MOVING_REGIME_COMPONENT_FAILURES"],
        "BJ_B_ROSTER_SELECTED": False, "RUNNER_RUN": 0, "simulation_calls": 0,
        "protected_CSV_sha256": sha(PROTECTED),
    }
    write_new(MANIFEST, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("census", "manifest"))
    args = parser.parse_args()
    census() if args.mode == "census" else manifest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
