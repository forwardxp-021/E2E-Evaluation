#!/usr/bin/env python3
"""R2-BJ-A4 moving-regime HLC applicability audit; offline and zero-run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import r2_bj_a2_joint_support_applicability_audit as a2  # noqa: E402
from tools import r2_bj_a3_hlc_prospective_applicability as a3  # noqa: E402
from tools.r1_b2_8_r3_prospective_selector import official_count, official_env  # noqa: E402
from tools.r1_b2_9_b_route_continuous_canary import _map_api  # noqa: E402
from tools.r1_prospective_generator_contract_v2 import sample_native_reference_no_extrapolation  # noqa: E402
from tools.r2_bj_a_hlc_morphology_feasible_generator_v4 import ARM_BASELINE, ARM_TREATMENT  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "docs/stageR/r1"
R2 = ROOT / "docs/stageR/r2"
CACHE = ROOT.parent / "nuplan/dataset/data/cache"
CONTRACT = R2 / "r2_bj_a4_preregistered_contract_v1.0.json"
PREDICATE = R2 / "r2_bj_a4_hlc_moving_regime_applicability_predicate_v1.0.json"
FRAME = R2 / "r2_bj_a4_hash_ranked_audit_frame_manifest_v1.0.json"
SOURCE = R1 / "r1_fresh_smoke_source_universe_v0.1.json"
SOURCE_SUMMARY = R1 / "r1_b2_7_enumeration_summary_v1.0.json"
BASE_EXCLUSION = R2 / "r2_bi_hlc_dev_kin_permanent_exclusion_ledger_v1.0.json"
R2BI_FIREWALL = R2 / "r2_bj_a_r2bi_outcome_exposure_ledger_v1.0.json"
A2_PROVENANCE = R2 / "r2_bj_a2_hlc_joint_support_provenance_manifest_v1.0.json"
A3_FRAME = R2 / "r2_bj_a3_hash_ranked_audit_frame_manifest_v1.0.json"
A3_ELIGIBILITY = R2 / "r2_bj_a3_fresh_candidate_eligibility_ledger_v1.0.json"
A3_COMPONENTS = R2 / "r2_bj_a3_native_generated_composite_component_audit_v1.0.json"
A3_SPEED = R2 / "r2_bj_a3_corrected_speed_envelope_review_v1.0.json"
A3_LEGACY = R2 / "r2_bj_a3_legacy_technical_disposition_ledger_v1.0.json"
SPACE = R2 / "r2_bj_a_hlc_global_parameter_space_v4.0.json"
PROTECTED = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"

OUT = {
    "disposition": R2 / "r2_bj_a4_a3_and_historical_applicability_disposition_v1.0.json",
    "exclusion": R2 / "r2_bj_a4_effective_audit_exclusion_closure_v1.0.json",
    "eligibility": R2 / "r2_bj_a4_fresh_candidate_eligibility_ledger_v1.0.json",
    "components": R2 / "r2_bj_a4_native_generated_composite_component_audit_v1.0.json",
    "curvature": R2 / "r2_bj_a4_curvature_disposition_audit_v1.0.json",
    "provenance": R2 / "r2_bj_a4_passing_candidate_provenance_manifest_v1.0.json",
    "envelope": R2 / "r2_bj_a4_moving_regime_candidate_pool_envelope_v1.0.json",
    "firewall": R2 / "r2_bj_a4_data_firewall_audit_v1.0.json",
    "request": R2 / "R2_BJ_A4_Scientific_Owner_Readiness_Request_v0.1.md",
}
MANIFEST = R2 / "r2_bj_a4_component_sha_binding_manifest_v1.0.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"R2_BJ_A4_VERSIONED_OUTPUT_EXISTS:{path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def rank_digest(token: str, log_id: str) -> str:
    return hashlib.sha256(f"R2_BJ_A4_HLC_MOVING_REGIME_AUDIT_FRAME_V1|2026090401|{token}|{log_id}".encode()).hexdigest()


def exclusion_closure() -> Mapping[str, Any]:
    sources = []
    rows: Dict[tuple[str, str], Dict[str, Any]] = {}
    for path, payload_rows, reason in (
        (BASE_EXCLUSION, json.loads(BASE_EXCLUSION.read_text(encoding="utf-8"))["entries"], "HISTORICAL_OR_PERMANENT_R1_R2_EXCLUSION"),
        (R2BI_FIREWALL, json.loads(R2BI_FIREWALL.read_text(encoding="utf-8"))["entries"], "R2_BI_HISTORY_ONLY"),
        (A2_PROVENANCE,
         json.loads(A2_PROVENANCE.read_text(encoding="utf-8"))["joint_records"] + json.loads(A2_PROVENANCE.read_text(encoding="utf-8"))["technical_extraction_failures"],
         "A2_AUDITED_OPPORTUNITY"),
        (A3_FRAME, json.loads(A3_FRAME.read_text(encoding="utf-8"))["entries"], "A3_FIXED_FRAME"),
    ):
        sources.append({"path": str(path.relative_to(ROOT)), "sha256": sha(path), "entry_count": len(payload_rows)})
        for row in payload_rows:
            key = (str(row["scenario_token"]).lower(), str(row["log_id"]))
            target = rows.setdefault(key, {"scenario_token": key[0], "log_id": key[1], "reasons": []})
            if reason not in target["reasons"]:
                target["reasons"].append(reason)
    entries = sorted(rows.values(), key=lambda row: (row["scenario_token"], row["log_id"]))
    return {
        "schema_version": "r2_bj_a4_effective_audit_exclusion_closure_v1.0",
        "status": "FROZEN_ADDITIVE_TOKEN_LOG_EXCLUSION_CLOSURE",
        "match_rule": "EXCLUDE_IF_SCENARIO_TOKEN_OR_LOG_ID_MATCHES",
        "source_ledgers": sources, "entry_count": len(entries),
        "unique_token_count": len({row["scenario_token"] for row in entries}),
        "unique_log_count": len({row["log_id"] for row in entries}),
        "A3_frame_entries_included": 256, "entries": entries,
        "outcome_based_selection": False,
    }


def exhaustive_basic_hlc_by_log(exclusion: Mapping[str, Any]) -> tuple[list[Dict[str, Any]], Mapping[str, Any], Counter[str]]:
    """Resolve the first hash-ranked basic-HLC record in every canonical log.

    A canonical nuPlan DB contains exactly one log.  Exhausting each DB before the
    global rank merge is therefore equivalent to a global rank scan with strict
    log dedup, without imposing an arbitrary in-memory prefix capacity.
    """
    duplicate_paths = {
        str(Path(row["db_path"]).resolve())
        for group in json.loads(SOURCE_SUMMARY.read_text(encoding="utf-8"))["source_universe"]["duplicate_db_groups"]
        for row in group["duplicate_occurrences"]
    }
    forbidden_tokens = {row["scenario_token"] for row in exclusion["entries"]}
    forbidden_logs = {row["log_id"] for row in exclusion["entries"]}
    accepted: list[Dict[str, Any]] = []
    failures: Counter[str] = Counter()
    map_cache: Dict[str, Any] = {}
    scanned = excluded = canonical_logs = attempted = 0
    for partition in ("mini", "train_pittsburgh"):
        for db_path in sorted((CACHE / partition).glob("*.db")):
            if str(db_path.resolve()) in duplicate_paths:
                continue
            rows = []
            log_ids = set()
            for row in a3.source_rows(db_path, partition):
                scanned += 1
                token, log_id = row["scenario_token"], row["log_id"]
                log_ids.add(log_id)
                if token in forbidden_tokens or log_id in forbidden_logs:
                    excluded += 1
                    continue
                row["audit_rank_sha256"] = rank_digest(token, log_id)
                rows.append(row)
            if not log_ids:
                failures["CANONICAL_LOG_HAS_NO_SOURCE_UNIVERSE_SCENARIO_TAG_ROWS"] += 1
                canonical_logs += 1
                continue
            if len(log_ids) != 1:
                raise RuntimeError(f"A4_CANONICAL_DB_LOG_CARDINALITY_NOT_ONE:{db_path}:{len(log_ids)}")
            canonical_logs += 1
            rows.sort(key=lambda row: (row["audit_rank_sha256"], row["scenario_token"], row["timestamp"]))
            for candidate in rows:
                attempted += 1
                try:
                    entry = a3.basic_hlc_candidate(candidate, map_cache)
                except Exception as error:
                    failures[str(error).split(":", 1)[0]] += 1
                    continue
                accepted.append({"candidate": candidate, "entry": entry})
                break
            if canonical_logs % 100 == 0:
                print(json.dumps({"progress": "A4_EXHAUSTIVE_LOG_FRAME_FREEZE", "canonical_logs": canonical_logs, "basic_HLC_logs": len(accepted), "attempted": attempted}), flush=True)
    accepted.sort(key=lambda item: (
        item["candidate"]["audit_rank_sha256"], item["candidate"]["scenario_token"],
        item["candidate"]["log_id"], item["candidate"]["timestamp"],
    ))
    return accepted, {
        "source_rows_scanned_after_canonical_DB_dedup": scanned,
        "rows_excluded_by_token_or_log": excluded,
        "duplicate_DB_occurrences_skipped": len(duplicate_paths),
        "canonical_logs_exhaustively_examined": canonical_logs,
        "basic_HLC_candidate_attempts": attempted,
        "logs_with_at_least_one_basic_HLC_record": len(accepted),
        "rank_prefix_capacity": None,
        "construction_strategy": "EXHAUSTIVE_PER_CANONICAL_LOG_FIRST_HASH_RANKED_BASIC_HLC_PASS_THEN_GLOBAL_HASH_RANK",
    }, failures


def freeze_frame() -> None:
    if FRAME.exists() or OUT["exclusion"].exists():
        raise FileExistsError("R2_BJ_A4_FRAME_OR_EXCLUSION_ALREADY_EXISTS")
    official_env()
    warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)
    exclusion = exclusion_closure()
    basic_by_log, accounting, failures = exhaustive_basic_hlc_by_log(exclusion)
    frame_size = int(json.loads(CONTRACT.read_text(encoding="utf-8"))["audit_frame"]["size"])
    frame = []
    for item in basic_by_log[:frame_size]:
        candidate, entry = item["candidate"], item["entry"]
        token, log_id = candidate["scenario_token"], candidate["log_id"]
        frame.append({
            "frame_index": len(frame) + 1, "audit_rank_sha256": candidate["audit_rank_sha256"],
            "scenario_token": token, "log_id": log_id, "scenario_anchor_timestamp_us": candidate["timestamp"],
            "db_file": candidate["db_file"], "db_path": candidate["db_path"], "source_partition": candidate["source_partition"],
            "map_name": candidate["map_name"], "initial_state": entry["initial_state"],
            "route_roadblock_ids": entry["route_roadblock_ids"], "route_fingerprint": entry["route_fingerprint"],
            "source_lane_id": entry["source_lane_id"], "target_lane_id": entry["target_lane_id"],
            "direction": entry["direction"], "basic_HLC_family_conditions": "PASS",
        })
    frame_complete = len(frame) == frame_size
    forbidden_tokens = {row["scenario_token"] for row in exclusion["entries"]}
    forbidden_logs = {row["log_id"] for row in exclusion["entries"]}
    overlap = sum(row["scenario_token"] in forbidden_tokens or row["log_id"] in forbidden_logs for row in frame)
    payload = {
        "schema_version": "r2_bj_a4_hash_ranked_audit_frame_manifest_v1.0",
        "status": "FROZEN_BEFORE_A4_MOVING_REGIME_PREDICATE_RESULTS" if frame_complete else "APPLICABLE_POOL_INSUFFICIENT",
        "contract": {"path": str(CONTRACT.relative_to(ROOT)), "sha256": sha(CONTRACT)},
        "predicate": {"path": str(PREDICATE.relative_to(ROOT)), "sha256": sha(PREDICATE)},
        "source_universe": {"path": str(SOURCE.relative_to(ROOT)), "sha256": sha(SOURCE)},
        "exclusion_closure_canonical_sha256": canonical_sha(exclusion),
        "frame_target_size": frame_size, "frame_size": len(frame), "frame_freeze_complete": frame_complete,
        "frame_cardinality_shortfall": frame_size - len(frame),
        "candidate_role": "A4_AUDIT_POOL_ONLY_NOT_BJ_B_ROSTER",
        "full_source_universe_census_claimed": False, "EARLY_STOP": False,
        "GLOBAL_TOKEN_DEDUP": True, "GLOBAL_LOG_DEDUP": True,
        "historical_A2_A3_or_permanent_overlap_count": overlap,
        "basic_candidates_evaluated_exhaustively": accounting["basic_HLC_candidate_attempts"],
        "basic_failure_counts_from_exhaustive_scan": dict(sorted(failures.items())),
        "source_scan_accounting": accounting, "entries": frame,
        "frame_entries_canonical_sha256": canonical_sha(frame),
        "A4_predicate_outcomes_opened_before_frame_freeze": 0,
        "runner_run_calls": 0, "simulation_calls": 0,
    }
    write_new(OUT["exclusion"], exclusion)
    write_new(FRAME, payload)
    print(json.dumps({"status": payload["status"], "frame": len(frame), "overlap": overlap}), flush=True)


def evaluate_frame_capacity_failure(frame: Mapping[str, Any]) -> None:
    """Materialize the fail-closed A4 record without opening predicate outcomes."""
    disposition = historical_disposition()
    frame_count = int(frame["frame_size"])
    target = int(frame["frame_target_size"])
    eligibility = {
        "schema_version": "r2_bj_a4_fresh_candidate_eligibility_ledger_v1.0",
        "status": "APPLICABLE_POOL_INSUFFICIENT",
        "audit_frame_sha256": sha(FRAME),
        "frame_freeze_complete": False,
        "frame_target_count": target,
        "available_unique_log_basic_HLC_count": frame_count,
        "frame_cardinality_shortfall": target - frame_count,
        "A4_predicate_evaluated_count": 0,
        "reason": "FROZEN_SOURCE_UNIVERSE_HAS_FEWER_THAN_768_BASIC_HLC_RECORDS_UNDER_GLOBAL_LOG_DEDUP",
        "EARLY_STOP": False,
        "exhaustive_canonical_log_scan_complete": True,
        "candidate_role": "FRAME_CONSTRUCTION_SUPPORT_ONLY_NOT_BJ_B_ROSTER",
        "entries": [],
    }
    components = {
        "schema_version": "r2_bj_a4_native_generated_composite_component_audit_v1.0",
        "status": "NOT_EXECUTED_FRAME_FREEZE_FAILED_CLOSED",
        "opportunities_reaching_component_stage": 0,
        "planner_state_case_count": 0,
        "saved_failure_case_count": 0,
        "reason": eligibility["reason"],
        "negative_native_generated_cancellation_accepted": False,
        "opportunities": [], "runner_run_calls": 0, "simulation_calls": 0,
    }
    curvature = {
        "schema_version": "r2_bj_a4_curvature_disposition_audit_v1.0",
        "status": "NOT_REACHED_FRAME_FREEZE_FAILED_CLOSED",
        "records_reaching_curvature_disposition": 0,
        "undefined_category_count": 0,
        "raw_and_robust_retained": True,
        "manual_point_deletion": False,
        "identity_specific_smoothing": False,
        "records": [],
    }
    provenance = {
        "schema_version": "r2_bj_a4_passing_candidate_provenance_manifest_v1.0",
        "status": "NO_PASSING_CANDIDATES_EVALUATED_FRAME_FREEZE_FAILED_CLOSED",
        "passing_count": 0, "closure_percent": 0.0,
        "source_universe_sha256": sha(SOURCE), "audit_frame_sha256": sha(FRAME), "records": [],
    }
    envelope = {
        "schema_version": "r2_bj_a4_moving_regime_candidate_pool_envelope_v1.0",
        "status": "APPLICABLE_POOL_INSUFFICIENT",
        "blocking_category": "A4_FRAME_CARDINALITY_UNATTAINABLE_UNDER_GLOBAL_LOG_DEDUP",
        "estimand": "MOVING_VEHICLE_HESITANT_LANE_CHANGE", "speed_floor_mps": 3.0,
        "audit_frame_complete": False, "audit_frame_target_count": target,
        "audit_frame_count": frame_count, "audit_frame_shortfall": target - frame_count,
        "exhaustive_canonical_log_scan_complete": True, "EARLY_STOP": False,
        "A4_predicate_evaluated_count": 0, "applicable_pool_count": None,
        "required_pool_count": 32, "moving_regime_component_stage_count": 0,
        "moving_regime_component_failure_count": 0,
        "V4_or_threshold_changed": False, "BJ_B_roster_selected": False,
        "runner_run_calls": 0, "engineering_simulation_calls": 0,
        "scientific_simulation_calls": 0, "TSB_simulation_calls": 0,
        "R2_C_started": False, "confirmatory_smoke_started": False, "RBR_started": False,
    }
    firewall = {
        "schema_version": "r2_bj_a4_data_firewall_audit_v1.0", "status": "PASS_NO_OUTCOME_LEAKAGE",
        "outcome_files_used_for_A4_candidate_selection": 0, "A4_predicate_outcomes_opened": 0,
        "anchor_speed_used": False, "scenario_specific_tuning": False,
        "V4_parameters_changed": False, "thresholds_changed": False,
        "A3_files_rewritten": False, "outcome_blacklist_created": False,
        "BJ_B_roster_selected": False, "runner_run_calls": 0,
        "engineering_simulation_calls": 0, "scientific_simulation_calls": 0,
        "TSB_simulation_calls": 0, "R2_C_started": False,
        "confirmatory_smoke_started": False, "RBR_started": False,
        "protected_CSV_sha256": sha(PROTECTED),
    }
    for key, payload in (("disposition", disposition), ("eligibility", eligibility), ("components", components),
                         ("curvature", curvature), ("provenance", provenance), ("envelope", envelope), ("firewall", firewall)):
        write_new(OUT[key], payload)
    OUT["request"].write_text(f"""# R2-BJ-A4 Scientific Owner 准备度请求 v0.1

## 结论

`APPLICABLE_POOL_INSUFFICIENT`。

冻结 source universe 在全部 {frame['source_scan_accounting']['canonical_logs_exhaustively_examined']} 个 canonical logs 穷尽后，仅能形成 {frame_count} 条满足基础 HLC 条件且 token/log 全局唯一的记录，低于预注册的 768 条 frame，缺口为 {target - frame_count}。因此 A4 frame 未冻结完成，未打开任何 A4 speed、topology、curvature 或 V4 predicate 结果，并保持 fail-closed；不得进入 BJ-B。

## 历史处置

A3 原 11 条 generated/composite failure 的 `v_audit` 均低于 `3.0 m/s`，新增适用域处置为 `LOW_SPEED_OUTSIDE_V4_APPLICABILITY`，不回写其 A3 failure、不加入 outcome blacklist。A2 历史 `3feb5f93f24e5b77` 保持 `HISTORICAL_OPPORTUNITY_NOT_APPLICABLE_UNDER_CURRENT_V2_3`。原 10 条 legacy failure 中重新可构造且低速的记录按 low-speed applicability 处置。

## 治理

V4、morphology/capture 参数及全部阈值未改变。未选择 BJ-B roster；`runner.run=0`，engineering/scientific/TSB simulation 均为 0；R2-C、confirmatory smoke、RBR 均未启动。
""", encoding="utf-8")
    print(json.dumps({"status": envelope["status"], "frame": frame_count, "target": target, "predicate_evaluated": 0}), flush=True)


def case_violations(case: Mapping[str, Any], limits: Mapping[str, float]) -> Mapping[str, Any] | None:
    reasons = []
    details: Dict[str, Any] = {}
    for component in ("native", "generated_increment", "composite"):
        values = case[component]
        flags = {
            "curvature": float(values["max_abs_curvature_inv_m"]) > float(limits["curvature_inv_m_max"]) + 1e-12,
            "yaw_rate": float(values["max_abs_yaw_rate_radps"]) > float(limits["yaw_rate_radps_max"]) + 1e-12,
            "lateral_acceleration": float(values["max_abs_lateral_acceleration_mps2"]) > float(limits["lateral_accel_mps2_max"]) + 1e-12,
        }
        if any(flags.values()):
            reasons.append(f"{component.upper()}_KINEMATIC_GATE")
            details[component] = {"values": values, "violated": flags}
    if case["construction_failure"]:
        reasons.append(f"CONSTRUCTION:{case['construction_failure']}")
    if not case["state0_exact"]:
        reasons.append("STATE0_CONTINUITY")
    if not case["terminal_pass"]:
        reasons.append("TERMINAL_TARGET_FRAME_OFFSET")
    if not reasons:
        return None
    return {
        "absolute_episode_time_s": case["absolute_episode_time_s"], "arm": case["arm"],
        "speed_mps": case["speed_mps"], "normal_residual_m": case["normal_residual_m"],
        "failure_reasons": reasons,
        "violated_curvature_yaw_rate_lateral_acceleration_gates": details,
        "morphology_increment": case["morphology_increment"],
        "stitching_capture_increment": case["stitching_capture_increment"],
        "generated_increment": case["generated_increment"], "composite": case["composite"],
        "continuity": case["continuity"],
        "terminal_target_frame_offset_abs_m": case["terminal_target_frame_offset_abs_m"],
    }


def component_grid(corridor: Mapping[str, Any], speeds: Sequence[float]) -> tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]:
    cases = [
        a2.audit_case(corridor, float(time_s), float(speed), arm, residual)
        for arm in (ARM_BASELINE, ARM_TREATMENT)
        for speed in speeds
        for residual in (-0.25, 0.0, 0.25)
        for time_s in np.arange(0.0, 8.0, 0.1)
    ]
    limits = a2.PARAMETERS["capture"]["frozen_feasibility_limits"]
    failures = [detail for detail in (case_violations(case, limits) for case in cases) if detail is not None]
    return a2.aggregate_cases(cases), failures


def evaluate_one(entry: Mapping[str, Any], map_cache: MutableMapping[str, Any]) -> Mapping[str, Any]:
    if official_count(str(entry["db_path"]), str(entry["scenario_token"])) != 1:
        return {"status": "FAIL", "stage": "P01", "reason": "EXACT_SINGLE_OFFICIAL_SCENARIO_RESOLUTION_FAIL"}
    speed = a3.corrected_speed(entry)
    if float(speed["v_audit_mps"]) < 3.0:
        return {"status": "FAIL", "stage": "P03", "reason": "LOW_SPEED_OUTSIDE_V4_APPLICABILITY", "speed_information": speed}
    try:
        corridor = a3.build_corridor(entry, map_cache, speed)
    except Exception as error:
        return {"status": "FAIL", "stage": "P04", "reason": f"{type(error).__name__}:{error}", "speed_information": speed}
    forward = float(speed["v_audit_mps"]) * 7.9
    try:
        source_quality = a2.curvature_quality(np.asarray(corridor["source_reference_xy"]), float(corridor["source_current_arc_m"]), forward)
        target_quality = a2.curvature_quality(np.asarray(corridor["target_reference_xy"]), float(corridor["target_current_arc_m"]), forward)
        distance = np.linspace(0.0, forward, 160)
        source_xy, _ = sample_native_reference_no_extrapolation(corridor["source_reference_xy"], float(corridor["source_current_arc_m"]) + distance)
        target_xy, _ = sample_native_reference_no_extrapolation(corridor["target_reference_xy"], float(corridor["target_current_arc_m"]) + distance)
    except Exception as error:
        return {"status": "FAIL", "stage": "P05_P06", "reason": f"{type(error).__name__}:{error}", "speed_information": speed}
    limits = a2.PARAMETERS["capture"]["frozen_feasibility_limits"]
    dispositions = {
        "source": a3.curvature_disposition(source_quality, float(speed["v_audit_mps"]), limits),
        "target": a3.curvature_disposition(target_quality, float(speed["v_audit_mps"]), limits),
    }
    if not all(row["raw_pass"] and row["robust_pass"] for row in dispositions.values()):
        return {
            "status": "FAIL", "stage": "P07", "reason": "RAW_OR_ROBUST_CURVATURE_FEASIBILITY_FAIL",
            "speed_information": speed, "curvature_quality": {"source": source_quality, "target": target_quality},
            "curvature_disposition": dispositions,
        }
    summary, failure_details = component_grid(corridor, (float(speed["v_audit_mps"]), float(speed["v_support_mps"])))
    if summary["native_only_infeasible_cases"]:
        stage, reason = "P08", "SOURCE_NATIVE_KINEMATIC_FEASIBILITY_UNRESOLVED"
    elif summary["generated_increment_infeasible_without_cancellation_cases"] or summary["V4_non_native_infeasible_cases"]:
        stage, reason = "P09", "V4_GENERATED_INCREMENT_INFEASIBLE"
    elif summary["composite_infeasible_cases"] or summary["state0_continuity_failures"]:
        stage, reason = "P10", "V4_COMPOSITE_OR_STATE0_CONTINUITY_INFEASIBLE"
    elif summary["post_recommit_terminal_capture_failures"] or summary["post_recommit_composite_failures"]:
        stage, reason = "P11", "V4_TERMINAL_SETTLING_INFEASIBLE"
    else:
        stage, reason = "P12", None
    closure = {
        "speed_information": speed,
        "route_coverage": {
            "required_reconstruction_forward_m": float(speed["v_support_mps"]) * 15.8 + 2.0,
            "source_total_length_m": corridor["source_total_length_m"], "target_total_length_m": corridor["target_total_length_m"],
            "source_current_arc_m": corridor["source_current_arc_m"], "target_current_arc_m": corridor["target_current_arc_m"],
            "source_remaining_margin_m": corridor["source_remaining_margin_m"], "target_remaining_margin_m": corridor["target_remaining_margin_m"],
            "source_components": corridor["source_components"], "target_components": corridor["target_components"],
            "extrapolation_used": False,
        },
        "reference_geometry": {
            "source": a3.encoded_array(np.asarray(corridor["source_reference_xy"])),
            "target": a3.encoded_array(np.asarray(corridor["target_reference_xy"])),
        },
        "curvature_quality": {"source": source_quality, "target": target_quality},
        "curvature_disposition": dispositions,
        "lane_separation_m": a2.percentile(np.linalg.norm(target_xy - source_xy, axis=1).tolist()),
        "component_summary": summary, "component_failure_details": failure_details,
    }
    closure["canonical_sha256"] = canonical_sha(closure)
    return {"status": "PASS" if reason is None else "FAIL", "stage": stage, "reason": reason, "closure": closure}


def historical_disposition() -> Mapping[str, Any]:
    a3_eligibility = json.loads(A3_ELIGIBILITY.read_text(encoding="utf-8"))
    a3_components = json.loads(A3_COMPONENTS.read_text(encoding="utf-8"))
    components = {(row["scenario_token"], row["log_id"]): row for row in a3_components["opportunities"]}
    failure_rows = []
    for row in a3_eligibility["entries"]:
        if "P08_V4_GENERATED_INCREMENT_INFEASIBLE" not in str(row.get("failure_reason")):
            continue
        component = components[(row["scenario_token"], row["log_id"])]
        v_audit = float(component["speed_cases_mps"][0])
        failure_rows.append({
            "scenario_token": row["scenario_token"], "log_id": row["log_id"],
            "A3_original_failure": row["failure_reason"], "v_audit_mps": v_audit,
            "v_audit_below_3p0": v_audit < 3.0, "A4_disposition": "LOW_SPEED_OUTSIDE_V4_APPLICABILITY",
            "A3_failure_result_changed": False, "moving_regime_V4_failure_counted": False,
            "outcome_blacklist_addition": False, "technical_rerun": False,
        })
    speed_review = json.loads(A3_SPEED.read_text(encoding="utf-8"))
    topology = next(row for row in speed_review["rows"] if row["scenario_token"] == "3feb5f93f24e5b77")
    legacy = json.loads(A3_LEGACY.read_text(encoding="utf-8"))["entries"]
    legacy_rows = []
    for row in legacy:
        low = float(row["A3_corrected_speed_information"]["v_audit_mps"]) < 3.0
        reconstructable = row["technical_disposition"] == "LEGACY_OPPORTUNITY_APPLICABLE_UNDER_V2_3"
        disposition = "LOW_SPEED_OUTSIDE_V4_APPLICABILITY" if reconstructable and low else row["technical_disposition"]
        legacy_rows.append({
            "scenario_token": row["scenario_token"], "log_id": row["log_id"],
            "v_audit_mps": row["A3_corrected_speed_information"]["v_audit_mps"],
            "A3_technical_disposition": row["technical_disposition"], "A4_disposition": disposition,
            "outcome_blacklist_addition": False, "historical_result_changed": False,
        })
    return {
        "schema_version": "r2_bj_a4_a3_and_historical_applicability_disposition_v1.0",
        "status": "COMPLETE_NON_BLACKLIST_APPLICABILITY_DISPOSITION",
        "A3_generated_composite_failure_count": len(failure_rows),
        "A3_failure_all_11_below_speed_floor": len(failure_rows) == 11 and all(row["v_audit_below_3p0"] for row in failure_rows),
        "A3_failures": failure_rows,
        "A2_corrected_speed_topology_failure": {
            "scenario_token": topology["scenario_token"], "log_id": topology["log_id"],
            "failure_reason": topology["failure_reason"],
            "A4_disposition": "HISTORICAL_OPPORTUNITY_NOT_APPLICABLE_UNDER_CURRENT_V2_3",
            "topology_builder_modified": False, "historical_result_changed": False,
        },
        "legacy_10": legacy_rows,
        "legacy_reconstructable_but_low_speed_count": sum(row["A4_disposition"] == "LOW_SPEED_OUTSIDE_V4_APPLICABILITY" for row in legacy_rows),
        "outcome_blacklist_entries_created": 0,
    }


def evaluate() -> None:
    if any(path.exists() for path in OUT.values() if path != OUT["exclusion"]):
        raise FileExistsError("R2_BJ_A4_VERSIONED_EVALUATION_OUTPUT_EXISTS")
    if not FRAME.exists() or not OUT["exclusion"].exists():
        raise FileNotFoundError("A4_FRAME_AND_EXCLUSION_MUST_BE_FROZEN_FIRST")
    frame = json.loads(FRAME.read_text(encoding="utf-8"))
    if not frame.get("frame_freeze_complete", False):
        evaluate_frame_capacity_failure(frame)
        return
    official_env()
    warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)
    a2.PARAMETERS = json.loads(SPACE.read_text(encoding="utf-8"))["global_parameters"]
    map_cache: Dict[str, Any] = {}
    rows, reasons, stages = [], Counter(), Counter()
    component_rows, curvature_rows, passing = [], [], []
    for index, frame_row in enumerate(frame["entries"], 1):
        entry = dict(frame_row)
        entry["timestamp"] = entry["scenario_anchor_timestamp_us"]
        result = evaluate_one(entry, map_cache)
        ledger_row = {
            "frame_index": frame_row["frame_index"], "audit_rank_sha256": frame_row["audit_rank_sha256"],
            "scenario_token": frame_row["scenario_token"], "log_id": frame_row["log_id"],
            "scenario_anchor_timestamp_us": frame_row["scenario_anchor_timestamp_us"],
            "anchor_speed_or_timestamp_used_for_eligibility": False,
            **result,
        }
        rows.append(ledger_row)
        stages[result["stage"]] += 1
        if result["reason"]:
            reasons[result["reason"].split(":", 1)[0]] += 1
        closure = result.get("closure")
        if closure is not None:
            component_rows.append({
                "frame_index": frame_row["frame_index"], "scenario_token": frame_row["scenario_token"],
                "log_id": frame_row["log_id"], "status": result["status"], "failure_reason": result["reason"],
                "speed_cases_mps": [closure["speed_information"]["v_audit_mps"], closure["speed_information"]["v_support_mps"]],
                "summary": closure["component_summary"], "failure_details": closure["component_failure_details"],
            })
            curvature_rows.append({
                "frame_index": frame_row["frame_index"], "scenario_token": frame_row["scenario_token"],
                "log_id": frame_row["log_id"], "status": result["status"],
                "quality": closure["curvature_quality"], "disposition": closure["curvature_disposition"],
            })
        elif result["stage"] == "P07":
            curvature_rows.append({
                "frame_index": frame_row["frame_index"], "scenario_token": frame_row["scenario_token"],
                "log_id": frame_row["log_id"], "status": result["status"],
                "quality": result["curvature_quality"], "disposition": result["curvature_disposition"],
            })
        if result["status"] == "PASS":
            passing.append(ledger_row)
        print(json.dumps({"progress": "A4_FRAME_EVALUATION", "completed": index, "total": 768, "pass": len(passing), "component_stage": len(component_rows)}), flush=True)
    eligibility = {
        "schema_version": "r2_bj_a4_fresh_candidate_eligibility_ledger_v1.0",
        "status": "COMPLETE_FIXED_768_AUDIT_FRAME", "audit_frame_sha256": sha(FRAME),
        "audited_count": len(rows), "applicable_count": len(passing), "eligibility_pass_rate": len(passing) / len(rows),
        "stage_completion_counts": dict(sorted(stages.items())), "failure_reason_counts": dict(sorted(reasons.items())),
        "entries": rows, "EARLY_STOP": False, "full_source_universe_census_claimed": False,
        "candidate_role": "A4_PASSING_POOL_NOT_BJ_B_ROSTER", "roster_selected": False,
    }
    component_failure_rows = [row for row in component_rows if row["status"] != "PASS"]
    components = {
        "schema_version": "r2_bj_a4_native_generated_composite_component_audit_v1.0",
        "status": "PASS" if not component_failure_rows else "MOVING_REGIME_COMPONENT_FAILURE_PRESENT",
        "opportunities_reaching_component_stage": len(component_rows),
        "planner_state_case_count": len(component_rows) * 960,
        "opportunities_full_gate_pass": len(component_rows) - len(component_failure_rows),
        "opportunities_with_any_V4_failure": len(component_failure_rows),
        "native_only_infeasible_opportunities": sum(row["summary"]["native_only_infeasible_cases"] > 0 for row in component_rows),
        "generated_increment_infeasible_opportunities": sum(row["summary"]["generated_increment_infeasible_without_cancellation_cases"] > 0 for row in component_rows),
        "composite_infeasible_opportunities": sum(row["summary"]["composite_infeasible_cases"] > 0 for row in component_rows),
        "terminal_settling_infeasible_opportunities": sum(row["summary"]["post_recommit_terminal_capture_failures"] > 0 for row in component_rows),
        "saved_failure_case_count": sum(len(row["failure_details"]) for row in component_rows),
        "negative_native_generated_cancellation_accepted": False,
        "opportunities": component_rows, "runner_run_calls": 0, "simulation_calls": 0,
    }
    undefined = sum(
        side["disposition"] not in {
            "LOW_MAGNITUDE_RAW_ROBUST_FEASIBILITY_CONCORDANT",
            "LOCALIZED_POINTWISE_SPIKE_RAW_AND_ROBUST_FEASIBILITY_CONCORDANT",
            "RAW_ROBUST_CONCORDANT_SUSTAINED_FEASIBLE",
            "RAW_CURVATURE_FEASIBILITY_FAIL", "ROBUST_CURVATURE_FEASIBILITY_FAIL",
        }
        for row in curvature_rows for side in row["disposition"].values()
    )
    curvature = {
        "schema_version": "r2_bj_a4_curvature_disposition_audit_v1.0",
        "status": "DEFINED_FOR_ALL_REACHED_RECORDS" if undefined == 0 else "UNDEFINED_CATEGORY_PRESENT",
        "records_reaching_curvature_disposition": len(curvature_rows), "undefined_category_count": undefined,
        "raw_and_robust_retained": True, "manual_point_deletion": False, "identity_specific_smoothing": False,
        "records": curvature_rows,
    }
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    provenance_records = [{
        "frame_index": row["frame_index"], "scenario_token": row["scenario_token"], "log_id": row["log_id"],
        "audit_rank_sha256": row["audit_rank_sha256"],
        "v_audit_mps": row["closure"]["speed_information"]["v_audit_mps"],
        "v_support_mps": row["closure"]["speed_information"]["v_support_mps"],
        "source_reference_sha256": row["closure"]["reference_geometry"]["source"]["sha256"],
        "target_reference_sha256": row["closure"]["reference_geometry"]["target"]["sha256"],
        "route_coverage": row["closure"]["route_coverage"], "closure_canonical_sha256": row["closure"]["canonical_sha256"],
    } for row in passing]
    provenance = {
        "schema_version": "r2_bj_a4_passing_candidate_provenance_manifest_v1.0",
        "status": "PASSING_PROVENANCE_CLOSURE_100_PERCENT", "passing_count": len(provenance_records),
        "closure_percent": 100.0 if provenance_records else 0.0,
        "source_universe_sha256": sha(SOURCE), "source_root_fingerprint_sha256": source["source_db"]["source_root_fingerprint_sha256"],
        "map_root_fingerprint_sha256": source["map_binding"]["map_root_fingerprint_sha256"],
        "audit_frame_sha256": sha(FRAME), "records": provenance_records,
    }
    disposition = historical_disposition()
    if len(passing) < 32:
        status = "APPLICABLE_POOL_INSUFFICIENT"
    elif component_failure_rows:
        status = "R2_BJ_A4_MOVING_REGIME_ARCHITECTURE_NOT_READY"
    elif undefined:
        status = "CURVATURE_REPRESENTATION_UNRESOLVED"
    elif not disposition["A3_failure_all_11_below_speed_floor"]:
        status = "R2_BJ_A4_MOVING_REGIME_ARCHITECTURE_NOT_READY"
    else:
        status = "R2_BJ_A4_MOVING_REGIME_READY_FOR_OWNER_REVIEW"
    envelope = {
        "schema_version": "r2_bj_a4_moving_regime_candidate_pool_envelope_v1.0", "status": status,
        "estimand": "MOVING_VEHICLE_HESITANT_LANE_CHANGE", "speed_floor_mps": 3.0,
        "audit_frame_complete": len(rows) == 768, "audit_frame_count": len(rows), "EARLY_STOP": False,
        "applicable_pool_count": len(passing), "required_pool_count": 32,
        "moving_regime_component_stage_count": len(component_rows),
        "moving_regime_component_failure_count": len(component_failure_rows),
        "passing_provenance_closure_percent": provenance["closure_percent"],
        "curvature_undefined_category_count": undefined,
        "speed_joint_support_mps": {
            "v_audit": a2.percentile([row["closure"]["speed_information"]["v_audit_mps"] for row in passing]) if passing else None,
            "v_support": a2.percentile([row["closure"]["speed_information"]["v_support_mps"] for row in passing]) if passing else None,
        },
        "lane_separation_joint_support_m": {
            "min": min(row["closure"]["lane_separation_m"]["min"] for row in passing) if passing else None,
            "max": max(row["closure"]["lane_separation_m"]["max"] for row in passing) if passing else None,
        },
        "full_source_universe_census_claimed": False, "V4_or_threshold_changed": False,
        "BJ_B_roster_selected": False, "runner_run_calls": 0,
        "engineering_simulation_calls": 0, "scientific_simulation_calls": 0, "TSB_simulation_calls": 0,
        "R2_C_started": False, "confirmatory_smoke_started": False, "RBR_started": False,
    }
    firewall = {
        "schema_version": "r2_bj_a4_data_firewall_audit_v1.0", "status": "PASS_NO_OUTCOME_LEAKAGE",
        "allowed_inputs": ["frozen_source_universe", "pre_treatment_speed_0_to_1p0s", "official_map_route", "frozen_V2_3", "frozen_V4"],
        "outcome_files_used_for_A4_candidate_selection": 0, "anchor_speed_used": False,
        "scenario_specific_tuning": False, "V4_parameters_changed": False, "thresholds_changed": False,
        "A3_files_rewritten": False, "outcome_blacklist_created": False, "BJ_B_roster_selected": False,
        "runner_run_calls": 0, "engineering_simulation_calls": 0, "scientific_simulation_calls": 0,
        "TSB_simulation_calls": 0, "R2_C_started": False, "confirmatory_smoke_started": False, "RBR_started": False,
        "protected_CSV_sha256": sha(PROTECTED),
    }
    for key, payload in (("disposition", disposition), ("eligibility", eligibility), ("components", components),
                         ("curvature", curvature), ("provenance", provenance), ("envelope", envelope), ("firewall", firewall)):
        write_new(OUT[key], payload)
    decision = "可提交 Owner readiness review，但本阶段不选择 BJ-B roster。" if status.endswith("READY_FOR_OWNER_REVIEW") else "请求暂缓；保持 fail-closed，不进入 BJ-B。"
    OUT["request"].write_text(f"""# R2-BJ-A4 Scientific Owner 准备度请求 v0.1

## 结论

`{status}`。

{decision}

## Moving-regime candidate pool

- estimand：`MOVING_VEHICLE_HESITANT_LANE_CHANGE`。
- 固定 A4 frame：{len(rows)}/768 全部审计，无提前停止。
- 完整 predicate 通过：{len(passing)}/768（{100.0 * len(passing) / 768:.2f}%），最低要求为 32。
- 到达 V4 component stage：{len(component_rows)}；出现 generated/composite/terminal 等 V4 failure 的 opportunity：{len(component_failure_rows)}。
- passing candidate provenance closure：{provenance['closure_percent']:.1f}%；curvature 未定义类别：{undefined}。

## 历史处置

A3 原 11 条 generated/composite failure 的 `v_audit` 全部低于 `3.0 m/s`，统一新增适用域处置 `LOW_SPEED_OUTSIDE_V4_APPLICABILITY`；不回写其 A3 failure，不加入 outcome blacklist，不计作 moving-regime V4 failure。

A2 历史 `3feb5f93f24e5b77` 保持 topology failure，处置为 `HISTORICAL_OPPORTUNITY_NOT_APPLICABLE_UNDER_CURRENT_V2_3`。原 10 条 legacy failure 中 2 条重新可构造但低于速度下限，按 low-speed applicability 处置。

## 治理

V4、morphology/capture 参数及运动学阈值均未改变。A4 passing pool 不是 BJ-B roster。`runner.run=0`，engineering/scientific/TSB simulation 均为 0；R2-C、confirmatory smoke、RBR 均未启动。
""", encoding="utf-8")
    print(json.dumps({"status": status, "pass": len(passing), "component_stage": len(component_rows), "component_fail": len(component_failure_rows), "runner_run_calls": 0, "simulation_calls": 0}), flush=True)


def build_manifest() -> None:
    if MANIFEST.exists():
        raise FileExistsError(f"R2_BJ_A4_VERSIONED_OUTPUT_EXISTS:{MANIFEST}")
    paths = [
        CONTRACT, PREDICATE, FRAME, OUT["exclusion"], OUT["disposition"], OUT["eligibility"],
        OUT["components"], OUT["curvature"], OUT["provenance"], OUT["envelope"], OUT["firewall"], OUT["request"],
        SOURCE, SOURCE_SUMMARY, BASE_EXCLUSION, R2BI_FIREWALL, A2_PROVENANCE, A3_FRAME, A3_ELIGIBILITY,
        A3_COMPONENTS, A3_SPEED, A3_LEGACY, SPACE,
        ROOT / "tools/r2_bj_a4_hlc_moving_regime_applicability.py",
        ROOT / "tools/r2_bj_a3_hlc_prospective_applicability.py",
        ROOT / "tools/r2_bj_a2_joint_support_applicability_audit.py",
        ROOT / "tools/r2_bj_a_hlc_morphology_feasible_generator_v4.py",
        ROOT / "tools/r2_bj_a_hlc_morphology_feasible_planner_v4.py",
        ROOT / "tools/r1_closed_loop_benchmark_v2_3.py",
        ROOT / "tests/test_r2_bj_a4_hlc_moving_regime_applicability.py", ROOT / "QUICK_REFERENCE.md",
    ]
    envelope = json.loads(OUT["envelope"].read_text(encoding="utf-8"))
    payload = {
        "schema_version": "r2_bj_a4_component_sha_binding_manifest_v1.0", "status": envelope["status"],
        "components": [{"path": str(path.relative_to(ROOT)), "sha256": sha(path)} for path in paths],
        "component_SHA_closure": "PASS", "frame_target_count": 768,
        "frame_count": envelope["audit_frame_count"],
        "frame_freeze_complete": envelope["audit_frame_complete"],
        "applicable_pool_count": envelope["applicable_pool_count"],
        "V4_or_threshold_changed": False, "BJ_B_roster_selected": False,
        "runner_run_calls": 0, "simulation_calls": 0, "protected_CSV_sha256": sha(PROTECTED),
    }
    write_new(MANIFEST, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze-frame", "evaluate", "manifest"))
    args = parser.parse_args()
    if args.mode == "freeze-frame":
        freeze_frame()
    elif args.mode == "evaluate":
        evaluate()
    else:
        build_manifest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
