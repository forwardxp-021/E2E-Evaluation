#!/usr/bin/env python3
"""Prepare, execute, and adjudicate the frozen B1 TSB qualification."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/stageR/b1_tsb_qualification"
RUN_ROOT = ROOT / "outputs/stageR/b1_tsb_qualification_v1"
S2R = ROOT / "docs/stageR/s2r_engineering_closure"
ROSTER = OUT / "B1_TSB_Qualification_Roster_v1.csv"
SPECS = OUT / "B1_TSB_Qualification_Arm_Specs_v1.json"
BINDINGS = OUT / "B1_TSB_Qualification_Pair_Bindings_v1.json"
AUTH = OUT / "B1_TSB_Qualification_Authorization_v1.json"
PRE_MANIFEST = OUT / "B1_TSB_Qualification_PreRun_Manifest_v1.json"
SELECTION_AUDIT = OUT / "B1_TSB_Qualification_Selection_Audit_v1.csv"
PARAMETERS = ROOT / "docs/stageR/r2/r2_b_calibration_rounds/r2_b_tsb_round_0_parameters_v1.0.json"
EXECUTOR = ROOT / "tools/b1_tsb_qualification_executor.py"
ANALYZER = ROOT / "tools/r1_b2_8_r3_2_post_run_evaluator_dispatcher.py"
PROTECTED = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"
PROTECTED_SHA = "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8"
OWNER_PROMPT = Path("/Users/liuqing/.codex/attachments/24ac0752-9196-459a-9127-37d3941d9f80/pasted-text.txt")
NU_PYTHON = Path("/Users/liuqing/miniconda3/envs/nuplan/bin/python")
SALT = "B1_FROZEN_TSB_QUALIFICATION_V1"
TARGET_PAIRS = 20
MAX_ARMS = 40
SOURCE_FILES = (
    "tools/b1_tsb_qualification_executor.py",
    "tools/b1_tsb_qualification.py",
    "tools/s2r_production_executor.py",
    "tools/r2_b_controller_aware_planner_v1.py",
    "tools/r2_bj_b0_2_passive_actual_lqr_recorder.py",
    "tools/r1_b2_8_r3_2_post_run_evaluator_dispatcher.py",
    "tools/s1_protocol_schema.py",
    "docs/stageR/r2/r2_b_calibration_rounds/r2_b_tsb_round_0_parameters_v1.0.json",
)


def ensure_runtime() -> None:
    if importlib.util.find_spec("numpy") is None:
        if not NU_PYTHON.is_file():
            raise RuntimeError("nuPlan Python runtime is unavailable")
        os.execv(str(NU_PYTHON), [str(NU_PYTHON), "-B", str(Path(__file__).resolve()), *sys.argv[1:]])


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def validate_frozen_parameters() -> Mapping[str, Any]:
    payload = read_json(PARAMETERS)
    expected = {
        "start_s": 1.1,
        "baseline_brake_mps2": -1.45,
        "baseline_duration_s": 1.8,
        "first_brake_mps2": -2.4,
        "first_brake_duration_s": 0.9,
        "release_mps2": 1.4,
        "release_duration_s": 1.3,
        "second_brake_mps2": -2.4,
        "second_brake_duration_s": 0.9,
    }
    if payload.get("status") != "FROZEN_BEFORE_THIS_ROUND_SIMULATION" or payload.get("parameters") != expected:
        raise RuntimeError("FROZEN_TSB_PARAMETERS_DO_NOT_MATCH_OWNER_AUTHORIZATION")
    return payload


def _pair_binding(winner: Mapping[str, Any], map_cache: dict[str, Any]) -> Mapping[str, Any]:
    from tools import r1_b2_7_freeze_official_smoke_roster_v2 as frozen
    from tools.r1_closed_loop_context_adapter_v2_1 import build_closed_loop_context_v2_1
    from tools.r1_context_mechanism_core import canonical_json_sha256
    from tools.r1_official_map_query_bridge_v2_1 import R1OfficialMapQueryBridgeV2_1

    initial = winner["initial"]
    replay = frozen._sampled_replay(winner, initial)
    api = frozen.map_api(ROOT.parent / "nuplan/dataset/maps", str(winner["map_name"]), map_cache)
    bridge = R1OfficialMapQueryBridgeV2_1(api)
    route_fingerprint = canonical_sha([str(value) for value in winner["route_roadblock_ids"]])
    canonical_context = build_closed_loop_context_v2_1(
        family="R-TSB",
        scenario_token=str(winner["scenario_token"]),
        map_version=str(winner["map_name"]),
        route_fingerprint=route_fingerprint,
        initial_state_fingerprint=str(initial["initial_state_fingerprint"]),
        log_id=str(winner["log_id"]),
        route_roadblock_ids=winner["route_roadblock_ids"],
        frames=replay["frames"],
        map_query=bridge,
    )
    context = {
        "pre_context_raw_hash": canonical_json_sha256(replay["frames"][:10]),
        "canonical_context_json_hash": canonical_context["canonical_context_json_hash"],
        "canonical_context": canonical_context,
        "frozen_temporal_semantics": {"pre_iterations": list(range(10)), "anchor_iteration": 10},
        "stage5d_slot_semantics": "AUTHORITATIVE_STAGE5D_EXACT_PARITY_LANE_AWARE_ONLY",
    }
    return {
        "family": "R-TSB",
        "scenario_token": winner["scenario_token"],
        "log_id": winner["log_id"],
        "baseline_context": context,
        "treatment_context": context,
        "context_source": "B1_STATIC_CERTIFIED_IDENTITY_PLUS_READ_ONLY_OFFICIAL_REPLAY_CONTEXT_V2_1",
        "map_route_binding_sha256": canonical_sha({
            "map_name": winner["map_name"],
            "route": winner["route_roadblock_ids"],
            "initial": initial["initial_state_fingerprint"],
        }),
        "pretreatment_clearance": None,
    }


def prepare(authorized_source_git_sha: str) -> None:
    ensure_runtime()
    validate_frozen_parameters()
    if git("branch", "--show-current") != "20260825_stageR_new":
        raise RuntimeError("B1 preparation requires branch 20260825_stageR_new")
    if sha(PROTECTED) != PROTECTED_SHA:
        raise RuntimeError("protected asset hash mismatch")
    if git("rev-parse", authorized_source_git_sha) != authorized_source_git_sha:
        raise RuntimeError("authorized source git SHA is not an exact commit")
    from tools import s2r_engineering_closure as s2r

    roles = s2r.role_rows()
    old = s2r.read_json(s2r.OLD_CENSUS)
    ranked, candidate_counts = s2r.ranked_candidates(old, roles)
    _, selected, _ = s2r.census_rows(roles, ranked, candidate_counts)
    eligible_by_class: dict[str, list[tuple[str, str, Mapping[str, Any]]]] = {
        "E2_BENCHMARK_ENGINEERING": [],
        "E1_UNRELATED_HISTORICAL_USE": [],
    }
    for session_id, winner in selected.items():
        exposure = roles[session_id]["exposure_class"]
        if exposure not in eligible_by_class:
            continue
        selection_hash = hashlib.sha256(f"{SALT}|{session_id}|{winner['scenario_token']}".encode()).hexdigest()
        eligible_by_class[exposure].append((selection_hash, session_id, winner))
    for values in eligible_by_class.values():
        values.sort(key=lambda item: (item[0], item[1], item[2]["scenario_token"]))
    roster_rows: list[dict[str, Any]] = []
    arms: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    schedule: list[dict[str, Any]] = []
    selection_audit: list[dict[str, Any]] = []
    map_cache: dict[str, Any] = {}
    chosen: list[tuple[str, str, Mapping[str, Any], Mapping[str, Any], str]] = []
    ordered_candidates = [
        (exposure, selection_hash, session_id, winner)
        for exposure in ("E2_BENCHMARK_ENGINEERING", "E1_UNRELATED_HISTORICAL_USE")
        for selection_hash, session_id, winner in eligible_by_class[exposure]
    ]
    for candidate_rank, (exposure, selection_hash, session_id, winner) in enumerate(ordered_candidates, 1):
        try:
            binding = _pair_binding(winner, map_cache)
        except Exception as exc:
            selection_audit.append({
                "candidate_rank": candidate_rank,
                "selection_hash": selection_hash,
                "session_id": session_id,
                "scenario_token": winner["scenario_token"],
                "exposure_class": exposure,
                "decision": "STATIC_B1_CONTEXT_INCOMPATIBLE",
                "reason": f"{type(exc).__name__}:{exc}",
            })
            continue
        selection_audit.append({
            "candidate_rank": candidate_rank,
            "selection_hash": selection_hash,
            "session_id": session_id,
            "scenario_token": winner["scenario_token"],
            "exposure_class": exposure,
            "decision": "SELECTED",
            "reason": "",
        })
        chosen.append((selection_hash, session_id, winner, binding, exposure))
        if len(chosen) == TARGET_PAIRS:
            break
    if len(chosen) != TARGET_PAIRS or len({item[1] for item in chosen}) != TARGET_PAIRS:
        write_csv(SELECTION_AUDIT, selection_audit)
        raise RuntimeError("B1_CONTEXT_COMPATIBLE_UNIQUE_E2_E1_SESSION_CAPACITY_BELOW_20")

    for pair_index, (selection_hash, session_id, winner, frozen_binding, exposure) in enumerate(chosen, 1):
        pair_id = f"B1-TSB-{pair_index:02d}"
        baseline_id, treatment_id = f"{pair_id}-BASELINE", f"{pair_id}-TREATMENT"
        order = ["BASELINE", "TREATMENT"] if int(hashlib.sha256(pair_id.encode()).hexdigest(), 16) % 2 == 0 else ["TREATMENT", "BASELINE"]
        for arm_name in order:
            schedule.append({"execution_order": len(schedule) + 1, "pair_id": pair_id, "arm": arm_name, "run_id": baseline_id if arm_name == "BASELINE" else treatment_id})
        common = {
            "pair_id": pair_id,
            "session_id": session_id,
            "log_id": winner["log_id"],
            "scenario_token": winner["scenario_token"],
            "exposure_class": exposure,
            "role": "B",
            "db_path": winner["db_path"],
            "map_name": winner["map_name"],
            "route_roadblock_ids": winner["route_roadblock_ids"],
            "route_id": winner["route_id"],
            "precontext": winner["precontext"],
            "precontext_id": winner["precontext_id"],
            "start_timestamp_us": winner["initial"]["initial_time_us"],
        }
        arms.extend([
            {**common, "run_id": baseline_id, "arm": "BASELINE"},
            {**common, "run_id": treatment_id, "arm": "TREATMENT"},
        ])
        binding = dict(frozen_binding)
        binding.update({"pair_id": pair_id, "baseline_run_id": baseline_id, "treatment_run_id": treatment_id})
        bindings.append(binding)
        roster_rows.append({
            "pair_id": pair_id,
            "session_id": session_id,
            "log_id": winner["log_id"],
            "scenario_token": winner["scenario_token"],
            "exposure_class": exposure,
            "static_eligibility_status": "STATIC_ELIGIBLE",
            "initial_speed_mps": f"{float(winner['initial']['initial_speed_mps']):.6f}",
            "route_id": winner["route_id"],
            "precontext_id": winner["precontext_id"],
            "baseline_arm_id": baseline_id,
            "treatment_arm_id": treatment_id,
            "arm_order": "->".join(order),
            "selection_rank": pair_index,
            "selection_salt": SALT,
            "selection_hash": selection_hash,
            "execution_status": "NOT_RUN",
        })

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(ROSTER, roster_rows)
    write_csv(SELECTION_AUDIT, selection_audit)
    write_json(SPECS, {"schema_version": "b1_tsb_arm_specs_v1", "arms": arms, "schedule": schedule})
    write_json(BINDINGS, {"schema_version": "b1_tsb_pair_bindings_v1", "pairs": bindings})
    source_hashes = {relative: sha(ROOT / relative) for relative in SOURCE_FILES}
    authorization = {
        "schema_version": "B1_TSB_Qualification_Authorization_v1",
        "stage": "B1_FROZEN_TSB_QUALIFICATION",
        "authorization_status": "AUTHORIZED",
        "allowed_role": "B",
        "allowed_exposure_classes": ["E2_BENCHMARK_ENGINEERING", "E1_UNRELATED_HISTORICAL_USE"],
        "allowed_arms": ["BASELINE", "TREATMENT"],
        "max_scientific_arms": MAX_ARMS,
        "planned_session_pairs": TARGET_PAIRS,
        "roster_sha256": sha(ROSTER),
        "arm_specs_sha256": sha(SPECS),
        "pair_bindings_sha256": sha(BINDINGS),
        "selection_audit_sha256": sha(SELECTION_AUDIT),
        "executor_sha256": sha(EXECUTOR),
        "tsb_config_sha256": sha(PARAMETERS),
        "analyzer_sha256": sha(ANALYZER),
        "schema_sha256": sha(ROOT / "tools/s1_protocol_schema.py"),
        "planner_sha256": sha(ROOT / "tools/r2_b_controller_aware_planner_v1.py"),
        "recorder_sha256": sha(ROOT / "tools/r2_bj_b0_2_passive_actual_lqr_recorder.py"),
        "branch": "20260825_stageR_new",
        "authorized_source_git_sha": authorized_source_git_sha,
        "source_file_hashes": source_hashes,
        "owner_instruction_sha256": sha(OWNER_PROMPT),
        "stop_policy": "COMPLETE_PRE_REGISTERED_ROSTER_ON_SCIENTIFIC_FAILURE; STOP_ON_INFRASTRUCTURE_FAILURE",
        "success_rule": "ALL_20_PAIRS_PASS_FROZEN_JOINT_QUALIFICATION",
        "allowed_runs": schedule,
        "rbr_training_authorized": False,
        "primary_evaluation_authorized": False,
        "v_primary_execution_authorized": False,
        "c_confirmatory_execution_authorized": False,
    }
    write_json(AUTH, authorization)
    pre_artifacts = [
        "B1_TSB_Qualification_Protocol_v1.md",
        "B1_Stop_Policy_Addendum_v1.md",
        ROSTER.name,
        SPECS.name,
        BINDINGS.name,
        SELECTION_AUDIT.name,
        AUTH.name,
    ]
    write_json(PRE_MANIFEST, {
        "schema_version": "B1_TSB_Qualification_PreRun_Manifest_v1",
        "status": "B1_PRE_RUN_GATE_READY_FOR_COMMIT",
        "authorized_source_git_sha": authorized_source_git_sha,
        "artifacts": {f"docs/stageR/b1_tsb_qualification/{name}": sha(OUT / name) for name in pre_artifacts},
        "source_file_hashes": source_hashes,
        "protected_asset_sha256": sha(PROTECTED),
        "planned_pairs": TARGET_PAIRS,
        "planned_arms": MAX_ARMS,
        "all_pairs_not_run": all(row["execution_status"] == "NOT_RUN" for row in roster_rows),
        "e2_pairs": sum(row["exposure_class"] == "E2_BENCHMARK_ENGINEERING" for row in roster_rows),
        "e1_pairs": sum(row["exposure_class"] == "E1_UNRELATED_HISTORICAL_USE" for row in roster_rows),
        "e5_pairs": 0,
        "static_b1_context_incompatible_before_roster_freeze": sum(row["decision"] != "SELECTED" for row in selection_audit),
        "scientific_rollout_count": 0,
    })
    print(json.dumps({"status": "B1_PRE_RUN_GATE_READY_FOR_COMMIT", "pairs": 20, "arms": 40, "roster_sha256": sha(ROSTER)}, indent=2))


def verify_pre_run() -> Mapping[str, Any]:
    authorization = read_json(AUTH)
    manifest = read_json(PRE_MANIFEST)
    for relative, expected in manifest["artifacts"].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError(f"B1_PRE_RUN_ARTIFACT_HASH_MISMATCH:{relative}")
    for relative, expected in authorization["source_file_hashes"].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError(f"B1_SOURCE_HASH_MISMATCH:{relative}")
    if sha(PROTECTED) != PROTECTED_SHA:
        raise RuntimeError("B1_PROTECTED_ASSET_HASH_MISMATCH")
    with ROSTER.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 20 or len({row["session_id"] for row in rows}) != 20 or any(row["execution_status"] != "NOT_RUN" for row in rows):
        raise RuntimeError("B1_ROSTER_GATE_FAILED")
    return authorization


def execute() -> None:
    ensure_runtime()
    authorization = verify_pre_run()
    if RUN_ROOT.exists():
        raise RuntimeError(f"B1_OUTPUT_ROOT_ALREADY_EXISTS:{RUN_ROOT}")
    RUN_ROOT.mkdir(parents=True)
    log_path = RUN_ROOT / "B1_execution.log"
    schedule = sorted(authorization["allowed_runs"], key=lambda row: int(row["execution_order"]))
    for index, row in enumerate(schedule, 1):
        run_id = str(row["run_id"])
        print(f"B1 scientific arm {index}/{len(schedule)}: {run_id}", flush=True)
        command = [
            str(NU_PYTHON), "-B", str(EXECUTOR), "--execute-authorized-arm",
            str(AUTH), str(SPECS), str(ROSTER), run_id, str(RUN_ROOT),
        ]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n===== {index}/{len(schedule)} {run_id} exit={completed.returncode} =====\n")
            handle.write(completed.stdout)
            handle.write(completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError(f"B1_INFRASTRUCTURE_STOP:{run_id}:see {log_path}")
    print("B1 all 40 authorized arms completed; starting frozen pair adjudication", flush=True)
    finalize()


def _artifact_hashes(run_root: Path) -> Mapping[str, str]:
    paths = [run_root / "execution_manifest.json", run_root / "trace/realized_current_ego.jsonl"]
    paths.extend(sorted((run_root / "raw/metrics").glob("*.parquet")) if (run_root / "raw/metrics").is_dir() else [])
    return {str(path.relative_to(RUN_ROOT)): sha(path) for path in paths if path.is_file()}


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def finalize() -> None:
    ensure_runtime()
    from tools.r1_b2_8_r3_2_post_run_evaluator_dispatcher import _read_trace, evaluate_frozen_pair
    from tools.r1_closed_loop_benchmark_v2_1 import (
        calculate_tsb_option_a_v2_timestamp_aware,
        exact_realized_window_v1_1,
        trajectory_arrays_timestamp_aware,
    )

    specs = read_json(SPECS)
    spec_by_id = {row["run_id"]: row for row in specs["arms"]}
    bindings = {row["pair_id"]: row for row in read_json(BINDINGS)["pairs"]}
    with ROSTER.open(encoding="utf-8", newline="") as handle:
        roster = list(csv.DictReader(handle))
    budget_ledger = read_json(RUN_ROOT / "B1_Scientific_Arm_Budget_Ledger_v1.json")
    arm_budget_status = budget_ledger["runs"]
    execution_log = RUN_ROOT / "B1_execution.log"
    log_text = execution_log.read_text(encoding="utf-8") if execution_log.is_file() else ""
    exception_lines = [line.strip() for line in log_text.splitlines() if line.startswith(("ValueError:", "RuntimeError:"))]
    infrastructure_stop_reason = exception_lines[-1] if exception_lines else "RUNNER_EXCEPTION_SEE_B1_EXECUTION_LOG"
    results: list[dict[str, Any]] = []
    evaluation_dir = RUN_ROOT / "pair_evaluations"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(roster, 1):
        pair_id = row["pair_id"]
        baseline_id, treatment_id = row["baseline_arm_id"], row["treatment_arm_id"]
        baseline_root, treatment_root = RUN_ROOT / baseline_id, RUN_ROOT / treatment_id
        technical_reason = scientific_reason = ""
        status = "NOT_RUN"
        baseline_status = treatment_status = "NOT_RUN"
        precontext_match = route_match = False
        base_valid = treatment_valid = False
        base_phase = treatment_phase = ""
        release = second_peak = ""
        fmatch_status = safety_status = joint_mechanism = "NOT_RUN"
        low_speed = False
        try:
            if arm_budget_status.get(baseline_id) != "TECHNICAL_COMPLETE" or arm_budget_status.get(treatment_id) != "TECHNICAL_COMPLETE":
                baseline_status = "NOT_RUN" if arm_budget_status.get(baseline_id) == "NOT_RUN" else "TECHNICAL_INCOMPLETE"
                treatment_status = "NOT_RUN" if arm_budget_status.get(treatment_id) == "NOT_RUN" else "TECHNICAL_INCOMPLETE"
                detail = f"ARM_BUDGET_STATUS:BASELINE={arm_budget_status.get(baseline_id)},TREATMENT={arm_budget_status.get(treatment_id)}"
                if "TECHNICAL_INCOMPLETE" in {arm_budget_status.get(baseline_id), arm_budget_status.get(treatment_id)}:
                    detail += f";INFRASTRUCTURE_STOP={infrastructure_stop_reason}"
                raise RuntimeError(detail)
            base_manifest = read_json(baseline_root / "execution_manifest.json")
            treatment_manifest = read_json(treatment_root / "execution_manifest.json")
            baseline_status = base_manifest["execution_status"]
            treatment_status = treatment_manifest["execution_status"]
            if baseline_status != "TECHNICAL_COMPLETE_PENDING_PAIR_ANALYSIS" or treatment_status != "TECHNICAL_COMPLETE_PENDING_PAIR_ANALYSIS":
                raise RuntimeError("ARM_TECHNICAL_COMPLETENESS_FAILED")
            precontext_match = base_manifest["precontext_id"] == treatment_manifest["precontext_id"]
            route_match = base_manifest["route_id"] == treatment_manifest["route_id"]
            if not precontext_match or not route_match:
                status = "TECHNICAL_INCOMPLETE"
                technical_reason = "PAIR_INVALID_PRECONTEXT" if not precontext_match else "PAIR_INVALID_ROUTE"
            else:
                baseline_states = exact_realized_window_v1_1(_read_trace(baseline_root))
                treatment_states = exact_realized_window_v1_1(_read_trace(treatment_root))
                bt, _, _, bs = trajectory_arrays_timestamp_aware(baseline_states)
                tt, _, _, ts = trajectory_arrays_timestamp_aware(treatment_states)
                bm = calculate_tsb_option_a_v2_timestamp_aware(bt, bs)
                tm = calculate_tsb_option_a_v2_timestamp_aware(tt, ts)
                base_valid, treatment_valid = bm["status"] == "OK", tm["status"] == "OK"
                base_phase, treatment_phase = bm.get("brake_phase_count"), tm.get("brake_phase_count")
                release, second_peak = tm.get("interstage_release_fraction"), tm.get("second_brake_peak_ratio")
                low_speed = bm["status"] == "LOW_SPEED_ENDSTOP" or tm["status"] == "LOW_SPEED_ENDSTOP"
                evaluation = evaluate_frozen_pair(
                    pair_binding=bindings[pair_id], baseline_run_dir=baseline_root, treatment_run_dir=treatment_root
                )
                write_json(evaluation_dir / f"{pair_id}__evaluation.json", evaluation)
                fmatch_status = evaluation["evaluation"]["f_match"]["status"]
                safety_status = "PASS" if evaluation["official_safety_pair_pass"] else "SCIENTIFIC_FAIL"
                joint_mechanism = "PASS" if evaluation["evaluation"]["mechanism"]["pass"] else "SCIENTIFIC_FAIL"
                reasons = list(evaluation["evaluation"]["mechanism"].get("reasons", []))
                if not evaluation["evaluation"]["f_match"]["pass"]:
                    reasons.append("F_MATCH_FAIL:" + ",".join(key for key, value in evaluation["evaluation"]["f_match"]["pass_by_feature"].items() if not value))
                if not evaluation["official_safety_pair_pass"]:
                    reasons.append("OFFICIAL_SAFETY_FAIL")
                if low_speed:
                    reasons.append("LOW_SPEED_ENDSTOP")
                if base_valid and treatment_valid and not reasons:
                    status = "PASS"
                elif not base_valid or not treatment_valid:
                    status = "SCIENTIFIC_FAIL" if low_speed or bm["status"] in {"NO_BRAKE_PHASE"} or tm["status"] in {"NO_BRAKE_PHASE"} else "MEASUREMENT_INVALID"
                    reasons.extend([f"BASELINE_MEASUREMENT:{bm['status']}", f"TREATMENT_MEASUREMENT:{tm['status']}"])
                else:
                    status = "SCIENTIFIC_FAIL"
                scientific_reason = ";".join(dict.fromkeys(reasons))
        except (FileNotFoundError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
            if not technical_reason:
                status = "TECHNICAL_INCOMPLETE"
                technical_reason = f"{type(exc).__name__}:{exc}"
        except ValueError as exc:
            status = "MEASUREMENT_INVALID" if "NOT_EVALUABLE" in str(exc) else "TECHNICAL_INCOMPLETE"
            if status == "MEASUREMENT_INVALID":
                scientific_reason = f"ValueError:{exc}"
            else:
                technical_reason = f"ValueError:{exc}"
        results.append({
            "pair_id": pair_id,
            "session_id": row["session_id"],
            "log_id": row["log_id"],
            "scenario_token": row["scenario_token"],
            "exposure_class": row["exposure_class"],
            "baseline_arm_status": baseline_status,
            "treatment_arm_status": treatment_status,
            "precontext_match": str(precontext_match).lower(),
            "route_match": str(route_match).lower(),
            "baseline_measurement_valid": str(base_valid).lower(),
            "treatment_measurement_valid": str(treatment_valid).lower(),
            "baseline_phase_count": base_phase,
            "treatment_phase_count": treatment_phase,
            "release_fraction": release,
            "second_peak_ratio": second_peak,
            "F_match_status": fmatch_status,
            "safety_status": safety_status,
            "LOW_SPEED_ENDSTOP": str(low_speed).lower(),
            "joint_mechanism_status": joint_mechanism,
            "joint_scientific_status": status,
            "technical_failure_reason": technical_reason,
            "scientific_failure_reason": scientific_reason,
            "artifact_hashes": canonical({"baseline": _artifact_hashes(baseline_root), "treatment": _artifact_hashes(treatment_root)}),
        })
        print(f"B1 pair adjudication {index}/{len(roster)}: {pair_id}={status}", flush=True)

    pair_results_path = OUT / "B1_TSB_Qualification_Pair_Results_v1.csv"
    write_csv(pair_results_path, results)
    counts = Counter(row["joint_scientific_status"] for row in results)
    technical_complete = sum(row["baseline_arm_status"] == row["treatment_arm_status"] == "TECHNICAL_COMPLETE_PENDING_PAIR_ANALYSIS" for row in results)
    evaluable = sum(row["joint_scientific_status"] in {"PASS", "SCIENTIFIC_FAIL"} for row in results)
    baseline_success = sum(row["baseline_measurement_valid"] == "true" and str(row["baseline_phase_count"]) == "1" for row in results)
    treatment_success = sum(row["treatment_measurement_valid"] == "true" and str(row["treatment_phase_count"]) == "2" and float(row["release_fraction"] or -math.inf) >= 0.15 and float(row["second_peak_ratio"] or -math.inf) >= 0.50 for row in results)
    mechanism_success = sum(row["joint_mechanism_status"] == "PASS" for row in results)
    fmatch_success = sum(row["F_match_status"] == "F_MATCH_PASS" for row in results)
    safety_success = sum(row["safety_status"] == "PASS" for row in results)
    low_speed_count = sum(row["LOW_SPEED_ENDSTOP"] == "true" for row in results)
    joint_success = counts["PASS"]
    interval = _wilson(joint_success, TARGET_PAIRS)
    if counts["TECHNICAL_INCOMPLETE"] or len(results) < TARGET_PAIRS:
        main_status = "B1_TSB_BENCHMARK_QUALIFICATION_INCOMPLETE_INFRASTRUCTURE"
    elif joint_success == TARGET_PAIRS:
        main_status = "B1_TSB_BENCHMARK_QUALIFICATION_PASS"
    else:
        main_status = "B1_TSB_BENCHMARK_QUALIFICATION_FAIL"
    summary = {
        "schema_version": "B1_TSB_Qualification_Summary_v1",
        "main_status": main_status,
        "planned_session_pairs": TARGET_PAIRS,
        "attempted_scientific_arms": sum(value != "NOT_RUN" for value in arm_budget_status.values()),
        "technical_complete_arms": sum(value == "TECHNICAL_COMPLETE" for value in arm_budget_status.values()),
        "executed_pairs": sum(row["baseline_arm_status"] != "NOT_RUN" or row["treatment_arm_status"] != "NOT_RUN" for row in results),
        "technical_complete_pairs": technical_complete,
        "scientifically_evaluable_pairs": evaluable,
        "baseline_one_phase_success": baseline_success,
        "treatment_two_stage_success": treatment_success,
        "joint_mechanism_success": mechanism_success,
        "f_match_pass": fmatch_success,
        "official_safety_pass": safety_success,
        "low_speed_endstop_count": low_speed_count,
        "joint_scientific_qualification": joint_success,
        "joint_scientific_wilson_95_interval": [round(interval[0], 6), round(interval[1], 6)],
        "success_rule": "ALL_20_PAIRS_PASS_FROZEN_JOINT_QUALIFICATION",
        "success_rule_satisfied": joint_success == TARGET_PAIRS,
        "status_counts": dict(sorted(counts.items())),
        "low_order_nuisance_eliminated": "NOT_ESTABLISHED",
        "tsb_clean_residual_task": "NOT_ESTABLISHED",
        "rbr_h_bdd_inspected": False,
        "rbr_training_authorized": False,
        "primary_evaluation_authorized": False,
    }
    summary_path = OUT / "B1_TSB_Qualification_Summary_v1.json"
    write_json(summary_path, summary)
    failure_lines = [f"- `{row['pair_id']}` — `{row['joint_scientific_status']}`: {row['technical_failure_reason'] or row['scientific_failure_reason']}" for row in results if row["joint_scientific_status"] != "PASS"]
    exposure_counts = Counter(row["exposure_class"] for row in roster)
    report = f"""# B1 TSB Qualification Execution Report v1

Main status: `{main_status}`.

The frozen B1 roster contained {TARGET_PAIRS} independent SESSION pairs and {MAX_ARMS} authorized arms. Attempted arms: {summary['attempted_scientific_arms']}; technically complete arms: {summary['technical_complete_arms']}. Executed pairs: {summary['executed_pairs']}; technical complete pairs: {technical_complete}; scientifically evaluable pairs: {evaluable}.

- Baseline one-phase success: `{baseline_success}/{TARGET_PAIRS}`
- Treatment two-stage success: `{treatment_success}/{TARGET_PAIRS}`
- Joint mechanism success: `{mechanism_success}/{TARGET_PAIRS}`
- F_match pass: `{fmatch_success}/{TARGET_PAIRS}`
- Official safety pass: `{safety_success}/{TARGET_PAIRS}`
- LOW_SPEED_ENDSTOP: `{low_speed_count}`
- Joint scientific qualification: `{joint_success}/{TARGET_PAIRS}`, descriptive Wilson 95% interval `[{interval[0]:.6f}, {interval[1]:.6f}]`
- Frozen all-20 success rule satisfied: `{str(joint_success == TARGET_PAIRS).upper()}`

Failure ledger:
{chr(10).join(failure_lines) if failure_lines else '- None.'}

B1 exposure was E2={exposure_counts['E2_BENCHMARK_ENGINEERING']}, E1={exposure_counts['E1_UNRELATED_HISTORICAL_USE']}, E5=0. RBR, H, BDD, z64, MMD, detector performance, and Primary comparison were not read or computed. `LOW_ORDER_NUISANCE_ELIMINATED = NOT_ESTABLISHED`; `TSB_CLEAN_RESIDUAL_TASK = NOT_ESTABLISHED`.
"""
    report_path = OUT / "B1_TSB_Qualification_Execution_Report_v1.md"
    report_path.write_text(report, encoding="utf-8")
    archive = OUT / "execution_manifests"
    archive.mkdir(parents=True, exist_ok=True)
    for row in specs["arms"]:
        source = RUN_ROOT / row["run_id"] / "execution_manifest.json"
        target = archive / f"{row['run_id']}.json"
        if source.is_file():
            shutil.copyfile(source, target)
        else:
            write_json(target, {
                "schema_version": "b1_tsb_unstarted_arm_record_v1",
                "stage": "B1_FROZEN_TSB_QUALIFICATION",
                "run_id": row["run_id"],
                "pair_id": row["pair_id"],
                "arm": row["arm"],
                "session_id": row["session_id"],
                "log_id": row["log_id"],
                "scenario_token": row["scenario_token"],
                "exposure_class": row["exposure_class"],
                "role": row["role"],
                "precontext_id": row["precontext_id"],
                "route_id": row["route_id"],
                "execution_status": "NOT_RUN",
                "reason": "NOT_RUN_AFTER_INFRASTRUCTURE_STOP",
            })
    stop_record = OUT / "B1_Infrastructure_Stop_Record_v1.json"
    write_json(stop_record, {
        "schema_version": "B1_Infrastructure_Stop_Record_v1",
        "status": "INFRASTRUCTURE_STOP_NO_RETRY",
        "failing_run_id": next((run_id for run_id, value in arm_budget_status.items() if value == "TECHNICAL_INCOMPLETE"), None),
        "reason": infrastructure_stop_reason,
        "attempted_scientific_arms": summary["attempted_scientific_arms"],
        "technical_complete_arms": summary["technical_complete_arms"],
        "not_run_arms": sum(value == "NOT_RUN" for value in arm_budget_status.values()),
        "execution_log_sha256": sha(execution_log),
        "retry_performed": False,
        "replacement_performed": False,
        "offline_finalization_only": True,
    })
    result_artifacts = [pair_results_path, summary_path, report_path, stop_record, *sorted(archive.glob("*.json"))]
    write_json(OUT / "B1_TSB_Qualification_Manifest_v1.json", {
        "schema_version": "B1_TSB_Qualification_Manifest_v1",
        "main_status": main_status,
        "pre_run_manifest_sha256": sha(PRE_MANIFEST),
        "authorization_sha256": sha(AUTH),
        "roster_sha256": sha(ROSTER),
        "pair_results_sha256": sha(pair_results_path),
        "summary_sha256": sha(summary_path),
        "execution_report_sha256": sha(report_path),
        "execution_manifest_hashes": {path.name: sha(path) for path in sorted(archive.glob("*.json"))},
        "result_artifacts": {str(path.relative_to(ROOT)): sha(path) for path in result_artifacts},
        "protected_asset_sha256": sha(PROTECTED),
        "scientific_arm_budget_ledger_sha256": sha(RUN_ROOT / "B1_Scientific_Arm_Budget_Ledger_v1.json"),
        "pair_evaluation_hashes": {path.name: sha(path) for path in sorted(evaluation_dir.glob("*.json"))},
        "rbr_h_bdd_inspected": False,
        "authorizations_after_b1": {
            "RBR_TRAINING": "NOT_AUTHORIZED",
            "PRIMARY_EVALUATION": "NOT_AUTHORIZED",
            "V_EXECUTION": "NOT_AUTHORIZED",
            "C_EXECUTION": "NOT_AUTHORIZED",
        },
    })
    print(json.dumps(summary, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--authorized-source-git-sha", required=True)
    sub.add_parser("verify-pre-run")
    sub.add_parser("execute")
    sub.add_parser("finalize")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.authorized_source_git_sha)
    elif args.command == "verify-pre-run":
        verify_pre_run()
        print("B1_PRE_RUN_GATE=PASS")
    elif args.command == "execute":
        execute()
    else:
        finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
