#!/usr/bin/env python3
"""Pre-outcome frozen analyzer for the only BJ-B0 two-arm canary pair."""

from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tools.r1_b2_8_r3_2_post_run_evaluator_dispatcher import evaluate_frozen_pair
from tools.r1_closed_loop_benchmark_v2_1 import trajectory_arrays_timestamp_aware


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "docs/stageR/r2"
BINDINGS = R2 / "r2_bj_b0_exact_pair_binding_manifest_v1.0.json"
CENSUS = R2 / "r2_bj_a5_557_entry_eligibility_census_ledger_v1.0.json"
PARAMETERS = R2 / "r2_bj_a_hlc_global_parameter_space_v4.0.json"
B0_COMPONENT = R2 / "r2_bj_b0_component_sha_binding_manifest_v1.0.json"
B01_COMPONENT = R2 / "r2_bj_b0_1_execution_component_sha_manifest_v1.0.json"
B02_COMPONENT = R2 / "r2_bj_b0_2_execution_observability_sha_manifest_v1.0.json"
PAIR_ID = "R2BJB0-HLC-01"
EXPECTED_RUN_IDS = ("R2BJB0-HLC-01-BASELINE", "R2BJB0-HLC-01-TREATMENT")
SHADOW_TOLERANCE = 1e-12
POST_DEADLINE_LATERAL_JUMP_LIMIT_M = 0.25
RESULT_STATES = (
    "R2_BJ_B1_CANARY_ARCHITECTURE_FAILURE_STOPPED",
    "R2_BJ_B1_CANARY_INFRASTRUCTURE_FAILURE_STOPPED",
    "R2_BJ_B1_CANARY_TECHNICAL_COMPLETE_MECHANISM_OR_ENDPOINT_FAIL",
    "R2_BJ_B1_CANARY_COMPLETE_READY_FOR_REMAINING_COHORT_OWNER_REVIEW",
)


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _jsonl(path: Path, expected: int) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected:
        raise ValueError(f"CARDINALITY:{path.name}:{len(rows)}!={expected}")
    return rows


def _decode(value: Mapping[str, Any]) -> list[list[float]]:
    raw = base64.b64decode(value["base64"])
    array = np.frombuffer(raw, dtype=np.dtype(value["dtype"])).reshape(value["shape"])
    if sha(array.tobytes(order="C")) != value["sha256"] or not np.isfinite(array).all():
        raise ValueError("REFERENCE_GEOMETRY_SHA_OR_FINITE_FAILURE")
    return array.astype(float).tolist()


def frozen_analysis_binding() -> Mapping[str, Any]:
    document = read(BINDINGS)
    rows = document.get("pair_bindings", document.get("bindings", []))
    matches = [row for row in rows if row["pair_id"] == PAIR_ID]
    if len(matches) != 1:
        raise ValueError("B0_PAIR_BINDING_NOT_EXACTLY_ONE")
    original = matches[0]
    shared = original["shared_binding"]
    index = int(shared["reference_geometry_locator"]["census_index"])
    census_matches = [row for row in read(CENSUS)["entries"] if int(row["census_index"]) == index]
    if len(census_matches) != 1:
        raise ValueError("A5_REFERENCE_LOCATOR_NOT_EXACTLY_ONE")
    reference = census_matches[0]["predicate_result"]["closure"]["reference_geometry"]
    source, target = _decode(reference["source"]), _decode(reference["target"])
    if sha(np.asarray(source, dtype=np.float64).tobytes()) != shared["source_reference_sha256"]:
        raise ValueError("SOURCE_REFERENCE_SHA_MISMATCH")
    if sha(np.asarray(target, dtype=np.float64).tobytes()) != shared["target_reference_sha256"]:
        raise ValueError("TARGET_REFERENCE_SHA_MISMATCH")
    context_hash = shared["shared_binding_canonical_sha256"] if "shared_binding_canonical_sha256" in shared else original["shared_binding_canonical_sha256"]
    context = {"pre_context_raw_hash": context_hash, "canonical_context_json_hash": context_hash}
    return {
        "pair_id": PAIR_ID,
        "family": "R-HLC",
        "scenario_token": shared["scenario_token"],
        "log_id": shared["log_id"],
        "baseline_context": context,
        "treatment_context": dict(context),
        "pretreatment_clearance": {
            "status": "FROZEN_B0_OUTCOME_BLIND_TECHNICAL_APPLICABILITY_PASS",
            "eligible": True,
            "pass": True,
            "pretreatment_only": True,
            "posthoc_recalculation_forbidden": True,
            "source": "B0_EXACT_PAIR_BINDING_AND_A5_APPLICABILITY_CLOSURE",
        },
        "source_reference_xy": source,
        "target_reference_xy": target,
        "native_route_reference_xy": source,
        "native_route_reference_source": "A5_V2_3_ROUTE_CONTINUOUS_SOURCE_REFERENCE_PRE_OUTCOME",
        "baseline_run_id": original["baseline_run_id"],
        "treatment_run_id": original["treatment_run_id"],
        "binding_source_sha256": hashlib.sha256(BINDINGS.read_bytes()).hexdigest(),
        "census_source_sha256": hashlib.sha256(CENSUS.read_bytes()).hexdigest(),
    }


def _signed_offsets(points: np.ndarray, reference: Sequence[Sequence[float]]) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float64)
    starts, vectors = ref[:-1], np.diff(ref, axis=0)
    denom = np.sum(vectors * vectors, axis=1)
    result = []
    for point in points:
        u = np.clip(np.sum((point - starts) * vectors, axis=1) / np.maximum(denom, 1e-12), 0.0, 1.0)
        projected = starts + u[:, None] * vectors
        index = int(np.argmin(np.sum((point - projected) ** 2, axis=1)))
        tangent = vectors[index] / max(float(np.linalg.norm(vectors[index])), 1e-12)
        result.append(float(np.dot(point - projected[index], np.asarray([-tangent[1], tangent[0]]))))
    return np.asarray(result)


def _arm_audit(run_root: Path, run: Mapping[str, Any], binding: Mapping[str, Any]) -> Mapping[str, Any]:
    trace_rows = _jsonl(run_root / "trace/realized_current_ego.jsonl", 80)
    planner_rows = _jsonl(run_root / "telemetry/planner_v4_online_gate.jsonl", 80)
    planner_reference_rows = _jsonl(run_root / "telemetry/controller_visible_telemetry.jsonl", 80)
    controller_rows = _jsonl(run_root / "telemetry/actual_lqr_controller_telemetry.jsonl", 79)
    if [int(row["iteration"]) for row in controller_rows] != list(range(79)):
        raise ValueError("ACTUAL_CONTROLLER_ITERATIONS_NOT_0_78")
    if any(row.get("behavior_changed") is not False for row in controller_rows):
        raise ValueError("PASSIVE_RECORDER_BEHAVIOR_FLAG_INVALID")
    expected_shas = {
        "B0_component_manifest_sha256": hashlib.sha256(B0_COMPONENT.read_bytes()).hexdigest(),
        "B0_1_execution_component_manifest_sha256": hashlib.sha256(B01_COMPONENT.read_bytes()).hexdigest(),
        "B0_2_execution_observability_manifest_sha256": hashlib.sha256(B02_COMPONENT.read_bytes()).hexdigest(),
    }
    for row in controller_rows:
        if any(str(row.get(key)) != str(run[key]) for key in ("run_id", "pair_id", "arm")):
            raise ValueError("ACTUAL_CONTROLLER_RUN_PAIR_ARM_PROVENANCE_MISMATCH")
        if row.get("component_sha256") != expected_shas:
            raise ValueError("ACTUAL_CONTROLLER_COMPONENT_SHA_PROVENANCE_MISMATCH")
    values = [
        float(row[key])
        for row in controller_rows
        for key in ("actual_acceleration_command_mps2", "actual_tire_steering_rate_command_radps", "shadow_acceleration_command_mps2", "shadow_tire_steering_rate_command_radps")
    ]
    if not np.isfinite(values).all():
        raise ValueError("ACTUAL_CONTROLLER_NONFINITE")
    direction_agreement = sum(
        bool(row["acceleration_direction_agreement"] and row["steering_direction_agreement"])
        for row in controller_rows
    )
    max_accel_diff = max(float(row["absolute_acceleration_difference_mps2"]) for row in controller_rows)
    max_steer_diff = max(float(row["absolute_steering_rate_difference_radps"]) for row in controller_rows)
    exact_shadow = max(max_accel_diff, max_steer_diff) <= SHADOW_TOLERANCE

    states = [row.get("current_ego", row) for row in trace_rows]
    times, xy, _, _ = trajectory_arrays_timestamp_aware(states)
    absolute = times - times[0]
    offsets = np.abs(_signed_offsets(xy, binding["target_reference_xy"]))
    capture = read(PARAMETERS)["global_parameters"]["capture"]
    start_s, deadline_s = float(capture["capture_start_abs_s"]), float(capture["capture_end_abs_s"])
    start_index = int(np.argmin(np.abs(absolute - start_s)))
    lateral_steps = []
    post_deadline_steps = []
    for row in planner_rows:
        lookahead = row["controller_lookahead"]["states_0_to_10"]
        p0, p1 = lookahead[0]["rear_axle"], lookahead[1]["rear_axle"]
        heading = float(p0["heading"])
        dx, dy = float(p1["x"] - p0["x"]), float(p1["y"] - p0["y"])
        lateral = abs(-math.sin(heading) * dx + math.cos(heading) * dy)
        lateral_steps.append(lateral)
        if float(row["absolute_episode_time_s"]) >= deadline_s - 1e-9:
            post_deadline_steps.append(lateral)
    return {
        "trace_rows": len(trace_rows),
        "planner_gate_rows": len(planner_rows),
        "planner_reference_steering": {
            "rows": len(planner_reference_rows),
            "semantic_role": "PLANNER_REFERENCE_STEERING",
            "actual_controller_command": False,
            "disposition": "NOT_ACTUAL_CONTROLLER_COMMAND",
        },
        "actual_lqr_rows": len(controller_rows),
        "actual_shadow_direction_agreement_rows": direction_agreement,
        "actual_shadow_exact_tolerance_pass": exact_shadow,
        "actual_shadow_max_acceleration_difference_mps2": max_accel_diff,
        "actual_shadow_max_steering_difference_radps": max_steer_diff,
        "realized_target_frame_abs_offset_at_capture_start_m": float(offsets[start_index]),
        "realized_target_frame_abs_offset_terminal_m": float(offsets[-1]),
        "realized_treatment_offset_declines": bool(offsets[-1] < offsets[start_index] - 1e-9),
        "post_deadline_hard_jump": bool(post_deadline_steps and max(post_deadline_steps) > POST_DEADLINE_LATERAL_JUMP_LIMIT_M),
        "maximum_state0_to_state1_lateral_step_m": max(lateral_steps),
    }


def _technical_artifacts(run_root: Path) -> None:
    if (run_root / "telemetry/architecture_failure_audit.json").exists():
        raise RuntimeError("ARCHITECTURE_FAILURE_AUDIT_PRESENT")
    for name in ("no_ego_at_fault_collisions.parquet", "drivable_area_compliance.parquet"):
        if len(list(run_root.rglob(name))) != 1:
            raise ValueError(f"OFFICIAL_SAFETY_ARTIFACT_NOT_EXACTLY_ONE:{name}")
    reports = [path for path in run_root.rglob("*.parquet") if "runner_report" in path.name]
    if len(reports) != 1:
        raise ValueError("RUNNER_REPORT_NOT_EXACTLY_ONE")


def analyze_frozen_canary_pair(
    output_root: Path,
    runs: Sequence[Mapping[str, Any]],
    evaluator: Any = evaluate_frozen_pair,
) -> Mapping[str, Any]:
    """Evaluate the frozen pair twice and fail closed if repeat output differs."""
    try:
        if tuple(str(row["run_id"]) for row in runs) != EXPECTED_RUN_IDS:
            raise ValueError("RUN_ORDER_OR_IDS_MISMATCH")
        if any(str(row["pair_id"]) != PAIR_ID for row in runs):
            raise ValueError("PAIR_ID_MISMATCH")
        if any(str(row["scenario_token"]) != "cc1abd3989065d8d" for row in runs):
            raise ValueError("SCENARIO_TOKEN_MISMATCH")
        if any(str(row["log_id"]) != "2021.10.01.16.53.37_veh-44_01126_01602" for row in runs):
            raise ValueError("LOG_ID_MISMATCH")
        roots = [Path(output_root) / str(row["run_id"]) for row in runs]
        if any((root / "telemetry/architecture_failure_audit.json").exists() for root in roots):
            return {"result_state": RESULT_STATES[0], "reason": "PERSISTED_ARCHITECTURE_FAILURE", "ordinary_failure": False}
        for root in roots:
            _technical_artifacts(root)
        binding = frozen_analysis_binding()
        arm_audits = {str(run["arm"]).lower(): _arm_audit(root, run, binding) for run, root in zip(runs, roots)}
        first = evaluator(pair_binding=binding, baseline_run_dir=roots[0], treatment_run_dir=roots[1])
        second = evaluator(pair_binding=binding, baseline_run_dir=roots[0], treatment_run_dir=roots[1])
        if canonical(first) != canonical(second):
            raise ValueError("NONDETERMINISTIC_REPEAT_ANALYSIS_OUTPUT")
        evaluation = first["evaluation"]
        endpoint = evaluation["endpoint"]
        engineering = evaluation["engineering"]
        mechanism_pass = bool(evaluation["mechanism"]["pass"])
        endpoint_pass = bool(endpoint["baseline"]["pass"] and endpoint["treatment"]["pass"])
        fmatch_pass = bool(evaluation["f_match"]["pass"])
        engineering_pass = all(
            arm["max_abs_lateral_accel_mps2"] <= arm["frozen_limits"]["lateral_accel_mps2_max"]
            and arm["max_abs_yaw_rate_radps"] <= arm["frozen_limits"]["yaw_rate_radps_max"]
            and arm["max_abs_curvature_inv_m"] <= arm["frozen_limits"]["curvature_inv_m_max"]
            for arm in engineering.values()
        )
        safety_pass = bool(first["official_safety_pair_pass"])
        observability_pass = all(
            arm_audits[arm]["actual_lqr_rows"] == 79
            and arm_audits[arm]["actual_shadow_direction_agreement_rows"] == 79
            and arm_audits[arm]["actual_shadow_exact_tolerance_pass"]
            for arm in ("baseline", "treatment")
        )
        capture_pass = bool(arm_audits["treatment"]["realized_treatment_offset_declines"])
        hard_jump_absent = not any(arm_audits[arm]["post_deadline_hard_jump"] for arm in arm_audits)
        ready = all((mechanism_pass, endpoint_pass, fmatch_pass, engineering_pass, safety_pass, observability_pass, capture_pass, hard_jump_absent))
        state = RESULT_STATES[3] if ready else RESULT_STATES[2]
        return {
            "schema_version": "r2_bj_b0_2_frozen_canary_pair_analysis_v1.0",
            "result_state": state,
            "technical_complete": True,
            "pair_id": PAIR_ID,
            "gates": {
                "mechanism_pass": mechanism_pass,
                "endpoint_pass": endpoint_pass,
                "F_match_pass": fmatch_pass,
                "engineering_pass": engineering_pass,
                "official_safety_pass": safety_pass,
                "actual_shadow_observability_pass": observability_pass,
                "treatment_target_offset_declines": capture_pass,
                "post_deadline_hard_jump_absent": hard_jump_absent,
            },
            "arm_audits": arm_audits,
            "frozen_evaluation": first,
            "deterministic_repeat_output": True,
            "remaining_14_runs_automatically_authorized": False,
        }
    except Exception as error:
        return {
            "schema_version": "r2_bj_b0_2_frozen_canary_pair_analysis_v1.0",
            "result_state": RESULT_STATES[1],
            "technical_complete": False,
            "reason": f"{type(error).__name__}:{error}",
            "remaining_14_runs_automatically_authorized": False,
        }


__all__ = ["RESULT_STATES", "analyze_frozen_canary_pair", "frozen_analysis_binding"]
