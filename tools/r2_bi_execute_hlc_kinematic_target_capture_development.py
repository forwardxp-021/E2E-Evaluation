#!/usr/bin/env python3
"""Conditionally execute at most two frozen R2-BI HLC DEV-KIN rounds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_b2_8_r3_2_post_run_evaluator_dispatcher import evaluate_frozen_pair  # noqa: E402
from tools.r1_b2_8_r3_prospective_selector import official_count, official_env  # noqa: E402
from tools.r1_b2_9_e_official_run_lifecycle import run_one_with_full_nuplan_lifecycle  # noqa: E402
from tools.r1_closed_loop_benchmark_v2_1 import calculate_hlc_option_b_v2_timestamp_aware, trajectory_arrays_timestamp_aware  # noqa: E402
from tools.r1_hlc_measurement_conformance_v1 import hlc_realized_lane_transition_progress_v1_0  # noqa: E402
from tools.r2_bi_hlc_kinematic_target_capture_generator_v3 import (  # noqa: E402
    ARM_BASELINE, ARM_TREATMENT, validate_parameters,
)
from tools.r2_bi_hlc_kinematic_target_capture_planner_v3 import R2BIHLCKinematicTargetCapturePlannerV3  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "docs/stageR/r2"
ROSTER = R2 / "r2_bi_hlc_dev_kin_roster_v1.0.json"
PAIRS = R2 / "r2_bi_hlc_dev_kin_pair_bindings_v1.0.json"
SPACE = R2 / "r2_bi_hlc_kinematic_capture_parameter_space_v3.0.json"
CONTRACT = R2 / "r2_bi_hlc_kinematic_capture_architecture_contract_v3.0.json"
TAXONOMY = R2 / "r2_bi_hlc_architecture_failure_taxonomy_v1.0.json"
ENTRY = R2 / "r2_bi_mandatory_zero_run_entry_gate_audit_v1.json"
LEDGER = R2 / "r2_bi_hlc_dev_kin_run_ledger_v1.0.json"
AUTHORIZATION = R2 / "r2_bi_scientific_owner_engineering_authorization_v1.0.json"
ROUND_DIR = R2 / "r2_bi_hlc_dev_kin_rounds"
PROTECTED = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"
PROTECTED_SHA = "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8"


def read(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Mapping[str, Any], update: bool = False) -> None:
    if path.exists() and not update:
        raise FileExistsError(f"R2_BI_VERSIONED_OUTPUT_EXISTS:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def build_planner_from_environment(
    run_id: str, trace_dir: str, telemetry_dir: str, parameter_file: str,
) -> R2BIHLCKinematicTargetCapturePlannerV3:
    payload = read(Path(parameter_file))
    matches = [row for row in payload["runs"] if row["run_id"] == run_id]
    if len(matches) != 1:
        raise ValueError(f"R2_BI_ENV_RUN_MATCH_COUNT_NOT_ONE:{run_id}:{len(matches)}")
    run = matches[0]
    entries = [row for row in read(ROSTER)["entries"] if row["scenario_token"] == run["scenario_token"]]
    if len(entries) != 1:
        raise ValueError("R2_BI_ROSTER_LOOKUP_NOT_ONE")
    return R2BIHLCKinematicTargetCapturePlannerV3(entries[0], run["arm"], payload["parameters"], trace_dir, telemetry_dir)


class ControllerCommandRecorderV2:
    """Passively record actual LQR return and an independently recomputed exact frozen shadow."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.rows = 0

    def install(self, controller: Any) -> None:
        tracker = getattr(controller, "_tracker", None)
        if tracker is None or tracker.__class__.__name__ != "LQRTracker":
            raise TypeError(f"R2_BI_EXPECTED_LQR_TRACKER:{type(tracker).__name__}")
        original = tracker.track_trajectory
        recorder = self

        def wrapped(_self: Any, current_iteration: Any, next_iteration: Any, initial_state: Any, trajectory: Any) -> Any:
            initial_velocity, initial_lateral = tracker._compute_initial_velocity_and_lateral_state(
                current_iteration, initial_state, trajectory
            )
            reference_velocity, curvature = tracker._compute_reference_velocity_and_curvature_profile(
                current_iteration, trajectory
            )
            result = original(current_iteration, next_iteration, initial_state, trajectory)
            should_stop = reference_velocity <= tracker._stopping_velocity and initial_velocity <= tracker._stopping_velocity
            if should_stop:
                shadow = 0.0
            else:
                acceleration = tracker._longitudinal_lqr_controller(initial_velocity, reference_velocity)
                from nuplan.planning.simulation.controller.tracker.tracker_utils import _generate_profile_from_initial_condition_and_derivatives
                velocity_profile = _generate_profile_from_initial_condition_and_derivatives(
                    initial_condition=initial_velocity,
                    derivatives=np.ones(tracker._tracking_horizon, dtype=np.float64) * acceleration,
                    discretization_time=tracker._discretization_time,
                )[: tracker._tracking_horizon]
                shadow = tracker._lateral_lqr_controller(initial_lateral, velocity_profile, curvature)
            row = {
                "schema_version": "r2_bi_two_stage_lqr_actual_and_shadow_v1.0",
                "instrumentation": "PASSIVE_RETURN_WRAPPER_PLUS_EXACT_FROZEN_LQR_RECOMPUTATION",
                "iteration": int(current_iteration.index),
                "actual_steering_rate_command_radps": float(result.tire_steering_rate),
                "shadow_steering_rate_command_radps": float(shadow),
                "initial_lateral_state": [float(value) for value in initial_lateral],
                "reference_curvature_profile": [float(value) for value in curvature],
                "direction_agreement": bool(
                    abs(float(result.tire_steering_rate)) <= 1e-12 and abs(float(shadow)) <= 1e-12
                    or np.sign(float(result.tire_steering_rate)) == np.sign(float(shadow))
                ),
                "absolute_command_difference_radps": abs(float(result.tire_steering_rate) - float(shadow)),
            }
            recorder.output_path.parent.mkdir(parents=True, exist_ok=True)
            with recorder.output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
            recorder.rows += 1
            return result

        tracker.track_trajectory = types.MethodType(wrapped, tracker)


def _overrides(run: Mapping[str, Any], entry: Mapping[str, Any], raw: Path) -> list[str]:
    return [
        "+simulation=closed_loop_nonreactive_agents", "planner=r2_bi_hlc_kinematic_target_capture_dev_v3",
        "scenario_builder=nuplan_mini", f"scenario_builder.db_files=[{entry['db_path']}]",
        "scenario_filter=all_scenarios", f"scenario_filter.scenario_tokens=[{run['scenario_token']}]",
        "simulation_time_controller._target_=tools.r1_primary80_scientific_time_controller_v1.R1Primary80ScientificTimeControllerV1",
        "worker=sequential", "disable_callback_parallelization=true", "scenario_builder.max_workers=1",
        "max_callback_workers=1", "number_of_cpus_allocated_per_simulation=1",
        "number_of_gpus_allocated_per_simulation=0", "gpu=false", "seed=2026090301",
        "run_metric=true", "enable_simulation_progress_bar=false",
        "experiment_name=r2_bi_hlc_kinematic_target_capture_dev", f"job_name={run['run_id']}", f"output_dir={raw}",
        f"hydra.searchpath=[file://{ROOT}/configs/r1_official_technical_smoke_hydra,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]",
    ]


def _execute_one(
    run: Mapping[str, Any], entry: Mapping[str, Any], parameters: Mapping[str, Any],
    parameter_file: Path, output_root: Path,
) -> Dict[str, Any]:
    official_env()
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from nuplan.planning.script.builders.simulation_callback_builder import build_callbacks_worker, build_simulation_callbacks
    from nuplan.planning.script.builders.simulation_builder import build_simulations
    from nuplan.planning.script.utils import set_up_common_builder
    from tools.r1_primary80_scientific_time_controller_v1 import R1Primary80ScientificTimeControllerV1

    run_root = output_root / run["run_id"]
    trace_dir, telemetry_dir, raw = run_root / "trace", run_root / "telemetry", run_root / "raw"
    if run_root.exists():
        raise FileExistsError(f"R2_BI_FRESH_RUN_ROOT_REQUIRED:{run_root}")
    if official_count(entry["db_path"], run["scenario_token"]) != 1:
        raise RuntimeError(f"R2_BI_EXACT_SCENARIO_RESOLUTION_NOT_ONE:{run['run_id']}")
    trace_dir.mkdir(parents=True)
    planner = R2BIHLCKinematicTargetCapturePlannerV3(entry, run["arm"], parameters, str(trace_dir), str(telemetry_dir))
    os.environ.update({
        "R2_BI_RUN_ID": run["run_id"], "R2_BI_TRACE_DIR": str(trace_dir),
        "R2_BI_TELEMETRY_DIR": str(telemetry_dir), "R2_BI_PARAMETER_FILE": str(parameter_file),
    })
    config_root = ROOT.parent / "nuplan-devkit/nuplan/planning/script/config/simulation"
    with initialize_config_dir(config_dir=str(config_root)):
        cfg = compose(config_name="default_simulation", overrides=_overrides(run, entry, raw))
    if "${" in json.dumps(OmegaConf.to_container(cfg, resolve=True), sort_keys=True):
        raise RuntimeError("R2_BI_UNRESOLVED_HYDRA")
    common = set_up_common_builder(cfg, "r2_bi_hlc_kinematic_target_capture_build")
    callback_worker = build_callbacks_worker(cfg)
    callbacks = build_simulation_callbacks(cfg, common.output_dir, callback_worker)
    runners = build_simulations(cfg, common.worker, callbacks, callback_worker, pre_built_planners=[planner])
    if len(runners) != 1:
        raise RuntimeError("R2_BI_RUNNER_COUNT_NOT_ONE")
    simulation = runners[0]._simulation
    time_controller, ego_controller = simulation._time_controller, simulation._ego_controller
    if time_controller.__class__ is not R1Primary80ScientificTimeControllerV1 or time_controller.number_of_iterations() != 81:
        raise RuntimeError("R2_BI_PRIMARY80_BINDING_FAIL")
    if ego_controller.__class__.__name__ != "TwoStageController":
        raise RuntimeError("R2_BI_TWO_STAGE_CONTROLLER_BINDING_FAIL")
    recorder = ControllerCommandRecorderV2(telemetry_dir / "controller_actual_shadow.jsonl")
    recorder.install(ego_controller)
    lifecycle = run_one_with_full_nuplan_lifecycle(
        runners=runners, common_builder=common, profiler_name="r2_bi_hlc_kinematic_target_capture_running",
        cfg=cfg, run_output_root=run_root,
    )
    paths = {
        "trace": trace_dir / "realized_current_ego.jsonl",
        "planner": telemetry_dir / "planner_kinematic_capture.jsonl",
        "controller": telemetry_dir / "controller_actual_shadow.jsonl",
    }
    counts = {key: sum(bool(line.strip()) for line in path.read_text().splitlines()) for key, path in paths.items()}
    if counts != {"trace": 80, "planner": 80, "controller": 79}:
        raise RuntimeError(f"R2_BI_PRIMARY80_ARTIFACT_COUNT_FAIL:{run['run_id']}:{counts}")
    return {
        **run, "status": "TECHNICAL_COMPLETE", "run_root": str(run_root.relative_to(ROOT)),
        "trace_path": str(paths["trace"].relative_to(ROOT)),
        "planner_telemetry_path": str(paths["planner"].relative_to(ROOT)),
        "controller_telemetry_path": str(paths["controller"].relative_to(ROOT)),
        "telemetry_counts": counts, "full_lifecycle": lifecycle,
        "planner_class": planner.__class__.__name__, "controller_class": ego_controller.__class__.__name__,
        "tracker_class": ego_controller._tracker.__class__.__name__, "runner_run_calls": 1,
    }


def _trace(run: Mapping[str, Any]) -> list[Dict[str, Any]]:
    return [json.loads(line)["current_ego"] for line in (ROOT / run["trace_path"]).read_text().splitlines() if line.strip()]


def _telemetry(run: Mapping[str, Any], key: str) -> list[Dict[str, Any]]:
    return [json.loads(line) for line in (ROOT / run[key]).read_text().splitlines() if line.strip()]


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


def _arm_diagnostic(run: Mapping[str, Any], binding: Mapping[str, Any], parameters: Mapping[str, Any]) -> Dict[str, Any]:
    trace = _trace(run)
    planner = _telemetry(run, "planner_telemetry_path")
    controls = _telemetry(run, "controller_telemetry_path")
    times, xy, _, _ = trajectory_arrays_timestamp_aware(trace)
    absolute = times - times[0]
    realized = _signed_offsets(xy, binding["target_reference_xy"])
    start, end = float(parameters["capture"]["capture_start_abs_s"]), float(parameters["capture"]["capture_end_abs_s"])
    start_index = int(np.argmin(np.abs(absolute - start)))
    terminal_index = len(realized) - 1
    post_deadline = [row for row in planner if float(row["absolute_episode_time_s"]) >= end - 1e-9]
    hard_jumps = []
    inconsistencies = []
    feasibility_failures = []
    for row in planner:
        states = row["controller_lookahead"]["states_0_to_10"]
        s0, s1 = states[0]["rear_axle"], states[1]["rear_axle"]
        h0 = float(s0["heading"])
        dx, dy = float(s1["x"] - s0["x"]), float(s1["y"] - s0["y"])
        lateral = -math.sin(h0) * dx + math.cos(h0) * dy
        hard_jumps.append(abs(lateral))
        capture = row["target_capture"]
        inconsistencies.append(max(
            float(capture["pose_consistency"]["state0_tangent_mismatch_abs_rad"]),
            float(capture["pose_consistency"]["max_future_declared_heading_vs_final_xy_tangent_abs_rad"]),
        ))
        feasibility_failures.append(not bool(capture["feasibility"]["pass"]))
    direction_disagreement = sum(not row["direction_agreement"] for row in controls)
    command_max_diff = max(float(row["absolute_command_difference_radps"]) for row in controls)
    planned_offsets = {
        "capture_start_state1_m": float(planner[start_index]["actual_planned_state1_target_frame_offset_m"]),
        "terminal_state1_m": float(planner[terminal_index]["actual_planned_state1_target_frame_offset_m"]),
    }
    return {
        "realized_capture_start_abs_offset_m": abs(float(realized[start_index])),
        "realized_terminal_abs_offset_m": abs(float(realized[terminal_index])),
        "realized_offset_decreased": bool(abs(realized[terminal_index]) < abs(realized[start_index]) - 1e-9),
        "planned_actual_target_offsets": planned_offsets,
        "controller_shadow_actual_direction_disagreement_rows": direction_disagreement,
        "controller_shadow_actual_max_abs_difference_radps": command_max_diff,
        "maximum_xy_heading_inconsistency_rad": max(inconsistencies),
        "kinematic_feasibility_failure_rows": sum(feasibility_failures),
        "maximum_state0_to_state1_lateral_step_m": max(hard_jumps),
        "post_deadline_planner_rows": len(post_deadline),
        "post_deadline_hard_jump": bool(post_deadline and max(hard_jumps[-len(post_deadline):]) > 0.25),
    }


def _pair_diagnostic(
    binding: Mapping[str, Any], baseline: Mapping[str, Any], treatment: Mapping[str, Any], parameters: Mapping[str, Any],
) -> Dict[str, Any]:
    dispatched = evaluate_frozen_pair(
        pair_binding=binding, baseline_run_dir=ROOT / baseline["run_root"], treatment_run_dir=ROOT / treatment["run_root"]
    )
    evaluation = dispatched["evaluation"]
    base_trace, treat_trace = _trace(baseline), _trace(treatment)
    bt, bxy, _, bs = trajectory_arrays_timestamp_aware(base_trace)
    tt, txy, _, ts = trajectory_arrays_timestamp_aware(treat_trace)
    bp = hlc_realized_lane_transition_progress_v1_0(
        source_reference_xy=binding["source_reference_xy"], target_reference_xy=binding["target_reference_xy"], realized_ego_xy=bxy
    )
    tp = hlc_realized_lane_transition_progress_v1_0(
        source_reference_xy=binding["source_reference_xy"], target_reference_xy=binding["target_reference_xy"], realized_ego_xy=txy
    )
    bm = calculate_hlc_option_b_v2_timestamp_aware(bt, bp["clipped_progress_for_frozen_mechanism"], bs)
    tm = calculate_hlc_option_b_v2_timestamp_aware(tt, tp["clipped_progress_for_frozen_mechanism"], ts)
    endpoint = evaluation["endpoint"]
    gate_pair = {key: bool(endpoint["baseline"]["pass_by_gate"][key] and endpoint["treatment"]["pass_by_gate"][key]) for key in ("offset", "heading", "lateral_velocity", "paired_route_progress_delta")}
    engineering = evaluation["engineering"]
    engineering_pass = all(
        arm["max_abs_lateral_accel_mps2"] <= arm["frozen_limits"]["lateral_accel_mps2_max"]
        and arm["max_abs_yaw_rate_radps"] <= arm["frozen_limits"]["yaw_rate_radps_max"]
        and arm["max_abs_curvature_inv_m"] <= arm["frozen_limits"]["curvature_inv_m_max"]
        for arm in engineering.values()
    )
    return {
        "pair_id": binding["pair_id"], "mechanism_pass": bool(evaluation["mechanism"]["pass"]),
        "F_match_pass": bool(evaluation["f_match"]["pass"]), "safety_pass": bool(dispatched["official_safety_pair_pass"]),
        "endpoint_pass": bool(endpoint["baseline"]["pass"] and endpoint["treatment"]["pass"]),
        "endpoint_pass_by_gate_pair": gate_pair, "engineering_pass": engineering_pass,
        "baseline_measurement": bm, "treatment_measurement": tm,
        "baseline_architecture": _arm_diagnostic(baseline, binding, parameters),
        "treatment_architecture": _arm_diagnostic(treatment, binding, parameters),
        "evaluation": evaluation,
    }


def _dist(values: Iterable[float]) -> Dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    q = np.quantile(array, [0, .25, .5, .75, 1]) if len(array) else [math.nan] * 5
    return {"n": len(array), **dict(zip(("min", "p25", "median", "p75", "max"), [None if not np.isfinite(x) else round(float(x), 6) for x in q]))}


def _summary(pairs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counts = {
        "pairs": 8, "mechanism_pass": sum(x["mechanism_pass"] for x in pairs),
        "endpoint_pass": sum(x["endpoint_pass"] for x in pairs),
        "endpoint_offset_pass": sum(x["endpoint_pass_by_gate_pair"]["offset"] for x in pairs),
        "heading_pass": sum(x["endpoint_pass_by_gate_pair"]["heading"] for x in pairs),
        "lateral_velocity_pass": sum(x["endpoint_pass_by_gate_pair"]["lateral_velocity"] for x in pairs),
        "route_progress_pass": sum(x["endpoint_pass_by_gate_pair"]["paired_route_progress_delta"] for x in pairs),
        "F_match_pass": sum(x["F_match_pass"] for x in pairs), "engineering_pass": sum(x["engineering_pass"] for x in pairs),
        "safety_pass": sum(x["safety_pass"] for x in pairs),
    }
    no_departure_unfinished = sum(
        x["treatment_measurement"].get("status") in {"NO_DEPARTURE", "UNFINISHED_TRANSITION"} for x in pairs
    )
    direction_disagreement = sum(
        x[arm]["controller_shadow_actual_direction_disagreement_rows"] for x in pairs for arm in ("baseline_architecture", "treatment_architecture")
    )
    inconsistent = sum(
        x[arm]["maximum_xy_heading_inconsistency_rad"] > 0.03 for x in pairs for arm in ("baseline_architecture", "treatment_architecture")
    )
    declining = sum(x["treatment_architecture"]["realized_offset_decreased"] for x in pairs)
    hard_jump = any(x[arm]["post_deadline_hard_jump"] for x in pairs for arm in ("baseline_architecture", "treatment_architecture"))
    feasibility_fail = sum(
        x[arm]["kinematic_feasibility_failure_rows"] > 0 for x in pairs for arm in ("baseline_architecture", "treatment_architecture")
    )
    architecture_stops = {
        "treatment_NO_DEPARTURE_plus_UNFINISHED_TRANSITION_gte_4": no_departure_unfinished >= 4,
        "controller_shadow_actual_direction_disagreement": direction_disagreement > 0,
        "systematic_xy_heading_curvature_inconsistency": inconsistent > 8,
        "realized_offset_not_decreasing_in_more_than_half": declining < 4,
        "post_deadline_hard_jump": hard_jump,
        "systematic_engineering_trajectory_feasibility_failure": feasibility_fail > 8,
    }
    architecture_stop = any(architecture_stops.values())
    success = all(counts[key] == 8 for key in ("mechanism_pass", "endpoint_pass", "F_match_pass", "engineering_pass")) and counts["safety_pass"] >= 4
    return {
        "counts": counts, "treatment_NO_DEPARTURE_plus_UNFINISHED_TRANSITION": no_departure_unfinished,
        "realized_treatment_offset_decrease_count": declining,
        "controller_direction_disagreement_rows": direction_disagreement,
        "xy_heading_inconsistent_arms": inconsistent, "feasibility_failed_arms": feasibility_fail,
        "architecture_stop_checks": architecture_stops,
        "architecture_stop_before_round1": architecture_stop,
        "round1_failure_class": "ARCHITECTURE_FAILURE_STOP" if architecture_stop else "NUMERICAL_GLOBAL_CALIBRATION_FAILURE",
        "realized_treatment_terminal_abs_offset_distribution_m": _dist(x["treatment_architecture"]["realized_terminal_abs_offset_m"] for x in pairs),
        "development_success": success,
    }


def _next(previous: Mapping[str, Any], summary: Mapping[str, Any]) -> Dict[str, Any]:
    if summary["architecture_stop_before_round1"]:
        raise PermissionError("R2_BI_ROUND1_FORBIDDEN_ARCHITECTURE_FAILURE")
    space = read(SPACE)
    result = json.loads(json.dumps(previous))
    if summary["counts"]["mechanism_pass"] < 8:
        result["morphology"]["retreat_depth"] += 0.06
        result["morphology"]["retreat_duration_s"] += 0.15
    if summary["counts"]["endpoint_pass"] < 8:
        result["capture"]["capture_end_abs_s"] -= 0.30
    for key, bounds in space["bounds"]["morphology"].items():
        result["morphology"][key] = round(float(np.clip(result["morphology"][key], *bounds)), 6)
    for key, bounds in space["bounds"]["capture"].items():
        result["capture"][key] = round(float(np.clip(result["capture"][key], *bounds)), 6)
    validate_parameters(result)
    return result


def execute(output_root: Path) -> Mapping[str, Any]:
    if sha(PROTECTED) != PROTECTED_SHA:
        raise PermissionError("PROTECTED_CSV_SHA_MISMATCH")
    if read(ENTRY)["status"] != "R2_BI_ZERO_RUN_ENTRY_GATES_PASS":
        raise PermissionError("R2_BI_SIMULATION_NOT_AUTHORIZED")
    authorization = read(AUTHORIZATION)
    if authorization.get("R2_BI_ENGINEERING_ONLY_HLC_SIMULATION_AUTHORIZED_CONDITIONALLY") is not True:
        raise PermissionError("R2_BI_ENGINEERING_SIMULATION_NOT_AUTHORIZED")
    roster, pair_doc, ledger = read(ROSTER), read(PAIRS), read(LEDGER)
    if ledger["rounds"]:
        raise RuntimeError("R2_BI_LEDGER_ALREADY_CONTAINS_ROUNDS")
    entries = {row["scenario_token"]: row for row in roster["entries"]}
    bindings = {row["pair_id"]: row for row in pair_doc["pairs"]}
    parameters = read(SPACE)["round0"]
    rounds = []
    for round_index in range(2):
        if round_index == 0:
            parameter_file = ROUND_DIR / "r2_bi_hlc_dev_kin_round_0_parameters_v3.0.json"
            payload = read(parameter_file)
        else:
            parameter_file = ROUND_DIR / "r2_bi_hlc_dev_kin_round_1_parameters_v3.0.json"
            prior_summary = rounds[-1]["summary"]
            parameters = _next(parameters, prior_summary)
            prior_runs = read(ROUND_DIR / "r2_bi_hlc_dev_kin_round_0_parameters_v3.0.json")["runs"]
            payload = {
                "schema_version": "r2_bi_hlc_dev_kin_round_1_parameters_v3.0",
                "status": "FROZEN_BEFORE_ROUND1_SIMULATION", "round_index": 1, "parameters": parameters,
                "runs": [{**run, "run_id": run["run_id"].replace("-R0-", "-R1-")} for run in prior_runs],
                "global_no_identity_specific_parameters": True,
                "source": "PRE_REGISTERED_DETERMINISTIC_AGGREGATE_UPDATE",
            }
            write(parameter_file, payload)
        validate_parameters(payload["parameters"])
        run_results = []
        for run in payload["runs"]:
            try:
                run_results.append(_execute_one(
                    run, entries[run["scenario_token"]], payload["parameters"], parameter_file,
                    output_root / f"round_{round_index}",
                ))
            except Exception as error:
                ledger["status"] = "R2_BI_ARCHITECTURE_OR_TECHNICAL_FAILURE_STOP"
                ledger.setdefault("execution_failures", []).append({
                    "round_index": round_index, "run_id": run["run_id"],
                    "error": f"{type(error).__name__}:{error}", "remaining_schedule_stopped": True,
                })
                ledger["rounds"] = rounds
                ledger["actual_HLC_engineering_runs"] = sum(len(x["runs"]) for x in rounds)
                write(LEDGER, ledger, update=True)
                raise
        pair_results = []
        for index in range(1, 9):
            pair_id = f"R2BI-DEV-KIN-HLC-{index:02d}"
            pair_runs = [x for x in run_results if x["pair_id"] == pair_id]
            pair_results.append(_pair_diagnostic(
                bindings[pair_id], next(x for x in pair_runs if x["arm"] == ARM_BASELINE),
                next(x for x in pair_runs if x["arm"] == ARM_TREATMENT), payload["parameters"],
            ))
        summary = _summary(pair_results)
        result = {
            "schema_version": f"r2_bi_hlc_dev_kin_round_{round_index}_results_v1.0",
            "round_index": round_index, "parameter_file": str(parameter_file.relative_to(ROOT)),
            "parameter_sha256": sha(parameter_file), "parameters": payload["parameters"],
            "runs": run_results, "pairs": pair_results, "summary": summary,
            "scientific_confirmation": False, "failed_results_deleted": False,
        }
        result_file = ROUND_DIR / f"r2_bi_hlc_dev_kin_round_{round_index}_results_v1.0.json"
        write(result_file, result)
        rounds.append({
            "round_index": round_index, "parameter_file": str(parameter_file.relative_to(ROOT)),
            "parameter_sha256": sha(parameter_file), "result_file": str(result_file.relative_to(ROOT)),
            "result_sha256": sha(result_file), "runs": run_results, "summary": summary,
        })
        ledger["rounds"] = rounds
        ledger["actual_HLC_engineering_runs"] = 16 * len(rounds)
        ledger["status"] = "R2_BI_ROUND_COMPLETE"
        write(LEDGER, ledger, update=True)
        if summary["development_success"] or summary["architecture_stop_before_round1"]:
            break
    ledger["status"] = "R2_BI_ENGINEERING_EXECUTION_COMPLETE"
    ledger["rounds"] = rounds
    ledger["TSB_simulation_calls"] = 0
    write(LEDGER, ledger, update=True)
    return {"rounds": rounds, "ledger": ledger}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/r2_bi_hlc_kinematic_target_capture_dev_v1")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "R2_BI_ZERO_RUN_INPUT_CLOSURE_PASS", "simulation": 0}))
        return 0
    result = execute(args.output_root.resolve())
    print(json.dumps({
        "status": result["ledger"]["status"], "rounds": len(result["rounds"]),
        "actual_HLC_engineering_runs": result["ledger"].get("actual_HLC_engineering_runs", 0),
        "TSB_simulation_calls": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
