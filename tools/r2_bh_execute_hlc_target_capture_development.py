#!/usr/bin/env python3
"""Execute at most three frozen R2-BH HLC DEV-ARCH rounds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_b2_8_r3_prospective_selector import official_count, official_env  # noqa: E402
from tools.r1_b2_9_e_official_run_lifecycle import run_one_with_full_nuplan_lifecycle  # noqa: E402
from tools.r1_b2_8_r3_2_post_run_evaluator_dispatcher import evaluate_frozen_pair  # noqa: E402
from tools.r1_closed_loop_benchmark_v2_1 import calculate_hlc_option_b_v2_timestamp_aware, trajectory_arrays_timestamp_aware  # noqa: E402
from tools.r1_hlc_measurement_conformance_v1 import hlc_realized_lane_transition_progress_v1_0  # noqa: E402
from tools.r2_bh_hlc_target_capture_generator_v2 import ARM_BASELINE, ARM_TREATMENT, validate_parameters  # noqa: E402
from tools.r2_bh_hlc_target_capture_planner_v2 import R2BHHLCTargetCapturePlannerV2  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "docs/stageR/r2"
ROSTER = R2 / "r2_bh_hlc_arch_dev_roster_v1.0.json"
PAIRS = R2 / "r2_bh_hlc_arch_pair_bindings_v1.0.json"
SPACE = R2 / "r2_bh_hlc_arch_parameter_space_v2.0.json"
CONTRACT = R2 / "r2_bh_hlc_architecture_contract_v2.0.json"
LEDGER = R2 / "r2_bh_hlc_arch_run_ledger_v1.0.json"
AUTHORIZATION = R2 / "r2_bh_scientific_owner_engineering_authorization_v1.0.json"
ROUND_DIR = R2 / "r2_bh_hlc_arch_rounds"
PROTECTED = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"
PROTECTED_SHA = "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8"


def read(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Mapping[str, Any], update: bool = False) -> None:
    if path.exists() and not update:
        raise FileExistsError(f"R2_BH_VERSIONED_OUTPUT_EXISTS:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_frozen_or_verify(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        if read(path) != value:
            raise PermissionError(f"R2_BH_EXISTING_FROZEN_PARAMETER_MISMATCH:{path}")
        return
    write(path, value)


def build_planner_from_environment(
    run_id: str, trace_dir: str, telemetry_dir: str, parameter_file: str
) -> R2BHHLCTargetCapturePlannerV2:
    payload = read(Path(parameter_file))
    run = next((row for row in payload["runs"] if row["run_id"] == run_id), None)
    if run is None:
        raise ValueError(f"R2_BH_ENV_RUN_NOT_FOUND:{run_id}")
    entry = next(row for row in read(ROSTER)["entries"] if row["scenario_token"] == run["scenario_token"])
    return R2BHHLCTargetCapturePlannerV2(entry, run["arm"], payload["parameters"], trace_dir, telemetry_dir)


def _clip(value: float, bounds: Sequence[float]) -> float:
    return round(float(min(max(value, float(bounds[0])), float(bounds[1]))), 6)


def _next(previous: Mapping[str, Any], summary: Mapping[str, Any]) -> Dict[str, Any]:
    space = read(SPACE)
    value = json.loads(json.dumps(previous))
    if summary["counts"]["mechanism_pass"] < 8:
        value["morphology"]["retreat_depth"] += 0.06
        value["morphology"]["retreat_duration_s"] += 0.15
    if summary["counts"]["endpoint_pass"] < 8:
        value["capture"]["capture_duration_s"] -= 0.20
    if summary["counts"]["endpoint_offset_pass"] < 8:
        value["capture"]["capture_start_abs_s"] -= 0.20
    if summary["counts"]["engineering_pass"] < 8:
        value["morphology"]["recommit_duration_s"] += 0.10
    for partition in ("morphology", "capture"):
        for key, bounds in space["bounds"][partition].items():
            value[partition][key] = _clip(value[partition][key], bounds)
    validate_parameters(value)
    return value


def _runs(round_index: int, roster: Mapping[str, Any]) -> list[Dict[str, Any]]:
    result = []
    for index, row in enumerate(roster["entries"], 1):
        pair_id = f"R2BH-ARCH-HLC-{index:02d}"
        for arm in (ARM_BASELINE, ARM_TREATMENT):
            result.append({
                "run_order": len(result) + 1, "run_id": f"R2BH-HLC-R{round_index}-{index:02d}-{arm}",
                "pair_id": pair_id, "family": "R-HLC", "arm": arm,
                "scenario_token": row["scenario_token"], "log_id": row["log_id"],
            })
    return result


def _overrides(run: Mapping[str, Any], entry: Mapping[str, Any], raw: Path) -> list[str]:
    return [
        "+simulation=closed_loop_nonreactive_agents", "planner=r2_bh_hlc_target_capture_dev_v2",
        "scenario_builder=nuplan_mini", f"scenario_builder.db_files=[{entry['db_path']}]",
        "scenario_filter=all_scenarios", f"scenario_filter.scenario_tokens=[{run['scenario_token']}]",
        "simulation_time_controller._target_=tools.r1_primary80_scientific_time_controller_v1.R1Primary80ScientificTimeControllerV1",
        "worker=sequential", "disable_callback_parallelization=true", "scenario_builder.max_workers=1",
        "max_callback_workers=1", "number_of_cpus_allocated_per_simulation=1",
        "number_of_gpus_allocated_per_simulation=0", "gpu=false", "seed=2026082701",
        "run_metric=true", "enable_simulation_progress_bar=false",
        "experiment_name=r2_bh_hlc_target_capture_dev", f"job_name={run['run_id']}", f"output_dir={raw}",
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
        raise FileExistsError(f"R2_BH_FRESH_RUN_ROOT_REQUIRED:{run_root}")
    if official_count(entry["db_path"], run["scenario_token"]) != 1:
        raise RuntimeError(f"R2_BH_EXACT_SCENARIO_RESOLUTION_NOT_ONE:{run['run_id']}")
    trace_dir.mkdir(parents=True)
    planner = R2BHHLCTargetCapturePlannerV2(entry, run["arm"], parameters, str(trace_dir), str(telemetry_dir))
    os.environ.update({
        "R2_BH_RUN_ID": run["run_id"], "R2_BH_TRACE_DIR": str(trace_dir),
        "R2_BH_TELEMETRY_DIR": str(telemetry_dir), "R2_BH_PARAMETER_FILE": str(parameter_file),
    })
    config_root = ROOT.parent / "nuplan-devkit/nuplan/planning/script/config/simulation"
    with initialize_config_dir(config_dir=str(config_root)):
        cfg = compose(config_name="default_simulation", overrides=_overrides(run, entry, raw))
    if "${" in json.dumps(OmegaConf.to_container(cfg, resolve=True), sort_keys=True):
        raise RuntimeError("R2_BH_UNRESOLVED_HYDRA")
    common = set_up_common_builder(cfg, "r2_bh_hlc_target_capture_dev_build")
    callback_worker = build_callbacks_worker(cfg)
    callbacks = build_simulation_callbacks(cfg, common.output_dir, callback_worker)
    runners = build_simulations(cfg, common.worker, callbacks, callback_worker, pre_built_planners=[planner])
    if len(runners) != 1:
        raise RuntimeError("R2_BH_RUNNER_COUNT_NOT_ONE")
    controller = runners[0]._simulation._time_controller
    if controller.__class__ is not R1Primary80ScientificTimeControllerV1 or controller.number_of_iterations() != 81:
        raise RuntimeError("R2_BH_PRIMARY80_BINDING_FAIL")
    lifecycle = run_one_with_full_nuplan_lifecycle(
        runners=runners, common_builder=common, profiler_name="r2_bh_hlc_target_capture_dev_running",
        cfg=cfg, run_output_root=run_root,
    )
    trace_file = trace_dir / "realized_current_ego.jsonl"
    telemetry_file = telemetry_dir / "planner_target_capture.jsonl"
    trace_rows = [line for line in trace_file.read_text().splitlines() if line.strip()]
    telemetry_rows = [line for line in telemetry_file.read_text().splitlines() if line.strip()]
    if len(trace_rows) != 80 or len(telemetry_rows) != 80:
        raise RuntimeError(f"R2_BH_PRIMARY80_ARTIFACT_COUNT_FAIL:{run['run_id']}:{len(trace_rows)}:{len(telemetry_rows)}")
    return {
        **run, "status": "TECHNICAL_COMPLETE", "run_root": str(run_root.relative_to(ROOT)),
        "trace_path": str(trace_file.relative_to(ROOT)), "telemetry_path": str(telemetry_file.relative_to(ROOT)),
        "trace_rows": 80, "planner_telemetry_rows": 80, "full_lifecycle": lifecycle,
        "runner_run_calls": 1, "run_runners_calls": 1,
    }


def _trace(run_root: str) -> list[Dict[str, Any]]:
    path = ROOT / run_root / "trace/realized_current_ego.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [row["current_ego"] for row in rows]


def _recover_completed_run(run: Mapping[str, Any], run_root: Path) -> Dict[str, Any]:
    trace = run_root / "trace/realized_current_ego.jsonl"
    telemetry = run_root / "telemetry/planner_target_capture.jsonl"
    runner_report = run_root / "raw/runner_report.parquet"
    metrics = run_root / "raw/metrics"
    if not all(path.is_file() for path in (trace, telemetry, runner_report)):
        raise RuntimeError(f"R2_BH_RECOVERY_ARTIFACT_MISSING:{run['run_id']}")
    trace_rows = [line for line in trace.read_text().splitlines() if line.strip()]
    telemetry_rows = [line for line in telemetry.read_text().splitlines() if line.strip()]
    if len(trace_rows) != 80 or len(telemetry_rows) != 80 or not any(metrics.glob("*.parquet")):
        raise RuntimeError(f"R2_BH_RECOVERY_ARTIFACT_INCOMPLETE:{run['run_id']}")
    return {
        **run, "status": "TECHNICAL_COMPLETE_RECOVERED_POSTPROCESSING_ONLY",
        "run_root": str(run_root.relative_to(ROOT)),
        "trace_path": str(trace.relative_to(ROOT)), "telemetry_path": str(telemetry.relative_to(ROOT)),
        "trace_rows": 80, "planner_telemetry_rows": 80,
        "full_lifecycle": {"recovered_from_complete_frozen_artifacts": True},
        "runner_run_calls": 1, "run_runners_calls": 1, "additional_runner_run_calls_for_recovery": 0,
    }


def _telemetry(run_root: str) -> list[Dict[str, Any]]:
    path = ROOT / run_root / "telemetry/planner_target_capture.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _signed_offsets(points: np.ndarray, reference: Sequence[Sequence[float]]) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float64)
    starts, vectors = ref[:-1], np.diff(ref, axis=0)
    denom = np.sum(vectors * vectors, axis=1)
    result = []
    for point in np.asarray(points, dtype=np.float64):
        u = np.clip(np.sum((point - starts) * vectors, axis=1) / np.maximum(denom, 1e-12), 0.0, 1.0)
        projected = starts + u[:, None] * vectors
        distance = np.sum((point - projected) ** 2, axis=1)
        index = int(np.argmin(distance))
        tangent = vectors[index] / max(math.sqrt(denom[index]), 1e-12)
        normal = np.asarray([-tangent[1], tangent[0]])
        result.append(float(np.dot(point - projected[index], normal)))
    return np.asarray(result)


def _capture_audit(binding: Mapping[str, Any], run_root: str, parameters: Mapping[str, Any]) -> Dict[str, Any]:
    trace, telemetry = _trace(run_root), _telemetry(run_root)
    times, xy, _, _ = trajectory_arrays_timestamp_aware(trace)
    absolute = times - times[0]
    realized = _signed_offsets(xy, binding["target_reference_xy"])
    start = float(parameters["capture"]["capture_start_abs_s"])
    end = start + float(parameters["capture"]["capture_duration_s"])
    landmarks = {"capture_start": start, "capture_midpoint": 0.5 * (start + end), "capture_end": end, "Primary_terminal": float(absolute[-1])}
    rows = {}
    for name, target in landmarks.items():
        index = int(np.argmin(np.abs(absolute - target)))
        rows[name] = {
            "requested_absolute_s": target, "actual_absolute_s": float(absolute[index]), "iteration": index,
            "planner_state1_commanded_target_frame_offset_m": float(telemetry[index]["planned_state1_target_frame_offset_command_m"]),
            "realized_target_frame_offset_m": float(realized[index]),
        }
    return {
        "landmarks": rows,
        "terminal_realized_target_frame_offset_m": float(realized[-1]),
        "capture_end_state1_command_is_zero": abs(rows["capture_end"]["planner_state1_commanded_target_frame_offset_m"]) <= 1e-9,
        "post_capture_end_zero_command_rows": sum(
            row["absolute_episode_time_s"] >= end - 1e-9
            and abs(row["planned_state1_target_frame_offset_command_m"]) <= 1e-9
            for row in telemetry
        ),
    }


def _pair_diagnostic(
    binding: Mapping[str, Any], dispatched: Mapping[str, Any], baseline: Mapping[str, Any],
    treatment: Mapping[str, Any], parameters: Mapping[str, Any],
) -> Dict[str, Any]:
    evaluation = dispatched["evaluation"]
    baseline_trace, treatment_trace = _trace(baseline["run_root"]), _trace(treatment["run_root"])
    bt, bxy, _, bs = trajectory_arrays_timestamp_aware(baseline_trace)
    tt, txy, _, ts = trajectory_arrays_timestamp_aware(treatment_trace)
    bp = hlc_realized_lane_transition_progress_v1_0(
        source_reference_xy=binding["source_reference_xy"], target_reference_xy=binding["target_reference_xy"], realized_ego_xy=bxy
    )
    tp = hlc_realized_lane_transition_progress_v1_0(
        source_reference_xy=binding["source_reference_xy"], target_reference_xy=binding["target_reference_xy"], realized_ego_xy=txy
    )
    bm = calculate_hlc_option_b_v2_timestamp_aware(bt, bp["clipped_progress_for_frozen_mechanism"], bs)
    tm = calculate_hlc_option_b_v2_timestamp_aware(tt, tp["clipped_progress_for_frozen_mechanism"], ts)
    endpoint, engineering = evaluation["endpoint"], evaluation["engineering"]
    engineering_pass = all(
        arm["max_abs_lateral_accel_mps2"] <= arm["frozen_limits"]["lateral_accel_mps2_max"]
        and arm["max_abs_yaw_rate_radps"] <= arm["frozen_limits"]["yaw_rate_radps_max"]
        and arm["max_abs_curvature_inv_m"] <= arm["frozen_limits"]["curvature_inv_m_max"]
        for arm in engineering.values()
    )
    gate_pair = {
        key: bool(endpoint["baseline"]["pass_by_gate"][key] and endpoint["treatment"]["pass_by_gate"][key])
        for key in ("offset", "heading", "lateral_velocity", "paired_route_progress_delta")
    }
    return {
        "pair_id": binding["pair_id"], "mechanism_pass": bool(evaluation["mechanism"]["pass"]),
        "F_match_pass": bool(evaluation["f_match"]["pass"]), "safety_pass": bool(dispatched["official_safety_pair_pass"]),
        "endpoint_pass": bool(endpoint["baseline"]["pass"] and endpoint["treatment"]["pass"]),
        "endpoint_pass_by_gate_pair": gate_pair, "engineering_pass": engineering_pass,
        "baseline_measurement": bm, "treatment_measurement": tm,
        "mechanism_margins": {
            "treatment_retreat_count_minus_1": float(tm["hesitation_retreat_count"] - 1),
            "latency_delta_minus_0p5_s": None if evaluation["mechanism"].get("delta_commit_latency_s") is None else float(evaluation["mechanism"]["delta_commit_latency_s"] - 0.5),
            "monotonic_delta_beyond_minus_0p10": None if evaluation["mechanism"].get("delta_monotonic_fraction") is None else float(-0.10 - evaluation["mechanism"]["delta_monotonic_fraction"]),
        },
        "endpoint": endpoint, "engineering": engineering, "evaluation": evaluation,
        "target_capture": {
            "baseline": _capture_audit(binding, baseline["run_root"], parameters),
            "treatment": _capture_audit(binding, treatment["run_root"], parameters),
        },
    }


def _dist(values: Iterable[float | None]) -> Dict[str, Any]:
    array = np.asarray([float(value) for value in values if value is not None], dtype=np.float64)
    if not len(array):
        return {"n": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None}
    q = np.quantile(array, [0, .25, .5, .75, 1])
    return {"n": len(array), **dict(zip(("min", "p25", "median", "p75", "max"), [round(float(value), 6) for value in q]))}


def _summary(pairs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counts = {
        "pairs": 8, "mechanism_pass": sum(row["mechanism_pass"] for row in pairs),
        "endpoint_pass": sum(row["endpoint_pass"] for row in pairs),
        "endpoint_offset_pass": sum(row["endpoint_pass_by_gate_pair"]["offset"] for row in pairs),
        "heading_pass": sum(row["endpoint_pass_by_gate_pair"]["heading"] for row in pairs),
        "lateral_velocity_pass": sum(row["endpoint_pass_by_gate_pair"]["lateral_velocity"] for row in pairs),
        "route_progress_pass": sum(row["endpoint_pass_by_gate_pair"]["paired_route_progress_delta"] for row in pairs),
        "F_match_pass": sum(row["F_match_pass"] for row in pairs),
        "engineering_pass": sum(row["engineering_pass"] for row in pairs),
        "safety_pass": sum(row["safety_pass"] for row in pairs),
    }
    margins = {
        key: _dist(row["mechanism_margins"][key] for row in pairs)
        for key in ("treatment_retreat_count_minus_1", "latency_delta_minus_0p5_s", "monotonic_delta_beyond_minus_0p10")
    }
    terminal = {
        arm: _dist(abs(row["target_capture"][arm]["terminal_realized_target_frame_offset_m"]) for row in pairs)
        for arm in ("baseline", "treatment")
    }
    capture_zero = all(
        row["target_capture"][arm]["capture_end_state1_command_is_zero"]
        for row in pairs for arm in ("baseline", "treatment")
    )
    success = all(counts[key] == 8 for key in ("mechanism_pass", "endpoint_pass", "F_match_pass", "engineering_pass")) and counts["safety_pass"] >= 4
    return {
        "counts": counts, "mechanism_margin_distributions": margins,
        "absolute_terminal_target_offset_distributions_m": terminal,
        "capture_end_zero_command_16_of_16": capture_zero,
        "development_success": success,
    }


def execute(output_root: Path, recover_completed_round0: bool = False) -> Dict[str, Any]:
    if sha(PROTECTED) != PROTECTED_SHA:
        raise PermissionError("PROTECTED_CSV_SHA_MISMATCH")
    authorization = read(AUTHORIZATION)
    if authorization.get("R2_BH_ENGINEERING_ONLY_HLC_SIMULATION_AUTHORIZED") is not True:
        raise PermissionError("R2_BH_ENGINEERING_SIMULATION_NOT_AUTHORIZED")
    roster, pair_doc, space, ledger = read(ROSTER), read(PAIRS), read(SPACE), read(LEDGER)
    if ledger["rounds"]:
        raise RuntimeError("R2_BH_LEDGER_ALREADY_CONTAINS_ROUNDS")
    if ledger["maximum_rounds"] != 3 or space["maximum_rounds"] != 3:
        raise RuntimeError("R2_BH_MAX_ROUNDS_NOT_THREE")
    entry_by_token = {row["scenario_token"]: row for row in roster["entries"]}
    binding_by_id = {row["pair_id"]: row for row in pair_doc["pairs"]}
    parameters = space["round0"]
    validate_parameters(parameters)
    all_rounds = []
    for round_index in range(3):
        runs = _runs(round_index, roster)
        parameter_file = ROUND_DIR / f"r2_bh_hlc_arch_round_{round_index}_parameters_v2.0.json"
        parameter_payload = {
            "schema_version": f"r2_bh_hlc_arch_round_{round_index}_parameters_v2.0",
            "status": "FROZEN_BEFORE_THIS_ROUND_SIMULATION", "round_index": round_index,
            "architecture": "BEHAVIOR_MORPHOLOGY_PLUS_FIXED_ABSOLUTE_TARGET_CAPTURE",
            "parameters": parameters, "runs": runs, "global_no_identity_specific_parameters": True,
            "source": "NEW_TARGET_CAPTURE_ARCHITECTURE_ROUND0" if round_index == 0 else "DETERMINISTIC_PRE_REGISTERED_AGGREGATE_UPDATE",
        }
        write_frozen_or_verify(parameter_file, parameter_payload)
        run_results = []
        for run in runs:
            try:
                if recover_completed_round0 and round_index == 0:
                    run_results.append(_recover_completed_run(run, output_root / "round_0" / run["run_id"]))
                else:
                    run_results.append(_execute_one(
                        run, entry_by_token[run["scenario_token"]], parameters, parameter_file,
                        output_root / f"round_{round_index}",
                    ))
            except Exception as exc:
                ledger["status"] = "TECHNICAL_FAILURE_STOPPED_REMAINING_SCHEDULE"
                ledger.setdefault("technical_failures", []).append({
                    "run_id": run["run_id"], "round_index": round_index,
                    "error": f"{type(exc).__name__}:{exc}", "scientific_or_behavior_failure": False,
                })
                write(LEDGER, ledger, update=True)
                raise
        pair_results = []
        for index in range(1, 9):
            pair_id = f"R2BH-ARCH-HLC-{index:02d}"
            pair_runs = [row for row in run_results if row["pair_id"] == pair_id]
            baseline = next(row for row in pair_runs if row["arm"] == ARM_BASELINE)
            treatment = next(row for row in pair_runs if row["arm"] == ARM_TREATMENT)
            binding = binding_by_id[pair_id]
            dispatched = evaluate_frozen_pair(
                pair_binding=binding, baseline_run_dir=ROOT / baseline["run_root"],
                treatment_run_dir=ROOT / treatment["run_root"],
            )
            pair_results.append(_pair_diagnostic(binding, dispatched, baseline, treatment, parameters))
        summary = _summary(pair_results)
        result = {
            "schema_version": f"r2_bh_hlc_arch_round_{round_index}_results_v1.0",
            "round_index": round_index, "parameter_file": str(parameter_file.relative_to(ROOT)),
            "parameter_sha256": sha(parameter_file), "parameters": parameters,
            "runs": run_results, "pairs": pair_results, "summary": summary,
            "scientific_confirmation": False, "failed_results_deleted": False,
        }
        result_file = ROUND_DIR / f"r2_bh_hlc_arch_round_{round_index}_results_v1.0.json"
        write(result_file, result)
        item = {
            "round_index": round_index, "parameter_file": str(parameter_file.relative_to(ROOT)),
            "parameter_sha256": sha(parameter_file), "result_file": str(result_file.relative_to(ROOT)),
            "result_sha256": sha(result_file), "summary": summary,
        }
        all_rounds.append(item)
        ledger["rounds"] = all_rounds
        ledger["status"] = "R2_BH_ENGINEERING_CALIBRATION_IN_PROGRESS"
        ledger["actual_engineering_runs"] = 16 * len(all_rounds)
        write(LEDGER, ledger, update=True)
        if summary["development_success"]:
            break
        if round_index < 2:
            parameters = _next(parameters, summary)
    ledger["status"] = "R2_BH_ENGINEERING_EXECUTION_COMPLETE"
    ledger["TSB_simulation_calls"] = 0
    write(LEDGER, ledger, update=True)
    return {"rounds": all_rounds, "ledger": ledger}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--recover-completed-round0", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/r2_bh_hlc_target_capture_dev_v1")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "R2_BH_ZERO_RUN_INPUT_CLOSURE_PASS", "simulation": 0}))
        return 0
    result = execute(args.output_root.resolve(), args.recover_completed_round0)
    print(json.dumps({
        "status": result["ledger"]["status"], "rounds": len(result["rounds"]),
        "actual_engineering_runs": result["ledger"]["actual_engineering_runs"], "TSB_simulation_calls": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
