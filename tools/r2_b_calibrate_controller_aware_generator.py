#!/usr/bin/env python3
"""Execute at most four deterministic R2-B DEV-CAL rounds per family."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_b2_8_r3_prospective_selector import official_count, official_env  # noqa: E402
from tools.r1_b2_9_e_official_run_lifecycle import run_one_with_full_nuplan_lifecycle  # noqa: E402
from tools.r1_b2_8_r3_2_post_run_evaluator_dispatcher import evaluate_frozen_pair  # noqa: E402
from tools.r1_closed_loop_benchmark_v2_1 import (  # noqa: E402
    calculate_hlc_option_b_v2_timestamp_aware, calculate_tsb_option_a_v2_timestamp_aware,
    trajectory_arrays_timestamp_aware,
)
from tools.r1_hlc_measurement_conformance_v1 import hlc_realized_lane_transition_progress_v1_0  # noqa: E402
from tools.r2_b_controller_aware_planner_v1 import R2BControllerAwarePlannerV1  # noqa: E402
from tools.r2_b_controller_aware_generator_v1 import ARM_BASELINE, ARM_TREATMENT, validate_global_parameters  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "docs/stageR/r2"
ROSTER = R2 / "r2_b_generator_calibration_roster_v1.0.json"
PAIR_BINDINGS = R2 / "r2_b_generator_calibration_pair_bindings_v1.0.json"
HLC_SPACE = R2 / "r2_b_hlc_calibration_parameter_space_v1.0.json"
TSB_SPACE = R2 / "r2_b_tsb_calibration_parameter_space_v1.0.json"
OBJECTIVE = R2 / "r2_b_generator_calibration_objective_v1.0.json"
LEDGER = R2 / "r2_b_generator_calibration_run_ledger_v1.0.json"
AUTHORIZATION = R2 / "r2_b_scientific_owner_dev_calibration_authorization_v1.0.json"
ROUND_DIR = R2 / "r2_b_calibration_rounds"
PROTECTED = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"
PROTECTED_SHA = "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8"
EXPECTED_ROSTER_SHA = "2b6c0d8459fadd6fb9b80a6623de31d11c36710bbf2e4666c999f27e87d68dcc"
EXPECTED_PAIR_SHA = "b55e7f3bae8221b4c35cd7b95e6425dccb9a0ba67cd8e4ff1ba64bdef0afafa5"


def read(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def write(path: Path, value: Mapping[str, Any], update: bool = False) -> None:
    if path.exists() and not update:
        raise FileExistsError(f"R2_B_VERSIONED_OUTPUT_EXISTS:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_frozen_or_verify(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        if read(path) != value:
            raise PermissionError(f"R2_B_EXISTING_FROZEN_ROUND_PARAMETER_MISMATCH:{path}")
        return
    write(path, value)


def build_planner_from_environment(run_id: str, trace_dir: str, telemetry_dir: str, parameter_file: str) -> R2BControllerAwarePlannerV1:
    payload = read(Path(parameter_file))
    run = next((row for row in payload["runs"] if row["run_id"] == run_id), None)
    if run is None:
        raise ValueError(f"R2_B_ENV_RUN_NOT_FOUND:{run_id}")
    roster = read(ROSTER)
    entry = next(row for row in roster["entries"] if row["scenario_token"] == run["scenario_token"])
    return R2BControllerAwarePlannerV1(entry, run["family"], run["arm"], payload["parameters"], trace_dir, telemetry_dir)


def _load_frozen() -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if sha(PROTECTED) != PROTECTED_SHA:
        raise PermissionError("PROTECTED_CSV_SHA_MISMATCH")
    if sha(ROSTER) != EXPECTED_ROSTER_SHA or sha(PAIR_BINDINGS) != EXPECTED_PAIR_SHA:
        raise PermissionError("R2_B_FROZEN_ROSTER_OR_PAIR_BINDING_SHA_MISMATCH")
    authorization = read(AUTHORIZATION)
    if authorization.get("R2_B_DEV_CALIBRATION_SIMULATION_AUTHORIZED") is not True:
        raise PermissionError("R2_B_DEV_CALIBRATION_NOT_AUTHORIZED")
    roster, pairs, objective, ledger = read(ROSTER), read(PAIR_BINDINGS), read(OBJECTIVE), read(LEDGER)
    if len(roster["entries"]) != 16 or len(pairs["pairs"]) != 16:
        raise ValueError("R2_B_FROZEN_CARDINALITY_FAIL")
    if ledger["maximum_rounds_per_family"] != 4 or objective["maximum_rounds_per_family"] != 4:
        raise ValueError("R2_B_MAXIMUM_ROUNDS_NOT_FOUR")
    return roster, pairs, objective, ledger


def _clip(value: float, bounds: Sequence[float]) -> float:
    return round(float(min(max(value, float(bounds[0])), float(bounds[1]))), 6)


def _initial(family: str) -> Dict[str, float]:
    space = read(HLC_SPACE if family == "R-HLC" else TSB_SPACE)
    params = dict(space["round0"])
    validate_global_parameters(family, params)
    return params


def _next(family: str, previous: Mapping[str, float], summary: Mapping[str, Any]) -> Dict[str, float]:
    space = read(HLC_SPACE if family == "R-HLC" else TSB_SPACE)
    bounds, value = space["bounds"], dict(previous)
    if family == "R-HLC":
        if summary["counts"]["mechanism_pass"] < 8:
            value["retreat_depth"] += 0.06
            value["retreat_duration_s"] += 0.15
        if summary["counts"]["endpoint_pass"] < 8 or summary["counts"]["engineering_pass"] < 8:
            value["recommit_duration_s"] -= 0.15
        for key in bounds:
            value[key] = _clip(value[key], bounds[key])
    else:
        if summary["counts"]["baseline_one_phase"] < 8:
            value["baseline_brake_mps2"] -= 0.20
            value["baseline_duration_s"] += 0.10
        if summary["counts"]["treatment_two_phase"] < 8:
            value["first_brake_mps2"] -= 0.30
            value["second_brake_mps2"] -= 0.30
            value["first_brake_duration_s"] += 0.10
            value["second_brake_duration_s"] += 0.10
        if summary["counts"]["measurement_OK"] < 8 or summary["counts"]["treatment_two_phase"] < 8:
            value["release_mps2"] += 0.20
            value["release_duration_s"] += 0.15
        if summary["counts"]["F_match_pass"] < 8:
            value["baseline_duration_s"] += 0.15
        for key in bounds:
            value[key] = _clip(value[key], bounds[key])
    validate_global_parameters(family, value)
    return value


def _runs(family: str, round_index: int, roster: Mapping[str, Any]) -> list[Dict[str, Any]]:
    rows = [row for row in roster["entries"] if row["family"] == family]
    prefix = "HLC" if family == "R-HLC" else "TSB"
    result = []
    for index, row in enumerate(rows, 1):
        pair_id = f"R2B-CAL-{prefix}-{index:02d}"
        for arm in (ARM_BASELINE, ARM_TREATMENT):
            result.append({
                "run_order": len(result) + 1,
                "run_id": f"R2B-{prefix}-R{round_index}-{index:02d}-{arm}",
                "pair_id": pair_id, "family": family, "arm": arm,
                "scenario_token": row["scenario_token"], "log_id": row["log_id"],
            })
    return result


def _overrides(run: Mapping[str, Any], entry: Mapping[str, Any], raw: Path) -> list[str]:
    return [
        "+simulation=closed_loop_nonreactive_agents", "planner=r2_b_controller_aware_dev_v1",
        "scenario_builder=nuplan_mini", f"scenario_builder.db_files=[{entry['db_path']}]",
        "scenario_filter=all_scenarios", f"scenario_filter.scenario_tokens=[{run['scenario_token']}]",
        "simulation_time_controller._target_=tools.r1_primary80_scientific_time_controller_v1.R1Primary80ScientificTimeControllerV1",
        "worker=sequential", "disable_callback_parallelization=true", "scenario_builder.max_workers=1",
        "max_callback_workers=1", "number_of_cpus_allocated_per_simulation=1",
        "number_of_gpus_allocated_per_simulation=0", "gpu=false", "seed=2026082701",
        "run_metric=true", "enable_simulation_progress_bar=false",
        "experiment_name=r2_b_controller_aware_dev_calibration", f"job_name={run['run_id']}",
        f"output_dir={raw}",
        f"hydra.searchpath=[file://{ROOT}/configs/r1_official_technical_smoke_hydra,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]",
    ]


def _execute_one(run: Mapping[str, Any], entry: Mapping[str, Any], parameters: Mapping[str, Any], parameter_file: Path, output_root: Path) -> Dict[str, Any]:
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
        raise FileExistsError(f"R2_B_FRESH_RUN_ROOT_REQUIRED:{run_root}")
    if official_count(entry["db_path"], run["scenario_token"]) != 1:
        raise RuntimeError(f"R2_B_EXACT_SCENARIO_RESOLUTION_NOT_ONE:{run['run_id']}")
    trace_dir.mkdir(parents=True)
    planner = R2BControllerAwarePlannerV1(entry, run["family"], run["arm"], parameters, str(trace_dir), str(telemetry_dir))
    os.environ.update({
        "R2_B_RUN_ID": run["run_id"], "R2_B_TRACE_DIR": str(trace_dir),
        "R2_B_TELEMETRY_DIR": str(telemetry_dir), "R2_B_PARAMETER_FILE": str(parameter_file),
    })
    config_root = ROOT.parent / "nuplan-devkit/nuplan/planning/script/config/simulation"
    with initialize_config_dir(config_dir=str(config_root)):
        cfg = compose(config_name="default_simulation", overrides=_overrides(run, entry, raw))
    if "${" in json.dumps(OmegaConf.to_container(cfg, resolve=True), sort_keys=True):
        raise RuntimeError("R2_B_UNRESOLVED_HYDRA")
    common = set_up_common_builder(cfg, "r2_b_controller_aware_dev_build")
    callback_worker = build_callbacks_worker(cfg)
    callbacks = build_simulation_callbacks(cfg, common.output_dir, callback_worker)
    runners = build_simulations(cfg, common.worker, callbacks, callback_worker, pre_built_planners=[planner])
    if len(runners) != 1:
        raise RuntimeError("R2_B_RUNNER_COUNT_NOT_ONE")
    controller = runners[0]._simulation._time_controller
    if controller.__class__ is not R1Primary80ScientificTimeControllerV1 or controller.number_of_iterations() != 81:
        raise RuntimeError("R2_B_PRIMARY80_BINDING_FAIL")
    lifecycle = run_one_with_full_nuplan_lifecycle(
        runners=runners, common_builder=common, profiler_name="r2_b_controller_aware_dev_running",
        cfg=cfg, run_output_root=run_root,
    )
    trace_file, telemetry_file = trace_dir / "realized_current_ego.jsonl", telemetry_dir / "planner_transfer.jsonl"
    trace_rows = [json.loads(line) for line in trace_file.read_text().splitlines() if line.strip()]
    telemetry_rows = [json.loads(line) for line in telemetry_file.read_text().splitlines() if line.strip()]
    if len(trace_rows) != 80 or len(telemetry_rows) != 80:
        raise RuntimeError(f"R2_B_PRIMARY80_ARTIFACT_COUNT_FAIL:{run['run_id']}:{len(trace_rows)}:{len(telemetry_rows)}")
    return {
        **run, "status": "TECHNICAL_COMPLETE", "run_root": str(run_root.relative_to(ROOT)),
        "trace_path": str(trace_file.relative_to(ROOT)), "telemetry_path": str(telemetry_file.relative_to(ROOT)),
        "trace_rows": 80, "planner_telemetry_rows": 80, "full_lifecycle": lifecycle,
        "runner_run_calls": 1, "run_runners_calls": 1,
    }


def _trace(run_root: str) -> list[Dict[str, Any]]:
    path = ROOT / run_root / "trace/realized_current_ego.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    # The frozen Primary80 artifact wraps the ego state with provenance fields.
    # Frozen measurement functions consume the inner state objects.
    return [row["current_ego"] for row in rows]


def _recover_completed_run(run: Mapping[str, Any], run_root: Path) -> Dict[str, Any]:
    """Recover post-processing after a technical failure without rerunning simulation."""
    trace_file = run_root / "trace/realized_current_ego.jsonl"
    telemetry_file = run_root / "telemetry/planner_transfer.jsonl"
    runner_report = run_root / "raw/runner_report.parquet"
    metric_dir = run_root / "raw/metrics"
    if not all(path.is_file() for path in (trace_file, telemetry_file, runner_report)):
        raise RuntimeError(f"R2_B_RECOVERY_ARTIFACT_MISSING:{run['run_id']}")
    trace_rows = [line for line in trace_file.read_text().splitlines() if line.strip()]
    telemetry_rows = [line for line in telemetry_file.read_text().splitlines() if line.strip()]
    if len(trace_rows) != 80 or len(telemetry_rows) != 80 or not any(metric_dir.glob("*.parquet")):
        raise RuntimeError(f"R2_B_RECOVERY_ARTIFACT_INCOMPLETE:{run['run_id']}")
    return {
        **run,
        "status": "TECHNICAL_COMPLETE_RECOVERED_POSTPROCESSING_ONLY",
        "run_root": str(run_root.relative_to(ROOT)),
        "trace_path": str(trace_file.relative_to(ROOT)),
        "telemetry_path": str(telemetry_file.relative_to(ROOT)),
        "trace_rows": 80,
        "planner_telemetry_rows": 80,
        "full_lifecycle": {"recovered_from_complete_frozen_artifacts": True},
        "runner_run_calls": 1,
        "run_runners_calls": 1,
        "additional_runner_run_calls_for_recovery": 0,
    }


def _pair_diagnostic(binding: Mapping[str, Any], result: Mapping[str, Any], baseline_root: str, treatment_root: str) -> Dict[str, Any]:
    evaluation = result["evaluation"]
    baseline, treatment = _trace(baseline_root), _trace(treatment_root)
    if binding["family"] == "R-HLC":
        bt, bxy, _, bs = trajectory_arrays_timestamp_aware(baseline)
        tt, txy, _, ts = trajectory_arrays_timestamp_aware(treatment)
        bp = hlc_realized_lane_transition_progress_v1_0(
            source_reference_xy=binding["source_reference_xy"], target_reference_xy=binding["target_reference_xy"], realized_ego_xy=bxy
        )
        tp = hlc_realized_lane_transition_progress_v1_0(
            source_reference_xy=binding["source_reference_xy"], target_reference_xy=binding["target_reference_xy"], realized_ego_xy=txy
        )
        bm = calculate_hlc_option_b_v2_timestamp_aware(bt, bp["clipped_progress_for_frozen_mechanism"], bs)
        tm = calculate_hlc_option_b_v2_timestamp_aware(tt, tp["clipped_progress_for_frozen_mechanism"], ts)
        endpoint = evaluation["endpoint"]
        engineering = evaluation["engineering"]
        engineering_pass = all(
            arm["max_abs_lateral_accel_mps2"] <= arm["frozen_limits"]["lateral_accel_mps2_max"]
            and arm["max_abs_yaw_rate_radps"] <= arm["frozen_limits"]["yaw_rate_radps_max"]
            and arm["max_abs_curvature_inv_m"] <= arm["frozen_limits"]["curvature_inv_m_max"]
            for arm in engineering.values()
        )
        mechanism_margins = {
            "treatment_retreat_count_minus_1": float(tm["hesitation_retreat_count"] - 1),
            "latency_delta_minus_0p5_s": float(evaluation["mechanism"]["delta_commit_latency_s"] - 0.5),
            "monotonic_delta_beyond_minus_0p10": float(-0.10 - evaluation["mechanism"]["delta_monotonic_fraction"]),
        }
        detail = {
            "baseline_measurement": bm, "treatment_measurement": tm,
            "endpoint": endpoint, "engineering": engineering,
            "mechanism_margins": mechanism_margins,
            "endpoint_pass": bool(endpoint["baseline"]["pass"] and endpoint["treatment"]["pass"]),
            "engineering_pass": engineering_pass,
        }
    else:
        bt, _, _, bs = trajectory_arrays_timestamp_aware(baseline)
        tt, _, _, ts = trajectory_arrays_timestamp_aware(treatment)
        bm = calculate_tsb_option_a_v2_timestamp_aware(bt, bs)
        tm = calculate_tsb_option_a_v2_timestamp_aware(tt, ts)
        detail = {
            "baseline_measurement": bm, "treatment_measurement": tm,
            "mechanism_margins": {
                "baseline_exactly_one_phase": 1.0 if bm["brake_phase_count"] == 1 else 0.0,
                "treatment_exactly_two_phases": 1.0 if tm["brake_phase_count"] == 2 else 0.0,
                "release_fraction_minus_0p15": None if tm.get("interstage_release_fraction") is None else float(tm["interstage_release_fraction"] - 0.15),
                "second_peak_ratio_minus_0p50": None if tm.get("second_brake_peak_ratio") is None else float(tm["second_brake_peak_ratio"] - 0.50),
            },
        }
    return {
        "pair_id": binding["pair_id"], "family": binding["family"],
        "mechanism_pass": bool(evaluation["mechanism"]["pass"]),
        "F_match_pass": bool(evaluation["f_match"]["pass"]),
        "safety_pass": bool(result["official_safety_pair_pass"]),
        "evaluation": evaluation, **detail,
    }


def _dist(values: Iterable[float | None]) -> Dict[str, Any]:
    array = np.asarray([float(v) for v in values if v is not None], dtype=np.float64)
    if not len(array):
        return {"n": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None}
    q = np.quantile(array, [0, .25, .5, .75, 1])
    return dict(zip(("min", "p25", "median", "p75", "max"), [round(float(v), 6) for v in q]), n=len(array))


def _summary(family: str, pairs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counts = {
        "pairs": 8,
        "mechanism_pass": sum(row["mechanism_pass"] for row in pairs),
        "F_match_pass": sum(row["F_match_pass"] for row in pairs),
        "safety_pass": sum(row["safety_pass"] for row in pairs),
    }
    if family == "R-HLC":
        counts.update({
            "endpoint_pass": sum(row["endpoint_pass"] for row in pairs),
            "engineering_pass": sum(row["engineering_pass"] for row in pairs),
        })
        margins = {
            key: _dist(row["mechanism_margins"][key] for row in pairs)
            for key in ("treatment_retreat_count_minus_1", "latency_delta_minus_0p5_s", "monotonic_delta_beyond_minus_0p10")
        }
        converged = all(counts[key] == 8 for key in ("mechanism_pass", "F_match_pass", "endpoint_pass", "engineering_pass")) and counts["safety_pass"] >= 4
    else:
        counts.update({
            "measurement_OK": sum(row["baseline_measurement"]["status"] == "OK" and row["treatment_measurement"]["status"] == "OK" for row in pairs),
            "baseline_one_phase": sum(row["baseline_measurement"]["brake_phase_count"] == 1 for row in pairs),
            "treatment_two_phase": sum(row["treatment_measurement"]["brake_phase_count"] == 2 for row in pairs),
        })
        margins = {
            key: _dist(row["mechanism_margins"][key] for row in pairs)
            for key in ("baseline_exactly_one_phase", "treatment_exactly_two_phases", "release_fraction_minus_0p15", "second_peak_ratio_minus_0p50")
        }
        converged = all(counts[key] == 8 for key in ("measurement_OK", "mechanism_pass", "F_match_pass")) and counts["safety_pass"] >= 4
    return {"counts": counts, "mechanism_margin_distributions": margins, "development_success": converged}


def execute_calibration(output_root: Path, recover_completed_hlc_round0: bool = False) -> Dict[str, Any]:
    roster, pair_doc, _, ledger = _load_frozen()
    if ledger["rounds"]:
        raise RuntimeError("R2_B_LEDGER_ALREADY_CONTAINS_ROUNDS")
    entry_by_token = {row["scenario_token"]: row for row in roster["entries"]}
    binding_by_id = {row["pair_id"]: row for row in pair_doc["pairs"]}
    all_rounds = []
    for family in ("R-HLC", "R-TSB"):
        parameters = _initial(family)
        for round_index in range(4):
            runs = _runs(family, round_index, roster)
            prefix = "hlc" if family == "R-HLC" else "tsb"
            parameter_file = ROUND_DIR / f"r2_b_{prefix}_round_{round_index}_parameters_v1.0.json"
            parameter_payload = {
                "schema_version": f"r2_b_{prefix}_round_{round_index}_parameters_v1.0",
                "status": "FROZEN_BEFORE_THIS_ROUND_SIMULATION",
                "family": family, "round_index": round_index,
                "parameters": parameters, "runs": runs,
                "global_no_identity_specific_parameters": True,
                "source": "R2_A_SURROGATE_INITIALIZATION_ONLY" if round_index == 0 else "DETERMINISTIC_PRE_FROZEN_AGGREGATE_UPDATE",
            }
            write_frozen_or_verify(parameter_file, parameter_payload)
            run_results = []
            for run in runs:
                effective_run = dict(run)
                if run["run_id"] == "R2B-HLC-R0-01-BASELINE" and ledger.get("technical_failures"):
                    effective_run["frozen_run_id"] = run["run_id"]
                    effective_run["run_id"] = "R2B-HLC-R0-01-BASELINE-TECHRERUN01"
                    effective_run["technical_rerun_reason"] = "PRE_SIMULATOR_INTERFACE_BINDING_FAILURE"
                family_round_root = output_root / prefix / f"round_{round_index}"
                existing_root = family_round_root / effective_run["run_id"]
                if recover_completed_hlc_round0 and family == "R-HLC" and round_index == 0:
                    run_results.append(_recover_completed_run(effective_run, existing_root))
                else:
                    run_results.append(_execute_one(effective_run, entry_by_token[run["scenario_token"]], parameters, parameter_file, family_round_root))
            pair_results = []
            for pair_id in [f"R2B-CAL-{prefix.upper()}-{i:02d}" for i in range(1, 9)]:
                pair_runs = [row for row in run_results if row["pair_id"] == pair_id]
                baseline = next(row for row in pair_runs if row["arm"] == ARM_BASELINE)
                treatment = next(row for row in pair_runs if row["arm"] == ARM_TREATMENT)
                binding = binding_by_id[pair_id]
                dispatched = evaluate_frozen_pair(
                    pair_binding=binding, baseline_run_dir=ROOT / baseline["run_root"], treatment_run_dir=ROOT / treatment["run_root"]
                )
                pair_results.append(_pair_diagnostic(binding, dispatched, baseline["run_root"], treatment["run_root"]))
            summary = _summary(family, pair_results)
            round_result = {
                "schema_version": f"r2_b_{prefix}_calibration_round_{round_index}_result_v1.0",
                "family": family, "round_index": round_index,
                "parameter_file": str(parameter_file.relative_to(ROOT)), "parameter_sha256": sha(parameter_file),
                "parameters": parameters, "runs": run_results, "pairs": pair_results, "summary": summary,
                "scientific_confirmation": False, "failed_results_deleted": False,
            }
            result_file = ROUND_DIR / f"r2_b_{prefix}_round_{round_index}_results_v1.0.json"
            write(result_file, round_result)
            round_result["result_file"] = str(result_file.relative_to(ROOT))
            round_result["result_sha256"] = sha(result_file)
            all_rounds.append(round_result)
            ledger["rounds"] = [
                {key: row[key] for key in ("family", "round_index", "parameter_file", "parameter_sha256", "result_file", "result_sha256", "summary")}
                for row in all_rounds
            ]
            ledger["status"] = "R2_B_DEV_CALIBRATION_IN_PROGRESS"
            ledger["actual_engineering_runs"] = sum(len(row["runs"]) for row in all_rounds)
            write(LEDGER, ledger, update=True)
            if summary["development_success"]:
                break
            if round_index < 3:
                parameters = _next(family, parameters, summary)
    ledger["status"] = "R2_B_DEV_CALIBRATION_EXECUTION_COMPLETE"
    write(LEDGER, ledger, update=True)
    return {"rounds": all_rounds, "ledger": ledger}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--recover-completed-hlc-round0", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/r2_b_controller_aware_calibration_v1")
    args = parser.parse_args()
    if not args.execute:
        roster, pairs, _, _ = _load_frozen()
        print(json.dumps({"status": "R2_B_ZERO_RUN_INPUT_CLOSURE_PASS", "roster": len(roster["entries"]), "pairs": len(pairs["pairs"]), "simulation": 0}))
        return 0
    result = execute_calibration(args.output_root.resolve(), args.recover_completed_hlc_round0)
    print(json.dumps({"status": result["ledger"]["status"], "rounds": len(result["rounds"]), "actual_engineering_runs": result["ledger"]["actual_engineering_runs"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
