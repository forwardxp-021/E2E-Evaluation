#!/usr/bin/env python3
"""Zero-run B1-IR2 recovered-safety and route adjudication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.b1_native_route_precheck import MAP_ROOT, precheck_with_production_builder
from tools.r1_b2_8_r3_1_official_safety_adapter import (
    FROZEN_PAIR_PASS_SEMANTICS,
    adapt_official_safety,
)

ROOT = Path(__file__).resolve().parents[1]
B1_DOCS = ROOT / "docs/stageR/b1_tsb_qualification"
IR_DOCS = ROOT / "docs/stageR/b1_infrastructure_remediation"
B1_OUTPUT = ROOT / "outputs/stageR/b1_tsb_qualification_v1"
SPECS = B1_DOCS / "B1_TSB_Qualification_Arm_Specs_v1.json"
PAIR_RESULTS = B1_DOCS / "B1_TSB_Qualification_Pair_Results_v1.csv"
LEDGER = B1_DOCS / "B1_Scientific_Arm_Budget_Ledger_v1.json"
RECOVERY_AUDIT = IR_DOCS / "B1_IR_Offline_Finalize_Audit_v1.json"
PARAMETERS = ROOT / "docs/stageR/r2/r2_b_calibration_rounds/r2_b_tsb_round_0_parameters_v1.0.json"
SAFETY_ADAPTER = ROOT / "tools/r1_b2_8_r3_1_official_safety_adapter.py"
CANONICALIZER = ROOT / "tools/r1_official_metric_canonicalizer.py"
CASES = (6, 10, 14, 18, 20)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise RuntimeError(f"EMPTY_CSV_FORBIDDEN:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)


def adjudicate_safety() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply the already-frozen official adapter to 34 recovered arms."""
    audit = json.loads(RECOVERY_AUDIT.read_text(encoding="utf-8"))
    by_run = {row["run_id"]: row for row in audit["arms"]}
    arm_rows: list[dict[str, Any]] = []
    for pair_number in range(1, 18):
        pair_id = f"B1-TSB-{pair_number:02d}"
        for arm in ("BASELINE", "TREATMENT"):
            run_id = f"{pair_id}-{arm}"
            recovered = by_run[run_id]
            root = B1_OUTPUT / run_id
            adapted = adapt_official_safety(root)
            collision = adapted["canonical_payload"]["collision"]["number_of_all_at_fault_collisions_stat_value"]
            drivable = adapted["canonical_payload"]["drivable_area"]["drivable_area_compliance_stat_value"]
            provenance = adapted["raw_metric_artifact_provenance"]
            for artifact in provenance.values():
                path = root / artifact["relative_path"]
                if sha256_file(path) != recovered["output_sha256"][path.stem]:
                    raise RuntimeError(f"SAFETY_ARTIFACT_HASH_MISMATCH:{run_id}:{path.name}")
            for relative, expected in recovered["original_artifact_sha256"].items():
                if sha256_file(root / relative) != expected:
                    raise RuntimeError(f"ORIGINAL_ARTIFACT_HASH_MISMATCH:{run_id}:{relative}")
            arm_rows.append(
                {
                    "arm_id": run_id,
                    "pair_id": pair_id,
                    "arm": arm,
                    "collision_metric_available": "YES",
                    "collision_count": collision,
                    "collision_status": "PASS" if collision == 0 else "FAIL",
                    "drivable_metric_available": "YES",
                    "drivable_value": str(drivable).lower(),
                    "drivable_status": "PASS" if drivable else "FAIL",
                    "joint_arm_safety": "PASS" if adapted["frozen_arm_safety_pass"] else "FAIL",
                    "collision_metric_path": str(root / provenance["collision"]["relative_path"]),
                    "collision_metric_sha256": provenance["collision"]["artifact_provenance_sha256"],
                    "drivable_metric_path": str(root / provenance["drivable_area"]["relative_path"]),
                    "drivable_metric_sha256": provenance["drivable_area"]["artifact_provenance_sha256"],
                    "canonical_payload_sha256": adapted["canonical_payload_sha256"],
                    "original_execution_manifest_sha256": recovered["original_artifact_sha256"]["execution_manifest.json"],
                    "metric_config_sha256": audit["official_metric_config_sha256"],
                    "offline_finalize_tool_sha256": audit["offline_finalize_tool_sha256"],
                    "safety_adapter_sha256": sha256_file(SAFETY_ADAPTER),
                    "canonicalizer_sha256": sha256_file(CANONICALIZER),
                    "adjudication_rule": FROZEN_PAIR_PASS_SEMANTICS,
                }
            )
    pair_rows: list[dict[str, Any]] = []
    for pair_number in range(1, 18):
        pair_id = f"B1-TSB-{pair_number:02d}"
        arms = {row["arm"]: row for row in arm_rows if row["pair_id"] == pair_id}
        pair_pass = arms["BASELINE"]["joint_arm_safety"] == "PASS" and arms["TREATMENT"]["joint_arm_safety"] == "PASS"
        pair_rows.append(
            {
                "pair_id": pair_id,
                "baseline_safety": arms["BASELINE"]["joint_arm_safety"],
                "treatment_safety": arms["TREATMENT"]["joint_arm_safety"],
                "pair_safety": "PAIR_SAFETY_PASS" if pair_pass else "PAIR_SAFETY_FAIL",
                "adjudication_rule": FROZEN_PAIR_PASS_SEMANTICS,
            }
        )
    return arm_rows, pair_rows


def audit_arm_lifecycle(spec: Mapping[str, Any], recovered: Mapping[str, Any]) -> Mapping[str, bool]:
    """Audit preserved execution evidence without relabeling the old manifest."""
    from tools.s2r_production_executor import precontext_id, route_id
    root = B1_OUTPUT / str(spec["run_id"])
    manifest = json.loads((root / "execution_manifest.json").read_text(encoding="utf-8"))
    trace = [json.loads(line) for line in (root / "trace/realized_current_ego.jsonl").read_text().splitlines()]
    telemetry = [json.loads(line) for line in (root / "telemetry/planner_transfer.jsonl").read_text().splitlines()]
    controller = [json.loads(line) for line in (root / "telemetry/actual_lqr_controller_telemetry.jsonl").read_text().splitlines()]
    logs = list((root / "raw/simulation_log").rglob("*.msgpack.xz"))
    identity_fields = ("run_id", "pair_id", "arm", "session_id", "log_id", "scenario_token", "precontext_id", "route_id")
    original_hash_valid = all(sha256_file(root / rel) == expected for rel, expected in recovered["original_artifact_sha256"].items())
    metric_hash_valid = all(sha256_file(root / "raw/metrics" / f"{name}.parquet") == expected for name, expected in recovered["output_sha256"].items())
    return {
        "runner_complete": manifest["execution_status"] == "TECHNICAL_COMPLETE_PENDING_PAIR_ANALYSIS" and [row["iteration_index"] for row in trace] == list(range(80)) and [row["iteration"] for row in telemetry] == list(range(80)),
        "recorder_complete": [row["iteration"] for row in controller] == list(range(79)) and all(row["run_id"] == spec["run_id"] and row["pair_id"] == spec["pair_id"] for row in controller),
        "official_metrics_complete": len(recovered["output_sha256"]) == 16 and metric_hash_valid,
        "serializer_complete": len(logs) == 1 and logs[0].stat().st_size > 0,
        "manifest_complete": all(manifest.get(key) == spec[key] for key in identity_fields),
        "artifact_hash_valid": original_hash_valid and metric_hash_valid,
        "precontext_valid": manifest["precontext_id"] == precontext_id(spec["precontext"]),
        "route_identity_valid": manifest["route_id"] == route_id(spec["map_name"], spec["route_roadblock_ids"]),
    }


def full_contract_rows(pair_safety: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join unchanged mechanism/F-match results with audited original evidence."""
    old = {row["pair_id"]: row for row in _read_csv(PAIR_RESULTS)}
    safety = {row["pair_id"]: row for row in pair_safety}
    specs = {row["run_id"]: row for row in json.loads(SPECS.read_text())["arms"]}
    audit = {row["run_id"]: row for row in json.loads(RECOVERY_AUDIT.read_text())["arms"]}
    rows: list[dict[str, Any]] = []
    for pair_number in range(1, 18):
        pair_id = f"B1-TSB-{pair_number:02d}"
        source = old[pair_id]
        arm_gates = [audit_arm_lifecycle(specs[f"{pair_id}-{arm}"], audit[f"{pair_id}-{arm}"]) for arm in ("BASELINE", "TREATMENT")]
        technical = {key: all(g[key] for g in arm_gates) for key in arm_gates[0]}
        technical.update(precontext_match=source["precontext_match"] == "true", route_identity_match=source["route_match"] == "true")
        scientific = {
            "mechanism_valid": source["joint_mechanism_status"] == "PASS" and source["LOW_SPEED_ENDSTOP"] == "false",
            "f_match_valid": source["F_match_status"] == "F_MATCH_PASS",
            "safety_valid": safety[pair_id]["pair_safety"] == "PAIR_SAFETY_PASS",
        }
        complete = all(technical.values())
        rows.append({
            "pair_id": pair_id,
            **{key: "YES" if value else "NO" for key, value in {**technical, **scientific}.items()},
            "full_contract_status": "FULL_CONTRACT_COMPLETE" if complete else "AMBIGUOUS",
            "scientific_status": ("SCIENTIFIC_PASS" if all(scientific.values()) else "SCIENTIFIC_FAIL") if complete else "AMBIGUOUS",
            "historical_execution_valid": "YES" if complete else "AMBIGUOUS",
            "lifecycle_basis": "AUDITED_ORIGINAL_EXECUTION_PLUS_OFFLINE_METRICS;NO_NEW_ARM_COMPLETE_MANIFEST",
        })
    return rows


def forward_capacity_probe(spec: Mapping[str, Any], map_api: Any, current: Mapping[str, Any]) -> Mapping[str, Any]:
    """Locate the failing forward-request boundary using only the exact builder."""
    from tools.r1_closed_loop_benchmark_v2_1 import build_native_route_reference_v1_1
    lower, upper = 0.0, 1_000_000.0
    result = None
    for _ in range(36):
        middle = (lower + upper) * 0.5
        try:
            candidate = build_native_route_reference_v1_1(map_api, spec["route_roadblock_ids"], current, middle)
        except ValueError as error:
            if not str(error).startswith("NATIVE_ROUTE_FAIL:"):
                raise
            upper = middle
        else:
            lower, result = middle, candidate
    return {
        "max_resolvable_forward_m_lower_bound": lower,
        "max_resolvable_forward_m_upper_bound": upper,
        "last_resolvable_native_edge_ids": [] if result is None else result["native_edge_ids"],
        "boundary_probe_method": "36_BISECTIONS_OF_EXACT_PRODUCTION_BUILDER;NO_TRAJECTORY_GENERATION",
    }


def replay_route(spec: Mapping[str, Any], map_api: Any) -> Mapping[str, Any]:
    """Replay exact production route calls from already-recorded current states."""
    from tools.r1_closed_loop_benchmark_v2_1 import build_native_route_reference_v1_1
    from tools.r2_b_controller_aware_planner_v1 import _tsb_required_distance

    parameters = json.loads(PARAMETERS.read_text(encoding="utf-8"))["parameters"]
    run_id = str(spec["run_id"])
    trace = B1_OUTPUT / run_id / "trace/realized_current_ego.jsonl"
    original_rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()] if trace.is_file() else []
    successes: list[Mapping[str, Any]] = []
    failures: list[Mapping[str, Any]] = []
    if trace.is_file():
        for row in original_rows:
            iteration = int(row["iteration_index"])
            absolute = iteration * 0.1
            effective_arm = "BASELINE" if absolute < 1.1 - 1e-9 else str(spec["arm"])
            current = row["current_ego"]
            required = _tsb_required_distance(float(current["speed_mps"]), absolute, effective_arm, parameters)
            try:
                result = build_native_route_reference_v1_1(map_api, spec["route_roadblock_ids"], current, required)
                successes.append(
                    {
                        "iteration": iteration,
                        "route_cursor": int(result["route_occurrence_cursor"]),
                        "max_required_route_index": int(result["route_occurrence_cursor"]) + len(result["native_edge_ids"]) - 1,
                        "native_edge_ids": result["native_edge_ids"],
                        "required_forward_m": required,
                    }
                )
            except ValueError as error:
                failures.append(
                    {
                        "iteration": iteration,
                        "failure_code": str(error),
                        "current_ego": current,
                        "required_forward_m": required,
                    }
                )
    route = [str(value) for value in spec["route_roadblock_ids"]]
    problematic = [index for index, value in enumerate(route) if value == "19339"]
    return {
        "trace_available": trace.is_file(),
        "last_recorded_input_capacity": forward_capacity_probe(spec, map_api, original_rows[-1]["current_ego"]) if original_rows else None,
        "trace_rows": len(successes) + len(failures),
        "successful_production_route_calls": len(successes),
        "failed_production_route_calls": len(failures),
        "runtime_cursor_min": min((row["route_cursor"] for row in successes), default=None),
        "runtime_cursor_max": max((row["route_cursor"] for row in successes), default=None),
        "runtime_max_required_route_index": max((row["max_required_route_index"] for row in successes), default=None),
        "problematic_successor_route_indices": problematic,
        "first_failure": failures[0] if failures else None,
        "route_segment_sequence": route,
        "successful_calls": successes,
        "trace_sha256": sha256_file(trace) if trace.is_file() else None,
        "planning_horizon_s": 7.9,
        "realized_trace_last_nominal_time_s": (len(successes) + len(failures) - 1) * 0.1 if trace.is_file() else None,
        "successor_traversal_native_edge_ids": sorted({edge for row in successes for edge in row["native_edge_ids"]}),
    }


def route_case_rows(specs: list[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    from tools.r1_b2_8_r3_prospective_selector import official_env

    official_env()
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}
    for pair_number in CASES:
        for arm in ("BASELINE", "TREATMENT"):
            run_id = f"B1-TSB-{pair_number:02d}-{arm}"
            spec = next(row for row in specs if row["run_id"] == run_id)
            map_name = str(spec["map_name"])
            if map_name not in cache:
                cache[map_name] = get_maps_api(str(MAP_ROOT), "nuplan-maps-v1.0", map_name)
            static = precheck_with_production_builder(spec, cache[map_name])
            replay = replay_route(spec, cache[map_name])
            from tools.b1_native_route_precheck import initial_current
            initial = initial_current(spec)
            detail[run_id] = {
                "frozen_route_id": spec["route_id"],
                "map_name": spec["map_name"],
                "initial_production_and_precheck_current_ego": initial,
                "production_and_precheck_initial_required_forward_m": static["initial_production_required_forward_m"],
                "old_precheck_required_forward_m": "POSITIVE_INFINITY",
                "initial_forward_capacity": forward_capacity_probe(spec, cache[map_name], initial),
                "static_precheck": static,
                "retrospective_production_replay": replay,
            }
            historical = "YES" if replay["trace_rows"] == 80 and replay["failed_production_route_calls"] == 0 else ("NO_ATTEMPT_1_INFRASTRUCTURE_FAILURE" if replay["failed_production_route_calls"] else "NOT_RUN")
            classification = "ROUTE_AMBIGUOUS" if static["route_precheck_status"] == "AMBIGUOUS" else f"ROUTE_{static['route_precheck_status']}"
            rows.append(
                {
                    "pair_id": str(spec["pair_id"]),
                    "arm_id": run_id,
                    "arm": arm,
                    "route_id": str(spec["route_id"]),
                    "route_identity_valid": "YES",
                    "initial_production_probe": static["initial_production_probe_status"],
                    "full_route_topology": static["latent_full_route_topology_status"],
                    "missing_successor_segment": static["missing_successor_segment"] or "",
                    "route_length": len(spec["route_roadblock_ids"]),
                    "problematic_route_indices": json.dumps(replay["problematic_successor_route_indices"]),
                    "runtime_trace_rows": replay["trace_rows"],
                    "runtime_successful_route_calls": replay["successful_production_route_calls"],
                    "runtime_failed_route_calls": replay["failed_production_route_calls"],
                    "runtime_cursor_range": f"{replay['runtime_cursor_min']}..{replay['runtime_cursor_max']}",
                    "runtime_max_required_route_index": replay["runtime_max_required_route_index"],
                    "first_failure_iteration": "" if replay["first_failure"] is None else replay["first_failure"]["iteration"],
                    "first_failure_code": "" if replay["first_failure"] is None else replay["first_failure"]["failure_code"],
                    "historical_execution_valid": historical,
                    "future_route_status": classification,
                    "future_runner_entry": "ALLOW_IF_OWNER_AUTHORIZED" if classification == "ROUTE_COMPATIBLE" else "DO_NOT_ENTER_RUNNER",
                }
            )
    return rows, detail


def resume_rows(specs: list[Mapping[str, Any]], route_cases: list[Mapping[str, Any]], all_route_results: Mapping[str, Any]) -> list[dict[str, Any]]:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))["runs"]
    route_by_run = {row["run_id"]: "ROUTE_" + row["route_precheck_status"] for row in all_route_results["arms"]}
    rows: list[dict[str, Any]] = []
    for spec in specs:
        run_id, pair_id = str(spec["run_id"]), str(spec["pair_id"])
        number = int(pair_id.rsplit("-", 1)[1])
        route_status = route_by_run[run_id]
        if number <= 17:
            action = "NO_RERUN;PRESERVE_FULL_CONTRACT_SCIENTIFIC_RESULT"
            owner = "NO"
        elif run_id == "B1-TSB-18-TREATMENT":
            action = "ATTEMPT_1_RETAINED;ATTEMPT_2_NOT_EXECUTABLE_UNDER_CURRENT_ROUTE_CONTRACT"
            owner = "YES_AFTER_ROUTE_CLOSURE"
        elif number == 18:
            action = "FIRST_ATTEMPT_NOT_EXECUTABLE_UNDER_CURRENT_ROUTE_CONTRACT"
            owner = "YES_AFTER_ROUTE_CLOSURE"
        elif number == 19:
            action = "READY_FOR_FIRST_ATTEMPT_IF_OWNER_AUTHORIZED"
            owner = "YES"
        else:
            action = "ORIGINAL_IDENTITY_RETAINED;TECHNICAL_INVALIDITY_DECISION_REQUIRED;DO_NOT_ENTER_RUNNER"
            owner = "YES_AFTER_ROUTE_CLOSURE"
        rows.append(
            {
                "pair_id": pair_id,
                "arm_id": run_id,
                "arm": spec["arm"],
                "current_attempt_state": ledger[run_id],
                "route_status": route_status,
                "next_action": action,
                "owner_authorization_required": owner,
                "denominator_rule": "ORIGINAL_20_PAIRS_UNCHANGED",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if (args.output_dir / "B1_IR2_Manifest_v1.json").exists():
        raise FileExistsError(f"FROZEN_IR2_OUTPUT_EXISTS:{args.output_dir}; use a new directory")
    specs = json.loads(SPECS.read_text(encoding="utf-8"))["arms"]
    arms, pairs = adjudicate_safety()
    full = full_contract_rows(pairs)
    route_cases, route_detail = route_case_rows(specs)
    from tools.b1_native_route_precheck import precheck_specs
    all_route_results = precheck_specs(SPECS)
    resume = resume_rows(specs, route_cases, all_route_results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "B1_IR2_Route_Precheck_v2.json").write_text(json.dumps(all_route_results, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    _write_csv(args.output_dir / "B1_IR2_Recovered_Safety_Arm_Results_v1.csv", arms)
    _write_csv(args.output_dir / "B1_IR2_Recovered_Safety_Pair_Results_v1.csv", pairs)
    _write_csv(args.output_dir / "B1_IR2_Existing17_Full_Contract_Status_v1.csv", full)
    _write_csv(args.output_dir / "B1_IR2_Route_Case_Comparison_v1.csv", route_cases)
    _write_csv(args.output_dir / "B1_IR2_Resume_Execution_Plan_v2.csv", resume)
    (args.output_dir / "B1_IR2_Route_Audit_v1.json").write_text(
        json.dumps(
            {
                "schema_version": "B1_IR2_Route_Audit_v1",
                "simulation_count": 0,
                "runner_run_count": 0,
                "planner_step_count": 0,
                "production_route_builder_sha256": sha256_file(ROOT / "tools/r1_closed_loop_benchmark_v2_1.py"),
                "cases": route_detail,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"official_safety_arm_pass": sum(row["joint_arm_safety"] == "PASS" for row in arms), "official_safety_pair_pass": sum(row["pair_safety"] == "PAIR_SAFETY_PASS" for row in pairs), "full_contract_pass": sum(row["scientific_status"] == "SCIENTIFIC_PASS" for row in full), "route_case_rows": len(route_cases)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
