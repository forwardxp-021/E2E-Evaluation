#!/usr/bin/env python3
"""Reproduce B1-IR3 static native-route closure without any scientific execution."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.b1_ir2_adjudication import B1_DOCS, B1_OUTPUT, PARAMETERS, ROOT, SPECS, sha256_file
from tools.b1_native_route_infrastructure import (
    VERSION, build_native_route_reference_v1_2, production_route_precheck_v1_2,
)
from tools.r1_closed_loop_benchmark_v2_1 import build_native_route_reference_v1_1

BASELINE_COMMIT = "b8673502ae9b69c752ba197dc9c2480cf11afe49"
ZERO_RUN = {name: 0 for name in (
    "SIMULATION", "RUNNER_RUN", "TSB_ROLLOUT", "HLC_ROLLOUT", "RBR_TRAINING",
    "PRIMARY_EVALUATION", "V_EXECUTION", "C_EXECUTION", "NEW_SCIENTIFIC_OUTCOME_EXPOSURE",
)}


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def reference_bytes(result: Mapping[str, Any]) -> bytes:
    """Exact ndarray bytes plus every returned metadata field, with no rounding."""
    array = np.asarray(result["reference_xy"])
    metadata = {key: value for key, value in result.items() if key != "reference_xy"}
    return canonical({"dtype": array.dtype.str, "shape": list(array.shape), "metadata": metadata}) + b"\n" + array.tobytes(order="C")


def historical_hashes() -> dict[str, str]:
    files = list(B1_OUTPUT.rglob("*"))
    for directory in ("b1_tsb_qualification", "b1_infrastructure_remediation", "b1_ir2_adjudication"):
        files.extend((ROOT / "docs/stageR" / directory).rglob("*"))
    files.append(ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv")
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(set(files)) if path.is_file()}


def source_proof() -> Mapping[str, Any]:
    immutable = [
        "tools/r1_closed_loop_benchmark_v2_1.py", "tools/r2_b_controller_aware_generator_v1.py",
        "tools/r2_bj_b0_2_passive_actual_lqr_recorder.py", "tools/s1_protocol_schema.py",
        "tools/r1_b2_8_r3_2_post_run_evaluator_dispatcher.py", "tools/r1_b2_8_r3_1_official_safety_adapter.py",
        "tools/r1_official_metric_canonicalizer.py", str(PARAMETERS.relative_to(ROOT)),
    ]
    results = {}
    for name in immutable:
        before = subprocess.check_output(["git", "show", f"{BASELINE_COMMIT}:{name}"], cwd=ROOT)
        after = (ROOT / name).read_bytes()
        if before != after:
            raise RuntimeError(f"FROZEN_SCIENTIFIC_SOURCE_CHANGED:{name}")
        results[name] = {"unchanged": True, "sha256": hashlib.sha256(after).hexdigest()}
    bindings = {}
    for name in ("tools/r2_b_controller_aware_planner_v1.py", "tools/b1_tsb_qualification_executor.py"):
        before = subprocess.check_output(["git", "show", f"{BASELINE_COMMIT}:{name}"], cwd=ROOT).decode()
        after = (ROOT / name).read_text()
        # This exact additional fail-closed authorization check is the only
        # permitted executor body change beyond the route import binding.
        authorization_guard = '    if not {"tools/b1_native_route_infrastructure.py", "tools/r1_closed_loop_benchmark_v2_1.py"}.issubset(frozen_files):\n        raise B1ExecutionError("B1_ROUTE_INFRASTRUCTURE_SOURCE_BINDING_MISSING")\n'
        comparison_after = after.replace(authorization_guard, "") if name.endswith("qualification_executor.py") else after
        def non_import_ast(source: str) -> str:
            tree = ast.parse(source)
            tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
            return ast.dump(tree, include_attributes=False)
        if non_import_ast(before) != non_import_ast(comparison_after):
            raise RuntimeError(f"ROUTE_BINDING_ONLY_CHANGE_REQUIRED:{name}")
        bindings[name] = {"route_import_and_exact_source_authorization_guard_only": True,
                          "scientific_function_bodies_ast_identical": True,
                          "before_sha256": hashlib.sha256(before.encode()).hexdigest(), "after_sha256": sha256_file(ROOT / name)}
    return {"immutable_scientific_files": results, "infrastructure_import_bindings": bindings}


def audit_queries(specs: list[Mapping[str, Any]], map_cache: Mapping[str, Any]) -> list[dict[str, Any]]:
    from tools.r2_b_controller_aware_planner_v1 import _tsb_required_distance
    parameters = json.loads(PARAMETERS.read_text())["parameters"]
    rows = []
    for spec in specs:
        trace = B1_OUTPUT / spec["run_id"] / "trace/realized_current_ego.jsonl"
        if not trace.is_file():
            continue
        for line in trace.read_text().splitlines():
            row = json.loads(line)
            iteration = int(row["iteration_index"])
            absolute = iteration * 0.1
            effective_arm = "BASELINE" if absolute < 1.1 - 1e-9 else spec["arm"]
            current = row["current_ego"]
            required = _tsb_required_distance(current["speed_mps"], absolute, effective_arm, parameters)
            args = (map_cache[spec["map_name"]], spec["route_roadblock_ids"], current, required)
            try:
                old = build_native_route_reference_v1_1(*args)
            except ValueError as error:
                old, old_error = None, str(error)
            else:
                old_error = ""
            new = build_native_route_reference_v1_2(*args)
            old_bytes = reference_bytes(old) if old is not None else None
            new_bytes = reference_bytes(new)
            if old_bytes is not None and old_bytes != new_bytes:
                raise RuntimeError(f"HISTORICAL_REFERENCE_CHANGED:{spec['run_id']}:{iteration}")
            rows.append({
                "pair_id": spec["pair_id"], "arm_id": spec["run_id"], "iteration": iteration,
                "route_id": spec["route_id"], "precontext_id": spec["precontext_id"],
                "frozen_trace_sha256": sha256_file(trace),
                "route_input_sha256": hashlib.sha256(canonical({"current": current, "required_forward_m": required, "route": spec["route_roadblock_ids"]})).hexdigest(),
                "required_forward_m": required,
                "pre_status": "PASS" if old is not None else "INFRASTRUCTURE_FAILURE",
                "pre_failure": old_error, "post_status": "PASS",
                "pre_reference_sha256": hashlib.sha256(old_bytes).hexdigest() if old_bytes is not None else "",
                "post_reference_sha256": hashlib.sha256(new_bytes).hexdigest(),
                "full_return_byte_identical": "YES" if old_bytes == new_bytes else "NOT_APPLICABLE_OLD_FAILURE",
                "new_native_edge_ids": json.dumps(new["native_edge_ids"], separators=(",", ":")),
                "builder_version_returned": new["builder_version"],
                "max_junction_gap_m": new.get("native_successor_repair", {}).get("max_junction_gap_m", ""),
            })
    if len(rows) != 2774 or sum(row["pre_status"] == "PASS" for row in rows) != 2773:
        raise RuntimeError("FROZEN_CALL_ENTRY_CARDINALITY_MISMATCH")
    return rows


def topology_evidence(api: Any) -> Mapping[str, Any]:
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as Layer
    rows = []
    for roadblock_id in ("19000", "19512", "18999", "19339"):
        roadblock = api.get_map_object(roadblock_id, Layer.ROADBLOCK) or api.get_map_object(roadblock_id, Layer.ROADBLOCK_CONNECTOR)
        for edge in sorted(roadblock.interior_edges, key=lambda edge: str(edge.id)):
            rows.append({"roadblock_id": roadblock_id, "native_edge_id": str(edge.id),
                         "baseline_start_xy": list(edge.baseline_path.linestring.coords[0]),
                         "baseline_end_xy": list(edge.baseline_path.linestring.coords[-1]),
                         "native_outgoing": [{"edge_id": str(child.id), "roadblock_id": str(child.get_roadblock_id())} for child in sorted(edge.outgoing_edges, key=lambda item: str(item.id))]})
    return {"native_graph": rows, "old_greedy_prefix": ["19812", "20202", "18519"],
            "legal_repaired_prefix": ["19812", "20203", "18520", "19953"],
            "same_ordered_roadblocks": ["19000", "19512", "18999", "19339"]}


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def generate(output_dir: Path) -> Mapping[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"IR3_FRESH_OUTPUT_DIRECTORY_REQUIRED:{output_dir}")
    from tools.r1_b2_8_r3_prospective_selector import official_env, MAPS
    official_env()
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    from tools.s2r_production_executor import precontext_id, route_id
    before = historical_hashes()
    protected = "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"
    if before[protected] != "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8":
        raise RuntimeError("PROTECTED_ASSET_SHA_MISMATCH")
    sources = source_proof()
    specs = json.loads(SPECS.read_text())["arms"]
    if len(specs) != 40 or len({spec["run_id"] for spec in specs}) != 40 or len({spec["pair_id"] for spec in specs}) != 20 or len({spec["session_id"] for spec in specs}) != 20:
        raise RuntimeError("ORIGINAL_20_PAIR_DENOMINATOR_REQUIRED")
    cache = {name: get_maps_api(str(MAPS), "nuplan-maps-v1.0", name) for name in sorted({spec["map_name"] for spec in specs})}
    for spec in specs:
        if precontext_id(spec["precontext"]) != spec["precontext_id"] or route_id(spec["map_name"], spec["route_roadblock_ids"]) != spec["route_id"]:
            raise RuntimeError(f"FROZEN_IDENTITY_MISMATCH:{spec['run_id']}")
    rows = audit_queries(specs, cache)
    gates = {spec["run_id"]: production_route_precheck_v1_2(spec, cache[spec["map_name"]]) for spec in specs}
    if any(gate["route_precheck_status"] != "COMPATIBLE" for gate in gates.values()):
        raise RuntimeError("IR3_STATIC_GATE_NOT_CLOSED; preserve ambiguity and inspect evidence")
    ledger = json.loads((B1_DOCS / "B1_Scientific_Arm_Budget_Ledger_v1.json").read_text())["runs"]
    plan = []
    for spec in specs:
        run_id = spec["run_id"]
        number = int(spec["pair_id"][-2:])
        action = "NO_RERUN;PRESERVE_FROZEN_SCIENTIFIC_PASS" if number <= 17 else (
            "ATTEMPT_1_RETAINED;ATTEMPT_2_ELIGIBLE_ONLY_IF_OWNER_AUTHORIZED" if run_id == "B1-TSB-18-TREATMENT" else "FIRST_ATTEMPT_ELIGIBLE_ONLY_IF_OWNER_AUTHORIZED")
        plan.append({"pair_id": spec["pair_id"], "arm_id": run_id, "arm": spec["arm"],
                     "route_id": spec["route_id"], "precontext_id": spec["precontext_id"],
                     "original_state": ledger[run_id], "route_status": gates[run_id]["route_precheck_status"],
                     "next_action": action, "execution_authorized": False,
                     "max_total_technical_attempts": 2, "planned_pair_denominator": 20})
    after = historical_hashes()
    if before != after:
        raise RuntimeError("HISTORICAL_ARTIFACT_MUTATION_DETECTED")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "B1_IR3_Pre_Post_Equivalence_Audit_v1.csv", rows)
    write_csv(output_dir / "B1_IR3_Resume_Execution_Plan_v3.csv", plan)
    evidence = {"schema_version": "B1_IR3_Static_Evidence_v1", "baseline_git_sha": BASELINE_COMMIT,
                "successor_version": VERSION, "zero_run_proof": ZERO_RUN, "source_proof": sources,
                "topology": topology_evidence(cache["us-pa-pittsburgh-hazelwood"]), "gates": gates,
                "historical_artifact_sha256": before,
                "map_sha256": {str(path.relative_to(ROOT.parent)): sha256_file(path) for path in sorted((MAPS / "us-pa-pittsburgh-hazelwood").rglob("*.gpkg"))},
                "summary": {"old_success_byte_identical": 2773, "old_failure_repaired": 1,
                            "existing17_reference_queries": 2720, "focus_06_10_14_queries": 480,
                            "static_compatible_arms": 40, "new_simulations": 0}}
    (output_dir / "B1_IR3_Static_Evidence_v1.json").write_text(json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    templates_path = ROOT / "tools/b1_ir3_report_templates_v1.json"
    for filename, content in json.loads(templates_path.read_text(encoding="utf-8")).items():
        if Path(filename).name != filename or not filename.startswith("B1_IR3_") or not filename.endswith(".md"):
            raise RuntimeError(f"INVALID_IR3_REPORT_TEMPLATE_NAME:{filename}")
        (output_dir / filename).write_text(content, encoding="utf-8")
    sources_to_bind = [
        "tools/b1_native_route_infrastructure.py", "tools/b1_ir3_route_closure.py",
        "tools/b1_ir3_report_templates_v1.json", "tools/r2_b_controller_aware_planner_v1.py",
        "tools/b1_tsb_qualification_executor.py", "tests/test_b1_ir3_route_closure.py",
    ]
    ir2_manifest_path = ROOT / "docs/stageR/b1_ir2_adjudication/B1_IR2_Manifest_v1.json"
    ir2_manifest = json.loads(ir2_manifest_path.read_text())
    for filename, expected in ir2_manifest["input_sha256"].items():
        if sha256_file(ROOT / filename) != expected:
            raise RuntimeError(f"GOVERNING_INPUT_CHANGED:{filename}")
    manifest = {
        "schema_version": "B1_IR3_Manifest_v1",
        "status": "B1_IR3_READY_FOR_RESUME_OWNER_REVIEW",
        "source_git_sha_before_ir3": BASELINE_COMMIT,
        "source_successor_version": VERSION,
        "scientific_contract_change_required": False,
        "governing_b1_status": "B1_TSB_BENCHMARK_QUALIFICATION_INCOMPLETE_INFRASTRUCTURE",
        "planned_pairs": 20, "planned_arms": 40,
        "existing17_scientific_pass_preserved": True,
        "B1_01_TO_17_RERUN": "FORBIDDEN",
        "B1_18_original_failure_input": "FIXED_BY_EXISTING_NATIVE_SUCCESSOR_BACKTRACKING",
        "B1_20_route_status": "COMPATIBLE",
        "runtime_evidence_scope": "INITIAL_PRODUCTION_QUERY_AND_COMPLETE_NATIVE_SUFFIX;NO_NEW_CLOSED_LOOP_RESULT",
        "remaining_uncertainties": [
            "No new full closed-loop execution; no claim of success for all theoretical future ego states",
            "Actual off-route, route-end coverage or other runtime failures still stop and remain in denominator",
            "New Owner authorization must bind commit, source hashes, original identities, remaining budgets and new attempt namespaces",
        ],
        "authorization": {name: "NOT_AUTHORIZED" for name in ("B1_RESUME_EXECUTION", "RBR_TRAINING", "PRIMARY_EVALUATION")},
        "zero_run_proof": ZERO_RUN,
        "max_total_technical_attempts_per_arm": 2,
        "summary": evidence["summary"],
        "source_sha256": {name: sha256_file(ROOT / name) for name in sources_to_bind},
        "governing_input_sha256": {**ir2_manifest["input_sha256"], str(ir2_manifest_path.relative_to(ROOT)): sha256_file(ir2_manifest_path)},
        "artifact_sha256": {path.name: sha256_file(path) for path in sorted(output_dir.iterdir()) if path.is_file()},
        "protected_asset": {"path": "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv",
                            "sha256": "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8"},
        "validation": {
            "suite": "tests/test_b1_ir3_route_closure.py plus B1-IR2, B1-IR and B1 regression suites",
            "runtime_guard": "runner.run, planner trajectory generation, controller update, LQR track and motion-model propagation forbidden in tests",
            "equivalence_rule": "dtype,shape,ndarray bytes and all returned metadata byte equality; no rounding/tolerance",
            "deterministic_regeneration": "Generate to a new directory and compare every report/CSV/JSON/manifest byte",
            "compile_commands": ["python3 -m py_compile tools/*.py", "nuplan Python -m py_compile modified tools/tests"],
            "additional_checks": ["tools/check_no_tmp_dependencies.py", "git diff --check", "historical asset hashes before/after"],
        },
    }
    (output_dir / "B1_IR3_Manifest_v1.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    evidence = generate(args.output_dir)
    print(json.dumps(evidence["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
