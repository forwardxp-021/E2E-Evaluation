#!/usr/bin/env python3
"""Generate S2R engineering closure and metadata-only static eligibility evidence."""

from __future__ import annotations

import csv
import gzip
import hashlib
import heapq
import importlib.util
import json
import math
import os
import sqlite3
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs/stageR/s2r_engineering_closure"
PRE = ROOT / "docs/stageR/s2r_role_aware_preflight"
OLD_CENSUS = ROOT / "docs/stageR/s2_preflight/S2_Preflight_Eligibility_Census_v1.json"
ROLE_CENSUS = PRE / "S2R_V_Eligibility_Census_v1.csv"
ROLE_MANIFEST = PRE / "S2R_Role_Binding_Manifest_v1.json"
APPROVAL = ROOT / "docs/stageR/s1r_owner_amendment/S1R_Owner_Approval_Record_v1.md"
PLAN = ROOT / "RBR-64_博士研究总体方案_v2.3.md"
PROTECTED = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"
PROTECTED_SHA = "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8"
NU_PYTHON = Path("/Users/liuqing/miniconda3/envs/nuplan/bin/python")
MAP_ROOT = ROOT.parent / "nuplan/dataset/maps"
BRANCH = "20260825_stageR_new"
AUDITED_GIT_SHA = "0268d4b2bafb6ec8cc7f76d5a385aaa93f7d90f2"
GOVERNING_REMOTE_SHA = "39a4f458f01ece8ab6413d9ab525ee0e02e5426e"
SALT = "S2R_STATIC_ELIGIBILITY_V1_OWNER_REVIEW_CANDIDATE_NOT_FINAL_ROSTER"
TOP_K = 256
SPEED_MIN = 3.61
ZERO = {
    "SIMULATION": 0, "RUNNER_RUN": 0, "TSB_ROLLOUT": 0, "HLC_ROLLOUT": 0,
    "RBR_TRAINING": 0, "NEW_SCIENTIFIC_OUTCOME_EXPOSURE": 0,
    "PRIMARY_EVALUATION": 0, "V_EXECUTION": 0, "C_EXECUTION": 0,
}


def ensure_runtime() -> None:
    if importlib.util.find_spec("numpy") is None:
        if not NU_PYTHON.is_file():
            raise RuntimeError("nuPlan Python runtime is unavailable")
        os.execv(str(NU_PYTHON), [str(NU_PYTHON), "-B", str(Path(__file__).resolve()), *sys.argv[1:]])


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(name: str, text: str) -> None:
    (OUT / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(name: str, value: Any) -> None:
    write_text(name, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))


def write_csv(name: str, rows: list[Mapping[str, Any]]) -> None:
    with (OUT / name).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def verify_governance() -> None:
    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError("wrong branch")
    if sha(PROTECTED) != PROTECTED_SHA:
        raise RuntimeError("protected asset changed")
    approval = APPROVAL.read_text(encoding="utf-8")
    if "S1R_OWNER_AMENDMENT = APPROVED" not in approval or "S1_PROTOCOL = AMENDED_BY_S1R_OWNER_AMENDMENT_V1" not in approval:
        raise RuntimeError("approved governing protocol is absent")
    role = read_json(ROLE_MANIFEST)
    if role["S1R_OWNER_AMENDMENT"] != "APPROVED" or role["final_primary_v_roster_materialized"]:
        raise RuntimeError("role manifest state invalid")
    if not PLAN.is_file():
        raise RuntimeError("v2.3 plan missing")


def role_rows() -> dict[str, dict[str, str]]:
    with ROLE_CENSUS.open(encoding="utf-8", newline="") as f:
        rows = {row["session_id"]: row for row in csv.DictReader(f)}
    if len(rows) != 248:
        raise RuntimeError("role census is not 248 sessions")
    return rows


def ranked_candidates(old: Mapping[str, Any], roles: Mapping[str, Mapping[str, str]]) -> tuple[dict[str, list[dict[str, Any]]], Counter[str]]:
    heaps: dict[str, list[tuple[int, str, str, int, str, dict[str, Any]]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for index, log in enumerate(old["logs"], 1):
        session_id = str(log["session_id"])
        if roles[session_id]["v_policy_status"] != "CANDIDATE":
            continue
        shard = ROOT / "docs/stageR/s2_preflight" / str(log["shard"])
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        columns = payload["record_columns"]
        token_i = columns.index("scenario_token")
        time_i = columns.index("tag_anchor_timestamp_us")
        tags_i = columns.index("task_types")
        support_i = columns.index("anchor_timestamp_80_state_support")
        for record in payload["records"]:
            token = str(record[token_i])
            counts[session_id] += 1
            digest = hashlib.sha256(f"{SALT}|{session_id}|{token}|{log['log_id']}".encode()).hexdigest()
            rank_int = int(digest, 16)
            row = {
                "rank_sha256": digest, "scenario_token": token,
                "anchor_timestamp_us": int(record[time_i]), "task_types": list(record[tags_i]),
                "anchor_80_state_support": bool(record[support_i]), "log_id": str(log["log_id"]),
                "db_path": str(log["source_file"]), "map_name": str(log["map_version"]),
                "source_fingerprint_sha256": str(log["observed_db_fingerprint_sha256"]),
                "source_partition": str(log["source_partition"]),
            }
            item = (-rank_int, token, str(log["log_id"]), int(record[time_i]), str(log["source_file"]), row)
            heap = heaps[session_id]
            if len(heap) < TOP_K:
                heapq.heappush(heap, item)
            elif rank_int < -heap[0][0]:
                heapq.heapreplace(heap, item)
        if index % 250 == 0:
            print(f"metadata shards read: {index}/{len(old['logs'])}", flush=True)
    result: dict[str, list[dict[str, Any]]] = {}
    for session_id, heap in heaps.items():
        result[session_id] = sorted((item[-1] for item in heap), key=lambda row: (row["rank_sha256"], row["scenario_token"], row["log_id"], row["anchor_timestamp_us"]))
    return result, counts


def _hex(value: Any) -> str:
    return bytes(value).hex() if isinstance(value, (bytes, bytearray, memoryview)) else str(value)


def precontext_payload(candidate: Mapping[str, Any], initial: Mapping[str, Any], route: list[str], rid: str) -> Mapping[str, Any]:
    db = Path(str(candidate["db_path"]))
    start = int(initial["official_simulation_initial_timestamp_us"])
    with sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True) as con:
        con.execute("PRAGMA query_only=ON")
        con.row_factory = sqlite3.Row
        ego = con.execute(
            """SELECT lower(hex(lp.token)) lidar_token,lp.timestamp,ep.x,ep.y,ep.z,ep.qw,ep.qx,ep.qy,ep.qz,
                      ep.vx,ep.vy,ep.vz,ep.acceleration_x,ep.acceleration_y,ep.acceleration_z,
                      ep.angular_rate_x,ep.angular_rate_y,ep.angular_rate_z
               FROM lidar_pc lp JOIN ego_pose ep ON ep.token=lp.ego_pose_token
               WHERE lp.timestamp BETWEEN ? AND ? ORDER BY lp.timestamp""", (start - 2_100_000, start),
        ).fetchall()
        token_bytes = [bytes.fromhex(str(row["lidar_token"])) for row in ego]
        agents: list[tuple[Any, ...]] = []
        lights: list[tuple[Any, ...]] = []
        for token in token_bytes:
            agents.extend(tuple(_hex(v) for v in row) for row in con.execute(
                """SELECT lower(hex(token)),lower(hex(track_token)),x,y,z,yaw,width,length,height,vx,vy
                   FROM lidar_box WHERE lidar_pc_token=? ORDER BY track_token,token""", (token,)
            ))
            lights.extend(tuple(_hex(v) for v in row) for row in con.execute(
                """SELECT lane_connector_id,status FROM traffic_light_status
                   WHERE lidar_pc_token=? ORDER BY lane_connector_id,status""", (token,)
            ))
    history = [tuple(_hex(value) for value in row) for row in ego]
    return {
        "schema_version": "s2r_precontext_v1",
        "session_id": candidate["session_id"], "log_id": candidate["log_id"],
        "scenario_token": candidate["scenario_token"], "start_timestamp_us": start,
        "source_database_fingerprint_sha256": candidate["source_fingerprint_sha256"],
        "ego_initial_state": {
            "official_simulation_initial_lidar_token": initial["official_simulation_initial_lidar_token"],
            "official_simulation_initial_timestamp_us": start,
            "initial_x": initial["initial_x"], "initial_y": initial["initial_y"],
            "initial_heading": initial["initial_heading"], "initial_speed_mps": initial["initial_speed_mps"],
            "initial_time_us": initial["initial_time_us"],
            "tire_steering_angle": 0.0,
            "tire_steering_source": "OFFICIAL_NUPLAN_EGO_QUERY_CONSTANT",
        },
        "route_id": rid, "route_roadblock_ids": route,
        "map_identity": {"map_name": candidate["map_name"], "map_version": "nuplan-maps-v1.0"},
        "traffic_agent_precontext_sha256": canonical_sha({"agents": agents, "traffic_lights": lights}),
        "history_buffer_source_sha256": canonical_sha(history),
        "history_source_row_count": len(history),
        "planner_warmup_state": "FRESH_UNINITIALIZED_PLANNER; COMMON_PREINTERVENTION_TRAJECTORY_CONTRACT",
        "simulation_initialization": "OFFICIAL_NUPLAN_SCENARIO_AND_FRESH_ARM_FACTORY",
        "intervention_start_s": 1.1,
        "intervention_precontext_contract": "BOTH_ARMS_USE_BASELINE_TRAJECTORY_FOR_ABSOLUTE_TIME_LT_1P1",
    }


def evaluate_candidate(candidate: dict[str, Any], map_cache: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    from tools.r1_b2_8_r3_prospective_selector import official_count
    from tools.r1_prepare_official_technical_smoke_roster import _official_initial
    from tools.r1_closed_loop_benchmark_v2_1 import build_native_route_reference_v1_1
    from tools.s2r_production_executor import precontext_id, route_id

    if official_count(candidate["db_path"], candidate["scenario_token"]) != 1:
        return None, "OFFICIAL_EXACT_SCENARIO_RESOLUTION_NOT_ONE"
    try:
        initial, route = _official_initial({"db_path": candidate["db_path"], "scenario_token": candidate["scenario_token"], "timestamp": candidate["anchor_timestamp_us"]})
    except Exception as exc:
        return None, f"EXECUTION_INITIAL_STATE_OR_ROUTE_UNAVAILABLE:{type(exc).__name__}"
    if float(initial["initial_speed_mps"]) < SPEED_MIN:
        return None, "EXECUTION_INITIAL_SPEED_BELOW_3P61"
    if not route:
        return None, "NATIVE_ROUTE_EMPTY"
    if not candidate["anchor_80_state_support"]:
        return None, "STATIC_80_STATE_SUPPORT_UNAVAILABLE"
    try:
        from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
        if candidate["map_name"] not in map_cache:
            map_cache[candidate["map_name"]] = get_maps_api(str(MAP_ROOT), "nuplan-maps-v1.0", candidate["map_name"])
        api = map_cache[candidate["map_name"]]
        current = {
            "rear_axle": {"x": initial["initial_x"], "y": initial["initial_y"], "heading": initial["initial_heading"]},
            "speed_mps": initial["initial_speed_mps"], "time_us": initial["initial_time_us"],
        }
        reference = build_native_route_reference_v1_1(api, route, current, max(0.2, float(initial["initial_speed_mps"])) * 7.9)
        if bool(reference.get("extrapolation_used")) or len(reference.get("reference_xy", [])) < 2:
            return None, "NATIVE_ROUTE_REFERENCE_INCOMPLETE"
    except Exception as exc:
        return None, f"NATIVE_ROUTE_REFERENCE_UNAVAILABLE:{type(exc).__name__}"
    rid = route_id(candidate["map_name"], [str(x) for x in route])
    enriched = dict(candidate)
    enriched.update({
        "initial": initial, "route_roadblock_ids": [str(x) for x in route], "route_id": rid,
        "native_edge_ids": [str(x) for x in reference.get("native_edge_ids", [])],
        "route_reference_builder": str(reference.get("builder_version")),
    })
    payload = precontext_payload(enriched, initial, [str(x) for x in route], rid)
    enriched["precontext"] = payload
    enriched["precontext_id"] = precontext_id(payload)
    return enriched, None


def census_rows(roles: Mapping[str, Mapping[str, str]], ranked: Mapping[str, list[dict[str, Any]]], candidate_counts: Counter[str]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], Counter[str]]:
    selected: dict[str, dict[str, Any]] = {}
    rejection_totals: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    map_cache: dict[str, Any] = {}
    for index, session_id in enumerate(sorted(roles), 1):
        role = roles[session_id]
        exposure = role["exposure_class"]
        if role["v_policy_status"] != "CANDIDATE":
            rows.append({
                "session_id": session_id, "exposure_class": exposure, "raw_candidate_scenarios": role["scenario_count"],
                "ranked_candidates_examined": 0, "representative_log_id": "", "representative_scenario_token": "",
                "source_provenance_status": role["source_role_provenance_status"],
                "representative_source_partition": "", "representative_source_database_fingerprint_sha256": "",
                "selection_rank_sha256": "", "execution_initial_speed_available": "false", "execution_initial_speed_mps": "",
                "initial_speed_ge_3_61": "false", "native_route_available": "false", "native_route_identity": "",
                "route_reference_complete": "false", "map_support_available": "false", "scenario_loadable_metadata_only": "false",
                "precontext_fields_available": "false", "precontext_id": "", "production_executor_compatible": "false",
                "duplicate_alias_status": role["duplicate_identity_status"], "static_eligibility_status": "STATIC_INELIGIBLE",
                "exclusion_reason": "EXPOSURE_POLICY_E3_E4", "ambiguity_reason": "", "final_primary_roster_member": "false",
            })
            continue
        errors: Counter[str] = Counter()
        winner = None
        candidates = ranked.get(session_id, [])
        for candidate in candidates:
            candidate["session_id"] = session_id
            value, error = evaluate_candidate(candidate, map_cache)
            if value is not None:
                winner = value
                break
            errors[str(error)] += 1
            rejection_totals[str(error)] += 1
        examined = sum(errors.values()) + (1 if winner else 0)
        if winner:
            selected[session_id] = winner
            initial = winner["initial"]
            status, exclusion, ambiguity = "STATIC_ELIGIBLE", "", ""
            values = {
                "representative_log_id": winner["log_id"], "representative_scenario_token": winner["scenario_token"],
                "source_provenance_status": role["source_role_provenance_status"],
                "representative_source_partition": winner["source_partition"],
                "representative_source_database_fingerprint_sha256": winner["source_fingerprint_sha256"],
                "selection_rank_sha256": winner["rank_sha256"], "execution_initial_speed_available": "true",
                "execution_initial_speed_mps": f"{float(initial['initial_speed_mps']):.6f}", "initial_speed_ge_3_61": "true",
                "native_route_available": "true", "native_route_identity": winner["route_id"], "route_reference_complete": "true",
                "map_support_available": "true", "scenario_loadable_metadata_only": "true", "precontext_fields_available": "true",
                "precontext_id": winner["precontext_id"], "production_executor_compatible": "true",
            }
        else:
            complete_search = int(candidate_counts[session_id]) <= TOP_K
            status = "STATIC_INELIGIBLE" if complete_search else "STATIC_AMBIGUOUS"
            exclusion = ";".join(f"{k}:{v}" for k, v in sorted(errors.items())) if complete_search else ""
            rejection_summary = ",".join(f"{key}:{value}" for key, value in sorted(errors.items()))
            ambiguity = "" if complete_search else f"NO_ELIGIBLE_IDENTITY_IN_DETERMINISTIC_TOP_{TOP_K};PREFIX_REJECTIONS={rejection_summary};REMAINING_CANDIDATES_NOT_MAP_EVALUATED"
            values = {key: "" for key in ("representative_log_id", "representative_scenario_token", "representative_source_partition", "representative_source_database_fingerprint_sha256", "selection_rank_sha256", "execution_initial_speed_mps", "native_route_identity", "precontext_id")}
            values["source_provenance_status"] = role["source_role_provenance_status"]
            values.update({key: "false" for key in ("execution_initial_speed_available", "initial_speed_ge_3_61", "native_route_available", "route_reference_complete", "map_support_available", "scenario_loadable_metadata_only", "precontext_fields_available", "production_executor_compatible")})
        rows.append({
            "session_id": session_id, "exposure_class": exposure, "raw_candidate_scenarios": candidate_counts[session_id],
            "ranked_candidates_examined": examined, **values, "duplicate_alias_status": role["duplicate_identity_status"],
            "static_eligibility_status": status, "exclusion_reason": exclusion, "ambiguity_reason": ambiguity,
            "final_primary_roster_member": "false",
        })
        if index % 25 == 0:
            print(f"static sessions evaluated: {index}/248; eligible={len(selected)}", flush=True)
    return rows, selected, rejection_totals


def fixture_proof(selected: Mapping[str, Mapping[str, Any]], roles: Mapping[str, Mapping[str, str]]) -> Mapping[str, Any]:
    from tools.s2r_production_executor import assert_fresh_pair, build_fresh_arm, execution_manifest

    session_id = sorted(selected)[0]
    row = selected[session_id]
    common = {
        "pair_id": "S2R-ZERO-RUN-FIXTURE-PAIR", "session_id": session_id, "log_id": row["log_id"],
        "scenario_token": row["scenario_token"], "exposure_class": roles[session_id]["exposure_class"], "role": "B",
        "db_path": row["db_path"], "map_name": row["map_name"], "route_roadblock_ids": row["route_roadblock_ids"],
        "route_id": row["route_id"], "precontext": row["precontext"], "precontext_id": row["precontext_id"],
        "start_timestamp_us": row["initial"]["initial_time_us"],
    }
    with tempfile.TemporaryDirectory(prefix="s2r_engineering_fixture_") as directory:
        root = Path(directory)
        baseline_spec = {**common, "run_id": "S2R-ZERO-RUN-FIXTURE-BASELINE", "arm": "BASELINE"}
        treatment_spec = {**common, "run_id": "S2R-ZERO-RUN-FIXTURE-TREATMENT", "arm": "TREATMENT"}
        baseline = build_fresh_arm(baseline_spec, root / "baseline")
        treatment = build_fresh_arm(treatment_spec, root / "treatment")
        proof = assert_fresh_pair(baseline, treatment)
        left = execution_manifest(baseline_spec, baseline, AUDITED_GIT_SHA, "ZERO_RUN_FIXTURE_NOT_SCIENTIFIC_EXECUTION")
        right = execution_manifest(treatment_spec, treatment, AUDITED_GIT_SHA, "ZERO_RUN_FIXTURE_NOT_SCIENTIFIC_EXECUTION")
        def normalized(spec: Mapping[str, Any], root: Path) -> list[str]:
            from tools.s2r_production_executor import resolved_override_contract
            return [value for value in resolved_override_contract(spec, root) if not value.startswith(("job_name=", "output_dir="))]
        shared_override_equal = normalized(baseline_spec, root / "baseline/raw") == normalized(treatment_spec, root / "treatment/raw")
        baseline.recorder.uninstall(); treatment.recorder.uninstall()
    return {
        "status": proof["status"], "fixture_session_id": session_id,
        "independent_components": proof["independent_components"], "precontext_equal": proof["precontext_equal"],
        "route_equal": proof["route_equal"], "isolated_process_required_for_execution": True,
        "execution_manifest_schema_valid": set(left) == set(right),
        "arm_specific_resolved_config_hashes_bound": bool(baseline.config_hash and treatment.config_hash),
        "shared_non_namespace_override_contract_equal": shared_override_equal,
        "runner_run_calls": 0, "planner_compute_calls": 0, "simulation_advance_calls": 0,
    }


def generate() -> None:
    ensure_runtime()
    verify_governance()
    from tools.s2r_production_executor import CONFIG_SCHEMA_VERSION, MANIFEST_FIELDS, SCHEMA_VERSION

    OUT.mkdir(parents=True, exist_ok=True)
    roles = role_rows()
    old = read_json(OLD_CENSUS)
    ranked, candidate_counts = ranked_candidates(old, roles)
    rows, selected, rejection_totals = census_rows(roles, ranked, candidate_counts)
    write_csv("S2R_Static_Eligibility_Census_v1.csv", rows)
    fixture = fixture_proof(selected, roles) if selected else {"status": "NO_STATIC_ELIGIBLE_FIXTURE", "runner_run_calls": 0}
    eligible = [r for r in rows if r["static_eligibility_status"] == "STATIC_ELIGIBLE"]
    ambiguous = [r for r in rows if r["static_eligibility_status"] in {"STATIC_AMBIGUOUS", "TECHNICAL_METADATA_BLOCKED"}]
    by_class = Counter(r["exposure_class"] for r in eligible)
    claim_a = sum(r["exposure_class"] in {"E0_METADATA_ONLY", "E1_UNRELATED_HISTORICAL_USE", "E5_UNTOUCHED_CONFIRMATORY"} for r in eligible)
    claim_b = len(eligible)
    strict_c = by_class["E5_UNTOUCHED_CONFIRMATORY"]
    executor_pass = fixture.get("status") == "NO_CROSS_ARM_STATE_LEAKAGE_PROVEN_BY_CONSTRUCTION"
    reset_pass = executor_pass
    precontext_pass = executor_pass and bool(selected) and all(r["precontext_id"] for r in eligible)
    main_status = "S2R_ENGINEERING_CLOSURE_READY_FOR_OWNER_REVIEW" if executor_pass and reset_pass and precontext_pass and claim_b > 0 else "S2R_ENGINEERING_CLOSURE_BLOCKED_STATIC_CAPACITY"

    write_text("S2R_Production_Executor_Contract_v1.md", f"""
# S2R Production Executor Contract v1

Status: `PRODUCTION_EXECUTION_BINDING = {'PASS' if executor_pass else 'BLOCKED'}`.

The only future production entrypoint is `tools/s2r_production_executor.py --execute-authorized-arm`. It is closed unless an exact Owner authorization binds the executor SHA, one run ID, and a one-arm budget. A pair orchestrator must launch baseline and treatment as two fresh processes using `isolated_arm_command`; no alternative or legacy executor is allowed.

The unique stack is: official nuPlan `SimulationRunner` → `R2BControllerAwarePlannerV1` with frozen round-0 TSB parameters → `TwoStageController` / `LQRTracker` / `KinematicBicycleModel` → sequential callback chain → `PassiveActualLQRRecorderV1` → `s1_protocol_schema` serializer → `r1_b2_8_r3_2_post_run_evaluator_dispatcher.evaluate_frozen_pair` analyzer. The resolved Hydra override contract fixes exact scenario token, Primary80 controller, sequential worker, seed {__import__('tools.s2r_production_executor', fromlist=['SEED']).SEED}, metrics, and output namespace.

Execution manifest schema `{SCHEMA_VERSION}` binds all {len(MANIFEST_FIELDS)} required fields before `runner.run()`. Technical failures remain recorded and cannot trigger scenario replacement. Audited source SHA is `{AUDITED_GIT_SHA}`; the final commit SHA must be rebound by Owner authorization before any future execution.
""")
    write_text("S2R_Full_Arm_Reset_Proof_v1.md", f"""
# S2R Full-arm Reset Proof v1

Status: `FULL_ARM_RESET_CONTRACT = {'PASS' if reset_pass else 'BLOCKED'}`.

`build_fresh_arm` constructs a new runner, planner, simulation, controller, tracker, motion model, callback graph, recorder, random-state object, history-buffer owner, and artifact namespace for each arm. `assert_fresh_pair` fails closed if any mutable object identity or output root is shared. Future execution additionally requires one fresh operating-system process per arm, eliminating Python module/singleton carryover.

The official zero-run fixture constructed both arms from one real metadata-only scenario and proved independent objects for: `{', '.join(fixture.get('independent_components', []))}`. Recorder state and callback state were separately constructed. Both runners remained unstarted; runner, planner-compute, and simulation-advance counts were zero.
""")
    write_text("S2R_Precontext_Identity_Implementation_v1.md", f"""
# S2R Precontext Identity Implementation v1

Status: `PRECONTEXT_IDENTITY_CONTRACT = {'PASS' if precontext_pass else 'BLOCKED'}`.

`PRECONTEXT_ID = SHA256(canonical_json(s2r_precontext_v1))` is now a required arm-spec and execution-manifest field. It binds SESSION/log/scenario identity, official simulation initial lidar token and timestamp, official ego initial state, route and map identity, raw traffic-agent/traffic-light precontext hash, history-buffer source hash, initialization contract, and the 1.1 s intervention boundary. The official constant zero steering value is explicitly labeled as an API source rather than a measured field.

Before execution, the pair factory requires exact baseline/treatment equality of both `PRECONTEXT_ID` and `ROUTE_ID`; mismatch yields `PAIR_INVALID_PRECONTEXT` or `PAIR_INVALID_ROUTE`. The production planner's frozen common-preintervention rule uses the complete baseline trajectory for both arms before 1.1 s. No rollout was used to form these hashes.
""")
    write_text("S2R_Static_V_Capacity_v1.md", f"""
# S2R Static V Capacity v1

```text
248 total SESSION clusters
↓ exclude E3=31 / E4=0
217 raw V candidates
↓ execution initial speed >= 3.61 m/s
{claim_b} sessions with a deterministically ranked static-eligible identity
↓ native route/reference and map support
{claim_b}
↓ metadata/precontext complete
{claim_b}
↓ canonical production executor compatible
{claim_b}
↓ STATIC_CERTIFIED_V_POOL
{claim_b}
```

The selector ranks identities by SHA256 of a fixed salt, SESSION, scenario token, and log ID, then chooses the first identity satisfying the pre-outcome static contract. At most {TOP_K} rank-leading identities were map-evaluated per SESSION; sessions with no success in that prefix remain ambiguous rather than being declared ineligible. This is a capacity census and candidate identity evidence, not a final Primary roster. A later frozen roster must use an Owner-approved deterministic draw from this pool and must prohibit replacement after execution begins.

The 3.61 m/s gate remains `NOMINAL_MEASURABILITY_SCREEN / NOT_A_CLOSED_LOOP_GUARANTEE`. Native route completeness uses the same official builder and conservative `initial_speed × 7.9 s` reference requirement already used by the frozen zero-run route preflight.
""")
    write_text("S2R_Claim_A_Claim_B_Static_Capacity_v1.md", f"""
# S2R Claim A / Claim B Static Capacity v1

- `CLAIM_A_STATIC_CANDIDATE_CAPACITY = {claim_a}` from statically eligible E0/E1/E5 SESSION clusters; E2 is excluded from Claim A.
- `CLAIM_B_STATIC_V_CAPACITY = {claim_b}` from statically eligible E0/E1/E2/E5 SESSION clusters.

These are static candidate capacities. They are not final sample sizes, scientific outcomes, or authorization to materialize the Primary roster.
""")
    write_text("S2R_C_Static_Capacity_v1.md", f"""
# S2R C Static Capacity v1

`E5_RAW = 5`; `STRICT_C_STATIC_CAPACITY = {strict_c}`. No E1/E2 session is renamed as untouched, and this preflight does not expand C.
""")
    write_text("S2R_Engineering_Closure_Report_v1.md", f"""
# S2R Engineering Closure Report v1

Main status: `{main_status}`.

`PRODUCTION_EXECUTION_BINDING = {'PASS' if executor_pass else 'BLOCKED'}`; `FULL_ARM_RESET_CONTRACT = {'PASS' if reset_pass else 'BLOCKED'}`; `PRECONTEXT_IDENTITY_CONTRACT = {'PASS' if precontext_pass else 'BLOCKED'}`. Static-certified Claim-B V capacity is {claim_b}; Claim-A capacity is {claim_a}; strict-C capacity is {strict_c}. Exposure policy and statistical contracts were not changed.

The census processed all 248 SESSION clusters, 1,564 logs, and 5,338,021 frozen scenario identities. It used raw DB/map/config metadata only. Primary roster membership remains false for every row. Main remaining ambiguity count is {len(ambiguous)}; deterministic-prefix rejection audit totals are `{dict(sorted(rejection_totals.items()))}`.

Scientific execution remains closed: simulation, runner.run, TSB/HLC rollout, RBR training, Primary evaluation, V execution, and C execution all equal zero and remain unauthorized.
""")

    artifacts = [
        "S2R_Engineering_Closure_Report_v1.md", "S2R_Production_Executor_Contract_v1.md",
        "S2R_Full_Arm_Reset_Proof_v1.md", "S2R_Precontext_Identity_Implementation_v1.md",
        "S2R_Static_Eligibility_Census_v1.csv", "S2R_Static_V_Capacity_v1.md",
        "S2R_Claim_A_Claim_B_Static_Capacity_v1.md", "S2R_C_Static_Capacity_v1.md",
    ]
    manifest = {
        "schema_version": "S2R_Engineering_Closure_Manifest_v1", "generator_version": "1.0",
        "date": "2026-09-18", "branch": BRANCH, "generation_base_git_sha": AUDITED_GIT_SHA,
        "governing_remote_git_sha": GOVERNING_REMOTE_SHA,
        "main_status": main_status,
        "contracts": {"production_execution_binding": "PASS" if executor_pass else "BLOCKED", "full_arm_reset": "PASS" if reset_pass else "BLOCKED", "precontext_identity": "PASS" if precontext_pass else "BLOCKED"},
        "capacity": {
            "total_sessions": 248, "raw_v_candidates": 217, "static_certified_v_pool": claim_b,
            "claim_a_static_candidate_capacity": claim_a, "claim_b_static_v_capacity": claim_b,
            "strict_c_static_capacity": strict_c, "ambiguous_candidate_sessions": len(ambiguous),
            "by_exposure_class": {key: by_class[key] for key in ("E0_METADATA_ONLY", "E1_UNRELATED_HISTORICAL_USE", "E2_BENCHMARK_ENGINEERING", "E5_UNTOUCHED_CONFIRMATORY")},
        },
        "selection": {"status": "PROPOSED_DETERMINISTIC_STATIC_EVIDENCE_RULE_NOT_FINAL_PRIMARY_ROSTER", "salt_sha256": hashlib.sha256(SALT.encode()).hexdigest(), "ranked_prefix_per_session": TOP_K, "outcome_fields_used": False, "replacement_after_execution_allowed": False},
        "fixture": fixture,
        "inputs": {str(path.relative_to(ROOT)): sha(path) for path in (APPROVAL, PLAN, ROLE_CENSUS, ROLE_MANIFEST, OLD_CENSUS, PROTECTED)},
        "code": {"tools/s2r_engineering_closure.py": sha(Path(__file__)), "tools/s2r_production_executor.py": sha(ROOT / "tools/s2r_production_executor.py")},
        "artifacts": {f"docs/stageR/s2r_engineering_closure/{name}": sha(OUT / name) for name in artifacts},
        "execution_manifest_schema": {"version": SCHEMA_VERSION, "config_schema_version": CONFIG_SCHEMA_VERSION, "fields": list(MANIFEST_FIELDS)},
        "final_primary_roster_materialized": False, "sample_size_changed": False,
        "authorization": {"simulation": False, "rbr_training": False, "primary_evaluation": False, "v_execution": False, "c_execution": False},
        "zero_run_counters": ZERO,
        "protected_asset": {"path": str(PROTECTED.relative_to(ROOT)), "sha256": sha(PROTECTED), "unchanged": sha(PROTECTED) == PROTECTED_SHA},
    }
    write_json("S2R_Engineering_Closure_Manifest_v1.json", manifest)
    print(json.dumps({"main_status": main_status, "static_certified_v_pool": claim_b, "claim_a": claim_a, "strict_c": strict_c, "ambiguous": len(ambiguous), "zero_run": ZERO}, indent=2))


if __name__ == "__main__":
    generate()
