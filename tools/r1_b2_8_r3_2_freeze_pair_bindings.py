#!/usr/bin/env python3
"""Materialize immutable R3.2 pair-evaluation bindings without simulation."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import r1_b2_7_freeze_official_smoke_roster_v2 as frozen  # noqa: E402
from tools.r1_b2_8_r3_prospective_selector import official_env  # noqa: E402
from tools.r1_closed_loop_benchmark_v2_1 import (  # noqa: E402
    build_hlc_native_geometry_v1_1,
    build_native_route_reference_v1_1,
)
from tools.r1_closed_loop_context_adapter_v2_1 import build_closed_loop_context_v2_1  # noqa: E402
from tools.r1_context_mechanism_core import canonical_json_sha256  # noqa: E402
from tools.r1_hlc_dynamic_clearance_v1_1 import evaluate_r1_hlc_dynamic_clearance_v1_1  # noqa: E402
from tools.r1_official_ego_vehicle_binding_v1 import official_ego_vehicle_binding_v1  # noqa: E402
from tools.r1_official_map_query_bridge_v2_1 import R1OfficialMapQueryBridgeV2_1  # noqa: E402
from tools.r1_prospective_generator_contract_v2 import HLC_BASELINE, HLC_TREATMENT  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "docs/stageR/r1"
ROSTER = R1 / "r1_official_compliant_technical_smoke_roster_v2.1.json"
SCHEDULE = R1 / "r1_official_compliant_technical_smoke_schedule_v2.1.json"
OUT = R1 / "r1_b2_8_r3_2_frozen_pair_evaluation_bindings_v1.0.json"
ROSTER_SHA = "b977b802a7b25f0be37d04f3277cba2b2e98e521a2e30938ec40af9f278c1973"
SCHEDULE_SHA = "6733dc623cce2e2b64b9eb71cd407982b54dcaf5ecd48b644058c767c89d552f"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _context(entry: dict[str, Any], replay: dict[str, Any], bridge: R1OfficialMapQueryBridgeV2_1) -> dict[str, Any]:
    canonical = build_closed_loop_context_v2_1(
        family=str(entry["family"]), scenario_token=str(entry["scenario_token"]), map_version=str(entry["map_name"]),
        route_fingerprint=str(entry["route_fingerprint"]), initial_state_fingerprint=str(entry["initial_state_fingerprint"]),
        log_id=str(entry["log_id"]), route_roadblock_ids=entry["route_roadblock_ids"], frames=replay["frames"][:11],
        map_query=bridge, intended_lane_change_direction=entry.get("direction"),
    )
    return {
        "pre_context_raw_hash": canonical_json_sha256(replay["frames"][:10]),
        "canonical_context_json_hash": canonical["canonical_context_json_hash"],
        "canonical_context": canonical,
        "frozen_temporal_semantics": {"pre_iterations": list(range(10)), "anchor_iteration": 10},
        "stage5d_slot_semantics": "AUTHORITATIVE_STAGE5D_EXACT_PARITY_LANE_AWARE_ONLY",
    }


def _one(entry: dict[str, Any], cache: dict[str, Any]) -> dict[str, Any]:
    candidate = {**entry, "timestamp": int(entry["scenario_anchor_timestamp_us"])}
    initial, route = frozen._official_initial(candidate)
    if [str(x) for x in route] != [str(x) for x in entry["route_roadblock_ids"]]:
        raise ValueError("FROZEN_ROUTE_BINDING_MISMATCH")
    api = frozen.map_api(ROOT.parent / "nuplan/dataset/maps", str(entry["map_name"]), cache)
    bridge = R1OfficialMapQueryBridgeV2_1(api)
    current = {"rear_axle": {"x": float(initial["initial_x"]), "y": float(initial["initial_y"]), "heading": float(initial["initial_heading"])}, "speed_mps": float(initial["initial_speed_mps"]), "time_us": int(initial["initial_time_us"])}
    replay = frozen._sampled_replay(entry, initial)
    context = _context(entry, replay, bridge)
    base = {
        "pair_id": None, "family": entry["family"], "scenario_token": entry["scenario_token"], "log_id": entry["log_id"],
        "baseline_context": context, "treatment_context": context,
        "context_source": "frozen_R3_roster_plus_read_only_official_replay_Context_V2_1",
        "map_route_binding_sha256": canonical_json_sha256({"map_name": entry["map_name"], "route": entry["route_roadblock_ids"], "initial": entry["initial_state_fingerprint"]}),
    }
    if entry["family"] == "R-TSB":
        return {**base, "pretreatment_clearance": None}
    source = bridge.native_reference_xy(str(entry["source_lane_id"])); target = bridge.native_reference_xy(str(entry["target_lane_id"]))
    route_ref = build_native_route_reference_v1_1(api, entry["route_roadblock_ids"], current, max(0.2, current["speed_mps"]) * 7.9)
    source_project = bridge.project(str(entry["source_lane_id"]), (current["rear_axle"]["x"], current["rear_axle"]["y"]))
    target_project = bridge.project(str(entry["target_lane_id"]), (current["rear_axle"]["x"], current["rear_axle"]["y"]))
    baseline = build_hlc_native_geometry_v1_1(current, 0.0, source, target, float(source_project["arc_m"]), float(target_project["arc_m"]), HLC_BASELINE)
    treatment = build_hlc_native_geometry_v1_1(current, 0.0, source, target, float(source_project["arc_m"]), float(target_project["arc_m"]), HLC_TREATMENT)
    clearance = evaluate_r1_hlc_dynamic_clearance_v1_1(
        baseline_xy=[[x["rear_axle"]["x"], x["rear_axle"]["y"]] for x in baseline],
        treatment_xy=[[x["rear_axle"]["x"], x["rear_axle"]["y"]] for x in treatment],
        official_runtime_vehicle_parameters=official_ego_vehicle_binding_v1(), original_replay_tracks=replay["tracks"],
        official_replay_observation_timestamps_s=replay["timestamps_s"],
    )
    if clearance.get("pretreatment_only") is not True or not clearance.get("pass"):
        raise ValueError("FROZEN_HLC_PRETREATMENT_CLEARANCE_UNAVAILABLE")
    return {**base, "pretreatment_clearance": clearance, "source_reference_xy": source.tolist(), "target_reference_xy": target.tolist(), "native_route_reference_xy": route_ref["reference_xy"].tolist(), "native_route_reference_source": "OFFICIAL_NUPLAN_NATIVE_ROUTE_REFERENCE_V1_1"}


def main() -> int:
    if OUT.exists(): raise FileExistsError(f"VERSIONED_OUTPUT_EXISTS:{OUT}")
    if sha(ROSTER) != ROSTER_SHA or sha(SCHEDULE) != SCHEDULE_SHA: raise ValueError("IMMUTABLE_ROSTER_SCHEDULE_SHA_MISMATCH")
    official_env()
    roster, schedule = json.loads(ROSTER.read_text()), json.loads(SCHEDULE.read_text())
    entries = {(x["scenario_token"], x["log_id"]): x for x in roster["entries"]}; cache: dict[str, Any] = {}
    pairs: list[dict[str, Any]] = []
    for pair_id in sorted({x["pair_id"] for x in schedule["runs"]}):
        runs = [x for x in schedule["runs"] if x["pair_id"] == pair_id]
        if len(runs) != 2 or {"BASELINE", "TREATMENT"} != {"BASELINE" if "BASELINE" in x["run_id"] else "TREATMENT" for x in runs}: raise ValueError(f"SCHEDULE_PAIR_ARM_MISMATCH:{pair_id}")
        run = runs[0]; binding = _one(entries[(run["scenario_token"], run["log_id"])], cache)
        binding.update({"pair_id": pair_id, "baseline_run_id": next(x["run_id"] for x in runs if "BASELINE" in x["run_id"]), "treatment_run_id": next(x["run_id"] for x in runs if "TREATMENT" in x["run_id"]), "schedule_rows": runs})
        pairs.append(binding)
    if len(pairs) != 24 or len({x["pair_id"] for x in pairs}) != 24: raise ValueError("PAIR_ID_CLOSURE_FAILED")
    if sum(x["family"] == "R-HLC" for x in pairs) != 12 or sum(x["family"] == "R-TSB" for x in pairs) != 12: raise ValueError("FAMILY_PAIR_CLOSURE_FAILED")
    payload = {"schema_version": "r1_b2_8_r3_2_frozen_pair_evaluation_bindings_v1.0", "status": "FROZEN_24_PAIR_EVALUATION_BINDINGS_COMPLETE", "roster_sha256": ROSTER_SHA, "schedule_sha256": SCHEDULE_SHA, "implementation_sha256": {"context_v2_1": sha(ROOT / "tools/r1_closed_loop_context_adapter_v2_1.py"), "map_bridge": sha(ROOT / "tools/r1_official_map_query_bridge_v2_1.py"), "clearance": sha(ROOT / "tools/r1_hlc_dynamic_clearance_v1_1.py")}, "pairs": pairs, "counts": {"total": 24, "HLC_PAIR_BINDING_COMPLETE": 12, "TSB_PAIR_BINDING_COMPLETE": 12}, "no_rollout": True, "no_reselection": True}
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "pair_binding_sha256": sha(OUT), "counts": payload["counts"]}, ensure_ascii=False))

if __name__ == "__main__": main()
