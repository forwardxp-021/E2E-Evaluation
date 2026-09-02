#!/usr/bin/env python3
"""Freeze B2.9-D pair evaluation bindings before any scientific rollout."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

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
ROSTER = R1 / "r1_official_compliant_technical_smoke_roster_v3.0.json"
SCHEDULE = R1 / "r1_official_compliant_technical_smoke_schedule_v3.0.json"
OUT = R1 / "r1_b2_9_d_frozen_pair_evaluation_bindings_v2.0.json"
ROSTER_SHA = "efe8e9d680ca0bcacb367bc9b616610ca78c260195e53b8f025a7bd1d92c23e6"
SCHEDULE_SHA = "47b5512bc235eb533d44bf3c8106c97ea5467533fe62d0902a23316e5827b0cf"
PLANNER_REFERENCE_SEMANTICS = "ROUTE_CONTINUOUS_V2_3"
MEASUREMENT_REFERENCE_SEMANTICS = "FROZEN_NATIVE_SOURCE_TARGET_MEASUREMENT_CONTRACT"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def _context(
    entry: Mapping[str, Any], replay: Mapping[str, Any], bridge: R1OfficialMapQueryBridgeV2_1
) -> Dict[str, Any]:
    canonical = build_closed_loop_context_v2_1(
        family=str(entry["family"]),
        scenario_token=str(entry["scenario_token"]),
        map_version=str(entry["map_name"]),
        route_fingerprint=str(entry["route_fingerprint"]),
        initial_state_fingerprint=str(entry["initial_state_fingerprint"]),
        log_id=str(entry["log_id"]),
        route_roadblock_ids=entry["route_roadblock_ids"],
        frames=replay["frames"][:11],
        map_query=bridge,
        intended_lane_change_direction=entry.get("direction"),
    )
    return {
        "pre_context_raw_hash": canonical_json_sha256(replay["frames"][:10]),
        "canonical_context_json_hash": canonical["canonical_context_json_hash"],
        "canonical_context": canonical,
        "frozen_temporal_semantics": {"PRE_CONTEXT_iterations": list(range(10)), "ANCHOR_iteration": 10},
        "stage5d_slot_semantics": "AUTHORITATIVE_STAGE5D_EXACT_PARITY_LANE_AWARE_ONLY",
    }


def _one(entry: Mapping[str, Any], cache: Dict[str, Any]) -> Dict[str, Any]:
    candidate = {**entry, "timestamp": int(entry["scenario_anchor_timestamp_us"])}
    initial, route = frozen._official_initial(candidate)
    if [str(value) for value in route] != [str(value) for value in entry["route_roadblock_ids"]]:
        raise ValueError("FROZEN_ROUTE_BINDING_MISMATCH")
    api = frozen.map_api(ROOT.parent / "nuplan/dataset/maps", str(entry["map_name"]), cache)
    bridge = R1OfficialMapQueryBridgeV2_1(api)
    current = {
        "rear_axle": {
            "x": float(initial["initial_x"]),
            "y": float(initial["initial_y"]),
            "heading": float(initial["initial_heading"]),
        },
        "speed_mps": float(initial["initial_speed_mps"]),
        "time_us": int(initial["initial_time_us"]),
    }
    replay = frozen._sampled_replay(entry, initial)
    context = _context(entry, replay, bridge)
    base = {
        "pair_id": None,
        "family": entry["family"],
        "scenario_token": entry["scenario_token"],
        "log_id": entry["log_id"],
        "baseline_context": context,
        "treatment_context": context,
        "context_source": "frozen_v3_roster_plus_read_only_official_replay_Context_V2_1",
        "map_route_binding_sha256": canonical_json_sha256(
            {"map_name": entry["map_name"], "route": entry["route_roadblock_ids"], "initial": entry["initial_state_fingerprint"]}
        ),
        "PLANNER_REFERENCE_SEMANTICS": PLANNER_REFERENCE_SEMANTICS,
        "MEASUREMENT_REFERENCE_SEMANTICS": MEASUREMENT_REFERENCE_SEMANTICS,
        "measurement_numerics_changed": False,
        "future_realized_trace_used": False,
        "future_safety_result_used": False,
        "future_scientific_gate_result_used": False,
    }
    if entry["family"] == "R-TSB":
        return {**base, "pretreatment_clearance": None}
    source = bridge.native_reference_xy(str(entry["source_lane_id"]))
    target = bridge.native_reference_xy(str(entry["target_lane_id"]))
    route_ref = build_native_route_reference_v1_1(
        api, entry["route_roadblock_ids"], current, max(0.2, current["speed_mps"]) * 7.9
    )
    source_projection = bridge.project(
        str(entry["source_lane_id"]), (current["rear_axle"]["x"], current["rear_axle"]["y"])
    )
    target_projection = bridge.project(
        str(entry["target_lane_id"]), (current["rear_axle"]["x"], current["rear_axle"]["y"])
    )
    baseline = build_hlc_native_geometry_v1_1(
        current, 0.0, source, target,
        float(source_projection["arc_m"]), float(target_projection["arc_m"]), HLC_BASELINE,
    )
    treatment = build_hlc_native_geometry_v1_1(
        current, 0.0, source, target,
        float(source_projection["arc_m"]), float(target_projection["arc_m"]), HLC_TREATMENT,
    )
    clearance = evaluate_r1_hlc_dynamic_clearance_v1_1(
        baseline_xy=[[state["rear_axle"]["x"], state["rear_axle"]["y"]] for state in baseline],
        treatment_xy=[[state["rear_axle"]["x"], state["rear_axle"]["y"]] for state in treatment],
        official_runtime_vehicle_parameters=official_ego_vehicle_binding_v1(),
        original_replay_tracks=replay["tracks"],
        official_replay_observation_timestamps_s=replay["timestamps_s"],
    )
    if clearance.get("pretreatment_only") is not True or clearance.get("pass") is not True:
        raise ValueError("FROZEN_HLC_PRETREATMENT_CLEARANCE_UNAVAILABLE")
    return {
        **base,
        "pretreatment_clearance": clearance,
        "source_reference_xy": source.tolist(),
        "target_reference_xy": target.tolist(),
        "native_route_reference_xy": route_ref["reference_xy"].tolist(),
        "native_route_reference_source": "OFFICIAL_NUPLAN_NATIVE_ROUTE_REFERENCE_V1_1",
        "planner_route_continuous_reference_is_not_measurement_reference": True,
    }


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"VERSIONED_OUTPUT_EXISTS:{OUT}")
    if sha256(ROSTER) != ROSTER_SHA or sha256(SCHEDULE) != SCHEDULE_SHA:
        raise ValueError("IMMUTABLE_V3_ROSTER_OR_SCHEDULE_SHA_MISMATCH")
    official_env()
    roster, schedule = read_json(ROSTER), read_json(SCHEDULE)
    entries = {(row["scenario_token"], row["log_id"]): row for row in roster["entries"]}
    cache: Dict[str, Any] = {}
    pairs = []
    for pair_id in sorted({row["pair_id"] for row in schedule["runs"]}):
        runs = sorted(
            [row for row in schedule["runs"] if row["pair_id"] == pair_id],
            key=lambda row: int(row["run_order"]),
        )
        if len(runs) != 2 or not ("BASELINE" in runs[0]["run_id"] and "TREATMENT" in runs[1]["run_id"]):
            raise ValueError(f"SCHEDULE_PAIR_ARM_MISMATCH:{pair_id}")
        entry = entries[(runs[0]["scenario_token"], runs[0]["log_id"])]
        binding = _one(entry, cache)
        binding.update(
            {
                "pair_id": pair_id,
                "baseline_run_id": runs[0]["run_id"],
                "treatment_run_id": runs[1]["run_id"],
                "schedule_rows": runs,
            }
        )
        pairs.append(binding)
    if len(pairs) != 24 or len({row["pair_id"] for row in pairs}) != 24:
        raise ValueError("PAIR_BINDING_24_CLOSURE_FAILED")
    if sum(row["family"] == "R-HLC" for row in pairs) != 12 or sum(row["family"] == "R-TSB" for row in pairs) != 12:
        raise ValueError("PAIR_BINDING_FAMILY_CLOSURE_FAILED")
    if any(row.get("pretreatment_clearance") is not None for row in pairs if row["family"] == "R-TSB"):
        raise ValueError("TSB_HLC_ONLY_REFERENCE_LEAK")
    payload = {
        "schema_version": "r1_b2_9_d_frozen_pair_evaluation_bindings_v2.0",
        "status": "FROZEN_24_OF_24_PRE_OUTCOME_PAIR_BINDINGS_COMPLETE",
        "roster_sha256": ROSTER_SHA,
        "schedule_sha256": SCHEDULE_SHA,
        "implementation_sha256": {
            "pair_binding_freezer": sha256(Path(__file__)),
            "context_v2_1": sha256(ROOT / "tools/r1_closed_loop_context_adapter_v2_1.py"),
            "map_bridge": sha256(ROOT / "tools/r1_official_map_query_bridge_v2_1.py"),
            "clearance": sha256(ROOT / "tools/r1_hlc_dynamic_clearance_v1_1.py"),
        },
        "pairs": pairs,
        "counts": {"total": 24, "HLC_PAIR_BINDING_COMPLETE": 12, "TSB_PAIR_BINDING_COMPLETE": 12},
        "pre_outcome_complete": True,
        "no_rollout": True,
        "no_reselection": True,
        "measurement_contract_versioning_required": False,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"status": payload["status"], "pair_binding_sha256": sha256(OUT), "counts": payload["counts"]},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
