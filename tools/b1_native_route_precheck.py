#!/usr/bin/env python3
"""Zero-run production route prerequisites and separate latent topology risk for B1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = Path(__file__).resolve().parents[1]
MAP_ROOT = ROOT.parent / "nuplan/dataset/maps"
PRODUCTION_ROUTE_SOURCE = ROOT / "tools/r1_closed_loop_benchmark_v2_1.py"
PARAMETERS = ROOT / "docs/stageR/r2/r2_b_calibration_rounds/r2_b_tsb_round_0_parameters_v1.0.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initial_current(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    initial = spec["precontext"]["ego_initial_state"]
    return {
        "rear_axle": {
            "x": float(initial["initial_x"]),
            "y": float(initial["initial_y"]),
            "heading": float(initial["initial_heading"]),
        },
        "speed_mps": float(initial["initial_speed_mps"]),
        "time_us": int(initial["initial_time_us"]),
    }


def precheck_with_production_builder(spec: Mapping[str, Any], map_api: Any) -> Mapping[str, Any]:
    """Separate current production prerequisites from latent full-route risk."""
    from tools.r1_closed_loop_benchmark_v2_1 import build_native_route_reference_v1_1
    from tools.r2_b_controller_aware_planner_v1 import _tsb_required_distance

    parameters = json.loads(PARAMETERS.read_text(encoding="utf-8"))["parameters"]
    current = initial_current(spec)
    required = _tsb_required_distance(float(current["speed_mps"]), 0.0, "BASELINE", parameters)
    base = {
        "route_id": str(spec["route_id"]),
        "native_route_resolution_status": "NOT_RESOLVED",
        "missing_successor_segment": None,
        "failure_code": None,
        "production_equivalence_version": "build_native_route_reference_v1_1",
        "production_equivalence_hash": sha256_file(PRODUCTION_ROUTE_SOURCE),
        "initial_production_required_forward_m": required,
        "initial_production_probe_status": "NOT_RUN",
        "compatibility_scope": "INITIAL_PRODUCTION_QUERY;FUTURE_REALIZED_INPUTS_NOT_KNOWN",
        "precheck_contract_version": "B1_IR2_PRODUCTION_INPUT_AND_LATENT_RISK_V2",
        "latent_full_route_topology_status": "NOT_RUN",
    }
    try:
        build_native_route_reference_v1_1(
            map_api,
            [str(value) for value in spec["route_roadblock_ids"]],
            current,
            required,
        )
    except ValueError as error:
        message = str(error)
        match = re.search(r"no native outgoing successor into (.+)$", message)
        return {
            **base,
            "route_precheck_status": "INCOMPATIBLE" if message.startswith("NATIVE_ROUTE_FAIL:") else "TECHNICAL_ERROR",
            "native_route_resolution_status": "INITIAL_PRODUCTION_REQUIREMENT_FAILED",
            "initial_production_probe_status": "FAILED",
            "missing_successor_segment": None if match is None else match.group(1),
            "failure_code": message,
        }
    except Exception as error:
        return {**base, "route_precheck_status": "TECHNICAL_ERROR", "failure_code": f"{type(error).__name__}:{error}"}
    base = {**base, "initial_production_probe_status": "PASS"}
    try:
        build_native_route_reference_v1_1(
            map_api,
            [str(value) for value in spec["route_roadblock_ids"]],
            current,
            float("inf"),
        )
    except ValueError as error:
        message = str(error)
        if message == "NATIVE_ROUTE_FAIL: insufficient native forward coverage":
            return {
                **base,
                "route_precheck_status": "COMPATIBLE",
                "native_route_resolution_status": "INITIAL_REQUIREMENT_AND_FULL_TOPOLOGY_PASS",
                "latent_full_route_topology_status": "PASS",
            }
        match = re.search(r"no native outgoing successor into (.+)$", message)
        if match:
            return {
                **base,
                "route_precheck_status": "AMBIGUOUS",
                "native_route_resolution_status": "INITIAL_REQUIREMENT_PASS_WITH_LATENT_RUNTIME_RISK",
                "latent_full_route_topology_status": "MISSING_SUCCESSOR",
                "missing_successor_segment": match.group(1),
                "failure_code": message,
            }
        return {**base, "route_precheck_status": "TECHNICAL_ERROR", "failure_code": message}
    except Exception as error:
        return {**base, "route_precheck_status": "TECHNICAL_ERROR", "failure_code": f"{type(error).__name__}:{error}"}
    return {**base, "route_precheck_status": "TECHNICAL_ERROR", "failure_code": "EXHAUSTIVE_REQUEST_UNEXPECTEDLY_SATISFIED"}


def precheck_specs(specs_path: Path) -> Mapping[str, Any]:
    from tools.r1_b2_8_r3_prospective_selector import official_env

    official_env()
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    specs = json.loads(specs_path.read_text(encoding="utf-8"))["arms"]
    cache: dict[str, Any] = {}
    pair_results: dict[str, Mapping[str, Any]] = {}
    arm_results: list[Mapping[str, Any]] = []
    for spec in specs:
        pair_id = str(spec["pair_id"])
        if pair_id not in pair_results:
            map_name = str(spec["map_name"])
            if map_name not in cache:
                cache[map_name] = get_maps_api(str(MAP_ROOT), "nuplan-maps-v1.0", map_name)
            pair_results[pair_id] = {"pair_id": pair_id, **precheck_with_production_builder(spec, cache[map_name])}
        arm_results.append({"run_id": str(spec["run_id"]), "pair_id": pair_id, **{k: v for k, v in pair_results[pair_id].items() if k != "pair_id"}})
    return {
        "schema_version": "B1_IR2_native_route_precheck_v2",
        "simulation_count": 0,
        "runner_run_count": 0,
        "planner_step_count": 0,
        "pair_count": len(pair_results),
        "arm_count": len(arm_results),
        "pairs": list(pair_results.values()),
        "arms": arm_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"ROUTE_OUTPUT_ALREADY_EXISTS:{args.output}; use a new path")
    result = precheck_specs(args.specs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    counts = {status: sum(row["route_precheck_status"] == status for row in result["pairs"]) for status in ("COMPATIBLE", "INCOMPATIBLE", "AMBIGUOUS", "TECHNICAL_ERROR")}
    print(json.dumps({"pair_count": result["pair_count"], **counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
