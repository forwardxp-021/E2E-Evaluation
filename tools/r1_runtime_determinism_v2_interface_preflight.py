#!/usr/bin/env python3
"""Zero-budget nuPlan interface preflight for R1 runtime determinism V2.

This program never constructs a NuPlanScenario, reads an SQLite database, or
starts simulation.  It checks the repaired planner against nuPlan's
``AbstractPlanner`` interface and makes one in-memory PlannerInput call for
each frozen runtime family.  Its only filesystem output is a small diagnostic
JSON plus disposable mock traces outside the official V2 execution namespace.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration

from tools.r1_runtime_determinism_planner import R1RuntimeDeterminismPlanner


ROOT = Path(__file__).resolve().parents[1]
R1_DIR = ROOT / "docs/stageR/r1"
DEFAULT_AUTHORIZATION = R1_DIR / "r1_runtime_determinism_validation_v2_authorization_v1.0.json"
DEFAULT_ROSTER = R1_DIR / "r1_runtime_determinism_validation_roster_v1.0.json"
DEFAULT_OUTPUT = R1_DIR / "r1_runtime_determinism_v2_interface_preflight_v1.0.json"
DEFAULT_SCRATCH = ROOT / "outputs/r1_runtime_determinism_v2_interface_preflight_attempt3"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _signature_shape(method: Any) -> Dict[str, Any]:
    signature = inspect.signature(method)
    parameters: List[Dict[str, Any]] = []
    for parameter in signature.parameters.values():
        parameters.append(
            {
                "name": parameter.name,
                "kind": parameter.kind.name,
                "has_default": parameter.default is not inspect.Parameter.empty,
                "default_repr": None if parameter.default is inspect.Parameter.empty else repr(parameter.default),
            }
        )
    return {"text": str(signature), "parameters": parameters}


def _runtime_signature_compatible(reference: Any, candidate: Any) -> Dict[str, Any]:
    expected = _signature_shape(reference)
    actual = _signature_shape(candidate)
    expected_parameters = expected["parameters"]
    actual_parameters = actual["parameters"]
    names_and_kinds_equal = [
        (row["name"], row["kind"], row["has_default"], row["default_repr"])
        for row in expected_parameters
    ] == [
        (row["name"], row["kind"], row["has_default"], row["default_repr"])
        for row in actual_parameters
    ]
    return {
        "abstract_planner_signature": expected,
        "candidate_signature": actual,
        "parameter_contract_exact": names_and_kinds_equal,
    }


class _MockMap:
    """Only provides the native adjacency that the frozen HLC initialization checks."""

    def __init__(self, entry: Mapping[str, Any]) -> None:
        self._source = SimpleNamespace(adjacent_edges=[SimpleNamespace(id=str(entry["target_lane_id"]))])
        self._target = SimpleNamespace(adjacent_edges=[])
        self._source_id = str(entry["source_lane_id"])

    def get_map_object(self, object_id: str, _layer: Any) -> Any:
        return self._source if str(object_id) == self._source_id else self._target


def _mock_input(entry: Mapping[str, Any]) -> PlannerInput:
    state = entry["initial_state"]
    ego = EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(float(state["initial_x"]), float(state["initial_y"]), float(state["initial_heading"])),
        rear_axle_velocity_2d=StateVector2D(float(state["initial_speed_mps"]), 0.0),
        rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
        tire_steering_angle=0.0,
        time_point=TimePoint(int(state["initial_time_us"])),
        vehicle_parameters=get_pacifica_parameters(),
        angular_vel=0.0,
        angular_accel=0.0,
    )
    observation = DetectionsTracks(TrackedObjects([]))
    history = SimulationHistoryBuffer.initialize_from_list(
        buffer_size=1, ego_states=[ego], observations=[observation], sample_interval=0.1
    )
    return PlannerInput(
        iteration=SimulationIteration(TimePoint(int(state["initial_time_us"])), 0),
        history=history,
        traffic_light_data=[],
    )


def _run_family_mock(entry: Mapping[str, Any], roster_path: Path, scratch_dir: Path) -> Dict[str, Any]:
    family = str(entry["family"])
    trace_dir = scratch_dir / f"{family}__{entry['scenario_token']}"
    scenario = SimpleNamespace(token=str(entry["scenario_token"]))
    planner = R1RuntimeDeterminismPlanner(
        scenario=scenario,
        roster_path=str(roster_path),
        runtime_family=family,
        trace_dir=str(trace_dir),
    )
    initialization = PlannerInitialization(
        route_roadblock_ids=[str(value) for value in entry.get("route_roadblock_ids", [])],
        mission_goal=StateSE2(float(entry["initial_state"]["initial_x"]), float(entry["initial_state"]["initial_y"]), 0.0),
        map_api=_MockMap(entry) if family == "R-HLC" else SimpleNamespace(),
    )
    planner.initialize(initialization)
    trajectory = planner.compute_trajectory(_mock_input(entry))
    report = planner.generate_planner_report(clear_stats=False)
    trace_path = trace_dir / "planner_trace.jsonl"
    binding_path = trace_dir / "planner_binding.json"
    return {
        "family": family,
        "scenario_token": str(entry["scenario_token"]),
        "planner_name": planner.name(),
        "observation_type": planner.observation_type().__name__,
        "trajectory_type": type(trajectory).__name__,
        "trajectory_sample_count": len(trajectory.get_sampled_trajectory()),
        "planner_report_runtime_samples": len(report.compute_trajectory_runtimes),
        "trace_present": trace_path.is_file(),
        "binding_present": binding_path.is_file(),
        "trace_sha256": sha256_file(trace_path) if trace_path.is_file() else None,
        "binding_sha256": sha256_file(binding_path) if binding_path.is_file() else None,
    }


def _assert_authorization(authority: Mapping[str, Any], roster_path: Path) -> None:
    if authority.get("status") != "AUTHORIZED_ONCE":
        raise ValueError("V2 authorization is absent or not authorized once")
    expected_roster = authority["binding"]["original_frozen_runtime_roster_sha256"]
    actual_roster = sha256_file(roster_path)
    if expected_roster != actual_roster:
        raise ValueError("V2 authorization roster hash does not match the frozen roster")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, default=DEFAULT_AUTHORIZATION)
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scratch-dir", type=Path, default=DEFAULT_SCRATCH)
    args = parser.parse_args()

    authority = read_json(args.authorization)
    _assert_authorization(authority, args.roster)
    roster = read_json(args.roster)
    entries = list(roster.get("entries", []))
    families = {str(entry.get("family")) for entry in entries}
    if len(entries) != 4 or families != {"R-HLC", "R-TSB"}:
        raise ValueError("frozen roster must contain exactly two R-HLC and two R-TSB rows")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite interface-preflight result: {args.output}")
    if args.scratch_dir.exists():
        raise FileExistsError(f"refusing to overwrite interface-preflight scratch: {args.scratch_dir}")

    methods = ("name", "observation_type", "initialize", "compute_planner_trajectory", "compute_trajectory", "generate_planner_report")
    signatures = {
        name: _runtime_signature_compatible(getattr(AbstractPlanner, name), getattr(R1RuntimeDeterminismPlanner, name))
        for name in methods
    }
    if not all(row["parameter_contract_exact"] for row in signatures.values()):
        raise RuntimeError("repaired planner has a non-compatible AbstractPlanner callable signature")
    if not issubclass(R1RuntimeDeterminismPlanner, AbstractPlanner):
        raise RuntimeError("repaired planner is not an AbstractPlanner subclass")

    mocks = [_run_family_mock(next(entry for entry in entries if entry["family"] == family), args.roster, args.scratch_dir) for family in ("R-HLC", "R-TSB")]
    if not all(
        row["trace_present"] and row["binding_present"] and row["trajectory_sample_count"] > 0 and row["planner_report_runtime_samples"] == 1
        for row in mocks
    ):
        raise RuntimeError("in-memory planner smoke did not produce the expected interface artifacts")

    current_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    payload = {
        "schema_version": "r1_runtime_determinism_v2_interface_preflight_v1.0",
        "status": "PASS_NO_OFFICIAL_RUN_BUDGET_CONSUMED",
        "scope": "IN_MEMORY_MOCK_ONLY_NO_SCENARIO_DB_NO_SIMULATION",
        "current_git_commit_sha": current_commit,
        "authorization_sha256": sha256_file(args.authorization),
        "roster_sha256": sha256_file(args.roster),
        "planner_sha256": sha256_file(ROOT / "tools/r1_runtime_determinism_planner.py"),
        "abstract_planner_subclass": True,
        "interface_signature_comparisons": signatures,
        "family_mock_results": mocks,
        "official_closed_loop_runs_claimed": 0,
        "official_closed_loop_runs_started": 0,
        "budget_consumed": 0,
        "prohibited_actions_not_performed": ["NuPlanScenario construction", "SQLite database read", "simulation start", "official V2 output write"],
    }
    write_json(args.output, payload)
    print(json.dumps({"status": payload["status"], "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
