#!/usr/bin/env python3
"""Official-like PlannerInput dispatch preflight; never starts nuPlan simulation."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence

from tools.r1_official_technical_smoke_planner_v2_1 import R1OfficialTechnicalSmokePlannerV2_1


def _official_input(initial: Mapping[str, Any], iteration_index: int) -> Any:
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
    from nuplan.common.actor_state.tracked_objects import TrackedObjects
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
    from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
    from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration

    time_us = int(initial["initial_time_us"]) + int(iteration_index) * 100_000
    ego = EgoState.build_from_rear_axle(rear_axle_pose=StateSE2(float(initial["initial_x"]), float(initial["initial_y"]), float(initial["initial_heading"])), rear_axle_velocity_2d=StateVector2D(float(initial["initial_speed_mps"]), 0.0), rear_axle_acceleration_2d=StateVector2D(0.0, 0.0), tire_steering_angle=0.0, time_point=TimePoint(time_us), vehicle_parameters=get_pacifica_parameters())
    observation = DetectionsTracks(TrackedObjects([]))
    history = SimulationHistoryBuffer.initialize_from_list(buffer_size=1, ego_states=[ego], observations=[observation], sample_interval=0.1)
    return PlannerInput(iteration=SimulationIteration(time_point=TimePoint(time_us), index=int(iteration_index)), history=history, traffic_light_data=[])


def run_official_dispatch_preflight_v1(*, future_roster_row: Mapping[str, Any], smoke_arm: str, map_api: Any, iteration_indices: Sequence[int] = tuple(range(26))) -> Dict[str, Any]:
    """Call the real public dispatch with bound nuPlan 1.2.2 dataclasses."""
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization
    from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

    indices = [int(value) for value in iteration_indices]
    if not indices or indices[0] != 0 or any(right <= left for left, right in zip(indices, indices[1:])):
        raise ValueError("DISPATCH_PREFLIGHT_ITERATIONS_MUST_START_ZERO_AND_INCREASE")
    planner = R1OfficialTechnicalSmokePlannerV2_1(future_roster_row, str(future_roster_row["family"]), smoke_arm)
    planner.initialize(PlannerInitialization(route_roadblock_ids=[str(value) for value in future_roster_row["route_roadblock_ids"]], mission_goal=None, map_api=map_api))
    initial = future_roster_row["initial_state"]
    calls = []
    for index in indices:
        current_input = _official_input(initial, index)
        trajectory = planner.compute_trajectory(current_input)
        if not isinstance(trajectory, InterpolatedTrajectory):
            raise TypeError("ABSTRACT_PLANNER_DISPATCH_DID_NOT_RETURN_INTERPOLATED_TRAJECTORY")
        current_ego, _ = current_input.history.current_state
        sampled = trajectory.get_sampled_trajectory()
        state0, state1 = sampled[0], sampled[1]
        identity = bool(state0.time_us == current_ego.time_us and state0.rear_axle == current_ego.rear_axle and state0.dynamic_car_state.speed == current_ego.dynamic_car_state.speed)
        if not identity or not planner.last_construction_audit or not planner.last_construction_audit["pass"]:
            raise ValueError("OFFICIAL_DISPATCH_CURRENT_EGO_OR_CONSTRUCTION_PARITY_FAIL")
        dx, dy, heading = float(state1.rear_axle.x - state0.rear_axle.x), float(state1.rear_axle.y - state0.rear_axle.y), float(state0.rear_axle.heading)
        calls.append({"iteration_index": index, "returned_type": type(trajectory).__name__, "current_ego_state0_identity": identity, "construction_parity": True, "state1_speed_delta_mps": float(state1.dynamic_car_state.speed - state0.dynamic_car_state.speed), "state1_lateral_displacement_m": -math.sin(heading) * dx + math.cos(heading) * dy, "phase": dict(planner.phase_history[-1])})
    nominal = [float(call["phase"]["absolute_episode_time_s"]) for call in calls]
    if any(right <= left for left, right in zip(nominal, nominal[1:])):
        raise ValueError("OFFICIAL_DISPATCH_PHASE_CLOCK_DID_NOT_ADVANCE")
    return {"status": "OFFICIAL_DISPATCH_PREFLIGHT_PASS_ZERO_ROLLOUT", "planner": planner.name(), "nuplan_interface": "1.2.2_BOUND_ABSTRACT_PLANNER_PLANNERINPUT_INTERPOLATEDTRAJECTORY", "compute_trajectory_dispatch_called": len(calls), "compute_planner_trajectory_delegate_verified": True, "calls": calls, "phase_clock_monotonic": True, "route_builder": "build_native_route_reference_v1_1", "historical_b2_1_planner_imported": False, "simulation_launched": False, "candidate_enumeration_count": 0, "new_roster_created": False, "new_rollout_count": 0}


def launch_simulation_forbidden_b2_6(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("B2_6_RUN_SIMULATION_HARD_BLOCK")


__all__ = ["launch_simulation_forbidden_b2_6", "run_official_dispatch_preflight_v1"]
