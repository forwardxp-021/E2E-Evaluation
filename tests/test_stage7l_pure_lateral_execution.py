import json
import math
from pathlib import Path

import numpy as np

from tools.stage7l_pure_lateral_execution_planner import (
    CanonicalLongitudinalProgressGenerator,
    DOSE_TRANSITION_LENGTH_M,
    FrozenLaneChangeManeuver,
    build_lateral_positions,
    derive_trajectory_states,
    dynamic_consistency_audit,
    quintic_blend,
    quintic_blend_d1,
    quintic_blend_d2,
)


def maneuver() -> FrozenLaneChangeManeuver:
    source = tuple((float(x), 0.0) for x in np.linspace(0.0, 240.0, 241))
    target = tuple((float(x), 3.6) for x in np.linspace(0.0, 240.0, 241))
    return FrozenLaneChangeManeuver(
        scenario_token="0123456789abcdef", log_name="log", db_file="log.db",
        initial_state_fingerprint="fp", initial_x=0.0, initial_y=0.0,
        initial_heading=0.0, initial_speed_mps=5.0,
        source_lane_id="s", target_lane_id="t", source_roadblock_id="r", target_roadblock_id="r",
        direction="left", route_roadblock_ids=("r",), route_fingerprint="rfp",
        trigger_s_route_m=12.0, source_start_arc_m=0.0, target_start_arc_m=0.0,
        nominal_lane_width_m=3.6, horizon_s=15.0,
        background_mode="closed_loop_nonreactive_agents", background_agent_model="TracksObservation",
        background_config_sha256="bg", source_reference_xy=source, target_reference_xy=target,
        planner_profile_ids=tuple(DOSE_TRANSITION_LENGTH_M),
    )


def test_quintic_boundaries() -> None:
    assert float(quintic_blend(0.0)) == 0.0
    assert float(quintic_blend(1.0)) == 1.0
    assert float(quintic_blend_d1(0.0)) == 0.0
    assert float(quintic_blend_d1(1.0)) == 0.0
    assert float(quintic_blend_d2(0.0)) == 0.0
    assert float(quintic_blend_d2(1.0)) == 0.0


def test_dose_order_and_only_lateral_channel_changes() -> None:
    lengths = list(DOSE_TRANSITION_LENGTH_M.values())
    assert all(a > b for a, b in zip(lengths, lengths[1:]))
    generator = CanonicalLongitudinalProgressGenerator(5.0, 8.0, 1.0)
    time = np.linspace(0.0, 8.0, 81)
    expected_progress, _, _ = generator.sample(time)
    positions = []
    for transition_length in lengths:
        progress, _, _ = generator.sample(time)
        np.testing.assert_array_equal(progress, expected_progress)
        xy, _ = build_lateral_positions(maneuver(), progress, transition_length)
        positions.append(xy)
    assert not np.array_equal(positions[0], positions[-1])


def test_manifest_identity_across_doses() -> None:
    fixed = maneuver().dose_invariant_payload()
    serializations = []
    for dose, length in DOSE_TRANSITION_LENGTH_M.items():
        row = {"maneuver": fixed, "dose": {"dose_id": dose, "transition_length_m": length}}
        serializations.append(json.dumps(row["maneuver"], sort_keys=True))
    assert len(set(serializations)) == 1


def test_dynamic_consistency_is_finite_and_smooth() -> None:
    time = np.linspace(0.0, 8.0, 81)
    progress, _, _ = CanonicalLongitudinalProgressGenerator(5.0, 8.0, 1.0).sample(time)
    xy, _ = build_lateral_positions(maneuver(), progress, DOSE_TRANSITION_LENGTH_M["dose50"])
    states = derive_trajectory_states(xy, time, wheel_base_m=3.089)
    audit = dynamic_consistency_audit(xy, time, states)
    assert audit["finite"]
    assert audit["time_strictly_monotonic"]
    assert audit["max_velocity_derivative_error_mps"] < 1e-9
    assert audit["max_heading_step_rad"] < 0.25
    assert audit["max_abs_lateral_accel_mps2"] < 5.0


def test_canonical_generator_is_independent_of_dose() -> None:
    generator = CanonicalLongitudinalProgressGenerator(4.5, 8.0, 1.0)
    time = np.arange(0.0, 8.1, 0.1)
    arrays = [generator.sample(time)[0] for _ in DOSE_TRANSITION_LENGTH_M]
    for array in arrays[1:]:
        np.testing.assert_array_equal(arrays[0], array)


def test_safe_development_lengths_preserve_one_dimensional_lateral_axis() -> None:
    safe_lengths = [60.0, 58.5, 57.0, 55.5, 54.0]
    assert all(a > b for a, b in zip(safe_lengths, safe_lengths[1:]))
    time = np.arange(0.0, 8.1, 0.1)
    progress, _, _ = CanonicalLongitudinalProgressGenerator(5.0, 8.0, 1.0).sample(time)
    trajectories = [build_lateral_positions(maneuver(), progress, length)[0] for length in safe_lengths]
    assert all(np.array_equal(progress, CanonicalLongitudinalProgressGenerator(5.0, 8.0, 1.0).sample(time)[0]) for _ in safe_lengths)
    assert not np.array_equal(trajectories[0], trajectories[-1])
