import argparse
import csv
import inspect
from pathlib import Path

import numpy as np

from tools.stage7l_dynamic_clearance import (
    DynamicClearanceConfig,
    common_envelope_conflict,
    interpolate_track_state,
)
from tools import stage7l_audit_dynamic_lane_change_clearance as audit_tool


def test_time_alignment_interpolates_without_extrapolation() -> None:
    times = np.asarray([0.0, 0.2], dtype=float)
    states = np.asarray([[0.0, 0.0, 4.0, 2.0, 0.0], [2.0, 0.0, 4.0, 2.0, 0.0]], dtype=float)
    value = interpolate_track_state(times, states, 0.1, 0.25)
    assert value is not None
    np.testing.assert_allclose(value[:2], [1.0, 0.0])
    assert interpolate_track_state(times, states, 0.3, 0.25) is None


def test_common_envelope_accepts_distant_agent_and_rejects_future_transition_conflict() -> None:
    config = DynamicClearanceConfig()
    source = np.asarray([10.0, 0.0])
    target = np.asarray([10.0, 3.5])
    far = np.asarray([50.0, 20.0, 4.5, 2.0, 0.0])
    conflict, _, _ = common_envelope_conflict(source, target, 0.0, far, "transition", config)
    assert not conflict
    # This agent could be farther than the initial 15 m target-lane gap at t=0,
    # yet enters the shared source-to-target strip at a future aligned time.
    future = np.asarray([10.0, 1.75, 4.5, 2.0, 0.0])
    conflict, _, _ = common_envelope_conflict(source, target, 0.0, future, "transition", config)
    assert conflict


def test_direction_symmetry_for_mirrored_geometry() -> None:
    config = DynamicClearanceConfig()
    right_source = np.asarray([5.0, 0.0]); right_target = np.asarray([5.0, 3.5])
    left_source = np.asarray([5.0, 0.0]); left_target = np.asarray([5.0, -3.5])
    right_agent = np.asarray([5.0, 1.75, 4.0, 2.0, 0.0])
    left_agent = np.asarray([5.0, -1.75, 4.0, 2.0, 0.0])
    assert common_envelope_conflict(right_source, right_target, 0.0, right_agent, "transition", config)[0]
    assert common_envelope_conflict(left_source, left_target, 0.0, left_agent, "transition", config)[0]


def test_buffer_boundary_and_dose_independence() -> None:
    config = DynamicClearanceConfig(ego_length_m=5.0, ego_width_m=2.0, longitudinal_buffer_m=3.0, lateral_buffer_m=0.5)
    source = np.asarray([0.0, 0.0]); target = np.asarray([0.0, 3.5])
    # Agent half length=2.0; permitted boundary is 2.5 + 2.0 + 3.0 = 7.5 m.
    boundary = np.asarray([7.5, 1.75, 4.0, 2.0, 0.0])
    outside = np.asarray([7.5001, 1.75, 4.0, 2.0, 0.0])
    assert common_envelope_conflict(source, target, 0.0, boundary, "transition", config)[0]
    assert not common_envelope_conflict(source, target, 0.0, outside, "transition", config)[0]
    assert "dose" not in inspect.signature(common_envelope_conflict).parameters


def test_missing_track_is_explicitly_not_interpolated() -> None:
    times = np.asarray([0.0, 1.0], dtype=float)
    states = np.asarray([[0.0, 0.0, 4.0, 2.0, 0.0], [1.0, 0.0, 4.0, 2.0, 0.0]], dtype=float)
    assert interpolate_track_state(times, states, 0.5, 0.25) is None


def test_historical_audit_reads_tokens_without_token_specific_rule(tmp_path: Path, monkeypatch) -> None:
    """The CSV audit is generic: a development token is only input provenance."""
    candidate_csv = tmp_path / "development.csv"
    with candidate_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scenario_token", "eligible"])
        writer.writeheader()
        writer.writerow({"scenario_token": "historical-development-token", "eligible": "True"})

    def fake_dynamic_audit(candidate, _db_root, _config):
        return {
            "dynamic_clearance_pass": False,
            "dynamic_reason_code": "TRANSITION_CORRIDOR_DYNAMIC_CONFLICT",
            "audit_token_echo": candidate["scenario_token"],
        }

    monkeypatch.setattr(audit_tool, "dynamic_clearance_audit", fake_dynamic_audit)
    args = argparse.Namespace(
        candidate_csv=candidate_csv,
        nuplan_db_root=tmp_path,
        output_dir=tmp_path / "audit",
        only_static_eligible=True,
        horizon_seconds=15.0,
        time_step_seconds=0.1,
        maximum_track_interpolation_gap_seconds=0.25,
        trigger_route_progress_m=12.0,
        gentle_transition_length_m=60.0,
        settling_margin_m=10.0,
        target_speed_mps=5.0,
        accel_limit_mps2=1.0,
        ego_length_m=5.0,
        ego_width_m=2.0,
        longitudinal_buffer_m=3.0,
        lateral_buffer_m=0.5,
    )
    summary = audit_tool.run(args)
    assert summary["candidate_count"] == 1
    with (args.output_dir / "dynamic_clearance_audit.csv").open(encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["audit_token_echo"] == "historical-development-token"
    assert row["dynamic_reason_code"] == "TRANSITION_CORRIDOR_DYNAMIC_CONFLICT"
    assert "historical-development-token" not in inspect.getsource(audit_tool.dynamic_clearance_audit)
