import argparse
import csv
import json

import pytest

from tools import stage6j_freeze_pure_longitudinal_confirmation as stage6j
from tools.stage7c1_run_nuplan_simulation import PLANNER_PROFILES, format_planner_hydra_overrides


PLANNER_A = "pdm_closed_assertive_longitudinal_v1"
PLANNER_B = "pdm_closed_conservative_longitudinal_v1"


def design(expected_counts=None):
    expected_counts = expected_counts or {
        "following_interaction": 1,
        "stop_go_control": 1,
        "longitudinal_high_motion": 1,
    }
    return {
        "schema_version": stage6j.DESIGN_SCHEMA_VERSION,
        "issue": "https://example.invalid/issue/249",
        "planner_a": PLANNER_A,
        "planner_b": PLANNER_B,
        "shared_lateral_parameters": {"lateral_offsets": [-0.5, 0.5]},
        "allowed_different_longitudinal_parameters": [
            "idm_policies.speed_limit_fraction",
            "idm_policies.fallback_target_velocity",
            "idm_policies.min_gap_to_lead_agent",
            "idm_policies.headway_time",
            "idm_policies.accel_max",
            "idm_policies.decel_max",
        ],
        "included_scenario_types": {
            "following_interaction": ["near_long_vehicle"],
            "stop_go_control": ["stationary_in_traffic"],
            "longitudinal_high_motion": ["high_magnitude_speed"],
        },
        "expected_selected_counts": expected_counts,
        "expected_selected_total": sum(expected_counts.values()),
        "excluded_tasks": ["lane_change"],
        "excluded_scenario_types": ["high_lateral_acceleration"],
    }


def write_csv(path, rows):
    fields = ["global_scenario_index", "task", "scenario_type", "log_name", "scenario_token", "db_file"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_registered_profiles_have_identical_lateral_and_different_longitudinal_parameters():
    rows, summary = stage6j.audit_planner_treatment(design())
    assert summary["pure_longitudinal_treatment"] is True
    assert summary["lateral_difference_count"] == 0
    assert summary["longitudinal_difference_count"] == 6
    by_name = {row["parameter"]: row for row in rows}
    assert by_name["lateral_offsets"]["same_value"] is True
    for planner in [PLANNER_A, PLANNER_B]:
        assert PLANNER_PROFILES[planner]["style_scope"] == "pure_longitudinal_closed_loop_planner"
        overrides = format_planner_hydra_overrides(planner)
        assert "planner.pdm_closed_planner.lateral_offsets=[-0.5,0.5]" in overrides


def test_planner_treatment_fails_if_shared_lateral_design_changes():
    changed = design()
    changed["shared_lateral_parameters"]["lateral_offsets"] = [0.0]
    with pytest.raises(ValueError, match="Shared lateral parameter"):
        stage6j.audit_planner_treatment(changed)


def test_freeze_selects_only_longitudinal_primary_and_writes_context(tmp_path):
    db_root = tmp_path / "db"
    db_root.mkdir()
    rows = [
        {"global_scenario_index": 0, "task": "following_interaction", "scenario_type": "near_long_vehicle", "log_name": "log-a", "scenario_token": "token-a", "db_file": "log-a.db"},
        {"global_scenario_index": 1, "task": "stop_go_control", "scenario_type": "stationary_in_traffic", "log_name": "log-b", "scenario_token": "token-b", "db_file": "log-b.db"},
        {"global_scenario_index": 2, "task": "high_motion_dynamics", "scenario_type": "high_magnitude_speed", "log_name": "log-c", "scenario_token": "token-c", "db_file": "log-c.db"},
        {"global_scenario_index": 3, "task": "high_motion_dynamics", "scenario_type": "high_lateral_acceleration", "log_name": "log-d", "scenario_token": "token-d", "db_file": "log-d.db"},
        {"global_scenario_index": 4, "task": "lane_change", "scenario_type": "changing_lane_to_left", "log_name": "log-e", "scenario_token": "token-e", "db_file": "log-e.db"},
    ]
    for row in rows:
        (db_root / row["db_file"]).touch()
    ledger = tmp_path / "ledger.csv"
    write_csv(ledger, rows)
    design_path = tmp_path / "design.json"
    design_path.write_text(json.dumps(design()), encoding="utf-8")
    output = tmp_path / "out"
    result = stage6j.freeze(
        argparse.Namespace(
            design_json=design_path,
            confirmation_ledger_csv=ledger,
            nuplan_db_root=db_root,
            output_dir=output,
            overwrite=False,
        )
    )
    assert result["status"] == "FROZEN_BEFORE_PURE_LONGITUDINAL_ROLLOUTS"
    assert result["selection_audit"]["selected_scenario_count"] == 3
    assert result["selection_audit"]["selected_rollout_count"] == 6
    assert result["treatment_audit"]["pure_longitudinal_treatment"] is True
    with (output / "stage6j_locked_scenarios.csv").open(encoding="utf-8") as handle:
        selected = list(csv.DictReader(handle))
    assert {row["scenario_token"] for row in selected} == {"token-a", "token-b", "token-c"}
    assert (output / "stage7c_context" / "merged_metadata.csv").is_file()
    assert (output / "stage6j_freeze_manifest.json").is_file()


def test_select_scenarios_fails_closed_on_missing_db(tmp_path):
    rows = [
        {"global_scenario_index": "0", "task": "following_interaction", "scenario_type": "near_long_vehicle", "log_name": "log-a", "scenario_token": "token-a", "db_file": "missing.db"},
    ]
    one = design({"following_interaction": 1})
    one["included_scenario_types"] = {"following_interaction": ["near_long_vehicle"]}
    one["expected_selected_total"] = 1
    with pytest.raises(FileNotFoundError, match="missing"):
        stage6j.select_scenarios(one, rows, tmp_path)
