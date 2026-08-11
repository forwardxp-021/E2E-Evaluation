from __future__ import annotations

import csv
import json
from argparse import Namespace
from pathlib import Path

import pytest

from tools.stage6k_freeze_longitudinal_dose_response import freeze
from tools.stage6k_run_longitudinal_dose_rollouts import selected_rows
from tools.stage7c1_run_nuplan_simulation import PLANNER_PROFILES


ROOT = Path(__file__).resolve().parents[1]


def test_registered_profiles_are_exact_linear_interpolations() -> None:
    baseline = PLANNER_PROFILES["pdm_closed_conservative_longitudinal_v1"]["parameters"]
    anchor = PLANNER_PROFILES["pdm_closed_assertive_longitudinal_v1"]["parameters"]
    profiles = {
        0.25: "pdm_closed_assertive_longitudinal_dose25_v1",
        0.50: "pdm_closed_assertive_longitudinal_dose50_v1",
        0.75: "pdm_closed_assertive_longitudinal_dose75_v1",
    }
    numeric = [
        "idm_policies.speed_limit_fraction", "idm_policies.fallback_target_velocity",
        "idm_policies.min_gap_to_lead_agent", "idm_policies.headway_time",
        "idm_policies.accel_max", "idm_policies.decel_max",
    ]
    for dose, name in profiles.items():
        actual = PLANNER_PROFILES[name]["parameters"]
        assert actual["lateral_offsets"] == [-0.5, 0.5]
        for key in numeric:
            left, right, value = baseline[key], anchor[key], actual[key]
            if isinstance(left, list):
                expected = [a + dose * (b - a) for a, b in zip(left, right)]
                assert value == pytest.approx(expected)
            else:
                assert value == pytest.approx(left + dose * (right - left))


def test_real_freeze_builds_549_ordered_jobs_without_bdd(tmp_path: Path) -> None:
    result = freeze(Namespace(
        design_json=ROOT / "configs/stage6k_longitudinal_dose_response.json",
        stage6j_locked_scenarios_csv=ROOT / "outputs/stage6j_pure_longitudinal_freeze_v1/stage6j_locked_scenarios.csv",
        output_dir=tmp_path / "freeze",
        overwrite=False,
    ))
    assert result["status"] == "FROZEN_BEFORE_LONGITUDINAL_DOSE_ROLLOUTS"
    assert result["embedding_or_bdd_read"] is False
    assert result["job_audit"] == {"job_count": 549, "planned_rollout_count": 1098, "dose_count": 3}
    with (tmp_path / "freeze/stage6k_locked_jobs.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["collection_order"]) for row in rows] == list(range(1, 550))
    assert {row["dose_label"] for row in rows[:183]} == {"dose25"}
    assert {row["dose_label"] for row in rows[183:366]} == {"dose50"}
    assert {row["dose_label"] for row in rows[366:]} == {"dose75"}
    assert len({row["scenario_token"] for row in rows[:183]}) == 183
    manifest = json.loads((tmp_path / "freeze/stage6k_freeze_manifest.json").read_text())
    assert manifest["profile_audit"]["interpolation_passed"] is True


def test_runner_range_can_select_one_smoke_per_dose() -> None:
    rows = [{"collection_order": str(index)} for index in range(1, 550)]
    for start in [1, 184, 367]:
        args = Namespace(start_order=start, end_order=0, max_jobs=1)
        assert selected_rows(args, rows) == [{"collection_order": str(start)}]
