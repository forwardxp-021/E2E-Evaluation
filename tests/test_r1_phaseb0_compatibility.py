from __future__ import annotations

import unittest

import numpy as np

from tools.r1_context_mechanism_core import calculate_tsb_option_a, qualify_tsb_pair
from tools.r1_phaseb0_compatibility_audit import (
    TSB_GEN_V2_OPTIONS,
    build_results,
    integrate_tsb_profile,
    synthetic_hlc_progress,
    synthetic_hlc_witness,
    synthetic_tsb_options,
)
from tools.r1_residual_generators import (
    HLC_SMOKE_CANDIDATES,
    TSB_SMOKE_CANDIDATES,
    generate_hlc_trajectory,
    generate_tsb_trajectory,
)
from tools.stage7l_pure_lateral_execution_planner import quintic_blend_d1, quintic_blend_d2
from tools.stageR_execute_r1_technical_smoke import (
    AUTHORIZED_CORE_CONSTRUCTION_CAP,
    CoreConstructionBudget,
    build_core_construction_schedule,
)


def synthetic_parallel_lane_maneuver() -> dict:
    x = np.linspace(0.0, 100.0, 101)
    return {
        "scenario_token": "SYNTHETIC_ONLY",
        "log_name": "NO_REAL_LOG",
        "db_file": "NO_DATABASE",
        "initial_state_fingerprint": "SYNTHETIC",
        "initial_x": 0.0,
        "initial_y": 0.0,
        "initial_heading": 0.0,
        "initial_speed_mps": 13.292885,
        "source_lane_id": "SYNTHETIC_SOURCE",
        "target_lane_id": "SYNTHETIC_TARGET",
        "source_roadblock_id": "SYNTHETIC_ROADBLOCK",
        "target_roadblock_id": "SYNTHETIC_ROADBLOCK",
        "direction": "LEFT",
        "route_roadblock_ids": ["SYNTHETIC_ROADBLOCK"],
        "route_fingerprint": "SYNTHETIC",
        "trigger_s_route_m": 0.0,
        "source_start_arc_m": 0.0,
        "target_start_arc_m": 0.0,
        "nominal_lane_width_m": 2.7,
        "horizon_s": 4.5,
        "background_mode": "NONE",
        "background_agent_model": "NONE",
        "background_config_sha256": "SYNTHETIC",
        "source_reference_xy": [[float(value), 0.0] for value in x],
        "target_reference_xy": [[float(value), 2.7] for value in x],
        "planner_profile_ids": ["SYNTHETIC"],
    }


class HlcCompatibilityTest(unittest.TestCase):
    def test_analytical_synthetic_witness_passes_both_frozen_gates(self) -> None:
        witness = synthetic_hlc_witness()
        self.assertTrue(witness["mechanism_pair"]["pass"])
        self.assertTrue(witness["f_match"]["pass"])
        self.assertGreaterEqual(witness["treatment_mechanism"]["hesitation_retreat_count"], 1)
        self.assertLessEqual(
            witness["treatment_mechanism"]["monotonic_transition_fraction"],
            witness["baseline_mechanism"]["monotonic_transition_fraction"] - 0.1,
        )

    def test_current_hlc_generator_has_common_prefix_and_terminal_match(self) -> None:
        maneuver = synthetic_parallel_lane_maneuver()
        baseline = generate_hlc_trajectory(maneuver)
        prefix = baseline["time_s"] < 1.1
        for candidate_id in HLC_SMOKE_CANDIDATES:
            treatment = generate_hlc_trajectory(maneuver, candidate_id)
            np.testing.assert_allclose(treatment["xy"][prefix], baseline["xy"][prefix], atol=1e-12)
            self.assertAlmostEqual(float(treatment["progress_p"][-1]), 1.0)
            self.assertTrue(np.isfinite(treatment["states"]["curvature"]).all())
            self.assertTrue(np.isfinite(treatment["states"]["yaw_rate"]).all())

    def test_quintic_phase_boundaries_are_c2_and_position_continuous(self) -> None:
        for u in (0.0, 1.0):
            self.assertAlmostEqual(float(quintic_blend_d1(u)), 0.0)
            self.assertAlmostEqual(float(quintic_blend_d2(u)), 0.0)
        time_s = np.arange(0.0, 6.05, 0.1)
        treatment = synthetic_hlc_progress(time_s, "TREATMENT")
        expected = {1.0: 0.0, 1.8: 0.35, 2.3: 0.35, 3.1: 0.20, 4.6: 1.0}
        for boundary_s, expected_p in expected.items():
            index = int(round(boundary_s / 0.1))
            self.assertAlmostEqual(float(treatment[index]), expected_p, places=12)


class TsbCompatibilityTest(unittest.TestCase):
    def test_all_proposed_v2_profiles_pass_frozen_mechanism_and_fmatch(self) -> None:
        results = synthetic_tsb_options()
        self.assertEqual(set(results["options"]), set(TSB_GEN_V2_OPTIONS))
        for row in results["options"].values():
            self.assertTrue(row["mechanism_pair"]["pass"])
            self.assertTrue(row["f_match"]["pass"])
            self.assertEqual(row["mechanism"]["brake_phase_count"], 2)

    def test_current_profile_failure_is_reproducible_without_timestamp_bug(self) -> None:
        baseline = generate_tsb_trajectory(8.0)
        baseline_mechanism = calculate_tsb_option_a(baseline["time_s"], baseline["speed_mps"])
        self.assertEqual(baseline_mechanism["brake_phase_count"], 1)
        for candidate_id in TSB_SMOKE_CANDIDATES:
            treatment = generate_tsb_trajectory(8.0, candidate_id)
            self.assertTrue(np.allclose(np.diff(treatment["time_s"]), 0.1))
            mechanism = calculate_tsb_option_a(treatment["time_s"], treatment["speed_mps"])
            self.assertIn(mechanism["brake_phase_count"], (1, 2))
            self.assertFalse(qualify_tsb_pair(baseline_mechanism, mechanism)["pass"])

    def test_synthetic_integrator_avoids_low_speed_endstop(self) -> None:
        trajectory = integrate_tsb_profile(8.0, ((-1.0, 0.7), (0.8, 0.6), (-1.0, 0.5)))
        self.assertGreater(float(np.min(trajectory["speed_mps"])), 1.0)


class ExecutorCapTest(unittest.TestCase):
    def test_schedule_is_exactly_48_and_reuses_one_baseline_per_scenario(self) -> None:
        families = {
            "R-HLC": [f"HLC_SYNTH_{index}" for index in range(6)],
            "R-TSB": [f"TSB_SYNTH_{index}" for index in range(6)],
        }
        candidates = {
            "R-HLC": ["HLC_A", "HLC_B", "HLC_C"],
            "R-TSB": ["TSB_A", "TSB_B", "TSB_C"],
        }
        schedule = build_core_construction_schedule(families, candidates)
        self.assertEqual(len(schedule), AUTHORIZED_CORE_CONSTRUCTION_CAP)
        for family, scenario_ids in families.items():
            family_rows = [row for row in schedule if row["family"] == family]
            self.assertEqual(len(family_rows), 24)
            for scenario_id in scenario_ids:
                rows = [row for row in family_rows if row["scenario_id"] == scenario_id]
                self.assertEqual(sum(row["arm"] == "BASELINE" for row in rows), 1)
                self.assertEqual(sum(row["arm"].startswith("TREATMENT::") for row in rows), 3)

    def test_budget_stops_before_call_49(self) -> None:
        budget = CoreConstructionBudget(authorized_cap=48)
        for index in range(48):
            budget.claim("SYNTHETIC", f"ID_{index}", "ARM")
        with self.assertRaises(RuntimeError):
            budget.claim("SYNTHETIC", "ID_48", "ARM")
        self.assertEqual(budget.actual_calls, 48)
        budget.assert_exact(48)

    def test_budget_ledger_schema_and_planned_actual_counters(self) -> None:
        families = {
            "R-HLC": [f"HLC_{index}" for index in range(6)],
            "R-TSB": [f"TSB_{index}" for index in range(6)],
        }
        candidates = {
            "R-HLC": ["A", "B", "C"],
            "R-TSB": ["A", "B", "C"],
        }
        schedule = build_core_construction_schedule(families, candidates)
        budget = CoreConstructionBudget(planned_schedule=schedule)
        for row in schedule:
            budget.claim(row["family"], row["scenario_id"], row["arm"])
        budget.assert_exact(48)
        self.assertEqual(tuple(budget.ledger[0]), CoreConstructionBudget.LEDGER_SCHEMA)
        self.assertEqual(
            budget.counters(),
            {
                "planned_core_construction_calls": 48,
                "actual_core_construction_calls": 48,
                "authorized_cap": 48,
            },
        )
        self.assertTrue(
            all(row["claim_status"] == "CLAIMED_BEFORE_CONSTRUCTION" for row in budget.ledger)
        )

    def test_duplicate_baseline_fails_closed_before_increment(self) -> None:
        budget = CoreConstructionBudget(authorized_cap=48)
        budget.claim("R-HLC", "S0", "BASELINE")
        with self.assertRaisesRegex(RuntimeError, "duplicate baseline construction blocked"):
            budget.claim("R-HLC", "S0", "BASELINE")
        self.assertEqual(budget.actual_calls, 1)

    def test_unplanned_call_fails_closed_before_increment(self) -> None:
        schedule = [
            {"family": "R-HLC", "scenario_id": "S0", "arm": "BASELINE"},
        ]
        budget = CoreConstructionBudget(authorized_cap=48, planned_schedule=schedule)
        with self.assertRaisesRegex(RuntimeError, "unplanned trajectory construction blocked"):
            budget.claim("R-HLC", "S1", "BASELINE")
        self.assertEqual(budget.actual_calls, 0)

    def test_combined_result_builder_opens_only_allowlisted_inputs(self) -> None:
        results = build_results()
        self.assertEqual(results["hlc"]["classification"], "MARGINALLY_FEASIBLE")
        self.assertEqual(results["tsb"]["classification"], "JOINTLY_FEASIBLE")


if __name__ == "__main__":
    unittest.main()
