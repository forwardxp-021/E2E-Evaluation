"""Unit checks for fail-closed R1 runtime-determinism bookkeeping only."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.r1_prepare_runtime_determinism_roster import rank_digest
from tools.r1_run_runtime_determinism_validation import OfficialRunBudget
from tools.r1_runtime_determinism_planner import R1RuntimeDeterminismPlanner


class TestR1RuntimeDeterminism(unittest.TestCase):
    def test_frozen_rank_is_deterministic_and_family_specific(self) -> None:
        salt = "f176f57a242a0ea5a2a7c38fb6c3527e9f6385aca7d5ac21d522e6b097d1a3b3"
        first = rank_digest(salt, "R-HLC", "a" * 16, "log-a")
        self.assertEqual(first, rank_digest(salt, "R-HLC", "a" * 16, "log-a"))
        self.assertNotEqual(first, rank_digest(salt, "R-TSB", "a" * 16, "log-a"))

    def test_budget_rejects_ninth_before_simulation(self) -> None:
        schedule = [{"run_id": f"run-{index}", "scenario_token": str(index)} for index in range(8)]
        with tempfile.TemporaryDirectory() as temporary:
            ledger = OfficialRunBudget.create(8, schedule, Path(temporary) / "ledger.json")
            for run in schedule:
                ledger.claim(run)
            self.assertEqual(8, len(ledger.records))
            with self.assertRaisesRegex(RuntimeError, "refusing unplanned official run"):
                ledger.claim({"run_id": "ninth", "scenario_token": "9"})
            written = json.loads((Path(temporary) / "ledger.json").read_text(encoding="utf-8"))
            self.assertEqual(8, written["claimed_count"])

    def test_planner_exposes_required_official_entry_points(self) -> None:
        self.assertTrue(callable(getattr(R1RuntimeDeterminismPlanner, "compute_trajectory", None)))
        self.assertTrue(callable(getattr(R1RuntimeDeterminismPlanner, "generate_planner_report", None)))


if __name__ == "__main__":
    unittest.main()
