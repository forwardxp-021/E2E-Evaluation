import unittest

from tools.r1_phaseb1_freeze_preparation import (
    HLC_GEN_V2_OPTIONS,
    MASTER_SEED,
    analyze_hlc_options,
    seeded_rank,
)


class R1PhaseB1FreezePreparationTest(unittest.TestCase):
    def test_hlc_three_options_are_synthetic_proposals_only(self) -> None:
        result = analyze_hlc_options()
        self.assertEqual(set(result["options"]), set(HLC_GEN_V2_OPTIONS))
        self.assertEqual(
            result["status"],
            "SYNTHETIC_DESIGN_COMPLETE_ALL_OPTIONS_PROPOSED_NOT_FROZEN",
        )
        for option in result["options"].values():
            self.assertEqual(option["status"], "PROPOSED_NOT_FROZEN")
            self.assertTrue(option["mechanism"]["pair"]["pass"])
            self.assertTrue(option["new_primary_f_match_all_cells_pass"])
            self.assertTrue(option["endpoint_validity_all_exact_on_parallel_lane_fixture"])
            self.assertEqual(len(option["cells"]), 12)
            self.assertEqual(sum(row["new_primary_f_match_pass"] for row in option["cells"]), 12)
            self.assertEqual(sum(row["engineering_safety_pass"] for row in option["cells"]), 12)

    def test_seeded_selector_rank_is_stable_and_identity_sensitive(self) -> None:
        first = seeded_rank("V1", "R-HLC", "abc", "log-a")
        self.assertEqual(first, seeded_rank("V1", "R-HLC", "abc", "log-a"))
        self.assertNotEqual(first, seeded_rank("V1", "R-HLC", "abd", "log-a"))
        self.assertEqual(MASTER_SEED, 2026082701)


if __name__ == "__main__":
    unittest.main()
