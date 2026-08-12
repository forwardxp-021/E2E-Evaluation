import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.stage6p_run_representation_unpaired_release import calibrate, median_bandwidth
from tools.stage6q_audit_waymo_raw_interaction_coverage import event_flags


ROOT = Path(__file__).resolve().parents[1]


def test_stage6p_frozen_config_forbids_cross_representation_raw_mmd_comparison() -> None:
    config = json.loads((ROOT / "configs/stage6p_representation_unpaired_release.json").read_text())
    assert config["statistic"]["cross_representation_raw_mmd2_comparison_forbidden"] is True
    assert [row["id"] for row in config["representations"]] == [
        "full64",
        "ego13",
        "handcrafted46",
        "neighbor_zero64",
    ]
    assert config["representations"][-1]["role"] == "diagnostic_only"


def test_stage6p_bandwidth_and_threshold_are_representation_specific() -> None:
    values = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    assert median_bandwidth(values, seed=7, max_pairs=200) > 0
    rows = []
    for representation, offset in [("full64", 0.0), ("ego13", 10.0)]:
        for index, family in enumerate(["AA_CALIBRATION"] * 4 + ["AA_EVALUATION", "AB_EVALUATION"]):
            rows.append(
                {
                    "representation": representation,
                    "target_scenarios_per_release": 200,
                    "family": family,
                    "raw_mmd2_within_representation_only": offset + index,
                }
            )
    thresholds, evaluated = calibrate(pd.DataFrame(rows), 0.05)
    assert len(thresholds) == 2
    assert thresholds.set_index("representation").loc["ego13", "q95_threshold_within_representation_only"] == 13.0
    assert evaluated.groupby("representation")["threshold_within_representation_only"].nunique().eq(1).all()


def test_stage6q_detects_entry_exit_and_intermittent_without_neighbor_valid_filter() -> None:
    lead_ids = [None] * 10 + ["12"] * 25 + [None] * 45
    gaps = np.asarray([np.nan] * 10 + [30.0] * 25 + [np.nan] * 45)
    closing = np.asarray([np.nan] * 10 + [1.0] * 10 + [0.0] * 15 + [np.nan] * 45)
    flags = event_flags(lead_ids, gaps, closing, minimum=5)
    assert flags["lead_entry"] is True
    assert flags["lead_exit"] is True
    assert flags["intermittent_following_primary"] is True
    assert flags["intermittent_following_strict"] is True
    assert flags["following_to_free_flow"] is True


def test_stage6q_config_preserves_stage6o_gate_and_style_definition() -> None:
    config = json.loads((ROOT / "configs/stage6q_waymo_raw_interaction_coverage_audit.json").read_text())
    assert config["root_cause_rule"]["threshold_change_forbidden"] is True
    assert config["root_cause_rule"]["builder_structural_filter"].find(">=5000") >= 0
    assert "behavior style" in config["interpretation_constraints"][1]
    assert "modify_or_overwrite_stage6o_v1" in config["forbidden_actions"]
