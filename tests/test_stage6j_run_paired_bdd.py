from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools import stage6j_run_paired_bdd as bdd


def test_task_masks_use_pair_matched_pretreatment_scenario_type() -> None:
    metadata = pd.DataFrame(
        [
            {"global_row": 0, "scenario_type": "near_long_vehicle"},
            {"global_row": 1, "scenario_type": "near_long_vehicle"},
            {"global_row": 2, "scenario_type": "high_magnitude_speed"},
            {"global_row": 3, "scenario_type": "high_magnitude_speed"},
        ]
    )
    masks, types = bdd.build_task_masks(
        metadata,
        np.asarray([[0, 1], [2, 3]], dtype=np.int64),
        {
            "following": ["near_long_vehicle"],
            "high_motion": ["high_magnitude_speed"],
        },
    )
    assert types == ["near_long_vehicle", "high_magnitude_speed"]
    assert masks["following"].tolist() == [True, False]
    assert masks["high_motion"].tolist() == [False, True]


def test_task_masks_reject_unmapped_or_pair_mismatched_types() -> None:
    metadata = pd.DataFrame(
        [
            {"global_row": 0, "scenario_type": "near_long_vehicle"},
            {"global_row": 1, "scenario_type": "high_magnitude_speed"},
        ]
    )
    with pytest.raises(ValueError, match="unequal pre-treatment"):
        bdd.build_task_masks(
            metadata,
            np.asarray([[0, 1]], dtype=np.int64),
            {"following": ["near_long_vehicle"]},
        )
