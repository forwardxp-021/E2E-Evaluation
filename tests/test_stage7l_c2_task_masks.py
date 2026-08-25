import json

import pytest

from tools.stage7l_generate_pretreatment_task_masks import build_task_mask_rows


def roster(token: str, log: str = "log") -> dict:
    return {"scenario_token": token, "log_name": log}


def source(token: str, types: list[str]) -> dict:
    return {
        "scenario_token": token,
        "official_scenario_types_json": json.dumps(types),
    }


def test_lane_change_is_roster_membership_and_dynamics_is_pretreatment_type() -> None:
    rows = build_task_mask_rows(
        [roster("a"), roster("b")],
        [
            source("a", ["high_lateral_acceleration", "unrelated"]),
            source("b", ["changing_lane_to_left"]),
        ],
    )
    assert [row["LAT.LANE_CHANGE"] for row in rows] == [True, True]
    assert [row["LAT.DYNAMICS"] for row in rows] == [True, False]
    assert all(row["selection_timing"] == "pre_treatment" for row in rows)


def test_missing_pretreatment_metadata_fails_closed() -> None:
    with pytest.raises(ValueError, match="missing from pre-treatment source"):
        build_task_mask_rows([roster("missing")], [source("other", [])])


def test_duplicate_source_token_fails_closed() -> None:
    with pytest.raises(ValueError, match="duplicate scenario_token"):
        build_task_mask_rows(
            [roster("a")],
            [source("a", []), source("a", ["high_magnitude_speed"])],
        )
