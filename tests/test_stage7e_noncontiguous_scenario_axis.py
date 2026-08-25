import json
from pathlib import Path

import pytest

from tools.build_nuplan_5neighbor_context_dataset import (
    load_stage7c_scenario_axis,
    metadata_rows,
    validate_scenario_planner_alignment,
)
from tools.stage7d_extract_neighbors_from_nuplan import find_msgpack


PLANNERS = ["planner_conservative", "planner_assertive"]


def _index_rows():
    rows = []
    tokens = {0: "token_zero", 2: "token_two"}
    for scenario_index in [0, 1, 2]:
        for planner_id, planner_name in enumerate(PLANNERS):
            rows.append(
                {
                    "scenario_index": str(scenario_index),
                    "planner_id": str(planner_id),
                    "planner_name": planner_name,
                    "status": "failed" if scenario_index == 1 else "succeeded",
                    "scenario_token": tokens.get(scenario_index, "failed_token"),
                    "scene_token": tokens.get(scenario_index, "failed_token"),
                    "log_name": f"log_{scenario_index}",
                }
            )
    return rows


def test_noncontiguous_scenario_axis_preserves_original_indices(tmp_path: Path):
    axis_path = tmp_path / "simulated_ego_seq_index.json"
    axis_path.write_text(
        json.dumps(
            {
                "scenario_axis": ["0", "2"],
                "planner_axis": ["0", "1"],
                "planner_axis_names": PLANNERS,
                "shape": [2, 2, 150, 8],
            }
        ),
        encoding="utf-8",
    )
    scenario_axis = load_stage7c_scenario_axis(axis_path, 2, 2, PLANNERS)
    by_pair = validate_scenario_planner_alignment(_index_rows(), scenario_axis, PLANNERS)
    rows = metadata_rows(_index_rows(), [], PLANNERS, scenario_axis)

    assert scenario_axis == [0, 2]
    assert sorted(by_pair) == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    assert [row["scenario_index"] for row in rows] == [0, 0, 2, 2]
    assert [row["tensor_scenario_position"] for row in rows] == [0, 0, 1, 1]
    assert [row["scenario_token"] for row in rows] == [
        "token_zero",
        "token_zero",
        "token_two",
        "token_two",
    ]


def test_alignment_rejects_failed_axis_row():
    with pytest.raises(ValueError, match="non-successful"):
        validate_scenario_planner_alignment(_index_rows(), [0, 1, 2], PLANNERS)


def test_find_msgpack_has_no_global_fallback_and_checks_token(tmp_path: Path):
    sim_dir = tmp_path / "sim"
    correct = (
        sim_dir
        / "official_nuplan_runs"
        / "scenario_2"
        / PLANNERS[0]
        / "simulation_log"
        / "token_two"
        / "token_two.msgpack.xz"
    )
    correct.parent.mkdir(parents=True)
    correct.write_bytes(b"test")
    unrelated = (
        sim_dir
        / "official_nuplan_runs"
        / "scenario_0"
        / PLANNERS[0]
        / "simulation_log"
        / "token_zero"
        / "token_zero.msgpack.xz"
    )
    unrelated.parent.mkdir(parents=True)
    unrelated.write_bytes(b"test")

    row = {
        "scenario_index": "2",
        "planner_id": "0",
        "planner_name": PLANNERS[0],
        "scenario_token": "token_two",
    }
    assert find_msgpack(sim_dir, row) == correct.resolve()

    missing_row = {
        "scenario_index": "3",
        "planner_id": "0",
        "planner_name": PLANNERS[0],
        "scenario_token": "token_three",
    }
    assert find_msgpack(sim_dir, missing_row) is None

    wrong_token_row = {**row, "scenario_token": "wrong_token"}
    with pytest.raises(ValueError, match="No msgpack token match"):
        find_msgpack(sim_dir, wrong_token_row)
