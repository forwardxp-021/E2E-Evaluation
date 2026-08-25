import pytest

from tools.stage7_m3_final_audit import symmetric_failed_scenarios


def test_symmetric_failed_scenarios_accepts_complete_planner_pair() -> None:
    records = [
        {"scenario_index": 7, "planner": "a", "status": "failed"},
        {"scenario_index": 7, "planner": "b", "status": "failed"},
        {"scenario_index": 8, "planner": "a", "status": "succeeded"},
        {"scenario_index": 8, "planner": "b", "status": "succeeded"},
    ]
    assert sorted(symmetric_failed_scenarios(records, ["a", "b"])) == [7]


def test_symmetric_failed_scenarios_rejects_one_planner_failure() -> None:
    records = [{"scenario_index": 7, "planner": "a", "status": "failed"}]
    with pytest.raises(ValueError, match="not planner-symmetric"):
        symmetric_failed_scenarios(records, ["a", "b"])
