import inspect
import json

from tools.stage7l_freeze_confirmation_roster import select_roster_rows


def make_row(direction: str, index: int, log_name: str) -> dict:
    source = [[0.0, 0.0], [10.0, 0.0], [20.0, 0.1 * index], [30.0, 0.2 * index]]
    target = [[0.0, 3.5], [10.0, 3.5], [20.0, 3.5 + 0.1 * index], [30.0, 3.5 + 0.2 * index]]
    return {
        "scenario_token": f"{index:016x}",
        "log_name": log_name,
        "db_file": f"{log_name}.db",
        "direction": direction,
        "initial_speed_mps": str(3.0 + index % 12),
        "source_reference_xy_json": json.dumps(source),
        "target_reference_xy_json": json.dumps(target),
        "nominal_lane_width_m": str(3.0 + (index % 5) * 0.2),
        "dynamic_replay_track_count": str(2 + index % 9),
        "paired_reference_remaining_m": str(95.0 + index),
        "source_roadblock_id": str(index % 7),
        "route_fingerprint": f"route-{index % 11}",
    }


def protocol() -> dict:
    return {"selection": {"direction_quotas": {"left": 15, "right": 65}}}


def test_seeded_selection_is_reproducible_and_respects_direction_quota() -> None:
    rows = []
    # 19 left candidates across 14 logs forces exactly one log reuse for 15 selections.
    for index in range(19):
        rows.append(make_row("left", index, f"left-log-{index % 14}"))
    for index in range(19, 89):
        rows.append(make_row("right", index, f"right-log-{index}"))
    first, first_trace = select_roster_rows(rows, protocol())
    second, second_trace = select_roster_rows(rows, protocol())
    assert [row["scenario_token"] for row in first] == [row["scenario_token"] for row in second]
    assert [row["scenario_token"] for row in first_trace] == [row["scenario_token"] for row in second_trace]
    assert len(first) == 80
    assert sum(row["direction"] == "left" for row in first) == 15
    assert sum(row["direction"] == "right" for row in first) == 65
    assert len({row["log_name"] for row in first}) == 79
    assert sum(row["selection_global_log_action"] == "REQUIRED_DIRECTION_LOG_REUSE" for row in first) == 1


def test_selection_uses_pretreatment_geometry_not_outcomes() -> None:
    source = inspect.getsource(select_roster_rows)
    for forbidden in ("rollout", "collision", "completion", "embedding", "bdd", "mmd"):
        assert forbidden not in source.lower()
