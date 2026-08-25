from tools.audit_stage7_m2a_lane_assignment import evaluate_m2a


def test_m2a_audit_requires_scenario_local_cache_and_material_fallback_reduction() -> None:
    baseline = {"fallback_assignment_used_rate": 0.86}
    repaired = {
        "pass": True,
        "fallback_assignment_used_rate": 0.20,
        "lane_cache_scope": "map_name_plus_source_scenario",
        "lane_cache_entry_count": 2,
        "map_api_cache_entry_count": 1,
        "map_names_used": ["us-nv-las-vegas-strip"],
        "log_db_map_resolution_count": 2,
        "ego_seq_mask_shape": [4, 150],
        "map_query_success": True,
        "ego_lane_projection_success_rate": 0.80,
        "candidate_lane_projection_success_rate": 0.75,
        "lane_assignment_fallback_reason_counts": {"lateral_distance_exceeded": 100},
    }
    rows = [
        {"scenario_index": str(scenario), "planner_name": planner}
        for scenario in (0, 4)
        for planner in ("a", "b")
    ]
    result = evaluate_m2a(
        baseline,
        repaired,
        rows,
        max_fallback_rate=0.5,
        min_absolute_improvement=0.3,
    )
    assert result["overall_verdict"] == "PASS"
    assert all(result["checks"].values())


def test_m2a_audit_fails_when_one_scenario_is_missing_a_planner() -> None:
    baseline = {"fallback_assignment_used_rate": 0.86}
    repaired = {
        "pass": True,
        "fallback_assignment_used_rate": 0.20,
        "lane_cache_scope": "map_name_plus_source_scenario",
        "lane_cache_entry_count": 2,
        "map_api_cache_entry_count": 1,
        "map_names_used": ["us-nv-las-vegas-strip"],
        "log_db_map_resolution_count": 2,
        "ego_seq_mask_shape": [3, 150],
        "map_query_success": True,
        "ego_lane_projection_success_rate": 0.80,
        "candidate_lane_projection_success_rate": 0.75,
    }
    rows = [
        {"scenario_index": "0", "planner_name": "a"},
        {"scenario_index": "0", "planner_name": "b"},
        {"scenario_index": "4", "planner_name": "a"},
    ]
    result = evaluate_m2a(
        baseline,
        repaired,
        rows,
        max_fallback_rate=0.5,
        min_absolute_improvement=0.3,
    )
    assert result["overall_verdict"] == "FAIL"
    assert result["checks"]["every_scenario_has_all_planners"] is False
