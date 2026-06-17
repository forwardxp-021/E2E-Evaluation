import csv
import json
from pathlib import Path

import pytest

from tools.compare_lane_aware_diagnostics import diagnose, summarize_nuplan, summarize_waymo
from tools.nuplan_projection_debug import write_projection_debug_artifacts


def test_compare_inconclusive_when_waymo_projection_and_fallback_missing():
    waymo = {"fallback_assignment_used_rate": None, "candidate_projection_success_rate": None}
    nuplan = {"fallback_assignment_used_rate": 0.4, "candidate_projection_success_rate": 0.03}
    out = diagnose(waymo, nuplan, 0.2)
    assert out["verdict"] == "inconclusive_missing_comparable_metrics"
    assert out["confidence"] == "inconclusive"


def test_compare_consumes_waymo_diagnostic_export(tmp_path: Path):
    d = tmp_path / "waymo"; d.mkdir()
    (d / "waymo_lane_aware_diagnostics.json").write_text(json.dumps({
        "fallback_assignment_used_rate": 0.1,
        "candidate_projection_success_rate": 0.9,
        "slot_coverage_metric_source": "array_derived",
    }), encoding="utf-8")
    metrics = summarize_waymo(d, max_rows=10)
    assert metrics["candidate_projection_success_rate"] == 0.9
    assert metrics["slot_coverage_metric_source"] == "array_derived"


def test_compare_consumes_nuplan_projection_debug_summary(tmp_path: Path):
    d = tmp_path / "nuplan"; d.mkdir()
    (d / "warnings.json").write_text(json.dumps({"validation": {"fallback_assignment_used_rate": 0.4}}), encoding="utf-8")
    (d / "nuplan_lane_projection_debug_summary.json").write_text(json.dumps({"candidate_projection_success_rate": 0.25}), encoding="utf-8")
    metrics = summarize_nuplan(d, max_rows=10)
    assert metrics["projection_debug_summary_available"] is True
    assert metrics["candidate_projection_success_rate"] == 0.25


def test_projection_debug_csv_is_bounded_by_given_rows(tmp_path: Path):
    rows = [{"global_row": i, "timestep": 0, "candidate_index": 0} for i in range(3)]
    write_projection_debug_artifacts(tmp_path, rows, {"sampled_candidate_rows": len(rows)}, True)
    with (tmp_path / "nuplan_lane_projection_debug.csv").open("r", encoding="utf-8", newline="") as f:
        assert len(list(csv.DictReader(f))) == 3


def test_diagnostics_do_not_define_slot_names_locally():
    for path in [Path("tools/nuplan_projection_debug.py"), Path("tools/export_waymo_lane_aware_diagnostics.py")]:
        text = path.read_text(encoding="utf-8")
        assert "SLOT_NAMES =" not in text


def test_stage5d_assignment_only_implementation_unchanged():
    text = Path("tools/build_nuplan_5neighbor_context_dataset.py").read_text(encoding="utf-8")
    assert "assign_stage5d_slots(" in text
    assert "def assign_neighbors_lane_aware" not in text



def test_nuplan_lane_info_topology_and_geometric_enrichment():
    pytest.importorskip("numpy")
    from tools.nuplan_lane_utils import lane_info_from_nuplan_object, enrich_geometric_adjacency, build_lane_topology_debug_summary

    class PathObj:
        def __init__(self, pts):
            self.discrete_path = [type("P", (), {"x": float(x), "y": float(y)})() for x, y in pts]

    class Obj:
        def __init__(self, oid, y, left=None, right=None, incoming=None, outgoing=None):
            self.id = oid
            self.baseline_path = PathObj([(0, y), (10, y)])
            self.left_neighbors = left or []
            self.right_neighbors = right or []
            self.incoming_edges = incoming or []
            self.outgoing_edges = outgoing or []

    lane0 = Obj("lane0", 0.0, incoming=[Obj("in0", -1.0)], outgoing=[Obj("out0", 1.0)])
    lane1 = Obj("lane1", 3.5)
    lanes = {"lane0": lane_info_from_nuplan_object(lane0, "lane"), "lane1": lane_info_from_nuplan_object(lane1, "lane")}
    assert lanes["lane0"].entry_lane_ids == ["in0"]
    assert lanes["lane0"].exit_lane_ids == ["out0"]
    counts = enrich_geometric_adjacency(lanes)
    summary = build_lane_topology_debug_summary(lanes, counts)
    assert summary["lane_info_count"] == 2
    assert summary["left_adjacency_non_empty_count"] >= 1
    assert counts["geometric_left_added"] >= 1


def test_projection_debug_reports_relation_unknown_breakdown(tmp_path: Path):
    rows = [{"lane_relation_used_by_assignment": "unknown", "relation_failure_category": "missing_adjacency", "candidate_projection_success": True, "ego_lane_projection_success": True, "candidate_distance_to_ego": 5.0, "accepted_by_lane_aware": False, "rejection_reason": "wrong_lane"}]
    from tools.nuplan_projection_debug import summarize_projection_debug, write_projection_debug_artifacts
    summary = summarize_projection_debug(rows, [])
    assert summary["lane_relation_unknown_breakdown"]["missing_adjacency"] == 1
    artifacts = write_projection_debug_artifacts(tmp_path, rows, summary, False)
    assert "relation_unknown_csv" in artifacts


def test_no_stage7_specific_assignment_function_introduced():
    for path in Path("tools").glob("stage7*.py"):
        text = path.read_text(encoding="utf-8")
        assert "def assign_neighbors_lane_aware" not in text
        assert "def assign_stage7" not in text
    assert Path("tools/lane_aware_assignment.py").read_text(encoding="utf-8").count("def assign_neighbors_lane_aware") == 1


def test_filtering_mismatch_downgrades_confidence():
    out = diagnose(
        {"filtering_mode": "strict_filter_lane_aware_only", "fallback_assignment_used_rate": 0.0, "candidate_projection_success_rate": 0.9},
        {"filtering_mode": "fallback_preserving", "fallback_assignment_used_rate": 0.419, "candidate_projection_success_rate": 0.7},
        0.2,
    )
    assert out["verdict"] == "inconclusive_due_to_filtering_mismatch"
    assert out["confidence"] == "downgraded"
    assert out["fallback_rate_comparable"] is False


def test_strict_filter_diagnostic_is_loaded_for_fair_comparison(tmp_path: Path):
    waymo = tmp_path / "waymo"; waymo.mkdir()
    nuplan = tmp_path / "nuplan"; nuplan.mkdir()
    (waymo / "waymo_lane_aware_diagnostics.json").write_text(json.dumps({
        "filtering_mode": "strict_filter_lane_aware_only",
        "fallback_assignment_used_rate": 0.0,
        "candidate_projection_success_rate": 0.95,
    }), encoding="utf-8")
    (nuplan / "nuplan_laneaware_strict_filter_summary.json").write_text(json.dumps({
        "original_rows": 10,
        "rows_kept": 8,
        "rows_dropped": 2,
        "kept_row_rate": 0.8,
        "fallback_assignment_used_rate": 0.0,
        "candidate_projection_success_rate": 0.9,
    }), encoding="utf-8")
    w = summarize_waymo(waymo, max_rows=None)
    n = summarize_nuplan(nuplan, max_rows=None)
    assert n["rows_kept"] == 8
    out = diagnose(w, n, 0.2)
    assert out["verdict"] == "comparable_strict_filter_pass"


def test_nuplan_strict_filter_low_keep_rate_verdict():
    out = diagnose(
        {"filtering_mode": "strict_filter_lane_aware_only", "fallback_assignment_used_rate": 0.0},
        {"filtering_mode": "strict_filter_lane_aware_only", "fallback_assignment_used_rate": 0.0, "kept_row_rate": 0.2},
        0.2,
    )
    assert out["verdict"] == "nuplan_strict_filter_low_keep_rate"
