import csv
import json
from pathlib import Path

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
