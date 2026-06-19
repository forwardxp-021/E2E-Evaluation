import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools.stage7p_find_lane_change_candidates as finder


def _base_args(ctx: Path, out: Path, **kwargs):
    values = dict(
        context_dir=str(ctx),
        output_dir=str(out),
        top_k=20,
        behavior_events_dir="",
        nuplan_db_root="",
        nuplan_map_root="",
        max_scenarios_scan=50,
        enable_kinematic_scan=False,
        min_lateral_displacement=2.0,
        min_heading_change=0.25,
        min_yaw_rate_proxy=0.05,
    )
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_find_lane_change_candidates_from_metadata_and_events(tmp_path: Path):
    ctx = tmp_path / "ctx"
    events = ctx / "behavior_events_v2"
    out = tmp_path / "out"
    events.mkdir(parents=True)
    (ctx / "merged_metadata.csv").write_text(
        "scenario_id,scenario_type,log_name\n"
        "s0,following,log_a\n"
        "s1,changing_lane,log_b\n"
        "s2,unknown,cut_in_merge_log\n",
        encoding="utf-8",
    )
    (events / "behavior_event_bins_v2.csv").write_text(
        "global_row,task_lane_change\n0,0\n1,0\n2,1\n",
        encoding="utf-8",
    )

    rc = finder.run(_base_args(ctx, out))

    assert rc == 0
    summary = json.loads((out / "lane_change_candidate_summary.json").read_text(encoding="utf-8"))
    assert summary["metadata_rows"] == 3
    assert summary["candidate_rows"] == 2
    assert summary["text_match_candidates"] == 2
    assert summary["behavior_event_candidates"] == 1
    assert summary["kinematic_candidates"] == 0
    assert summary["behavior_events"]["available"] is True
    rows = list(csv.DictReader((out / "lane_change_candidate_metadata.csv").open(encoding="utf-8")))
    assert [row["scenario_id"] for row in rows] == ["s2", "s1"]
    assert "behavior_event_bins_v2:task_lane_change" in rows[0]["match_sources"]
    report = (out / "lane_change_candidate_report.md").read_text(encoding="utf-8")
    assert "text_match_candidates" in report
    assert "behavior_event_candidates" in report
    assert "kinematic_candidates" in report


def test_missing_behavior_event_bins_does_not_crash_and_reports_metadata_only_zero(tmp_path: Path):
    ctx = tmp_path / "ctx"
    out = tmp_path / "out"
    ctx.mkdir()
    (ctx / "merged_metadata.csv").write_text(
        "scenario_id,scenario_type,log_name\n"
        "s0,following,log_a\n",
        encoding="utf-8",
    )

    rc = finder.run(_base_args(ctx, out))

    assert rc == 0
    summary = json.loads((out / "lane_change_candidate_summary.json").read_text(encoding="utf-8"))
    assert summary["candidate_rows"] == 0
    assert summary["behavior_events"]["available"] is False
    assert "behavior_event_bins_v2.csv not found" in summary["behavior_events"]["reason"]
    assert "PDM lane-change capability" in summary["warnings"][0]
    report = (out / "lane_change_candidate_report.md").read_text(encoding="utf-8")
    assert "metadata-only / optional-kinematic candidate discovery" in report
    assert "not that PDM lacks lane-change capability" in report


def test_kinematic_scan_arguments_exist():
    # The real parser is exercised by monkeypatching sys.argv rather than duplicating parser internals.
    old_argv = sys.argv
    try:
        sys.argv = [
            "stage7p_find_lane_change_candidates.py",
            "--context_dir",
            "ctx",
            "--output_dir",
            "out",
            "--nuplan_db_root",
            "db",
            "--nuplan_map_root",
            "maps",
            "--max_scenarios_scan",
            "7",
            "--enable_kinematic_scan",
            "--min_lateral_displacement",
            "1.5",
            "--min_heading_change",
            "0.1",
            "--min_yaw_rate_proxy",
            "0.02",
        ]
        parsed_args = finder.parse_args()
    finally:
        sys.argv = old_argv
    assert parsed_args.nuplan_db_root == "db"
    assert parsed_args.nuplan_map_root == "maps"
    assert parsed_args.max_scenarios_scan == 7
    assert parsed_args.enable_kinematic_scan is True
    assert parsed_args.min_lateral_displacement == 1.5
    assert parsed_args.min_heading_change == 0.1
    assert parsed_args.min_yaw_rate_proxy == 0.02


def test_compute_kinematic_metrics_mock_trajectory():
    metrics = finder.compute_kinematic_metrics(
        [
            {"x": 0.0, "y": 0.0, "yaw": 0.0, "timestamp": 0.0},
            {"x": 5.0, "y": 1.0, "yaw": 0.1, "timestamp": 1.0},
            {"x": 10.0, "y": 3.5, "yaw": 0.3, "timestamp": 2.0},
        ],
        log_name="log_a",
        scenario_token="tok",
        scenario_id="sid",
        scenario_type="mock",
    )
    assert metrics["lateral_displacement_in_start_ego_frame"] == 3.5
    assert metrics["abs_lateral_displacement"] == 3.5
    assert metrics["heading_change_abs"] == 0.3
    assert metrics["yaw_rate_proxy"] == 0.15
    assert metrics["candidate_score"] == 2.0 * 3.5 + 5.0 * 0.3 + 2.0 * 0.15
    assert metrics["max_lateral_speed_proxy"] == 2.5
