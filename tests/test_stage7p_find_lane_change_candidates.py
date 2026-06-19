import csv
import json
import sqlite3
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
        scan_db_scenario_tags=False,
        max_db_files=0,
        max_candidates_per_type=0,
        max_per_log=2,
        write_stage7c_context_dir=False,
        prefer_exact_changing_lane=False,
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


def _make_mock_nuplan_db(path: Path, blob_tokens: bool = False, duplicate_types: bool = False, log_name: str = "mock_log") -> None:
    conn = sqlite3.connect(path)
    with conn:
        conn.execute("CREATE TABLE scenario_tag(token BLOB, lidar_pc_token BLOB, type TEXT, agent_track_token BLOB)")
        conn.execute("CREATE TABLE lidar_pc(token BLOB, scene_token BLOB, ego_pose_token BLOB)")
        conn.execute("CREATE TABLE log(logfile TEXT, token BLOB)")
        st_token = b"\x01\x02tag" if blob_tokens else "tag_text"
        lidar_token = b"\x03\x04lidar" if blob_tokens else "lidar_text"
        scene_token = b"\x05scene" if blob_tokens else "scene_text"
        ego_pose_token = b"\x06ego" if blob_tokens else "ego_text"
        conn.execute("INSERT INTO log(logfile, token) VALUES (?, ?)", (log_name, b"log"))
        conn.execute("INSERT INTO lidar_pc(token, scene_token, ego_pose_token) VALUES (?, ?, ?)", (lidar_token, scene_token, ego_pose_token))
        conn.execute(
            "INSERT INTO scenario_tag(token, lidar_pc_token, type, agent_track_token) VALUES (?, ?, ?, ?)",
            (st_token, lidar_token, "changing_lane", b"agent"),
        )
        if duplicate_types:
            conn.execute(
                "INSERT INTO scenario_tag(token, lidar_pc_token, type, agent_track_token) VALUES (?, ?, ?, ?)",
                ("tag_right", lidar_token, "changing_lane_to_right", b"agent"),
            )
        conn.execute(
            "INSERT INTO scenario_tag(token, lidar_pc_token, type, agent_track_token) VALUES (?, ?, ?, ?)",
            ("ignore_text", "missing_lidar", "following_lane", b"agent"),
        )


def test_db_scenario_tag_scan_discovers_changing_lane_blob_and_writes_stage7c_context(tmp_path: Path):
    ctx = tmp_path / "ctx"
    db_root = tmp_path / "dbs"
    out = tmp_path / "out"
    ctx.mkdir()
    db_root.mkdir()
    (ctx / "merged_metadata.csv").write_text("scenario_id,scenario_type,log_name\ns0,following,log_a\n", encoding="utf-8")
    _make_mock_nuplan_db(db_root / "mini.db", blob_tokens=True)

    rc = finder.run(
        _base_args(
            ctx,
            out,
            nuplan_db_root=str(db_root),
            scan_db_scenario_tags=True,
            write_stage7c_context_dir=True,
        )
    )

    assert rc == 0
    summary = json.loads((out / "lane_change_candidate_summary.json").read_text(encoding="utf-8"))
    assert summary["metadata_text_candidate_rows"] == 0
    assert summary["db_scenario_tag_candidate_rows"] == 1
    assert summary["final_candidate_rows"] == 1
    assert summary["scenario_type_counts"] == {"changing_lane": 1}
    rows = list(csv.DictReader((out / "lane_change_candidate_metadata.csv").open(encoding="utf-8")))
    assert rows[0]["source"] == "db_scenario_tag"
    assert rows[0]["scenario_tag_token"] == b"\x01\x02tag".hex()
    assert rows[0]["lidar_pc_token"] == b"\x03\x04lidar".hex()
    assert rows[0]["scenario_token"] == b"\x03\x04lidar".hex()
    assert rows[0]["scene_token"] == b"\x03\x04lidar".hex()
    assert rows[0]["db_scene_token"] == b"\x05scene".hex()
    assert rows[0]["log_name"] == "mock_log"
    stage7c_path = out / "stage7c_candidate_context" / "merged_metadata.csv"
    assert stage7c_path.is_file()
    stage7c_rows = list(csv.DictReader(stage7c_path.open(encoding="utf-8")))
    assert stage7c_rows[0]["scenario_token"] == b"\x03\x04lidar".hex()
    assert stage7c_rows[0]["scene_token"] == b"\x03\x04lidar".hex()
    assert stage7c_rows[0]["db_scene_token"] == b"\x05scene".hex()
    assert stage7c_rows[0]["log_name"] == "mock_log"
    assert stage7c_rows[0]["source"] == "db_scenario_tag"
    assert summary["raw_db_scenario_tag_rows"] == 2
    assert summary["unique_scenario_token_rows"] == 1
    assert summary["selected_rows"] == 1
    assert summary["selected_log_counts"] == {"mock_log": 1}
    report = (out / "lane_change_candidate_report.md").read_text(encoding="utf-8")
    assert "metadata_text candidates: `0`" in report
    assert "db_scenario_tag candidates: `1`" in report
    assert "Stage7B merged subset is not lane-change-rich" in report


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
            "--scan_db_scenario_tags",
            "--max_db_files",
            "2",
            "--max_candidates_per_type",
            "3",
            "--max_per_log",
            "2",
            "--write_stage7c_context_dir",
            "--prefer_exact_changing_lane",
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
    assert parsed_args.scan_db_scenario_tags is True
    assert parsed_args.max_db_files == 2
    assert parsed_args.max_candidates_per_type == 3
    assert parsed_args.max_per_log == 2
    assert parsed_args.write_stage7c_context_dir is True
    assert parsed_args.prefer_exact_changing_lane is True
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


def test_db_scenario_tag_scan_deduplicates_lidar_pc_token_and_prefers_strict_type(tmp_path: Path):
    ctx = tmp_path / "ctx"
    db_root = tmp_path / "dbs"
    out = tmp_path / "out"
    ctx.mkdir()
    db_root.mkdir()
    (ctx / "merged_metadata.csv").write_text("scenario_id,scenario_type,log_name\ns0,following,log_a\n", encoding="utf-8")
    _make_mock_nuplan_db(db_root / "mini.db", duplicate_types=True)

    rc = finder.run(_base_args(ctx, out, nuplan_db_root=str(db_root), scan_db_scenario_tags=True, write_stage7c_context_dir=True))

    assert rc == 0
    summary = json.loads((out / "lane_change_candidate_summary.json").read_text(encoding="utf-8"))
    assert summary["raw_db_scenario_tag_rows"] == 3
    assert summary["unique_scenario_token_rows"] == 1
    assert summary["duplicate_scenario_token_count_removed"] == 1
    rows = list(csv.DictReader((out / "stage7c_candidate_context" / "merged_metadata.csv").open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["scenario_type"] == "changing_lane_to_right"
    assert rows[0]["scenario_token"] == rows[0]["scene_token"]
    assert rows[0]["db_scene_token"] == "scene_text"


def test_db_scenario_tag_scan_max_per_log_limits_selected_rows(tmp_path: Path):
    ctx = tmp_path / "ctx"
    db_path = tmp_path / "mini.db"
    out = tmp_path / "out"
    ctx.mkdir()
    (ctx / "merged_metadata.csv").write_text("scenario_id,scenario_type,log_name\ns0,following,log_a\n", encoding="utf-8")
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute("CREATE TABLE scenario_tag(token TEXT, lidar_pc_token TEXT, type TEXT, agent_track_token TEXT)")
        conn.execute("CREATE TABLE lidar_pc(token TEXT, scene_token TEXT, ego_pose_token TEXT)")
        conn.execute("CREATE TABLE log(logfile TEXT, token TEXT)")
        conn.execute("INSERT INTO log(logfile, token) VALUES (?, ?)", ("one_log", "log"))
        for idx in range(3):
            lidar = f"lidar_{idx}"
            conn.execute("INSERT INTO lidar_pc(token, scene_token, ego_pose_token) VALUES (?, ?, ?)", (lidar, f"scene_{idx}", f"ego_{idx}"))
            conn.execute("INSERT INTO scenario_tag(token, lidar_pc_token, type, agent_track_token) VALUES (?, ?, ?, ?)", (f"tag_{idx}", lidar, "changing_lane_to_right", "agent"))

    rc = finder.run(_base_args(ctx, out, nuplan_db_root=str(db_path), scan_db_scenario_tags=True, write_stage7c_context_dir=True, max_per_log=2))

    assert rc == 0
    rows = list(csv.DictReader((out / "stage7c_candidate_context" / "merged_metadata.csv").open(encoding="utf-8")))
    summary = json.loads((out / "lane_change_candidate_summary.json").read_text(encoding="utf-8"))
    assert len(rows) == 2
    assert summary["selected_log_counts"] == {"one_log": 2}


def test_prefer_exact_changing_lane_prioritizes_strict_then_fallback_and_respects_max_per_log(tmp_path: Path):
    ctx = tmp_path / "ctx"
    db_path = tmp_path / "mini.db"
    out = tmp_path / "out"
    ctx.mkdir()
    (ctx / "merged_metadata.csv").write_text("scenario_id,scenario_type,log_name\ns0,following,log_a\n", encoding="utf-8")
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute("CREATE TABLE scenario_tag(token TEXT, lidar_pc_token TEXT, type TEXT, agent_track_token TEXT)")
        conn.execute("CREATE TABLE lidar_pc(token TEXT, scene_token TEXT, ego_pose_token TEXT)")
        conn.execute("CREATE TABLE log(logfile TEXT, token TEXT)")
        conn.execute("INSERT INTO log(logfile, token) VALUES (?, ?)", ("one_log", "log"))
        scenario_types = ["high_lateral_acceleration", "cut_in", "merge", "changing_lane", "changing_lane_to_left"]
        for idx, scenario_type in enumerate(scenario_types):
            lidar = f"lidar_{idx}"
            conn.execute("INSERT INTO lidar_pc(token, scene_token, ego_pose_token) VALUES (?, ?, ?)", (lidar, f"scene_{idx}", f"ego_{idx}"))
            conn.execute("INSERT INTO scenario_tag(token, lidar_pc_token, type, agent_track_token) VALUES (?, ?, ?, ?)", (f"tag_{idx}", lidar, scenario_type, "agent"))

    rc = finder.run(
        _base_args(
            ctx,
            out,
            nuplan_db_root=str(db_path),
            scan_db_scenario_tags=True,
            write_stage7c_context_dir=True,
            prefer_exact_changing_lane=True,
            max_per_log=2,
            top_k=4,
        )
    )

    assert rc == 0
    rows = list(csv.DictReader((out / "stage7c_candidate_context" / "merged_metadata.csv").open(encoding="utf-8")))
    summary = json.loads((out / "lane_change_candidate_summary.json").read_text(encoding="utf-8"))
    assert [row["scenario_type"] for row in rows] == ["changing_lane_to_left", "changing_lane"]
    assert all(row["scenario_token"] == row["scene_token"] for row in rows)
    assert summary["selected_log_counts"] == {"one_log": 2}
    assert summary["strict_changing_lane_candidate_rows"] == 2
    assert summary["selected_strict_changing_lane_rows"] == 2
    assert summary["selected_scenario_type_counts"] == {"changing_lane": 1, "changing_lane_to_left": 1}
