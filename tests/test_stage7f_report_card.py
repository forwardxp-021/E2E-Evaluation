import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "stage7f_run_report_card.py"


def make_embedding_dir(base: Path, rows):
    d = base / "emb"
    d.mkdir()
    pd.DataFrame(rows).to_csv(d / "metadata.csv", index=False)
    np.save(d / "embedding.npy", np.arange(len(rows) * 3, dtype=np.float32).reshape(len(rows), 3))
    (d / "embedding_manifest.json").write_text(json.dumps({"context_dataset_dir": str(base / "ctx")}), encoding="utf-8")
    return d


def run_tool(emb, out, mode="full"):
    return subprocess.run(
        [sys.executable, str(TOOL), "--embedding_dir", str(emb), "--output_dir", str(out), "--mode", mode, "--overwrite"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def test_stage7f_full_requires_complete_scenario_planner_alignment(tmp_path):
    emb = make_embedding_dir(tmp_path, [
        {"scenario_token": "s0", "planner_name": "p0", "fallback_used": 0},
        {"scenario_token": "s0", "planner_name": "p1", "fallback_used": 0},
        {"scenario_token": "s1", "planner_name": "p0", "fallback_used": 0},
    ])
    proc = run_tool(emb, tmp_path / "out", "full")
    assert proc.returncode != 0
    assert "requires complete scenario" in (proc.stderr + proc.stdout)


def test_stage7f_strict_sensitivity_reports_incomplete_alignment_and_mode(tmp_path):
    emb = make_embedding_dir(tmp_path, [
        {"scenario_token": "s0", "planner_name": "p0", "fallback_used": 0},
        {"scenario_token": "s0", "planner_name": "p1", "fallback_used": 0},
        {"scenario_token": "s1", "planner_name": "p0", "fallback_used": 0},
    ])
    proc = run_tool(emb, tmp_path / "out", "strict_sensitivity")
    assert proc.returncode == 0, proc.stderr
    summary = json.loads((tmp_path / "out" / "stage7f_summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "strict_sensitivity"
    assert summary["alignment"]["all_scenarios_have_all_planners"] is False
    assert summary["alignment"]["scenarios_missing_any_planner"] == 1
    report = (tmp_path / "out" / "stage7f_report.md").read_text(encoding="utf-8")
    assert "not the main planner-evaluation dataset" in report


def test_stage7f_full_records_fallback_preserving_mode(tmp_path):
    emb = make_embedding_dir(tmp_path, [
        {"scenario_token": "s0", "planner_name": "p0", "fallback_used": 0},
        {"scenario_token": "s0", "planner_name": "p1", "fallback_used": 1},
        {"scenario_token": "s1", "planner_name": "p0", "fallback_used": 0},
        {"scenario_token": "s1", "planner_name": "p1", "fallback_used": 0},
    ])
    proc = run_tool(emb, tmp_path / "out", "full")
    assert proc.returncode == 0, proc.stderr
    summary = json.loads((tmp_path / "out" / "stage7f_summary.json").read_text(encoding="utf-8"))
    assert summary["fallback"]["fallback_preserving_status"] is True
    assert summary["fallback"]["fallback_rate"] == 0.25


def test_stage7f_wrapper_delegates_stage6_and_does_not_touch_stage5d_core_files():
    src = TOOL.read_text(encoding="utf-8")
    assert "tools/stage6_compare_unpaired_style.py" in src
    assert "tools/stage6_generate_report_card.py" in src
    assert "def compute_mmd" not in src
    assert "def mmd" not in src
    assert "lane_aware_assignment.py" not in src
    assert "stage5d_context_core.py" not in src


def test_stage7f_auto_loads_warnings_json_from_context_dataset_dir(tmp_path):
    emb = make_embedding_dir(tmp_path, [
        {"scenario_token": "s0", "planner_name": "p0", "fallback_used": 0},
        {"scenario_token": "s0", "planner_name": "p1", "fallback_used": 0},
        {"scenario_token": "s1", "planner_name": "p0", "fallback_used": 0},
        {"scenario_token": "s1", "planner_name": "p1", "fallback_used": 0},
    ])
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    (ctx / "warnings.json").write_text(json.dumps({
        "validation": {
            "fallback_assignment_used_rate": 0.75,
            "lane_assignment_available_rate": 0.25,
            "map_name_resolved_rate": 1.0,
            "map_query_success": True,
            "lane_info_count": 12,
        }
    }), encoding="utf-8")

    proc = run_tool(emb, tmp_path / "out", "full")

    assert proc.returncode == 0, proc.stderr
    summary = json.loads((tmp_path / "out" / "stage7f_summary.json").read_text(encoding="utf-8"))
    assert summary["context_diagnostics_source"] == str(ctx / "warnings.json")
    assert summary["fallback"]["fallback_assignment_used_rate"] == 0.75
    assert summary["fallback"]["fallback_rate"] == 0.75
    assert summary["fallback"]["lane_assignment_available_rate"] == 0.25
    assert summary["fallback"]["map_query_success"] is True
    report = (tmp_path / "out" / "stage7f_report.md").read_text(encoding="utf-8")
    assert f"context diagnostics source: `{ctx / 'warnings.json'}`" in report
    assert "fallback rate: `0.75`" in report
