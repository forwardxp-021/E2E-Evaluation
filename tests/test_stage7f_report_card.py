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


def _make_pair(pair_dir: Path, mmd2: float, with_optional: bool = False):
    pair_dir.mkdir(parents=True)
    (pair_dir / "bdd_summary.json").write_text(json.dumps({
        "mmd2": mmd2, "ci95_low": mmd2 - 0.1, "ci95_high": mmd2 + 0.1,
        "p_value": 0.2, "n_A": 20, "n_B": 20, "embedding_dim": 64,
    }), encoding="utf-8")
    (pair_dir / "style_report_card.md").write_text("# Style Report Card\n", encoding="utf-8")
    (pair_dir / "stage6_warnings.json").write_text(json.dumps({"warnings": ["w1"]}), encoding="utf-8")
    if with_optional:
        pd.DataFrame([
            {"category": "speed", "delta": 0.1, "cohen_d": 0.2, "p_value": 0.4},
            {"category": "spacing", "delta": -0.3, "cohen_d": -0.6, "p_value": 0.1},
        ]).to_csv(pair_dir / "category_delta.csv", index=False)
        pd.DataFrame([
            {"feature": "mean_speed", "delta_normalized": -0.5, "cohen_d": -0.5, "permutation_p_value": 0.3},
        ]).to_csv(pair_dir / "feature_delta.csv", index=False)


def test_stage7f_collect_pairwise_summary_minimal_missing_optional_and_ranking(tmp_path):
    from tools.stage7f_collect_pairwise_summary import collect_pairwise_summary

    stage7f = tmp_path / "stage7f"
    (stage7f / "stage6_pairwise").mkdir(parents=True)
    (stage7f / "stage7f_summary.json").write_text(json.dumps({
        "mode": "full",
        "row_semantics": "scenario × planner-controlled nuPlan ego rollout",
        "alignment": {"num_scenarios": 5, "num_planners": 3, "total_rows": 15},
        "fallback": {"fallback_preserving_status": True, "fallback_rate": 0.5191275167785235},
    }), encoding="utf-8")
    _make_pair(stage7f / "stage6_pairwise" / "planner_a_vs_planner_b", 0.2, with_optional=False)
    _make_pair(stage7f / "stage6_pairwise" / "planner_a_vs_planner_c", 0.7, with_optional=True)

    result = collect_pairwise_summary(stage7f, overwrite=True)

    assert Path(result["csv"]).exists()
    rows = json.loads((stage7f / "stage7f_pairwise_summary.json").read_text(encoding="utf-8"))["rows"]
    assert [r["pair_name"] for r in rows] == ["planner_a_vs_planner_c", "planner_a_vs_planner_b"]
    assert rows[0]["bdd_rank_desc"] == 1
    assert rows[1]["top_category_1"] is None
    assert rows[1]["has_scenario_slice_summary"] is False
    md = (stage7f / "stage7f_pairwise_summary.md").read_text(encoding="utf-8")
    assert "planner_a_vs_planner_c" in md
    assert "n_A=20, n_B=20" in md
    assert "0.5191275167785235" in md
    assert "n_A=n_B=5" not in md
    assert "41.9%" not in md


def test_stage7f_runner_creates_pairwise_summary_after_pairwise(monkeypatch, tmp_path):
    import tools.stage7f_run_report_card as runner

    emb = make_embedding_dir(tmp_path, [
        {"scenario_token": "s0", "planner_name": "p0", "fallback_used": 0},
        {"scenario_token": "s0", "planner_name": "p1", "fallback_used": 0},
        {"scenario_token": "s1", "planner_name": "p0", "fallback_used": 0},
        {"scenario_token": "s1", "planner_name": "p1", "fallback_used": 0},
    ])
    ctx = tmp_path / "ctx"
    ctx.mkdir(exist_ok=True)

    def fake_pairwise(embedding_dir, context_dir, output_dir, idx_paths, args):
        pair = output_dir / "stage6_pairwise" / "p0_vs_p1"
        _make_pair(pair, 0.4, with_optional=True)
        return [{"planner_a": "p0", "planner_b": "p1", "output_dir": str(pair)}]

    monkeypatch.setattr(runner, "run_stage6_pairwise", fake_pairwise)
    args = runner.parse_args.__globals__["argparse"].Namespace(
        embedding_dir=str(emb), output_dir=str(tmp_path / "out"), context_dataset_dir=str(ctx),
        context_diagnostics_json=None, mode="full", strict_filter_min_laneaware_ratio=0.8,
        run_stage6_pairwise=True, num_bootstrap=1, num_permutation=1, min_slice_size=2, top_k=2, overwrite=True,
    )
    runner.run(args)

    assert (tmp_path / "out" / "stage7f_pairwise_summary.csv").exists()
    assert (tmp_path / "out" / "stage7f_pairwise_summary.json").exists()
    assert (tmp_path / "out" / "stage7f_pairwise_summary.md").exists()
    assert "stage7f_pairwise_summary.md" in (tmp_path / "out" / "stage7f_report.md").read_text(encoding="utf-8")


def test_stage7f_collector_does_not_import_stage6_metrics_or_forbidden_stage5_modules():
    src = (ROOT / "tools" / "stage7f_collect_pairwise_summary.py").read_text(encoding="utf-8")
    assert "stage6_compare_unpaired_style" not in src
    assert "compute_mmd" not in src
    assert "lane_aware_assignment.py" not in src
    assert "stage5d_context_core.py" not in src
