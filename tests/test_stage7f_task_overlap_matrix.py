import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.stage7f_task_overlap_matrix import run


def _make_inputs(tmp_path: Path):
    events = tmp_path / "events"; bdd = tmp_path / "bdd"; emb = tmp_path / "emb"; ctx = tmp_path / "ctx"; s7 = tmp_path / "s7"
    for d in [events, bdd, emb, ctx, s7 / "planner_indices"]:
        d.mkdir(parents=True)
    pd.DataFrame({
        "global_row": [0, 1, 2, 3, 4, 5],
        "task_following": ["following", "following", "not_following", "following", "not_following", "not_following"],
        "task_queue_approach": ["queue_approach", "queue_approach", "no_queue_approach", "queue_approach", "no_queue_approach", "no_queue_approach"],
        "task_lead_brake_response": ["no_lead_brake_response", "lead_brake_response", "no_lead_brake_response", "no_lead_brake_response", "lead_brake_response", "no_lead_brake_response"],
        "task_cutin_response": ["no_cutin_response"] * 6,
        "task_yield_conflict": ["no_yield_conflict"] * 6,
    }).to_csv(events / "behavior_event_bins_v2.csv", index=False)
    pd.DataFrame({"global_row": [0, 1, 2, 3, 4, 5]}).to_csv(events / "behavior_event_metrics_v2.csv", index=False)
    pd.DataFrame({
        "global_row": [0, 1, 2, 3, 4, 5],
        "scenario_token": ["s0", "s1", "s2", "s0", "s1", "s2"],
        "planner_name": ["A", "A", "A", "B", "B", "B"],
    }).to_csv(emb / "metadata.csv", index=False)
    np.save(s7 / "planner_indices" / "A.npy", np.array([0, 1, 2]))
    np.save(s7 / "planner_indices" / "B.npy", np.array([3, 4, 5]))
    pd.DataFrame({"task_key": ["task_following"], "bdd_mmd": [0.5], "p_value": [0.01]}).to_csv(bdd / "task_bdd_summary.csv", index=False)
    return events, bdd, emb, ctx, s7


def _args(tmp_path: Path):
    events, bdd, emb, ctx, s7 = _make_inputs(tmp_path)
    class Args: pass
    args = Args()
    args.events_dir = str(events); args.stage7f_task_bdd_dir = str(bdd); args.embedding_dir = str(emb); args.context_dataset_dir = str(ctx)
    args.stage7f_dir = str(s7); args.planner_a = "A"; args.planner_b = "B"
    args.task_keys = "task_following,task_lead_brake_response,task_queue_approach,task_cutin_response,task_yield_conflict"
    args.output_dir = str(tmp_path / "out"); args.overwrite = True
    return args


def test_task_overlap_matrix_counts_jaccard_identical_and_paired_scenarios(tmp_path):
    summary = run(_args(tmp_path))
    assert summary["positive_counts"]["task_following"] == {"all": 3, "planner_a": 2, "planner_b": 1, "paired_scenarios": 1}
    fq = summary["following_vs_queue"]
    assert fq["all"]["overlap_count"] == 3
    assert fq["all"]["jaccard"] == 1.0
    assert fq["identical_positive_rows_all"] is True
    assert fq["identical_positive_rows_planner_a"] is True
    assert fq["identical_positive_rows_planner_b"] is True
    assert fq["identical_paired_scenarios"] is True
    paired = pd.read_csv(tmp_path / "out" / "task_overlap_matrix_paired_scenarios.csv")
    row = paired[(paired.task_i == "task_following") & (paired.task_j == "task_lead_brake_response")].iloc[0]
    assert row.overlap_count == 0
    assert row.union_count == 2
    assert row.jaccard == 0.0
    report = (tmp_path / "out" / "task_overlap_report.md").read_text(encoding="utf-8")
    assert "following-vs-queue diagnostic" in report
    assert "not independent evidence" in report


def test_quick_reference_contains_task_overlap_command():
    text = Path("QUICK_REFERENCE.md").read_text(encoding="utf-8")
    assert "tools/stage7f_task_overlap_matrix.py" in text
    assert "task_overlap_matrix_paired_scenarios.csv" in text


def test_stage7f_diagnostic_does_not_modify_forbidden_metric_modules():
    src = Path("tools/stage7f_task_overlap_matrix.py").read_text(encoding="utf-8")
    assert "mmd_with_stats" not in src
    assert "compute_mmd" not in src
    assert "stage5d_context_core.py" not in src
    assert "lane_aware_assignment.py" not in src
