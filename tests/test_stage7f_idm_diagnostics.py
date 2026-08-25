import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.stage7f_aggressive_conservative_paired_delta import align_pairs
from tools.stage7f_run_task_conditioned_bdd import build_commands
from tools.stage7f_idm_diagnostic_common import idm_parameter_markdown


def test_task_wrapper_resolves_indices_and_reuses_stage6c(tmp_path):
    emb = tmp_path / "emb"; ctx = tmp_path / "ctx"; s7 = tmp_path / "s7"; out = tmp_path / "out"
    (emb).mkdir(); (ctx).mkdir(); (s7 / "planner_indices").mkdir(parents=True)
    (emb / "embedding_manifest.json").write_text("{}")
    (ctx / "shard_manifest.json").write_text("{}")
    (ctx / "feature_schema.json").write_text("{}")
    np.save(s7 / "planner_indices" / "A.npy", np.array([0, 1]))
    np.save(s7 / "planner_indices" / "B.npy", np.array([2, 3]))
    class A: pass
    a = A(); a.embedding_dir=str(emb); a.context_dataset_dir=str(ctx); a.stage7f_dir=str(s7); a.planner_a="A"; a.planner_b="B"; a.output_dir=str(out); a.task_keys="task_following"; a.min_bin_size=2; a.num_bootstrap=3; a.num_permutation=4; a.overwrite=True; a.overwrite_events=False
    build, report, resolved = build_commands(a, tmp_path / "events")
    assert "tools/stage6c_build_behavior_events_v2.py" in build
    assert "tools/stage6c_task_conditioned_bdd_report.py" in report
    assert "--min_bin_size" in report and report[report.index("--min_bin_size") + 1] == "2"
    assert resolved["a_indices_path"].endswith("A.npy")


def test_paired_delta_align_fails_on_duplicate_pairs():
    meta = pd.DataFrame({"scenario_token": ["s1", "s1"], "planner_name": ["A", "A"]})
    with pytest.raises(ValueError, match="Duplicate scenario-planner"):
        align_pairs(meta, "A", "B")


def test_paired_delta_align_requires_paired_scenarios():
    meta = pd.DataFrame({"scenario_token": ["s1", "s2"], "planner_name": ["A", "B"]})
    with pytest.raises(ValueError, match="No paired scenarios"):
        align_pairs(meta, "A", "B")


def make_dataset(root: Path):
    emb = root / "emb"; ctx = root / "ctx"; s7 = root / "s7"; emb.mkdir(); ctx.mkdir(); s7.mkdir()
    meta = pd.DataFrame({"scenario_token": ["s1", "s1"], "planner_name": ["idm_longitudinal_aggressive", "idm_longitudinal_conservative"], "fallback_used": [0, 1], "laneaware_available": [1, 1]})
    meta.to_csv(emb / "metadata.csv", index=False)
    np.save(emb / "embedding.npy", np.array([[1.0, 0.0], [0.0, 1.0]]))
    ego = np.zeros((2, 4, 8), dtype=float)
    ego[0, :, 5] = [2, 3, 4, 5]; ego[1, :, 5] = [1, 2, 3, 4]
    ego[0, :, 6] = [0.1, 0.2, 0.3, 0.4]; ego[1, :, 6] = [0.0, 0.1, 0.1, 0.2]
    ego[:, :, 7] = 0.1
    np.save(ctx / "ego_seq.npy", ego)
    nei = np.zeros((2, 1, 4, 12), dtype=float); nei[:, 0, :, 0] = 1; nei[0,0,:,5]=[5,6,7,8]; nei[1,0,:,5]=[6,7,8,9]; nei[0,0,:,10]=[1,2,3,4]; nei[1,0,:,10]=[2,3,4,5]
    np.save(ctx / "neighbor_seq.npy", nei)
    return emb, ctx, s7


def test_paired_delta_script_computes_distances_and_report(tmp_path):
    emb, ctx, s7 = make_dataset(tmp_path); out = tmp_path / "out"
    subprocess.run(["python", "tools/stage7f_aggressive_conservative_paired_delta.py", "--embedding_dir", str(emb), "--context_dataset_dir", str(ctx), "--stage7f_dir", str(s7), "--planner_a", "idm_longitudinal_aggressive", "--planner_b", "idm_longitudinal_conservative", "--output_dir", str(out), "--overwrite"], check=True)
    df = pd.read_csv(out / "paired_delta_by_scenario.csv")
    assert df["embedding_l2_distance"].iloc[0] > 0
    assert "delta_mean_speed" in df.columns
    text = (out / "paired_delta_report.md").read_text()
    assert "target_velocity = 12.0 m/s" in text
    assert "target_velocity: +4.0 m/s, +50.0%" in text


def test_quick_reference_contains_stage7f_commands():
    text = Path("QUICK_REFERENCE.md").read_text(encoding="utf-8")
    assert "tools/stage7f_run_task_conditioned_bdd.py" in text
    assert "tools/stage7f_aggressive_conservative_paired_delta.py" in text
