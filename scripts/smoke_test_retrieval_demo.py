"""Smoke test for tools/embedding_retrieval_demo.py.

Validates that:
  1) retrieval returns correct shapes for global and within-source modes
  2) plotting functions run without error (uses matplotlib Agg backend)
  3) run_demo() writes expected output files (summary.json, retrieval_table.csv,
     traj_overlay.png, timeseries.png) to a temp directory

Run with:
    python scripts/smoke_test_retrieval_demo.py
or from repo root:
    python -m scripts.smoke_test_retrieval_demo
"""

import os
import sys
import json
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")

from tools.embedding_retrieval_demo import (
    _make_synthetic_data,
    retrieve,
    compute_traj_signals,
    _align_traj,
    plot_traj_overlay,
    plot_timeseries,
    run_demo,
    parse_args,
)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_global_retrieval_shapes():
    """retrieve() in global mode returns a DataFrame with correct columns."""
    embeddings, meta, traj, front, split = _make_synthetic_data(n_items=60, T=20)
    result = retrieve(
        query_idx=0,
        embeddings=embeddings,
        meta=meta,
        split=split,
        split_filter="test",
        mode="global",
        distance="euclidean",
        topk=5,
        exclude_same_scenario=False,
        exclude_same_source=False,
    )
    assert not result.empty, "global retrieval result should not be empty"
    required_cols = {"index", "scenario_id", "start", "window_len", "front_id", "distance", "excluded"}
    assert required_cols.issubset(result.columns), f"Missing columns: {required_cols - set(result.columns)}"
    non_excl = result[~result["excluded"]]
    assert len(non_excl) <= 5, f"Expected ≤5 non-excluded results, got {len(non_excl)}"
    # Distances should be non-negative and sorted ascending
    assert (non_excl["distance"].values >= 0).all(), "Distances must be non-negative"
    assert list(non_excl["distance"]) == sorted(non_excl["distance"].tolist()), "Results not sorted by distance"
    print("  [PASS] test_global_retrieval_shapes")


def test_within_source_retrieval():
    """within-source retrieval finds only same-meta-key rows (excluding query)."""
    embeddings, meta, traj, front, split = _make_synthetic_data(n_items=60, T=20, n_groups=20)
    # Query at index 0: meta key = (scenario_000, 0, 20, front_000)
    # Same meta key: indices 0, 20, 40 (all i%20==0)
    result = retrieve(
        query_idx=0,
        embeddings=embeddings,
        meta=meta,
        split=split,
        split_filter="test",
        mode="within-source",
        distance="cosine",
        topk=10,
        exclude_same_scenario=False,
        exclude_same_source=False,
    )
    # Should find exactly 2 candidates (indices 20 and 40, since query itself is excluded)
    assert len(result) == 2, f"Expected 2 within-source candidates, got {len(result)}"
    assert 0 not in result["index"].values, "Query itself must not appear in results"
    print("  [PASS] test_within_source_retrieval")


def test_cosine_distance_ordering():
    """Cosine distance retrieval should rank the most-similar item first."""
    rng = np.random.default_rng(7)
    N, D = 30, 8
    embeddings = rng.standard_normal((N, D)).astype(np.float32)
    # Make item 5 almost identical to query (index 0)
    embeddings[5] = embeddings[0] + rng.standard_normal(D).astype(np.float32) * 1e-4
    meta = np.empty(N, dtype=object)
    for i in range(N):
        meta[i] = (f"sc_{i}", i, 10, f"f_{i}")
    split = np.array(["test"] * N, dtype=object)

    result = retrieve(
        query_idx=0,
        embeddings=embeddings,
        meta=meta,
        split=split,
        split_filter="test",
        mode="global",
        distance="cosine",
        topk=3,
        exclude_same_scenario=False,
        exclude_same_source=False,
    )
    top1_idx = int(result[~result["excluded"]].iloc[0]["index"])
    assert top1_idx == 5, f"Expected idx=5 as nearest, got {top1_idx}"
    print("  [PASS] test_cosine_distance_ordering")


def test_exclude_same_scenario():
    """exclude_same_scenario flag correctly marks same-scenario rows as excluded."""
    embeddings, meta, traj, front, split = _make_synthetic_data(n_items=60, T=20, n_groups=20)
    # Use "train" which has the majority of rows in synthetic data
    result = retrieve(
        query_idx=1,  # query index 1 -> scenario_001, which has rows 1, 21, 41
        embeddings=embeddings,
        meta=meta,
        split=split,
        split_filter="train",
        mode="global",
        distance="euclidean",
        topk=5,
        exclude_same_scenario=True,
        exclude_same_source=False,
    )
    assert not result.empty, "Result should not be empty when using 'train' split"
    q_scenario = str(meta[1][0])
    excl_rows = result[result["excluded"]]
    for _, row in excl_rows.iterrows():
        assert str(row["scenario_id"]) == q_scenario, (
            f"Only same-scenario rows should be excluded, found scenario_id={row['scenario_id']}"
        )
    print("  [PASS] test_exclude_same_scenario")


def test_traj_signals_shape():
    """compute_traj_signals returns correct keys and shapes."""
    T = 25
    traj = np.zeros((T, 4), dtype=np.float32)
    traj[:, 2] = 10.0  # constant vx
    sigs = compute_traj_signals(traj, dt=0.1)
    for key in ("speed", "accel", "jerk", "curvature_proxy"):
        assert key in sigs
        assert len(sigs[key]) == T, f"Signal {key}: expected length {T}, got {len(sigs[key])}"
    # Constant speed → accel ≈ 0
    assert np.allclose(sigs["accel"], 0.0, atol=1e-5), "Constant-speed accel should be ~0"
    print("  [PASS] test_traj_signals_shape")


def test_align_traj_origin():
    """_align_traj places query first point at origin and aligns heading to +x."""
    T = 10
    # Trajectory moving diagonally (45 deg)
    ref = np.zeros((T, 4), dtype=float)
    for t in range(T):
        ref[t] = [t * 1.0, t * 1.0, 1.0, 1.0]  # vx=vy=1 → 45° heading

    aligned_ref = _align_traj(ref, ref)
    assert abs(aligned_ref[0, 0]) < 1e-9, "Aligned[0, 0] should be 0"
    assert abs(aligned_ref[0, 1]) < 1e-9, "Aligned[0, 1] should be 0"
    # After aligning 45° heading to +x, motion should be approximately along +x
    assert aligned_ref[-1, 0] > 0, "Aligned trajectory should move in +x"
    assert abs(aligned_ref[-1, 1]) < 1e-6, "Aligned y displacement should be ~0 for straight diagonal"
    print("  [PASS] test_align_traj_origin")


def test_plot_functions_no_error():
    """plot_traj_overlay and plot_timeseries run without raising exceptions."""
    embeddings, meta, traj, front, split = _make_synthetic_data(n_items=30, T=15)
    result = retrieve(
        query_idx=0,
        embeddings=embeddings,
        meta=meta,
        split=split,
        split_filter="train",
        mode="global",
        distance="euclidean",
        topk=3,
        exclude_same_scenario=False,
        exclude_same_source=False,
    )
    with tempfile.TemporaryDirectory() as tmp:
        plot_traj_overlay(
            query_idx=0, result_df=result, traj=traj, front=front, meta=meta,
            out_path=os.path.join(tmp, "traj.png"), topk=3, quiet=True,
        )
        plot_timeseries(
            query_idx=0, result_df=result, traj=traj, meta=meta,
            out_path=os.path.join(tmp, "ts.png"), topk=3, dt=0.1, quiet=True,
        )
        assert os.path.exists(os.path.join(tmp, "traj.png")), "traj_overlay not created"
        assert os.path.exists(os.path.join(tmp, "ts.png")), "timeseries not created"
    print("  [PASS] test_plot_functions_no_error")


def test_run_demo_output_files():
    """run_demo() writes all expected output files."""
    embeddings, meta, traj, front, split = _make_synthetic_data(n_items=60, T=20)

    with tempfile.TemporaryDirectory() as tmp:
        # Save synthetic arrays to files
        emb_path = os.path.join(tmp, "feat_style.npy")
        meta_path = os.path.join(tmp, "meta.npy")
        traj_path = os.path.join(tmp, "traj.npy")
        front_path = os.path.join(tmp, "front.npy")
        split_path = os.path.join(tmp, "split.npy")
        np.save(emb_path, embeddings)
        np.save(meta_path, meta)
        np.save(traj_path, traj)
        np.save(front_path, front)
        np.save(split_path, split)

        out_dir = os.path.join(tmp, "outputs")
        args = parse_args([
            "--emb_path", emb_path,
            "--meta_path", meta_path,
            "--traj_path", traj_path,
            "--front_path", front_path,
            "--split_path", split_path,
            "--split_filter", "train",
            "--query_index", "0",
            "--topk", "3",
            "--mode", "global",
            "--output_dir", out_dir,
            "--run_id", "test_run",
            "--quiet",
        ])
        result_dir = run_demo(args)

        result_path = Path(result_dir)
        assert (result_path / "summary.json").exists(), "summary.json not created"
        assert (result_path / "retrieval_table.csv").exists(), "retrieval_table.csv not created"
        assert (result_path / "traj_overlay.png").exists(), "traj_overlay.png not created"
        assert (result_path / "timeseries.png").exists(), "timeseries.png not created"

        summary = json.loads((result_path / "summary.json").read_text())
        assert summary["mode"] == "global"
        assert summary["topk"] == 3
        assert "query_meta" in summary
    print("  [PASS] test_run_demo_output_files")


def test_query_selection_by_scenario_id():
    """--query_scenario_id finds the correct row."""
    embeddings, meta, traj, front, split = _make_synthetic_data(n_items=30, T=10)
    # meta[5] = (scenario_005, ...)
    target_sc = str(meta[5][0])

    with tempfile.TemporaryDirectory() as tmp:
        emb_path = os.path.join(tmp, "emb.npy")
        meta_path = os.path.join(tmp, "meta.npy")
        traj_path = os.path.join(tmp, "traj.npy")
        front_path = os.path.join(tmp, "front.npy")
        split_path = os.path.join(tmp, "split.npy")
        np.save(emb_path, embeddings)
        np.save(meta_path, meta)
        np.save(traj_path, traj)
        np.save(front_path, front)
        np.save(split_path, split)

        out_dir = os.path.join(tmp, "outputs")
        args = parse_args([
            "--emb_path", emb_path,
            "--meta_path", meta_path,
            "--traj_path", traj_path,
            "--front_path", front_path,
            "--split_path", split_path,
            "--split_filter", "train",
            "--query_scenario_id", target_sc,
            "--topk", "2",
            "--mode", "global",
            "--output_dir", out_dir,
            "--run_id", "test_sc",
            "--quiet",
        ])
        result_dir = run_demo(args)
        summary = json.loads((Path(result_dir) / "summary.json").read_text())
        assert summary["query_meta"]["scenario_id"] == target_sc, (
            f"Expected scenario_id={target_sc}, got {summary['query_meta']['scenario_id']}"
        )
    print("  [PASS] test_query_selection_by_scenario_id")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Running smoke/unit tests for tools/embedding_retrieval_demo.py …")

    tests = [
        test_global_retrieval_shapes,
        test_within_source_retrieval,
        test_cosine_distance_ordering,
        test_exclude_same_scenario,
        test_traj_signals_shape,
        test_align_traj_origin,
        test_plot_functions_no_error,
        test_run_demo_output_files,
        test_query_selection_by_scenario_id,
    ]

    for i, fn in enumerate(tests, 1):
        print(f"\n[{i}] {fn.__name__}")
        fn()

    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
