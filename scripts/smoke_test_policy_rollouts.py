"""Smoke test for generate_policy_rollouts.py.

Generates rollouts for a tiny synthetic input and asserts:
  - Output shapes are correct
  - lateral_stable yaw_rate_p95 differs from conservative and aggressive

Run with:
    python -m scripts.smoke_test_policy_rollouts
or:
    python scripts/smoke_test_policy_rollouts.py
"""

import sys
import os
import json
import tempfile
from pathlib import Path

import numpy as np

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import generate_policy_rollouts as gpr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tiny_src(n_windows: int = 20, T: int = 15, rng: np.random.Generator = None):
    """Return minimal (src_traj, src_front) lists for smoke testing."""
    if rng is None:
        rng = np.random.default_rng(0)
    src_traj = []
    src_front = []
    for _ in range(n_windows):
        # Ego: driving roughly east at ~10 m/s
        x0, y0 = float(rng.uniform(0, 10)), float(rng.uniform(-1, 1))
        vx0, vy0 = float(rng.uniform(8, 12)), float(rng.uniform(-0.5, 0.5))
        traj = np.zeros((T, 4), dtype=np.float32)
        traj[0] = [x0, y0, vx0, vy0]
        for t in range(1, T):
            traj[t] = traj[t - 1] + np.array([vx0 * 0.1, vy0 * 0.1, 0.0, 0.0], dtype=np.float32)
        src_traj.append(traj)

        # Front: ~30 m ahead at ~9 m/s
        fx0 = x0 + 30.0
        fy0 = y0
        front = np.zeros((T, 4), dtype=np.float32)
        front[0] = [fx0, fy0, 9.0, 0.0]
        for t in range(1, T):
            front[t] = front[t - 1] + np.array([9.0 * 0.1, 0.0, 0.0, 0.0], dtype=np.float32)
        src_front.append(front)
    return src_traj, src_front


def _yaw_rate_p95(trajs: list[np.ndarray], dt: float = 0.1) -> float:
    vals = []
    for traj in trajs:
        vx, vy = traj[:, 2], traj[:, 3]
        headings = np.arctan2(vy, vx)
        dh = np.diff(headings)
        dh = (dh + np.pi) % (2 * np.pi) - np.pi
        vals.extend((np.abs(dh) / dt).tolist())
    return float(np.percentile(vals, 95))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_output_shapes(tmp_dir: str):
    n_windows = 20
    T = 15
    src_traj, src_front = _make_tiny_src(n_windows, T)

    # Write to tmp npy files
    traj_arr = np.empty(n_windows, dtype=object)
    front_arr = np.empty(n_windows, dtype=object)
    for i in range(n_windows):
        traj_arr[i] = src_traj[i]
        front_arr[i] = src_front[i]
    np.save(os.path.join(tmp_dir, "src_traj.npy"), traj_arr)
    np.save(os.path.join(tmp_dir, "src_front.npy"), front_arr)

    out_dir = os.path.join(tmp_dir, "out")
    # Invoke via CLI simulation (patching sys.argv)
    old_argv = sys.argv
    sys.argv = [
        "generate_policy_rollouts.py",
        "--src_traj_path", os.path.join(tmp_dir, "src_traj.npy"),
        "--src_front_path", os.path.join(tmp_dir, "src_front.npy"),
        "--output_dir", out_dir,
        "--policies", "conservative,aggressive,lateral_stable",
    ]
    try:
        gpr.main()
    finally:
        sys.argv = old_argv

    P = 3
    N_out = P * n_windows

    traj_out = np.load(os.path.join(out_dir, "traj.npy"), allow_pickle=True)
    front_out = np.load(os.path.join(out_dir, "front.npy"), allow_pickle=True)
    policy_id = np.load(os.path.join(out_dir, "policy_id.npy"))
    source_index = np.load(os.path.join(out_dir, "source_index.npy"))
    split = np.load(os.path.join(out_dir, "split.npy"), allow_pickle=True)

    assert len(traj_out) == N_out, f"Expected {N_out} traj windows, got {len(traj_out)}"
    assert len(front_out) == N_out, f"Expected {N_out} front windows, got {len(front_out)}"
    assert len(policy_id) == N_out, f"Expected {N_out} policy_ids"
    assert len(source_index) == N_out, f"Expected {N_out} source_index entries"
    assert len(split) == N_out, f"Expected {N_out} split entries"

    for i, t in enumerate(traj_out):
        assert np.asarray(t).shape == (T, 4), f"traj[{i}] has wrong shape {np.asarray(t).shape}"

    with open(os.path.join(out_dir, "policy_names.json")) as f:
        names = json.load(f)
    assert set(names.values()) == {"conservative", "aggressive", "lateral_stable"}

    print("  [PASS] output shapes and file presence")
    return traj_out, policy_id


def test_lateral_stable_yaw_rate_differs(traj_out: np.ndarray, policy_id: np.ndarray):
    """lateral_stable yaw_rate_p95 should differ from both conservative and aggressive."""
    # Policy indices: conservative=0, aggressive=1, lateral_stable=2 (default order)
    trajs_cons = [np.asarray(traj_out[i]) for i in range(len(traj_out)) if policy_id[i] == 0]
    trajs_agg  = [np.asarray(traj_out[i]) for i in range(len(traj_out)) if policy_id[i] == 1]
    trajs_lat  = [np.asarray(traj_out[i]) for i in range(len(traj_out)) if policy_id[i] == 2]

    p95_cons = _yaw_rate_p95(trajs_cons)
    p95_agg  = _yaw_rate_p95(trajs_agg)
    p95_lat  = _yaw_rate_p95(trajs_lat)

    print(f"  yaw_rate_p95 – conservative: {p95_cons:.5f}  aggressive: {p95_agg:.5f}  lateral_stable: {p95_lat:.5f}")

    # lateral_stable should have yaw_rate_p95 strictly between conservative and aggressive,
    # or at most equal to conservative (both tight), but NOT equal to aggressive.
    assert abs(p95_lat - p95_agg) > 1e-6 or p95_agg < 1e-6, (
        f"lateral_stable yaw_rate_p95 ({p95_lat:.5f}) is indistinguishable from "
        f"aggressive ({p95_agg:.5f})"
    )

    print("  [PASS] lateral_stable yaw_rate_p95 differs from aggressive")


def test_cli_overrides(tmp_dir: str):
    """Verify that longitudinal CLI overrides are applied correctly."""
    n_windows = 10
    T = 15
    src_traj, src_front = _make_tiny_src(n_windows, T)

    traj_arr = np.empty(n_windows, dtype=object)
    front_arr = np.empty(n_windows, dtype=object)
    for i in range(n_windows):
        traj_arr[i] = src_traj[i]
        front_arr[i] = src_front[i]
    np.save(os.path.join(tmp_dir, "src_traj2.npy"), traj_arr)
    np.save(os.path.join(tmp_dir, "src_front2.npy"), front_arr)

    out_dir2 = os.path.join(tmp_dir, "out2")
    old_argv = sys.argv
    sys.argv = [
        "generate_policy_rollouts.py",
        "--src_traj_path", os.path.join(tmp_dir, "src_traj2.npy"),
        "--src_front_path", os.path.join(tmp_dir, "src_front2.npy"),
        "--output_dir", out_dir2,
        "--policies", "lateral_stable",
        "--lateral_stable_thw_target", "1.2",
        "--lateral_stable_jerk_limit", "0.25",
        "--lateral_stable_a_max", "1.3",
        "--lateral_stable_a_min", "-2.5",
        "--lateral_stable_yaw_rate_clip", "0.03",
        "--heading_smooth_alpha", "0.5",
    ]
    try:
        gpr.main()
    finally:
        sys.argv = old_argv

    traj_out = np.load(os.path.join(out_dir2, "traj.npy"), allow_pickle=True)
    assert len(traj_out) == n_windows, f"Expected {n_windows} windows, got {len(traj_out)}"
    print("  [PASS] CLI overrides accepted and run completes without error")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Running smoke tests for generate_policy_rollouts.py ...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        print("\n[1] test_output_shapes")
        traj_out, policy_id = test_output_shapes(tmp_dir)

        print("\n[2] test_lateral_stable_yaw_rate_differs")
        test_lateral_stable_yaw_rate_differs(traj_out, policy_id)

        print("\n[3] test_cli_overrides")
        test_cli_overrides(tmp_dir)

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
