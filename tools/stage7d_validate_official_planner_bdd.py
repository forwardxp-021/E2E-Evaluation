#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import itertools
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

REQUIRED_PLANNERS = [
    "simple_planner",
    "idm_longitudinal_conservative",
    "idm_longitudinal_comfort",
    "idm_longitudinal_aggressive",
]
CHANNELS = ["x", "y", "yaw", "speed", "velocity_y", "acceleration", "acceleration_y", "time_s"]
FEATURES = [
    "mean_speed", "max_speed", "final_displacement", "mean_accel", "mean_abs_accel",
    "max_accel", "min_accel", "rms_accel", "jerk_mean", "jerk_rms",
    "jerk_abs_mean", "jerk_abs_p95", "low_speed_ratio", "path_length",
    "yaw_rate_rms", "yaw_rate_abs_mean",
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def require_files(sim_dir: Path) -> Dict[str, Path]:
    names = [
        "simulated_ego_seq.npy", "simulated_ego_seq_mask.npy", "simulated_ego_seq_index.json",
        "simulated_planner_metadata.csv", "scenario_planner_index.csv", "simulation_schema.json", "warnings.json",
    ]
    paths = {name: sim_dir / name for name in names}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required Stage 7C.2C official output files: " + ", ".join(missing))
    return paths


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def planner_axis(metadata_path: Path, index_path: Path, p_count: int) -> List[str]:
    rows = read_csv_rows(metadata_path)
    pairs = []
    for row in rows:
        if row.get("planner_name") and row.get("planner_id", "").strip() != "":
            try:
                pairs.append((int(row["planner_id"]), row["planner_name"]))
            except ValueError:
                pass
    if not pairs:
        idx_rows = read_csv_rows(index_path)
        seen = {}
        for row in idx_rows:
            if row.get("planner_name") and row.get("planner_id", "").strip() != "":
                try:
                    seen[int(row["planner_id"])] = row["planner_name"]
                except ValueError:
                    pass
        pairs = sorted(seen.items())
    axis = [name for _, name in sorted(set(pairs))]
    if len(axis) != p_count:
        raise ValueError(f"Planner axis length mismatch: tensor P={p_count}, metadata planners={axis}")
    return axis


def validate_schema(schema: Dict[str, Any], warnings_in: Dict[str, Any]) -> None:
    if schema.get("pseudo_rollout") is not False:
        raise ValueError("simulation_schema.json validation failed: pseudo_rollout must be false for Stage 7D official validation.")
    if schema.get("uses_official_nuplan_simulation") is not True:
        raise ValueError("simulation_schema.json validation failed: uses_official_nuplan_simulation must be true.")
    missing_pair_count = schema.get("missing_pair_count", warnings_in.get("missing_pair_count"))
    if missing_pair_count is not None and int(missing_pair_count) != 0:
        raise ValueError(f"Stage 7D requires complete planner pairs; missing_pair_count={missing_pair_count}.")


def valid_dt(time_s: np.ndarray, warn: List[Dict[str, Any]], scenario_index: int, planner_name: str) -> np.ndarray:
    t = np.asarray(time_s, dtype=float)
    dt = np.diff(t)
    positive = dt[np.isfinite(dt) & (dt > 1e-6)]
    if positive.size:
        fill = float(np.median(positive))
    else:
        fill = 0.1
        warn.append({"type": "time_s_invalid_dt_fallback", "scenario_index": scenario_index, "planner_name": planner_name, "fallback_dt": fill})
    dt_safe = np.where(np.isfinite(dt) & (dt > 1e-6), dt, fill)
    return dt_safe


def extract_features(seq: np.ndarray, mask: np.ndarray, scenario_index: int, planner_name: str, warn: List[Dict[str, Any]]) -> Dict[str, float]:
    valid = mask.astype(bool) & np.all(np.isfinite(seq), axis=1)
    if not np.any(valid):
        raise ValueError(f"Mask has no valid timesteps for scenario_index={scenario_index}, planner_name={planner_name}.")
    arr = seq[valid]
    x, y, yaw, speed, _vy, accel, _ay, time_s = [arr[:, i].astype(float) for i in range(8)]
    if arr.shape[0] >= 2:
        dx, dy = np.diff(x), np.diff(y)
        step_dist = np.sqrt(dx * dx + dy * dy)
        path_length = float(np.sum(step_dist))
        final_displacement = float(math.hypot(float(x[-1] - x[0]), float(y[-1] - y[0])))
        dt = valid_dt(time_s, warn, scenario_index, planner_name)
        jerk = np.diff(accel) / dt
        yaw_delta = np.unwrap(yaw)
        yaw_rate = np.diff(yaw_delta) / dt
    else:
        path_length = 0.0
        final_displacement = 0.0
        jerk = np.array([0.0])
        yaw_rate = np.array([0.0])
    jerk = jerk[np.isfinite(jerk)] if jerk.size else np.array([0.0])
    yaw_rate = yaw_rate[np.isfinite(yaw_rate)] if yaw_rate.size else np.array([0.0])
    return {
        "mean_speed": float(np.mean(speed)),
        "max_speed": float(np.max(speed)),
        "final_displacement": final_displacement,
        "mean_accel": float(np.mean(accel)),
        "mean_abs_accel": float(np.mean(np.abs(accel))),
        "max_accel": float(np.max(accel)),
        "min_accel": float(np.min(accel)),
        "rms_accel": float(np.sqrt(np.mean(accel * accel))),
        "jerk_mean": float(np.mean(jerk)),
        "jerk_rms": float(np.sqrt(np.mean(jerk * jerk))),
        "jerk_abs_mean": float(np.mean(np.abs(jerk))),
        "jerk_abs_p95": float(np.percentile(np.abs(jerk), 95)),
        "low_speed_ratio": float(np.mean(speed < 0.5)),
        "path_length": path_length,
        "yaw_rate_rms": float(np.sqrt(np.mean(yaw_rate * yaw_rate))),
        "yaw_rate_abs_mean": float(np.mean(np.abs(yaw_rate))),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def standardize(mat: np.ndarray) -> np.ndarray:
    mu = np.nanmean(mat, axis=0)
    sd = np.nanstd(mat, axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (mat - mu) / sd


def rbf_mmd2(a: np.ndarray, b: np.ndarray) -> float:
    both = np.vstack([a, b])
    d2 = np.sum((both[:, None, :] - both[None, :, :]) ** 2, axis=2)
    vals = d2[d2 > 1e-12]
    gamma = 1.0 / (2.0 * (float(np.median(vals)) if vals.size else 1.0))
    def k(x, y):
        return np.exp(-gamma * np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2))
    return float(np.mean(k(a, a)) + np.mean(k(b, b)) - 2.0 * np.mean(k(a, b)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 7D.1/7D.2 first-pass BDD validation on official nuPlan planner trajectories.")
    ap.add_argument("--sim_dir", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dir exists: {args.output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    paths = require_files(args.sim_dir)
    schema = read_json(paths["simulation_schema.json"])
    input_warnings = read_json(paths["warnings.json"])
    if not isinstance(input_warnings, dict):
        input_warnings = {"source_warnings": input_warnings}
    validate_schema(schema, input_warnings)

    seq = np.load(paths["simulated_ego_seq.npy"], mmap_mode="r")
    mask = np.load(paths["simulated_ego_seq_mask.npy"], mmap_mode="r")
    if seq.ndim != 4 or seq.shape[-1] != 8:
        raise ValueError(f"simulated_ego_seq.npy shape must be [N, P, T, 8], got {list(seq.shape)}.")
    if mask.shape != seq.shape[:3]:
        raise ValueError(f"simulated_ego_seq_mask.npy shape must match [N, P, T]={list(seq.shape[:3])}, got {list(mask.shape)}.")
    if not np.any(mask):
        raise ValueError("simulated_ego_seq_mask.npy has no valid timesteps.")
    planners = planner_axis(paths["simulated_planner_metadata.csv"], paths["scenario_planner_index.csv"], seq.shape[1])
    missing = [p for p in REQUIRED_PLANNERS if p not in planners]
    if missing:
        raise ValueError(f"Planner axis missing required longitudinal controls: {missing}; observed={planners}")

    diag: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []
    for i in range(seq.shape[0]):
        for p, planner in enumerate(planners):
            feats = extract_features(np.asarray(seq[i, p]), np.asarray(mask[i, p]), i, planner, diag)
            feature_rows.append({"scenario_index": i, "planner_id": p, "planner_name": planner, **feats})
    write_csv(args.output_dir / "planner_kinematic_features.csv", feature_rows, ["scenario_index", "planner_id", "planner_name", *FEATURES])

    summary_rows = []
    mat = np.array([[r[f] for f in FEATURES] for r in feature_rows], dtype=float)
    zmat = standardize(mat)
    row_key = [(int(r["scenario_index"]), str(r["planner_name"])) for r in feature_rows]
    for planner in planners:
        vals = np.array([[r[f] for f in FEATURES] for r in feature_rows if r["planner_name"] == planner], dtype=float)
        row = {"planner_name": planner, "count": vals.shape[0]}
        for j, f in enumerate(FEATURES):
            row[f"{f}_mean"] = float(np.mean(vals[:, j])); row[f"{f}_std"] = float(np.std(vals[:, j], ddof=0))
        summary_rows.append(row)
    write_csv(args.output_dir / "planner_feature_summary.csv", summary_rows, list(summary_rows[0].keys()))

    z_by_key = {k: zmat[idx] for idx, k in enumerate(row_key)}
    paired_delta = []
    paired_dist = []
    for a, b in itertools.combinations(planners, 2):
        dists = []
        for i in range(seq.shape[0]):
            va, vb = z_by_key[(i, a)], z_by_key[(i, b)]
            dists.append(float(np.linalg.norm(va - vb)))
            row = {"scenario_index": i, "planner_a": a, "planner_b": b, "standardized_feature_distance": dists[-1]}
            for j, f in enumerate(FEATURES):
                row[f"delta_{f}"] = float(vb[j] - va[j])
            paired_delta.append(row)
        paired_dist.append({"planner_a": a, "planner_b": b, "mean_within_scenario_feature_distance": float(np.mean(dists)), "std_within_scenario_feature_distance": float(np.std(dists))})
    write_csv(args.output_dir / "paired_planner_delta.csv", paired_delta, list(paired_delta[0].keys()))
    write_csv(args.output_dir / "paired_distance_matrix.csv", paired_dist, ["planner_a", "planner_b", "mean_within_scenario_feature_distance", "std_within_scenario_feature_distance"])

    dist_rows = []
    z_by_planner = {p: zmat[[k[1] == p for k in row_key]] for p in planners}
    for a, b in itertools.product(planners, planners):
        za, zb = z_by_planner[a], z_by_planner[b]
        eu = float(np.linalg.norm(np.mean(za, axis=0) - np.mean(zb, axis=0)))
        wass = float(np.mean([abs(float(np.mean(za[:, j]) - np.mean(zb[:, j]))) for j in range(zmat.shape[1])]))
        dist_rows.append({"planner_a": a, "planner_b": b, "euclidean_mean_feature_distance": eu, "rbf_mmd2_standardized": rbf_mmd2(za, zb), "avg_1d_mean_distance_standardized": wass})
    write_csv(args.output_dir / "bdd_distance_matrix.csv", dist_rows, ["planner_a", "planner_b", "euclidean_mean_feature_distance", "rbf_mmd2_standardized", "avg_1d_mean_distance_standardized"])

    report = [
        "# Stage 7D.1 / 7D.2 Official Planner BDD Validation Report", "",
        "## PASS/FAIL Summary", "", "- validation.pass: **PASS**", "- This stage uses official Stage 7C.2C nuPlan simulation outputs only; it does not run pseudo rollout or nuPlan simulation.",
        "- Interpretation: Stage 7D first-pass validates whether BDD / feature-distance can detect controlled longitudinal behavior differences among official nuPlan planner rollouts.", "",
        "## Input Tensor", "", f"- simulated_ego_seq.npy shape: `{list(seq.shape)}`", f"- simulated_ego_seq_mask.npy shape: `{list(mask.shape)}`", "",
        "## Planner Axis", "", *[f"- {idx}: `{name}`" for idx, name in enumerate(planners)], "",
        "## Feature List", "", *[f"- `{f}`" for f in FEATURES], "",
        "## BDD / Distance Matrix Files", "", "- `bdd_distance_matrix.csv`", "- `paired_distance_matrix.csv`", "- `paired_planner_delta.csv`", "",
        "## Top Observations", "", "- The validation passed all official-output guardrails, required planner-axis checks, tensor-shape checks, and non-empty-mask checks.", "- Use `bdd_distance_matrix.csv` for planner-level distribution distances and `paired_distance_matrix.csv` for within-scenario controlled distances.", "",
        "## Limitations", "", "- Longitudinal-only controlled validation.", "- 5 logs mini smoke, not a full benchmark.", "- IDM profiles are longitudinal-only positive controls, not full driving styles.",
    ]
    (args.output_dir / "stage7d_validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(args.output_dir / "stage7d_schema.json", {"stage": "7D.1/7D.2", "input_stage": "7C.2C-2", "uses_official_nuplan_simulation_outputs": True, "pseudo_rollout": False, "longitudinal_only_controlled_validation": True, "feature_channels": CHANNELS, "features": FEATURES, "input_tensor_shape": list(seq.shape), "planner_axis": planners})
    write_json(args.output_dir / "warnings.json", {"validation": {"pass": True}, "diagnostics": diag, "input_warnings_summary": input_warnings})
    print(f"Stage 7D validation PASS. Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
