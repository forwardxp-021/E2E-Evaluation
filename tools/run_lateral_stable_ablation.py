#!/usr/bin/env python3
"""Experiment 2: lateral_stable ablation + parameter sweep orchestrator."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import generate_policy_rollouts as gpr

PARAM_KEYS = ["heading_smooth_alpha", "yaw_rate_clip", "thw_target", "jerk_limit", "a_max", "a_min"]
REQ_ROLLOUT_FILES = ["source_index.npy", "policy_id.npy", "policy_name.npy", "split.npy", "traj.npy", "front.npy", "meta.npy", "feat_style.npy"]


def parse_args():
    p = argparse.ArgumentParser(description="Run lateral_stable ablation and population evaluator sweep.")
    p.add_argument("--source_data_dir", required=True)
    p.add_argument("--base_output_dir", required=True)
    p.add_argument("--embedding", default="feat_style")
    p.add_argument("--split", default="test")
    p.add_argument("--distance", default="euclidean")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--max_sources", type=int, default=None)
    p.add_argument("--configs", type=str, default=None)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_generation", action="store_true")
    p.add_argument("--skip_evaluation", action="store_true")
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def zscore(x):
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return arr
    sd = arr.std()
    return np.zeros_like(arr) if sd < 1e-12 else (arr - arr.mean()) / sd


def run_cmd(cmd, dry):
    print("[CMD]", " ".join(cmd))
    if not dry:
        subprocess.run(cmd, check=True)


def build_config_grid():
    base = dict(gpr.POLICY_PARAMS["lateral_stable"])
    cons = gpr.POLICY_PARAMS["conservative"]
    aggr = gpr.POLICY_PARAMS["aggressive"]
    for k in PARAM_KEYS:
        if k not in base:
            raise KeyError(f"Missing expected lateral_stable parameter in generator: {k}")

    def cfg(**kw):
        out = dict(base)
        out.update(kw)
        return out

    return {
        "baseline_current": cfg(),
        "no_lateral_smoothing": cfg(heading_smooth_alpha=0.0),
        "weak_lateral_stable": cfg(heading_smooth_alpha=max(0.15, base["heading_smooth_alpha"] * 0.7), yaw_rate_clip=min(base["yaw_rate_clip"] * 1.25, aggr["yaw_rate_clip"])),
        "strong_yaw_clip": cfg(yaw_rate_clip=max(0.005, base["yaw_rate_clip"] * 0.6)),
        "strong_heading_smoothing": cfg(heading_smooth_alpha=min(0.85, base["heading_smooth_alpha"] + 0.25)),
        "comfort_only": cfg(heading_smooth_alpha=0.0, yaw_rate_clip=cons["yaw_rate_clip"]),
        "lateral_only": cfg(thw_target=aggr["thw_target"], jerk_limit=aggr["jerk_limit"], a_max=aggr["a_max"], a_min=aggr["a_min"]),
        "full_strong_lateral_stable": cfg(heading_smooth_alpha=min(0.85, base["heading_smooth_alpha"] + 0.3), yaw_rate_clip=max(0.005, base["yaw_rate_clip"] * 0.5), thw_target=min(base["thw_target"] + 0.3, cons["thw_target"]), jerk_limit=max(0.1, base["jerk_limit"] * 0.7), a_max=max(0.8, base["a_max"] * 0.85), a_min=min(-0.5, base["a_min"] * 0.9)),
    }


def find_source(source_dir: Path):
    files = {k: source_dir / f"{k}.npy" for k in ["traj", "front", "split", "meta"]}
    for req in ["traj", "front"]:
        if not files[req].exists():
            raise FileNotFoundError(f"Missing required source file: {files[req]}")
    return files


def subset_sources(files, out_root: Path, n):
    if n is None:
        return files
    tmp = out_root / "_tmp_subset_sources"
    tmp.mkdir(parents=True, exist_ok=True)
    for k, f in files.items():
        if f.exists():
            arr = np.load(f, allow_pickle=True)
            np.save(tmp / f"{k}.npy", arr[:n], allow_pickle=True)
    return find_source(tmp)


def read_eval_row(cfg_name, cfg, eval_dir: Path):
    s = json.loads((eval_dir / "population_summary.json").read_text())
    style_csv = eval_dir / "per_source_style_summary.csv"
    import pandas as pd
    st = pd.read_csv(style_csv)
    p = {pid: st[st["policy_id"] == pid] for pid in [0, 1, 2]}
    row = {"config_name": cfg_name, "generation_status": "success", "evaluation_status": "success", **{k: cfg[k] for k in PARAM_KEYS}}
    row.update({
        "n_complete_sources": s.get("n_complete_sources"), "n_rows_after_split": s.get("n_rows_after_split"), "warnings_count": len(s.get("warnings", [])),
        "centroid_accuracy_overall": s.get("centroid_accuracy_overall"), "centroid_accuracy_p0": s.get("centroid_accuracy_by_policy", {}).get("0"), "centroid_accuracy_p1": s.get("centroid_accuracy_by_policy", {}).get("1"), "centroid_accuracy_p2": s.get("centroid_accuracy_by_policy", {}).get("2"),
        "retrieval_hit_at_1": s.get("retrieval_hit_at_1"), "retrieval_hit_at_k": s.get("retrieval_hit_at_k"), "retrieval_mean_same_policy_count_topk": s.get("retrieval_mean_same_policy_count_topk"), "retrieval_mean_same_policy_fraction_topk": s.get("retrieval_mean_same_policy_fraction_topk"),
        "d_p0_p1_mean": s.get("d_p0_p1_stats", {}).get("mean"), "d_p0_p2_mean": s.get("d_p0_p2_stats", {}).get("mean"), "d_p1_p2_mean": s.get("d_p1_p2_stats", {}).get("mean"),
        "d_p0_p1_median": s.get("d_p0_p1_stats", {}).get("median"), "d_p0_p2_median": s.get("d_p0_p2_stats", {}).get("median"), "d_p1_p2_median": s.get("d_p1_p2_stats", {}).get("median"),
        "p2_farthest_rate": s.get("p2_farthest_rate"), "mean_p2_separation_margin": s.get("mean_p2_separation_margin"), "median_p2_separation_margin": s.get("median_p2_separation_margin"), "pct_p2_separation_margin_gt_0": s.get("pct_p2_separation_margin_gt_0"),
        "p2_mean_speed": float(p[2]["mean_speed"].mean()), "p2_rms_jerk_mean": float(p[2]["rms_jerk"].mean()), "p2_rms_yaw_rate_proxy_mean": float(p[2]["rms_yaw_rate_proxy"].mean()), "p2_rms_curvature_proxy_mean": float(p[2]["rms_curvature_proxy"].mean()), "p2_mean_thw": float(p[2]["mean_thw"].mean()), "p2_min_thw": float(p[2]["min_thw"].mean()),
        "p2_vs_p0_rms_jerk_delta": float(p[2]["rms_jerk"].mean() - p[0]["rms_jerk"].mean()), "p2_vs_p1_rms_jerk_delta": float(p[2]["rms_jerk"].mean() - p[1]["rms_jerk"].mean()),
        "p2_vs_p0_rms_yaw_delta": float(p[2]["rms_yaw_rate_proxy"].mean() - p[0]["rms_yaw_rate_proxy"].mean()), "p2_vs_p1_rms_yaw_delta": float(p[2]["rms_yaw_rate_proxy"].mean() - p[1]["rms_yaw_rate_proxy"].mean()),
        "p2_vs_p0_curvature_delta": float(p[2]["rms_curvature_proxy"].mean() - p[0]["rms_curvature_proxy"].mean()), "p2_vs_p1_curvature_delta": float(p[2]["rms_curvature_proxy"].mean() - p[1]["rms_curvature_proxy"].mean()),
    })
    return row


def main():
    args = parse_args()
    out_root = Path(args.base_output_dir); out_root.mkdir(parents=True, exist_ok=True)
    files = subset_sources(find_source(Path(args.source_data_dir)), out_root, args.max_sources)
    grid = build_config_grid()
    selected = list(grid.keys()) if not args.configs else [x.strip() for x in args.configs.split(",") if x.strip()]
    bad = [x for x in selected if x not in grid]
    if bad:
        raise ValueError(f"Unknown config(s): {bad}; available={list(grid)}")

    print("=== Resolved configs ===")
    for n in selected:
        print(n, json.dumps({k: grid[n][k] for k in PARAM_KEYS}, sort_keys=True))

    rows = []
    for n in selected:
        cfg = grid[n]
        cdir = out_root / n; rdir = cdir / "rollouts"; edir = cdir / "population_eval"
        rdir.mkdir(parents=True, exist_ok=True); edir.mkdir(parents=True, exist_ok=True)

        if not args.skip_generation:
            gcmd = [sys.executable, "generate_policy_rollouts.py", "--src_traj_path", str(files["traj"]), "--src_front_path", str(files["front"]), "--output_dir", str(rdir), "--seed", str(args.seed), "--heading_smooth_alpha", str(cfg["heading_smooth_alpha"]), "--lateral_stable_yaw_rate_clip", str(cfg["yaw_rate_clip"]), "--lateral_stable_thw_target", str(cfg["thw_target"]), "--lateral_stable_jerk_limit", str(cfg["jerk_limit"]), "--lateral_stable_a_max", str(cfg["a_max"]), "--lateral_stable_a_min", str(cfg["a_min"])]
            if args.num_workers is not None:
                print("[WARN] --num_workers provided but generator has no such argument; ignoring")
            if files["split"].exists(): gcmd += ["--src_split_path", str(files["split"])]
            if files["meta"].exists(): gcmd += ["--src_meta_path", str(files["meta"])]
            run_cmd(gcmd, args.dry_run)

        if not args.skip_evaluation:
            ecmd = [sys.executable, "tools/evaluate_policy_population.py", "--data_dir", str(rdir), "--out_dir", str(edir), "--embedding", args.embedding, "--split", args.split, "--distance", args.distance, "--topk", str(args.topk), "--projection", "pca"]
            run_cmd(ecmd, args.dry_run)

        if not args.dry_run and not args.skip_evaluation:
            missing = [f for f in REQ_ROLLOUT_FILES if not (rdir / f).exists()]
            if missing:
                raise FileNotFoundError(f"{n}: missing rollout files {missing}")
            rows.append(read_eval_row(n, cfg, edir))

    if args.dry_run or args.skip_evaluation:
        return

    cols = list(rows[0].keys())
    with (out_root / "ablation_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    (out_root / "ablation_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    sep = zscore([r["mean_p2_separation_margin"] for r in rows]); far = zscore([r["p2_farthest_rate"] for r in rows]); cp2 = zscore([r["centroid_accuracy_p2"] for r in rows]); ret = zscore([r["retrieval_mean_same_policy_fraction_topk"] for r in rows])
    jerk_th, thw_th = 1.8, 0.8
    scored = []
    for i, r in enumerate(rows):
        pen_jerk = max(0.0, float((r["p2_rms_jerk_mean"] - jerk_th) / max(jerk_th, 1e-6)))
        pen_thw = max(0.0, float((thw_th - r["p2_min_thw"]) / max(thw_th, 1e-6)))
        score = float(sep[i] + far[i] + cp2[i] + ret[i] - pen_jerk - pen_thw)
        warns = [w for w, c in [("high_p2_jerk", pen_jerk > 0), ("low_p2_min_thw", pen_thw > 0)] if c]
        scored.append({"config_name": r["config_name"], "p2_independence_score": score, "score_components": {"z_mean_p2_separation_margin": float(sep[i]), "z_p2_farthest_rate": float(far[i]), "z_centroid_accuracy_p2": float(cp2[i]), "z_retrieval_mean_same_policy_fraction_topk": float(ret[i]), "penalty_high_p2_rms_jerk": pen_jerk, "penalty_low_p2_min_thw": pen_thw}, "warning_flags": warns})
    best = max(scored, key=lambda x: x["p2_independence_score"])
    rec = {"best_config_name": best["config_name"], "p2_independence_score": best["p2_independence_score"], "score_components": best["score_components"], "warning_flags": best["warning_flags"], "reason": f"Best combined separation/discriminability with penalties (jerk>{jerk_th}, min_thw<{thw_th}).", "all_config_scores": scored}
    (out_root / "ablation_recommendation.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [r["config_name"] for r in rows]
    x = np.arange(len(names))
    plt.figure(figsize=(11, 5)); vals = [r["mean_p2_separation_margin"] for r in rows]; plt.bar(names, vals); plt.axhline(0.0, color="red", linestyle="--", linewidth=1); plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.savefig(out_root / "ablation_p2_separation_margin.png"); plt.close()
    plt.figure(figsize=(11, 5)); plt.bar(names, [r["p2_farthest_rate"] for r in rows]); plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.savefig(out_root / "ablation_p2_farthest_rate.png"); plt.close()

    def grouped(path, series):
        plt.figure(figsize=(12, 5)); w = 0.8 / len(series)
        for i, (lab, vals) in enumerate(series.items()): plt.bar(x + i * w, vals, w, label=lab)
        plt.xticks(x + w * (len(series) - 1) / 2, names, rotation=30, ha="right"); plt.legend(); plt.tight_layout(); plt.savefig(out_root / path); plt.close()

    grouped("ablation_pairwise_distances.png", {"d_p0_p1_mean": [r["d_p0_p1_mean"] for r in rows], "d_p0_p2_mean": [r["d_p0_p2_mean"] for r in rows], "d_p1_p2_mean": [r["d_p1_p2_mean"] for r in rows]})
    grouped("ablation_retrieval_classification.png", {"centroid_accuracy_overall": [r["centroid_accuracy_overall"] for r in rows], "centroid_accuracy_p2": [r["centroid_accuracy_p2"] for r in rows], "retrieval_hit_at_1": [r["retrieval_hit_at_1"] for r in rows], "retrieval_hit_at_k": [r["retrieval_hit_at_k"] for r in rows]})
    grouped("ablation_p2_style_metrics.png", {"p2_rms_jerk_mean": [r["p2_rms_jerk_mean"] for r in rows], "p2_rms_yaw_rate_proxy_mean": [r["p2_rms_yaw_rate_proxy_mean"] for r in rows], "p2_rms_curvature_proxy_mean": [r["p2_rms_curvature_proxy_mean"] for r in rows], "p2_mean_thw": [r["p2_mean_thw"] for r in rows]})

    xs = [r["p2_rms_yaw_rate_proxy_mean"] for r in rows]; ys = [r["mean_p2_separation_margin"] for r in rows]
    plt.figure(figsize=(8, 5)); plt.scatter(xs, ys)
    for xi, yi, nm in zip(xs, ys, names): plt.annotate(nm, (xi, yi), fontsize=8)
    plt.xlabel("p2_rms_yaw_rate_proxy_mean"); plt.ylabel("mean_p2_separation_margin"); plt.tight_layout(); plt.savefig(out_root / "ablation_tradeoff_plot.png"); plt.close()

    report = f"""# Experiment 2: Lateral_stable Ablation and Parameter Sweep

## 1. Experiment goal
Experiment 1 showed embedding discriminability is strong, but p2/lateral_stable was not consistently farthest. This ablation evaluates which lateral_stable mechanisms improve p2 independence.

## 2. Dataset / split / config overview
- Split: `{args.split}`
- Embedding: `{args.embedding}`
- Distance: `{args.distance}`
- topk: `{args.topk}`
- Config count: {len(rows)}

## 3. Config table
See `ablation_summary.csv`.

## 4. Main metrics table
See `ablation_summary.csv` / `ablation_summary.json`.

## 5. P2 separation result
Primary metrics: `mean_p2_separation_margin`, `p2_farthest_rate`, `pct_p2_separation_margin_gt_0`.

## 6. Classification and retrieval result
Use centroid accuracy (overall + p2) and retrieval hit@1/hit@k.

## 7. P2 style metric comparison
Use p2 jerk/yaw/curvature/THW metrics to check comfort and lateral stability tradeoff.

## 8. Recommended config
- Best config: `{rec['best_config_name']}`
- p2_independence_score: {rec['p2_independence_score']:.4f}

## 9. Interpretation
Higher p2 separation with low jerk/yaw is preferred. PCA/UMAP outputs (if generated elsewhere) are visualization-only.

## 10. Limitations
Synthetic policy rollouts only (not human-driver validation); replayed front vehicle; no sensor rendering/perception stack.

## 11. Next suggested experiment
Perform a local fine-grained sweep around the recommended config and repeat on additional splits.
"""
    (out_root / "ablation_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
