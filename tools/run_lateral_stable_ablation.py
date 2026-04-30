#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import generate_policy_rollouts as gpr

REQ_FILES = [
    "source_index.npy",
    "policy_id.npy",
    "policy_name.npy",
    "split.npy",
    "feat_style.npy",
    "traj.npy",
    "front.npy",
    "meta.npy",
]


def parse_args():
    p = argparse.ArgumentParser(description="Run lateral_stable ablation and population evaluation.")
    p.add_argument("--source_data_dir", "--input_data_dir", dest="source_data_dir", required=True)
    p.add_argument("--base_output_dir", required=True)
    p.add_argument("--embedding", default="feat_style", choices=["feat_style", "feat_style_raw", "feat", "feat_legacy"])
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--distance", default="euclidean", choices=["euclidean", "cosine"])
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--max_sources", type=int, default=None)
    p.add_argument("--configs", type=str, default=None)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_generation", action="store_true")
    p.add_argument("--skip_evaluation", action="store_true")
    p.add_argument("--num_workers", type=int, default=None)
    return p.parse_args()


def zscore(vals):
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return arr
    std = arr.std()
    if std < 1e-12:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


def run_cmd(cmd, dry_run=False):
    print("[CMD]", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def find_source_files(source_data_dir: Path):
    mapping = {
        "traj": source_data_dir / "traj.npy",
        "front": source_data_dir / "front.npy",
        "split": source_data_dir / "split.npy",
        "meta": source_data_dir / "meta.npy",
    }
    for k in ["traj", "front"]:
        if not mapping[k].exists():
            raise FileNotFoundError(f"Required source file missing: {mapping[k]}")
    return mapping


def build_config_grid():
    base = dict(gpr.POLICY_PARAMS["lateral_stable"])
    for k in ["heading_smooth_alpha", "yaw_rate_clip", "thw_target", "jerk_limit", "a_max", "a_min"]:
        if k not in base:
            raise KeyError(f"Missing expected lateral_stable parameter: {k}")

    def c(**kw):
        out = dict(base)
        out.update(kw)
        return out

    return {
        "baseline_current": c(),
        "no_lateral_smoothing": c(heading_smooth_alpha=0.0),
        "weak_lateral_stable": c(
            heading_smooth_alpha=max(0.0, base["heading_smooth_alpha"] * 0.6),
            yaw_rate_clip=min(gpr.POLICY_PARAMS["conservative"]["yaw_rate_clip"], base["yaw_rate_clip"] * 1.25),
        ),
        "strong_yaw_clip": c(yaw_rate_clip=max(0.005, base["yaw_rate_clip"] * 0.6)),
        "strong_heading_smoothing": c(heading_smooth_alpha=min(0.85, base["heading_smooth_alpha"] + 0.2)),
        "comfort_only": c(
            heading_smooth_alpha=0.0,
            yaw_rate_clip=gpr.POLICY_PARAMS["conservative"]["yaw_rate_clip"],
        ),
        "lateral_only": c(
            thw_target=gpr.POLICY_PARAMS["conservative"]["thw_target"],
            jerk_limit=gpr.POLICY_PARAMS["conservative"]["jerk_limit"],
            a_max=gpr.POLICY_PARAMS["conservative"]["a_max"],
            a_min=gpr.POLICY_PARAMS["conservative"]["a_min"],
        ),
        "full_strong_lateral_stable": c(
            heading_smooth_alpha=min(0.85, base["heading_smooth_alpha"] + 0.25),
            yaw_rate_clip=max(0.005, base["yaw_rate_clip"] * 0.5),
            thw_target=min(base["thw_target"] + 0.2, gpr.POLICY_PARAMS["conservative"]["thw_target"]),
            jerk_limit=max(0.1, base["jerk_limit"] * 0.75),
            a_max=max(0.8, base["a_max"] * 0.85),
            a_min=min(-0.5, base["a_min"] * 0.9),
        ),
    }


def collect_summary(summary_path: Path):
    s = json.loads(summary_path.read_text())
    by_policy = s.get("centroid_accuracy_by_policy", {})
    out = {
        "centroid_accuracy_overall": s.get("centroid_accuracy_overall"),
        "centroid_accuracy_p0": by_policy.get("0"),
        "centroid_accuracy_p1": by_policy.get("1"),
        "centroid_accuracy_p2": by_policy.get("2"),
        "retrieval_hit_at_1": s.get("retrieval_hit_at_1"),
        "retrieval_hit_at_k": s.get("retrieval_hit_at_k"),
        "retrieval_mean_same_policy_fraction_topk": s.get("retrieval_mean_same_policy_fraction_topk"),
        "d_p0_p1_mean": s.get("d_p0_p1_stats", {}).get("mean"),
        "d_p0_p2_mean": s.get("d_p0_p2_stats", {}).get("mean"),
        "d_p1_p2_mean": s.get("d_p1_p2_stats", {}).get("mean"),
        "p2_farthest_rate": s.get("p2_farthest_rate"),
        "mean_p2_separation_margin": s.get("mean_p2_separation_margin"),
        "median_p2_separation_margin": s.get("median_p2_separation_margin"),
        "pct_p2_separation_margin_gt_0": s.get("pct_p2_separation_margin_gt_0"),
    }
    return out


def add_style_metrics(cfg_dir: Path, row: dict):
    import pandas as pd
    style = pd.read_csv(cfg_dir / "population_eval" / "per_source_style_summary.csv")
    p2 = style[style["policy_id"] == 2]
    p0 = style[style["policy_id"] == 0]
    p1 = style[style["policy_id"] == 1]
    row.update({
        "p2_mean_speed": float(p2["mean_speed"].mean()),
        "p2_rms_jerk_mean": float(p2["rms_jerk"].mean()),
        "p2_rms_yaw_rate_proxy_mean": float(p2["rms_yaw_rate_proxy"].mean()),
        "p2_rms_curvature_proxy_mean": float(p2["rms_curvature_proxy"].mean()),
        "p2_mean_thw": float(p2["mean_thw"].mean()),
        "p2_min_thw": float(p2["min_thw"].mean()),
        "p2_vs_p0_rms_jerk_delta": float(p2["rms_jerk"].mean() - p0["rms_jerk"].mean()),
        "p2_vs_p1_rms_jerk_delta": float(p2["rms_jerk"].mean() - p1["rms_jerk"].mean()),
        "p2_vs_p0_rms_yaw_delta": float(p2["rms_yaw_rate_proxy"].mean() - p0["rms_yaw_rate_proxy"].mean()),
        "p2_vs_p1_rms_yaw_delta": float(p2["rms_yaw_rate_proxy"].mean() - p1["rms_yaw_rate_proxy"].mean()),
        "p2_vs_p0_curvature_delta": float(p2["rms_curvature_proxy"].mean() - p0["rms_curvature_proxy"].mean()),
        "p2_vs_p1_curvature_delta": float(p2["rms_curvature_proxy"].mean() - p1["rms_curvature_proxy"].mean()),
    })


def main():
    args = parse_args()
    source_dir = Path(args.source_data_dir)
    out_root = Path(args.base_output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    files = find_source_files(source_dir)

    if args.max_sources is not None:
        tmp = out_root / "_tmp_subset_sources"
        tmp.mkdir(parents=True, exist_ok=True)
        for key in ["traj", "front", "split", "meta"]:
            src = files[key]
            if not src.exists():
                continue
            arr = np.load(src, allow_pickle=True)
            np.save(tmp / f"{key}.npy", arr[: args.max_sources], allow_pickle=True)
        files = find_source_files(tmp)

    grid = build_config_grid()
    selected = list(grid.keys()) if args.configs is None else [x.strip() for x in args.configs.split(",") if x.strip()]
    unknown = [c for c in selected if c not in grid]
    if unknown:
        raise ValueError(f"Unknown configs requested: {unknown}; available={list(grid.keys())}")

    print("=== Effective lateral_stable config parameters ===")
    for name in selected:
        print(name, json.dumps(grid[name], sort_keys=True))

    rows = []
    for name in selected:
        cfg = grid[name]
        cfg_dir = out_root / name
        roll_dir = cfg_dir / "rollouts"
        eval_dir = cfg_dir / "population_eval"
        roll_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)

        if not args.skip_generation:
            cmd = [
                sys.executable,
                "generate_policy_rollouts.py",
                "--src_traj_path", str(files["traj"]),
                "--src_front_path", str(files["front"]),
                "--output_dir", str(roll_dir),
                "--seed", "42",
                "--heading_smooth_alpha", str(cfg["heading_smooth_alpha"]),
                "--lateral_stable_yaw_rate_clip", str(cfg["yaw_rate_clip"]),
                "--lateral_stable_thw_target", str(cfg["thw_target"]),
                "--lateral_stable_jerk_limit", str(cfg["jerk_limit"]),
                "--lateral_stable_a_max", str(cfg["a_max"]),
                "--lateral_stable_a_min", str(cfg["a_min"]),
            ]
            if files["split"].exists():
                cmd += ["--src_split_path", str(files["split"])]
            if files["meta"].exists():
                cmd += ["--src_meta_path", str(files["meta"])]
            run_cmd(cmd, dry_run=args.dry_run)

        if not args.dry_run:
            missing = [f for f in REQ_FILES if not (roll_dir / f).exists()]
            if missing:
                raise FileNotFoundError(f"{name}: Missing required rollout files: {missing}")

        if not args.skip_evaluation:
            ecmd = [
                sys.executable,
                "tools/evaluate_policy_population.py",
                "--data_dir", str(roll_dir),
                "--out_dir", str(eval_dir),
                "--embedding", args.embedding,
                "--split", args.split,
                "--distance", args.distance,
                "--topk", str(args.topk),
                "--projection", "pca",
            ]
            run_cmd(ecmd, dry_run=args.dry_run)

        if not args.dry_run and not args.skip_evaluation:
            row = {"config_name": name}
            for k in ["heading_smooth_alpha", "yaw_rate_clip", "thw_target", "jerk_limit", "a_max", "a_min"]:
                row[k] = cfg[k]
            row.update(collect_summary(eval_dir / "population_summary.json"))
            add_style_metrics(cfg_dir, row)
            rows.append(row)

    if args.dry_run or args.skip_evaluation:
        return

    cols = list(rows[0].keys())
    csv_path = out_root / "ablation_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)
    (out_root / "ablation_summary.json").write_text(json.dumps(rows, indent=2))

    # recommendation
    sep = zscore([r["mean_p2_separation_margin"] for r in rows])
    far = zscore([r["p2_farthest_rate"] for r in rows])
    cp2 = zscore([r["centroid_accuracy_p2"] for r in rows])
    ret = zscore([r["retrieval_mean_same_policy_fraction_topk"] for r in rows])
    jerk_vals = np.asarray([r["p2_rms_jerk_mean"] for r in rows], dtype=float)
    thw_vals = np.asarray([r["p2_min_thw"] for r in rows], dtype=float)
    jerk_pen = np.maximum(0.0, zscore(jerk_vals))
    thw_pen = np.maximum(0.0, zscore(np.maximum(0.0, thw_vals.mean() - thw_vals)))

    best_i, best_score = 0, -1e9
    scored = []
    for i, r in enumerate(rows):
        score = float(sep[i] + far[i] + cp2[i] + ret[i] - jerk_pen[i] - thw_pen[i])
        warnings = []
        if jerk_pen[i] > 0: warnings.append("high_p2_jerk")
        if thw_pen[i] > 0: warnings.append("low_p2_min_thw")
        rec = {"config_name": r["config_name"], "score": score, "warning_flags": warnings,
               "score_components": {"z_mean_p2_separation_margin": float(sep[i]), "z_p2_farthest_rate": float(far[i]), "z_centroid_accuracy_p2": float(cp2[i]), "z_retrieval_mean_same_policy_fraction_topk": float(ret[i]), "penalty_high_jerk": float(jerk_pen[i]), "penalty_low_min_thw": float(thw_pen[i])}}
        scored.append(rec)
        if score > best_score:
            best_i, best_score = i, score
    best = scored[best_i]
    rec_json = {
        "best_config_name": best["config_name"],
        "reason": "Highest p2_independence_score balancing p2 separation, p2 farthest rate, p2 centroid accuracy, retrieval consistency, and comfort penalties.",
        "score_components": best["score_components"],
        "warning_flags": best["warning_flags"],
        "all_config_scores": scored,
    }
    (out_root / "ablation_recommendation.json").write_text(json.dumps(rec_json, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["config_name"] for r in rows]
    def bar_plot(path, title, ys_map):
        plt.figure(figsize=(10,5))
        x = np.arange(len(names))
        width = 0.8 / max(1, len(ys_map))
        for i, (lab, vals) in enumerate(ys_map.items()):
            plt.bar(x + i*width, vals, width, label=lab)
        plt.xticks(x + width*(len(ys_map)-1)/2, names, rotation=30, ha="right")
        plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(path); plt.close()

    bar_plot(out_root / "ablation_p2_separation_margin.png", "P2 Mean Separation Margin", {"mean_p2_separation_margin": [r["mean_p2_separation_margin"] for r in rows]})
    bar_plot(out_root / "ablation_p2_farthest_rate.png", "P2 Farthest Rate", {"p2_farthest_rate": [r["p2_farthest_rate"] for r in rows]})
    bar_plot(out_root / "ablation_pairwise_distances.png", "Pairwise Distances", {"d_p0_p1_mean": [r["d_p0_p1_mean"] for r in rows], "d_p0_p2_mean": [r["d_p0_p2_mean"] for r in rows], "d_p1_p2_mean": [r["d_p1_p2_mean"] for r in rows]})
    bar_plot(out_root / "ablation_retrieval_classification.png", "Retrieval/Classification", {"centroid_accuracy_overall": [r["centroid_accuracy_overall"] for r in rows], "retrieval_hit_at_1": [r["retrieval_hit_at_1"] for r in rows], "retrieval_hit_at_k": [r["retrieval_hit_at_k"] for r in rows]})
    bar_plot(out_root / "ablation_p2_style_metrics.png", "P2 Style Metrics", {"rms_jerk": [r["p2_rms_jerk_mean"] for r in rows], "rms_yaw_rate_proxy": [r["p2_rms_yaw_rate_proxy_mean"] for r in rows], "rms_curvature_proxy": [r["p2_rms_curvature_proxy_mean"] for r in rows], "mean_thw": [r["p2_mean_thw"] for r in rows]})

    plt.figure(figsize=(8,5))
    xs = [r["p2_rms_yaw_rate_proxy_mean"] for r in rows]
    ys = [r["mean_p2_separation_margin"] for r in rows]
    plt.scatter(xs, ys)
    for x, y, n in zip(xs, ys, names):
        plt.text(x, y, n)
    plt.xlabel("p2_rms_yaw_rate_proxy_mean")
    plt.ylabel("mean_p2_separation_margin")
    plt.tight_layout(); plt.savefig(out_root / "ablation_tradeoff_plot.png"); plt.close()

    report = out_root / "ablation_report.md"
    report.write_text(
        "# Lateral_stable Ablation Report\n\n"
        "## 1. Experiment goal\nPopulation-level ablation to improve p2 independence while preserving smoothness/comfort.\n\n"
        "## 2. Config table\nSee `ablation_summary.csv`.\n\n"
        "## 3. Main result table\nSee `ablation_summary.csv` and `ablation_summary.json`.\n\n"
        "## 4. Which config improves p2 separation?\nUse `mean_p2_separation_margin` + `p2_farthest_rate` jointly across all sources.\n\n"
        "## 5. Which config best preserves comfort?\nInspect p2 jerk/thw metrics and warning flags.\n\n"
        f"## 6. Which config is recommended?\n`{rec_json['best_config_name']}` from `ablation_recommendation.json`.\n\n"
        "## 7. Limitations\nNo public dataset validation included yet. No sensor rendering/perception stack used.\n\n"
        "## 8. Next suggested experiment\nPerform finer local sweep around recommended config and validate generalization across splits.\n"
    )


if __name__ == "__main__":
    main()
