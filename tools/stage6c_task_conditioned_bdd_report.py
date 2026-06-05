#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tools.stage6b_compare_baselines import mmd_with_stats
from tools.stage6c_common import iter_progress, load_embeddings, write_json

TASK_SPECS = {
    "task_following": ("following", "not_following"),
    "task_lead_brake_response": ("lead_brake_response", "no_lead_brake_response"),
    "task_queue_approach": ("queue_approach", "no_queue_approach"),
    "task_lane_change": ("lane_change", "no_lane_change"),
    "task_cutin_response": ("cutin_response", "no_cutin_response"),
    "task_overtake_opportunity": ("overtake_opportunity", "no_overtake_opportunity"),
    "task_overtake_executed": ("overtake_executed", "no_overtake_executed"),
    "task_hesitation": ("hesitation", "no_hesitation"),
    "task_yield_conflict": ("yield_conflict", "no_yield_conflict"),
}
DEFAULT_TASK_KEYS = list(TASK_SPECS.keys())
META_COLUMNS = ["global_row", "shard_id", "local_row", "scenario_id", "target_agent_id", "start", "window_len", "split"]
YIELD_METRICS = [
    "yield_conflict_score",
    "yielding_score",
    "assertiveness_score",
    "gap_pressure_score",
    "conflict_accel_score",
    "small_gap_speed_maintain_score",
    "rear_pressure_response_score",
    "courtesy_score",
]


def relevant_metrics(task_key: str, metric_columns: Sequence[str]) -> List[str]:
    if task_key == "task_following":
        prefix = "following_"
    elif task_key == "task_lead_brake_response":
        prefix = "lead_brake_"
    elif task_key == "task_queue_approach":
        prefix = "queue_"
    elif task_key == "task_lane_change":
        prefix = "lc_"
    elif task_key == "task_cutin_response":
        prefix = "cutin_"
    elif task_key in {"task_overtake_opportunity", "task_overtake_executed"}:
        prefix = "overtake_"
    elif task_key == "task_hesitation":
        prefix = "hesitation_"
    elif task_key == "task_yield_conflict":
        return [m for m in YIELD_METRICS if m in metric_columns]
    else:
        return []
    return [m for m in metric_columns if m.startswith(prefix)]


def finite_mean(x) -> float:
    arr = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(arr)
    return float(np.mean(arr[ok])) if ok.any() else np.nan


def cohens_d(a, b) -> float:
    xa = pd.to_numeric(pd.Series(a), errors="coerce").to_numpy(dtype=float)
    xb = pd.to_numeric(pd.Series(b), errors="coerce").to_numpy(dtype=float)
    xa = xa[np.isfinite(xa)]
    xb = xb[np.isfinite(xb)]
    if len(xa) < 2 or len(xb) < 2:
        return np.nan
    pooled = np.sqrt(((len(xa) - 1) * np.var(xa, ddof=1) + (len(xb) - 1) * np.var(xb, ddof=1)) / max(len(xa) + len(xb) - 2, 1))
    if not np.isfinite(pooled) or pooled < 1e-12:
        return np.nan
    return float((np.mean(xb) - np.mean(xa)) / pooled)



def detector_strength_summary(events_df: pd.DataFrame, task_key: str, pos_label: str, pos_mask: pd.Series = None) -> Dict[str, str]:
    strength_col = f"{task_key}_strength"
    if strength_col not in events_df.columns:
        return {"dominant_detector_strength": "unknown", "detector_strength_counts": "unavailable", "proxy_fraction": np.nan}
    pos_rows = events_df[pos_mask] if pos_mask is not None else events_df[events_df[task_key].astype(str) == pos_label]
    if len(pos_rows) == 0:
        return {"dominant_detector_strength": "unknown", "detector_strength_counts": "{}", "proxy_fraction": np.nan}
    counts = pos_rows[strength_col].fillna("unknown").astype(str).value_counts().to_dict()
    dominant = max(counts, key=counts.get) if counts else "unknown"
    proxy_count = int(counts.get("proxy", 0) + counts.get("weak_proxy", 0))
    return {
        "dominant_detector_strength": dominant,
        "detector_strength_counts": json.dumps({str(k): int(v) for k, v in counts.items()}, sort_keys=True),
        "proxy_fraction": float(proxy_count / max(len(pos_rows), 1)),
    }

def task_validity(events_df: pd.DataFrame, task_key: str) -> Dict:
    pos, neg = TASK_SPECS[task_key]
    vals = events_df[task_key].astype(str)
    pos_count = int((vals == pos).sum())
    neg_count = int((vals == neg).sum())
    unk_count = int((vals == "unknown").sum())
    denom = pos_count + neg_count
    positive_ratio = float(pos_count / denom) if denom else 0.0
    if unk_count == len(events_df):
        validity = "all_unknown"
    elif positive_ratio < 0.01 or positive_ratio > 0.95:
        validity = "degenerate"
    else:
        validity = "valid"
    return {"positive_count": pos_count, "negative_count": neg_count, "unknown_count": unk_count, "positive_ratio": positive_ratio, "event_validity": validity}


def ensure_global_alignment(events_df: pd.DataFrame, metrics_df: pd.DataFrame, n_embeddings: int):
    for name, df in [("behavior_event_bins_path", events_df), ("behavior_event_metrics_path", metrics_df)]:
        if "global_row" not in df.columns:
            raise ValueError(f"{name} is missing required column global_row")
        if df["global_row"].duplicated().any():
            dup = df.loc[df["global_row"].duplicated(), "global_row"].head().tolist()
            raise ValueError(f"{name} has duplicate global_row values, examples={dup}")
    if len(events_df) != len(metrics_df):
        raise ValueError(f"behavior_event_bins and behavior_event_metrics row counts differ: {len(events_df)} vs {len(metrics_df)}")
    if not events_df["global_row"].equals(metrics_df["global_row"]):
        raise ValueError("behavior_event_bins_v2.csv and behavior_event_metrics_v2.csv are not aligned by global_row")
    max_row = int(max(events_df["global_row"].max(), metrics_df["global_row"].max())) if len(events_df) else -1
    if max_row >= n_embeddings:
        raise ValueError(f"global_row exceeds embedding row count: max_global_row={max_row}, embeddings={n_embeddings}")


def top_case_rows(task_key: str, task_value: str, task_global_rows: np.ndarray, group_name: str, group_rows: np.ndarray, emb: np.ndarray, opposite_centroid: np.ndarray, metrics_df: pd.DataFrame, dominant_metrics: str, top_k: int) -> List[Dict]:
    if len(group_rows) == 0:
        return []
    z = emb[group_rows]
    dist = np.sqrt(np.sum((z - opposite_centroid.reshape(1, -1)) ** 2, axis=1))
    order = np.argsort(-dist)[:top_k]
    rows = []
    metrics_index = metrics_df.set_index("global_row", drop=False)
    for i in order:
        gr = int(group_rows[i])
        meta = metrics_index.loc[gr] if gr in metrics_index.index else pd.Series(dtype=object)
        row = {
            "global_row": gr,
            "source_group": group_name,
            "task_key": task_key,
            "task_value": task_value,
            "embedding_distance_to_opposite_centroid": float(dist[i]),
            "dominant_task_metrics": dominant_metrics,
        }
        for col in ["shard_id", "local_row", "scenario_id", "target_agent_id", "start", "window_len", "split"]:
            row[col] = meta.get(col, np.nan)
        rows.append(row)
    return rows


def plot_outputs(out: Path, bdd_df: pd.DataFrame, delta_df: pd.DataFrame):
    plot_dir = out / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if len(bdd_df):
        top = bdd_df.sort_values("bdd_mmd", ascending=False)
        plt.figure(figsize=(max(8, len(top) * 0.8), 4))
        plt.bar(top["task_key"], top["bdd_mmd"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("BDD MMD")
        plt.title("Task-conditioned behavior-event BDD")
        plt.tight_layout()
        plt.savefig(plot_dir / "task_bdd_bar.png", dpi=150)
        plt.close()
    else:
        (plot_dir / "task_bdd_bar.png").write_text("No valid task BDD rows to plot.\n", encoding="utf-8")
    if len(delta_df):
        top_delta = delta_df.reindex(delta_df["effect_size"].abs().sort_values(ascending=False).index).head(20)
        labels = [f"{r.task_key}\n{r.metric}" for r in top_delta.itertuples()]
        plt.figure(figsize=(max(10, len(top_delta) * 0.6), 5))
        plt.bar(labels, top_delta["delta_B_minus_A"])
        plt.xticks(rotation=60, ha="right")
        plt.ylabel("B - A metric delta")
        plt.title("Task-specific style metric deltas")
        plt.tight_layout()
        plt.savefig(plot_dir / "task_style_delta_bar.png", dpi=150)
        plt.close()
    else:
        (plot_dir / "task_style_delta_bar.png").write_text("No style metric delta rows to plot.\n", encoding="utf-8")


def write_report(out: Path, bdd_df: pd.DataFrame, delta_df: pd.DataFrame, skipped: List[Dict], warnings: List[Dict]):
    lines = [
        "# Stage 6C v2 task-conditioned behavior-event BDD report",
        "",
        "BDD detects distribution shift in learned embedding space. Task-specific metrics explain the drift direction.",
        "",
        "本报告的主评价单元是 driving task / behavior-event slice 内的 BDD；hard_brake、late_brake 等 outcome-style 表现只应作为可选 post-hoc 诊断，而不是主结果。",
        "",
        "## Task BDD summary",
        "",
    ]
    if len(bdd_df):
        lines.extend(["| task_key | task_value | strength_filter | detector_strength | detector_strength_counts | n_A(before) | n_B(before) | n_A(after) | n_B(after) | BDD_MMD | bootstrap_mean | bootstrap_std | in_CI | CI95 | p_value | interpretation |", "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|"])
        for row in bdd_df.sort_values("bdd_mmd", ascending=False).itertuples():
            lines.append(f"| {row.task_key} | {row.task_value} | {row.detector_strength_filter} | {row.dominant_detector_strength} | {row.detector_strength_counts} | {row.n_A_before_strength_filter} | {row.n_B_before_strength_filter} | {row.n_A} | {row.n_B} | {row.bdd_mmd:.6g} | {row.bootstrap_mean:.6g} | {row.bootstrap_std:.6g} | {row.observed_in_bootstrap_ci} | [{row.ci95_low:.6g}, {row.ci95_high:.6g}] | {row.p_value:.6g} | {row.interpretation} |")
    else:
        lines.append("- No task passed the min_bin_size / validity filters.")
    lines.extend(["", "## Style metric explanation layer", ""])
    if len(delta_df):
        lines.extend(["| task_key | metric | n_A | n_B | mean_A | mean_B | B_minus_A | effect_size |", "|---|---|---:|---:|---:|---:|---:|---:|"])
        for row in delta_df.reindex(delta_df["effect_size"].abs().sort_values(ascending=False).index).head(40).itertuples():
            lines.append(f"| {row.task_key} | {row.metric} | {row.n_A_valid} | {row.n_B_valid} | {row.mean_A:.6g} | {row.mean_B:.6g} | {row.delta_B_minus_A:.6g} | {row.effect_size:.6g} |")
    else:
        lines.append("- No valid task-specific metric deltas were available.")
    metric_quality = [w for w in warnings if w.get("warning") in {"metric_physical_range_warning", "raw_metric_physically_implausible", "physical_metric_clipping_applied"}]
    lines.extend(["", "## Metric quality warnings", ""])
    lines.extend(["- None"] if not metric_quality else [f"- {w.get('warning')}: {w}" for w in metric_quality[:80]])
    lines.extend(["", "## Skipped tasks", ""])
    lines.extend(["- None"] if not skipped else [f"- `{s['task_key']}`: {s['reason']} (n_A={s.get('n_A')}, n_B={s.get('n_B')}, validity={s.get('event_validity')})" for s in skipped])
    lines.extend(["", "## Interpretation guide", "", "- `negative_control_random`: sanity check; task BDD should be low and not systematic.", "- `pseudo_agg_vs_cons`: positive control; style drift should localize to relevant behavior tasks.", "- `scene_confounding_control`: confounding diagnosis; drift may concentrate where task exposure or dynamic interaction pressure differs.", "", "## Warnings", ""])
    lines.extend(["- None"] if not warnings else [f"- {w.get('warning')}: {w}" for w in warnings[:200]])
    (out / "task_report_card.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    out = Path(args.output_dir)
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"output_dir exists: {out}; use --overwrite")
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    (out / "plots").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    warnings: List[Dict] = []
    emb, emb_meta = load_embeddings(args.shard_manifest, args.embedding_manifest, progress_enabled=not args.no_progress)
    events_df = pd.read_csv(args.behavior_event_bins_path)
    metrics_df = pd.read_csv(args.behavior_event_metrics_path)
    behavior_schema_path = Path(args.behavior_event_bins_path).parent / "behavior_event_schema_v2.json"
    if behavior_schema_path.exists():
        try:
            behavior_schema = json.loads(behavior_schema_path.read_text(encoding="utf-8"))
            for warning in behavior_schema.get("metric_quality_warnings", []):
                warnings.append(warning)
        except Exception as exc:
            warnings.append({"warning": "behavior_event_schema_metric_quality_load_failed", "path": str(behavior_schema_path), "detail": str(exc)})
    ensure_global_alignment(events_df, metrics_df, len(emb))
    if args.feature_schema_path and not Path(args.feature_schema_path).exists():
        raise FileNotFoundError(f"feature_schema_path does not exist: {args.feature_schema_path}")

    a_idx = np.asarray(np.load(args.a_indices_path), dtype=np.int64)
    b_idx = np.asarray(np.load(args.b_indices_path), dtype=np.int64)
    if a_idx.size == 0 or b_idx.size == 0:
        raise ValueError("A/B index arrays must both be non-empty")
    if max(int(a_idx.max()), int(b_idx.max())) >= len(emb):
        raise ValueError(f"A/B index exceeds embedding row count: embeddings={len(emb)}")

    task_keys = DEFAULT_TASK_KEYS if not args.task_keys else [x.strip() for x in args.task_keys.split(",") if x.strip()]
    unknown_tasks = [t for t in task_keys if t not in TASK_SPECS]
    if unknown_tasks:
        raise ValueError(f"Unknown task_keys: {unknown_tasks}; valid keys={DEFAULT_TASK_KEYS}")

    global_to_pos = pd.Series(np.arange(len(events_df), dtype=np.int64), index=events_df["global_row"].astype(np.int64))
    a_set = set(int(x) for x in a_idx.tolist())
    b_set = set(int(x) for x in b_idx.tolist())
    bdd_rows: List[Dict] = []
    delta_rows: List[Dict] = []
    skipped: List[Dict] = []
    top_rows: List[Dict] = []

    for task_key in iter_progress(task_keys, enabled=not args.no_progress, desc="task-conditioned BDD", unit="task"):
        if task_key not in events_df.columns:
            warnings.append({"warning": "task_column_missing", "task_key": task_key})
            skipped.append({"task_key": task_key, "reason": "task_column_missing"})
            continue
        pos_label, _ = TASK_SPECS[task_key]
        validity = task_validity(events_df, task_key)
        pos_mask = events_df[task_key].astype(str) == pos_label
        n_positive_before_strength_filter = int(pos_mask.sum())
        strength_col = f"{task_key}_strength"
        if args.detector_strength_filter == "strong":
            if strength_col not in events_df.columns:
                warnings.append({"warning": "detector_strength_filter_column_missing", "task_key": task_key, "filter": args.detector_strength_filter})
                pos_mask = pd.Series(False, index=events_df.index)
            else:
                pos_mask = pos_mask & (events_df[strength_col].fillna("unknown").astype(str) == "strong")
        elif args.detector_strength_filter == "strong_or_proxy":
            if strength_col not in events_df.columns:
                warnings.append({"warning": "detector_strength_filter_column_missing", "task_key": task_key, "filter": args.detector_strength_filter})
                pos_mask = pd.Series(False, index=events_df.index)
            else:
                pos_mask = pos_mask & events_df[strength_col].fillna("unknown").astype(str).isin(["strong", "proxy"])
        task_global_rows = events_df.loc[pos_mask, "global_row"].astype(np.int64).to_numpy()
        task_set = set(int(x) for x in task_global_rows.tolist())
        ai = np.asarray(sorted(task_set & a_set), dtype=np.int64)
        bi = np.asarray(sorted(task_set & b_set), dtype=np.int64)
        n_A_before_strength_filter = int(len(np.asarray(sorted(set(int(x) for x in events_df.loc[events_df[task_key].astype(str) == pos_label, "global_row"].astype(np.int64).tolist()) & a_set), dtype=np.int64)))
        n_B_before_strength_filter = int(len(np.asarray(sorted(set(int(x) for x in events_df.loc[events_df[task_key].astype(str) == pos_label, "global_row"].astype(np.int64).tolist()) & b_set), dtype=np.int64)))
        if validity["event_validity"] != "valid" and not args.include_degenerate_tasks:
            skipped.append({"task_key": task_key, "reason": "degenerate_or_all_unknown_skipped", "n_A": int(len(ai)), "n_B": int(len(bi)), **validity})
            warnings.append({"warning": "task_skipped_degenerate_or_all_unknown", "task_key": task_key, **validity})
            continue
        if len(ai) < args.min_bin_size or len(bi) < args.min_bin_size:
            skipped.append({"task_key": task_key, "reason": "below_min_bin_size", "n_A": int(len(ai)), "n_B": int(len(bi)), "n_A_before_strength_filter": n_A_before_strength_filter, "n_B_before_strength_filter": n_B_before_strength_filter, "detector_strength_filter": args.detector_strength_filter, **validity})
            continue
        stats = mmd_with_stats(emb[ai], emb[bi], rng, args.num_bootstrap, args.num_permutation, args.max_mmd_samples)
        interp = "Task-conditioned BDD is computed within the same behavior-event slice; inspect task-specific metrics below for semantic drift direction."
        strength_summary = detector_strength_summary(events_df, task_key, pos_label, pos_mask)
        if strength_summary["dominant_detector_strength"] in {"proxy", "weak_proxy"} or (np.isfinite(strength_summary["proxy_fraction"]) and strength_summary["proxy_fraction"] > 0.5):
            warnings.append({
                "warning": "task_bdd_uses_proxy_detector",
                "task_key": task_key,
                "dominant_detector_strength": strength_summary["dominant_detector_strength"],
                "detector_strength_counts": strength_summary["detector_strength_counts"],
                "proxy_fraction": strength_summary["proxy_fraction"],
            })
        bdd_rows.append({
            "task_key": task_key,
            "task_value": pos_label,
            "n_A": int(len(ai)),
            "n_B": int(len(bi)),
            "n_A_before_strength_filter": n_A_before_strength_filter,
            "n_B_before_strength_filter": n_B_before_strength_filter,
            "n_positive_before_strength_filter": n_positive_before_strength_filter,
            "n_positive_after_strength_filter": int(len(task_global_rows)),
            "detector_strength_filter": args.detector_strength_filter,
            "bdd_mmd": stats["mmd2"],
            "ci95_low": stats["ci95_low"],
            "ci95_high": stats["ci95_high"],
            "bootstrap_mean": stats.get("bootstrap_mean", np.nan),
            "bootstrap_std": stats.get("bootstrap_std", np.nan),
            "observed_in_bootstrap_ci": stats.get("observed_in_bootstrap_ci", False),
            "mmd_estimator_config": json.dumps(stats.get("mmd_estimator_config", {}), sort_keys=True),
            "p_value": stats["p_value"],
            "positive_count_total": validity["positive_count"],
            "positive_ratio_total": validity["positive_ratio"],
            "event_validity": validity["event_validity"],
            "dominant_detector_strength": strength_summary["dominant_detector_strength"],
            "detector_strength_counts": strength_summary["detector_strength_counts"],
            "interpretation": interp,
        })

        if not stats.get("observed_in_bootstrap_ci", False):
            warnings.append({"warning": "observed_bdd_outside_bootstrap_ci", "task_key": task_key, "bdd_mmd": stats["mmd2"], "ci95_low": stats["ci95_low"], "ci95_high": stats["ci95_high"], "mmd_estimator_config": stats.get("mmd_estimator_config", {})})
        metric_cols = relevant_metrics(task_key, metrics_df.columns)
        dominant_parts = []
        for metric in metric_cols:
            a_pos = global_to_pos.loc[ai].to_numpy(dtype=np.int64)
            b_pos = global_to_pos.loc[bi].to_numpy(dtype=np.int64)
            avals = pd.to_numeric(metrics_df.iloc[a_pos][metric], errors="coerce")
            bvals = pd.to_numeric(metrics_df.iloc[b_pos][metric], errors="coerce")
            mean_a = finite_mean(avals)
            mean_b = finite_mean(bvals)
            delta = float(mean_b - mean_a) if np.isfinite(mean_a) and np.isfinite(mean_b) else np.nan
            eff = cohens_d(avals, bvals)
            n_a_valid = int(np.isfinite(pd.to_numeric(avals, errors="coerce")).sum())
            n_b_valid = int(np.isfinite(pd.to_numeric(bvals, errors="coerce")).sum())
            delta_rows.append({
                "task_key": task_key,
                "task_value": pos_label,
                "metric": metric,
                "n_A_valid": n_a_valid,
                "n_B_valid": n_b_valid,
                "mean_A": mean_a,
                "mean_B": mean_b,
                "delta_B_minus_A": delta,
                "effect_size": eff,
            })
            if np.isfinite(eff):
                dominant_parts.append((abs(eff), f"{metric}: Δ={delta:.4g}, d={eff:.3g}"))
        dominant_metrics = "; ".join(x[1] for x in sorted(dominant_parts, reverse=True)[:3])
        if not dominant_metrics:
            dominant_metrics = "no valid task-specific metric delta"
        centroid_a = np.mean(emb[ai], axis=0)
        centroid_b = np.mean(emb[bi], axis=0)
        top_rows.extend(top_case_rows(task_key, pos_label, task_global_rows, "A", ai, emb, centroid_b, metrics_df, dominant_metrics, max(1, args.top_k // 2)))
        top_rows.extend(top_case_rows(task_key, pos_label, task_global_rows, "B", bi, emb, centroid_a, metrics_df, dominant_metrics, max(1, args.top_k // 2)))

    bdd_df = pd.DataFrame(bdd_rows)
    delta_df = pd.DataFrame(delta_rows)
    top_df = pd.DataFrame(top_rows)
    if len(top_df):
        top_df = top_df.sort_values("embedding_distance_to_opposite_centroid", ascending=False).head(args.top_k)
    for df, name in [(bdd_df, "task_bdd_summary.csv"), (delta_df, "task_style_delta.csv"), (top_df, "top_task_drift_cases.csv")]:
        df.to_csv(out / name, index=False)
    warnings.append({"warning": "completed", "valid_task_count": int(len(bdd_df)), "skipped_task_count": int(len(skipped)), "embedding_rows": int(len(emb)), "event_rows": int(len(events_df))})
    write_json(out / "warnings.json", {"warnings": warnings, "skipped_tasks": skipped})
    plot_outputs(out, bdd_df, delta_df)
    write_report(out, bdd_df, delta_df, skipped, warnings)


def parse_args():
    p = argparse.ArgumentParser(description="Compute Stage 6C v2 task-conditioned behavior-event BDD report.")
    p.add_argument("--embedding_manifest", required=True, help="Path to embedding_manifest.json.")
    p.add_argument("--shard_manifest", required=True, help="Path to sharded dataset manifest JSON.")
    p.add_argument("--feature_schema_path", required=True, help="Path to feature_schema.json for provenance/validation.")
    p.add_argument("--a_indices_path", required=True, help="Path to A global-row indices .npy.")
    p.add_argument("--b_indices_path", required=True, help="Path to B global-row indices .npy.")
    p.add_argument("--behavior_event_bins_path", required=True, help="Path to behavior_event_bins_v2.csv.")
    p.add_argument("--behavior_event_metrics_path", required=True, help="Path to behavior_event_metrics_v2.csv.")
    p.add_argument("--output_dir", required=True, help="Output directory for task-conditioned BDD report.")
    p.add_argument("--task_keys", default="", help="Optional comma-separated task keys; defaults to all Stage 6C v2 tasks.")
    p.add_argument("--num_bootstrap", type=int, default=50, help="Number of bootstrap samples for BDD CI.")
    p.add_argument("--num_permutation", type=int, default=100, help="Number of permutations for BDD p-value.")
    p.add_argument("--max_mmd_samples", type=int, default=2000, help="Max samples per group for MMD computation.")
    p.add_argument("--min_bin_size", type=int, default=100, help="Minimum A and B rows in positive task bin.")
    p.add_argument("--top_k", type=int, default=20, help="Top drift cases to export.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output_dir if it exists.")
    p.add_argument("--no_progress", action="store_true", help="Disable progress bars.")
    p.add_argument("--include_degenerate_tasks", action="store_true", help="Include degenerate/all_unknown tasks instead of skipping them by default.")
    p.add_argument("--detector_strength_filter", choices=["all", "strong", "strong_or_proxy"], default="all", help="Filter positive task rows by detector strength before BDD; all keeps existing behavior.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
