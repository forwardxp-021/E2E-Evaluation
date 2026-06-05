#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tools.stage6b_compare_baselines import mmd_with_stats
from tools.stage6c_common import iter_progress, load_embeddings, load_schema_names, write_json

DEFAULT_EVENT_KEYS = [
    "exposure_following",
    "exposure_cut_in",
    "exposure_overtake_opportunity",
    "exposure_dense_traffic",
    "exposure_front_pressure",
    "exposure_side_pressure",
    "exposure_gap_pressure",
    "exposure_yield_conflict",
    "exposure_free_cruising",
]

EVENT_METRIC_PREFIXES = {
    "exposure_following": ["following_", "peak_decel", "jerk_p95", "hard_brake_score"],
    "exposure_cut_in": ["cutin_", "peak_decel", "jerk_p95"],
    "exposure_overtake_opportunity": ["overtake_", "lc_", "assertiveness_score"],
    "exposure_dense_traffic": ["gap_pressure_score", "yielding_score", "assertiveness_score", "hard_brake_score"],
    "exposure_front_pressure": ["following_", "gap_pressure_score", "peak_decel", "jerk_p95"],
    "exposure_side_pressure": ["lc_", "gap_pressure_score", "assertiveness_score"],
    "exposure_gap_pressure": ["gap_pressure_score", "assertiveness_score", "small_gap_speed_maintain_score", "hard_brake_score"],
    "exposure_yield_conflict": ["yielding_score", "assertiveness_score", "conflict_accel_score", "gap_pressure_score"],
    "exposure_free_cruising": ["cruise_"],
}


def metric_columns_for(event_key, metrics_df):
    non_metrics = {"global_row", "shard_id", "local_row"}
    cols = [c for c in metrics_df.columns if c not in non_metrics]
    prefixes = EVENT_METRIC_PREFIXES.get(event_key)
    if not prefixes:
        return cols
    selected = []
    for c in cols:
        if any(c.startswith(p) or c == p for p in prefixes):
            selected.append(c)
    return selected or cols


def interpretation_for_delta(event_key, summary, deltas):
    def delta(metric):
        r = deltas[deltas["metric_name"] == metric]
        if r.empty:
            return np.nan
        return float(r.iloc[0]["delta_B_minus_A"])

    text = []
    if event_key == "exposure_following":
        if delta("following_min_thw") < 0 and delta("following_min_front_distance") < 0 and (delta("following_peak_decel") > 0 or delta("following_jerk_p95") > 0):
            text.append("Model B is closer-following and more abrupt in braking under following exposure.")
    elif event_key == "exposure_cut_in":
        if delta("cutin_min_ttc_proxy") < 0 and delta("cutin_peak_decel_proxy") > 0:
            text.append("Model B reacts later and brakes harder after cut-in exposure.")
    elif event_key in {"exposure_side_pressure", "exposure_overtake_opportunity"}:
        if (delta("lc_rms_yaw_rate") > 0 or delta("lc_lateral_sharpness_score") > 0) and (delta("lc_duration_proxy") < 0 or delta("lc_min_front_gap") < 0 or delta("lc_min_rear_gap") < 0):
            text.append("Model B performs sharper and more assertive lane changes with smaller accepted gaps.")
    if event_key == "exposure_overtake_opportunity":
        if delta("overtake_execution_score") > 0 and delta("overtake_peak_accel") > 0:
            text.append("Model B is more willing to overtake and uses stronger acceleration during overtake.")
    if event_key in {"exposure_gap_pressure", "exposure_yield_conflict", "exposure_dense_traffic"}:
        if delta("assertiveness_score") > 0 and (delta("yielding_score") < 0 or delta("gap_pressure_score") > 0):
            text.append("Model B is more assertive under interaction pressure.")
    if event_key == "exposure_free_cruising" and float(summary.get("bdd_mmd", np.nan)) < 0.01:
        text.append("Basic cruising behavior remains stable; drift is concentrated in interaction events.")
    if not text:
        bdd = summary.get("bdd_mmd", np.nan)
        if np.isfinite(bdd):
            text.append("Event-specific BDD and metric deltas were computed; inspect event_style_delta.csv for semantic direction.")
        else:
            text.append("Insufficient valid rows or embeddings for event-specific BDD.")
    return " ".join(text)


def scalar_delta_interpretation(metric, delta):
    if not np.isfinite(delta):
        return "insufficient_valid_values"
    if abs(delta) < 1e-9:
        return "no_mean_shift"
    direction = "higher_in_B" if delta > 0 else "lower_in_B"
    if any(k in metric for k in ["jerk", "decel", "pressure", "assert", "sharp", "oscillation", "hesitation"]):
        return f"{direction}; larger values usually indicate stronger interaction response or lower comfort for this proxy"
    if any(k in metric for k in ["thw", "gap", "distance", "ttc"]):
        return f"{direction}; smaller values usually indicate reduced margin for this proxy"
    return direction


def effect_size_from_embeddings(xa, xb):
    ca = np.mean(xa, axis=0)
    cb = np.mean(xb, axis=0)
    pooled = np.sqrt(0.5 * (np.mean(np.sum((xa - ca) ** 2, axis=1)) + np.mean(np.sum((xb - cb) ** 2, axis=1))))
    return float(np.linalg.norm(cb - ca) / max(pooled, 1e-9))


def top_cases_for_event(event_key, event_value, ai, bi, emb, metrics, bins, top_k):
    rows = []
    if len(ai) == 0 or len(bi) == 0:
        return rows
    ca = np.mean(emb[ai], axis=0)
    cb = np.mean(emb[bi], axis=0)
    metric_cols = metric_columns_for(event_key, metrics)
    mean_a = metrics.loc[metrics.global_row.isin(ai), metric_cols].mean(numeric_only=True)
    mean_b = metrics.loc[metrics.global_row.isin(bi), metric_cols].mean(numeric_only=True)
    delta_abs = (mean_b - mean_a).abs().sort_values(ascending=False)
    dominant = ";".join(delta_abs.head(5).index.tolist())
    for group, idx, centroid in [("A", ai, cb), ("B", bi, ca)]:
        d = np.linalg.norm(emb[idx] - centroid, axis=1)
        order = np.argsort(-d)[:top_k]
        sub_meta = bins.set_index("global_row", drop=False)
        for pos in order:
            gr = int(idx[pos])
            m = sub_meta.loc[gr] if gr in sub_meta.index else {}
            row = {
                "global_row": gr,
                "source_group": group,
                "event_key": event_key,
                "event_value": event_value,
                "embedding_distance_to_opposite_centroid": float(d[pos]),
                "dominant_style_metrics": dominant,
                "shard_id": int(m.get("shard_id", -1)) if hasattr(m, "get") else -1,
                "local_row": int(m.get("local_row", -1)) if hasattr(m, "get") else -1,
            }
            if hasattr(m, "get") and "scenario_id" in m:
                row["scenario_id"] = m.get("scenario_id")
            rows.append(row)
    return rows


def maybe_plot(out, bdd_df, delta_df):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return f"matplotlib unavailable; plots skipped: {exc}"
    plots = out / "plots"
    plots.mkdir(exist_ok=True)
    if not bdd_df.empty:
        top = bdd_df.sort_values("bdd_mmd", ascending=False).head(20)
        labels = [f"{r.event_key}\n{r.event_value}" for r in top.itertuples()]
        plt.figure(figsize=(max(8, len(top) * 0.55), 5))
        plt.bar(range(len(top)), top["bdd_mmd"])
        plt.xticks(range(len(top)), labels, rotation=80, ha="right", fontsize=7)
        plt.ylabel("BDD MMD")
        plt.tight_layout()
        plt.savefig(plots / "event_bdd_bar.png", dpi=180)
        plt.close()
    if not delta_df.empty:
        topd = delta_df.reindex(delta_df["delta_B_minus_A"].abs().sort_values(ascending=False).index).head(20)
        labels = [f"{r.event_key}\n{r.metric_name}" for r in topd.itertuples()]
        plt.figure(figsize=(max(8, len(topd) * 0.55), 5))
        plt.bar(range(len(topd)), topd["delta_B_minus_A"])
        plt.xticks(range(len(topd)), labels, rotation=80, ha="right", fontsize=7)
        plt.ylabel("Delta B - A")
        plt.tight_layout()
        plt.savefig(plots / "event_style_delta_bar.png", dpi=180)
        plt.close()
    return None


def main(args):
    t0 = time.time()
    out = Path(args.output_dir)
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"output_dir exists: {out}; use --overwrite")
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    progress_enabled = not args.no_progress
    _ = load_schema_names(args.feature_schema_path)  # validates path for clear diagnostics and records protocol input.
    emb, emb_meta = load_embeddings(args.shard_manifest, args.embedding_manifest, progress_enabled=progress_enabled)
    bins = pd.read_csv(args.dynamic_event_bins_path)
    metrics = pd.read_csv(args.event_style_metrics_path)
    a_idx = np.load(args.a_indices_path).astype(int)
    b_idx = np.load(args.b_indices_path).astype(int)
    rng = np.random.default_rng(args.seed)

    if not np.array_equal(bins["global_row"].to_numpy(), emb_meta["global_row"].to_numpy()):
        raise ValueError("dynamic_event_bins_path is not row-aligned with embedding global_row order")
    if not np.array_equal(metrics["global_row"].to_numpy(), emb_meta["global_row"].to_numpy()):
        raise ValueError("event_style_metrics_path is not row-aligned with embedding global_row order")

    event_keys = [x.strip() for x in args.event_keys.split(",") if x.strip()] if args.event_keys else DEFAULT_EVENT_KEYS
    missing_keys = [k for k in event_keys if k not in bins.columns]
    if missing_keys:
        raise ValueError(f"Requested event_keys missing from dynamic_event_bins_path: {missing_keys}")

    warnings = {"skipped_bins": [], "plot_warning": None}
    bdd_rows = []
    delta_rows = []
    top_rows = []
    bin_index = bins.set_index("global_row")

    for key in iter_progress(event_keys, enabled=progress_enabled, desc="computing event BDD", unit="event"):
        values = [v for v in bins[key].dropna().unique().tolist() if v != "unknown"]
        for val in iter_progress(values, enabled=progress_enabled, desc=f"{key} values", unit="value", leave=False):
            event_rows = bins.loc[bins[key] == val, "global_row"].to_numpy(dtype=int)
            ai = np.intersect1d(a_idx, event_rows, assume_unique=False)
            bi = np.intersect1d(b_idx, event_rows, assume_unique=False)
            if len(ai) < args.min_bin_size or len(bi) < args.min_bin_size:
                warnings["skipped_bins"].append({"event_key": key, "event_value": val, "n_A": int(len(ai)), "n_B": int(len(bi)), "reason": "below_min_bin_size"})
                continue
            st = mmd_with_stats(emb[ai], emb[bi], rng, args.num_bootstrap, args.num_permutation, args.max_mmd_samples)
            bdd_row = {
                "event_key": key,
                "event_value": val,
                "n_A": int(len(ai)),
                "n_B": int(len(bi)),
                "bdd_mmd": st["mmd2"],
                "ci95_low": st["ci95_low"],
                "ci95_high": st["ci95_high"],
                "p_value": st["p_value"],
                "effect_size": effect_size_from_embeddings(emb[ai], emb[bi]),
                "interpretation": "pending_metric_delta_interpretation",
                "warnings": "",
            }
            bdd_rows.append(bdd_row)

            metric_cols = metric_columns_for(key, metrics)
            event_delta_rows = []
            for metric in metric_cols:
                ma = metrics.loc[metrics.global_row.isin(ai), metric].to_numpy(dtype=float)
                mb = metrics.loc[metrics.global_row.isin(bi), metric].to_numpy(dtype=float)
                va = ma[np.isfinite(ma)]
                vb = mb[np.isfinite(mb)]
                mean_a = float(np.mean(va)) if len(va) else float("nan")
                mean_b = float(np.mean(vb)) if len(vb) else float("nan")
                delta = mean_b - mean_a if np.isfinite(mean_a) and np.isfinite(mean_b) else float("nan")
                rel = 100.0 * delta / max(abs(mean_a), 1e-9) if np.isfinite(delta) and np.isfinite(mean_a) else float("nan")
                row = {
                    "event_key": key,
                    "event_value": val,
                    "metric_name": metric,
                    "n_A_valid": int(len(va)),
                    "n_B_valid": int(len(vb)),
                    "mean_A": mean_a,
                    "mean_B": mean_b,
                    "delta_B_minus_A": float(delta),
                    "relative_delta_percent": float(rel),
                    "direction_label": "B_higher" if np.isfinite(delta) and delta > 0 else ("B_lower" if np.isfinite(delta) and delta < 0 else "no_valid_delta"),
                    "interpretation": scalar_delta_interpretation(metric, delta),
                }
                delta_rows.append(row)
                event_delta_rows.append(row)
            interp = interpretation_for_delta(key, bdd_row, pd.DataFrame(event_delta_rows))
            bdd_rows[-1]["interpretation"] = interp

            for _ in iter_progress([0], enabled=progress_enabled, desc="top-case retrieval", unit="event", leave=False):
                top_rows.extend(top_cases_for_event(key, val, ai, bi, emb, metrics, bins, args.top_k))

    bdd_df = pd.DataFrame(bdd_rows, columns=["event_key", "event_value", "n_A", "n_B", "bdd_mmd", "ci95_low", "ci95_high", "p_value", "effect_size", "interpretation", "warnings"])
    delta_df = pd.DataFrame(delta_rows, columns=["event_key", "event_value", "metric_name", "n_A_valid", "n_B_valid", "mean_A", "mean_B", "delta_B_minus_A", "relative_delta_percent", "direction_label", "interpretation"])
    top_df = pd.DataFrame(top_rows)
    bdd_df.to_csv(out / "event_bdd_summary.csv", index=False)
    delta_df.to_csv(out / "event_style_delta.csv", index=False)
    top_df.to_csv(out / "top_event_drift_cases.csv", index=False)
    warnings["plot_warning"] = maybe_plot(out, bdd_df, delta_df)
    write_json(out / "warnings.json", warnings)

    lines = ["# Stage 6C event style report", "", f"- n_A total: {len(a_idx)}", f"- n_B total: {len(b_idx)}", f"- events requested: {', '.join(event_keys)}", f"- runtime seconds: {time.time() - t0:.3f}", ""]
    lines += ["## Human-readable conclusions", ""]
    if bdd_df.empty:
        lines.append("No event bin satisfied the min_bin_size requirement; inspect warnings.json for skipped bins.")
    else:
        for r in bdd_df.sort_values("bdd_mmd", ascending=False).head(20).itertuples():
            lines.append(f"- `{r.event_key}={r.event_value}`: BDD={r.bdd_mmd:.6g}, n_A={r.n_A}, n_B={r.n_B}, p={r.p_value:.4g}. {r.interpretation}")
    lines += ["", "## Interpretation rule", "", "Embedding-based BDD provides a unified behavior distribution metric across heterogeneous driving events, while event-specific features provide semantic diagnosis of the detected drift.", "", "Exposure bins are candidates for dynamic matching/control. Outcome bins should mainly be used for reporting/localization rather than pure scenario control."]
    (out / "event_report_card.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate Stage 6C event-level BDD, style deltas, and top drift cases.")
    p.add_argument("--embedding_manifest", required=True)
    p.add_argument("--shard_manifest", required=True)
    p.add_argument("--feature_schema_path", required=True)
    p.add_argument("--a_indices_path", required=True)
    p.add_argument("--b_indices_path", required=True)
    p.add_argument("--dynamic_event_bins_path", required=True)
    p.add_argument("--event_style_metrics_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--event_keys")
    p.add_argument("--num_bootstrap", type=int, default=50)
    p.add_argument("--num_permutation", type=int, default=100)
    p.add_argument("--max_mmd_samples", type=int, default=2000)
    p.add_argument("--min_bin_size", type=int, default=100)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no_progress", action="store_true")
    main(p.parse_args())
