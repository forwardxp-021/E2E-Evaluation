#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BDD_COLUMNS = [
    "experiment",
    "task_key",
    "bdd_mmd",
    "p_value",
    "ci95_low",
    "ci95_high",
    "observed_in_bootstrap_ci",
    "dominant_detector_strength",
    "detector_strength_counts",
    "n_A",
    "n_B",
    "event_validity",
]

PRIMARY_TASKS = {
    "task_following",
    "task_lane_change",
    "task_yield_conflict",
    "task_hesitation",
}

AUXILIARY_PROXY_TASKS = {
    "task_cutin_response",
    "task_queue_approach",
    "task_lead_brake_response",
    "task_overtake_opportunity",
    "task_overtake_executed",
}


def parse_csv_arg(value: str, name: str):
    items = [x.strip() for x in str(value).split(",") if x.strip()]
    if not items:
        raise ValueError(f"{name} must contain at least one item")
    return items


def read_csv_or_empty(path: Path, warnings: list, label: str) -> pd.DataFrame:
    if not path.exists():
        warnings.append({"warning": "missing_csv", "label": label, "path": str(path)})
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        warnings.append({"warning": "empty_csv", "label": label, "path": str(path)})
        return pd.DataFrame()
    if df.empty:
        warnings.append({"warning": "empty_csv", "label": label, "path": str(path)})
    return df


def read_json_or_empty(path: Path, warnings: list, label: str):
    if not path.exists():
        warnings.append({"warning": "missing_json", "label": label, "path": str(path)})
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        warnings.append({"warning": "invalid_json", "label": label, "path": str(path), "error": str(exc)})
        return {}


def normalize_bdd_rows(df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=BDD_COLUMNS)
    out = pd.DataFrame(index=df.index)
    out["experiment"] = experiment
    for col in BDD_COLUMNS:
        if col == "experiment":
            continue
        out[col] = df[col] if col in df.columns else np.nan
    return out[BDD_COLUMNS]


def skipped_rows_from_warnings(obj, experiment: str) -> pd.DataFrame:
    rows = []
    for item in obj.get("skipped_tasks", []) if isinstance(obj, dict) else []:
        rows.append({
            "experiment": experiment,
            "task_key": item.get("task_key"),
            "bdd_mmd": np.nan,
            "p_value": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "observed_in_bootstrap_ci": np.nan,
            "dominant_detector_strength": "skipped",
            "detector_strength_counts": json.dumps({
                "reason": item.get("reason"),
                "positive_count": item.get("positive_count"),
                "negative_count": item.get("negative_count"),
                "unknown_count": item.get("unknown_count"),
            }, ensure_ascii=False),
            "n_A": item.get("n_A"),
            "n_B": item.get("n_B"),
            "event_validity": item.get("event_validity", item.get("reason", "skipped")),
        })
    return pd.DataFrame(rows, columns=BDD_COLUMNS)


def top_style_delta(df: pd.DataFrame, experiment: str, top_k: int) -> pd.DataFrame:
    if df.empty:
        cols = ["experiment", "task_key", "task_value", "metric", "n_A_valid", "n_B_valid", "mean_A", "mean_B", "delta_B_minus_A", "effect_size", "abs_effect_size"]
        return pd.DataFrame(columns=cols)
    out = df.copy()
    out.insert(0, "experiment", experiment)
    if "effect_size" in out.columns:
        out["abs_effect_size"] = pd.to_numeric(out["effect_size"], errors="coerce").abs()
    else:
        out["abs_effect_size"] = np.nan
    return out.sort_values("abs_effect_size", ascending=False).head(top_k)


def reliability_tier(task_key: str) -> str:
    if task_key in PRIMARY_TASKS:
        return "primary"
    if task_key in AUXILIARY_PROXY_TASKS:
        return "auxiliary_proxy"
    return "review_required"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    safe = df.copy()
    safe = safe.replace({np.nan: ""})
    columns = list(safe.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in safe.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown(out_path: Path, pivot: pd.DataFrame, delta: pd.DataFrame, warnings: list) -> None:
    lines = [
        "# Stage 6C v2 Cross-Experiment Summary",
        "",
        "## Interpretation Guide",
        "",
        "- `negative`: sanity check; task-conditioned BDD should be low and non-systematic.",
        "- `pseudo`: positive control; should show strong BDD in behavior-style tasks.",
        "- `scene`: confounding diagnosis; BDD may appear, but the pattern should differ from pseudo.",
        "",
        "## Reliability Tier",
        "",
        "- Primary tasks: `task_following`, `task_lane_change`, `task_yield_conflict`, `task_hesitation`.",
        "- Auxiliary proxy tasks: `task_cutin_response`, `task_queue_approach`, `task_lead_brake_response`, `task_overtake_opportunity`, `task_overtake_executed`.",
        "- Do not interpret skipped tasks, especially `task_overtake_executed` when it is skipped due to sample size.",
        "",
        "## BDD Pivot",
        "",
    ]
    lines.append(dataframe_to_markdown(pivot) if not pivot.empty else "No BDD pivot rows available.")
    lines.extend(["", "## Delta vs Negative", ""])
    lines.append(dataframe_to_markdown(delta) if not delta.empty else "Delta vs negative is unavailable until negative/pseudo/scene experiments are all present.")
    lines.extend(["", "## Warnings", ""])
    if warnings:
        for item in warnings:
            lines.append(f"- `{item.get('warning', 'warning')}`: {json.dumps(item, ensure_ascii=False)}")
    else:
        lines.append("- No summarizer warnings.")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    experiment_dirs = [Path(p) for p in parse_csv_arg(args.experiment_dirs, "--experiment_dirs")]
    experiment_names = parse_csv_arg(args.experiment_names, "--experiment_names")
    if len(experiment_dirs) != len(experiment_names):
        raise ValueError(
            f"--experiment_dirs and --experiment_names length mismatch: {len(experiment_dirs)} vs {len(experiment_names)}"
        )

    out_dir = Path(args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir exists and is not empty: {out_dir}. Use --overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings = []
    bdd_frames = []
    style_frames = []
    for exp_dir, exp_name in zip(experiment_dirs, experiment_names):
        if not exp_dir.exists():
            warnings.append({"warning": "missing_experiment_dir", "experiment": exp_name, "path": str(exp_dir)})
            continue
        bdd_df = read_csv_or_empty(exp_dir / "task_bdd_summary.csv", warnings, f"{exp_name}:task_bdd_summary")
        warnings_obj = read_json_or_empty(exp_dir / "warnings.json", warnings, f"{exp_name}:warnings")
        normalized = normalize_bdd_rows(bdd_df, exp_name)
        skipped = skipped_rows_from_warnings(warnings_obj, exp_name)
        bdd_frames.append(pd.concat([normalized, skipped], ignore_index=True))

        style_df = read_csv_or_empty(exp_dir / "task_style_delta.csv", warnings, f"{exp_name}:task_style_delta")
        style_frames.append(top_style_delta(style_df, exp_name, args.top_style_delta_k))

    if bdd_frames:
        bdd_all = pd.concat(bdd_frames, ignore_index=True)
    else:
        bdd_all = pd.DataFrame(columns=BDD_COLUMNS)
    bdd_all.to_csv(out_dir / "task_bdd_cross_experiment.csv", index=False)

    if bdd_all.empty:
        pivot = pd.DataFrame(columns=["task_key"])
    else:
        pivot = bdd_all.pivot_table(index="task_key", columns="experiment", values="bdd_mmd", aggfunc="first").reset_index()
    pivot.to_csv(out_dir / "task_bdd_pivot.csv", index=False)

    delta = build_delta_vs_negative(pivot)
    delta.to_csv(out_dir / "task_bdd_delta_vs_negative.csv", index=False)

    if style_frames:
        style_all = pd.concat(style_frames, ignore_index=True)
    else:
        style_all = pd.DataFrame()
    style_all.to_csv(out_dir / "top_style_delta_by_experiment.csv", index=False)

    write_markdown(out_dir / "stage6c_v2_cross_experiment_summary.md", pivot, delta, warnings)
    (out_dir / "summarizer_warnings.json").write_text(json.dumps({"warnings": warnings}, indent=2, ensure_ascii=False), encoding="utf-8")


def build_delta_vs_negative(pivot: pd.DataFrame) -> pd.DataFrame:
    if pivot.empty or "task_key" not in pivot.columns:
        return pd.DataFrame(columns=[
            "task_key",
            "bdd_negative",
            "bdd_pseudo",
            "bdd_scene",
            "pseudo_minus_negative",
            "scene_minus_negative",
            "strongest_experiment",
        ])
    out = pd.DataFrame()
    out["task_key"] = pivot["task_key"]
    out["bdd_negative"] = pivot["negative"] if "negative" in pivot.columns else np.nan
    out["bdd_pseudo"] = pivot["pseudo"] if "pseudo" in pivot.columns else np.nan
    out["bdd_scene"] = pivot["scene"] if "scene" in pivot.columns else np.nan
    out["pseudo_minus_negative"] = out["bdd_pseudo"] - out["bdd_negative"]
    out["scene_minus_negative"] = out["bdd_scene"] - out["bdd_negative"]
    value_cols = ["bdd_negative", "bdd_pseudo", "bdd_scene"]
    labels = {"bdd_negative": "negative", "bdd_pseudo": "pseudo", "bdd_scene": "scene"}

    def strongest(row):
        vals = row[value_cols].dropna()
        if vals.empty:
            return np.nan
        return labels[vals.idxmax()]

    out["strongest_experiment"] = out.apply(strongest, axis=1)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize Stage 6C v2 task-conditioned BDD experiments.")
    parser.add_argument("--experiment_dirs", required=True, help="Comma-separated result directories.")
    parser.add_argument("--experiment_names", required=True, help="Comma-separated names matching experiment_dirs.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_style_delta_k", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
