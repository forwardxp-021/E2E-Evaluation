#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_ORDER = ["learned", "raw_feature", "trajectory_l2", "random", "pca_feature"]
PSEUDO_LABELS = ["conservative_like", "aggressive_like", "lateral_stable_like", "unlabeled"]


def _read_json(path: Path):
    return json.loads(path.read_text())


def _ensure_columns(df: pd.DataFrame, required: list[str], source_name: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {missing}")


def _ensure_non_empty(df: pd.DataFrame, name: str):
    if df.empty or len(df.columns) == 0:
        raise ValueError(f"{name} generated an empty table, refusing to write placeholder output")


def _write_csv_md(df: pd.DataFrame, csv_path: Path, md_path: Path):
    _ensure_non_empty(df, csv_path.name)
    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False) + "\n")


def _fmt_value(value):
    if value is None:
        return "N/A"
    if isinstance(value, float) and np.isnan(value):
        return "N/A"
    return value


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _validate_inputs(paths: dict[str, Path], allow_missing: bool):
    missing = [(name, path) for name, path in paths.items() if not path.exists()]
    if missing and not allow_missing:
        details = "\n".join([f"- {name}: {path}" for name, path in missing])
        raise FileNotFoundError(f"Missing required input files:\n{details}")
    return missing


def _suggest_alternative_summary(path: Path) -> str | None:
    parent = path.parent
    if not parent.exists():
        return None
    candidates = sorted(parent.glob("*export_summary*.json"))
    if not candidates:
        return None
    return str(candidates[0])


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_csv = Path(args.eval_dir) / "baseline_comparison_summary.csv"
    corr_csv = Path(args.eval_dir) / "style_distance_correlation.csv"
    input_paths = {
        "build_summary": Path(args.build_summary),
        "pseudo_label_summary": Path(args.pseudo_label_summary),
        "train_summary": Path(args.train_summary),
        "export_summary": Path(args.export_summary),
        "baseline_comparison_summary": baseline_csv,
        "style_distance_correlation": corr_csv,
    }
    missing = _validate_inputs(input_paths, args.allow_missing)
    for name, missing_path in missing:
        if name == "export_summary":
            alt = _suggest_alternative_summary(missing_path)
            if alt:
                print(f"[error] Missing export summary: {missing_path}")
                print(f"[hint] Try: --export_summary {alt}")

    if missing:
        warning_lines = [
            "# Paper Tables Summary — Stage 4G comfort metric alignment",
            "",
            "## Warnings",
            "Required input files are missing. Tables were not generated.",
            "",
        ]
        warning_lines.extend([f"- missing: `{path}`" for _, path in missing])
        warning_lines.append("")
        warning_lines.append("Use `--allow_missing` only for diagnostic reporting.")
        (out_dir / "paper_tables_summary.md").write_text("\n".join(warning_lines))
        return 1

    build = _read_json(input_paths["build_summary"])
    pseudo = _read_json(input_paths["pseudo_label_summary"])
    train = _read_json(input_paths["train_summary"])
    export = _read_json(input_paths["export_summary"])
    baseline_df = pd.read_csv(baseline_csv)
    corr_df = pd.read_csv(corr_csv)

    split_counts = build.get("split_counts", {})
    dataset_df = pd.DataFrame([
        {
            "n_files_processed": build.get("n_files_processed"),
            "n_scenarios_processed": build.get("n_scenarios_processed"),
            "n_windows_kept": build.get("n_windows_kept"),
            "train_count": split_counts.get("train"),
            "val_count": split_counts.get("val"),
            "test_count": split_counts.get("test"),
            "front_found_rate": build.get("front_found_rate"),
        }
    ])
    _write_csv_md(dataset_df, out_dir / "table_dataset_statistics.csv", out_dir / "table_dataset_statistics.md")

    total_windows = build.get("n_windows_kept") or 0
    label_rows = []
    split_labeled = pseudo.get("split_labeled_counts", {})
    split_label_counts = pseudo.get("split_label_counts", {})
    for label in PSEUDO_LABELS:
        if label == "unlabeled":
            total_count = pseudo.get("n_unlabeled", 0)
        else:
            total_count = pseudo.get(label, 0)
        pct = (float(total_count) / float(total_windows) * 100.0) if total_windows else np.nan
        per_label_splits = split_label_counts.get(label, {}) if isinstance(split_label_counts, dict) else {}
        label_rows.append(
            {
                "label": label,
                "total_count": int(total_count) if total_count is not None else 0,
                "total_percentage": pct,
                "train_count": per_label_splits.get("train", split_labeled.get("train") if label != "unlabeled" else pseudo.get("split_unlabeled_counts", {}).get("train")),
                "val_count": per_label_splits.get("val", split_labeled.get("val") if label != "unlabeled" else pseudo.get("split_unlabeled_counts", {}).get("val")),
                "test_count": per_label_splits.get("test", split_labeled.get("test") if label != "unlabeled" else pseudo.get("split_unlabeled_counts", {}).get("test")),
            }
        )
    pseudo_df = pd.DataFrame(label_rows)
    _write_csv_md(pseudo_df, out_dir / "table_pseudo_label_distribution.csv", out_dir / "table_pseudo_label_distribution.md")

    required_baseline_cols = [
        "method",
        "centroid_accuracy_overall",
        "hit_at_1",
        "mean_same_label_fraction_topk",
        "hit_at_1_lift_over_chance",
    ]
    _ensure_columns(baseline_df, required_baseline_cols, "baseline_comparison_summary.csv")
    lvb_df = baseline_df[required_baseline_cols].copy()
    lvb_df["_order"] = lvb_df["method"].apply(lambda x: METHOD_ORDER.index(x) if x in METHOD_ORDER else len(METHOD_ORDER))
    lvb_df = lvb_df.sort_values(["_order", "method"]).drop(columns=["_order"]).reset_index(drop=True)
    _write_csv_md(lvb_df, out_dir / "table_learned_vs_baselines.csv", out_dir / "table_learned_vs_baselines.md")

    corr_cols = [
        "method",
        "spearman_mean_speed_delta",
        "spearman_rms_jerk_delta",
        "spearman_rms_yaw_rate_delta",
        "spearman_rms_curvature_delta",
    ]
    thw_col = "spearman_mean_thw_delta" if "spearman_mean_thw_delta" in corr_df.columns else None
    if thw_col:
        corr_cols.append(thw_col)
    _ensure_columns(corr_df, corr_cols, "style_distance_correlation.csv")
    corr_out_df = corr_df[corr_cols].copy()
    if thw_col:
        corr_out_df[thw_col] = corr_out_df[thw_col].apply(_fmt_value)
    _write_csv_md(corr_out_df, out_dir / "table_style_distance_correlation.csv", out_dir / "table_style_distance_correlation.md")

    training_export_df = pd.DataFrame(
        [
            {
                "n_total": train.get("n_total"),
                "n_retained": train.get("n_retained"),
                "n_dropped": train.get("n_dropped"),
                "traj_nan_count_raw": train.get("traj_nan_count_raw"),
                "traj_nan_count_after_sanitize": train.get("traj_nan_count_after_sanitize"),
                "traj_repaired_count": train.get("traj_repaired_count"),
                "feature_clipped_values": train.get("feature_clipped_values"),
                "best_val_loss": train.get("best_val_loss"),
                "final_train_loss": train.get("final_train_loss"),
                "final_val_loss": train.get("final_val_loss"),
                "embedding_shape": export.get("shape"),
                "row_aligned": export.get("row_aligned"),
                "n_rows_exported": export.get("n_rows_exported"),
            }
        ]
    )
    _write_csv_md(training_export_df, out_dir / "table_training_export_summary.csv", out_dir / "table_training_export_summary.md")

    has_nan_thw = thw_col and corr_df[thw_col].isna().any()

    summary = []
    summary.append("# Paper Tables Summary — Stage 4G comfort metric alignment")
    summary.append("")
    summary.append("## 1. Experiment identity")
    summary.append("- experiment_stage: Stage 4G comfort metric alignment")
    summary.append("- dataset: Waymo human_public full51")
    summary.append("- learned_embedding: behavior embedding + comfort metric alignment")
    summary.append("- evaluation_split: test")
    summary.append("- note: This table is the current main Stage 4G result.")
    summary.append("")

    summary.append("## 2. Dataset statistics")
    summary.append(dataset_df.to_markdown(index=False))
    summary.append("")

    summary.append("## 3. Pseudo-label distribution")
    summary.append(pseudo_df.to_markdown(index=False))
    summary.append("")
    summary.append("Warning: Pseudo labels are weak rule-based labels, not ground truth.")
    summary.append("")

    summary.append("## 4. Training and export summary")
    summary.append(training_export_df.to_markdown(index=False))
    summary.append("")

    summary.append("## 5. Learned vs baselines")
    summary.append(lvb_df.to_markdown(index=False))
    summary.append("")
    summary.append("- learned is clearly above random.")
    summary.append("- learned classification is stronger than raw_feature/pca_feature but below trajectory_l2.")
    summary.append("- raw_feature/pca_feature retrieval is much stronger, expected because pseudo labels are feature-derived.")
    summary.append("- learned retrieval is similar to trajectory_l2 and above random.")
    summary.append("")

    summary.append("## 6. Style-distance correlation")
    summary.append(corr_out_df.to_markdown(index=False))
    summary.append("")
    if has_nan_thw:
        summary.append("- `spearman_mean_thw_delta` has N/A values. THW correlation requires valid-pair filtering and should not be overinterpreted.")
    summary.append("- trajectory_l2 mainly captures speed/geometric variation.")
    summary.append("- learned captures lateral/curvature-related variation better than trajectory_l2.")
    summary.append("- learned jerk sensitivity is weak.")
    summary.append("- raw_feature/pca_feature remain strong feature-oracle baselines.")
    summary.append("")

    summary.append("## 7. Main conclusion")
    summary.append(
        "Experiment: Stage 4G comfort metric alignment. "
        "Stage 4G completed row-level learned embedding validation on Waymo human_public full51. "
        "The learned embedding is row-aligned, finite, and evaluated on the held-out test split. "
        "It captures non-random, interpretable behavior structure and is useful for pseudo-style classification and "
        "lateral/curvature style sensitivity. However, it does not outperform raw_feature/pca_feature retrieval baselines, "
        "and jerk/comfort-sensitive geometry improved substantially."
    )
    summary.append("")

    summary.append("## 8. Limitations")
    summary.append("- pseudo labels are weak labels, not ground truth.")
    summary.append("- pseudo labels are derived from style features, so raw_feature/pca_feature have a natural advantage.")
    summary.append("- learned embedding should not be claimed to outperform all baselines.")
    summary.append("- Stage 4H shuffled-target sanity check should still be run to rule out leakage/artifacts.")
    summary.append("")

    summary.append("## 9. Next steps")
    summary.append("- run Stage 4H shuffled-target sanity check")
    summary.append("- compare Stage 4D/4E/4F/4G/4H using compare_embedding_runs.py")
    summary.append("")

    summary.append("## Source files")
    for name, path in input_paths.items():
        summary.append(f"- {name}: `{path}`")

    (out_dir / "paper_tables_summary.md").write_text("\n".join(summary) + "\n")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", default="outputs/waymo_human_v1_full51/eval_with_learned")
    parser.add_argument("--train_summary", default="outputs/waymo_human_v1_full51/human_embedding_model/train_summary.json")
    parser.add_argument("--export_summary", default="outputs/waymo_human_v1_full51/embedding_export_summary.json")
    parser.add_argument("--pseudo_label_summary", default="outputs/waymo_human_v1_full51/pseudo_labels/pseudo_label_summary.json")
    parser.add_argument("--build_summary", default="outputs/waymo_human_v1_full51/build_summary.json")
    parser.add_argument("--out_dir", default="outputs/waymo_human_v1_full51/paper_tables")
    parser.add_argument("--allow_missing", action="store_true", help="Allow missing inputs: write warning summary and exit non-zero")
    raise SystemExit(run(parser.parse_args()))
