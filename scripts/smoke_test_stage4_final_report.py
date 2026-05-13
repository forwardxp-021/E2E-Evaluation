#!/usr/bin/env python3
import json
import subprocess
import tempfile
from pathlib import Path


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        out_dir = root / "out"
        comp = root / "comparison.csv"
        comp.write_text(
            "run,centroid_accuracy_overall,hit_at_1,mean_same_label_fraction_topk,spearman_rms_jerk_delta,spearman_rms_yaw_rate_delta,spearman_rms_curvature_delta,spearman_mean_speed_delta\n"
            "stage4d_v1,0.45,0.41,0.43,0.20,0.30,0.31,0.10\n"
            "stage4e_jerk_comfort,0.46,0.40,0.42,0.21,0.31,0.32,0.11\n"
            "stage4f_comfort_aux,0.47,0.42,0.44,0.22,0.33,0.34,0.12\n"
            "stage4g_comfort_metric,0.70,0.69,0.68,0.72,0.60,0.58,0.25\n"
            "stage4h_metric_shuffled,0.50,0.48,0.49,0.10,0.62,0.59,0.20\n"
        )
        eval_dir = root / "eval"
        (eval_dir / "baseline_comparison_summary.csv").parent.mkdir(parents=True, exist_ok=True)
        (eval_dir / "baseline_comparison_summary.csv").write_text(
            "method,centroid_accuracy_overall,hit_at_1,mean_same_label_fraction_topk\n"
            "learned,0.7,0.69,0.68\nraw_feature,0.6,0.58,0.57\ntrajectory_l2,0.5,0.49,0.48\nrandom,0.2,0.2,0.2\npca_feature,0.55,0.54,0.53\n"
        )
        aux_path = root / "aux.json"
        write_json(aux_path, {
            "metrics": {
                "rms_accel": {"mae": 0.1, "rmse": 0.2, "spearman": 0.5, "valid_pairs": 100},
                "rms_jerk": {"mae": 0.11, "rmse": 0.21, "spearman": 0.723, "valid_pairs": 101},
                "max_abs_accel": {"mae": 0.12, "rmse": 0.22, "spearman": 0.51, "valid_pairs": 102},
                "max_abs_jerk": {"mae": 0.13, "rmse": 0.23, "spearman": 0.700, "valid_pairs": 103},
                "mean_thw": {"mae": 0.14, "rmse": 0.24, "spearman": 0.616, "valid_pairs": 104},
                "min_thw": {"mae": 0.15, "rmse": 0.25, "spearman": 0.52, "valid_pairs": 105}
            }
        })
        write_json(root / "build.json", {"n_files_processed": 1, "n_scenarios_processed": 1, "n_windows_kept": 1, "split_counts": {"train": 1}})
        write_json(root / "pseudo.json", {"n_labeled": 1, "n_unlabeled": 0})

        subprocess.run([
            "python", "tools/generate_stage4_final_report.py",
            "--out_dir", str(out_dir),
            "--comparison_csv", str(comp),
            "--stage4g_eval_dir", str(eval_dir),
            "--stage4g_aux_json", str(aux_path),
            "--build_summary", str(root / "build.json"),
            "--pseudo_label_summary", str(root / "pseudo.json"),
        ], check=True)

        aux_md = (out_dir / "table_stage4g_aux_prediction.md").read_text()
        report_md = (out_dir / "stage4_final_report.md").read_text()
        sanity_md = (out_dir / "table_stage4h_sanity_check.md").read_text()

        assert "nan" not in aux_md.lower(), aux_md
        assert "0.723" in report_md, report_md
        assert "Shuffling comfort metric target collapses jerk-sensitive geometry" in sanity_md, sanity_md
        assert "Not the primary sanity-check target" in sanity_md, sanity_md
        print("Smoke test passed")


if __name__ == "__main__":
    main()
