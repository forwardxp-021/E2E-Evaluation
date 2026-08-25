#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np


COMPARISONS = [
    ("idm_longitudinal_conservative", "idm_longitudinal_comfort", "conservative_vs_comfort"),
    ("idm_longitudinal_conservative", "idm_longitudinal_aggressive", "conservative_vs_aggressive"),
    ("idm_longitudinal_comfort", "idm_longitudinal_aggressive", "comfort_vs_aggressive"),
]


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def reset_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"output_dir exists: {path}. Use --overwrite.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def run_stage6_compare(dataset_dir: Path, embedding_dir: Path, out_dir: Path, a_path: Path, b_path: Path, args) -> None:
    cmd = [
        sys.executable, "tools/stage6_compare_unpaired_style.py",
        "--embedding_path", str(require_file(embedding_dir / "embedding.npy", "Stage 7E embedding.npy")),
        "--feature_path", str(require_file(dataset_dir / "shards" / "shard_000" / "interaction_feat_style.npy", "Stage 7D interaction features")),
        "--feature_schema_path", str(require_file(dataset_dir / "feature_schema.json", "Stage 7D feature schema")),
        "--a_indices_path", str(a_path),
        "--b_indices_path", str(b_path),
        "--output_dir", str(out_dir),
        "--num_bootstrap", str(args.num_bootstrap),
        "--num_permutation", str(args.num_permutation),
        "--min_slice_size", str(args.min_slice_size),
        "--top_k", str(args.top_k),
        "--overwrite",
    ]
    subprocess.run(cmd, check=True)


def run(args) -> None:
    dataset_dir = Path(args.dataset_dir)
    embedding_dir = Path(args.embedding_dir)
    output_dir = Path(args.output_dir)
    reset_dir(output_dir, args.overwrite)
    require_file(dataset_dir / "shards" / "shard_000" / "metadata.csv", "Stage 7D metadata.csv")
    require_file(embedding_dir / "embedding_manifest.json", "Stage 7E embedding_manifest.json")

    validation = {"pass": True, "comparisons": {}}
    manifest = {"stage": "7F", "purpose": "run_existing_stage6_bdd_report_card_on_stage7e_idm_embeddings", "comparisons": []}
    warnings = []
    for a_name, b_name, out_name in COMPARISONS:
        a_path = require_file(dataset_dir / "planner_policy_indices" / f"{a_name}.npy", f"A index {a_name}")
        b_path = require_file(dataset_dir / "planner_policy_indices" / f"{b_name}.npy", f"B index {b_name}")
        a_idx = np.load(a_path)
        b_idx = np.load(b_path)
        comp_dir = output_dir / out_name
        if len(a_idx) == 0 or len(b_idx) == 0:
            raise ValueError(f"Empty A/B indices for {out_name}: A={len(a_idx)} B={len(b_idx)}")
        run_stage6_compare(dataset_dir, embedding_dir, comp_dir, a_path, b_path, args)
        produced = {
            "bdd_summary.json": (comp_dir / "bdd_summary.json").exists(),
            "style_report_card.md": (comp_dir / "style_report_card.md").exists(),
            "feature_delta.csv": (comp_dir / "feature_delta.csv").exists(),
            "category_delta.csv": (comp_dir / "category_delta.csv").exists(),
        }
        ok = bool(len(a_idx) == args.expected_rows_per_policy and len(b_idx) == args.expected_rows_per_policy and produced["bdd_summary.json"] and produced["style_report_card.md"])
        validation["comparisons"][out_name] = {"pass": ok, "A": a_name, "B": b_name, "n_A": int(len(a_idx)), "n_B": int(len(b_idx)), "produced": produced}
        validation["pass"] = bool(validation["pass"] and ok)
        manifest["comparisons"].append({"name": out_name, "A": a_name, "B": b_name, "a_indices_path": str(a_path), "b_indices_path": str(b_path), "output_dir": str(comp_dir)})
        if not ok:
            warnings.append(f"Validation failed for {out_name}; expected {args.expected_rows_per_policy} rows per side and Stage 6 outputs.")

    write_json(output_dir / "comparison_manifest.json", manifest)
    write_json(output_dir / "warnings.json", {"warnings": warnings, "validation": validation})
    lines = [
        "# Stage 7F IDM Stage 6 BDD/Report-Card Smoke Summary", "",
        f"- validation.pass: **{str(validation['pass']).upper()}**",
        "- Purpose: interface validation, not statistical significance.",
        "- Bridge validated: official nuPlan simulation → Stage 6-compatible data → embedding → BDD/report card.",
        "- Results are exploratory positive-control evidence only for this 5-log smoke test.", "",
        "## Comparisons",
    ]
    for name, info in validation["comparisons"].items():
        lines.append(f"- `{name}`: pass={info['pass']}, n_A={info['n_A']}, n_B={info['n_B']}")
    (output_dir / "stage7f_idm_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not validation["pass"]:
        raise RuntimeError("Stage 7F validation failed; see warnings.json")


def parse_args():
    p = argparse.ArgumentParser(description="Stage 7F smoke: run existing Stage 6 BDD/report-card comparisons on Stage 7E IDM embeddings.")
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--embedding_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_bootstrap", type=int, default=50)
    p.add_argument("--num_permutation", type=int, default=100)
    p.add_argument("--min_slice_size", type=int, default=2)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--expected_rows_per_policy", type=int, default=5)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
