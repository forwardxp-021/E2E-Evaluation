#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import itertools
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROW_SEMANTICS = "scenario × planner-controlled nuPlan ego rollout"
STAGE6_COMPARE = "tools/stage6_compare_unpaired_style.py"
STAGE6_REPORT = "tools/stage6_generate_report_card.py"


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


def read_json_if_exists(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def resolve_context_dir(embedding_dir: Path, explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        return Path(explicit)
    manifest = read_json_if_exists(embedding_dir / "embedding_manifest.json")
    raw = manifest.get("context_dataset_dir")
    return Path(raw) if raw else None


def planner_column(meta: pd.DataFrame) -> str:
    for col in ["planner_name", "policy_style", "planner", "planner_id"]:
        if col in meta.columns:
            return col
    raise ValueError(f"metadata.csv lacks planner axis column; tried planner_name/policy_style/planner/planner_id. columns={list(meta.columns)}")


def scenario_column(meta: pd.DataFrame) -> str:
    for col in ["scenario_token", "scenario_id", "scenario_index", "log_scenario_id"]:
        if col in meta.columns:
            return col
    raise ValueError(f"metadata.csv lacks scenario axis column; tried scenario_token/scenario_id/scenario_index/log_scenario_id. columns={list(meta.columns)}")


def alignment_summary(meta: pd.DataFrame, mode: str) -> Dict:
    pcol = planner_column(meta)
    scol = scenario_column(meta)
    tmp = meta[[scol, pcol]].copy()
    tmp[scol] = tmp[scol].astype(str)
    tmp[pcol] = tmp[pcol].astype(str)
    planners = sorted(tmp[pcol].dropna().unique().tolist())
    scenarios = sorted(tmp[scol].dropna().unique().tolist())
    counts = tmp.groupby(scol)[pcol].nunique()
    complete = counts[counts == len(planners)].index.astype(str).tolist()
    missing = [s for s in scenarios if s not in set(complete)]
    duplicate_pairs = int(tmp.duplicated([scol, pcol]).sum())
    all_complete = bool(len(missing) == 0 and duplicate_pairs == 0 and len(tmp) == len(scenarios) * len(planners))
    if mode == "full" and not all_complete:
        raise ValueError(
            "Stage7F full mode requires complete scenario × planner alignment. "
            f"scenarios={len(scenarios)} planners={len(planners)} rows={len(tmp)} missing_scenarios={len(missing)} duplicate_pairs={duplicate_pairs}"
        )
    return {
        "scenario_column": scol,
        "planner_column": pcol,
        "planner_axis": planners,
        "num_scenarios": int(len(scenarios)),
        "num_planners": int(len(planners)),
        "total_rows": int(len(tmp)),
        "all_scenarios_have_all_planners": all_complete,
        "scenarios_with_all_planners": int(len(complete)),
        "scenarios_missing_any_planner": int(len(missing)),
        "missing_scenario_ids": missing[:50],
        "duplicate_scenario_planner_pairs": duplicate_pairs,
    }


def fallback_summary(meta: pd.DataFrame, diagnostics: Dict, mode: str) -> Dict:
    out = {"fallback_preserving_status": mode == "full"}
    if "fallback_used" in meta.columns:
        vals = pd.to_numeric(meta["fallback_used"], errors="coerce").fillna(0)
        out["fallback_rows"] = int((vals > 0).sum())
        out["fallback_rate"] = float((vals > 0).mean()) if len(vals) else 0.0
    for key in ["fallback_rate", "map_name_resolution_status", "slot_sanity", "strict_filter_min_laneaware_ratio", "rows_kept", "kept_row_rate"]:
        if key in diagnostics:
            out[key] = diagnostics[key]
    return out


def write_indices(meta: pd.DataFrame, pcol: str, out: Path) -> Dict[str, str]:
    idx_dir = out / "planner_indices"
    idx_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for planner, rows in meta.groupby(pcol, sort=True).groups.items():
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(planner))
        path = idx_dir / f"{safe}.npy"
        np.save(path, np.asarray(sorted(rows), dtype=np.int64))
        paths[str(planner)] = str(path)
    return paths


def run_stage6_pairwise(embedding_dir: Path, context_dir: Path, output_dir: Path, idx_paths: Dict[str, str], args) -> List[Dict]:
    feature_path = context_dir / "interaction_feat_style.npy"
    schema_path = context_dir / "feature_schema.json"
    if not feature_path.exists() or not schema_path.exists():
        return [{"warning": "stage6_pairwise_skipped_missing_feature_inputs", "feature_path": str(feature_path), "feature_schema_path": str(schema_path)}]
    results = []
    for a, b in itertools.combinations(sorted(idx_paths), 2):
        pair_name = f"{a}_vs_{b}".replace("/", "_").replace(" ", "_")
        pair_dir = output_dir / "stage6_pairwise" / pair_name
        cmd = [
            sys.executable, STAGE6_COMPARE,
            "--embedding_path", str(require_file(embedding_dir / "embedding.npy", "embedding.npy")),
            "--feature_path", str(feature_path),
            "--feature_schema_path", str(schema_path),
            "--a_indices_path", idx_paths[a],
            "--b_indices_path", idx_paths[b],
            "--output_dir", str(pair_dir),
            "--num_bootstrap", str(args.num_bootstrap),
            "--num_permutation", str(args.num_permutation),
            "--min_slice_size", str(args.min_slice_size),
            "--top_k", str(args.top_k),
            "--overwrite",
        ]
        subprocess.run(cmd, check=True)
        subprocess.run([sys.executable, STAGE6_REPORT, "--input_dir", str(pair_dir), "--overwrite"], check=True)
        results.append({"planner_a": a, "planner_b": b, "output_dir": str(pair_dir), "reused_stage6_tools": [STAGE6_COMPARE, STAGE6_REPORT]})
    return results


def write_report(out: Path, summary: Dict) -> None:
    align = summary["alignment"]
    strict_warning = ""
    if summary["mode"] == "strict_sensitivity":
        strict_warning = "\n> WARNING: strict-filter output is a clean-subset sensitivity diagnostic, not the main planner-evaluation dataset, because scenario × planner alignment may be incomplete.\n"
    lines = [
        "# Stage7F Report Card Runner Summary", "", strict_warning,
        f"- input embedding path: `{summary['embedding_dir']}`",
        f"- context source path: `{summary.get('context_dataset_dir')}`",
        f"- row semantics: `{ROW_SEMANTICS}`",
        f"- mode: `{summary['mode']}`",
        f"- number of scenarios: `{align['num_scenarios']}`",
        f"- number of planners: `{align['num_planners']}`",
        f"- total rows: `{align['total_rows']}`",
        f"- planner axis: `{align['planner_axis']}`",
        f"- all scenarios have all planners: `{align['all_scenarios_have_all_planners']}`",
        f"- scenarios_with_all_planners: `{align['scenarios_with_all_planners']}`",
        f"- scenarios_missing_any_planner: `{align['scenarios_missing_any_planner']}`",
        f"- fallback-preserving status: `{summary['fallback'].get('fallback_preserving_status')}`",
        f"- fallback rate: `{summary['fallback'].get('fallback_rate', 'unavailable')}`",
        f"- map_name resolution status: `{summary['fallback'].get('map_name_resolution_status', 'unavailable')}`",
        f"- embedding shape: `{summary['embedding_shape']}`", "",
        "## Stage6 reuse", "",
        "- This wrapper validates Stage7E row semantics and delegates pairwise report-card computation to existing Stage6 tools when feature inputs are available.",
    ]
    for item in summary.get("stage6_pairwise_outputs", []):
        lines.append(f"- `{item}`")
    (out / "stage7f_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args) -> None:
    embedding_dir = Path(args.embedding_dir)
    output_dir = Path(args.output_dir)
    reset_dir(output_dir, args.overwrite)
    emb = np.load(require_file(embedding_dir / "embedding.npy", "Stage7E embedding.npy"), mmap_mode="r")
    manifest = read_json_if_exists(require_file(embedding_dir / "embedding_manifest.json", "Stage7E embedding_manifest.json"))
    meta = pd.read_csv(require_file(embedding_dir / "metadata.csv", "Stage7E metadata.csv"))
    if emb.shape[0] != len(meta):
        raise ValueError(f"embedding.npy rows != metadata.csv rows: {emb.shape[0]} vs {len(meta)}")
    context_dir = resolve_context_dir(embedding_dir, args.context_dataset_dir)
    diagnostics = read_json_if_exists(Path(args.context_diagnostics_json)) if args.context_diagnostics_json else {}
    align = alignment_summary(meta, args.mode)
    fallback = fallback_summary(meta, diagnostics, args.mode)
    idx_paths = write_indices(meta, align["planner_column"], output_dir)
    stage6_outputs = []
    if args.run_stage6_pairwise:
        if context_dir is None:
            stage6_outputs = [{"warning": "stage6_pairwise_skipped_missing_context_dataset_dir"}]
        else:
            stage6_outputs = run_stage6_pairwise(embedding_dir, context_dir, output_dir, idx_paths, args)
    if args.mode == "strict_sensitivity":
        fallback.setdefault("strict_filter_min_laneaware_ratio", args.strict_filter_min_laneaware_ratio)
        fallback.setdefault("rows_kept", int(len(meta)))
        fallback.setdefault("kept_row_rate", None)
    summary = {
        "stage": "7F",
        "mode": args.mode,
        "embedding_dir": str(embedding_dir),
        "context_dataset_dir": str(context_dir) if context_dir else None,
        "row_semantics": ROW_SEMANTICS,
        "embedding_shape": list(emb.shape),
        "embedding_manifest": manifest,
        "alignment": align,
        "fallback": fallback,
        "planner_index_paths": idx_paths,
        "stage6_pairwise_outputs": stage6_outputs,
        "stage6_metric_definitions_modified": False,
        "new_metric_logic_implemented": False,
    }
    write_json(output_dir / "stage7f_summary.json", summary)
    write_report(output_dir, summary)


def parse_args():
    p = argparse.ArgumentParser(description="Stage7F thin runner: validate Stage7E embeddings and reuse Stage6 report-card/BDD tools.")
    p.add_argument("--embedding_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--context_dataset_dir", help="Optional Stage7E context dataset dir; otherwise read from embedding_manifest.json.")
    p.add_argument("--context_diagnostics_json", help="Optional Stage7E context diagnostic summary JSON.")
    p.add_argument("--mode", choices=["full", "strict_sensitivity"], default="full")
    p.add_argument("--strict_filter_min_laneaware_ratio", type=float, default=0.8)
    p.add_argument("--run_stage6_pairwise", action="store_true", help="Run existing Stage6 pairwise report-card commands when feature inputs are available.")
    p.add_argument("--num_bootstrap", type=int, default=50)
    p.add_argument("--num_permutation", type=int, default=100)
    p.add_argument("--min_slice_size", type=int, default=2)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
