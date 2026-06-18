#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from tools.stage7f_idm_diagnostic_common import find_planner_index

TASK_POS_LABELS = {
    "task_following": "following",
    "task_lead_brake_response": "lead_brake_response",
    "task_queue_approach": "queue_approach",
    "task_cutin_response": "cutin_response",
    "task_yield_conflict": "yield_conflict",
    "task_lane_change": "lane_change",
    "task_overtake_opportunity": "overtake_opportunity",
    "task_overtake_executed": "overtake_executed",
    "task_hesitation": "hesitation",
}

OUTPUTS = [
    "task_overlap_matrix_all.csv",
    "task_overlap_matrix_planner_a.csv",
    "task_overlap_matrix_planner_b.csv",
    "task_overlap_matrix_paired_scenarios.csv",
    "task_overlap_summary.json",
    "task_overlap_report.md",
]


def require_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def load_metadata(embedding_dir: Path, context_dataset_dir: Path) -> pd.DataFrame:
    candidates = [embedding_dir / "metadata.csv", context_dataset_dir / "metadata.csv"]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            if "global_row" not in df.columns:
                df = df.copy()
                df["global_row"] = np.arange(len(df), dtype=int)
            if "planner_name" not in df.columns:
                raise ValueError(f"metadata.csv is missing required column planner_name: {path}")
            if "scenario_token" not in df.columns and "scenario_id" not in df.columns:
                raise ValueError(f"metadata.csv is missing scenario_token or scenario_id: {path}")
            if "scenario_token" not in df.columns:
                df = df.rename(columns={"scenario_id": "scenario_token"})
            return df
    raise FileNotFoundError(f"Missing metadata.csv in embedding_dir or context_dataset_dir: {candidates}")


def positive_rows(events_df: pd.DataFrame, task_key: str) -> Set[int]:
    if task_key not in TASK_POS_LABELS:
        raise ValueError(f"Unknown task_key={task_key}; known={sorted(TASK_POS_LABELS)}")
    if task_key not in events_df.columns:
        raise ValueError(f"behavior_event_bins_v2.csv is missing task column {task_key}")
    if "global_row" not in events_df.columns:
        raise ValueError("behavior_event_bins_v2.csv is missing required column global_row")
    pos_label = TASK_POS_LABELS[task_key]
    mask = events_df[task_key].astype(str) == pos_label
    return set(pd.to_numeric(events_df.loc[mask, "global_row"], errors="raise").astype(int).tolist())


def jaccard(a: Set, b: Set) -> float:
    union = len(a | b)
    return float(len(a & b) / union) if union else 1.0


def matrix_rows(task_keys: List[str], sets: Dict[str, Set]) -> List[Dict]:
    rows = []
    for left in task_keys:
        for right in task_keys:
            a, b = sets[left], sets[right]
            rows.append({
                "task_i": left,
                "task_j": right,
                "overlap_count": int(len(a & b)),
                "union_count": int(len(a | b)),
                "jaccard": jaccard(a, b),
                "task_i_count": int(len(a)),
                "task_j_count": int(len(b)),
            })
    return rows


def scenario_sets_for_both_positive(meta: pd.DataFrame, task_sets: Dict[str, Set], a_rows: Set[int], b_rows: Set[int]) -> Dict[str, Set[str]]:
    idx = meta.set_index("global_row", drop=False)
    out: Dict[str, Set[str]] = {}
    for task, rows in task_sets.items():
        a_pos = rows & a_rows
        b_pos = rows & b_rows
        a_scenarios = set(idx.loc[list(a_pos), "scenario_token"].astype(str).tolist()) if a_pos else set()
        b_scenarios = set(idx.loc[list(b_pos), "scenario_token"].astype(str).tolist()) if b_pos else set()
        out[task] = a_scenarios & b_scenarios
    return out


def load_task_bdd(stage7f_task_bdd_dir: Path) -> Dict[str, Dict]:
    csv_path = stage7f_task_bdd_dir / "task_bdd_summary.csv"
    json_path = stage7f_task_bdd_dir / "stage7f_task_bdd_summary.json"
    out: Dict[str, Dict] = {}
    if csv_path.exists():
        for row in pd.read_csv(csv_path).to_dict("records"):
            key = row.get("task_key")
            if isinstance(key, str):
                out[key] = row
    elif json_path.exists():
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        for row in raw.get("rows", []):
            key = row.get("task_key")
            if isinstance(key, str):
                out[key] = row
    return out


def write_matrix(path: Path, rows: List[Dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def sorted_unique_pairs(rows: List[Dict]) -> List[Dict]:
    return sorted([r for r in rows if r["task_i"] < r["task_j"]], key=lambda r: (-float(r["jaccard"]), r["task_i"], r["task_j"]))


def build_report(summary: Dict, all_rows: List[Dict], paired_rows: List[Dict]) -> str:
    lines = [
        "# Stage7F task overlap diagnostic", "",
        f"* planner_a: `{summary['planner_a']}`",
        f"* planner_b: `{summary['planner_b']}`",
        f"* task_keys: `{', '.join(summary['task_keys'])}`", "",
        "## Positive counts", "",
        "| task_key | all_positive_rows | planner_a_positive_rows | planner_b_positive_rows | paired_positive_scenarios | BDD | p_value |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in summary["task_keys"]:
        c = summary["positive_counts"][task]
        b = summary.get("task_bdd", {}).get(task, {})
        bdd = b.get("bdd_mmd", b.get("mmd2", ""))
        p = b.get("p_value", b.get("permutation_p_value", ""))
        lines.append(f"| {task} | {c['all']} | {c['planner_a']} | {c['planner_b']} | {c['paired_scenarios']} | {bdd} | {p} |")
    lines += ["", "## All-row overlap pairs sorted by Jaccard", "", "| task_i | task_j | overlap_count | union_count | jaccard |", "|---|---|---:|---:|---:|"]
    for r in sorted_unique_pairs(all_rows):
        lines.append(f"| {r['task_i']} | {r['task_j']} | {r['overlap_count']} | {r['union_count']} | {r['jaccard']:.6f} |")
    fq = summary["following_vs_queue"]
    lines += ["", "## following-vs-queue diagnostic", "",
              f"* all-row overlap_count: `{fq['all']['overlap_count']}`",
              f"* all-row Jaccard: `{fq['all']['jaccard']}`",
              f"* planner_a Jaccard: `{fq['planner_a']['jaccard']}`",
              f"* planner_b Jaccard: `{fq['planner_b']['jaccard']}`",
              f"* paired-scenario overlap_count: `{fq['paired_scenarios']['overlap_count']}`",
              f"* paired-scenario Jaccard: `{fq['paired_scenarios']['jaccard']}`",
              f"* identical_positive_rows_all: `{fq['identical_positive_rows_all']}`",
              f"* identical_positive_rows_planner_a: `{fq['identical_positive_rows_planner_a']}`",
              f"* identical_positive_rows_planner_b: `{fq['identical_positive_rows_planner_b']}`",
              f"* identical_paired_scenarios: `{fq['identical_paired_scenarios']}`", "",
              "## Interpretation", "",
              "* If following and queue are identical/highly overlapping, their BDD results are not independent evidence.",
              "* Report them as one combined longitudinal interaction evidence cluster.",
              "* Lead-brake can be treated as additional evidence if its positive rows are not identical to following/queue.",
              "* Cut-in is proxy-based and should be interpreted cautiously.",
              "* Yield-conflict is strong detector but not significant in current A/B result."]
    return "\n".join(lines) + "\n"


def run(args) -> Dict:
    out = Path(args.output_dir)
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    for name in OUTPUTS:
        if (out / name).exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {out / name}. Use --overwrite.")
    events = pd.read_csv(require_path(Path(args.events_dir) / "behavior_event_bins_v2.csv", "behavior_event_bins_v2.csv"))
    require_path(Path(args.events_dir) / "behavior_event_metrics_v2.csv", "behavior_event_metrics_v2.csv")
    meta = load_metadata(Path(args.embedding_dir), Path(args.context_dataset_dir))
    a_idx = set(np.load(find_planner_index(Path(args.stage7f_dir), args.planner_a)).astype(int).tolist())
    b_idx = set(np.load(find_planner_index(Path(args.stage7f_dir), args.planner_b)).astype(int).tolist())
    task_keys = [x.strip() for x in args.task_keys.split(",") if x.strip()]
    task_sets = {t: positive_rows(events, t) for t in task_keys}
    all_rows_set = set(pd.to_numeric(meta["global_row"], errors="raise").astype(int).tolist())
    task_sets = {t: rows & all_rows_set for t, rows in task_sets.items()}
    a_sets = {t: rows & a_idx for t, rows in task_sets.items()}
    b_sets = {t: rows & b_idx for t, rows in task_sets.items()}
    paired_sets = scenario_sets_for_both_positive(meta, task_sets, a_idx, b_idx)
    matrices = {
        "all": matrix_rows(task_keys, task_sets),
        "planner_a": matrix_rows(task_keys, a_sets),
        "planner_b": matrix_rows(task_keys, b_sets),
        "paired_scenarios": matrix_rows(task_keys, paired_sets),
    }
    write_matrix(out / "task_overlap_matrix_all.csv", matrices["all"])
    write_matrix(out / "task_overlap_matrix_planner_a.csv", matrices["planner_a"])
    write_matrix(out / "task_overlap_matrix_planner_b.csv", matrices["planner_b"])
    write_matrix(out / "task_overlap_matrix_paired_scenarios.csv", matrices["paired_scenarios"])
    def pair_stats(sets, left="task_following", right="task_queue_approach"):
        a, b = sets.get(left, set()), sets.get(right, set())
        return {"overlap_count": int(len(a & b)), "union_count": int(len(a | b)), "jaccard": jaccard(a, b)}
    fq = {
        "all": pair_stats(task_sets), "planner_a": pair_stats(a_sets), "planner_b": pair_stats(b_sets), "paired_scenarios": pair_stats(paired_sets),
        "identical_positive_rows_all": task_sets.get("task_following", set()) == task_sets.get("task_queue_approach", set()),
        "identical_positive_rows_planner_a": a_sets.get("task_following", set()) == a_sets.get("task_queue_approach", set()),
        "identical_positive_rows_planner_b": b_sets.get("task_following", set()) == b_sets.get("task_queue_approach", set()),
        "identical_paired_scenarios": paired_sets.get("task_following", set()) == paired_sets.get("task_queue_approach", set()),
    }
    summary = {
        "planner_a": args.planner_a, "planner_b": args.planner_b, "task_keys": task_keys,
        "positive_counts": {t: {"all": len(task_sets[t]), "planner_a": len(a_sets[t]), "planner_b": len(b_sets[t]), "paired_scenarios": len(paired_sets[t])} for t in task_keys},
        "following_vs_queue": fq,
        "task_bdd": load_task_bdd(Path(args.stage7f_task_bdd_dir)),
        "outputs": OUTPUTS,
    }
    (out / "task_overlap_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "task_overlap_report.md").write_text(build_report(summary, matrices["all"], matrices["paired_scenarios"]), encoding="utf-8")
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Diagnose overlap among Stage7F/Stage6C task-positive row and paired-scenario slices.")
    p.add_argument("--events_dir", required=True)
    p.add_argument("--stage7f_task_bdd_dir", required=True)
    p.add_argument("--embedding_dir", required=True)
    p.add_argument("--context_dataset_dir", required=True)
    p.add_argument("--stage7f_dir", required=True)
    p.add_argument("--planner_a", required=True)
    p.add_argument("--planner_b", required=True)
    p.add_argument("--task_keys", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
