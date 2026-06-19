#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import csv

PREFERRED_SCENARIO_TYPE_TERMS = [
    "changing_lane",
    "lane_change",
    "high_lateral_acceleration",
    "near_multiple_vehicles",
    "cut_in",
    "merge",
]
GENERAL_LANE_CHANGE_TERMS = [
    "lanechange",
    "lane change",
    "changing lane",
    "changing_lane",
    "lane_change",
    "lateral",
    "cutin",
    "cut-in",
    "cut_in",
    "merge",
    "merging",
]
SCENARIO_TYPE_COLUMNS = ["scenario_type", "scenario_label", "type"]
TEXT_COLUMNS = [
    "scenario_type",
    "scenario_label",
    "scenario_name",
    "scenario_id",
    "scenario_token",
    "log_name",
    "db_name",
    "map_name",
]


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def find_metadata_path(context_dir: Path) -> Path:
    candidates = [context_dir / "merged_metadata.csv", context_dir / "metadata.csv"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing Stage7 metadata CSV; tried: {', '.join(str(p) for p in candidates)}")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def score_metadata_row(row: Dict[str, Any]) -> Tuple[int, List[str]]:
    score = 0
    sources: List[str] = []
    for col in SCENARIO_TYPE_COLUMNS:
        if col not in row:
            continue
        text = normalize_text(row[col])
        for term in PREFERRED_SCENARIO_TYPE_TERMS:
            if term in text:
                score += 10
                sources.append(f"{col}:{term}")
    for col in TEXT_COLUMNS:
        if col not in row:
            continue
        text = normalize_text(row[col])
        for term in GENERAL_LANE_CHANGE_TERMS:
            if term in text:
                score += 2
                sources.append(f"{col}:{term}")
    return score, sorted(set(sources))


def find_event_bins(context_dir: Path, behavior_events_dir: Optional[Path]) -> Optional[Path]:
    candidates: List[Path] = []
    if behavior_events_dir is not None:
        candidates.append(behavior_events_dir / "behavior_event_bins_v2.csv")
    candidates.extend([
        context_dir / "behavior_events_v2" / "behavior_event_bins_v2.csv",
        context_dir.parent / "behavior_events_v2" / "behavior_event_bins_v2.csv",
    ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_lane_change_events(path: Optional[Path], metadata_len: int) -> Tuple[Dict[int, int], Dict[str, Any]]:
    if path is None:
        return {}, {"available": False, "path": "", "reason": "behavior_event_bins_v2.csv not found"}
    with require_file(path, "behavior_event_bins_v2.csv").open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}, {"available": True, "path": str(path), "rows": 0, "positive_rows": 0}
    if "task_lane_change" not in rows[0]:
        return {}, {"available": False, "path": str(path), "reason": "missing task_lane_change column"}
    event_map: Dict[int, int] = {}
    for fallback_idx, row in enumerate(rows):
        raw_idx = row.get("global_row", fallback_idx)
        try:
            idx = int(float(raw_idx))
        except (TypeError, ValueError):
            idx = fallback_idx
        try:
            value = int(float(row.get("task_lane_change", 0) or 0))
        except (TypeError, ValueError):
            value = 0
        if idx < metadata_len:
            event_map[idx] = value
    return event_map, {
        "available": True,
        "path": str(path),
        "rows": int(len(rows)),
        "positive_rows": int(sum(1 for v in event_map.values() if v > 0)),
    }


def build_candidates(metadata: List[Dict[str, Any]], fieldnames: List[str], event_map: Dict[int, int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(metadata):
        text_score, sources = score_metadata_row(row)
        event_positive = int(event_map.get(int(idx), 0) > 0)
        match_score = text_score + (20 if event_positive else 0)
        if match_score <= 0:
            continue
        candidate = {
            "metadata_index": int(idx),
            "match_score": int(match_score),
            "metadata_match_score": int(text_score),
            "event_task_lane_change": int(event_positive),
            "match_sources": ";".join(sources + (["behavior_event_bins_v2:task_lane_change"] if event_positive else [])),
        }
        for col in fieldnames:
            candidate[col] = row.get(col, "")
        rows.append(candidate)
    rows.sort(key=lambda r: (-int(r["match_score"]), int(r["metadata_index"])))
    for rank, row in enumerate(rows, 1):
        row["candidate_rank"] = rank
    ordered = ["candidate_rank", "metadata_index", "match_score", "metadata_match_score", "event_task_lane_change", "match_sources", *fieldnames]
    return [{key: row.get(key, "") for key in ordered} for row in rows]


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def write_report(out: Path, summary: Dict[str, Any], top: List[Dict[str, Any]]) -> None:
    lines = [
        "# Stage7P Lane-Change Candidate Report",
        "",
        "## Summary",
        f"- context_dir: `{summary['context_dir']}`",
        f"- metadata_path: `{summary['metadata_path']}`",
        f"- metadata_rows: `{summary['metadata_rows']}`",
        f"- candidate_rows: `{summary['candidate_rows']}`",
        f"- top_k_written: `{summary['top_k_written']}`",
        f"- behavior_event_detector_available: `{summary['behavior_events']['available']}`",
        "",
        "## Matching rules",
        "- Prefer `scenario_type`-like columns containing: `changing_lane`, `lane_change`, `high_lateral_acceleration`, `near_multiple_vehicles`, `cut_in`, `merge`.",
        "- Also scan metadata labels / scenario ids / log names for lane-change-like text.",
        "- If Stage7 behavior events are available, rows with `task_lane_change=1` receive an additional score boost.",
        "",
        "## Top candidates",
    ]
    if not top:
        lines.append("No lane-change-like candidates were found by metadata text or behavior-event matching.")
    else:
        display_cols = [c for c in ["candidate_rank", "metadata_index", "match_score", "event_task_lane_change", "scenario_type", "scenario_id", "scenario_token", "log_name", "db_name", "match_sources"] if c in top[0]]
        lines.append("| " + " | ".join(display_cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(display_cols)) + " |")
        for candidate in top:
            values = [str(candidate.get(col, "")).replace("\n", " ").replace("|", "\\|") for col in display_cols]
            lines.append("| " + " | ".join(values) + " |")
    (out / "lane_change_candidate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    context_dir = Path(args.context_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metadata_path = find_metadata_path(context_dir)
    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        metadata = list(reader)
        fieldnames = list(reader.fieldnames or [])
    event_path = find_event_bins(context_dir, Path(args.behavior_events_dir) if args.behavior_events_dir else None)
    event_map, event_summary = load_lane_change_events(event_path, len(metadata))
    candidates = build_candidates(metadata, fieldnames, event_map)
    top = candidates[: int(args.top_k)]
    output_fields = ["candidate_rank", "metadata_index", "match_score", "metadata_match_score", "event_task_lane_change", "match_sources", *fieldnames]
    write_csv(out / "lane_change_candidate_metadata.csv", top, output_fields)
    summary = {
        "context_dir": str(context_dir),
        "metadata_path": str(metadata_path),
        "metadata_rows": int(len(metadata)),
        "candidate_rows": int(len(candidates)),
        "top_k_requested": int(args.top_k),
        "top_k_written": int(len(top)),
        "preferred_scenario_type_terms": PREFERRED_SCENARIO_TYPE_TERMS,
        "behavior_events": event_summary,
        "outputs": {
            "report": "lane_change_candidate_report.md",
            "summary": "lane_change_candidate_summary.json",
            "metadata": "lane_change_candidate_metadata.csv",
        },
    }
    (out / "lane_change_candidate_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(out, summary, top)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find Stage7P lane-change-like nuPlan candidate scenarios from Stage7 metadata and optional Stage7 behavior events.")
    parser.add_argument("--context_dir", required=True, help="Stage7B/Stage7B4 context directory containing merged_metadata.csv or metadata.csv.")
    parser.add_argument("--output_dir", required=True, help="Output directory for lane-change candidate report/summary/metadata CSV.")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top-ranked candidates to write.")
    parser.add_argument("--behavior_events_dir", default="", help="Optional directory containing behavior_event_bins_v2.csv with task_lane_change.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
