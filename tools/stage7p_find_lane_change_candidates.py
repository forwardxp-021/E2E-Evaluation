#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
KINEMATIC_OUTPUT_FIELDS = [
    "candidate_rank",
    "metadata_index",
    "candidate_source",
    "candidate_score",
    "match_score",
    "metadata_match_score",
    "event_task_lane_change",
    "match_sources",
    "log_name",
    "scenario_token",
    "scenario_id",
    "scenario_type",
    "duration",
    "num_poses",
    "ego_start_x",
    "ego_start_y",
    "ego_start_yaw",
    "ego_end_x",
    "ego_end_y",
    "ego_end_yaw",
    "total_displacement",
    "lateral_displacement_in_start_ego_frame",
    "abs_lateral_displacement",
    "heading_change_abs",
    "yaw_rate_proxy",
    "max_lateral_speed_proxy",
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


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def compute_kinematic_metrics(
    poses: Sequence[Dict[str, Any]],
    *,
    log_name: str = "",
    scenario_token: str = "",
    scenario_id: str = "",
    scenario_type: str = "",
    text_match_bonus: float = 0.0,
) -> Dict[str, Any]:
    if len(poses) < 2:
        raise ValueError(f"Need at least 2 ego poses to compute kinematic metrics, got {len(poses)}")
    parsed: List[Tuple[float, float, float, float]] = []
    for idx, pose in enumerate(poses):
        try:
            x = float(pose["x"])
            y = float(pose["y"])
            yaw = float(pose.get("yaw", pose.get("heading")))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid ego pose at index {idx}; required numeric x/y/yaw or heading fields: {pose}") from exc
        raw_t = pose.get("timestamp", pose.get("time", idx))
        try:
            t = float(raw_t)
        except (TypeError, ValueError):
            t = float(idx)
        parsed.append((x, y, yaw, t))
    sx, sy, syaw, st = parsed[0]
    ex, ey, eyaw, et = parsed[-1]
    dx = ex - sx
    dy = ey - sy
    lateral = -math.sin(syaw) * dx + math.cos(syaw) * dy
    total = math.hypot(dx, dy)
    heading_change_abs = abs(_wrap_angle(eyaw - syaw))
    duration = max(float(et - st), float(len(parsed) - 1), 1e-6)
    yaw_rate_proxy = heading_change_abs / duration
    max_lateral_speed_proxy = 0.0
    prev_lat = 0.0
    prev_t = st
    for x, y, _yaw, t in parsed[1:]:
        rel_lat = -math.sin(syaw) * (x - sx) + math.cos(syaw) * (y - sy)
        dt = max(float(t - prev_t), 1e-6)
        max_lateral_speed_proxy = max(max_lateral_speed_proxy, abs(rel_lat - prev_lat) / dt)
        prev_lat = rel_lat
        prev_t = t
    candidate_score = 2.0 * abs(lateral) + 5.0 * heading_change_abs + 2.0 * yaw_rate_proxy + float(text_match_bonus)
    return {
        "log_name": log_name,
        "scenario_token": scenario_token,
        "scenario_id": scenario_id,
        "scenario_type": scenario_type,
        "duration": duration,
        "num_poses": int(len(parsed)),
        "ego_start_x": sx,
        "ego_start_y": sy,
        "ego_start_yaw": syaw,
        "ego_end_x": ex,
        "ego_end_y": ey,
        "ego_end_yaw": eyaw,
        "total_displacement": total,
        "lateral_displacement_in_start_ego_frame": lateral,
        "abs_lateral_displacement": abs(lateral),
        "heading_change_abs": heading_change_abs,
        "yaw_rate_proxy": yaw_rate_proxy,
        "max_lateral_speed_proxy": max_lateral_speed_proxy,
        "candidate_score": candidate_score,
    }


def build_text_event_candidates(metadata: List[Dict[str, Any]], fieldnames: List[str], event_map: Dict[int, int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(metadata):
        text_score, sources = score_metadata_row(row)
        event_positive = int(event_map.get(int(idx), 0) > 0)
        match_score = text_score + (20 if event_positive else 0)
        if match_score <= 0:
            continue
        candidate = {
            "metadata_index": int(idx),
            "candidate_source": "behavior_event" if event_positive else "text_match",
            "candidate_score": float(match_score),
            "match_score": int(match_score),
            "metadata_match_score": int(text_score),
            "event_task_lane_change": int(event_positive),
            "match_sources": ";".join(sources + (["behavior_event_bins_v2:task_lane_change"] if event_positive else [])),
        }
        for col in fieldnames:
            candidate[col] = row.get(col, "")
        rows.append(candidate)
    return rows


def discover_sqlite_schema(db_path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"db_path": str(db_path), "tables": [], "interesting_tables": [], "warnings": []}
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as exc:
        info["warnings"].append(f"Could not open SQLite DB {db_path}: {exc}")
        return info
    with conn:
        try:
            table_rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
        except sqlite3.Error as exc:
            info["warnings"].append(f"Could not list tables in {db_path}: {exc}")
            return info
        keywords = ("scenario", "type", "token", "log", "ego", "state", "pose", "x", "y", "yaw", "heading")
        for table_row in table_rows:
            table = str(table_row["name"])
            try:
                cols = [str(r["name"]) for r in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]
            except sqlite3.Error as exc:
                info["warnings"].append(f"Could not inspect table {table} in {db_path}: {exc}")
                continue
            entry = {"table": table, "columns": cols}
            info["tables"].append(entry)
            haystack = " ".join([table, *cols]).lower()
            if any(k in haystack for k in keywords):
                info["interesting_tables"].append(entry)
    return info


def _find_col(columns: Sequence[str], names: Sequence[str]) -> Optional[str]:
    lowered = {c.lower(): c for c in columns}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def scan_sqlite_kinematics(db_path: Path, max_scenarios: int, warnings: List[str]) -> List[Dict[str, Any]]:
    schema = discover_sqlite_schema(db_path)
    for warning in schema.get("warnings", []):
        warnings.append(warning)
    rows: List[Dict[str, Any]] = []
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as exc:
        warnings.append(f"Could not open SQLite DB for kinematic scan {db_path}: {exc}")
        return rows
    with conn:
        for table_info in schema.get("interesting_tables", []):
            table = table_info["table"]
            cols = table_info["columns"]
            x_col = _find_col(cols, ["x", "ego_x", "center_x"])
            y_col = _find_col(cols, ["y", "ego_y", "center_y"])
            yaw_col = _find_col(cols, ["yaw", "heading", "ego_yaw"])
            if not (x_col and y_col and yaw_col):
                continue
            time_col = _find_col(cols, ["timestamp", "time_us", "time", "iteration", "id"])
            log_col = _find_col(cols, ["log_name", "log_file", "filename"])
            token_col = _find_col(cols, ["scenario_token", "token", "scenario_id"])
            select_cols = [x_col, y_col, yaw_col] + ([time_col] if time_col else []) + ([log_col] if log_col else []) + ([token_col] if token_col else [])
            select_sql = ", ".join([f'"{c}"' for c in dict.fromkeys(select_cols)])
            try:
                fetched = conn.execute(f'SELECT {select_sql} FROM "{table}" LIMIT 200').fetchall()
            except sqlite3.Error as exc:
                warnings.append(f"Could not read pose-like table {table} in {db_path}: {exc}")
                continue
            if len(fetched) < 2:
                continue
            poses = []
            for idx, row in enumerate(fetched):
                poses.append({"x": row[x_col], "y": row[y_col], "yaw": row[yaw_col], "timestamp": row[time_col] if time_col else idx})
            try:
                metrics = compute_kinematic_metrics(
                    poses,
                    log_name=str(fetched[0][log_col]) if log_col else db_path.stem,
                    scenario_token=str(fetched[0][token_col]) if token_col else f"{db_path.stem}:{table}:first_{len(fetched)}_poses",
                    scenario_id=f"{db_path.stem}:{table}",
                    scenario_type="sqlite_pose_scan",
                )
            except ValueError as exc:
                warnings.append(f"Could not compute kinematic metrics for {db_path}:{table}: {exc}")
                continue
            rows.append(metrics)
            if len(rows) >= max_scenarios:
                break
    if not rows:
        warnings.append(f"No directly readable ego pose table found in {db_path}; schema discovery was written for inspection.")
    return rows


def run_kinematic_scan(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    enabled = bool(getattr(args, "enable_kinematic_scan", False))
    summary: Dict[str, Any] = {"enabled": enabled, "candidates": 0, "scanned_dbs": 0, "warnings": [], "schema_discovery": []}
    if not enabled:
        return [], summary
    db_root_value = getattr(args, "nuplan_db_root", "") or ""
    if not db_root_value:
        summary["warnings"].append("--enable_kinematic_scan was set but --nuplan_db_root is empty; skipped DB scan.")
        return [], summary
    db_root = Path(db_root_value)
    if not db_root.exists():
        summary["warnings"].append(f"--nuplan_db_root does not exist: {db_root}")
        return [], summary
    db_paths = sorted([p for p in db_root.rglob("*.db") if p.is_file()]) if db_root.is_dir() else [db_root]
    max_scenarios = int(getattr(args, "max_scenarios_scan", 50) or 50)
    selected: List[Dict[str, Any]] = []
    remaining = max_scenarios
    for db_path in db_paths:
        if remaining <= 0:
            break
        summary["scanned_dbs"] += 1
        schema = discover_sqlite_schema(db_path)
        summary["schema_discovery"].append(schema)
        rows = scan_sqlite_kinematics(db_path, remaining, summary["warnings"])
        for row in rows:
            if (
                float(row["abs_lateral_displacement"]) >= float(getattr(args, "min_lateral_displacement", 2.0))
                or float(row["heading_change_abs"]) >= float(getattr(args, "min_heading_change", 0.25))
                or float(row["yaw_rate_proxy"]) >= float(getattr(args, "min_yaw_rate_proxy", 0.05))
            ):
                row["candidate_source"] = "kinematic"
                row["match_score"] = ""
                row["metadata_match_score"] = ""
                row["event_task_lane_change"] = 0
                row["match_sources"] = "kinematic_scan"
                row["metadata_index"] = ""
                selected.append(row)
                remaining -= 1
    summary["candidates"] = len(selected)
    return selected, summary


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
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
        f"- text_match_candidates: `{summary['text_match_candidates']}`",
        f"- behavior_event_candidates: `{summary['behavior_event_candidates']}`",
        f"- kinematic_candidates: `{summary['kinematic_candidates']}`",
        f"- final_selected_candidates: `{summary['final_selected_candidates']}`",
        f"- candidate_rows: `{summary['candidate_rows']}`",
        f"- top_k_written: `{summary['top_k_written']}`",
        f"- behavior_event_detector_available: `{summary['behavior_events']['available']}`",
        f"- kinematic_scan_enabled: `{summary['kinematic_scan']['enabled']}`",
        "",
        "## Matching rules",
        "- Preserve metadata text matching over `scenario_type` / `scenario_name` / `log_name` / `scenario_id` for lane-change-like terms.",
        "- If Stage7 behavior events are available, rows with `task_lane_change=1` receive an additional score boost.",
        "- Optional kinematic scan computes expert-ego lateral displacement, heading change, yaw-rate proxy, and `candidate_score = 2.0 * abs_lateral_displacement + 5.0 * heading_change_abs + 2.0 * yaw_rate_proxy + text_match_bonus`.",
        "",
        "## Candidate source counts",
        f"- text_match_candidates: `{summary['text_match_candidates']}`",
        f"- behavior_event_candidates: `{summary['behavior_event_candidates']}`",
        f"- kinematic_candidates: `{summary['kinematic_candidates']}`",
        f"- final_selected_candidates: `{summary['final_selected_candidates']}`",
        "",
        "## Warnings",
    ]
    warnings = list(summary.get("warnings", [])) + list(summary.get("kinematic_scan", {}).get("warnings", []))
    if warnings:
        lines.extend([f"- WARNING: {w}" for w in warnings])
    else:
        lines.append("- None.")
    lines.extend(["", "## Top candidates"])
    if not top:
        lines.append("No lane-change-like candidates were found. This is a metadata-only / optional-kinematic candidate discovery result: zero candidates means no text/behavior/kinematic scan match was found under the configured thresholds, not that PDM lacks lane-change capability.")
    else:
        display_cols = [c for c in ["candidate_rank", "candidate_source", "candidate_score", "metadata_index", "match_score", "event_task_lane_change", "scenario_type", "scenario_id", "scenario_token", "log_name", "abs_lateral_displacement", "heading_change_abs", "yaw_rate_proxy", "match_sources"] if c in top[0]]
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
    event_path = find_event_bins(context_dir, Path(args.behavior_events_dir) if getattr(args, "behavior_events_dir", "") else None)
    event_map, event_summary = load_lane_change_events(event_path, len(metadata))
    text_event_candidates = build_text_event_candidates(metadata, fieldnames, event_map)
    kinematic_candidates, kinematic_summary = run_kinematic_scan(args)
    candidates = text_event_candidates + kinematic_candidates
    candidates.sort(key=lambda r: (-float(r.get("candidate_score") or r.get("match_score") or 0), str(r.get("scenario_id", "")), str(r.get("metadata_index", ""))))
    for rank, row in enumerate(candidates, 1):
        row["candidate_rank"] = rank
    top = candidates[: int(args.top_k)]
    output_fields = list(dict.fromkeys([*KINEMATIC_OUTPUT_FIELDS, *fieldnames]))
    write_csv(out / "lane_change_candidate_metadata.csv", top, output_fields)
    text_match_count = sum(1 for r in text_event_candidates if int(r.get("metadata_match_score") or 0) > 0)
    behavior_count = sum(1 for r in text_event_candidates if int(r.get("event_task_lane_change") or 0) > 0)
    warnings = []
    if not text_event_candidates and not kinematic_candidates:
        warnings.append("candidate_rows=0 because metadata text matching and available behavior/kinematic scans found no lane-change-like rows; this does not diagnose PDM lane-change capability.")
    summary = {
        "context_dir": str(context_dir),
        "metadata_path": str(metadata_path),
        "metadata_rows": int(len(metadata)),
        "candidate_rows": int(len(candidates)),
        "text_match_candidates": int(text_match_count),
        "behavior_event_candidates": int(behavior_count),
        "kinematic_candidates": int(len(kinematic_candidates)),
        "final_selected_candidates": int(len(candidates)),
        "top_k_requested": int(args.top_k),
        "top_k_written": int(len(top)),
        "preferred_scenario_type_terms": PREFERRED_SCENARIO_TYPE_TERMS,
        "behavior_events": event_summary,
        "kinematic_scan": kinematic_summary,
        "warnings": warnings,
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
    parser = argparse.ArgumentParser(description="Find Stage7P lane-change-like nuPlan candidate scenarios from Stage7 metadata, optional behavior events, and optional kinematic DB scan.")
    parser.add_argument("--context_dir", required=True, help="Stage7B/Stage7B4 context directory containing merged_metadata.csv or metadata.csv.")
    parser.add_argument("--output_dir", required=True, help="Output directory for lane-change candidate report/summary/metadata CSV.")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top-ranked candidates to write.")
    parser.add_argument("--behavior_events_dir", default="", help="Optional directory containing behavior_event_bins_v2.csv with task_lane_change.")
    parser.add_argument("--nuplan_db_root", default="", help="nuPlan mini DB root or a single SQLite .db file for optional kinematic scan.")
    parser.add_argument("--nuplan_map_root", default="", help="nuPlan map root reserved for scenario-builder based scans; SQLite fallback does not require maps.")
    parser.add_argument("--max_scenarios_scan", type=int, default=50, help="Maximum DB-derived scenarios / pose windows to inspect during kinematic scan.")
    parser.add_argument("--enable_kinematic_scan", action="store_true", help="Enable trajectory-driven high-lateral-motion candidate discovery from nuPlan DB pose-like tables.")
    parser.add_argument("--min_lateral_displacement", type=float, default=2.0, help="Minimum absolute lateral displacement in the start ego frame for kinematic candidate selection.")
    parser.add_argument("--min_heading_change", type=float, default=0.25, help="Minimum absolute heading change in radians for kinematic candidate selection.")
    parser.add_argument("--min_yaw_rate_proxy", type=float, default=0.05, help="Minimum heading-change-over-duration proxy for kinematic candidate selection.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
