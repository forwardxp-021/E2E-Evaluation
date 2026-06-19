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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

PREFERRED_SCENARIO_TYPE_TERMS = [
    "changing_lane_to_left",
    "changing_lane_to_right",
    "changing_lane",
    "lane_change",
    "high_lateral_acceleration",
    "cut_in",
    "merge",
    "near_multiple_vehicles",
]
STRICT_CHANGING_LANE_TYPES = {
    "changing_lane_to_left",
    "changing_lane_to_right",
    "changing_lane",
}
FALLBACK_EXACT_CHANGING_LANE_TYPES = {
    "high_lateral_acceleration",
    "cut_in",
    "merge",
    "near_multiple_vehicles",
}

DB_SCENARIO_TYPE_PRIORITY = {
    "changing_lane_to_left": 0,
    "changing_lane_to_right": 1,
    "changing_lane": 2,
    "high_lateral_acceleration": 3,
    "cut_in": 4,
    "merge": 5,
    "near_multiple_vehicles": 6,
}
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

DB_OUTPUT_FIELDS = [
    "log_name",
    "scenario_token",
    "scene_token",
    "db_scene_token",
    "scenario_type_db_tag",
    "actual_scenario_type",
    "actual_type_verified",
    "actual_type_verification_method",
    "actual_type_verification_error",
    "selected_by_db_tag_only",
    "candidate_source",
    "candidate_rank",
    "candidate_score",
    "selected_as_strict_changing_lane",
    "selected_as_fallback_lateral",
    "db_file",
    "scenario_tag_token",
    "lidar_pc_token",
    "ego_pose_token",
    "scenario_type",
    "source",
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


def token_to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, memoryview):
        return bytes(value).hex()
    return str(value)


def list_db_paths(db_root: Path, max_db_files: Optional[int] = None) -> List[Path]:
    db_paths = sorted([p for p in db_root.glob("*.db") if p.is_file()]) if db_root.is_dir() else [db_root]
    if max_db_files is not None and max_db_files > 0:
        return db_paths[:max_db_files]
    return db_paths


def scan_db_scenario_tags(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    enabled = bool(getattr(args, "scan_db_scenario_tags", False))
    summary: Dict[str, Any] = {
        "enabled": enabled,
        "candidates": 0,
        "raw_db_scenario_tag_rows": 0,
        "unique_scenario_token_rows": 0,
        "selected_rows": 0,
        "selected_scenario_type_counts": {},
        "selected_log_counts": {},
        "duplicate_scenario_token_count_removed": 0,
        "scanned_dbs": 0,
        "warnings": [],
    }
    if not enabled:
        return [], summary
    db_root_value = getattr(args, "nuplan_db_root", "") or ""
    if not db_root_value:
        summary["warnings"].append("--scan_db_scenario_tags was set but --nuplan_db_root is empty; skipped DB scenario_tag scan.")
        return [], summary
    db_root = Path(db_root_value)
    if not db_root.exists():
        summary["warnings"].append(f"--nuplan_db_root does not exist: {db_root}")
        return [], summary

    max_db_files = getattr(args, "max_db_files", None)
    db_paths = list_db_paths(db_root, int(max_db_files) if max_db_files else None)
    max_per_type = getattr(args, "max_candidates_per_type", None)
    max_per_type = int(max_per_type) if max_per_type else 0
    max_per_log = int(getattr(args, "max_per_log", 2) or 0)
    per_type_counts: Dict[str, int] = {}
    rows: List[Dict[str, Any]] = []
    terms = [t.lower() for t in DB_SCENARIO_TYPE_PRIORITY]

    for db_path in db_paths:
        summary["scanned_dbs"] += 1
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
        except sqlite3.Error as exc:
            summary["warnings"].append(f"Could not open SQLite DB for scenario_tag scan {db_path}: {exc}")
            continue
        with conn:
            try:
                log_row = conn.execute("SELECT logfile FROM log LIMIT 1").fetchone()
                default_log_name = token_to_str(log_row["logfile"]).strip() if log_row and "logfile" in log_row.keys() else ""
                if not default_log_name:
                    default_log_name = db_path.stem
                    summary["warnings"].append(f"log.logfile missing or empty in {db_path}; using db stem as log_name fallback: {default_log_name}")
                query = """
                    SELECT
                        st.token AS scenario_tag_token,
                        st.lidar_pc_token AS st_lidar_pc_token,
                        st.type AS scenario_type,
                        lp.token AS lidar_pc_token,
                        lp.scene_token AS scene_token,
                        lp.ego_pose_token AS ego_pose_token
                    FROM scenario_tag AS st
                    LEFT JOIN lidar_pc AS lp ON lp.token = st.lidar_pc_token
                """
                fetched = conn.execute(query).fetchall()
            except sqlite3.Error as exc:
                summary["warnings"].append(f"Could not read scenario_tag/lidar_pc/log from {db_path}: {exc}")
                continue
        for row in fetched:
            summary["raw_db_scenario_tag_rows"] += 1
            scenario_type = token_to_str(row["scenario_type"])
            scenario_type_lower = scenario_type.lower()
            if not any(term in scenario_type_lower for term in terms):
                continue
            if max_per_type > 0 and per_type_counts.get(scenario_type, 0) >= max_per_type:
                continue
            per_type_counts[scenario_type] = per_type_counts.get(scenario_type, 0) + 1
            priority = min((rank for term, rank in DB_SCENARIO_TYPE_PRIORITY.items() if term in scenario_type_lower), default=99)
            score = 1000.0 - float(priority) * 100.0 + max((len(term) for term in terms if term in scenario_type_lower), default=0) / 10.0
            lidar_pc_token = token_to_str(row["lidar_pc_token"] if row["lidar_pc_token"] is not None else row["st_lidar_pc_token"])
            if not lidar_pc_token:
                summary["warnings"].append(f"Skipped scenario_tag row in {db_path} with empty lidar_pc_token; scenario_type={scenario_type}")
                continue
            db_scene_token = token_to_str(row["scene_token"])
            rows.append(
                {
                    "db_file": str(db_path),
                    "log_name": default_log_name,
                    "scenario_type": scenario_type,
                    "scenario_type_db_tag": scenario_type,
                    "actual_scenario_type": "",
                    "actual_type_verified": "false",
                    "actual_type_verification_method": "",
                    "actual_type_verification_error": "",
                    "selected_by_db_tag_only": "false",
                    "selected_as_strict_changing_lane": "false",
                    "selected_as_fallback_lateral": "false",
                    "scenario_tag_token": token_to_str(row["scenario_tag_token"]),
                    "scenario_token": lidar_pc_token,
                    "lidar_pc_token": lidar_pc_token,
                    "scene_token": lidar_pc_token,
                    "db_scene_token": db_scene_token,
                    "ego_pose_token": token_to_str(row["ego_pose_token"]),
                    "source": "db_scenario_tag",
                    "candidate_source": "db_scenario_tag",
                    "candidate_score": float(score),
                    "match_score": "",
                    "metadata_match_score": "",
                    "event_task_lane_change": 0,
                    "match_sources": "scenario_tag.type",
                    "metadata_index": "",
                    "_db_priority": priority,
                }
            )
    best_by_token: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        token = str(row.get("scenario_token", ""))
        current = best_by_token.get(token)
        if current is None or (int(row.get("_db_priority", 99)), -float(row.get("candidate_score", 0.0))) < (int(current.get("_db_priority", 99)), -float(current.get("candidate_score", 0.0))):
            best_by_token[token] = row
    unique_rows = sorted(best_by_token.values(), key=lambda r: (int(r.get("_db_priority", 99)), str(r.get("log_name", "")), str(r.get("scenario_token", ""))))
    selected: List[Dict[str, Any]] = []
    per_log_counts: Dict[str, int] = {}
    for row in unique_rows:
        log_name = str(row.get("log_name", "") or "")
        if max_per_log > 0 and per_log_counts.get(log_name, 0) >= max_per_log:
            continue
        per_log_counts[log_name] = per_log_counts.get(log_name, 0) + 1
        row.pop("_db_priority", None)
        selected.append(row)
    summary["unique_scenario_token_rows"] = len(unique_rows)
    summary["duplicate_scenario_token_count_removed"] = max(0, len(rows) - len(unique_rows))
    summary["selected_rows"] = len(selected)
    summary["selected_scenario_type_counts"] = count_by_field(selected, "scenario_type")
    summary["selected_log_counts"] = count_by_field(selected, "log_name")
    summary["candidates"] = len(selected)
    return selected, summary


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



def parse_csv_set(value: str) -> Set[str]:
    return {item.strip().lower() for item in str(value or "").split(",") if item.strip()}


def _sqlite_table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    try:
        return [str(r["name"]) for r in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]
    except sqlite3.Error:
        return []


def lookup_actual_scenario_type_sqlite(row: Dict[str, Any]) -> Tuple[str, bool, str, str]:
    """Lightweight exact-token actual-type lookup for tests and DB sidecars.

    This does not treat scenario_tag.type as verified actual type.  It only accepts
    explicit actual-type tables/columns keyed by the Stage7C scenario token
    (scenario_tag.lidar_pc_token).
    """
    db_file = str(row.get("db_file", "") or "")
    scenario_token = str(row.get("scenario_token", "") or "")
    if not db_file or not scenario_token:
        return "", False, "sqlite_exact_token_lookup", "missing db_file or scenario_token"
    try:
        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as exc:
        return "", False, "sqlite_exact_token_lookup", f"could not open DB: {exc}"
    candidates = [
        ("scenario_actual_type", ["scenario_token", "lidar_pc_token", "token"], ["actual_scenario_type", "scenario_type", "type"]),
        ("scenario", ["scenario_token", "lidar_pc_token", "token"], ["actual_scenario_type", "scenario_type"]),
    ]
    with conn:
        for table, token_cols, type_cols in candidates:
            cols = _sqlite_table_columns(conn, table)
            if not cols:
                continue
            token_col = _find_col(cols, token_cols)
            type_col = _find_col(cols, type_cols)
            if not token_col or not type_col:
                continue
            try:
                found = conn.execute(f'SELECT "{type_col}" AS actual_type FROM "{table}" WHERE "{token_col}" = ? LIMIT 1', (scenario_token,)).fetchone()
            except sqlite3.Error as exc:
                return "", False, "sqlite_exact_token_lookup", f"lookup failed in {table}: {exc}"
            if found:
                actual = token_to_str(found["actual_type"]).strip()
                return actual, bool(actual), f"sqlite_exact_token_lookup:{table}.{type_col}", "" if actual else "empty actual scenario type"
    return "", False, "sqlite_exact_token_lookup", "no explicit actual scenario type table/column found for exact scenario_token"


def verify_actual_scenario_types(candidates: List[Dict[str, Any]]) -> None:
    for row in candidates:
        actual = str(row.get("actual_scenario_type", "") or "").strip()
        if actual:
            row["actual_type_verified"] = "true"
            row["actual_type_verification_method"] = row.get("actual_type_verification_method") or "input_metadata"
            row["actual_type_verification_error"] = ""
            continue
        actual, verified, method, error = lookup_actual_scenario_type_sqlite(row)
        row["actual_scenario_type"] = actual
        row["actual_type_verified"] = "true" if verified else "false"
        row["actual_type_verification_method"] = method
        row["actual_type_verification_error"] = error


def actual_type_allowed(row: Dict[str, Any], allowlist: Set[str]) -> bool:
    return str(row.get("actual_type_verified", "") or "").lower() == "true" and normalize_scenario_type(row.get("actual_scenario_type")) in allowlist


def actual_type_rejected(row: Dict[str, Any], allowlist: Set[str], fallback_allowlist: Optional[Set[str]] = None) -> bool:
    if str(row.get("actual_type_verified", "") or "").lower() != "true":
        return False
    actual = normalize_scenario_type(row.get("actual_scenario_type"))
    allowed = set(allowlist)
    if fallback_allowlist:
        allowed.update(fallback_allowlist)
    return bool(actual) and actual not in allowed


def db_tag_strict_but_actual_type_unverified(row: Dict[str, Any]) -> bool:
    return is_strict_changing_lane_row(row) and str(row.get("actual_type_verified", "") or "").lower() != "true"

def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def count_by_field(rows: Sequence[Dict[str, Any]], field: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        key = str(row.get(field, "") or "")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def count_by_field_nonempty(rows: Sequence[Dict[str, Any]], field: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        key = str(row.get(field, "") or "").strip()
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def normalize_scenario_type(value: Any) -> str:
    return str(value or "").strip().lower()


def is_strict_changing_lane_row(row: Dict[str, Any]) -> bool:
    return normalize_scenario_type(row.get("scenario_type_db_tag") or row.get("scenario_type")) in STRICT_CHANGING_LANE_TYPES


def is_fallback_exact_row(row: Dict[str, Any]) -> bool:
    return normalize_scenario_type(row.get("scenario_type")) in FALLBACK_EXACT_CHANGING_LANE_TYPES


def select_top_candidates(
    candidates: List[Dict[str, Any]],
    top_k: int,
    max_per_log: int,
    prefer_exact_changing_lane: bool,
    *,
    verify_actual_scenario_type: bool = False,
    actual_type_allowlist: Optional[Set[str]] = None,
    allow_fallback_lateral_types: bool = False,
    fallback_type_allowlist: Optional[Set[str]] = None,
    allow_db_tag_when_actual_type_unverified: bool = True,
    require_actual_type_verified: bool = False,
) -> List[Dict[str, Any]]:
    actual_type_allowlist = actual_type_allowlist or STRICT_CHANGING_LANE_TYPES
    fallback_type_allowlist = fallback_type_allowlist or {"high_lateral_acceleration"}
    selected: List[Dict[str, Any]] = []
    selected_tokens = set()
    per_log_counts: Dict[str, int] = {}

    def try_add(row: Dict[str, Any], *, strict: bool = False, fallback: bool = False) -> None:
        if len(selected) >= top_k:
            return
        token = str(row.get("scenario_token", "") or "")
        if token and token in selected_tokens:
            return
        log_name = str(row.get("log_name", "") or "")
        if max_per_log > 0 and log_name and per_log_counts.get(log_name, 0) >= max_per_log:
            return
        row["selected_as_strict_changing_lane"] = "true" if strict else "false"
        row["selected_as_fallback_lateral"] = "true" if fallback else "false"
        if strict and verify_actual_scenario_type and str(row.get("actual_type_verified", "") or "").lower() != "true":
            row["selected_by_db_tag_only"] = "true"
        elif strict or fallback:
            row["selected_by_db_tag_only"] = "false"
        selected.append(row)
        if token:
            selected_tokens.add(token)
        if log_name:
            per_log_counts[log_name] = per_log_counts.get(log_name, 0) + 1

    if verify_actual_scenario_type:
        for row in candidates:
            if actual_type_allowed(row, actual_type_allowlist):
                try_add(row, strict=True)
                if len(selected) >= top_k:
                    break
            elif (not require_actual_type_verified) and allow_db_tag_when_actual_type_unverified and db_tag_strict_but_actual_type_unverified(row):
                row["actual_type_verified"] = "false"
                row["actual_scenario_type"] = str(row.get("actual_scenario_type", "") or "")
                if not str(row.get("actual_type_verification_error", "") or ""):
                    row["actual_type_verification_error"] = "actual scenario type verification did not return a value"
                try_add(row, strict=True)
                if len(selected) >= top_k:
                    break
        if allow_fallback_lateral_types and len(selected) < top_k:
            for row in candidates:
                if actual_type_allowed(row, fallback_type_allowlist):
                    try_add(row, fallback=True)
                    if len(selected) >= top_k:
                        break
        return selected

    groups = [candidates]
    if prefer_exact_changing_lane:
        strict = [r for r in candidates if is_strict_changing_lane_row(r)]
        fallback = [r for r in candidates if (not is_strict_changing_lane_row(r)) and is_fallback_exact_row(r)]
        other = [r for r in candidates if (not is_strict_changing_lane_row(r)) and (not is_fallback_exact_row(r))]
        groups = [strict, fallback, other]
    for group in groups:
        for row in group:
            try_add(row, strict=is_strict_changing_lane_row(row), fallback=(not is_strict_changing_lane_row(row) and is_fallback_exact_row(row)))
            if len(selected) >= top_k:
                break
        if len(selected) >= top_k:
            break
    return selected

def write_stage7c_context(out: Path, top: List[Dict[str, Any]], original_fieldnames: Sequence[str]) -> str:
    context_dir = out / "stage7c_candidate_context"
    context_dir.mkdir(parents=True, exist_ok=True)
    required = [
        "log_name", "scenario_token", "scene_token", "db_scene_token", "scenario_type_db_tag",
        "actual_scenario_type", "actual_type_verified", "actual_type_verification_method",
        "actual_type_verification_error", "selected_by_db_tag_only", "candidate_source", "candidate_rank", "candidate_score",
        "selected_as_strict_changing_lane", "selected_as_fallback_lateral", "db_file",
        "scenario_tag_token", "lidar_pc_token", "ego_pose_token", "scenario_type", "source",
    ]
    fieldnames = list(dict.fromkeys([*required, *original_fieldnames]))
    rows = []
    seen_tokens = set()
    for row in top:
        scenario_token = str(row.get("scenario_token", "") or row.get("lidar_pc_token", "") or "").strip()
        if not scenario_token or scenario_token in seen_tokens:
            continue
        seen_tokens.add(scenario_token)
        stage7c_row = {col: row.get(col, "") for col in fieldnames}
        stage7c_row["log_name"] = str(row.get("log_name", "") or "").strip()
        stage7c_row["scenario_token"] = scenario_token
        stage7c_row["lidar_pc_token"] = str(row.get("lidar_pc_token", "") or scenario_token)
        stage7c_row["scene_token"] = scenario_token
        stage7c_row["db_scene_token"] = str(row.get("db_scene_token", "") or "")
        stage7c_row["source"] = row.get("source") or row.get("candidate_source", "")
        rows.append(stage7c_row)
    write_csv(context_dir / "merged_metadata.csv", rows, fieldnames)
    return "stage7c_candidate_context/merged_metadata.csv"


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
        f"- metadata_text candidates: `{summary['metadata_text_candidate_rows']}`",
        f"- db_scenario_tag candidates: `{summary['db_scenario_tag_candidate_rows']}`",
        f"- raw_db_scenario_tag_rows: `{summary.get('raw_db_scenario_tag_rows', 0)}`",
        f"- unique_scenario_token_rows: `{summary.get('unique_scenario_token_rows', 0)}`",
        f"- duplicate_scenario_token_count_removed: `{summary.get('duplicate_scenario_token_count_removed', 0)}`",
        f"- selected_scenario_type_counts: `{summary.get('selected_scenario_type_counts', {})}`",
        f"- selected_actual_scenario_type_counts: `{summary.get('selected_actual_scenario_type_counts', {})}`",
        f"- selected_db_tag_only_rows: `{summary.get('selected_db_tag_only_rows', 0)}`",
        f"- selected_actual_type_verified_rows: `{summary.get('selected_actual_type_verified_rows', 0)}`",
        f"- selected_actual_type_empty_rows: `{summary.get('selected_actual_type_empty_rows', 0)}`",
        f"- actual_type_verification_failed_rows: `{summary.get('actual_type_verification_failed_rows', 0)}`",
        f"- selected_fallback_lateral_rows: `{summary.get('selected_fallback_lateral_rows', 0)}`",
        f"- strict_db_tag_candidate_rows: `{summary.get('strict_db_tag_candidate_rows', 0)}`",
        f"- strict_actual_type_verified_rows: `{summary.get('strict_actual_type_verified_rows', 0)}`",
        f"- strict_actual_type_unverified_but_db_tag_selected_rows: `{summary.get('strict_actual_type_unverified_but_db_tag_selected_rows', 0)}`",
        f"- strict_actual_type_rejected_rows: `{summary.get('strict_actual_type_rejected_rows', 0)}`",
        f"- strict_db_tag_candidates_exist_but_none_selected: `{summary.get('strict_db_tag_candidates_exist_but_none_selected', False)}`",
        f"- insufficient_strict_changing_lane_warning: `{summary.get('insufficient_strict_changing_lane_warning', '')}`",
        f"- selected_log_counts: `{summary.get('selected_log_counts', {})}`",
        f"- strict_changing_lane_candidate_rows: `{summary.get('strict_changing_lane_candidate_rows', 0)}`",
        f"- strict_db_tag_candidate_rows: `{summary.get('strict_db_tag_candidate_rows', 0)}`",
        f"- strict_actual_type_verified_rows: `{summary.get('strict_actual_type_verified_rows', 0)}`",
        f"- strict_actual_type_unverified_but_db_tag_selected_rows: `{summary.get('strict_actual_type_unverified_but_db_tag_selected_rows', 0)}`",
        f"- strict_actual_type_rejected_rows: `{summary.get('strict_actual_type_rejected_rows', 0)}`",
        f"- selected_strict_changing_lane_rows: `{summary.get('selected_strict_changing_lane_rows', 0)}`",
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
        "- Optional DB scenario-tag scan reads nuPlan mini DB `scenario_tag.type` directly and joins `lidar_pc` / `log` for Stage7C-friendly context.",
        "- DB `scenario_tag.type` is only a candidate label; it is not the final `actual_scenario_type` that official nuPlan scenario filtering/building may resolve.",
        "- With `--verify_actual_scenario_type`, verified actual types in `changing_lane` / `changing_lane_to_left` / `changing_lane_to_right` enter the strict lane-change set.",
        "- If actual-type verification fails or returns empty and `--allow_db_tag_when_actual_type_unverified` is true, strict changing-lane DB tags are retained as DB-tag strict but actual-type-unverified rows with `actual_type_verified=false`.",
        "- If actual-type verification explicitly returns a non-lane-change type, the strict DB-tag row is rejected.",
        "- Verified fallback lateral rows are reported separately with `selected_as_fallback_lateral=true`; they are never counted as strict lane-change rows.",
        "",
        "## Candidate source counts",
        f"- text_match_candidates: `{summary['text_match_candidates']}`",
        f"- behavior_event_candidates: `{summary['behavior_event_candidates']}`",
        f"- metadata_text candidates: `{summary['metadata_text_candidate_rows']}`",
        f"- db_scenario_tag candidates: `{summary['db_scenario_tag_candidate_rows']}`",
        f"- strict_changing_lane_candidate_rows: `{summary.get('strict_changing_lane_candidate_rows', 0)}`",
        f"- strict_db_tag_candidate_rows: `{summary.get('strict_db_tag_candidate_rows', 0)}`",
        f"- strict_actual_type_verified_rows: `{summary.get('strict_actual_type_verified_rows', 0)}`",
        f"- strict_actual_type_unverified_but_db_tag_selected_rows: `{summary.get('strict_actual_type_unverified_but_db_tag_selected_rows', 0)}`",
        f"- strict_actual_type_rejected_rows: `{summary.get('strict_actual_type_rejected_rows', 0)}`",
        f"- selected_strict_changing_lane_rows: `{summary.get('selected_strict_changing_lane_rows', 0)}`",
        f"- kinematic_candidates: `{summary['kinematic_candidates']}`",
        f"- final_selected_candidates: `{summary['final_selected_candidates']}`",
        "",
        "## Warnings",
    ]
    warnings = (
        list(summary.get("warnings", []))
        + list(summary.get("db_scenario_tag_scan", {}).get("warnings", []))
        + list(summary.get("kinematic_scan", {}).get("warnings", []))
    )
    if warnings:
        lines.extend([f"- WARNING: {w}" for w in warnings])
    else:
        lines.append("- None.")
    if summary.get("metadata_text_candidate_rows") == 0 and summary.get("db_scenario_tag_candidate_rows", 0) > 0:
        lines.extend([
            "",
            "## Metadata-vs-DB interpretation",
            f"- metadata_text candidates: `{summary['metadata_text_candidate_rows']}`",
            f"- db_scenario_tag candidates: `{summary['db_scenario_tag_candidate_rows']}`",
            "- This means the original Stage7B merged subset is not lane-change-rich, but mini DB contains lane-change/lateral candidate tags.",
        ])
    lines.extend([
        "",
        "## Strict lane-change selection interpretation",
        f"- verified strict lane-change candidates: `{summary.get('selected_actual_type_verified_rows', 0)}`",
        f"- DB-tag-only strict lane-change candidates: `{summary.get('selected_db_tag_only_rows', 0)}`",
        "- Rows are called verified strict only when `actual_type_verified=true` and `actual_scenario_type` is non-empty.",
        "- Rows with empty/unavailable `actual_scenario_type` are DB-tag-only strict candidates, not verified strict candidates.",
        "",
        "## Top candidates",
    ])
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
    db_scenario_tag_candidates, db_scenario_tag_summary = scan_db_scenario_tags(args)
    kinematic_candidates, kinematic_summary = run_kinematic_scan(args)
    raw_candidates = text_event_candidates + db_scenario_tag_candidates + kinematic_candidates
    for row in raw_candidates:
        row.setdefault("scenario_type_db_tag", row.get("scenario_type", ""))
        row.setdefault("actual_scenario_type", "")
        row.setdefault("actual_type_verified", "false")
        row.setdefault("actual_type_verification_method", "")
        row.setdefault("actual_type_verification_error", "")
        row.setdefault("selected_by_db_tag_only", "false")
        row.setdefault("selected_as_strict_changing_lane", "false")
        row.setdefault("selected_as_fallback_lateral", "false")
        row.setdefault("candidate_source", row.get("source", ""))
        row.setdefault("source", row.get("candidate_source", ""))
        if not str(row.get("log_name", "") or "").strip():
            row["log_name"] = str(row.get("db_file", "") or row.get("scenario_id", "unknown_log")).strip() or "unknown_log"
    raw_candidate_rows = len(raw_candidates)
    best_by_token: Dict[str, Dict[str, Any]] = {}
    duplicate_all = 0
    for row in raw_candidates:
        token = str(row.get("scenario_token", "") or row.get("lidar_pc_token", "") or row.get("scenario_id", "") or "")
        if token:
            row["scenario_token"] = token
            row.setdefault("lidar_pc_token", token)
            current = best_by_token.get(token)
            row_pri = min((rank for term, rank in DB_SCENARIO_TYPE_PRIORITY.items() if term in normalize_scenario_type(row.get("scenario_type"))), default=99)
            if current is None:
                best_by_token[token] = row
            else:
                duplicate_all += 1
                cur_pri = min((rank for term, rank in DB_SCENARIO_TYPE_PRIORITY.items() if term in normalize_scenario_type(current.get("scenario_type"))), default=99)
                if (row_pri, -float(row.get("candidate_score") or row.get("match_score") or 0)) < (cur_pri, -float(current.get("candidate_score") or current.get("match_score") or 0)):
                    best_by_token[token] = row
        else:
            best_by_token[f"__row_{len(best_by_token)}"] = row
    candidates = list(best_by_token.values())
    def candidate_sort_key(r: Dict[str, Any]) -> Tuple[int, float, str, str]:
        source = str(r.get("candidate_source", "") or r.get("source", ""))
        if source == "db_scenario_tag" or bool(getattr(args, "prefer_exact_changing_lane", False)) or bool(getattr(args, "verify_actual_scenario_type", False)):
            priority = min((rank for term, rank in DB_SCENARIO_TYPE_PRIORITY.items() if term in normalize_scenario_type(r.get("scenario_type"))), default=99)
        else:
            priority = 99
        return (priority, -float(r.get("candidate_score") or r.get("match_score") or 0), str(r.get("log_name", "")), str(r.get("scenario_token", "")))

    candidates.sort(key=candidate_sort_key)
    if bool(getattr(args, "verify_actual_scenario_type", False)):
        verify_actual_scenario_types(candidates)
    for rank, row in enumerate(candidates, 1):
        row["candidate_rank"] = rank
    max_per_log = int(getattr(args, "max_per_log", 2) or 0)
    selected_k = int(getattr(args, "verified_top_k", None) or args.top_k)
    actual_allowlist = parse_csv_set(getattr(args, "actual_type_allowlist", "")) or STRICT_CHANGING_LANE_TYPES
    fallback_allowlist = parse_csv_set(getattr(args, "fallback_type_allowlist", "")) or {"high_lateral_acceleration"}
    top = select_top_candidates(
        candidates,
        selected_k,
        max_per_log,
        bool(getattr(args, "prefer_exact_changing_lane", False)),
        verify_actual_scenario_type=bool(getattr(args, "verify_actual_scenario_type", False)),
        actual_type_allowlist=actual_allowlist,
        allow_fallback_lateral_types=bool(getattr(args, "allow_fallback_lateral_types", False)),
        fallback_type_allowlist=fallback_allowlist,
        allow_db_tag_when_actual_type_unverified=bool(getattr(args, "allow_db_tag_when_actual_type_unverified", True)),
        require_actual_type_verified=bool(getattr(args, "require_actual_type_verified", False)),
    )
    output_fields = list(dict.fromkeys([*DB_OUTPUT_FIELDS, *KINEMATIC_OUTPUT_FIELDS, *fieldnames]))
    write_csv(out / "lane_change_candidate_metadata.csv", top, output_fields)
    text_match_count = sum(1 for r in text_event_candidates if int(r.get("metadata_match_score") or 0) > 0)
    behavior_count = sum(1 for r in text_event_candidates if int(r.get("event_task_lane_change") or 0) > 0)
    stage7c_context_path = ""
    if getattr(args, "write_stage7c_context_dir", False) and top:
        stage7c_context_path = write_stage7c_context(out, top, fieldnames)
    warnings = []
    if not text_event_candidates and not db_scenario_tag_candidates and not kinematic_candidates:
        warnings.append("candidate_rows=0 because metadata text matching and available behavior/kinematic scans found no lane-change-like rows; this does not diagnose PDM lane-change capability.")
    strict_db_tag_candidate_rows = sum(1 for r in candidates if is_strict_changing_lane_row(r) and str(r.get("scenario_token", "") or "").strip() and str(r.get("log_name", "") or "").strip())
    strict_actual_type_verified_rows = sum(1 for r in top if str(r.get("selected_as_strict_changing_lane", "") or "").lower() == "true" and str(r.get("actual_type_verified", "") or "").lower() == "true")
    strict_actual_type_unverified_but_db_tag_selected_rows = sum(1 for r in top if str(r.get("selected_as_strict_changing_lane", "") or "").lower() == "true" and str(r.get("actual_type_verified", "") or "").lower() != "true" and is_strict_changing_lane_row(r))
    strict_actual_type_rejected_rows = sum(1 for r in candidates if is_strict_changing_lane_row(r) and actual_type_rejected(r, actual_allowlist, fallback_allowlist if bool(getattr(args, "allow_fallback_lateral_types", False)) else None))
    strict_db_tag_candidates_exist_but_none_selected = bool(strict_db_tag_candidate_rows > 0 and len(top) == 0)
    if strict_db_tag_candidates_exist_but_none_selected:
        warnings.append("strict_db_tag_candidates_exist_but_none_selected=true; insufficient strict changing-lane candidates selected. Check actual-type verification and allow_db_tag_when_actual_type_unverified.")
    if getattr(args, "write_stage7c_context_dir", False) and not top:
        warnings.append("insufficient candidates: stage7c_candidate_context/merged_metadata.csv was not written because final_selected_rows=0.")
    summary = {
        "context_dir": str(context_dir),
        "metadata_path": str(metadata_path),
        "metadata_rows": int(len(metadata)),
        "raw_candidate_rows": int(raw_candidate_rows),
        "candidate_rows": int(len(candidates)),
        "metadata_text_candidate_rows": int(text_match_count),
        "behavior_event_candidate_rows": int(behavior_count),
        "db_scenario_tag_candidate_rows": int(len(db_scenario_tag_candidates)),
        "final_candidate_rows": int(len(candidates)),
        "unique_scenario_token_rows": int(len(candidates)),
        "actual_type_verified_rows": int(sum(1 for r in candidates if str(r.get("actual_type_verified", "")).lower() == "true")),
        "actual_type_verification_failed_rows": int(sum(1 for r in candidates if bool(getattr(args, "verify_actual_scenario_type", False)) and str(r.get("actual_type_verified", "")).lower() != "true")),
        "strict_changing_lane_actual_type_rows": int(sum(1 for r in candidates if actual_type_allowed(r, actual_allowlist))),
        "strict_db_tag_candidate_rows": int(strict_db_tag_candidate_rows),
        "strict_actual_type_verified_rows": int(strict_actual_type_verified_rows),
        "strict_actual_type_unverified_but_db_tag_selected_rows": int(strict_actual_type_unverified_but_db_tag_selected_rows),
        "strict_actual_type_rejected_rows": int(strict_actual_type_rejected_rows),
        "strict_db_tag_candidates_exist_but_none_selected": bool(strict_db_tag_candidates_exist_but_none_selected),
        "allow_db_tag_when_actual_type_unverified": bool(getattr(args, "allow_db_tag_when_actual_type_unverified", True)),
        "require_actual_type_verified": bool(getattr(args, "require_actual_type_verified", False)),
        "fallback_lateral_actual_type_rows": int(sum(1 for r in candidates if actual_type_allowed(r, fallback_allowlist))),
        "final_selected_rows": int(len(top)),
        "selected_fallback_lateral_rows": int(sum(1 for r in top if str(r.get("selected_as_fallback_lateral", "")).lower() == "true")),
        "text_match_candidates": int(text_match_count),
        "behavior_event_candidates": int(behavior_count),
        "db_scenario_tag_candidates": int(len(db_scenario_tag_candidates)),
        "kinematic_candidates": int(len(kinematic_candidates)),
        "final_selected_candidates": int(len(top)),
        "scenario_type_counts": count_by_field(candidates, "scenario_type"),
        "selected_scenario_type_db_tag_counts": count_by_field(top, "scenario_type_db_tag"),
        "selected_actual_scenario_type_counts": count_by_field_nonempty(top, "actual_scenario_type"),
        "selected_db_tag_only_rows": int(sum(1 for r in top if str(r.get("selected_by_db_tag_only", "") or "").lower() == "true")),
        "selected_actual_type_verified_rows": int(sum(1 for r in top if str(r.get("actual_type_verified", "") or "").lower() == "true")),
        "selected_actual_type_empty_rows": int(sum(1 for r in top if not str(r.get("actual_scenario_type", "") or "").strip())),
        "selected_scenario_type_counts": count_by_field(top, "scenario_type"),
        "top_k_requested": int(args.top_k),
        "top_k_written": int(len(top)),
        "preferred_scenario_type_terms": PREFERRED_SCENARIO_TYPE_TERMS,
        "behavior_events": event_summary,
        "db_scenario_tag_scan": db_scenario_tag_summary,
        "raw_db_scenario_tag_rows": int(db_scenario_tag_summary.get("raw_db_scenario_tag_rows", 0)),
        "selected_rows": int(db_scenario_tag_summary.get("selected_rows", 0)),
        "selected_log_counts": count_by_field(top, "log_name"),
        "strict_changing_lane_candidate_rows": int(sum(1 for r in candidates if is_strict_changing_lane_row(r))),
        "selected_strict_changing_lane_rows": int(sum(1 for r in top if str(r.get("selected_as_strict_changing_lane", "")).lower() == "true")),
        "prefer_exact_changing_lane": bool(getattr(args, "prefer_exact_changing_lane", False)),
        "duplicate_scenario_token_count_removed": int(duplicate_all + int(db_scenario_tag_summary.get("duplicate_scenario_token_count_removed", 0))),
        "insufficient_strict_changing_lane_warning": ("strict_db_tag_candidates_exist_but_none_selected=true" if strict_db_tag_candidates_exist_but_none_selected else ("" if (not bool(getattr(args, "verify_actual_scenario_type", False)) or sum(1 for r in top if str(r.get("selected_as_strict_changing_lane", "")).lower() == "true") >= min(selected_k, max(strict_db_tag_candidate_rows, 1))) else f"strict changing-lane rows selected {sum(1 for r in top if str(r.get('selected_as_strict_changing_lane', '')).lower() == 'true')} < requested {selected_k}")),
        "kinematic_scan": kinematic_summary,
        "warnings": warnings,
        "outputs": {
            "report": "lane_change_candidate_report.md",
            "summary": "lane_change_candidate_summary.json",
            "metadata": "lane_change_candidate_metadata.csv",
            "stage7c_context": stage7c_context_path,
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
    parser.add_argument("--scan_db_scenario_tags", action="store_true", help="Scan nuPlan mini DB scenario_tag.type directly for lane-change/lateral candidate tags.")
    parser.add_argument("--max_db_files", type=int, default=0, help="Optional cap on the number of nuPlan .db files scanned by --scan_db_scenario_tags.")
    parser.add_argument("--max_candidates_per_type", type=int, default=0, help="Optional cap on DB scenario-tag candidates retained for each scenario_tag.type.")
    parser.add_argument("--max_per_log", type=int, default=2, help="Maximum candidates selected per log before writing top_k outputs; use 0 to disable.")
    parser.add_argument("--prefer_exact_changing_lane", action="store_true", help="Prioritize exact changing_lane_to_left/right/changing_lane scenario_tag types before fallback lateral/cut-in/merge types during top_k selection.")
    parser.add_argument("--write_stage7c_context_dir", action="store_true", help="Write output_dir/stage7c_candidate_context/merged_metadata.csv for Stage7C.")
    parser.add_argument("--nuplan_map_root", default="", help="nuPlan map root reserved for scenario-builder based scans; SQLite fallback does not require maps.")
    parser.add_argument("--max_scenarios_scan", type=int, default=50, help="Maximum DB-derived scenarios / pose windows to inspect during kinematic scan.")
    parser.add_argument("--enable_kinematic_scan", action="store_true", help="Enable trajectory-driven high-lateral-motion candidate discovery from nuPlan DB pose-like tables.")
    parser.add_argument("--min_lateral_displacement", type=float, default=2.0, help="Minimum absolute lateral displacement in the start ego frame for kinematic candidate selection.")
    parser.add_argument("--min_heading_change", type=float, default=0.25, help="Minimum absolute heading change in radians for kinematic candidate selection.")
    parser.add_argument("--min_yaw_rate_proxy", type=float, default=0.05, help="Minimum heading-change-over-duration proxy for kinematic candidate selection.")
    parser.add_argument("--verify_actual_scenario_type", action="store_true", help="Verify exact-token actual scenario_type before selecting strict lane-change rows.")
    parser.add_argument("--actual_type_allowlist", default="changing_lane,changing_lane_to_left,changing_lane_to_right", help="Comma-separated verified actual scenario types allowed in the strict selected set.")
    parser.add_argument("--allow_fallback_lateral_types", action="store_true", help="Allow verified fallback lateral actual types to fill remaining slots after strict lane-change rows.")
    parser.add_argument("--allow_db_tag_when_actual_type_unverified", action="store_true", default=True, help="When actual-type verification fails or returns empty, keep strict changing_lane DB-tag rows in the strict selected set instead of dropping them.")
    parser.add_argument("--fallback_type_allowlist", default="high_lateral_acceleration", help="Comma-separated verified actual scenario types allowed only as fallback rows.")
    parser.add_argument("--verified_top_k", type=int, default=None, help="Optional selected-row cap for verified mode; defaults to --top_k.")
    parser.add_argument("--require_actual_type_verified", action="store_true", help="In verified mode, drop rows whose actual_scenario_type could not be verified instead of selecting DB-tag-only rows.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
