#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


EGO_COLUMNS = [
    "db_name",
    "scene_token",
    "scene_name",
    "lidar_pc_token",
    "lidar_pc_timestamp",
    "ego_pose_token",
    "ego_timestamp",
    "ego_x",
    "ego_y",
    "ego_z",
    "ego_heading",
    "ego_qw",
    "ego_qx",
    "ego_qy",
    "ego_qz",
    "ego_vx",
    "ego_vy",
    "ego_speed",
    "ego_accel",
    "ego_yaw_rate",
    "frame_index_in_scene",
]

OBJECT_COLUMNS = [
    "db_name",
    "scene_token",
    "scene_name",
    "lidar_pc_token",
    "lidar_pc_timestamp",
    "frame_index_in_scene",
    "track_token",
    "category_name",
    "object_x",
    "object_y",
    "object_z",
    "object_heading",
    "object_length",
    "object_width",
    "object_height",
    "relative_x",
    "relative_y",
    "relative_distance",
    "rank_by_distance",
]

SCENE_COLUMNS = [
    "db_name",
    "scene_token",
    "scene_name",
    "num_lidar_pcs_exported",
    "num_ego_rows",
    "num_object_rows",
    "duration_sec_estimate",
]

REQUIRED_TABLES = ["scene", "lidar_pc", "ego_pose", "lidar_box", "track", "category"]


class WarningCollector:
    def __init__(self) -> None:
        self.items: List[Dict[str, Any]] = []

    def add(self, code: str, message: str, db_name: Optional[str] = None, scene_token: Optional[str] = None) -> None:
        item: Dict[str, Any] = {"code": code, "message": message}
        if db_name:
            item["db_name"] = db_name
        if scene_token:
            item["scene_token"] = scene_token
        self.items.append(item)


class SchemaInfo:
    def __init__(self, tables: Dict[str, List[str]]) -> None:
        self.tables = tables

    def has_table(self, table: str) -> bool:
        return table in self.tables

    def columns(self, table: str) -> List[str]:
        return self.tables.get(table, [])

    def has_column(self, table: str, column: str) -> bool:
        return column in self.columns(table)

    def first_column(self, table: str, candidates: Sequence[str]) -> Optional[str]:
        cols = set(self.columns(table))
        for candidate in candidates:
            if candidate in cols:
                return candidate
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 7A.1 exporter for nuPlan mini expert ego trajectories and nearby object context. "
            "This script reads SQLite DBs directly, does not run planner simulation, and does not generate rollouts."
        )
    )
    parser.add_argument(
        "--nuplan_data_root",
        default=os.environ.get("NUPLAN_DATA_ROOT"),
        help="nuPlan data root. Default: NUPLAN_DATA_ROOT environment variable.",
    )
    parser.add_argument(
        "--mini_db_dir",
        default=None,
        help=(
            "Optional mini DB directory. If omitted, tries "
            "$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/mini then $NUPLAN_DATA_ROOT/data/cache/mini."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/stage7A_nuplan/expert_context_export",
        help="Directory for exported CSV/JSON/Markdown files.",
    )
    parser.add_argument("--max_dbs", type=int, default=5, help="Maximum number of mini DBs to export.")
    parser.add_argument("--max_scenes_per_db", type=int, default=5, help="Maximum scenes per DB to export.")
    parser.add_argument(
        "--max_lidar_pcs_per_scene", type=int, default=200, help="Maximum lidar_pc frames per scene to export."
    )
    parser.add_argument("--num_neighbors", type=int, default=10, help="Nearest objects retained per lidar_pc frame.")
    parser.add_argument("--overwrite", action="store_true", help="Remove output_dir before writing if it exists.")
    return parser.parse_args()


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def sql_value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.hex()
    return str(value)


def to_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bytes):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def normalize_timestamp_seconds(timestamp: Optional[float]) -> Optional[float]:
    if timestamp is None:
        return None
    abs_ts = abs(timestamp)
    if abs_ts > 1.0e17:
        return timestamp / 1.0e9
    if abs_ts > 1.0e12:
        return timestamp / 1.0e6
    if abs_ts > 1.0e10:
        return timestamp / 1.0e3
    return timestamp


def signed_angle_delta(a: float, b: float) -> float:
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion(qw: Optional[float], qx: Optional[float], qy: Optional[float], qz: Optional[float]) -> Optional[float]:
    if qw is None or qx is None or qy is None or qz is None:
        return None
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def resolve_mini_db_dir(data_root: Optional[str], explicit_dir: Optional[str], warnings: WarningCollector) -> Optional[Path]:
    if explicit_dir:
        return Path(explicit_dir).expanduser()
    if not data_root:
        warnings.add(
            "missing_nuplan_data_root",
            "Cannot infer mini DB directory because --nuplan_data_root is unset and NUPLAN_DATA_ROOT is not defined.",
        )
        return None
    root = Path(data_root).expanduser()
    candidates = [root / "nuplan-v1.1" / "splits" / "mini", root / "data" / "cache" / "mini"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    warnings.add(
        "mini_db_dir_candidates_missing",
        "No default mini DB directory exists under --nuplan_data_root. Tried: " + ", ".join(str(p) for p in candidates),
    )
    return candidates[0]


def find_db_files(mini_db_dir: Optional[Path], max_dbs: int, warnings: WarningCollector) -> List[Path]:
    if mini_db_dir is None:
        warnings.add("mini_db_dir_missing", "Mini DB directory is not configured.")
        return []
    if not mini_db_dir.exists():
        warnings.add("mini_db_dir_missing", f"Mini DB directory does not exist: {mini_db_dir}")
        return []
    if not mini_db_dir.is_dir():
        warnings.add("mini_db_dir_not_directory", f"Mini DB path is not a directory: {mini_db_dir}")
        return []
    dbs = sorted(
        path for path in mini_db_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}
    )
    if max_dbs > 0:
        dbs = dbs[:max_dbs]
    if not dbs:
        warnings.add("no_mini_db_files", f"No SQLite DB files found under: {mini_db_dir}")
    return dbs


def inspect_schema(conn: sqlite3.Connection) -> SchemaInfo:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    tables: Dict[str, List[str]] = {}
    for row in rows:
        table = row[0]
        info = conn.execute(f"PRAGMA table_info({quote_identifier(table)})").fetchall()
        tables[table] = [str(item[1]) for item in info]
    return SchemaInfo(tables)


def select_rows(
    conn: sqlite3.Connection,
    table: str,
    columns: Sequence[str],
    where: str = "",
    params: Sequence[Any] = (),
    order_by: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[sqlite3.Row]:
    cols = ", ".join(quote_identifier(c) for c in columns)
    sql = f"SELECT {cols} FROM {quote_identifier(table)}"
    if where:
        sql += f" WHERE {where}"
    if order_by:
        sql += f" ORDER BY {order_by}"
    if limit is not None and limit > 0:
        sql += f" LIMIT {int(limit)}"
    return list(conn.execute(sql, params))


def get_scenes(conn: sqlite3.Connection, schema: SchemaInfo, limit: int) -> List[sqlite3.Row]:
    columns = schema.columns("scene")
    order_col = schema.first_column("scene", ["name", "token", "id"])
    order_by = quote_identifier(order_col) if order_col else None
    return select_rows(conn, "scene", columns, order_by=order_by, limit=limit)


def row_get(row: sqlite3.Row, column: Optional[str]) -> Any:
    if column is None:
        return None
    try:
        return row[column]
    except (KeyError, IndexError):
        return None


def fetch_one_by_token(
    conn: sqlite3.Connection, schema: SchemaInfo, table: str, token_column: Optional[str], token_value: Any
) -> Optional[sqlite3.Row]:
    if token_column is None or token_value is None or not schema.has_table(table):
        return None
    rows = select_rows(
        conn,
        table,
        schema.columns(table),
        where=f"{quote_identifier(token_column)} = ?",
        params=(token_value,),
        limit=1,
    )
    return rows[0] if rows else None


def build_token_index(
    conn: sqlite3.Connection, schema: SchemaInfo, table: str, token_column: Optional[str]
) -> Dict[Any, sqlite3.Row]:
    if token_column is None or not schema.has_table(table):
        return {}
    rows = select_rows(conn, table, schema.columns(table))
    return {row[token_column]: row for row in rows}


def resolve_lidar_sequence(
    conn: sqlite3.Connection,
    schema: SchemaInfo,
    scene: sqlite3.Row,
    max_frames: int,
    db_name: str,
    warnings: WarningCollector,
) -> Tuple[List[sqlite3.Row], str]:
    scene_cols = schema.columns("scene")
    lidar_cols = schema.columns("lidar_pc")
    scene_token_col = schema.first_column("scene", ["token", "scene_token"])
    scene_token_value = row_get(scene, scene_token_col)

    lidar_scene_col = schema.first_column("lidar_pc", ["scene_token", "scene", "scene_id"])
    if lidar_scene_col and scene_token_value is not None:
        order_col = schema.first_column("lidar_pc", ["timestamp", "time_us", "utime", "id", "token"])
        order_by = quote_identifier(order_col) if order_col else None
        rows = select_rows(
            conn,
            "lidar_pc",
            lidar_cols,
            where=f"{quote_identifier(lidar_scene_col)} = ?",
            params=(scene_token_value,),
            order_by=order_by,
            limit=max_frames,
        )
        if rows:
            return rows, f"direct lidar_pc.{lidar_scene_col} = scene.{scene_token_col}"

    start_col = schema.first_column(
        "scene", ["start_lidar_pc_token", "first_lidar_pc_token", "initial_lidar_pc_token", "lidar_pc_token"]
    )
    end_col = schema.first_column("scene", ["end_lidar_pc_token", "last_lidar_pc_token", "final_lidar_pc_token"])
    lidar_token_col = schema.first_column("lidar_pc", ["token", "lidar_pc_token"])
    next_col = schema.first_column("lidar_pc", ["next_token", "next_lidar_pc_token"])
    if start_col and lidar_token_col:
        start_token = row_get(scene, start_col)
        end_token = row_get(scene, end_col)
        if start_token is not None:
            token_index = build_token_index(conn, schema, "lidar_pc", lidar_token_col)
            sequence: List[sqlite3.Row] = []
            current = start_token
            seen = set()
            while current is not None and current not in seen and len(sequence) < max_frames:
                seen.add(current)
                row = token_index.get(current)
                if row is None:
                    warnings.add(
                        "lidar_pc_chain_token_missing",
                        f"Scene references lidar_pc token that is absent from lidar_pc table via scene.{start_col}/{next_col}.",
                        db_name=db_name,
                        scene_token=sql_value_to_text(scene_token_value),
                    )
                    break
                sequence.append(row)
                if end_token is not None and current == end_token:
                    break
                if next_col is None:
                    break
                current = row_get(row, next_col)
            if sequence:
                if next_col is None:
                    warnings.add(
                        "lidar_pc_chain_without_next_token",
                        "Scene has a start lidar_pc token, but lidar_pc has no next_token column; only the start frame was exported.",
                        db_name=db_name,
                        scene_token=sql_value_to_text(scene_token_value),
                    )
                return sequence, f"scene.{start_col} chained through lidar_pc.{next_col or lidar_token_col}"

    lidar_ts_col = schema.first_column("lidar_pc", ["timestamp", "time_us", "utime", "time_stamp"])
    start_time_col = schema.first_column("scene", ["start_time", "start_timestamp", "start_time_us", "first_timestamp"])
    end_time_col = schema.first_column("scene", ["end_time", "end_timestamp", "end_time_us", "last_timestamp"])
    if lidar_ts_col and start_time_col and end_time_col:
        start_time = row_get(scene, start_time_col)
        end_time = row_get(scene, end_time_col)
        if start_time is not None and end_time is not None:
            rows = select_rows(
                conn,
                "lidar_pc",
                lidar_cols,
                where=f"{quote_identifier(lidar_ts_col)} >= ? AND {quote_identifier(lidar_ts_col)} <= ?",
                params=(start_time, end_time),
                order_by=quote_identifier(lidar_ts_col),
                limit=max_frames,
            )
            if rows:
                return rows, f"timestamp window scene.{start_time_col}/{end_time_col} -> lidar_pc.{lidar_ts_col}"

    candidate_scene_cols = ", ".join(scene_cols) if scene_cols else "<none>"
    warnings.add(
        "uncertain_scene_lidar_join",
        "Could not confidently join this scene to lidar_pc rows. scene columns: " + candidate_scene_cols,
        db_name=db_name,
        scene_token=sql_value_to_text(scene_token_value),
    )
    return [], "unresolved; no direct scene/lidar_pc relation, chain, or timestamp window found"


def extract_pose_fields(row: Optional[sqlite3.Row], schema: SchemaInfo, table: str) -> Dict[str, Optional[float]]:
    if row is None:
        return {key: None for key in ["x", "y", "z", "qw", "qx", "qy", "qz", "heading", "vx", "vy", "timestamp"]}
    x_col = schema.first_column(table, ["x", "translation_x", "ego_x", "center_x"])
    y_col = schema.first_column(table, ["y", "translation_y", "ego_y", "center_y"])
    z_col = schema.first_column(table, ["z", "translation_z", "ego_z", "center_z"])
    qw_col = schema.first_column(table, ["qw", "rotation_w", "orientation_w"])
    qx_col = schema.first_column(table, ["qx", "rotation_x", "orientation_x"])
    qy_col = schema.first_column(table, ["qy", "rotation_y", "orientation_y"])
    qz_col = schema.first_column(table, ["qz", "rotation_z", "orientation_z"])
    heading_col = schema.first_column(table, ["heading", "yaw", "ego_heading"])
    vx_col = schema.first_column(table, ["vx", "velocity_x", "ego_vx"])
    vy_col = schema.first_column(table, ["vy", "velocity_y", "ego_vy"])
    ts_col = schema.first_column(table, ["timestamp", "time_us", "utime", "time_stamp"])
    qw = to_float(row_get(row, qw_col))
    qx = to_float(row_get(row, qx_col))
    qy = to_float(row_get(row, qy_col))
    qz = to_float(row_get(row, qz_col))
    heading = to_float(row_get(row, heading_col))
    if heading is None:
        heading = yaw_from_quaternion(qw, qx, qy, qz)
    return {
        "x": to_float(row_get(row, x_col)),
        "y": to_float(row_get(row, y_col)),
        "z": to_float(row_get(row, z_col)),
        "qw": qw,
        "qx": qx,
        "qy": qy,
        "qz": qz,
        "heading": heading,
        "vx": to_float(row_get(row, vx_col)),
        "vy": to_float(row_get(row, vy_col)),
        "timestamp": to_float(row_get(row, ts_col)),
    }


def format_optional(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.9g}"


def compute_ego_kinematics(ego_rows: List[Dict[str, Any]], warnings: WarningCollector, db_name: str, scene_token: str) -> None:
    computed_speed_count = 0
    for index, row in enumerate(ego_rows):
        vx = row.get("_ego_vx")
        vy = row.get("_ego_vy")
        if vx is not None and vy is not None:
            row["ego_speed"] = math.hypot(vx, vy)
            computed_speed_count += 1
            continue
        if 0 < index < len(ego_rows):
            prev = ego_rows[index - 1]
            x0 = prev.get("_ego_x")
            y0 = prev.get("_ego_y")
            t0 = prev.get("_ego_time_sec")
            x1 = row.get("_ego_x")
            y1 = row.get("_ego_y")
            t1 = row.get("_ego_time_sec")
            if None not in (x0, y0, t0, x1, y1, t1) and t1 > t0:
                row["ego_speed"] = math.hypot(x1 - x0, y1 - y0) / (t1 - t0)
                computed_speed_count += 1
    if computed_speed_count == 0 and ego_rows:
        warnings.add(
            "cannot_compute_speed",
            "Ego speed could not be computed because velocity columns or adjacent timestamped ego poses were unavailable.",
            db_name=db_name,
            scene_token=scene_token,
        )

    accel_count = 0
    yaw_rate_count = 0
    for index in range(1, len(ego_rows)):
        prev = ego_rows[index - 1]
        row = ego_rows[index]
        t0 = prev.get("_ego_time_sec")
        t1 = row.get("_ego_time_sec")
        if t0 is None or t1 is None or t1 <= t0:
            continue
        speed0 = prev.get("ego_speed")
        speed1 = row.get("ego_speed")
        if isinstance(speed0, (int, float)) and isinstance(speed1, (int, float)):
            row["ego_accel"] = (speed1 - speed0) / (t1 - t0)
            accel_count += 1
        yaw0 = prev.get("_ego_heading")
        yaw1 = row.get("_ego_heading")
        if yaw0 is not None and yaw1 is not None:
            row["ego_yaw_rate"] = signed_angle_delta(yaw1, yaw0) / (t1 - t0)
            yaw_rate_count += 1
    if accel_count == 0 and len(ego_rows) > 1:
        warnings.add(
            "cannot_compute_accel",
            "Ego acceleration could not be computed from exported frames.",
            db_name=db_name,
            scene_token=scene_token,
        )
    if yaw_rate_count == 0 and len(ego_rows) > 1:
        warnings.add(
            "cannot_compute_yaw_rate",
            "Ego yaw rate could not be computed from exported frames.",
            db_name=db_name,
            scene_token=scene_token,
        )


def load_categories(conn: sqlite3.Connection, schema: SchemaInfo) -> Dict[Any, str]:
    token_col = schema.first_column("category", ["token", "category_token"])
    name_col = schema.first_column("category", ["name", "category_name", "description"])
    if token_col is None or name_col is None or not schema.has_table("category"):
        return {}
    rows = select_rows(conn, "category", schema.columns("category"))
    return {row[token_col]: sql_value_to_text(row[name_col]) for row in rows}


def fetch_lidar_boxes(
    conn: sqlite3.Connection,
    schema: SchemaInfo,
    lidar_pc_token: Any,
    max_neighbors: int,
    ego_x: Optional[float],
    ego_y: Optional[float],
) -> Tuple[List[sqlite3.Row], Optional[str]]:
    lidar_box_lidar_col = schema.first_column("lidar_box", ["lidar_pc_token", "lidar_token", "lidar_pc"])
    if lidar_box_lidar_col is None or lidar_pc_token is None:
        return [], lidar_box_lidar_col
    rows = select_rows(
        conn,
        "lidar_box",
        schema.columns("lidar_box"),
        where=f"{quote_identifier(lidar_box_lidar_col)} = ?",
        params=(lidar_pc_token,),
    )
    if ego_x is None or ego_y is None:
        return rows[:max_neighbors], lidar_box_lidar_col
    x_col = schema.first_column("lidar_box", ["x", "translation_x", "center_x"])
    y_col = schema.first_column("lidar_box", ["y", "translation_y", "center_y"])
    ranked: List[Tuple[float, sqlite3.Row]] = []
    unranked: List[sqlite3.Row] = []
    for row in rows:
        x = to_float(row_get(row, x_col))
        y = to_float(row_get(row, y_col))
        if x is None or y is None:
            unranked.append(row)
        else:
            ranked.append((math.hypot(x - ego_x, y - ego_y), row))
    ranked.sort(key=lambda item: item[0])
    return [row for _, row in ranked[:max_neighbors]] + unranked[: max(0, max_neighbors - len(ranked))], lidar_box_lidar_col


def export_db(
    db_path: Path,
    args: argparse.Namespace,
    ego_writer: csv.DictWriter,
    object_writer: csv.DictWriter,
    scene_writer: csv.DictWriter,
    warnings: WarningCollector,
    join_strategies: Dict[str, int],
    example_rows: Dict[str, List[Dict[str, str]]],
) -> Tuple[int, int, int]:
    db_name = db_path.name
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        schema = inspect_schema(conn)
        missing_tables = [table for table in REQUIRED_TABLES if not schema.has_table(table)]
        if missing_tables:
            warnings.add("missing_expected_tables", "Missing expected tables: " + ", ".join(missing_tables), db_name=db_name)
            return 0, 0, 0

        for table in REQUIRED_TABLES:
            if not schema.columns(table):
                warnings.add("missing_expected_columns", f"Table {table} has no discoverable columns.", db_name=db_name)

        scene_token_col = schema.first_column("scene", ["token", "scene_token"])
        scene_name_col = schema.first_column("scene", ["name", "scene_name"])
        lidar_token_col = schema.first_column("lidar_pc", ["token", "lidar_pc_token"])
        lidar_ts_col = schema.first_column("lidar_pc", ["timestamp", "time_us", "utime", "time_stamp"])
        lidar_ego_col = schema.first_column("lidar_pc", ["ego_pose_token", "ego_token"])
        ego_token_col = schema.first_column("ego_pose", ["token", "ego_pose_token"])
        track_token_col = schema.first_column("track", ["token", "track_token"])
        track_category_col = schema.first_column("track", ["category_token", "category"])
        box_track_col = schema.first_column("lidar_box", ["track_token", "track"])

        if lidar_token_col is None:
            warnings.add("missing_lidar_pc_token_column", "Could not find token column in lidar_pc.", db_name=db_name)
        if lidar_ego_col is None:
            warnings.add("missing_lidar_pc_ego_pose_token_column", "Could not find ego_pose_token column in lidar_pc.", db_name=db_name)
        if ego_token_col is None:
            warnings.add("missing_ego_pose_token_column", "Could not find token column in ego_pose.", db_name=db_name)

        categories = load_categories(conn, schema)
        tracks = build_token_index(conn, schema, "track", track_token_col)
        scenes = get_scenes(conn, schema, args.max_scenes_per_db)
        if not scenes:
            warnings.add("no_scene_rows", "The scene table has no rows selected for export.", db_name=db_name)
            return 0, 0, 0

        db_scene_count = 0
        db_ego_count = 0
        db_object_count = 0
        for scene in scenes:
            scene_token_raw = row_get(scene, scene_token_col)
            scene_token = sql_value_to_text(scene_token_raw)
            scene_name = sql_value_to_text(row_get(scene, scene_name_col))
            lidar_rows, join_strategy = resolve_lidar_sequence(conn, schema, scene, args.max_lidar_pcs_per_scene, db_name, warnings)
            join_strategies[join_strategy] = join_strategies.get(join_strategy, 0) + 1
            if not lidar_rows:
                warnings.add("scene_has_no_lidar_pc", "No lidar_pc rows were exported for this scene.", db_name=db_name, scene_token=scene_token)
                scene_writer.writerow(
                    {
                        "db_name": db_name,
                        "scene_token": scene_token,
                        "scene_name": scene_name,
                        "num_lidar_pcs_exported": 0,
                        "num_ego_rows": 0,
                        "num_object_rows": 0,
                        "duration_sec_estimate": "",
                    }
                )
                db_scene_count += 1
                continue

            scene_ego_rows: List[Dict[str, Any]] = []
            object_rows_to_write: List[Dict[str, str]] = []
            frame_times: List[float] = []
            scene_object_count = 0
            scene_missing_ego = 0
            scene_no_objects = 0
            for frame_index, lidar in enumerate(lidar_rows):
                lidar_token_raw = row_get(lidar, lidar_token_col)
                lidar_token = sql_value_to_text(lidar_token_raw)
                lidar_timestamp = row_get(lidar, lidar_ts_col)
                lidar_time_float = to_float(lidar_timestamp)
                lidar_time_sec = normalize_timestamp_seconds(lidar_time_float)
                if lidar_time_sec is not None:
                    frame_times.append(lidar_time_sec)
                else:
                    warnings.add(
                        "timestamp_missing",
                        "lidar_pc timestamp is missing or non-numeric for an exported frame.",
                        db_name=db_name,
                        scene_token=scene_token,
                    )

                ego_pose_token_raw = row_get(lidar, lidar_ego_col)
                ego_pose = fetch_one_by_token(conn, schema, "ego_pose", ego_token_col, ego_pose_token_raw)
                if ego_pose is None:
                    scene_missing_ego += 1
                ego = extract_pose_fields(ego_pose, schema, "ego_pose")
                ego_time_sec = normalize_timestamp_seconds(ego["timestamp"] if ego["timestamp"] is not None else lidar_time_float)
                ego_row: Dict[str, Any] = {
                    "db_name": db_name,
                    "scene_token": scene_token,
                    "scene_name": scene_name,
                    "lidar_pc_token": lidar_token,
                    "lidar_pc_timestamp": sql_value_to_text(lidar_timestamp),
                    "ego_pose_token": sql_value_to_text(ego_pose_token_raw),
                    "ego_timestamp": sql_value_to_text(row_get(ego_pose, schema.first_column("ego_pose", ["timestamp", "time_us", "utime", "time_stamp"])) if ego_pose is not None else ""),
                    "ego_x": format_optional(ego["x"]),
                    "ego_y": format_optional(ego["y"]),
                    "ego_z": format_optional(ego["z"]),
                    "ego_heading": format_optional(ego["heading"]),
                    "ego_qw": format_optional(ego["qw"]),
                    "ego_qx": format_optional(ego["qx"]),
                    "ego_qy": format_optional(ego["qy"]),
                    "ego_qz": format_optional(ego["qz"]),
                    "ego_vx": format_optional(ego["vx"]),
                    "ego_vy": format_optional(ego["vy"]),
                    "ego_speed": "",
                    "ego_accel": "",
                    "ego_yaw_rate": "",
                    "frame_index_in_scene": str(frame_index),
                    "_ego_x": ego["x"],
                    "_ego_y": ego["y"],
                    "_ego_time_sec": ego_time_sec,
                    "_ego_heading": ego["heading"],
                    "_ego_vx": ego["vx"],
                    "_ego_vy": ego["vy"],
                }
                scene_ego_rows.append(ego_row)

                object_box_rows, box_lidar_col = fetch_lidar_boxes(
                    conn, schema, lidar_token_raw, args.num_neighbors, ego["x"], ego["y"]
                )
                if box_lidar_col is None:
                    warnings.add(
                        "missing_lidar_box_lidar_pc_token_column",
                        "Could not find lidar_pc token reference column in lidar_box.",
                        db_name=db_name,
                        scene_token=scene_token,
                    )
                if not object_box_rows:
                    scene_no_objects += 1

                x_col = schema.first_column("lidar_box", ["x", "translation_x", "center_x"])
                y_col = schema.first_column("lidar_box", ["y", "translation_y", "center_y"])
                z_col = schema.first_column("lidar_box", ["z", "translation_z", "center_z"])
                length_col = schema.first_column("lidar_box", ["length", "size_x", "box_length"])
                width_col = schema.first_column("lidar_box", ["width", "size_y", "box_width"])
                height_col = schema.first_column("lidar_box", ["height", "size_z", "box_height"])
                heading_col = schema.first_column("lidar_box", ["heading", "yaw"])
                qw_col = schema.first_column("lidar_box", ["qw", "rotation_w", "orientation_w"])
                qx_col = schema.first_column("lidar_box", ["qx", "rotation_x", "orientation_x"])
                qy_col = schema.first_column("lidar_box", ["qy", "rotation_y", "orientation_y"])
                qz_col = schema.first_column("lidar_box", ["qz", "rotation_z", "orientation_z"])

                ranked_objects: List[Tuple[float, Dict[str, str]]] = []
                for box in object_box_rows:
                    object_x = to_float(row_get(box, x_col))
                    object_y = to_float(row_get(box, y_col))
                    object_z = to_float(row_get(box, z_col))
                    heading = to_float(row_get(box, heading_col))
                    qw = to_float(row_get(box, qw_col))
                    qx = to_float(row_get(box, qx_col))
                    qy = to_float(row_get(box, qy_col))
                    qz = to_float(row_get(box, qz_col))
                    if heading is None:
                        heading = yaw_from_quaternion(qw, qx, qy, qz)
                    relative_x = object_x - ego["x"] if object_x is not None and ego["x"] is not None else None
                    relative_y = object_y - ego["y"] if object_y is not None and ego["y"] is not None else None
                    relative_distance = math.hypot(relative_x, relative_y) if relative_x is not None and relative_y is not None else None
                    track_token_raw = row_get(box, box_track_col)
                    track_row = tracks.get(track_token_raw)
                    category_name = ""
                    if track_row is not None and track_category_col is not None:
                        category_name = categories.get(row_get(track_row, track_category_col), "")
                    object_row = {
                        "db_name": db_name,
                        "scene_token": scene_token,
                        "scene_name": scene_name,
                        "lidar_pc_token": lidar_token,
                        "lidar_pc_timestamp": sql_value_to_text(lidar_timestamp),
                        "frame_index_in_scene": str(frame_index),
                        "track_token": sql_value_to_text(track_token_raw),
                        "category_name": category_name,
                        "object_x": format_optional(object_x),
                        "object_y": format_optional(object_y),
                        "object_z": format_optional(object_z),
                        "object_heading": format_optional(heading),
                        "object_length": format_optional(to_float(row_get(box, length_col))),
                        "object_width": format_optional(to_float(row_get(box, width_col))),
                        "object_height": format_optional(to_float(row_get(box, height_col))),
                        "relative_x": format_optional(relative_x),
                        "relative_y": format_optional(relative_y),
                        "relative_distance": format_optional(relative_distance),
                        "rank_by_distance": "",
                    }
                    sort_distance = relative_distance if relative_distance is not None else float("inf")
                    ranked_objects.append((sort_distance, object_row))
                ranked_objects.sort(key=lambda item: item[0])
                for rank, (_, object_row) in enumerate(ranked_objects[: args.num_neighbors], start=1):
                    object_row["rank_by_distance"] = str(rank)
                    object_rows_to_write.append(object_row)
                    scene_object_count += 1

            if scene_missing_ego:
                warnings.add(
                    "lidar_pc_has_no_ego_pose",
                    f"{scene_missing_ego} exported lidar_pc rows did not join to ego_pose.",
                    db_name=db_name,
                    scene_token=scene_token,
                )
            if scene_no_objects:
                warnings.add(
                    "no_lidar_box_objects",
                    f"{scene_no_objects} exported lidar_pc frames had no joined lidar_box objects.",
                    db_name=db_name,
                    scene_token=scene_token,
                )

            compute_ego_kinematics(scene_ego_rows, warnings, db_name, scene_token)
            for ego_row in scene_ego_rows:
                for key in ["ego_speed", "ego_accel", "ego_yaw_rate"]:
                    ego_row[key] = format_optional(ego_row.get(key)) if ego_row.get(key) != "" else ""
                clean_row = {key: ego_row.get(key, "") for key in EGO_COLUMNS}
                ego_writer.writerow(clean_row)
                if len(example_rows["ego"]) < 3:
                    example_rows["ego"].append(clean_row)
            for object_row in object_rows_to_write:
                object_writer.writerow(object_row)
                if len(example_rows["objects"]) < 3:
                    example_rows["objects"].append(object_row)

            duration = ""
            if len(frame_times) >= 2:
                duration = format_optional(max(frame_times) - min(frame_times))
            elif lidar_rows:
                warnings.add(
                    "cannot_estimate_duration",
                    "Scene duration could not be estimated because fewer than two valid timestamps were exported.",
                    db_name=db_name,
                    scene_token=scene_token,
                )
            scene_writer.writerow(
                {
                    "db_name": db_name,
                    "scene_token": scene_token,
                    "scene_name": scene_name,
                    "num_lidar_pcs_exported": len(lidar_rows),
                    "num_ego_rows": len(scene_ego_rows),
                    "num_object_rows": scene_object_count,
                    "duration_sec_estimate": duration,
                }
            )
            db_scene_count += 1
            db_ego_count += len(scene_ego_rows)
            db_object_count += scene_object_count
        return db_scene_count, db_ego_count, db_object_count
    finally:
        conn.close()


def write_report(
    output_dir: Path,
    mini_db_dir: Optional[Path],
    selected_db_count: int,
    selected_scene_count: int,
    ego_count: int,
    object_count: int,
    warnings: WarningCollector,
    join_strategies: Dict[str, int],
    example_rows: Dict[str, List[Dict[str, str]]],
) -> None:
    warning_counts: Dict[str, int] = {}
    for warning in warnings.items:
        code = str(warning.get("code", "unknown"))
        warning_counts[code] = warning_counts.get(code, 0) + 1
    lines = [
        "# Stage 7A.1 Expert Context Export Report",
        "",
        "## Paths",
        "",
        f"- mini_db_dir: `{mini_db_dir or ''}`",
        f"- output_dir: `{output_dir}`",
        f"- expert_ego_trajectory.csv: `{output_dir / 'expert_ego_trajectory.csv'}`",
        f"- expert_nearby_objects.csv: `{output_dir / 'expert_nearby_objects.csv'}`",
        f"- selected_scenes.csv: `{output_dir / 'selected_scenes.csv'}`",
        f"- warnings.json: `{output_dir / 'warnings.json'}`",
        "",
        "## Summary",
        "",
        f"- selected DB count: {selected_db_count}",
        f"- selected scene count: {selected_scene_count}",
        f"- ego row count: {ego_count}",
        f"- nearby object row count: {object_count}",
        "",
        "## SQLite join strategies used",
        "",
    ]
    if join_strategies:
        for strategy, count in sorted(join_strategies.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {strategy}: {count} scene(s)")
    else:
        lines.append("- No scene/lidar_pc join strategy was used because no scenes were exported.")
    lines.extend(["", "## Examples", "", "### expert_ego_trajectory.csv", ""])
    if example_rows["ego"]:
        lines.append("```json")
        lines.append(json.dumps(example_rows["ego"], indent=2, ensure_ascii=False))
        lines.append("```")
    else:
        lines.append("No ego rows were exported.")
    lines.extend(["", "### expert_nearby_objects.csv", ""])
    if example_rows["objects"]:
        lines.append("```json")
        lines.append(json.dumps(example_rows["objects"], indent=2, ensure_ascii=False))
        lines.append("```")
    else:
        lines.append("No nearby object rows were exported.")
    lines.extend(["", "## Warnings", ""])
    if warning_counts:
        for code, count in sorted(warning_counts.items()):
            lines.append(f"- {code}: {count}")
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Next step",
            "",
            "Convert exported expert context into Stage 6-style `ego_seq.npy` / `neighbor_seq.npy`, metadata, "
            "`shard_manifest.json`, and `feature_schema.json`.",
            "",
            "This report is an expert/historical nuPlan data export only: no planner simulation, no fake rollout data, "
            "and no Stage 6C result modification is performed.",
        ]
    )
    (output_dir / "expert_context_export_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    warnings = WarningCollector()
    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, args.overwrite)
    mini_db_dir = resolve_mini_db_dir(args.nuplan_data_root, args.mini_db_dir, warnings)
    db_files = find_db_files(mini_db_dir, args.max_dbs, warnings)

    join_strategies: Dict[str, int] = {}
    example_rows: Dict[str, List[Dict[str, str]]] = {"ego": [], "objects": []}
    selected_scene_count = 0
    ego_count = 0
    object_count = 0

    with (output_dir / "expert_ego_trajectory.csv").open("w", newline="", encoding="utf-8") as ego_file, (
        output_dir / "expert_nearby_objects.csv"
    ).open("w", newline="", encoding="utf-8") as object_file, (output_dir / "selected_scenes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as scene_file:
        ego_writer = csv.DictWriter(ego_file, fieldnames=EGO_COLUMNS)
        object_writer = csv.DictWriter(object_file, fieldnames=OBJECT_COLUMNS)
        scene_writer = csv.DictWriter(scene_file, fieldnames=SCENE_COLUMNS)
        ego_writer.writeheader()
        object_writer.writeheader()
        scene_writer.writeheader()
        for db_index, db_path in enumerate(db_files, start=1):
            print(f"[{db_index}/{len(db_files)}] exporting {db_path.name}")
            try:
                scene_delta, ego_delta, object_delta = export_db(
                    db_path, args, ego_writer, object_writer, scene_writer, warnings, join_strategies, example_rows
                )
            except sqlite3.Error as exc:
                warnings.add("db_sqlite_error", f"SQLite error while exporting {db_path}: {exc}", db_name=db_path.name)
                continue
            selected_scene_count += scene_delta
            ego_count += ego_delta
            object_count += object_delta

    warning_payload = {
        "warning_count": len(warnings.items),
        "warnings": warnings.items,
    }
    (output_dir / "warnings.json").write_text(json.dumps(warning_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(
        output_dir,
        mini_db_dir,
        len(db_files),
        selected_scene_count,
        ego_count,
        object_count,
        warnings,
        join_strategies,
        example_rows,
    )
    print(f"Wrote Stage 7A.1 expert context export to {output_dir}")
    print(f"selected DBs={len(db_files)} scenes={selected_scene_count} ego_rows={ego_count} object_rows={object_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
