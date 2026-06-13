#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


KEY_TABLES = [
    "log",
    "scene",
    "scenario_tag",
    "ego_pose",
    "lidar_pc",
    "lidar_box",
    "track",
    "category",
    "traffic_light_status",
]

SAMPLE_TABLES = [
    "ego_pose",
    "lidar_pc",
    "lidar_box",
    "track",
    "scenario_tag",
]

SAMPLE_OUTPUTS = {
    "scenario_tag": "mini_scenario_tags_sample.csv",
    "ego_pose": "sample_ego_pose_rows.csv",
    "lidar_pc": "sample_lidar_pc_rows.csv",
    "lidar_box": "sample_lidar_box_rows.csv",
    "track": "sample_track_rows.csv",
}

INVENTORY_COLUMNS = [
    "db_name",
    "db_path",
    "size_mb",
    "open_ok",
    "error",
    "table_count",
    "tables",
    "has_log",
    "has_scene",
    "has_scenario_tag",
    "has_ego_pose",
    "has_lidar_pc",
    "has_lidar_box",
    "has_track",
    "has_category",
    "has_traffic_light_status",
    "count_log",
    "count_scene",
    "count_scenario_tag",
    "count_ego_pose",
    "count_lidar_pc",
    "count_lidar_box",
    "count_track",
    "count_category",
    "count_traffic_light_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 7A.0 lightweight readiness checker for nuPlan mini SQLite DBs "
            "and map.gpkg files. This script does not run planner simulation or "
            "generate rollout data."
        )
    )
    parser.add_argument(
        "--nuplan_data_root",
        default=os.environ.get("NUPLAN_DATA_ROOT"),
        help="nuPlan data root. Default: NUPLAN_DATA_ROOT environment variable.",
    )
    parser.add_argument(
        "--nuplan_maps_root",
        default=os.environ.get("NUPLAN_MAPS_ROOT"),
        help="nuPlan maps root. Default: NUPLAN_MAPS_ROOT environment variable.",
    )
    parser.add_argument(
        "--mini_db_dir",
        default=None,
        help=(
            "Optional explicit mini DB directory. If omitted, tries "
            "$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/mini then "
            "$NUPLAN_DATA_ROOT/data/cache/mini."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/stage7A_nuplan/mini_check",
        help="Directory for CSV/JSON/Markdown reports.",
    )
    parser.add_argument(
        "--max_dbs",
        type=int,
        default=0,
        help="Maximum number of DBs to scan. 0 means scan all DBs.",
    )
    parser.add_argument(
        "--sample_rows_per_table",
        type=int,
        default=20,
        help="Number of first rows to sample from selected tables in each DB.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output_dir first if it already exists.",
    )
    return parser.parse_args()


def add_warning(warnings: list, code: str, message: str, path: Path | None = None) -> None:
    item = {"code": code, "message": message}
    if path is not None:
        item["path"] = str(path)
    warnings.append(item)


def resolve_mini_db_dir(data_root: Path | None, explicit_dir: str | None, warnings: list) -> Path | None:
    if explicit_dir:
        return Path(explicit_dir).expanduser()
    if data_root is None:
        add_warning(
            warnings,
            "missing_data_root_for_mini_db_dir",
            "Cannot infer mini DB directory because --nuplan_data_root is unset and NUPLAN_DATA_ROOT is not defined.",
        )
        return None

    candidates = [
        data_root / "nuplan-v1.1" / "splits" / "mini",
        data_root / "data" / "cache" / "mini",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    add_warning(
        warnings,
        "mini_db_dir_candidates_missing",
        "No default mini DB directory exists under nuPlan data root.",
    )
    return candidates[0]


def find_db_files(mini_db_dir: Path | None, max_dbs: int, warnings: list) -> list[Path]:
    if mini_db_dir is None:
        add_warning(warnings, "mini_db_dir_missing", "Mini DB directory is not configured.")
        return []
    if not mini_db_dir.exists():
        add_warning(warnings, "mini_db_dir_missing", "Mini DB directory does not exist.", mini_db_dir)
        return []
    if not mini_db_dir.is_dir():
        add_warning(warnings, "mini_db_dir_not_directory", "Mini DB path is not a directory.", mini_db_dir)
        return []

    db_files = sorted(
        path
        for path in mini_db_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}
    )
    if not db_files:
        add_warning(warnings, "no_mini_db_files", "No mini SQLite DB files found.", mini_db_dir)
        return []
    if max_dbs > 0:
        return db_files[:max_dbs]
    return db_files


def count_map_files(maps_root: Path | None, warnings: list) -> int:
    if maps_root is None:
        add_warning(
            warnings,
            "maps_root_missing",
            "--nuplan_maps_root is unset and NUPLAN_MAPS_ROOT is not defined.",
        )
        return 0
    if not maps_root.exists():
        add_warning(warnings, "maps_root_missing", "nuPlan maps root does not exist.", maps_root)
        return 0
    if not maps_root.is_dir():
        add_warning(warnings, "maps_root_not_directory", "nuPlan maps root is not a directory.", maps_root)
        return 0
    count = sum(1 for path in maps_root.rglob("map.gpkg") if path.is_file())
    if count == 0:
        add_warning(warnings, "no_map_gpkg_files", "No map.gpkg files found.", maps_root)
    return count


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def to_jsonable(value):
    if isinstance(value, bytes):
        return value.hex()
    return value


def normalize_csv_value(value):
    value = to_jsonable(value)
    if value is None:
        return ""
    return value


def fetch_table_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    return [str(row[0]) for row in rows]


def fetch_schema(conn: sqlite3.Connection, table_name: str) -> list[dict]:
    rows = conn.execute(f"PRAGMA table_info({quote_identifier(table_name)})").fetchall()
    fields = ["cid", "name", "type", "notnull", "dflt_value", "pk"]
    return [dict(zip(fields, [to_jsonable(value) for value in row])) for row in rows]


def fetch_count(conn: sqlite3.Connection, table_name: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) FROM {quote_identifier(table_name)}").fetchone()
    return int(row[0])


def sample_rows(conn: sqlite3.Connection, table_name: str, limit: int, db_name: str) -> list[dict]:
    if limit <= 0:
        return []
    cursor = conn.execute(f"SELECT * FROM {quote_identifier(table_name)} LIMIT ?", (limit,))
    columns = [description[0] for description in cursor.description]
    output = []
    for row in cursor.fetchall():
        row_dict = {"db_name": db_name}
        for column, value in zip(columns, row):
            row_dict[column] = normalize_csv_value(value)
        output.append(row_dict)
    return output


def scan_db(db_path: Path, sample_rows_per_table: int) -> tuple[dict, dict, dict[str, list[dict]], dict | None]:
    inventory = {
        "db_name": db_path.name,
        "db_path": str(db_path),
        "size_mb": round(db_path.stat().st_size / (1024 * 1024), 3),
        "open_ok": False,
        "error": "",
        "table_count": 0,
        "tables": "",
    }
    for table_name in KEY_TABLES:
        inventory[f"has_{table_name}"] = False
        inventory[f"count_{table_name}"] = ""

    schema_entry = {
        "db_name": db_path.name,
        "db_path": str(db_path),
        "open_ok": False,
        "error": "",
        "tables": [],
        "schemas": {},
        "row_counts": {},
    }
    samples = {table_name: [] for table_name in SAMPLE_TABLES}
    failure_warning = None

    try:
        with sqlite3.connect(str(db_path)) as conn:
            tables = fetch_table_names(conn)
            table_set = set(tables)
            inventory["open_ok"] = True
            inventory["table_count"] = len(tables)
            inventory["tables"] = ";".join(tables)
            schema_entry["open_ok"] = True
            schema_entry["tables"] = tables

            for table_name in tables:
                schema_entry["schemas"][table_name] = fetch_schema(conn, table_name)

            for table_name in KEY_TABLES:
                present = table_name in table_set
                inventory[f"has_{table_name}"] = present
                if present:
                    count = fetch_count(conn, table_name)
                    inventory[f"count_{table_name}"] = count
                    schema_entry["row_counts"][table_name] = count

            for table_name in SAMPLE_TABLES:
                if table_name in table_set:
                    samples[table_name] = sample_rows(
                        conn, table_name, sample_rows_per_table, db_path.name
                    )
    except sqlite3.Error as exc:
        message = f"SQLite error while opening/scanning {db_path}: {exc}"
        inventory["error"] = message
        schema_entry["error"] = message
        failure_warning = {
            "code": "db_open_or_scan_failed",
            "message": message,
            "path": str(db_path),
        }
    except OSError as exc:
        message = f"OS error while opening/scanning {db_path}: {exc}"
        inventory["error"] = message
        schema_entry["error"] = message
        failure_warning = {
            "code": "db_open_or_scan_failed",
            "message": message,
            "path": str(db_path),
        }

    return inventory, schema_entry, samples, failure_warning


def write_csv(path: Path, rows: list[dict], preferred_columns: list[str] | None = None) -> None:
    if preferred_columns is None:
        columns = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    columns.append(key)
        if not columns:
            columns = ["db_name"]
    else:
        columns = preferred_columns

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def key_table_coverage(inventory_rows: list[dict]) -> dict[str, dict]:
    coverage = {}
    total = len(inventory_rows)
    for table_name in KEY_TABLES:
        present_count = sum(1 for row in inventory_rows if row.get(f"has_{table_name}") is True)
        coverage[table_name] = {
            "dbs_with_table": present_count,
            "scanned_dbs": total,
            "coverage_ratio": (present_count / total) if total else 0.0,
            "total_rows": sum(
                int(row.get(f"count_{table_name}") or 0)
                for row in inventory_rows
                if row.get(f"has_{table_name}") is True
            ),
        }
    return coverage


def render_report(
    data_root: Path | None,
    maps_root: Path | None,
    mini_db_dir: Path | None,
    found_db_count: int,
    scanned_db_count: int,
    map_gpkg_count: int,
    inventory_rows: list[dict],
    coverage: dict[str, dict],
    warnings: list[dict],
) -> str:
    open_success_count = sum(1 for row in inventory_rows if row.get("open_ok") is True)
    lines = [
        "# Stage 7A.0 — nuPlan mini readiness check",
        "",
        "## Paths",
        "",
        f"- data_root: `{data_root if data_root is not None else ''}`",
        f"- maps_root: `{maps_root if maps_root is not None else ''}`",
        f"- mini_db_dir: `{mini_db_dir if mini_db_dir is not None else ''}`",
        "",
        "## Summary",
        "",
        f"- mini DB count found: {found_db_count}",
        f"- mini DB count scanned: {scanned_db_count}",
        f"- map.gpkg count: {map_gpkg_count}",
        f"- DB open success count: {open_success_count}",
        f"- DB open failure count: {scanned_db_count - open_success_count}",
        "",
        "## Key table coverage",
        "",
        "| table | DBs with table | scanned DBs | coverage | total rows |",
        "|---|---:|---:|---:|---:|",
    ]
    for table_name in KEY_TABLES:
        item = coverage[table_name]
        lines.append(
            f"| {table_name} | {item['dbs_with_table']} | {item['scanned_dbs']} | "
            f"{item['coverage_ratio']:.3f} | {item['total_rows']} |"
        )

    lines.extend(
        [
            "",
            "## First 20 DB examples",
            "",
            "| db_name | ego_pose | lidar_pc | lidar_box | scenario_tag |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in inventory_rows[:20]:
        lines.append(
            f"| {row['db_name']} | {row.get('count_ego_pose') or 0} | "
            f"{row.get('count_lidar_pc') or 0} | {row.get('count_lidar_box') or 0} | "
            f"{row.get('count_scenario_tag') or 0} |"
        )

    lines.extend(["", "## Warnings", ""])
    if warnings:
        for warning in warnings:
            path_text = f" (`{warning['path']}`)" if warning.get("path") else ""
            lines.append(f"- {warning['code']}: {warning['message']}{path_text}")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Next step",
            "",
            "Export expert ego trajectory and nearby object context from selected mini scenarios.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    warnings = []

    data_root = Path(args.nuplan_data_root).expanduser() if args.nuplan_data_root else None
    maps_root = Path(args.nuplan_maps_root).expanduser() if args.nuplan_maps_root else None

    if data_root is None:
        add_warning(
            warnings,
            "data_root_missing",
            "--nuplan_data_root is unset and NUPLAN_DATA_ROOT is not defined.",
        )
    elif not data_root.exists():
        add_warning(warnings, "data_root_missing", "nuPlan data root does not exist.", data_root)
    elif not data_root.is_dir():
        add_warning(warnings, "data_root_not_directory", "nuPlan data root is not a directory.", data_root)

    mini_db_dir = resolve_mini_db_dir(data_root, args.mini_db_dir, warnings)
    map_gpkg_count = count_map_files(maps_root, warnings)
    db_files = find_db_files(mini_db_dir, args.max_dbs, warnings)

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Re-run with --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory_rows = []
    schema_entries = []
    all_samples = {table_name: [] for table_name in SAMPLE_TABLES}

    for db_path in db_files:
        inventory, schema_entry, samples, failure_warning = scan_db(
            db_path, args.sample_rows_per_table
        )
        inventory_rows.append(inventory)
        schema_entries.append(schema_entry)
        for table_name, rows in samples.items():
            all_samples[table_name].extend(rows)
        if failure_warning is not None:
            warnings.append(failure_warning)

    coverage = key_table_coverage(inventory_rows)
    for table_name, item in coverage.items():
        if inventory_rows and item["dbs_with_table"] == 0:
            add_warning(
                warnings,
                "key_table_missing_in_all_scanned_dbs",
                f"Key table '{table_name}' is missing in all scanned DBs.",
            )

    write_csv(output_dir / "mini_db_inventory.csv", inventory_rows, INVENTORY_COLUMNS)
    for table_name, filename in SAMPLE_OUTPUTS.items():
        write_csv(output_dir / filename, all_samples[table_name])

    schema_report = {
        "data_root": str(data_root) if data_root is not None else "",
        "maps_root": str(maps_root) if maps_root is not None else "",
        "mini_db_dir": str(mini_db_dir) if mini_db_dir is not None else "",
        "db_count": len(db_files),
        "map_gpkg_count": map_gpkg_count,
        "key_table_coverage": coverage,
        "databases": schema_entries,
    }
    write_json(output_dir / "mini_schema_report.json", schema_report)
    write_json(output_dir / "warnings.json", warnings)

    report = render_report(
        data_root,
        maps_root,
        mini_db_dir,
        len(db_files),
        len(inventory_rows),
        map_gpkg_count,
        inventory_rows,
        coverage,
        warnings,
    )
    (output_dir / "mini_check_report.md").write_text(report, encoding="utf-8")

    print(f"Stage 7A.0 nuPlan mini readiness check complete: {output_dir}")
    print(f"Scanned DBs: {len(inventory_rows)}")
    print(f"map.gpkg count: {map_gpkg_count}")
    print(f"warnings: {len(warnings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
