#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sqlite3
import subprocess
import sys
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


SCHEMA_VERSION = "stage7p_scenario_inventory_v1"
INVENTORY_FIELDS = [
    "db_file",
    "log_name",
    "scenario_token",
    "scene_token",
    "db_scene_token",
    "scenario_type",
    "scenario_tag_token",
]
INPUT_FIELDS = [
    "db_file",
    "db_path",
    "size_bytes",
    "mtime_ns",
    "sha256",
    "scenario_tag_rows",
    "inventory_rows",
    "duplicate_rows_removed",
    "log_names",
]
OUTPUT_FILENAMES = {
    "all_scenario_tags.csv",
    "scenario_inventory_summary.json",
    "scenario_inventory_inputs.csv",
    "scenario_inventory_report.md",
}
REQUIRED_SCHEMA = {
    "scenario_tag": {"token", "lidar_pc_token", "type"},
    "lidar_pc": {"token", "scene_token"},
    "scene": {"token", "log_token"},
    "log": {"token", "logfile"},
}


def token_to_text(value: Any, *, label: str, db_path: Path) -> str:
    if isinstance(value, memoryview):
        value = bytes(value)
    if isinstance(value, bytes):
        text = value.hex()
    elif value is None:
        text = ""
    else:
        text = str(value).strip().lower()
    if not text:
        raise ValueError(f"Empty {label} in SQLite DB: {db_path}")
    return text


def text_value(value: Any, *, label: str, db_path: Path) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(f"Empty {label} in SQLite DB: {db_path}")
    return text


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def git_value(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def discover_db_paths(db_roots: Sequence[Path]) -> List[Path]:
    if not db_roots:
        raise ValueError("At least one --db_root is required")
    discovered: Dict[Path, Path] = {}
    for raw_root in db_roots:
        root = raw_root.expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"nuPlan DB root does not exist or is not a directory: {root}")
        for path in sorted(root.glob("*.db"), key=lambda value: value.name):
            if path.is_file():
                resolved = path.resolve()
                discovered.setdefault(resolved, path.absolute())
    db_paths = sorted(discovered, key=lambda value: (value.name, str(value)))
    if not db_paths:
        roots = ", ".join(str(Path(value).expanduser().resolve()) for value in db_roots)
        raise FileNotFoundError(f"No direct-child *.db files found under DB roots: {roots}")

    by_basename: Dict[str, Path] = {}
    for path in db_paths:
        previous = by_basename.get(path.name)
        if previous is not None and previous != path:
            raise ValueError(
                "DB basename conflict prevents a flat M6.4 pool: "
                f"{path.name} resolves to both {previous} and {path}"
            )
        by_basename[path.name] = path
    return db_paths


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / name for name in sorted(OUTPUT_FILENAMES) if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Inventory outputs already exist; pass --overwrite to replace only known generated files: "
            + ", ".join(str(path) for path in existing)
        )
    if overwrite:
        for path in existing:
            if not path.is_file() and not path.is_symlink():
                raise IsADirectoryError(f"Refusing to replace non-file inventory output: {path}")
            path.unlink()


def sqlite_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    return {str(row[1]) for row in rows}


def validate_sqlite_schema(conn: sqlite3.Connection, db_path: Path) -> None:
    tables = {
        str(row[0])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    for table, required_columns in REQUIRED_SCHEMA.items():
        if table not in tables:
            raise ValueError(f"Missing required SQLite table '{table}' in DB: {db_path}")
        missing = sorted(required_columns - sqlite_columns(conn, table))
        if missing:
            raise ValueError(
                f"SQLite table '{table}' is missing required columns {missing} in DB: {db_path}"
            )


def open_readonly_sqlite(db_path: Path) -> sqlite3.Connection:
    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise ValueError(f"Could not open SQLite DB read-only: {db_path}: {exc}") from exc
    conn.row_factory = sqlite3.Row
    return conn


def create_staging_database(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute(
        """
        CREATE TABLE inventory (
            db_file TEXT NOT NULL,
            log_name TEXT NOT NULL,
            scenario_token TEXT NOT NULL,
            scene_token TEXT NOT NULL,
            db_scene_token TEXT NOT NULL,
            scenario_type TEXT NOT NULL,
            scenario_tag_token TEXT NOT NULL,
            UNIQUE(db_file, log_name, scenario_token, scenario_type)
        )
        """
    )
    conn.execute("CREATE INDEX inventory_token_idx ON inventory(scenario_token)")
    return conn


def scan_one_db(
    db_path: Path,
    staging: sqlite3.Connection,
    *,
    batch_size: int = 10_000,
) -> Dict[str, Any]:
    initial_stat = db_path.stat()
    with closing(open_readonly_sqlite(db_path)) as source:
        validate_sqlite_schema(source, db_path)
        raw_rows = int(source.execute("SELECT COUNT(*) FROM scenario_tag").fetchone()[0])
        query = """
            SELECT
                st.token AS scenario_tag_token,
                st.lidar_pc_token AS scenario_token,
                st.type AS scenario_type,
                lp.token AS linked_lidar_pc_token,
                lp.scene_token AS db_scene_token,
                sc.token AS linked_scene_token,
                sc.log_token AS scene_log_token,
                lg.token AS linked_log_token,
                lg.logfile AS log_name
            FROM scenario_tag AS st
            LEFT JOIN lidar_pc AS lp ON lp.token = st.lidar_pc_token
            LEFT JOIN scene AS sc ON sc.token = lp.scene_token
            LEFT JOIN log AS lg ON lg.token = sc.log_token
            ORDER BY st.lidar_pc_token, st.type, st.token
        """
        cursor = source.execute(query)
        fetched_rows = 0
        duplicate_rows = 0
        log_names: set[str] = set()
        before_total = staging.total_changes
        while True:
            batch = cursor.fetchmany(batch_size)
            if not batch:
                break
            parsed: List[Tuple[str, str, str, str, str, str, str]] = []
            for row in batch:
                fetched_rows += 1
                if row["linked_lidar_pc_token"] is None:
                    raise ValueError(
                        f"scenario_tag references a missing lidar_pc row in DB {db_path}; "
                        f"scenario_tag_token={token_to_text(row['scenario_tag_token'], label='scenario_tag.token', db_path=db_path)}"
                    )
                if row["linked_scene_token"] is None:
                    raise ValueError(
                        f"lidar_pc references a missing scene row in DB {db_path}; "
                        f"scenario_token={token_to_text(row['scenario_token'], label='scenario_tag.lidar_pc_token', db_path=db_path)}"
                    )
                if row["linked_log_token"] is None:
                    raise ValueError(
                        f"scene references a missing log row in DB {db_path}; "
                        f"db_scene_token={token_to_text(row['db_scene_token'], label='lidar_pc.scene_token', db_path=db_path)}"
                    )
                scenario_token = token_to_text(
                    row["scenario_token"], label="scenario_tag.lidar_pc_token", db_path=db_path
                )
                db_scene_token = token_to_text(
                    row["db_scene_token"], label="lidar_pc.scene_token", db_path=db_path
                )
                scenario_type = text_value(
                    row["scenario_type"], label="scenario_tag.type", db_path=db_path
                )
                log_name = text_value(row["log_name"], label="log.logfile", db_path=db_path)
                log_names.add(log_name)
                parsed.append(
                    (
                        db_path.name,
                        log_name,
                        scenario_token,
                        scenario_token,
                        db_scene_token,
                        scenario_type,
                        token_to_text(
                            row["scenario_tag_token"], label="scenario_tag.token", db_path=db_path
                        ),
                    )
                )
            changes_before_batch = staging.total_changes
            staging.executemany(
                """
                INSERT OR IGNORE INTO inventory (
                    db_file, log_name, scenario_token, scene_token,
                    db_scene_token, scenario_type, scenario_tag_token
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                parsed,
            )
            duplicate_rows += len(parsed) - (staging.total_changes - changes_before_batch)
        if fetched_rows != raw_rows:
            raise ValueError(
                f"scenario_tag row-count mismatch in {db_path}: table={raw_rows}, query={fetched_rows}"
            )
        staging.commit()
        inserted_rows = staging.total_changes - before_total

    db_sha256 = sha256_file(db_path)
    final_stat = db_path.stat()
    if (initial_stat.st_size, initial_stat.st_mtime_ns) != (
        final_stat.st_size,
        final_stat.st_mtime_ns,
    ):
        raise RuntimeError(f"SQLite DB changed while inventory was being built: {db_path}")
    return {
        "db_file": db_path.name,
        "db_path": str(db_path.resolve()),
        "size_bytes": int(final_stat.st_size),
        "mtime_ns": int(final_stat.st_mtime_ns),
        "sha256": db_sha256,
        "scenario_tag_rows": raw_rows,
        "inventory_rows": int(inserted_rows),
        "duplicate_rows_removed": int(duplicate_rows),
        "log_names": "|".join(sorted(log_names)),
    }


def token_location_conflicts(staging: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = staging.execute(
        """
        SELECT
            scenario_token,
            COUNT(DISTINCT log_name || char(31) || db_file) AS location_count,
            GROUP_CONCAT(DISTINCT log_name || '@' || db_file) AS locations
        FROM inventory
        GROUP BY scenario_token
        HAVING location_count > 1
        ORDER BY scenario_token
        """
    ).fetchall()
    return [
        {
            "scenario_token": str(row[0]),
            "location_count": int(row[1]),
            "locations": str(row[2]),
        }
        for row in rows
    ]


def preflight_flat_pool(flat_db_root: Path, db_paths: Sequence[Path]) -> None:
    for source in db_paths:
        target = flat_db_root / source.name
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise ValueError(
                    f"Flat DB pool contains a symlink with the wrong target: {target} -> {os.readlink(target)}"
                )
        elif target.exists():
            raise FileExistsError(
                f"Refusing to overwrite non-symlink entry in flat DB pool: {target}"
            )


def create_flat_pool(flat_db_root: Path, db_paths: Sequence[Path]) -> Dict[str, int]:
    flat_db_root.mkdir(parents=True, exist_ok=True)
    preflight_flat_pool(flat_db_root, db_paths)
    created = 0
    reused = 0
    for source in db_paths:
        target = flat_db_root / source.name
        if target.is_symlink():
            reused += 1
            continue
        relative_target = os.path.relpath(source.resolve(), target.parent.resolve())
        target.symlink_to(relative_target)
        created += 1
    return {"created": created, "reused": reused, "total": len(db_paths)}


def write_inventory_csv(path: Path, staging: sqlite3.Connection) -> int:
    cursor = staging.execute(
        """
        SELECT db_file, log_name, scenario_token, scene_token, db_scene_token,
               scenario_type, scenario_tag_token
        FROM inventory
        ORDER BY db_file, log_name, scenario_token, scenario_type, scenario_tag_token
        """
    )
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(INVENTORY_FIELDS)
        while True:
            batch = cursor.fetchmany(20_000)
            if not batch:
                break
            writer.writerows(batch)
            count += len(batch)
    return count


def write_inputs_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def grouped_counts(staging: sqlite3.Connection, field: str) -> Dict[str, int]:
    if field not in {"db_file", "log_name", "scenario_type"}:
        raise ValueError(f"Unsupported grouped-count field: {field}")
    rows = staging.execute(
        f'SELECT "{field}", COUNT(*) FROM inventory GROUP BY "{field}" ORDER BY "{field}"'
    ).fetchall()
    return {str(row[0]): int(row[1]) for row in rows}


def render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Stage7P scenario-tag inventory report",
        "",
        f"- schema version: `{summary['schema_version']}`",
        f"- status: `{summary['status']}`",
        f"- DB files: `{summary['db_file_count']}`",
        f"- logs: `{summary['log_count']}`",
        f"- source scenario_tag rows: `{summary['source_scenario_tag_rows']}`",
        f"- inventory rows after deduplication: `{summary['inventory_rows']}`",
        f"- unique scenario tokens: `{summary['unique_scenario_tokens']}`",
        f"- duplicate rows removed: `{summary['duplicate_rows_removed']}`",
        f"- token-location conflicts: `{summary['token_location_conflicts']}`",
        "",
        "## Data contract",
        "",
        "- Only SQLite pre-treatment metadata is read.",
        "- `scenario_token` and compatibility `scene_token` use `scenario_tag.lidar_pc_token`.",
        "- `db_scene_token` preserves the original `lidar_pc.scene_token`.",
        "- Multiple scenario types for the same token remain separate rows.",
        "- Planner outcomes, trajectories, embeddings and BDD are not read.",
        "",
        "## M6.4 handoff",
        "",
        f"- inventory: `{summary['outputs']['inventory_csv']}`",
        f"- flat DB root: `{summary['flat_db_pool']['path']}`",
        "- This builder does not launch M6.4 preflight or any rollout.",
        "",
    ]
    return "\n".join(lines)


def build_inventory(
    *,
    db_roots: Sequence[Path],
    output_dir: Path,
    flat_db_root: Path | None = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parent.parent
    output_dir = output_dir.expanduser().resolve()
    db_paths = discover_db_paths(db_roots)
    if flat_db_root is not None:
        flat_db_root = flat_db_root.expanduser().resolve()
        preflight_flat_pool(flat_db_root, db_paths)
    prepare_output_dir(output_dir, overwrite)

    input_rows: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="stage7p_inventory_") as temp_dir:
        staging = create_staging_database(Path(temp_dir) / "inventory.sqlite3")
        try:
            for index, db_path in enumerate(db_paths, 1):
                print(f"[{index}/{len(db_paths)}] scanning and hashing {db_path}", flush=True)
                input_rows.append(scan_one_db(db_path, staging))

            conflicts = token_location_conflicts(staging)
            if conflicts:
                preview = json.dumps(conflicts[:10], ensure_ascii=False)
                raise ValueError(
                    f"Found {len(conflicts)} scenario tokens assigned to multiple log/DB locations; "
                    f"examples: {preview}"
                )

            flat_stats = {"created": 0, "reused": 0, "total": 0}
            if flat_db_root is not None:
                flat_stats = create_flat_pool(flat_db_root, db_paths)

            inventory_path = output_dir / "all_scenario_tags.csv"
            inputs_path = output_dir / "scenario_inventory_inputs.csv"
            report_path = output_dir / "scenario_inventory_report.md"
            summary_path = output_dir / "scenario_inventory_summary.json"
            inventory_rows = write_inventory_csv(inventory_path, staging)
            write_inputs_csv(inputs_path, input_rows)

            unique_tokens = int(
                staging.execute("SELECT COUNT(DISTINCT scenario_token) FROM inventory").fetchone()[0]
            )
            log_counts = grouped_counts(staging, "log_name")
            scenario_type_counts = grouped_counts(staging, "scenario_type")
            db_counts = grouped_counts(staging, "db_file")
            source_rows = sum(int(row["scenario_tag_rows"]) for row in input_rows)
            duplicate_rows = sum(int(row["duplicate_rows_removed"]) for row in input_rows)
            summary: Dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "status": "COMPLETE_PRETREATMENT_INVENTORY",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "db_roots": [str(Path(root).expanduser().resolve()) for root in db_roots],
                "db_file_count": len(db_paths),
                "log_count": len(log_counts),
                "scenario_type_count": len(scenario_type_counts),
                "source_scenario_tag_rows": source_rows,
                "inventory_rows": inventory_rows,
                "unique_scenario_tokens": unique_tokens,
                "duplicate_rows_removed": duplicate_rows,
                "token_location_conflicts": 0,
                "inventory_fields": INVENTORY_FIELDS,
                "scenario_type_counts": scenario_type_counts,
                "log_counts": log_counts,
                "db_inventory_row_counts": db_counts,
                "selection_timing": "pre_treatment",
                "outcome_blind": True,
                "reads_planner_outcomes": False,
                "flat_db_pool": {
                    "enabled": flat_db_root is not None,
                    "path": str(flat_db_root) if flat_db_root is not None else "",
                    **flat_stats,
                },
                "provenance": {
                    "tool_path": str(Path(__file__).resolve()),
                    "tool_sha256": sha256_file(Path(__file__).resolve()),
                    "git_commit": git_value(repo_root, "rev-parse", "HEAD"),
                    "git_branch": git_value(repo_root, "branch", "--show-current"),
                    "python_version": sys.version,
                    "platform": platform.platform(),
                    "machine": platform.machine(),
                },
                "outputs": {
                    "inventory_csv": str(inventory_path),
                    "inputs_csv": str(inputs_path),
                    "report_md": str(report_path),
                    "summary_json": str(summary_path),
                },
            }
            report_path.write_text(render_report(summary), encoding="utf-8")
            summary["output_sha256"] = {
                "all_scenario_tags.csv": sha256_file(inventory_path),
                "scenario_inventory_inputs.csv": sha256_file(inputs_path),
                "scenario_inventory_report.md": sha256_file(report_path),
            }
            write_json(summary_path, summary)
            return summary
        finally:
            staging.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a reproducible, outcome-blind nuPlan scenario_tag inventory from one or "
            "more flat SQLite DB roots."
        )
    )
    parser.add_argument(
        "--db_root",
        type=Path,
        action="append",
        required=True,
        help="Directory whose direct-child *.db files will be scanned; repeat for multiple roots.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--flat_db_root",
        type=Path,
        default=None,
        help="Optional flat directory populated with relative symlinks for M6.4 --nuplan_db_root.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_inventory(
        db_roots=args.db_root,
        output_dir=args.output_dir,
        flat_db_root=args.flat_db_root,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
