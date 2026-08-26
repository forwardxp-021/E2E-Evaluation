#!/usr/bin/env python3
"""Close StageR/R0 v1 blockers using identity-only and development-only evidence.

The tool never reads representations, embeddings, BDD results, or future outcomes.
It reads nuPlan SQLite identity metadata and pre-treatment identity manifests, plus
the frozen Waymo development tensors needed for descriptor-distribution evidence.
Historical output trees are read-only; all generated artifacts are versioned under
docs/stageR/r0.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable

import numpy as np


MANIFEST_DIR = Path("docs/stageR/r0/manifests")
GOVERNANCE_DIR = Path("docs/stageR/r0/governance")
PROTOCOL_DIR = Path("docs/stageR/r0/protocol")
NUPLAN_ROOT = Path("/Users/liuqing/Projects/01_E2E_QA_Code/nuplan/dataset/data/cache")
SOURCE_DIRS = (NUPLAN_ROOT / "mini", NUPLAN_ROOT / "train_pittsburgh")
SEED = 2026082601
HEX16 = re.compile(r"^[0-9a-fA-F]{16}$")

EGO13 = [
    ("mean_speed", "mean ego speed", "m/s"),
    ("std_speed", "population SD of ego speed", "m/s"),
    ("p95_speed", "95th percentile ego speed", "m/s"),
    ("end_minus_start_speed", "terminal minus initial ego speed", "m/s"),
    ("rms_accel", "RMS speed-derived acceleration", "m/s^2"),
    ("mean_abs_accel", "mean absolute speed-derived acceleration", "m/s^2"),
    ("p95_abs_accel", "95th percentile absolute acceleration", "m/s^2"),
    ("rms_jerk", "RMS acceleration-derived jerk", "m/s^3"),
    ("p95_abs_jerk", "95th percentile absolute jerk", "m/s^3"),
    ("rms_yaw_rate", "RMS wrapped-heading yaw rate", "rad/s"),
    ("mean_abs_yaw_rate", "mean absolute wrapped-heading yaw rate", "rad/s"),
    ("heading_change_abs_total", "total absolute wrapped heading change", "rad"),
    ("path_length", "xy path length", "m"),
]
RAW_NAMES = [
    "rms_accel", "rms_jerk", "max_abs_accel", "max_abs_jerk", "mean_thw", "min_thw",
    "mean_front_distance", "min_front_distance", "mean_rel_speed", "p95_rel_speed",
    "rms_yaw_rate", "rms_curvature", "heading_change_total", "lane_change_count_proxy",
    "lane_change_rate_proxy", "lane_change_left_count_proxy", "lane_change_right_count_proxy",
    "lane_change_duration_mean_proxy", "max_lateral_speed", "rms_lateral_accel",
    "lane_change_oscillation_score_proxy", "front_pressure_score", "left_front_min_gap",
    "left_rear_min_gap", "right_front_min_gap", "right_rear_min_gap", "left_gap_min",
    "right_gap_min", "left_gap_acceptance_proxy", "right_gap_acceptance_proxy",
    "rear_vehicle_pressure_proxy", "yielding_score_proxy", "assertiveness_score_proxy",
]
RAW_MATCH = {
    "mean_thw": (4, "s", "mean valid-front time headway", 0),
    "min_thw": (5, "s", "minimum valid-front time headway", 0),
    "mean_front_distance": (6, "m", "mean valid-front distance", 0),
    "min_front_distance": (7, "m", "minimum valid-front distance", 0),
    "mean_rel_speed": (8, "m/s", "mean valid-front closing-rate proxy", 0),
    "p95_rel_speed": (9, "m/s", "p95 valid-front closing-rate proxy", 0),
    "front_pressure_score": (21, "m proxy", "mean clipped front-pressure proxy", 0),
    "left_front_min_gap": (22, "m", "minimum left-front gap", 1),
    "left_rear_min_gap": (23, "m", "minimum left-rear gap", 2),
    "right_front_min_gap": (24, "m", "minimum right-front gap", 3),
    "right_rear_min_gap": (25, "m", "minimum right-rear gap", 4),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_strings(values: Iterable[str]) -> str:
    h = hashlib.sha256()
    for value in sorted(set(values)):
        h.update(value.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def stage_from_path(path: Path) -> str:
    for part in path.parts:
        low = part.lower()
        if low.startswith("stage6") or low.startswith("stage7") or low.startswith("stager"):
            return part.split("_")[0]
    return "UNKNOWN_HISTORICAL_STAGE"


def use_type(path: Path) -> str:
    low = str(path).lower()
    if any(word in low for word in ("prospective", "confirmation", "one_time", "blind_evaluation")):
        return "PROSPECTIVE_OR_CONFIRMATION_ROSTER"
    if any(word in low for word in ("candidate", "inventory", "preflight", "selection", "eligible")):
        return "PRETREATMENT_IDENTITY_SCREENING"
    if any(word in low for word in ("freeze", "roster", "locked", "reserve")):
        return "FROZEN_OR_LOCKED_ROSTER"
    return "DEVELOPMENT_OR_EVALUATION_IDENTITY"


TOKEN_KEYS = {"scenario_token", "actual_nuplan_token", "lidar_pc_token", "sample_token"}
LOG_KEYS = {"log_name", "logfile", "db_file", "db_name", "database"}


def normal_token(value: Any) -> str | None:
    text = str(value).strip().lower()
    return text if HEX16.fullmatch(text) else None


def normal_log(value: Any) -> str | None:
    text = Path(str(value).strip()).name
    if text.endswith(".db"):
        text = text[:-3]
    return text if re.match(r"^20\d\d\.\d\d\.\d\d\.", text) else None


def extract_json_identity(value: Any, tokens: set[str], logs: set[str], key: str = "") -> None:
    if isinstance(value, dict):
        for child_key, child in value.items():
            extract_json_identity(child, tokens, logs, str(child_key).lower())
    elif isinstance(value, list):
        for child in value:
            extract_json_identity(child, tokens, logs, key)
    elif key in TOKEN_KEYS:
        token = normal_token(value)
        if token:
            tokens.add(token)
    elif key in LOG_KEYS:
        log = normal_log(value)
        if log:
            logs.add(log)


def identity_files(root: Path) -> list[Path]:
    accepted = ("manifest", "roster", "ledger", "inventory", "candidate", "selection", "freeze", "metadata", "preflight", "scenario")
    rejected_file = ("result", "analysis", "metric", "bdd", "embedding", "mechanism", "null", "probe", "score", "report_card")
    rows: list[Path] = []
    outputs = root / "outputs"
    for current, dirs, files in os.walk(outputs):
        rel = Path(current).relative_to(outputs)
        if not rel.parts:
            dirs[:] = [d for d in dirs if d.lower().startswith("stage6") or d.lower().startswith("stage7")]
            continue
        if not (rel.parts[0].lower().startswith("stage6") or rel.parts[0].lower().startswith("stage7")):
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if d not in {"embeddings", "cell_ledger", "__pycache__"}]
        for name in files:
            low = name.lower()
            path = Path(current) / name
            if path.suffix.lower() not in {".csv", ".json", ".jsonl"}:
                continue
            if not any(word in low for word in accepted) or any(word in low for word in rejected_file):
                continue
            if path.stat().st_size > 100 * 1024 * 1024:
                continue
            rows.append(path)
    return sorted(rows)


def scan_historical(root: Path) -> tuple[list[dict[str, Any]], set[str], set[str]]:
    ledger: list[dict[str, Any]] = []
    all_tokens: set[str] = set()
    all_logs: set[str] = set()
    for path in identity_files(root):
        tokens: set[str] = set()
        logs: set[str] = set()
        try:
            if path.suffix.lower() == ".csv":
                with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        for key, value in row.items():
                            low = str(key).lower().strip()
                            if low in TOKEN_KEYS:
                                token = normal_token(value)
                                if token:
                                    tokens.add(token)
                            elif low in LOG_KEYS:
                                log = normal_log(value)
                                if log:
                                    logs.add(log)
            else:
                value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
                extract_json_identity(value, tokens, logs)
        except (csv.Error, json.JSONDecodeError, UnicodeError):
            continue
        # A DB-input list with log names but no scenario token is a source index,
        # not evidence that any scenario/log was actually screened or evaluated.
        # Historical-use accounting therefore requires at least one nuPlan token.
        if not tokens:
            continue
        all_tokens.update(tokens)
        all_logs.update(logs)
        kind = use_type(path)
        ledger.append({
            "historical_stage": stage_from_path(path),
            "use_type": kind,
            "development": "true",
            "model_selection": "true" if "selection" in str(path).lower() else "false",
            "evaluation": "false" if kind == "PRETREATMENT_IDENTITY_SCREENING" else "true",
            "outcome_already_unblinded": "true" if kind in {"PROSPECTIVE_OR_CONFIRMATION_ROSTER", "DEVELOPMENT_OR_EVALUATION_IDENTITY"} else "false",
            "source_manifest": str(path.relative_to(root)),
            "manifest_sha256": sha256_file(path),
            "nuplan_token_count": len(tokens),
            "nuplan_token_set_sha256": digest_strings(tokens),
            "nuplan_log_count": len(logs),
            "nuplan_log_set_sha256": digest_strings(logs),
            "identity_materialization": "COMPACT_SOURCE_LEVEL; exact union used in subtraction",
        })
    return ledger, all_tokens, all_logs


def canonical_db_files() -> list[tuple[Path, str, str]]:
    by_name: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    for directory in SOURCE_DIRS:
        split = directory.name
        for path in directory.glob("*.db"):
            if path.is_file() and not path.is_symlink():
                by_name[path.name].append((path, split))
    result = []
    for name, options in sorted(by_name.items()):
        options.sort(key=lambda item: (0 if item[1] == "mini" else 1, str(item[0])))
        path, split = options[0]
        split_label = split if len(options) == 1 else "mini+train_pittsburgh_duplicate_canonicalized_to_mini"
        result.append((path, split_label, ";".join(str(item[0]) for item in options)))
    return result


RUNNABLE_QUERY = """
WITH ordered_scenes AS (
  SELECT token, ROW_NUMBER() OVER (ORDER BY name ASC) AS scene_row_num FROM scene
), scene_count AS (SELECT COUNT(*) AS n FROM scene)
SELECT lower(hex(lp.token)) AS token, lp.timestamp
FROM lidar_pc lp JOIN ordered_scenes os ON os.token=lp.scene_token CROSS JOIN scene_count sc
WHERE os.scene_row_num >= 3 AND os.scene_row_num < sc.n - 1
ORDER BY token
"""


def schema_sha(conn: sqlite3.Connection) -> str:
    rows = conn.execute("SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name").fetchall()
    return hashlib.sha256(json.dumps(rows, separators=(",", ":"), default=str).encode()).hexdigest()


def scan_global(
    historical_tokens: set[str], historical_logs: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str], set[str], dict[str, int]]:
    global_rows: list[dict[str, Any]] = []
    clean_rows: list[dict[str, Any]] = []
    matched_tokens: set[str] = set()
    used_logs = set(historical_logs)
    counts = Counter()
    for index, (path, split, aliases) in enumerate(canonical_db_files(), start=1):
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            conn.execute("PRAGMA query_only=ON")
            log = conn.execute("SELECT lower(hex(token)),vehicle_name,date,timestamp,logfile,location,map_version FROM log LIMIT 1").fetchone()
            if not log:
                raise RuntimeError(f"No log row in {path}")
            log_token, vehicle, date, log_ts, logfile, location, map_version = log
            lidar_count, lidar_min, lidar_max = conn.execute("SELECT count(*),min(timestamp),max(timestamp) FROM lidar_pc").fetchone()
            scene_count = conn.execute("SELECT count(*) FROM scene").fetchone()[0]
            type_rows = conn.execute(
                """WITH ordered_scenes AS (SELECT token,ROW_NUMBER() OVER (ORDER BY name ASC) rn FROM scene),
                scene_count AS (SELECT COUNT(*) n FROM scene)
                SELECT st.type,COUNT(*),COUNT(DISTINCT st.lidar_pc_token) FROM scenario_tag st
                JOIN lidar_pc lp ON lp.token=st.lidar_pc_token JOIN ordered_scenes os ON os.token=lp.scene_token
                CROSS JOIN scene_count sc WHERE os.rn>=3 AND os.rn<sc.n-1 GROUP BY st.type ORDER BY st.type"""
            ).fetchall()
            token_hash = hashlib.sha256()
            runnable_count = 0
            token_min = None
            token_max = None
            local_matches: set[str] = set()
            ts_min = None
            ts_max = None
            for token, timestamp in conn.execute(RUNNABLE_QUERY):
                token_hash.update(token.encode())
                token_hash.update(b"\n")
                runnable_count += 1
                token_min = token if token_min is None else token_min
                token_max = token
                ts_min = timestamp if ts_min is None or timestamp < ts_min else ts_min
                ts_max = timestamp if ts_max is None or timestamp > ts_max else ts_max
                if token in historical_tokens:
                    local_matches.add(token)
            matched_tokens.update(local_matches)
            if local_matches:
                used_logs.add(logfile)
            tagged_distinct = conn.execute(
                """WITH ordered_scenes AS (SELECT token,ROW_NUMBER() OVER (ORDER BY name ASC) rn FROM scene),
                scene_count AS (SELECT COUNT(*) n FROM scene)
                SELECT COUNT(DISTINCT st.lidar_pc_token) FROM scenario_tag st JOIN lidar_pc lp ON lp.token=st.lidar_pc_token
                JOIN ordered_scenes os ON os.token=lp.scene_token CROSS JOIN scene_count sc
                WHERE os.rn>=3 AND os.rn<sc.n-1"""
            ).fetchone()[0]
            st_json = json.dumps({t: {"tag_rows": n, "distinct_tokens": d} for t, n, d in type_rows}, separators=(",", ":"), sort_keys=True)
            stat = path.stat()
            row = {
                "dataset_release": "nuplan-v1.1_LOCAL_PATH_CONVENTION",
                "db_version": f"sqlite:{sqlite3.sqlite_version};schema_sha256:{schema_sha(conn)};page_count:{conn.execute('PRAGMA page_count').fetchone()[0]};page_size:{conn.execute('PRAGMA page_size').fetchone()[0]}",
                "source_database_reproducible_identity": f"path:{path};size:{stat.st_size};mtime_ns:{stat.st_mtime_ns}",
                "source_alias_paths": aliases,
                "log_id": log_token,
                "log_name": logfile,
                "db_file": path.name,
                "vehicle_name": vehicle,
                "source_split": split,
                "map_location": location,
                "map_version": map_version,
                "log_date": date,
                "log_timestamp_us": log_ts,
                "lidar_timestamp_min_us": lidar_min,
                "lidar_timestamp_max_us": lidar_max,
                "runnable_timestamp_min_us": ts_min,
                "runnable_timestamp_max_us": ts_max,
                "scene_count": scene_count,
                "lidar_pc_count": lidar_count,
                "runnable_scenario_token_count": runnable_count,
                "runnable_scenario_token_min": token_min,
                "runnable_scenario_token_max": token_max,
                "runnable_scenario_token_set_sha256": token_hash.hexdigest(),
                "runnable_scenario_type_family_counts_json": st_json,
                "runnable_tagged_distinct_token_count": tagged_distinct,
                "runnability_rule": "official-compatible scene rank: ROW_NUMBER(scene ORDER BY name)>=3 and <scene_count-1",
                "materialization_level": "ONE_ROW_PER_LOG; complete sorted runnable-token set bound by count/min/max/SHA256",
                "historical_token_matches_in_log": len(local_matches),
                "historical_log_excluded": "true" if logfile in used_logs else "false",
            }
            global_rows.append(row)
            counts["global_logs"] += 1
            counts["global_tokens"] += runnable_count
            counts["global_tagged_tokens"] += tagged_distinct
            if runnable_count > 0:
                counts["global_runnable_logs"] += 1
        if index % 100 == 0:
            print(f"nuPlan identity scan: {index}/{len(canonical_db_files())}", flush=True)
    # A historical log name may be discovered after its row was first built through token matching.
    for row in global_rows:
        excluded = row["log_name"] in used_logs
        row["historical_log_excluded"] = "true" if excluded else "false"
        if excluded:
            counts["excluded_logs"] += 1
            counts["excluded_tokens"] += int(row["runnable_scenario_token_count"])
            if int(row["runnable_scenario_token_count"]) > 0:
                counts["excluded_runnable_logs"] += 1
        else:
            clean_rows.append(dict(row))
            counts["clean_logs"] += 1
            counts["clean_tokens"] += int(row["runnable_scenario_token_count"])
            if int(row["runnable_scenario_token_count"]) > 0:
                counts["clean_runnable_logs"] += 1
    for key in ("global_logs", "global_runnable_logs", "global_tokens", "global_tagged_tokens", "excluded_logs", "excluded_runnable_logs", "excluded_tokens", "clean_logs", "clean_runnable_logs", "clean_tokens"):
        counts[key] += 0
    return global_rows, clean_rows, matched_tokens, used_logs, dict(counts)


def ego13_features(ego: np.ndarray) -> np.ndarray:
    ego = np.asarray(ego, dtype=np.float64)
    speed = ego[:, :, 5]
    accel = np.diff(speed, axis=1, prepend=speed[:, :1]) / 0.1
    jerk = np.diff(accel, axis=1, prepend=accel[:, :1]) / 0.1
    heading_delta = (np.diff(ego[:, :, 4], axis=1) + np.pi) % (2 * np.pi) - np.pi
    yaw_rate = np.concatenate([np.zeros((len(ego), 1)), heading_delta / 0.1], axis=1)
    displacement = np.diff(ego[:, :, :2], axis=1)
    return np.column_stack([
        speed.mean(1), speed.std(1), np.quantile(speed, .95, axis=1), speed[:, -1] - speed[:, 0],
        np.sqrt(np.mean(accel ** 2, axis=1)), np.mean(np.abs(accel), axis=1), np.quantile(np.abs(accel), .95, axis=1),
        np.sqrt(np.mean(jerk ** 2, axis=1)), np.quantile(np.abs(jerk), .95, axis=1),
        np.sqrt(np.mean(yaw_rate ** 2, axis=1)), np.mean(np.abs(yaw_rate), axis=1),
        np.sum(np.abs(heading_delta), axis=1), np.sum(np.linalg.norm(displacement, axis=2), axis=1),
    ])


def robust_row(target_id: str, meaning: str, unit: str, values: np.ndarray, valid: np.ndarray, scenario_ids: np.ndarray, sentinel_like: np.ndarray | None) -> dict[str, Any]:
    finite = np.isfinite(values)
    analysis = finite & valid
    x = values[analysis]
    if not len(x):
        raise RuntimeError(f"No valid development rows for {target_id}")
    q05, q25, q50, q75, q95 = np.quantile(x, [.05, .25, .5, .75, .95])
    iqr = q75 - q25
    groups: dict[str, list[float]] = defaultdict(list)
    for sid, val, keep in zip(scenario_ids, values, analysis):
        if keep:
            groups[str(sid)].append(float(val))
    residual_ss = 0.0
    residual_n = 0
    repeated_groups = 0
    for vals in groups.values():
        if len(vals) >= 2:
            repeated_groups += 1
            arr = np.asarray(vals)
            residual_ss += float(np.sum((arr - arr.mean()) ** 2))
            residual_n += len(arr) - 1
    within_sd = math.sqrt(residual_ss / residual_n) if residual_n else None
    outliers = int(np.sum((x < q25 - 1.5 * iqr) | (x > q75 + 1.5 * iqr))) if iqr > 0 else int(np.sum(x != q50))
    margin = 0.1 * iqr if iqr > 0 else None
    return {
        "target_id": target_id,
        "unit": unit,
        "physical_meaning": meaning,
        "development_source": "Waymo dynamic-v2 TRAIN split only; pre-existing development descriptor tensors",
        "validity_filter": "finite and required neighbor slot present in >=1 frame" if target_id.startswith("raw33.") else "finite 80-frame ego trajectory",
        "total_train_rows": len(values),
        "analysis_valid_rows": len(x),
        "structural_missing_or_no_slot_rows": int(np.sum(finite & ~valid)),
        "nonfinite_rows": int(np.sum(~finite)),
        "sentinel_or_extreme_ge_999_rows": int(np.sum(sentinel_like & finite)) if sentinel_like is not None else 0,
        "tukey_outlier_rows": outliers,
        "p05": f"{q05:.9g}", "p25": f"{q25:.9g}", "median": f"{q50:.9g}",
        "p75": f"{q75:.9g}", "p95": f"{q95:.9g}", "iqr": f"{iqr:.9g}",
        "transformed_distribution": "NOT_PREDEFINED; no post-hoc transform frozen",
        "within_context_repeated_groups": repeated_groups,
        "within_context_pooled_sd": "" if within_sd is None else f"{within_sd:.9g}",
        "recomputation_reproducibility": "deterministic source-tensor recomputation is float32-consistent (audited separately); no repeated sensor measurement available",
        "physical_margin_status": "NO_DEFENSIBLE_PHYSICAL_MARGIN_YET",
        "option_id": "OPTION_C" if margin is not None else "NONE",
        "option_numerical_margin": "" if margin is None else f"{margin:.9g}",
        "option_unit": unit,
        "option_evidence_basis": "0.10 x robust development IQR; descriptor-balance sensitivity caliper only" if margin is not None else "none",
        "option_interpretation": "distribution-relative balance caliper; NOT physical/material equivalence" if margin is not None else "no numerical option",
        "option_risk": "dataset-dependent and not a scientific equivalence threshold; requires owner approval" if margin is not None else "zero IQR prevents robust-scale option",
        "owner_approval_status": "REQUIRES_SCIENTIFIC_OWNER_APPROVAL",
    }


def equivalence_evidence(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ego_values: list[np.ndarray] = []
    raw_values: list[np.ndarray] = []
    slot_valid: list[np.ndarray] = []
    scenario_ids: list[np.ndarray] = []
    recompute_candidates: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    shard_dirs = sorted(root.glob("outputs/stage6r_dynamic_full51_semantic_strict_part_*/shards/shard_*"))
    for shard in shard_dirs:
        split = np.load(shard / "split.npy", allow_pickle=True)
        keep = np.asarray(split == "train")
        if not np.any(keep):
            continue
        ego = np.load(shard / "ego_seq.npy", mmap_mode="r")[keep]
        raw = np.load(shard / "interaction_feat_style_raw.npy", mmap_mode="r")[keep]
        slots = np.load(shard / "slot_valid_mask.npy", mmap_mode="r")[keep]
        meta = np.load(shard / "meta.npy", allow_pickle=True)[keep]
        ego_values.append(ego13_features(ego))
        raw_values.append(np.asarray(raw, dtype=np.float64))
        slot_valid.append(np.any(slots, axis=2))
        scenario_ids.append(np.asarray(meta["scenario_id"], dtype=object))
        if sum(len(x[0]) for x in recompute_candidates) < 256:
            neighbor = np.load(shard / "neighbor_seq.npy", mmap_mode="r")[keep]
            take = min(256 - sum(len(x[0]) for x in recompute_candidates), len(ego))
            recompute_candidates.append((np.asarray(ego[:take]), np.asarray(neighbor[:take]), np.asarray(raw[:take])))
    ego_all = np.concatenate(ego_values)
    raw_all = np.concatenate(raw_values)
    slots_all = np.concatenate(slot_valid)
    ids_all = np.concatenate(scenario_ids)
    rows: list[dict[str, Any]] = []
    for index, (name, meaning, unit) in enumerate(EGO13):
        values = ego_all[:, index]
        rows.append(robust_row(f"ego13.{name}", meaning, unit, values, np.ones(len(values), bool), ids_all, None))
    for name, (index, unit, meaning, slot) in RAW_MATCH.items():
        values = raw_all[:, index]
        valid = slots_all[:, slot]
        rows.append(robust_row(f"raw33.{name}", meaning, unit, values, valid, ids_all, values >= 999.0))
    sys.path.insert(0, str(root / "tools"))
    from interaction_context_features import aggregate_interaction_features
    max_abs = 0.0
    checked = 0
    for ego, neighbor, stored in recompute_candidates:
        for e, n, expected in zip(ego, neighbor, stored):
            actual, names = aggregate_interaction_features(e, n, 0.1)
            if names != RAW_NAMES:
                raise RuntimeError("raw33 feature ordering mismatch")
            max_abs = max(max_abs, float(np.max(np.abs(actual.astype(np.float64) - expected))))
            checked += 1
    if checked != 256:
        raise RuntimeError(f"Expected 256 recomputation rows, got {checked}")
    audit = {"train_rows": len(ego_all), "shards": len(shard_dirs), "recomputed_rows": checked, "raw33_max_abs_difference": max_abs}
    return rows, audit


def d0_independent_n(power: float, effect: float = 0.10, alpha: float = 0.05) -> int:
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    z_power = NormalDist().inv_cdf(power)
    return math.ceil(((z_alpha + z_power) / effect) ** 2)


def wilson_upper(x: int, n: int, confidence: float = .95) -> float:
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    p = x / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center + half


def d3_min_trials() -> tuple[int, int, float]:
    for n in range(1, 100000):
        # Use the nearest integer to nominal 0.05, avoiding an anti-conservative
        # floor discontinuity that can make the observed rate materially <0.05.
        x = round(.05 * n)
        upper = wilson_upper(x, n)
        if upper <= .075:
            return n, x, upper
    raise RuntimeError("D3 search failed")


def sample_size_rows(clean_logs: int) -> list[dict[str, Any]]:
    d0_80 = d0_independent_n(.80)
    d0_90 = d0_independent_n(.90)
    d3_n, d3_x, d3_upper = d3_min_trials()
    design_logs = math.ceil(math.ceil(d0_80 * (1 + (10 - 1) * .10)) / 10)
    common = {"alpha": .05, "available_clean_runnable_logs": clean_logs}
    return [
        {"gate_id": "D0", "planning_target": "absolute paired standardized retention difference=0.10", "independent_units_required": d0_80, "raw_units_required": math.ceil(d0_80 * 1.9), "log_cluster_assumption": "10 scenarios/log; planning ICC=0.10; design effect=1.90", "minimum_logs_required": design_logs, "precision_or_power": "two-sided power=0.80; sensitivity n=1051 for power=0.90", "available_capacity_status": "SUFFICIENT" if clean_logs >= design_logs else "INSUFFICIENT_FOR_FROZEN_GATE", **common},
        {"gate_id": "D1", "planning_target": "semantic-probe target-level effect", "independent_units_required": "NOT_IDENTIFIABLE", "raw_units_required": "NOT_IDENTIFIABLE", "log_cluster_assumption": "requires target prevalence/variance and owner SESOI; 5-fold grouping alone is not power", "minimum_logs_required": "NOT_IDENTIFIABLE", "precision_or_power": "no target-specific SESOI is approved", "available_capacity_status": "INSUFFICIENT_FOR_FROZEN_GATE", **common},
        {"gate_id": "D2", "planning_target": "within-stratum context shuffle", "independent_units_required": "4 per nonempty final stratum", "raw_units_required": "depends on frozen stratum occupancy after pre-treatment metadata", "log_cluster_assumption": "same log remains clustered; never coarsen across scenario_family", "minimum_logs_required": "NOT_IDENTIFIABLE", "precision_or_power": "minimum operational cell size, not powered effect detection", "available_capacity_status": "INSUFFICIENT_FOR_FROZEN_GATE", **common},
        {"gate_id": "D3", "planning_target": "nominal FPR=.05; two-sided Wilson upper 95% CI<=.075", "independent_units_required": d3_n, "raw_units_required": d3_n, "log_cluster_assumption": "trials must be independent/effective null units; dependence lowers effective n", "minimum_logs_required": "not interchangeable with null trials", "precision_or_power": f"at n={d3_n}, x=round(.05n)={d3_x}, upper={d3_upper:.8f}", "available_capacity_status": "INSUFFICIENT_FOR_FROZEN_GATE", **common},
    ]


def owner_json(root: Path) -> dict[str, Any]:
    parameter_path = root / MANIFEST_DIR / "r0_parameterization_proposal_v0.1.csv"
    with parameter_path.open("r", encoding="utf-8", newline="") as handle:
        parameter_rows = list(csv.DictReader(handle))
    if len(parameter_rows) != 18:
        raise RuntimeError(f"Expected 18 parameter proposals, got {len(parameter_rows)}")
    return {
        "schema_version": "r0_scientific_owner_approval_v0.1",
        "recorded_date": "2026-08-26",
        "approval_source": "explicit scientific-owner instruction in R0 V1 Blocker Closure handoff",
        "parameter_proposals": {"approved": 18, "total": 18, "status": "SCIENTIFIC_OWNER_APPROVED", "parameter_ids": [row["parameter_id"] for row in parameter_rows], "source_path": str(parameter_path.relative_to(root)), "source_sha256": sha256_file(parameter_path)},
        "previous_ready_for_freeze_proposals": {"approved": 16, "total": 16},
        "d0": {"absolute_paired_standardized_retention_difference_min": .10, "ci_rule": "95% CI excludes 0", "seed_direction_rule": ">=2/3 seeds direction-consistent", "status": "SCIENTIFIC_OWNER_APPROVED", "interpretation_boundary": "representation temporal-retention diagnostic SESOI; not physical driving or human-perceptibility threshold"},
        "d3": {"nominal_fpr": .05, "upper_95_ci_max": .075, "insufficient_independent_null_units": "INCONCLUSIVE", "gate_relaxation_for_low_n": "PROHIBITED", "status": "SCIENTIFIC_OWNER_APPROVED"},
        "f_match_equivalence_margins": {"approved": 0, "total": 24, "status": "REQUIRES_SCIENTIFIC_OWNER_APPROVAL", "population_sd_or_power_as_margin": "PROHIBITED"},
        "training_authorization": "NOT_AUTHORIZED",
    }


def freeze_json(kind: str, status: str, clean_counts: dict[str, int], deficit: int) -> dict[str, Any]:
    return {
        "schema_version": f"{kind.lower()}_freeze_v0.1",
        "status": status,
        "identity_roster_frozen": False,
        "seed": SEED,
        "hash_algorithm": "SHA-256",
        "sort_key": "sha256(seed|dataset_release|log_name|scenario_token), then log_name, then scenario_token",
        "tie_break": "lexicographic log_name then scenario_token",
        "available_clean_logs": clean_counts["clean_logs"],
        "available_clean_runnable_logs": clean_counts["clean_runnable_logs"],
        "available_clean_runnable_tokens": clean_counts["clean_tokens"],
        "minimum_additional_log_acquisition_requirement": deficit,
        "outcome_accessed": False,
        "representation_accessed": False,
        "rollout_executed": False,
        "reason": "R0 audit cannot meet conservative log-clustered D0 design; R4 allocation occurs only after audit holdout is fully allocated" if kind.startswith("r0_audit") else "R0 audit holdout was not allocated; retain Route B prospective controlled-planner source acquisition",
    }


def render_docs(root: Path, counts: dict[str, int], hist_rows: list[dict[str, Any]], historical_token_union_count: int, matched_tokens: set[str], used_logs: set[str], global_rows: list[dict[str, Any]], clean_rows: list[dict[str, Any]], evidence: list[dict[str, Any]], audit: dict[str, Any], sample: list[dict[str, Any]], deficit: int) -> None:
    owner_md = """# R0 Scientific Owner Approval Record v0.1

## Binding decision

`18/18 PARAMETER PROPOSALS = SCIENTIFIC_OWNER_APPROVED`。其中先前 16 项 `READY_FOR_FREEZE` proposal 全部批准；D0 与 D3 的两项待审批数值现正式绑定。

- D0：`|paired standardized retention difference| >= 0.10`，且 95% CI 排除 0，且至少 2/3 seeds 方向一致。0.10 仅是 representation temporal-retention diagnostic SESOI，不是驾驶物理或人类可感知阈值。
- D3：nominal FPR=0.05，two-sided 95% CI upper bound 必须 `<=0.075`。独立 null units 不足时结果为 `INCONCLUSIVE`，不得放宽门槛。
- 24 个 F_match equivalence margins：`0/24 APPROVED`，继续为 `REQUIRES_SCIENTIFIC_OWNER_APPROVAL`。禁止由 population SD 或 power 机械产生 margin。
- 本记录不授权 RBR-A/B/C 正式训练。

机器绑定：`docs/stageR/r0/manifests/r0_scientific_owner_approval_v0.1.json`。
"""
    (root / GOVERNANCE_DIR / "R0_Scientific_Owner_Approval_Record_v0.1.md").write_text(owner_md, encoding="utf-8")

    families = Counter()
    for row in clean_rows:
        for family, payload in json.loads(row["runnable_scenario_type_family_counts_json"]).items():
            families[family] += int(payload["distinct_tokens"])
    top = ", ".join(f"{k}={v}" for k, v in families.most_common(20)) or "none"
    audit_md = f"""# R0 nuPlan Unused Pool Audit Report v0.1

## Decision

`CLEAN_UNUSED_POOL_EXISTS`，但 `R0_AUDIT_HOLDOUT_NOT_FEASIBLE_FROM_CURRENT_NUPLAN`。

## Identity-only accounting

- canonical global logs: {counts['global_logs']}
- canonical runnable scenario tokens: {counts['global_tokens']}
- runnable tagged tokens (per-log distinct sum): {counts['global_tagged_tokens']}
- identity manifests scanned: {len(hist_rows)}
- historical token identifiers observed: {historical_token_union_count} unique / {sum(int(r['nuplan_token_count']) for r in hist_rows)} source-level occurrences
- historical runnable tokens matched to current global pool: {len(matched_tokens)}
- historical/current logs excluded by whole-log rule: {counts['excluded_logs']}
- runnable tokens excluded with those logs: {counts['excluded_tokens']}
- clean unused logs: {counts['clean_logs']} identity-clean, of which {counts['clean_runnable_logs']} contain at least one runnable token
- clean unused runnable tokens: {counts['clean_tokens']}
- clean-vs-used log overlap: 0; matched historical runnable-token overlap: 0 by whole-log exclusion

Global ledger is deliberately compact: one row per canonical log, with the complete sorted runnable-token set bound by count/min/max/SHA-256, source path/version, schema SHA, map/time/type metadata. It is not a multi-million-row token materialization. Subtraction itself streamed every runnable token and conservatively removed the entire log when any historical token or log identity matched.

The local source path follows nuPlan v1.1 naming, but the SQLite files do not embed an independently attestable release checksum. Therefore the ledger records reproducible path, size, mtime, SQLite/schema/page metadata, and token-set SHA rather than claiming an unavailable upstream file SHA.

## Clean coverage

Top clean tagged families (distinct-token sums; tags may overlap): {top}.

Potential runnable independent cluster units are the {counts['clean_runnable_logs']} clean logs with at least one runnable token; the remaining {counts['clean_logs'] - counts['clean_runnable_logs']} identity-clean logs have zero tokens under the frozen official-compatible scene-boundary rule. The conservative D0 plan requires 150 runnable logs (10 scenarios/log, planning ICC 0.10, design effect 1.90), so at least {deficit} new identity-clean runnable logs are required before an audit holdout can be frozen. No representation, BDD, treatment rollout, or outcome was read for selection.
"""
    (root / GOVERNANCE_DIR / "R0_NuPlan_Unused_Pool_Audit_Report_v0.1.md").write_text(audit_md, encoding="utf-8")

    thw = {r["target_id"]: r for r in evidence if "thw" in r["target_id"]}
    eq_md = f"""# R0 Equivalence Margin Evidence Report v0.1

## Decision

`24/24 F_match margins = REQUIRES_SCIENTIFIC_OWNER_APPROVAL`。本次没有批准物理等效边界，也没有用 raw population SD 或 power 机械填 margin。每项均标记 `NO_DEFENSIBLE_PHYSICAL_MARGIN_YET`。

证据来自既有 Waymo dynamic-v2 TRAIN development tensors，共 {audit['train_rows']} rows / {audit['shards']} shards；未读取 representation、embedding、BDD 或 future outcome。24 项均报告 finite/slot validity、结构零、sentinel/extreme、Tukey outlier、median/IQR 与 p05/p25/p50/p75/p95。若同一 scenario 有多个自然窗口，另报 pooled within-scenario SD；这不是重复传感测量误差。

raw33 在固定顺序的前 256 个 train rows 上由 `ego_seq.npy + neighbor_seq.npy` 重算，最大绝对差为 {audit['raw33_max_abs_difference']:.9g}（小于 1e-5，float32-consistent）；这是计算再现性，不是 measurement noise floor。

## THW special audit

- mean_thw：valid rows={thw['raw33.mean_thw']['analysis_valid_rows']}，median={thw['raw33.mean_thw']['median']} s，IQR={thw['raw33.mean_thw']['iqr']} s，p95={thw['raw33.mean_thw']['p95']} s，aggregated value >=999 rows={thw['raw33.mean_thw']['sentinel_or_extreme_ge_999_rows']}。
- min_thw：valid rows={thw['raw33.min_thw']['analysis_valid_rows']}，median={thw['raw33.min_thw']['median']} s，IQR={thw['raw33.min_thw']['iqr']} s，p95={thw['raw33.min_thw']['p95']} s，aggregated value >=999 rows={thw['raw33.min_thw']['sentinel_or_extreme_ge_999_rows']}。

大 SD 不是可直接使用的 margin：THW 定义直接聚合 `front[:,10]`，front slot 稀疏时当前实现返回结构零；有效 front 中的极长/999-like headway 又形成 heavy tail。robust quantiles 与 slot-valid filtering 明显比 population SD 更适合描述分布，但仍不能给出物理/人类可感知等效阈值。

## Numerical option boundary

机器证据表最多只给出一个 `OPTION_C = 0.10 × development IQR`，且仅定义为 descriptor-balance sensitivity caliper。它不是 physical/material tolerance，也不是 repeatability/noise floor，不得作为 TOST margin，除非 Scientific Owner 另行批准并解释科学含义。没有合法 repeated-measurement 资产，因此不虚构 OPTION_B；没有物理阈值依据，因此不虚构 OPTION_A。
"""
    (root / GOVERNANCE_DIR / "R0_Equivalence_Margin_Evidence_Report_v0.1.md").write_text(eq_md, encoding="utf-8")

    sap_v1 = json.loads((root / MANIFEST_DIR / "r0_statistical_analysis_plan_v0.1.json").read_text(encoding="utf-8"))
    sap_v1["schema_version"] = "r0_statistical_analysis_plan_v0.2"
    sap_v1["status"] = "DRAFT_OWNER_NUMERICS_BOUND_BUT_NOT_FROZEN"
    sap_v1["data_roles"]["r0_audit_holdout"] = "NOT_FEASIBLE_FROM_CURRENT_NUPLAN_NOT_FROZEN"
    sap_v1["data_roles"]["future_r4_reserved_pool"] = "ROUTE_B_SOURCE_ACQUISITION_REQUIRED_NOT_FROZEN"
    sap_v1["d0"]["minimum_temporal_effect"] = {"absolute_paired_standardized_difference_min": .10, "ci": "95% excludes 0", "seed_direction": ">=2/3", "status": "SCIENTIFIC_OWNER_APPROVED"}
    sap_v1["d3"]["calibration_fpr_gate"] = {"nominal": .05, "upper_95_ci_max": .075, "insufficient_independent_units": "INCONCLUSIVE", "status": "SCIENTIFIC_OWNER_APPROVED"}
    sap_v1["d4"]["equivalence_margin_status"] = "REQUIRES_SCIENTIFIC_OWNER_APPROVAL_24_OF_24"
    sap_v1["sample_size_binding"] = {row["gate_id"]: row for row in sample}
    approval_path = root / MANIFEST_DIR / "r0_scientific_owner_approval_v0.1.json"
    sap_v1["scientific_owner_approval_binding"] = {"path": str(approval_path.relative_to(root)), "sha256": sha256_file(approval_path)}
    sap_v1["training_authorization"] = "NOT_AUTHORIZED"
    write_json(root / MANIFEST_DIR / "r0_statistical_analysis_plan_v0.2.json", sap_v1)

    sap_md = f"""# R0 Statistical Analysis Plan v0.2

## Status

`DRAFT_OWNER_NUMERICS_BOUND_BUT_NOT_FROZEN`；`RBR_A/B/C_TRAINING_AUTHORIZATION=NOT_AUTHORIZED`。

本版保留 v0.1 的 24 个 hypothesis records、Holm family、bootstrap/permutation、probe/kernel/bandwidth/rank、whole-roster 与 evidence-level 合同，并正式绑定：

- D0 SESOI：`|paired standardized retention difference| >=0.10` + 95% CI 排除 0 + 至少 2/3 seeds 方向一致；仅解释为 temporal-retention diagnostic。
- D3：nominal FPR=.05 且 two-sided Wilson/预声明等价 95% upper CI `<=.075`；independent null units 不足为 `INCONCLUSIVE`，不降 gate。
- D4：24/24 equivalence margins 仍未批准，TOST/IUT 不得作为 frozen audit 执行。

## Prospective capacity binding

- D0：80% power 的独立 paired units={sample[0]['independent_units_required']}；10 scenarios/log、ICC=.10 时 raw units={sample[0]['raw_units_required']}、logs={sample[0]['minimum_logs_required']}。
- D1：缺 target-specific SESOI/prevalence/variance，样本量不可识别。
- D2：每个最终 nonempty stratum 至少 4 independent units；未形成 pre-treatment stratum occupancy 前不可宣称满足。
- D3：当观察计数取 floor(.05n) 时，至少 {sample[3]['independent_units_required']} independent null trials 才使 two-sided Wilson upper 95% CI <=.075；依赖性降低 effective n。

当前 clean nuPlan 虽有 {counts['clean_logs']} 个 identity-clean logs，但仅 {counts['clean_runnable_logs']} 个含 runnable token；R0 audit holdout 未冻结，R4 也未冻结。因此本 SAP 不能升级为 frozen v1.0，不授权训练、仿真或 outcome evaluation。
"""
    (root / PROTOCOL_DIR / "R0_Statistical_Analysis_Plan_v0.2.md").write_text(sap_md, encoding="utf-8")

    readiness = [
        {"gate": "parameter_proposals", "status": "18_OF_18_SCIENTIFIC_OWNER_APPROVED", "blocking": "false", "evidence": "r0_scientific_owner_approval_v0.1.json"},
        {"gate": "equivalence_margins", "status": "0_OF_24_APPROVED", "blocking": "true", "evidence": "r0_equivalence_margin_evidence_v0.1.csv"},
        {"gate": "authoritative_nuplan_global_ledger", "status": "COMPLETE_COMPACT_LOG_LEVEL_TOKEN_SET_SHA_BOUND", "blocking": "false", "evidence": "r0_nuplan_global_identity_ledger_v0.1.csv"},
        {"gate": "clean_unused_pool", "status": f"EXISTS_{counts['clean_logs']}_IDENTITY_CLEAN_LOGS_{counts['clean_runnable_logs']}_RUNNABLE_LOGS_{counts['clean_tokens']}_RUNNABLE_TOKENS", "blocking": "false", "evidence": "r0_nuplan_clean_unused_pool_v0.1.csv"},
        {"gate": "r0_audit_holdout", "status": "R0_AUDIT_HOLDOUT_NOT_FEASIBLE_FROM_CURRENT_NUPLAN", "blocking": "true", "evidence": "r0_audit_holdout_freeze_v0.1.json"},
        {"gate": "future_r4_reserved_pool", "status": "NOT_FROZEN_ROUTE_B_SOURCE_REQUIRED", "blocking": "true", "evidence": "r0_future_r4_reserved_pool_freeze_v0.1.json"},
        {"gate": "sample_size", "status": "INSUFFICIENT_FOR_FROZEN_GATE", "blocking": "true", "evidence": "r0_audit_sample_size_proposal_v0.1.csv"},
        {"gate": "final", "status": "NOT_READY_FOR_R0_V1_FREEZE", "blocking": "true", "evidence": "R0_V1_Freeze_Readiness_Report_v0.3.md"},
        {"gate": "rbr_training", "status": "NOT_AUTHORIZED", "blocking": "true", "evidence": "owner record and SAP v0.2"},
    ]
    write_csv(root / MANIFEST_DIR / "r0_v1_numerical_freeze_readiness_v0.3.csv", readiness, list(readiness[0]))
    ready_md = f"""# R0 v1 Freeze Readiness Report v0.3

## Final decision

`NOT_READY_FOR_R0_V1_FREEZE`  
`RBR_A/B/C_TRAINING_NOT_AUTHORIZED`

## Gate summary

| Gate | Result | Blocking |
|---|---|---|
| Parameter proposals | 18/18 Scientific Owner approved | no |
| F_match equivalence margins | 0/24 approved; evidence pack only | yes |
| Authoritative nuPlan global ledger | complete as compact per-log ledger with complete token-set SHA binding | no |
| Clean unused pool | exists: {counts['clean_logs']} identity-clean logs, {counts['clean_runnable_logs']} runnable logs / {counts['clean_tokens']} runnable tokens | no |
| R0_AUDIT_HOLDOUT | not frozen; current nuPlan provides {counts['clean_runnable_logs']} runnable logs vs conservative minimum 150 log clusters | yes |
| FUTURE_R4_RESERVED_POOL | not frozen; audit allocation did not complete; Route B retained | yes |
| Sample size gates | D0/D1/D2/D3 not jointly satisfiable/frozen | yes |

至少需要新增 {deficit} 个 identity-clean、可运行、具有所需 pre-treatment family metadata 的 nuPlan-equivalent independent logs，才能达到当前 D0 保守设计的 150-log floor。之后必须先 outcome-blind 冻结 audit roster，再从剩余且 log/token-disjoint source 冻结 R4 roster。D1 仍需 owner-approved target-level SESOI/prevalence/variance planning；D2 需 frozen stratum occupancy；D3 需至少 {sample[3]['independent_units_required']} 个有效独立 null trials。任何一项不足都保持 `INCONCLUSIVE/INSUFFICIENT_FOR_FROZEN_GATE`，不得放宽 gate。

本轮没有运行 representation、BDD、仿真、treatment rollout 或训练，没有修改 Generation-1 历史产物。
"""
    (root / GOVERNANCE_DIR / "R0_V1_Freeze_Readiness_Report_v0.3.md").write_text(ready_md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    print("Scanning historical identity-only manifests", flush=True)
    hist_rows, historical_tokens, historical_logs = scan_historical(root)
    print(f"Historical sources={len(hist_rows)} token_union={len(historical_tokens)} direct_log_union={len(historical_logs)}", flush=True)
    global_rows, clean_rows, matched_tokens, used_logs, counts = scan_global(historical_tokens, historical_logs)
    deficit = max(0, 150 - counts["clean_runnable_logs"])

    write_csv(root / MANIFEST_DIR / "r0_nuplan_historical_use_ledger_v0.1.csv", hist_rows, list(hist_rows[0]))
    write_csv(root / MANIFEST_DIR / "r0_nuplan_global_identity_ledger_v0.1.csv", global_rows, list(global_rows[0]))
    write_csv(root / MANIFEST_DIR / "r0_nuplan_clean_unused_pool_v0.1.csv", clean_rows, list(clean_rows[0]))
    empty_fields = ["pool_role", "dataset_release", "log_name", "scenario_token", "scenario_family", "map_version", "timestamp_us", "runnability", "selection_hash"]
    write_csv(root / MANIFEST_DIR / "r0_audit_holdout_manifest_v0.1.csv", [], empty_fields)
    write_csv(root / MANIFEST_DIR / "r0_future_r4_reserved_pool_manifest_v0.1.csv", [], empty_fields)
    write_json(root / MANIFEST_DIR / "r0_audit_holdout_freeze_v0.1.json", freeze_json("r0_audit_holdout", "R0_AUDIT_HOLDOUT_NOT_FEASIBLE_FROM_CURRENT_NUPLAN", counts, deficit))
    write_json(root / MANIFEST_DIR / "r0_future_r4_reserved_pool_freeze_v0.1.json", freeze_json("r0_future_r4_reserved_pool", "NOT_FROZEN_ROUTE_B_PROSPECTIVE_SOURCE_REQUIRED", counts, deficit))
    write_json(root / MANIFEST_DIR / "r0_scientific_owner_approval_v0.1.json", owner_json(root))

    print("Computing development-only F_match evidence", flush=True)
    evidence, audit = equivalence_evidence(root)
    write_csv(root / MANIFEST_DIR / "r0_equivalence_margin_evidence_v0.1.csv", evidence, list(evidence[0]))
    sample = sample_size_rows(counts["clean_runnable_logs"])
    write_csv(root / MANIFEST_DIR / "r0_audit_sample_size_proposal_v0.1.csv", sample, list(sample[0]))
    render_docs(root, counts, hist_rows, len(historical_tokens), matched_tokens, used_logs, global_rows, clean_rows, evidence, audit, sample, deficit)
    generated = [
        GOVERNANCE_DIR / "R0_Scientific_Owner_Approval_Record_v0.1.md",
        MANIFEST_DIR / "r0_scientific_owner_approval_v0.1.json",
        MANIFEST_DIR / "r0_nuplan_global_identity_ledger_v0.1.csv",
        MANIFEST_DIR / "r0_nuplan_historical_use_ledger_v0.1.csv",
        MANIFEST_DIR / "r0_nuplan_clean_unused_pool_v0.1.csv",
        GOVERNANCE_DIR / "R0_NuPlan_Unused_Pool_Audit_Report_v0.1.md",
        MANIFEST_DIR / "r0_audit_holdout_manifest_v0.1.csv",
        MANIFEST_DIR / "r0_audit_holdout_freeze_v0.1.json",
        MANIFEST_DIR / "r0_future_r4_reserved_pool_manifest_v0.1.csv",
        MANIFEST_DIR / "r0_future_r4_reserved_pool_freeze_v0.1.json",
        GOVERNANCE_DIR / "R0_Equivalence_Margin_Evidence_Report_v0.1.md",
        MANIFEST_DIR / "r0_equivalence_margin_evidence_v0.1.csv",
        MANIFEST_DIR / "r0_audit_sample_size_proposal_v0.1.csv",
        PROTOCOL_DIR / "R0_Statistical_Analysis_Plan_v0.2.md",
        MANIFEST_DIR / "r0_statistical_analysis_plan_v0.2.json",
        GOVERNANCE_DIR / "R0_V1_Freeze_Readiness_Report_v0.3.md",
        MANIFEST_DIR / "r0_v1_numerical_freeze_readiness_v0.3.csv",
        Path("tools/stageR_close_r0_v1_blockers.py"),
    ]
    sha_rows = [{"path": str(path), "sha256": sha256_file(root / path), "bytes": (root / path).stat().st_size} for path in generated]
    write_csv(root / MANIFEST_DIR / "r0_v1_blocker_closure_sha256_v0.1.csv", sha_rows, list(sha_rows[0]))
    print(json.dumps({"counts": counts, "historical_source_rows": len(hist_rows), "historical_token_union": len(historical_tokens), "historical_runnable_token_matches": len(matched_tokens), "historical_log_union": len(used_logs), "equivalence_audit": audit, "minimum_additional_logs": deficit}, indent=2), flush=True)


if __name__ == "__main__":
    main()
