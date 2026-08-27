#!/usr/bin/env python3
"""Prepare R1 Phase B1 synthetic generator and official-DB inventories.

This tool is deliberately limited to analytical/synthetic trajectories and
read-only SQLite/map inventory.  It never starts nuPlan simulation, selects a
roster, or opens representation, BDD, probe, checkpoint, or RBR artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_context_mechanism_core import (
    calculate_hlc_option_b,
    qualify_hlc_pair,
    trajectory_descriptors,
)
from tools.stage7l_pure_lateral_execution_planner import derive_trajectory_states, quintic_blend


ROOT = Path(__file__).resolve().parents[1]
R1_DIR = ROOT / "docs/stageR/r1"
DEFAULT_DB_CACHE_ROOT = ROOT.parent / "nuplan/dataset/data/cache"
DEFAULT_MAP_ROOT = ROOT.parent / "nuplan/dataset/maps"
BLACKLIST = R1_DIR / "r1_technical_smoke_v1_permanent_blacklist_v1.json"
RAW_EVIDENCE = R1_DIR / "r1_phasea_raw_trajectory_evidence_v0.1.json"
HLC_CONTRACT = R1_DIR / "r1_hlc_mechanism_contract_v1.0.json"
FUTURE_R4_FREEZE = ROOT / "docs/stageR/r0/manifests/r0_future_r4_reserved_pool_freeze_v0.1.json"
REPLAY_CONTRACT = R1_DIR / "r1_official_nuplan_replay_contract_v0.1.json"

MASTER_SEED = 2026082701
PRIMARY_HLC_CALIPERS = {
    "mean_speed": 0.708203939,
    "end_minus_start_speed": 0.978755681,
    "path_length": 5.38423459,
}
SECONDARY_HEADING_CALIPER = 0.0492160141
ENGINEERING_LIMITS = {
    "max_abs_lateral_accel_mps2": 6.0,
    "max_abs_yaw_rate_radps": 1.0,
    "max_abs_curvature_inv_m": 0.5,
}
HLC_GEN_V2_OPTIONS: Dict[str, Dict[str, float]] = {
    "HLC_GEN_V2_OPTION_A": {
        "advance_target_p": 0.35,
        "advance_seconds": 1.2,
        "hold_seconds": 0.5,
        "retreat_delta_p": 0.15,
        "retreat_seconds": 0.8,
        "recommit_seconds": 2.2,
    },
    "HLC_GEN_V2_OPTION_B": {
        "advance_target_p": 0.38,
        "advance_seconds": 1.4,
        "hold_seconds": 0.6,
        "retreat_delta_p": 0.16,
        "retreat_seconds": 1.0,
        "recommit_seconds": 2.4,
    },
    "HLC_GEN_V2_OPTION_C": {
        "advance_target_p": 0.42,
        "advance_seconds": 1.6,
        "hold_seconds": 0.7,
        "retreat_delta_p": 0.18,
        "retreat_seconds": 1.2,
        "recommit_seconds": 2.6,
    },
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Phase B1 artifact: {path}")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def apply_master_seed(master_seed: int = MASTER_SEED, torch_module: Any | None = None) -> Dict[str, str]:
    """Apply the versioned seed to available in-process random sources."""
    random.seed(master_seed)
    np.random.seed(master_seed % (2**32))
    status = {
        "python_random": "SEEDED",
        "numpy": "SEEDED",
        "torch": "DETERMINISTIC_NO_SEED_CONSUMPTION",
    }
    if torch_module is not None:
        torch_module.manual_seed(master_seed)
        if getattr(torch_module, "cuda", None) is not None and torch_module.cuda.is_available():
            torch_module.cuda.manual_seed_all(master_seed)
        if hasattr(torch_module, "use_deterministic_algorithms"):
            torch_module.use_deterministic_algorithms(True)
        status["torch"] = "SEEDED_AND_DETERMINISTIC_ALGORITHMS_REQUESTED"
    return status


def seeded_rank(selector_version: str, family: str, scenario_token: str, log_id: str) -> str:
    payload = f"{selector_version}|{MASTER_SEED}|{family}|{scenario_token}|{log_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _blend(start: float, end: float, elapsed: np.ndarray, duration: float) -> np.ndarray:
    return start + (end - start) * quintic_blend(elapsed / duration)


def hlc_progress(time_s: np.ndarray, option: Mapping[str, float] | None) -> np.ndarray:
    p = np.zeros_like(time_s)
    diverge_s = 1.1
    if option is None:
        active = time_s >= diverge_s
        p[active] = _blend(0.0, 1.0, time_s[active] - diverge_s, 2.0)
        p[time_s >= diverge_s + 2.0] = 1.0
        return p
    advance_target = float(option["advance_target_p"])
    retreat_target = advance_target - float(option["retreat_delta_p"])
    advance_seconds = float(option["advance_seconds"])
    hold_seconds = float(option["hold_seconds"])
    retreat_seconds = float(option["retreat_seconds"])
    recommit_seconds = float(option["recommit_seconds"])
    boundaries = (
        diverge_s,
        diverge_s + advance_seconds,
        diverge_s + advance_seconds + hold_seconds,
        diverge_s + advance_seconds + hold_seconds + retreat_seconds,
        diverge_s + advance_seconds + hold_seconds + retreat_seconds + recommit_seconds,
    )
    active = (time_s >= boundaries[0]) & (time_s < boundaries[1])
    p[active] = _blend(0.0, advance_target, time_s[active] - boundaries[0], advance_seconds)
    p[(time_s >= boundaries[1]) & (time_s < boundaries[2])] = advance_target
    active = (time_s >= boundaries[2]) & (time_s < boundaries[3])
    p[active] = _blend(advance_target, retreat_target, time_s[active] - boundaries[2], retreat_seconds)
    active = (time_s >= boundaries[3]) & (time_s < boundaries[4])
    p[active] = _blend(retreat_target, 1.0, time_s[active] - boundaries[3], recommit_seconds)
    p[time_s >= boundaries[4]] = 1.0
    return p


def _trajectory(time_s: np.ndarray, progress: np.ndarray, speed_mps: float, lane_width_m: float) -> Dict[str, Any]:
    xy = np.column_stack((speed_mps * time_s, lane_width_m * progress))
    states = derive_trajectory_states(xy, time_s, wheel_base_m=3.0)
    descriptors = trajectory_descriptors(time_s, xy, states["speed"])
    lateral_velocity = np.gradient(xy[:, 1], time_s, edge_order=2)
    return {
        "xy": xy,
        "states": states,
        "descriptors": descriptors,
        "terminal": {
            "target_center_offset_m": float(abs(xy[-1, 1] - lane_width_m)),
            "heading_error_rad": float(abs(states["heading"][-1])),
            "lateral_velocity_mps": float(abs(lateral_velocity[-1])),
            "route_progress_m": float(xy[-1, 0] - xy[0, 0]),
        },
    }


def analyze_hlc_options() -> Dict[str, Any]:
    with RAW_EVIDENCE.open("r", encoding="utf-8") as handle:
        evidence = json.load(handle)
    source = next(row for row in evidence["sources"] if row["source_id"] == "r_hlc_stage7l_dose0_raw")
    speeds = [
        float(source["metrics"]["speed_mps"]["q01"]),
        8.0,
        float(source["metrics"]["speed_mps"]["q95"]),
        float(source["metrics"]["speed_mps"]["q99"]),
    ]
    lane_widths = [2.7, 3.2, 4.2]
    options: Dict[str, Any] = {}
    for option_id, parameters in HLC_GEN_V2_OPTIONS.items():
        maneuver_duration = sum(
            float(parameters[key])
            for key in ("advance_seconds", "hold_seconds", "retreat_seconds", "recommit_seconds")
        )
        horizon_s = float(np.ceil((1.1 + maneuver_duration + 0.5) * 10.0) / 10.0)
        time_s = np.arange(0.0, horizon_s + 0.05, 0.1, dtype=np.float64)
        baseline_p = hlc_progress(time_s, None)
        treatment_p = hlc_progress(time_s, parameters)
        baseline_mechanism = calculate_hlc_option_b(time_s, baseline_p, np.full_like(time_s, 8.0))
        treatment_mechanism = calculate_hlc_option_b(time_s, treatment_p, np.full_like(time_s, 8.0))
        mechanism_pair = qualify_hlc_pair(baseline_mechanism, treatment_mechanism)
        cells: List[Dict[str, Any]] = []
        for speed_mps in speeds:
            for lane_width_m in lane_widths:
                baseline = _trajectory(time_s, baseline_p, speed_mps, lane_width_m)
                treatment = _trajectory(time_s, treatment_p, speed_mps, lane_width_m)
                primary_delta = {
                    feature: round(
                        abs(treatment["descriptors"][feature] - baseline["descriptors"][feature]), 6
                    )
                    for feature in PRIMARY_HLC_CALIPERS
                }
                primary_pass = all(
                    primary_delta[feature] <= PRIMARY_HLC_CALIPERS[feature] + 1e-12
                    for feature in PRIMARY_HLC_CALIPERS
                )
                safety_values = {
                    "max_abs_lateral_accel_mps2": round(float(np.max(np.abs(treatment["states"]["lateral_accel"]))), 6),
                    "max_abs_yaw_rate_radps": round(float(np.max(np.abs(treatment["states"]["yaw_rate"]))), 6),
                    "max_abs_curvature_inv_m": round(float(np.max(np.abs(treatment["states"]["curvature"]))), 6),
                }
                safety_pass = all(safety_values[key] <= limit + 1e-12 for key, limit in ENGINEERING_LIMITS.items())
                heading_delta = round(
                    abs(
                        treatment["descriptors"]["heading_change_abs_total"]
                        - baseline["descriptors"]["heading_change_abs_total"]
                    ),
                    6,
                )
                cells.append(
                    {
                        "speed_mps": speed_mps,
                        "lane_width_m": lane_width_m,
                        "new_primary_f_match_pass": primary_pass,
                        "new_primary_absolute_delta": primary_delta,
                        "secondary_heading_change_abs_total_delta_rad": heading_delta,
                        "secondary_heading_original_caliper_exceeded": heading_delta > SECONDARY_HEADING_CALIPER,
                        "engineering_safety_pass": safety_pass,
                        "engineering_values": safety_values,
                        "baseline_terminal": baseline["terminal"],
                        "treatment_terminal": treatment["terminal"],
                        "route_progress_pair_delta_m": round(abs(treatment["terminal"]["route_progress_m"] - baseline["terminal"]["route_progress_m"]), 6),
                    }
                )
        worst = {
            key: max(cell["engineering_values"][key] for cell in cells)
            for key in ENGINEERING_LIMITS
        }
        options[option_id] = {
            "status": "PROPOSED_NOT_FROZEN",
            "baseline_morphology": "DECISIVE_MONOTONIC_LANE_CHANGE",
            "treatment_morphology": "HESITANT_RETREAT_RECOMMIT",
            "interpolation": "QUINTIC_C2_PHASE_JOINS",
            "parameters": parameters,
            "retreat_target_p": round(parameters["advance_target_p"] - parameters["retreat_delta_p"], 6),
            "maneuver_duration_after_divergence_s": round(maneuver_duration, 6),
            "synthetic_horizon_s": horizon_s,
            "mechanism": {
                "baseline": baseline_mechanism,
                "treatment": treatment_mechanism,
                "pair": mechanism_pair,
            },
            "new_primary_f_match_all_cells_pass": all(cell["new_primary_f_match_pass"] for cell in cells),
            "secondary_heading_audit_only": {
                "original_caliper_rad": SECONDARY_HEADING_CALIPER,
                "delta_range_rad": [
                    min(cell["secondary_heading_change_abs_total_delta_rad"] for cell in cells),
                    max(cell["secondary_heading_change_abs_total_delta_rad"] for cell in cells),
                ],
                "primary_qualification_role": False,
            },
            "endpoint_validity_all_exact_on_parallel_lane_fixture": all(
                cell["treatment_terminal"]["target_center_offset_m"] < 1e-9
                and cell["treatment_terminal"]["heading_error_rad"] < 1e-9
                and cell["treatment_terminal"]["lateral_velocity_mps"] < 1e-9
                and cell["route_progress_pair_delta_m"] < 1e-9
                for cell in cells
            ),
            "feasible_speed_lane_width_envelope": {
                "speed_mps": [min(speeds), max(speeds)],
                "lane_width_m": [min(lane_widths), max(lane_widths)],
                "grid_cell_count": len(cells),
                "mechanism_primary_fmatch_engineering_pass_count": sum(
                    cell["new_primary_f_match_pass"] and cell["engineering_safety_pass"]
                    for cell in cells
                ),
            },
            "engineering_worst_case": worst,
            "engineering_margin_to_existing_limit": {
                key: round(ENGINEERING_LIMITS[key] - worst[key], 6) for key in ENGINEERING_LIMITS
            },
            "cells": cells,
        }
        if not mechanism_pair["pass"] or not options[option_id]["new_primary_f_match_all_cells_pass"]:
            raise RuntimeError(f"{option_id} failed its frozen-mechanism or amended-primary synthetic precheck")
    return {
        "schema_version": "r1_hlc_generator_v2_proposals_v0.1",
        "status": "SYNTHETIC_DESIGN_COMPLETE_ALL_OPTIONS_PROPOSED_NOT_FROZEN",
        "source_scope": "ANALYTICAL_SYNTHETIC_AND_TREATMENT_INDEPENDENT_RAW_SCALE_ONLY",
        "baseline_constraint": "DECISIVE_MONOTONIC_LANE_CHANGE_NO_OSCILLATORY_HEADING_MATCHING",
        "frozen_mechanism_contract_sha256": sha256_file(HLC_CONTRACT),
        "raw_scale_evidence_sha256": sha256_file(RAW_EVIDENCE),
        "prospective_primary_f_match": list(PRIMARY_HLC_CALIPERS),
        "primary_calipers_unchanged_from_r0_values": PRIMARY_HLC_CALIPERS,
        "secondary_heading_change_abs_total": {
            "role": "SECONDARY_MECHANISM_PROXIMAL_AUDIT",
            "descriptive_caliper_rad": SECONDARY_HEADING_CALIPER,
        },
        "engineering_limits": ENGINEERING_LIMITS,
        "options": options,
        "forbidden_inputs_not_opened": ["old smoke outcomes", "representation", "BDD", "probe", "checkpoint", "RBR"],
    }


def _schema_fingerprint(connection: sqlite3.Connection) -> str:
    rows = connection.execute(
        "SELECT type, name, tbl_name, COALESCE(sql, '') FROM sqlite_master "
        "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
    ).fetchall()
    return canonical_sha256(rows)


def _token_hash(tokens: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for token in sorted(set(tokens)):
        digest.update(token.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _map_inventory(map_root: Path) -> Tuple[List[Dict[str, Any]], str, set[str]]:
    rows: List[Dict[str, Any]] = []
    map_versions: set[str] = set()
    for path in sorted(item for item in map_root.rglob("*") if item.is_file() and ".maplocks" not in item.parts):
        relative = path.relative_to(map_root)
        stat = path.stat()
        rows.append(
            {
                "relative_path": str(relative),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256_file(path),
            }
        )
        if path.name == "map.gpkg" and len(relative.parts) >= 3:
            map_versions.add(relative.parts[0])
    return rows, canonical_sha256(rows), map_versions


def audit_databases(db_cache_root: Path, map_root: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    with BLACKLIST.open("r", encoding="utf-8") as handle:
        blacklist = json.load(handle)
    black_tokens = {str(row["scenario_token"]).lower() for row in blacklist["entries"]}
    black_logs = {str(row["log_id"]) for row in blacklist["entries"]}
    map_rows, map_root_fingerprint, available_map_versions = _map_inventory(map_root)
    inventory: List[Dict[str, Any]] = []
    all_tokens: set[str] = set()
    eligible_tokens: set[str] = set()
    all_logs: set[str] = set()
    eligible_logs: set[str] = set()
    excluded_by_token: set[str] = set()
    excluded_by_log: set[str] = set()
    identity_collisions: List[str] = []
    token_to_log: Dict[str, str] = {}
    partitions = ("mini", "train_pittsburgh")
    for partition in partitions:
        partition_root = db_cache_root / partition
        for db_path in sorted(partition_root.glob("*.db")):
            stat = db_path.stat()
            row: Dict[str, Any] = {
                "source_partition": partition,
                "db_path": str(db_path.resolve()),
                "db_file": db_path.name,
                "size_bytes": stat.st_size,
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "sqlite_readable": False,
                "schema_sha256": "",
                "db_fingerprint_sha256": "",
                "scenario_tag_count": 0,
                "unique_scenario_token_count": 0,
                "token_set_sha256": "",
                "log_count": 0,
                "log_ids": "",
                "locations": "",
                "map_versions": "",
                "map_compatible": False,
                "status": "NOT_READY",
                "error": "",
            }
            try:
                uri = f"file:{db_path.resolve()}?mode=ro"
                connection = sqlite3.connect(uri, uri=True)
                connection.execute("PRAGMA query_only=ON")
                schema_sha = _schema_fingerprint(connection)
                tag_count = int(connection.execute("SELECT count(*) FROM scenario_tag").fetchone()[0])
                tokens = [
                    str(value[0]).lower()
                    for value in connection.execute(
                        "SELECT DISTINCT lower(hex(lidar_pc_token)) FROM scenario_tag ORDER BY 1"
                    )
                ]
                log_rows = connection.execute(
                    "SELECT logfile, location, map_version FROM log ORDER BY logfile"
                ).fetchall()
                connection.close()
                logs = {str(value[0]) for value in log_rows}
                locations = {str(value[1]) for value in log_rows}
                versions = {str(value[2]) for value in log_rows}
                compatible = bool(versions) and versions.issubset(available_map_versions)
                token_sha = _token_hash(tokens)
                db_fingerprint = canonical_sha256(
                    {
                        "relative_path": str(db_path.relative_to(db_cache_root)),
                        "size_bytes": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "schema_sha256": schema_sha,
                        "token_set_sha256": token_sha,
                    }
                )
                row.update(
                    {
                        "sqlite_readable": True,
                        "schema_sha256": schema_sha,
                        "db_fingerprint_sha256": db_fingerprint,
                        "scenario_tag_count": tag_count,
                        "unique_scenario_token_count": len(tokens),
                        "token_set_sha256": token_sha,
                        "log_count": len(logs),
                        "log_ids": ";".join(sorted(logs)),
                        "locations": ";".join(sorted(locations)),
                        "map_versions": ";".join(sorted(versions)),
                        "map_compatible": compatible,
                        "status": "READY" if compatible and stat.st_size > 0 else "NOT_READY",
                    }
                )
                all_logs.update(logs)
                for token in tokens:
                    all_tokens.add(token)
                    log_id = next(iter(logs)) if len(logs) == 1 else row["log_ids"]
                    prior = token_to_log.setdefault(token, log_id)
                    if prior != log_id:
                        identity_collisions.append(token)
                    if token in black_tokens:
                        excluded_by_token.add(token)
                    elif logs.intersection(black_logs):
                        excluded_by_log.add(token)
                    else:
                        eligible_tokens.add(token)
                        eligible_logs.update(logs)
            except (sqlite3.Error, OSError, ValueError) as error:
                row["error"] = f"{type(error).__name__}: {error}"
            inventory.append(row)
    source_root_fingerprint = canonical_sha256(
        [
            {
                "partition": row["source_partition"],
                "db_file": row["db_file"],
                "db_fingerprint_sha256": row["db_fingerprint_sha256"],
            }
            for row in inventory
        ]
    )
    all_ready = bool(inventory) and all(row["status"] == "READY" for row in inventory)
    future_r4 = json.loads(FUTURE_R4_FREEZE.read_text(encoding="utf-8"))
    universe = {
        "schema_version": "r1_fresh_smoke_source_universe_v0.1",
        "status": "READY_FOR_OUTCOME_BLIND_SELECTION" if all_ready and eligible_tokens and not identity_collisions else "NOT_READY",
        "actual_smoke_roster_selected": False,
        "source_db": {
            "release": "LOCAL_NUPLAN_V1.1_CACHE_MINI_AND_TRAIN_PITTSBURGH",
            "cache_root": str(db_cache_root.resolve()),
            "partitions": list(partitions),
            "db_file_count": len(inventory),
            "total_size_bytes": sum(int(row["size_bytes"]) for row in inventory),
            "readable_ready_db_count": sum(row["status"] == "READY" for row in inventory),
            "source_root_fingerprint_sha256": source_root_fingerprint,
        },
        "map_binding": {
            "map_root": str(map_root.resolve()),
            "map_root_fingerprint_sha256": map_root_fingerprint,
            "map_files": map_rows,
            "available_map_versions": sorted(available_map_versions),
        },
        "unfiltered_universe": {
            "unique_scenario_token_count": len(all_tokens),
            "token_set_sha256": _token_hash(all_tokens),
            "unique_log_count": len(all_logs),
            "log_universe": sorted(all_logs),
        },
        "eligibility_exclusions": {
            "old_smoke_blacklist_path": str(BLACKLIST.relative_to(ROOT)),
            "old_smoke_blacklist_sha256": sha256_file(BLACKLIST),
            "blacklisted_scenario_count": len(black_tokens),
            "blacklisted_log_count": len(black_logs),
            "excluded_token_match_count": len(excluded_by_token),
            "excluded_log_match_count": len(excluded_by_log),
            "formal_r1_development_roster": "NOT_EXISTING_BY_DESIGN",
            "future_r4_identity_roster": future_r4["status"],
            "future_r4_current_identity_count": 0,
            "future_r4_freeze_sha256": sha256_file(FUTURE_R4_FREEZE),
            "historical_audit_identity_roster": "NOT_FOUND_NO_R0_AUDIT_HOLDOUT_ALLOCATED",
            "rule": "exclude if scenario_token OR log_id appears in any bound exclusion identity set",
        },
        "eligible_universe": {
            "unique_scenario_token_count": len(eligible_tokens),
            "token_set_sha256": _token_hash(eligible_tokens),
            "unique_log_count": len(eligible_logs),
            "log_universe": sorted(eligible_logs),
            "identity_collision_count": len(set(identity_collisions)),
        },
        "eligibility_query": {
            "sqlite": "SELECT DISTINCT lower(hex(st.lidar_pc_token)), l.logfile, l.location, l.map_version FROM scenario_tag st JOIN lidar_pc pc ON pc.token=st.lidar_pc_token JOIN scene s ON s.token=pc.scene_token JOIN log l ON l.token=s.log_token",
            "post_query_filters": [
                "nonzero readable DB",
                "map_version present under bound map root",
                "scenario_token and entire log absent from bound exclusion sets",
            ],
            "outcome_fields_used": False,
        },
        "selection_seed_contract": {
            "path": str(REPLAY_CONTRACT.relative_to(ROOT)),
            "master_seed": MASTER_SEED,
            "rank_rule": "SHA256(selector_version|MASTER_SEED|family|scenario_token|log_id)",
            "sha256": sha256_file(REPLAY_CONTRACT),
        },
        "selection_authorized": False,
        "smoke_authorized": False,
        "forbidden_inputs_not_opened": ["old smoke outcomes", "representation", "BDD", "probe", "checkpoint", "RBR"],
    }
    return inventory, universe


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare R1 Phase B1 synthetic and DB inventory artifacts.")
    parser.add_argument("--db-cache-root", type=Path, default=DEFAULT_DB_CACHE_ROOT)
    parser.add_argument("--map-root", type=Path, default=DEFAULT_MAP_ROOT)
    parser.add_argument("--out-dir", type=Path, default=R1_DIR)
    args = parser.parse_args()
    required = (args.db_cache_root / "mini", args.db_cache_root / "train_pittsburgh", args.map_root, BLACKLIST, RAW_EVIDENCE, HLC_CONTRACT, FUTURE_R4_FREEZE, REPLAY_CONTRACT)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Phase B1 required read-only inputs missing: {missing}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "hlc": args.out_dir / "r1_hlc_generator_v2_proposals_v0.1.json",
        "inventory_rows": args.out_dir / "r1_official_nuplan_db_inventory_rows_v0.1.json",
        "universe": args.out_dir / "r1_fresh_smoke_source_universe_v0.1.json",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite Phase B1 outputs: {existing}")
    hlc = analyze_hlc_options()
    inventory, universe = audit_databases(args.db_cache_root, args.map_root)
    write_new_json(paths["hlc"], hlc)
    write_new_json(
        paths["inventory_rows"],
        {
            "schema_version": "r1_official_nuplan_db_inventory_rows_v0.1",
            "status": "READ_ONLY_SQLITE_INVENTORY_COMPLETE",
            "columns": list(inventory[0]) if inventory else [],
            "rows": inventory,
        },
    )
    write_new_json(paths["universe"], universe)
    print(
        json.dumps(
            {
                "hlc_options": len(hlc["options"]),
                "db_files": len(inventory),
                "db_ready": sum(row["status"] == "READY" for row in inventory),
                "source_universe_status": universe["status"],
                "eligible_scenarios": universe["eligible_universe"]["unique_scenario_token_count"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
