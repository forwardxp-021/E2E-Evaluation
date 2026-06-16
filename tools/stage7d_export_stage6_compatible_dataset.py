#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

REQUIRED_PLANNERS = [
    "simple_planner",
    "idm_longitudinal_conservative",
    "idm_longitudinal_comfort",
    "idm_longitudinal_aggressive",
]
INPUT_CHANNELS = ["x", "y", "yaw", "speed", "velocity_y", "acceleration", "acceleration_y", "time_s"]
EGO_CHANNELS = ["x", "y", "vx", "vy", "heading", "speed", "accel", "yaw_rate"]
NEIGHBOR_CHANNELS = [
    "rel_x", "rel_y", "rel_vx", "rel_vy", "distance", "bearing", "heading_rel", "speed", "valid"
]
FEATURES = [
    "mean_speed", "max_speed", "speed_std", "path_length", "final_displacement",
    "mean_accel", "mean_abs_accel", "max_accel", "min_accel", "rms_accel",
    "jerk_mean", "jerk_rms", "jerk_abs_mean", "jerk_abs_p95", "low_speed_ratio",
    "yaw_rate_rms", "yaw_rate_abs_mean", "heading_change_total",
    "min_neighbor_distance", "mean_neighbor_distance", "min_ttc_proxy", "mean_thw_proxy",
    "min_thw_proxy", "following_valid_ratio", "cutin_proxy_score", "yield_conflict_proxy_score",
]
META_COLUMNS = [
    "global_row", "scenario_index", "planner_id", "planner_name", "log_name", "scenario_token",
    "scenario_type", "source_stage", "uses_official_nuplan_simulation", "pseudo_rollout",
    "style_scope", "policy_style", "nuplan_planner_config", "supported_behavior_tasks",
    "unsupported_behavior_tasks",
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def require_inputs(sim_dir: Path) -> Dict[str, Path]:
    names = [
        "simulated_ego_seq.npy", "simulated_ego_seq_mask.npy", "simulated_planner_metadata.csv",
        "scenario_planner_index.csv", "simulation_schema.json", "warnings.json",
    ]
    paths = {name: sim_dir / name for name in names}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required Stage 7C official output files: " + ", ".join(missing))
    return paths


def validate_official(schema: Dict[str, Any], warnings_in: Any) -> None:
    if schema.get("pseudo_rollout") is not False:
        raise ValueError("Stage 7D export is fatal: simulation_schema.json pseudo_rollout must be false.")
    if schema.get("uses_official_nuplan_simulation") is not True:
        raise ValueError("Stage 7D export is fatal: official nuPlan simulation output is not confirmed.")
    if isinstance(warnings_in, dict) and bool(warnings_in.get("pseudo_rollout", False)):
        raise ValueError("Stage 7D export is fatal: warnings.json indicates pseudo_rollout=true.")


def planner_axis(metadata_path: Path, index_path: Path, p_count: int) -> List[str]:
    seen: Dict[int, str] = {}
    for path in (metadata_path, index_path):
        for row in read_csv_rows(path):
            if row.get("planner_id", "").strip() and row.get("planner_name"):
                seen[int(row["planner_id"])] = row["planner_name"]
    planners = [seen[i] for i in sorted(seen)]
    if len(planners) != p_count:
        raise ValueError(f"Planner axis length mismatch: tensor P={p_count}, metadata planners={planners}")
    missing = [p for p in REQUIRED_PLANNERS if p not in planners]
    if missing:
        raise ValueError(f"Planner axis missing mandatory planners: {missing}; observed={planners}")
    return planners


def compute_yaw_rate(yaw: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    out = np.zeros_like(yaw, dtype=np.float32)
    if yaw.size < 2:
        return out
    dt = np.diff(time_s.astype(float))
    good = dt[np.isfinite(dt) & (dt > 1e-6)]
    fill = float(np.median(good)) if good.size else 0.1
    dt = np.where(np.isfinite(dt) & (dt > 1e-6), dt, fill)
    rate = np.diff(np.unwrap(yaw.astype(float))) / dt
    out[1:] = rate.astype(np.float32)
    out[0] = out[1]
    return out


def convert_ego(seq4: np.ndarray, mask: np.ndarray) -> np.ndarray:
    n, p, t, _ = seq4.shape
    ego = np.zeros((n * p, t, len(EGO_CHANNELS)), dtype=np.float32)
    row = 0
    for i in range(n):
        for j in range(p):
            s = np.asarray(seq4[i, j], dtype=np.float32)
            valid = mask[i, j].astype(bool)
            x, y, yaw, speed, _vy_in, accel, _ay, time_s = [s[:, k] for k in range(8)]
            ego[row, :, 0] = x
            ego[row, :, 1] = y
            ego[row, :, 2] = speed * np.cos(yaw)
            ego[row, :, 3] = speed * np.sin(yaw)
            ego[row, :, 4] = yaw
            ego[row, :, 5] = speed
            ego[row, :, 6] = accel
            ego[row, :, 7] = compute_yaw_rate(yaw, time_s)
            ego[row, ~valid, :] = 0.0
            row += 1
    return ego


def load_neighbor_source(sim_dir: Path, expected_rows: int, timesteps: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Load already extracted Stage 7D neighbor tensors.

    The exporter refuses to fabricate neighbor_seq. Upstream extraction may come from
    nuPlan msgpack/scenario DB, but it must materialize one of these audited files.
    """
    candidates = [
        (sim_dir / "stage7d_neighbor_seq.npy", sim_dir / "stage7d_neighbor_slot_ids.npy"),
        (sim_dir / "neighbor_seq.npy", sim_dir / "neighbor_slot_ids.npy"),
        (sim_dir / "extracted_neighbor_seq.npy", sim_dir / "extracted_neighbor_slot_ids.npy"),
    ]
    for seq_path, ids_path in candidates:
        if seq_path.exists() and ids_path.exists():
            neigh = np.load(seq_path, allow_pickle=False)
            slot = np.load(ids_path, allow_pickle=True)
            if neigh.ndim != 4:
                raise ValueError(f"{seq_path} must have shape [rows,K,T,D], got {list(neigh.shape)}")
            if neigh.shape[0] != expected_rows or neigh.shape[2] != timesteps:
                raise ValueError(f"{seq_path} row/T mismatch: expected [{expected_rows},K,{timesteps},D], got {list(neigh.shape)}")
            if slot.shape[:2] != neigh.shape[:2]:
                raise ValueError(f"{ids_path} shape must start with {list(neigh.shape[:2])}, got {list(slot.shape)}")
            return neigh.astype(np.float32), slot, seq_path.name
    msgpacks = list((sim_dir / "official_nuplan_runs").rglob("*.msgpack.xz")) if (sim_dir / "official_nuplan_runs").exists() else []
    raise FileNotFoundError(
        "neighbor_seq.npy and neighbor_slot_ids.npy are mandatory for Stage 7D PASS. "
        "No extracted neighbor source was found in sim_dir. Expected one of: "
        "stage7d_neighbor_seq.npy + stage7d_neighbor_slot_ids.npy, neighbor_seq.npy + neighbor_slot_ids.npy, "
        "or extracted_neighbor_seq.npy + extracted_neighbor_slot_ids.npy. "
        f"Discovered official msgpack files={len(msgpacks)}, but this exporter does not fabricate neighbors; "
        "extract/reload surrounding agents from nuPlan first and rerun."
    )


def safe_hash_id(text: Any) -> int:
    h = hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:15]
    return int(h, 16) % (2**31 - 1)


def normalize_slot_ids(slot: np.ndarray) -> np.ndarray:
    if np.issubdtype(slot.dtype, np.number):
        return slot.astype(np.int64)
    out = np.zeros(slot.shape, dtype=np.int64)
    it = np.nditer(slot, flags=["multi_index", "refs_ok"], op_flags=["readonly"])
    for x in it:
        out[it.multi_index] = safe_hash_id(x.item())
    return out


def interaction_features(ego: np.ndarray, neigh: np.ndarray) -> np.ndarray:
    rows = []
    for r in range(ego.shape[0]):
        e = ego[r]
        speed = e[:, 5]
        accel = e[:, 6]
        yaw_rate = e[:, 7]
        xy = e[:, :2]
        step = np.linalg.norm(np.diff(xy, axis=0), axis=1) if xy.shape[0] > 1 else np.array([0.0])
        jerk = np.diff(accel) / 0.1 if accel.size > 1 else np.array([0.0])
        valid_n = neigh[r, :, :, -1] > 0.5 if neigh.shape[-1] >= 1 else np.ones(neigh[r].shape[:2], dtype=bool)
        dist = neigh[r, :, :, 4] if neigh.shape[-1] > 4 else np.linalg.norm(neigh[r, :, :, :2], axis=-1)
        dist_valid = dist[valid_n & np.isfinite(dist) & (dist > 0)]
        rel_x = neigh[r, :, :, 0]
        rel_vx = neigh[r, :, :, 2] if neigh.shape[-1] > 2 else np.zeros_like(rel_x)
        following = valid_n & (rel_x > 0) & (np.abs(neigh[r, :, :, 1]) < 3.5)
        closing = np.maximum(-rel_vx[following], 1e-3)
        ttc = rel_x[following] / closing if np.any(following) else np.array([], dtype=float)
        thw = rel_x[following] / np.maximum(speed[None, :].repeat(neigh.shape[1], axis=0)[following], 1e-3) if np.any(following) else np.array([], dtype=float)
        rows.append([
            float(np.mean(speed)), float(np.max(speed)), float(np.std(speed)), float(np.sum(step)),
            float(np.linalg.norm(xy[-1] - xy[0])), float(np.mean(accel)), float(np.mean(np.abs(accel))),
            float(np.max(accel)), float(np.min(accel)), float(np.sqrt(np.mean(accel * accel))),
            float(np.mean(jerk)), float(np.sqrt(np.mean(jerk * jerk))), float(np.mean(np.abs(jerk))),
            float(np.percentile(np.abs(jerk), 95)), float(np.mean(speed < 0.5)),
            float(np.sqrt(np.mean(yaw_rate * yaw_rate))), float(np.mean(np.abs(yaw_rate))),
            float(np.sum(np.abs(np.diff(np.unwrap(e[:, 4]))))) if e.shape[0] > 1 else 0.0,
            float(np.min(dist_valid)) if dist_valid.size else math.inf,
            float(np.mean(dist_valid)) if dist_valid.size else math.inf,
            float(np.min(ttc)) if ttc.size else math.inf,
            float(np.mean(thw)) if thw.size else math.inf,
            float(np.min(thw)) if thw.size else math.inf,
            float(np.mean(following)) if following.size else 0.0,
            float(np.mean((valid_n) & (np.abs(neigh[r, :, :, 1]) < 2.0) & (np.abs(rel_x) < 15.0))),
            float(np.mean((following) & (rel_x < 10.0))),
        ])
    return np.asarray(rows, dtype=np.float32)


def metadata_rows(index_rows: List[Dict[str, str]], planners: List[str], n_scenarios: int) -> List[Dict[str, Any]]:
    by_pair = {}
    for row in index_rows:
        try:
            by_pair[(int(row.get("scenario_index", row.get("scenario_id", -1))), int(row.get("planner_id", -1)))] = row
        except ValueError:
            continue
    rows = []
    g = 0
    for i in range(n_scenarios):
        for pid, planner in enumerate(planners):
            src = by_pair.get((i, pid), {})
            rows.append({
                "global_row": g, "scenario_index": i, "planner_id": pid, "planner_name": planner,
                "log_name": src.get("log_name", ""),
                "scenario_token": src.get("actual_nuplan_scenario_token", src.get("scenario_token", "")),
                "scenario_type": src.get("scenario_type", ""),
                "source_stage": "stage7c_official_nuplan_simulation",
                "uses_official_nuplan_simulation": True, "pseudo_rollout": False,
                "style_scope": "planner_policy", "policy_style": planner,
                "nuplan_planner_config": planner,
                "supported_behavior_tasks": "stage6_bdd,stage6_report_card,stage6c_task_conditioned_bdd",
                "unsupported_behavior_tasks": "none_export_is_full_stage6_compatible",
            })
            g += 1
    return rows


def validate_outputs(out_dir: Path, planners: List[str], expected_rows: int, n_scenarios: int, n_planners: int) -> None:
    shard = out_dir / "shards" / "shard_000"
    req = [shard / n for n in ["ego_seq.npy", "neighbor_seq.npy", "neighbor_slot_ids.npy", "interaction_feat_style.npy", "metadata.csv"]]
    req += [out_dir / "feature_schema.json", out_dir / "shard_manifest.json"]
    req += [out_dir / "planner_policy_indices" / f"{p}.npy" for p in planners]
    missing = [str(p) for p in req if not p.exists()]
    if missing:
        raise FileNotFoundError("Stage 7D export validation failed; missing mandatory outputs: " + ", ".join(missing))
    ego = np.load(shard / "ego_seq.npy", mmap_mode="r")
    neigh = np.load(shard / "neighbor_seq.npy", mmap_mode="r")
    feat = np.load(shard / "interaction_feat_style.npy", mmap_mode="r")
    meta_count = len(read_csv_rows(shard / "metadata.csv"))
    if expected_rows != n_scenarios * n_planners:
        raise ValueError(
            f"Stage 7D row semantics violation: expected_rows={expected_rows} must equal "
            f"num_scenarios * num_planners = {n_scenarios} * {n_planners}."
        )
    if not (ego.shape[0] == neigh.shape[0] == feat.shape[0] == meta_count == expected_rows):
        raise ValueError(
            f"Row alignment failed: ego={ego.shape[0]} neighbor={neigh.shape[0]} "
            f"features={feat.shape[0]} metadata={meta_count} expected={expected_rows}. "
            "Stage 7D must not expand background/neighbor agents into additional ego rows."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Export official Stage 7C planner rollouts as a full Stage 6-compatible sharded dataset.")
    ap.add_argument("--sim_dir", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dir exists: {args.output_dir}. Use --overwrite.")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    paths = require_inputs(args.sim_dir)
    schema = read_json(paths["simulation_schema.json"])
    warnings_in = read_json(paths["warnings.json"])
    validate_official(schema, warnings_in)
    seq = np.load(paths["simulated_ego_seq.npy"], mmap_mode="r")
    mask = np.load(paths["simulated_ego_seq_mask.npy"], mmap_mode="r")
    if seq.ndim != 4 or seq.shape[-1] != 8 or mask.shape != seq.shape[:3]:
        raise ValueError(f"Invalid Stage 7C tensor shapes: seq={list(seq.shape)} mask={list(mask.shape)}")
    planners = planner_axis(paths["simulated_planner_metadata.csv"], paths["scenario_planner_index.csv"], seq.shape[1])
    n_scenarios = int(seq.shape[0])
    n_planners = int(seq.shape[1])
    expected_rows = int(n_scenarios * n_planners)
    ego = convert_ego(np.asarray(seq), np.asarray(mask))
    neighbor, slot_raw, neighbor_source = load_neighbor_source(args.sim_dir, expected_rows, seq.shape[2])
    slot_ids = normalize_slot_ids(slot_raw)
    feat = interaction_features(ego, neighbor)

    shard = args.output_dir / "shards" / "shard_000"
    idx_dir = args.output_dir / "planner_policy_indices"
    shard.mkdir(parents=True)
    idx_dir.mkdir()
    np.save(shard / "ego_seq.npy", ego)
    np.save(shard / "neighbor_seq.npy", neighbor)
    np.save(shard / "neighbor_slot_ids.npy", slot_ids)
    np.save(shard / "interaction_feat_style.npy", feat)
    rows = metadata_rows(read_csv_rows(paths["scenario_planner_index.csv"]), planners, seq.shape[0])
    write_csv(shard / "metadata.csv", rows, META_COLUMNS)
    for pid, planner in enumerate(planners):
        np.save(idx_dir / f"{planner}.npy", np.asarray([r for r in range(expected_rows) if r % len(planners) == pid], dtype=np.int64))
    write_json(args.output_dir / "shard_manifest.json", {"shards": [{"shard_path": "shards/shard_000"}]})
    write_json(args.output_dir / "feature_schema.json", {"feature_names": FEATURES, "features": [{"index": i, "name": n} for i, n in enumerate(FEATURES)], "ego_channels": EGO_CHANNELS, "neighbor_channels": NEIGHBOR_CHANNELS})
    export_schema = {
        "stage": "7D",
        "purpose": "full_stage6_compatible_dataset_export",
        "row_semantics": "scenario_planner_controlled_ego_rollout",
        "ego_definition": "nuPlan planner-controlled ego vehicle only",
        "neighbor_definition": "background road participants used only as context",
        "multi_agent_ego_expansion": False,
        "num_scenarios": n_scenarios,
        "num_planners": n_planners,
        "total_rows_expected": expected_rows,
        "rows": expected_rows,
        "input_channels": INPUT_CHANNELS,
        "ego_channels": EGO_CHANNELS,
        "neighbor_channels": NEIGHBOR_CHANNELS,
        "planner_axis": planners,
        "neighbor_source": neighbor_source,
        "pseudo_rollout": False,
        "uses_official_nuplan_simulation": True,
    }
    write_json(args.output_dir / "stage7d_export_schema.json", export_schema)
    write_json(args.output_dir / "warnings.json", {
        "warnings": [],
        "fatal_if_missing": ["neighbor_seq.npy", "neighbor_slot_ids.npy"],
        "validation": {
            "pass": True,
            "total_rows": expected_rows,
            "total_rows_expected": expected_rows,
            "num_scenarios": n_scenarios,
            "num_planners": n_planners,
            "row_semantics": "scenario_planner_controlled_ego_rollout",
            "no_multi_agent_ego_expansion": True,
            "neighbor_agents_used_as_context_only": True,
        },
        "input_warnings": warnings_in,
    })
    validate_outputs(args.output_dir, planners, expected_rows, n_scenarios, n_planners)
    report = ["# Stage 7D Full Stage 6-Compatible Export Report", "", "- validation.pass: **PASS**", "- Stage 7D exports data only; it does not run nuPlan simulation and does not compute final BDD.", f"- rows: `{expected_rows}` (= `{n_scenarios} scenarios × {n_planners} planners`)", "- row semantics: `one row = one scenario × one planner-controlled nuPlan ego rollout`", "- multi-agent ego expansion: `false`", f"- ego_seq.npy shape: `{list(ego.shape)}`", f"- neighbor_seq.npy shape: `{list(neighbor.shape)}`", f"- interaction_feat_style.npy shape: `{list(feat.shape)}`", "", "## Planner Axis", *[f"- {i}: `{p}`" for i, p in enumerate(planners)], "", "## Mandatory Outputs", "- `ego_seq.npy`", "- `neighbor_seq.npy`", "- `neighbor_slot_ids.npy`", "- `interaction_feat_style.npy`", "- `metadata.csv`", "- `feature_schema.json`", "- `shard_manifest.json`", "- `planner_policy_indices/*.npy`"]
    (args.output_dir / "export_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Stage 7D full Stage 6-compatible export PASS: {args.output_dir}")


if __name__ == "__main__":
    main()
