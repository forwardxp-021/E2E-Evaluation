#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import hashlib
import importlib
import json
import lzma
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

NEIGHBOR_CHANNELS = ["rel_x", "rel_y", "rel_vx", "rel_vy", "distance", "bearing", "heading_rel", "speed", "valid"]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def require_inputs(sim_dir: Path) -> Dict[str, Path]:
    names = ["simulated_ego_seq.npy", "simulated_ego_seq_mask.npy", "scenario_planner_index.csv", "simulated_planner_metadata.csv", "simulation_schema.json", "warnings.json"]
    out = {name: sim_dir / name for name in names}
    missing = [str(p) for p in out.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required Stage 7C official output files: " + ", ".join(missing))
    if not (sim_dir / "official_nuplan_runs").exists():
        raise FileNotFoundError(f"Missing official_nuplan_runs directory: {sim_dir / 'official_nuplan_runs'}")
    return out


def validate_official(schema: Dict[str, Any], warnings_in: Any) -> None:
    if schema.get("pseudo_rollout") is not False:
        raise ValueError("Stage 7D neighbor extraction is fatal: simulation_schema.json pseudo_rollout must be false.")
    if schema.get("uses_official_nuplan_simulation") is not True:
        raise ValueError("Stage 7D neighbor extraction is fatal: official nuPlan simulation output is not confirmed.")
    if isinstance(warnings_in, dict):
        validation = warnings_in.get("validation", {})
        if validation.get("pseudo_rollout") is True or warnings_in.get("pseudo_rollout") is True:
            raise ValueError("Stage 7D neighbor extraction is fatal: warnings.json indicates pseudo_rollout=true.")


def obj_value(obj: Any, names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def obj_path(obj: Any, paths: Iterable[str], default: Any = None) -> Any:
    for path in paths:
        cur = obj
        ok = True
        for part in path.split("."):
            cur = obj_value(cur, [part], None)
            if cur is None:
                ok = False
                break
        if ok:
            return cur
    return default


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def hash_id(text: Any) -> int:
    h = hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:15]
    return int(h, 16) % (2**31 - 1)


def load_sim_log(path: Path) -> Any:
    try:
        mod = importlib.import_module("nuplan.planning.simulation.simulation_log")
        cls = getattr(mod, "SimulationLog")
        for method in ["load_data", "deserialize", "load"]:
            if hasattr(cls, method):
                for arg in (path, str(path)):
                    try:
                        return getattr(cls, method)(arg)
                    except Exception:
                        pass
    except Exception:
        pass
    msgpack = importlib.import_module("msgpack")
    return msgpack.unpackb(lzma.open(path, "rb").read(), raw=False, strict_map_key=False)


def history_samples(sim_log: Any) -> List[Any]:
    hist = obj_value(sim_log, ["simulation_history"], sim_log)
    data = obj_value(hist, ["data"], hist if isinstance(hist, list) else [])
    if not isinstance(data, list):
        raise ValueError("msgpack simulation history has no list-like simulation_history.data")
    return data


def tracked_objects(sample: Any) -> List[Any]:
    candidates = [
        "observation.tracked_objects.tracked_objects", "observation.tracked_objects", "observation.detections.tracked_objects",
        "observations.tracked_objects.tracked_objects", "detections.tracked_objects.tracked_objects", "tracked_objects.tracked_objects",
        "tracked_objects", "observation",
    ]
    for path in candidates:
        val = obj_path(sample, [path], None)
        if isinstance(val, list):
            return val
        inner = obj_value(val, ["tracked_objects"], None)
        if isinstance(inner, list):
            return inner
    return []


def center_of(obj: Any) -> Tuple[float, float, float]:
    center = obj_path(obj, ["center", "box.center", "oriented_box.center"], obj)
    x = finite_float(obj_path(center, ["x", "point.x", "rear_axle.x"], obj_path(obj, ["x"], math.nan)))
    y = finite_float(obj_path(center, ["y", "point.y", "rear_axle.y"], obj_path(obj, ["y"], math.nan)))
    heading = finite_float(obj_path(center, ["heading", "yaw"], obj_path(obj, ["heading", "yaw"], 0.0)), 0.0)
    return x, y, heading


def velocity_of(obj: Any) -> Tuple[float, float, float]:
    vel = obj_path(obj, ["velocity", "metadata.velocity", "dynamic_car_state.rear_axle_velocity_2d"], None)
    vx = finite_float(obj_path(vel, ["x", "vx"], obj_path(obj, ["velocity_x", "vx"], 0.0)), 0.0)
    vy = finite_float(obj_path(vel, ["y", "vy"], obj_path(obj, ["velocity_y", "vy"], 0.0)), 0.0)
    return vx, vy, math.hypot(vx, vy)


def token_of(obj: Any, fallback: str) -> str:
    tok = obj_path(obj, ["track_token", "token", "metadata.track_token", "metadata.token", "tracked_object_type"], "")
    return str(tok or fallback)


def parse_neighbor_world_tracks(path: Path, timesteps: int) -> Tuple[Dict[str, Dict[int, Tuple[float, float, float, float, float, float]]], int]:
    samples = history_samples(load_sim_log(path))
    tracks: Dict[str, Dict[int, Tuple[float, float, float, float, float, float]]] = {}
    for t, sample in enumerate(samples[:timesteps]):
        for j, obj in enumerate(tracked_objects(sample)):
            x, y, heading = center_of(obj)
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            vx, vy, speed = velocity_of(obj)
            tok = token_of(obj, f"{path.name}:{j}")
            tracks.setdefault(tok, {})[t] = (x, y, vx, vy, heading, speed)
    return tracks, len(samples)


def find_msgpack(sim_dir: Path, row: Dict[str, str]) -> Path | None:
    direct = row.get("actual_msgpack_path") or row.get("msgpack_path")
    if direct and Path(direct).exists():
        return Path(direct)
    scenario_index = str(row.get("scenario_index", ""))
    planner_name = str(row.get("planner_name", ""))
    base = sim_dir / "official_nuplan_runs" / f"scenario_{scenario_index}" / planner_name
    roots = [base, sim_dir / "official_nuplan_runs"] if base.exists() else [sim_dir / "official_nuplan_runs"]
    msgpacks: List[Path] = []
    for root in roots:
        msgpacks.extend(sorted(root.rglob("*.msgpack.xz")))
        if msgpacks:
            return msgpacks[0]
    return None


def ego_world(seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y, yaw, speed = seq[:, 0], seq[:, 1], seq[:, 2], seq[:, 3]
    return x, y, yaw, speed * np.cos(yaw), speed * np.sin(yaw)


def build_row_neighbors(ego_seq: np.ndarray, ego_mask: np.ndarray, tracks: Dict[str, Dict[int, Tuple[float, float, float, float, float, float]]], k: int) -> Tuple[np.ndarray, np.ndarray]:
    t_count = ego_seq.shape[0]
    ex, ey, eyaw, evx, evy = ego_world(ego_seq)
    scores = []
    for tok, by_t in tracks.items():
        d = []
        for t, st in by_t.items():
            if t < t_count and ego_mask[t]:
                d.append(math.hypot(st[0] - float(ex[t]), st[1] - float(ey[t])))
        if d:
            scores.append((float(np.nanmedian(d)), tok))
    chosen = [tok for _, tok in sorted(scores)[:k]]
    arr = np.zeros((k, t_count, len(NEIGHBOR_CHANNELS)), dtype=np.float32)
    ids = np.full((k,), -1, dtype=np.int64)
    for slot, tok in enumerate(chosen):
        ids[slot] = hash_id(tok)
        for t, st in tracks[tok].items():
            if t >= t_count or not ego_mask[t]:
                continue
            dx = st[0] - float(ex[t]); dy = st[1] - float(ey[t])
            c = math.cos(float(eyaw[t])); s = math.sin(float(eyaw[t]))
            rel_x = c * dx + s * dy
            rel_y = -s * dx + c * dy
            dvx = st[2] - float(evx[t]); dvy = st[3] - float(evy[t])
            rel_vx = c * dvx + s * dvy
            rel_vy = -s * dvx + c * dvy
            dist = math.hypot(rel_x, rel_y)
            bearing = math.atan2(rel_y, rel_x)
            heading_rel = math.atan2(math.sin(st[4] - float(eyaw[t])), math.cos(st[4] - float(eyaw[t])))
            arr[slot, t, :] = [rel_x, rel_y, rel_vx, rel_vy, dist, bearing, heading_rel, st[5], 1.0]
    return arr, ids


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract Stage 7D mandatory ego-centric neighbor tensors from official nuPlan simulation logs.")
    ap.add_argument("--sim_dir", type=Path, default=Path("outputs/stage7c2c2_idm_longitudinal_5logs"))
    ap.add_argument("--max_neighbors", type=int, default=16)
    ap.add_argument("--low_coverage_threshold", type=float, default=0.05)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    paths = require_inputs(args.sim_dir)
    validate_official(read_json(paths["simulation_schema.json"]), read_json(paths["warnings.json"]))
    seq = np.load(paths["simulated_ego_seq.npy"], mmap_mode="r")
    mask = np.load(paths["simulated_ego_seq_mask.npy"], mmap_mode="r")
    if seq.ndim != 4 or seq.shape[-1] != 8 or mask.shape != seq.shape[:3]:
        raise ValueError(f"Invalid Stage 7C tensor shapes: seq={list(seq.shape)} mask={list(mask.shape)}")
    n_scenarios, n_planners, timesteps, _ = seq.shape
    expected_rows = n_scenarios * n_planners
    out_seq = args.sim_dir / "stage7d_neighbor_seq.npy"
    out_ids = args.sim_dir / "stage7d_neighbor_slot_ids.npy"
    if (out_seq.exists() or out_ids.exists()) and not args.overwrite:
        raise FileExistsError(f"Output exists: {out_seq} or {out_ids}. Use --overwrite.")
    index_rows = read_csv_rows(paths["scenario_planner_index.csv"])
    by_pair = {(int(r.get("scenario_index", -1)), int(r.get("planner_id", -1))): r for r in index_rows if r.get("scenario_index", "").strip() and r.get("planner_id", "").strip()}
    neighbor = np.zeros((expected_rows, args.max_neighbors, timesteps, len(NEIGHBOR_CHANNELS)), dtype=np.float32)
    slot_ids = np.full((expected_rows, args.max_neighbors), -1, dtype=np.int64)
    warnings: List[Dict[str, Any]] = []
    parsed_cache: Dict[Path, Dict[str, Dict[int, Tuple[float, float, float, float, float, float]]]] = {}
    parsed_files = 0
    row = 0
    for si in range(n_scenarios):
        for pi in range(n_planners):
            meta = by_pair.get((si, pi), {"scenario_index": str(si), "planner_id": str(pi)})
            msg = find_msgpack(args.sim_dir, meta)
            if msg is None:
                raise FileNotFoundError(f"No official nuPlan msgpack found for scenario_index={si}, planner_id={pi}, planner_name={meta.get('planner_name','')}")
            if msg not in parsed_cache:
                tracks, sample_count = parse_neighbor_world_tracks(msg, timesteps)
                parsed_cache[msg] = tracks
                parsed_files += 1
                if sample_count != timesteps:
                    warnings.append({"type": "msgpack_timestep_mismatch", "row": row, "path": str(msg), "message": f"msgpack samples={sample_count}, expected T={timesteps}; extraction uses overlapping prefix only."})
            arr, ids = build_row_neighbors(np.asarray(seq[si, pi]), np.asarray(mask[si, pi]).astype(bool), parsed_cache[msg], args.max_neighbors)
            neighbor[row] = arr
            slot_ids[row] = ids
            coverage = float(np.mean(arr[:, :, -1] > 0.5))
            if coverage < args.low_coverage_threshold:
                warnings.append({"type": "low_neighbor_coverage", "row": row, "scenario_index": si, "planner_id": pi, "coverage": coverage, "message": "Neighbor coverage is low; no neighbors were fabricated."})
            row += 1
    if row != expected_rows:
        raise ValueError(f"Output row mismatch: rows={row}, expected num_scenarios*num_planners={expected_rows}")
    if neighbor.shape != (expected_rows, args.max_neighbors, timesteps, 9):
        raise ValueError(f"neighbor_seq shape invalid: {list(neighbor.shape)}")
    if not np.isfinite(neighbor).all():
        raise ValueError("neighbor_seq contains NaN or +/-inf; extraction must write finite values and valid=0 for missing neighbors.")
    if not np.any(neighbor[:, :, :, -1] > 0.5):
        raise ValueError("All neighbor valid flags are zero; refusing to write empty neighbor tensors.")
    np.save(out_seq, neighbor)
    np.save(out_ids, slot_ids)
    schema = {"stage": "7D", "source_stage": "7C official nuPlan simulation", "row_semantics": "scenario_planner_controlled_ego_rollout", "multi_agent_ego_expansion": False, "num_scenarios": int(n_scenarios), "num_planners": int(n_planners), "rows": int(expected_rows), "shape": list(neighbor.shape), "slot_ids_shape": list(slot_ids.shape), "neighbor_layout": "ego_centric_relative_to_each_planner_controlled_simulated_ego", "neighbor_channels": NEIGHBOR_CHANNELS, "max_neighbors": int(args.max_neighbors), "uses_official_nuplan_simulation": True, "pseudo_rollout": False, "parsed_msgpack_files": parsed_files}
    write_json(args.sim_dir / "stage7d_neighbor_schema.json", schema)
    write_json(args.sim_dir / "stage7d_neighbor_warnings.json", {"warnings": warnings, "validation": {"pass": True, "all_valid_flags_zero": False, "low_coverage_rows": sum(1 for w in warnings if w.get("type") == "low_neighbor_coverage")}})
    report = f"""# Stage 7D Neighbor Extraction Report\n\n## Status\n\nPASS — extracted official nuPlan background tracks as ego-centric neighbor tensors.\n\n## Row semantics\n\n- row = scenario × planner-controlled ego rollout\n- num_scenarios: `{n_scenarios}`\n- num_planners: `{n_planners}`\n- total_rows: `{expected_rows}`\n- multi-agent ego expansion: `False`\n\n## Outputs\n\n- stage7d_neighbor_seq.npy shape: `{list(neighbor.shape)}`\n- stage7d_neighbor_slot_ids.npy shape: `{list(slot_ids.shape)}`\n- neighbor channels: `{NEIGHBOR_CHANNELS}`\n\n## Validation\n\n- official nuPlan simulation confirmed: `True`\n- pseudo_rollout: `False`\n- T matches simulated_ego_seq.npy: `True`\n- last dimension equals 9: `True`\n- all valid flags zero: `False`\n- warnings: `{len(warnings)}`\n"""
    (args.sim_dir / "stage7d_neighbor_report.md").write_text(report, encoding="utf-8")
    print(f"Stage 7D neighbor extraction PASS: {args.sim_dir}")


if __name__ == "__main__":
    main()