#!/usr/bin/env python3
"""Build a fixed-dimension BDD matrix from frozen Stage6/Stage7 assets.

This is a read-only evaluation tool.  It never trains, changes checkpoints,
changes planners, or selects new scenarios.  Stage7 A/B/C/ego13 embeddings are
exported only from the already locked 310 assertive/conservative pairs and are
explicitly labelled POST_HOC_STANDARDIZED_DESCRIPTIVE_EVALUATION.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage6k_run_longitudinal_dose_bdd import null_diagnostics  # noqa: E402
from tools.stage6l_prepare_context_representation_ablation import (  # noqa: E402
    apply_scaler,
    ego_kinematic_features,
)
from tools.stage6u_unified_abc_trainer import UnifiedABCModel, feature_group_indices  # noqa: E402
from tools.stage7_m6_scenario_conditioned_bdd import (  # noqa: E402
    exact_median_bandwidth,
    holm_adjust,
    rbf_kernel,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DEFAULT = ROOT / "configs/standardized_fixed_dimension_bdd_protocol_v1.json"
SCHEMA = ROOT / "configs/unified_bdd_reporting_schema_v1.json"
SCHEMA_FREEZE = ROOT / "docs/unified_bdd_reporting_schema_freeze_v1.json"
LEDGER = ROOT / "outputs/stage6u_abc_formal_training_v1/checkpoint_lock/stage6u_formal_checkpoint_ledger.json"
OLD_STAGE7_MANIFEST = ROOT / "outputs/stage7_m6_5_locked_confirmation_representations_v1/m6_5_representation_manifest.json"
SCALER = ROOT / "outputs/stage6l_context_representation_ablation_representations_v2_runtime_repaired/scalers/handcrafted_reference_scalers.npz"
STAGE6P_DECISIONS = ROOT / "outputs/stage6v_stage6p_unpaired_blind_v1/stage6v_stage6p_primary_decisions.csv"
STAGE6JK_DECISIONS = ROOT / "outputs/stage6v_stage6jk_paired_blind_v1/stage6v_stage6jk_decisions.csv"
WAYMO_DECISIONS = ROOT / "outputs/stage6v_waymo_dynamic_v2_test_v1/waymo_test_decisions.csv"
STAGE6V_FINAL = ROOT / "outputs/stage6v_one_time_blind_evaluation_final_v1/stage6v_blind_evaluation_final_manifest.json"
STAGE6S_INCREMENT = ROOT / "outputs/stage6s_v3_confirmation_representations_v1/stage6s_v3_c_context_increment.json"

REPS = ("old64", "A", "B", "C", "ego13")
STAGE7_TASKS: Mapping[str, tuple[str, ...]] = {
    "following_interaction": (
        "following_lane_with_lead",
        "following_lane_with_slow_lead",
        "near_long_vehicle",
    ),
    "lane_change": ("changing_lane_to_left", "changing_lane_to_right"),
    "stop_go_control": (
        "accelerating_at_traffic_light_with_lead",
        "stationary_at_traffic_light_without_lead",
        "stationary_in_traffic",
        "stopping_at_traffic_light_with_lead",
        "stopping_with_lead",
    ),
    "high_motion_dynamics": (
        "high_lateral_acceleration",
        "high_magnitude_speed",
        "medium_magnitude_speed",
    ),
    "dense_or_vulnerable_interaction": (
        "near_multiple_vehicles",
        "near_pedestrian_on_crosswalk",
    ),
}
DIMENSION_META = {
    "OVR.ALL": ("总体行为漂移", "Overall"),
    "LON.FREE_FLOW_SPEED": ("自由流速度", "Longitudinal"),
    "LON.ACCEL_DECEL": ("纵向加速/减速", "Longitudinal"),
    "LON.CAR_FOLLOWING": ("跟车行为", "Longitudinal"),
    "LON.CLOSING_RESPONSE": ("逼近前车响应", "Longitudinal"),
    "LON.COMFORT": ("纵向平顺性", "Longitudinal"),
    "LAT.LANE_KEEPING": ("车道保持", "Lateral"),
    "LAT.LANE_CHANGE": ("变道行为", "Lateral"),
    "LAT.DYNAMICS": ("横向动态", "Lateral"),
    "INT.FRONT_GAP_THW": ("前车间距/车头时距交互", "Interaction"),
    "INT.LONG_FOLLOWING": ("纵向跟车交互响应", "Interaction"),
    "INT.LATERAL_GAP": ("横向间隙接受/横向交互", "Interaction"),
    "INT.MERGE_YIELD_CUTIN": ("汇入/让行/切入响应", "Interaction"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_DEFAULT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/standardized_fixed_dimension_bdd_matrix_v1",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate all frozen inputs and checkpoint hashes without exporting embeddings or BDD.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def assert_status(payload: Mapping[str, Any], expected: str, label: str) -> None:
    if payload.get("status") != expected:
        raise ValueError(f"{label} status must be {expected!r}, got {payload.get('status')!r}")


def as_number(value: Any) -> float | None:
    if value is None or (isinstance(value, str) and value.strip().upper() in {"", "N/A", "NA"}):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def safe_text(value: Any, default: str = "N/A") -> str:
    if value is None:
        return default
    if isinstance(value, float) and not math.isfinite(value):
        return default
    text = str(value)
    return text if text and text.lower() != "nan" else default


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def schema_names(context_dir: Path) -> list[str]:
    schema = read_json(context_dir / "feature_schema.json")
    names = schema.get("feature_names") or [row["name"] for row in schema.get("features", [])]
    # The tensor is 83D (8 ego + 5 × 15 neighbour/context channels), while the
    # feature schema freezes the 33D global supervision names used by the A/B/C
    # output heads.  feature_group_indices intentionally consumes those 33 names.
    if len(names) != 33:
        raise ValueError(f"Expected 33 global raw supervision feature names in {context_dir}, got {len(names)}")
    return [str(name) for name in names]


def build_pairs(metadata: pd.DataFrame, pair_delta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required_meta = {"global_row", "planner_name", "scenario_token", "scenario_type", "log_name"}
    required_delta = {"scenario", "row_A", "row_B"}
    missing_meta = sorted(required_meta - set(metadata.columns))
    missing_delta = sorted(required_delta - set(pair_delta.columns))
    if missing_meta or missing_delta:
        raise ValueError(f"Stage7 pair inputs missing metadata={missing_meta}, paired_delta={missing_delta}")
    meta = metadata.sort_values("global_row").reset_index(drop=True)
    if list(meta["global_row"].astype(int)) != list(range(len(meta))):
        raise ValueError("Stage7 global_row must be contiguous and sorted 0..N-1")
    if len(pair_delta) != 310 or len(meta) != 620:
        raise ValueError(f"Expected frozen Stage7 310/620 pairs/rows, got {len(pair_delta)}/{len(meta)}")
    index = meta.set_index("global_row", drop=False)
    pairs: list[tuple[int, int]] = []
    types: list[str] = []
    logs: list[str] = []
    used: list[int] = []
    for pair_position, row in pair_delta.reset_index(drop=True).iterrows():
        a, b = int(row.row_A), int(row.row_B)
        if a == b or a < 0 or b < 0 or a >= len(meta) or b >= len(meta):
            raise ValueError(f"Invalid Stage7 pair at position {pair_position}: {a}/{b}")
        ma, mb = index.loc[a], index.loc[b]
        if str(ma.scenario_token) != str(row.scenario) or str(mb.scenario_token) != str(row.scenario):
            raise ValueError(f"Stage7 scenario mismatch at pair {pair_position}")
        if str(ma.planner_name) != "pdm_closed_assertive_v1" or str(mb.planner_name) != "pdm_closed_conservative_v1":
            raise ValueError(f"Stage7 planner order is not assertive/conservative at pair {pair_position}")
        if str(ma.scenario_type) != str(mb.scenario_type) or str(ma.log_name) != str(mb.log_name):
            raise ValueError(f"Stage7 pair metadata mismatch at pair {pair_position}")
        pairs.append((a, b))
        types.append(str(ma.scenario_type))
        logs.append(str(ma.log_name))
        used.extend((a, b))
    if len(set(used)) != 620 or set(used) != set(range(620)):
        raise ValueError("Stage7 paired_delta must cover each of the 620 frozen rows exactly once")
    return np.asarray(pairs, dtype=np.int64), np.asarray(types, dtype=str), np.asarray(logs, dtype=str)


def load_preflight(protocol: Mapping[str, Any]) -> dict[str, Any]:
    if protocol.get("status") != "FROZEN_STANDARDIZED_FIXED_DIMENSION_BDD_PROTOCOL":
        raise ValueError("The standardized protocol is not frozen")
    schema = read_json(SCHEMA)
    assert_status(schema, "UNIFIED_BDD_REPORTING_SCHEMA_FROZEN", "unified schema")
    assert_status(read_json(SCHEMA_FREEZE), "UNIFIED_BDD_REPORTING_SCHEMA_FROZEN", "unified schema freeze")
    sources = protocol["source_contracts"]
    stage6jk_manifest = ROOT / sources["stage6jk"]["result_manifest"]
    stage6s_results = ROOT / sources["stage6s_v3"]["result_csv"]
    assert_status(read_json(stage6jk_manifest), "FROZEN_STAGE6J_K_PAIRED_BLIND_COMPLETE", "Stage6J/K")
    stage6s_mechanism = read_json(ROOT / sources["stage6s_v3"]["mechanism_summary"])
    if stage6s_mechanism.get("mechanism_gate_passed") is not True:
        raise ValueError("Stage6S-v3 mechanism gate is not passed")
    if not stage6s_results.is_file():
        raise FileNotFoundError(stage6s_results)
    ledger = read_json(LEDGER)
    assert_status(ledger, "LOCKED_9_OF_9_READY_FOR_BLIND_EVALUATION_UNLOCK", "Stage6U checkpoint ledger")
    if ledger.get("primary_seed") != protocol["representations"]["primary_seed"]:
        raise ValueError("Stage6U primary seed differs from standardized protocol")
    context_dir = ROOT / sources["stage7"]["context_dir"]
    for name in ("context_traj.npy", "ego_seq.npy", "ego_seq_mask.npy", "metadata.csv", "feature_schema.json"):
        if not (context_dir / name).is_file():
            raise FileNotFoundError(f"Stage7 frozen context asset missing: {context_dir / name}")
    old_manifest = read_json(OLD_STAGE7_MANIFEST)
    old_embedding = ROOT / sources["stage7"]["old64_embedding"]
    expected_old_sha = old_manifest["output_hashes"]["learned_embedding.npy"]
    if sha256(old_embedding) != expected_old_sha:
        raise ValueError("Frozen Stage7 old64 embedding SHA differs from its representation manifest")
    context = np.load(context_dir / "context_traj.npy", mmap_mode="r")
    ego = np.load(context_dir / "ego_seq.npy", mmap_mode="r")
    mask = np.load(context_dir / "ego_seq_mask.npy", mmap_mode="r")
    old = np.load(old_embedding, mmap_mode="r")
    if context.shape != (620, 150, 83) or ego.shape != (620, 150, 8) or mask.shape != (620, 150) or old.shape != (620, 64):
        raise ValueError(f"Unexpected Stage7 frozen shapes: context={context.shape}, ego={ego.shape}, mask={mask.shape}, old={old.shape}")
    if not np.isfinite(old).all():
        raise ValueError("Frozen Stage7 old64 embedding has non-finite values")
    meta = pd.read_csv(context_dir / "metadata.csv")
    pairs, scenario_types, logs = build_pairs(meta, pd.read_csv(ROOT / sources["stage7"]["pair_delta_csv"]))
    task_csv = pd.read_csv(ROOT / sources["stage7"]["task_definition_csv"])
    observed_task_counts = {task: int(np.isin(scenario_types, kinds).sum()) for task, kinds in STAGE7_TASKS.items()}
    declared = {str(row.task): int(row.n_pairs) for _, row in task_csv.iterrows()}
    for task, count in observed_task_counts.items():
        if declared.get(task) != count:
            raise ValueError(f"Frozen Stage7 task count differs for {task}: {declared.get(task)} != {count}")
    return {
        "ledger": ledger,
        "context_dir": context_dir,
        "stage7_metadata": meta,
        "stage7_pairs": pairs,
        "stage7_scenario_types": scenario_types,
        "stage7_logs": logs,
        "stage7_task_counts": observed_task_counts,
    }


def embed(model: torch.nn.Module, context: np.ndarray, device: torch.device, label: str) -> np.ndarray:
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(context), 128), desc=f"导出 {label}", unit="batch"):
            batch = torch.from_numpy(np.asarray(context[start:start + 128], dtype=np.float32).copy()).to(device)
            chunks.append(model(batch).detach().cpu().numpy().astype(np.float64))
    values = np.concatenate(chunks, axis=0)
    if values.shape != (len(context), 64) or not np.isfinite(values).all():
        raise ValueError(f"Invalid {label} embedding: {values.shape}, finite={np.isfinite(values).all()}")
    return values


def build_stage7_representations(preflight: Mapping[str, Any], output_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    context_dir = Path(preflight["context_dir"])
    context = np.asarray(np.load(context_dir / "context_traj.npy", mmap_mode="r"), dtype=np.float32)
    old = np.asarray(np.load(ROOT / "outputs/stage7_m6_5_locked_confirmation_representations_v1/learned_embedding.npy", mmap_mode="r"), dtype=np.float64)
    result: dict[str, np.ndarray] = {"old64": old}
    checksums: dict[str, str] = {"old64": sha256(ROOT / "outputs/stage7_m6_5_locked_confirmation_representations_v1/learned_embedding.npy")}
    groups = feature_group_indices(schema_names(context_dir))
    ledger = preflight["ledger"]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    for candidate in ("A", "B", "C"):
        row = next((item for item in ledger["rows"] if item["candidate"] == candidate and int(item["seed"]) == 3407), None)
        if row is None:
            raise ValueError(f"No locked primary checkpoint for {candidate}")
        checkpoint = Path(row["best_checkpoint_path"])
        if sha256(checkpoint) != row["best_checkpoint_sha256"]:
            raise ValueError(f"Locked checkpoint SHA changed for {candidate}")
        model = UnifiedABCModel(candidate, groups)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        result[candidate] = embed(model.eval().to(device), context, device, candidate)
        checksums[candidate] = str(row["best_checkpoint_sha256"])
    ego = np.asarray(np.load(context_dir / "ego_seq.npy", mmap_mode="r"), dtype=np.float32)
    mask = np.asarray(np.load(context_dir / "ego_seq_mask.npy", mmap_mode="r"), dtype=bool)
    scaler = np.load(SCALER)
    ego13 = apply_scaler(ego_kinematic_features(ego, mask), scaler["ego_median"], scaler["ego_scale"])
    if ego13.shape != (620, 13) or not np.isfinite(ego13).all():
        raise ValueError(f"Invalid ego13 Stage7 representation: {ego13.shape}")
    result["ego13"] = np.asarray(ego13, dtype=np.float64)
    checksums["ego13"] = sha256(SCALER)
    representation_dir = output_dir / "stage7_posthoc_representations"
    representation_dir.mkdir()
    for name, values in result.items():
        np.save(representation_dir / f"{name}.npy", values.astype(np.float32))
    return result, checksums


def signed_swap_null(contrast: np.ndarray, repetitions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(contrast)
    samples = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 2000):
        stop = min(start + 2000, repetitions)
        signs = rng.integers(0, 2, size=(stop - start, n), dtype=np.int8).astype(np.float64) * 2.0 - 1.0
        samples[start:stop] = np.einsum("bi,ij,bj->b", signs, contrast, signs, optimize=True) / (n * n)
    return samples


def paired_bdd(values: np.ndarray, pairs: np.ndarray, *, repetitions: int, seed: int) -> tuple[dict[str, Any], np.ndarray]:
    if len(pairs) < 2:
        raise ValueError("A BDD task needs at least two complete pairs")
    reference = np.asarray(values[pairs[:, 1]], dtype=np.float64)
    target = np.asarray(values[pairs[:, 0]], dtype=np.float64)
    pooled = np.vstack((target, reference))
    bandwidth = exact_median_bandwidth(pooled)
    kernel = rbf_kernel(pooled, bandwidth)
    n = len(pairs)
    contrast = kernel[:n, :n] + kernel[n:, n:] - kernel[:n, n:] - kernel[n:, :n]
    observed = float(contrast.mean())
    samples = signed_swap_null(contrast, repetitions, seed)
    exceedance = int(np.sum(samples >= observed))
    row: dict[str, Any] = {
        "raw_mmd2": observed,
        "bandwidth": bandwidth,
        "null_repetitions": repetitions,
        "raw_p_value": float((exceedance + 1) / (repetitions + 1)),
        "exceedance_count": exceedance,
        "null_or_calibration_method": "within_scenario_pair_label_swap; common seeded sign stream across representations",
        **null_diagnostics(observed, samples),
    }
    return row, samples


def bootstrap_log_mean(values: np.ndarray, logs: np.ndarray, *, seed: int, repetitions: int = 10000) -> tuple[float | None, float | None, int]:
    finite = np.isfinite(values)
    if not finite.any():
        return None, None, 0
    frame = pd.DataFrame({"value": values[finite], "log": logs[finite]})
    grouped = frame.groupby("log", sort=True)["value"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(dtype=np.float64)
    counts = grouped["count"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(grouped), size=(repetitions, len(grouped)))
    weights = np.apply_along_axis(lambda row: np.bincount(row, minlength=len(grouped)), 1, choices)
    denominator = weights @ counts
    samples = (weights @ sums) / denominator
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975)), int(finite.sum())


def stage7_semantic_records(pair_delta: pd.DataFrame, logs: np.ndarray, scope: str) -> list[dict[str, Any]]:
    metric_map: dict[str, list[tuple[str, str, str]]] = {
        "overall": [("mean_speed", "delta_mean_speed", "m/s"), ("rms_accel", "delta_rms_accel", "m/s²")],
        "following_interaction": [("following_mean_speed", "delta_mean_speed", "m/s"), ("following_rms_accel", "delta_rms_accel", "m/s²")],
        "lane_change": [("mean_abs_yaw_rate", "delta_mean_abs_yaw_rate", "rad/s")],
        "stop_go_control": [("mean_speed", "delta_mean_speed", "m/s"), ("rms_accel", "delta_rms_accel", "m/s²")],
        "high_motion_dynamics": [("mean_abs_yaw_rate", "delta_mean_abs_yaw_rate", "rad/s")],
        "dense_or_vulnerable_interaction": [("mean_front_distance", "delta_mean_front_distance", "m")],
    }
    records: list[dict[str, Any]] = []
    for offset, (metric, column, unit) in enumerate(metric_map[scope]):
        values = pair_delta[column].to_numpy(dtype=np.float64)
        low, high, finite_count = bootstrap_log_mean(values, logs, seed=2026081411 + 31 * offset + len(scope))
        records.append({
            "semantic_metric": metric,
            "semantic_delta_target_minus_reference": float(np.nanmean(values)) if finite_count else None,
            "semantic_unit": unit,
            "semantic_ci95_low": low,
            "semantic_ci95_high": high,
            "semantic_finite_pair_count": finite_count,
        })
    return records


def semantic_from_stage6jk(kinematics: pd.DataFrame, scope: str, dimension: str) -> list[dict[str, Any]]:
    fields = {
        "LON.ACCEL_DECEL": ("delta_mean_speed", "delta_rms_accel"),
        "LON.CAR_FOLLOWING": ("delta_mean_speed", "delta_rms_accel"),
        "LON.COMFORT": ("delta_rms_jerk",),
    }[dimension]
    records: list[dict[str, Any]] = []
    for metric in fields:
        row = kinematics[(kinematics.dose_label == "dose100") & (kinematics.scope == scope) & (kinematics.metric == metric)]
        if len(row) != 1:
            raise ValueError(f"Missing frozen Stage6K semantic row {scope}/{metric}")
        value = row.iloc[0]
        records.append({
            "semantic_metric": metric,
            "semantic_delta_target_minus_reference": as_number(value.mean_delta_A_minus_B),
            "semantic_unit": safe_text(value.unit),
            "semantic_ci95_low": as_number(value.cluster_bootstrap_ci95_low),
            "semantic_ci95_high": as_number(value.cluster_bootstrap_ci95_high),
            "semantic_finite_pair_count": int(value.finite_pair_count),
        })
    return records


def semantic_from_stage6s(mechanism: Mapping[str, Any], dimension: str) -> list[dict[str, Any]]:
    specification = {
        "LON.CLOSING_RESPONSE": ("mean_accel_during_closing", "delta_mean_accel_during_closing_mps2", "m/s²"),
        "INT.FRONT_GAP_THW": ("median_front_gap", "delta_median_front_gap_m", "m"),
        "INT.LONG_FOLLOWING": ("mean_accel_during_following_pressure", "delta_mean_accel_during_following_pressure_mps2", "m/s²"),
    }[dimension]
    ci_name, aggregate_name, unit = specification
    interval = mechanism["log_cluster_bootstrap"][f"delta_{ci_name}"]
    records = [{
        "semantic_metric": aggregate_name,
        "semantic_delta_target_minus_reference": as_number(mechanism["aggregate"][aggregate_name]),
        "semantic_unit": unit,
        "semantic_ci95_low": as_number(interval["bootstrap95_low"]),
        "semantic_ci95_high": as_number(interval["bootstrap95_high"]),
        "semantic_finite_pair_count": int(mechanism["complete_pairs"]),
    }]
    if dimension == "INT.FRONT_GAP_THW":
        thw = mechanism["log_cluster_bootstrap"]["delta_median_finite_thw"]
        records.append({
            "semantic_metric": "delta_median_finite_thw_s",
            "semantic_delta_target_minus_reference": as_number(mechanism["aggregate"]["delta_median_finite_thw_s"]),
            "semantic_unit": "s",
            "semantic_ci95_low": as_number(thw["bootstrap95_low"]),
            "semantic_ci95_high": as_number(thw["bootstrap95_high"]),
            "semantic_finite_pair_count": int(mechanism["complete_pairs"]),
        })
    return records


def semantic_summary(records: Sequence[Mapping[str, Any]]) -> str:
    if not records:
        return "N/A"
    pieces = []
    for record in records:
        delta = record.get("semantic_delta_target_minus_reference")
        if delta is None:
            continue
        low, high = record.get("semantic_ci95_low"), record.get("semantic_ci95_high")
        interval = "" if low is None or high is None else f"; 95% CI [{float(low):+.3f}, {float(high):+.3f}]"
        pieces.append(f"{record['semantic_metric']} {float(delta):+.3f} {record['semantic_unit']}{interval}")
    return "; ".join(pieces) if pieces else "N/A"


def direction_for(dimension: str, records: Sequence[Mapping[str, Any]], *, scope: str = "") -> str:
    first = records[0] if records else {}
    low = first.get("semantic_ci95_low")
    high = first.get("semantic_ci95_high")
    positive = low is not None and float(low) > 0
    negative = high is not None and float(high) < 0
    if dimension == "LON.CAR_FOLLOWING":
        return "TARGET_MORE_ACTIVE_FOLLOWING" if positive else "NO_CLEAR_DIRECTION"
    if dimension == "LON.ACCEL_DECEL":
        return "TARGET_HIGHER_LONGITUDINAL_EXCITATION" if positive else ("TARGET_LOWER_LONGITUDINAL_EXCITATION" if negative else "NO_CLEAR_DIRECTION")
    if dimension == "LON.COMFORT":
        return "TARGET_HIGHER_LONGITUDINAL_JERK" if positive else ("TARGET_LOWER_LONGITUDINAL_JERK" if negative else "NO_CLEAR_DIRECTION")
    if dimension == "LON.CLOSING_RESPONSE":
        return "TARGET_MAINTAINS_MORE_ACCEL_DURING_CLOSING" if positive else "NO_CLEAR_DIRECTION"
    if dimension == "INT.FRONT_GAP_THW":
        return "TARGET_SHORTER_GAP_OR_THW" if negative else "NO_CLEAR_DIRECTION"
    if dimension == "INT.LONG_FOLLOWING":
        return "TARGET_MORE_ACCEL_UNDER_FOLLOWING_PRESSURE" if positive else "NO_CLEAR_DIRECTION"
    if dimension == "LAT.LANE_CHANGE":
        return "N/A_TASK_SLICE_DOES_NOT_CONFIRM_EXECUTED_LANE_CHANGE"
    if dimension == "LAT.DYNAMICS":
        return "TARGET_HIGHER_LATERAL_EXCITATION_PROXY" if positive else "NO_CLEAR_DIRECTION"
    if dimension == "INT.MERGE_YIELD_CUTIN":
        return "N/A_DENSE_OR_VULNERABLE_PROXY_NOT_A_MERGE_YIELD_CUTIN_EVENT"
    if dimension == "OVR.ALL":
        return "MIXED_NO_SINGLE_STYLE_DIRECTION"
    return "N/A"


def row_base(*, result_id: str, parent: str, dimension: str, reference: str, target: str, task: str, mode: str, n_pairs: int | None, n_logs: int | None, representation: str, source: str, classification: str, mapping_strength: str, semantic: Sequence[Mapping[str, Any]], direction: str, stats: Mapping[str, Any] | None = None, corrected: Any = None, detection: Any = None, note: str = "") -> dict[str, Any]:
    name, level = DIMENSION_META[dimension]
    row: dict[str, Any] = {
        "schema_version": "standardized_fixed_dimension_bdd_matrix_v1",
        "report_id": "standardized_fixed_dimension_bdd_matrix_v1",
        "result_id": result_id,
        "parent_bdd_result_id": parent,
        "dimension_id": dimension,
        "behavior_dimension": name,
        "behavior_level": level,
        "behavior_reference": reference,
        "target": target,
        "contrast_label": f"{target} | {reference}",
        "task_id": task,
        "evaluation_mode": mode,
        "n_pairs": n_pairs,
        "n_scenarios": n_pairs,
        "n_logs": n_logs,
        "representation_id": representation,
        "representation_baseline": "old64" if representation == "old64" else "old64_capability_baseline_not_raw_mmd2_reference",
        "statistic_name": "biased_single_rbf_mmd2",
        "null_reference": "N/A",
        "raw_mmd2": None,
        "null_q95": None,
        "bdd_to_null_q95_ratio": None,
        "z_bdd": None,
        "raw_p_value": None,
        "corrected_p_value": corrected,
        "detection_or_pass": detection,
        "semantic_metric": " + ".join(str(item["semantic_metric"]) for item in semantic) if semantic else "N/A",
        "semantic_delta_target_minus_reference": semantic_summary(semantic),
        "semantic_95ci": "; ".join(f"[{safe_text(item.get('semantic_ci95_low'))}, {safe_text(item.get('semantic_ci95_high'))}]" for item in semantic) if semantic else "N/A",
        "semantic_direction": direction,
        "mapping_strength": mapping_strength,
        "evidence_status": classification,
        "provenance_path": source,
        "shared_parent_bdd": bool(parent != result_id),
        "interpretation": note,
    }
    if stats is not None:
        row.update({
            "null_reference": safe_text(stats.get("null_or_calibration_method")),
            "raw_mmd2": as_number(stats.get("raw_mmd2", stats.get("mmd2"))),
            "null_q95": as_number(stats.get("paired_null_q95")),
            "bdd_to_null_q95_ratio": as_number(stats.get("bdd_to_null_q95_ratio")),
            "z_bdd": as_number(stats.get("null_standardized_z_bdd")),
            "raw_p_value": as_number(stats.get("raw_p_value", stats.get("raw_p"))),
            "bandwidth": as_number(stats.get("bandwidth")),
            "null_repetitions": int(stats["null_repetitions"] if "null_repetitions" in stats else stats.get("permutations", 0)),
        })
    return row


def stage6jk_rows(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    cfg = protocol["source_contracts"]["stage6jk"]
    results = pd.read_csv(ROOT / cfg["result_csv"])
    kinematics = pd.read_csv(ROOT / cfg["kinematic_csv"])
    expected = set(REPS) | {"old64"}
    if not set(REPS).issubset(set(results.representation.astype(str))):
        raise ValueError("Stage6J/K results do not contain every fixed representation")
    output: list[dict[str, Any]] = []
    scope_dimension = {
        "overall": "LON.ACCEL_DECEL",
        "following_interaction": "LON.CAR_FOLLOWING",
        "longitudinal_high_motion": "LON.ACCEL_DECEL",
        "stop_go_control": "LON.ACCEL_DECEL",
    }
    def frozen_log_count(dose: str, scope: str, pair_count: int) -> int:
        selected = kinematics[(kinematics.dose_label == dose) & (kinematics.scope == scope)]
        # Gap/THW metrics can be finite on only a subset of the fixed paired
        # roster, so their distinct-log count is not the denominator of the BDD.
        # Use a semantic metric whose finite pair count covers the complete BDD
        # slice (speed/accel do so in every frozen Stage6J/K scope).
        selected = selected[selected.finite_pair_count.astype(int) == int(pair_count)]
        counts = selected["distinct_log_count"].dropna().astype(int).unique().tolist()
        if len(counts) != 1:
            raise ValueError(f"Stage6K does not provide one distinct log count for {dose}/{scope}: {counts}")
        return int(counts[0])
    for _, source in results.iterrows():
        rep, scope, dose = str(source.representation), str(source.scope), str(source.dose_label)
        if rep not in REPS or scope not in scope_dimension:
            continue
        dimension = scope_dimension[scope]
        semantic = semantic_from_stage6jk(kinematics, scope, dimension) if dose == "dose100" else []
        parent = f"stage6jk:{rep}:{dose}:{scope}"
        stats = source.to_dict()
        stats["null_or_calibration_method"] = cfg["null_method"]
        row = row_base(
            result_id=parent,
            parent=parent,
            dimension=dimension,
            reference=cfg["reference"],
            target=cfg["target"],
            task=f"stage6jk_{scope}_{dose}",
            mode="paired",
            n_pairs=int(source.n_pairs),
            n_logs=frozen_log_count(dose, scope, int(source.n_pairs)),
            representation=rep,
            source=cfg["result_csv"],
            classification=cfg["classification"],
            mapping_strength="TREATMENT_ALIGNED_PROXY" if scope == "overall" else "TASK_SLICE_PROXY",
            semantic=semantic,
            direction=direction_for(dimension, semantic, scope=scope) if semantic else "N/A_DOSE_SEMANTIC_NOT_REPORTED_IN_PRIMARY_CARD",
            stats=stats,
            corrected=as_number(source.holm_p),
            detection=bool(source.reject_holm_0_05),
            note="Inherited Stage6J/K paired result. Raw MMD² is audit-only and is not ranked across representations.",
        )
        output.append(row)
    for rep in REPS:
        parent = f"stage6jk:{rep}:dose100:overall"
        semantic = semantic_from_stage6jk(kinematics, "overall", "LON.COMFORT")
        source = results[(results.representation == rep) & (results.dose_label == "dose100") & (results.scope == "overall")]
        if len(source) != 1:
            raise ValueError(f"Stage6J/K missing dose100 overall for {rep}")
        stats = source.iloc[0].to_dict()
        stats["null_or_calibration_method"] = cfg["null_method"]
        output.append(row_base(
            result_id=f"{parent}:comfort_child",
            parent=parent,
            dimension="LON.COMFORT",
            reference=cfg["reference"], target=cfg["target"], task="stage6jk_overall_dose100_shared_parent", mode="paired",
            n_pairs=int(source.iloc[0].n_pairs), n_logs=frozen_log_count("dose100", "overall", int(source.iloc[0].n_pairs)), representation=rep,
            source=cfg["result_csv"], classification="SHARED_PARENT_BDD_SEMANTIC_PROXY", mapping_strength="TREATMENT_ALIGNED_PROXY",
            semantic=semantic, direction=direction_for("LON.COMFORT", semantic), stats=stats,
            corrected=as_number(source.iloc[0].holm_p), detection=bool(source.iloc[0].reject_holm_0_05),
            note="Shared parent BDD with LON.ACCEL_DECEL; not an independent comfort BDD test.",
        ))
    return output


def stage6s_rows(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    cfg = protocol["source_contracts"]["stage6s_v3"]
    results = pd.read_csv(ROOT / cfg["result_csv"])
    mechanism = read_json(ROOT / cfg["mechanism_summary"])
    output: list[dict[str, Any]] = []
    for _, source in results.iterrows():
        rep = str(source.representation)
        if rep not in REPS and rep != "C_neighbor_zero":
            continue
        dimensions = ("LON.CLOSING_RESPONSE", "INT.FRONT_GAP_THW", "INT.LONG_FOLLOWING")
        stats = source.to_dict()
        stats["null_or_calibration_method"] = cfg["null_method"]
        for position, dimension in enumerate(dimensions):
            parent = f"stage6s_v3:{rep}:following_interaction"
            semantic = semantic_from_stage6s(mechanism, dimension)
            output.append(row_base(
                result_id=parent if position == 0 else f"{parent}:{dimension}", parent=parent, dimension=dimension,
                reference=cfg["reference"], target=cfg["target"], task="stage6s_v3_following_interaction_confirmation", mode="paired",
                n_pairs=int(source.n_pairs), n_logs=int(cfg["log_count"]), representation=rep, source=cfg["result_csv"],
                classification=cfg["classification"] if rep != "C_neighbor_zero" else "DIAGNOSTIC_C_NEIGHBOR_ZERO_NOT_A_MAIN_MATRIX_COLUMN",
                mapping_strength="EXACT_DIMENSION", semantic=semantic, direction=direction_for(dimension, semantic), stats=stats,
                corrected="N/A_NOT_A_FROZEN_MULTIPLICITY_FAMILY", detection=bool(source.candidate_detection_gate_pass),
                note=("Shared parent BDD across three interaction semantic child rows; it is not three independent BDD tests. "
                      "C_neighbor_zero is a diagnostic, not a primary representation.") if position else "Stage6S-v3 inherited confirmation result.",
            ))
    return output


def stage7_rows(protocol: Mapping[str, Any], preflight: Mapping[str, Any], representations: Mapping[str, np.ndarray], output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = protocol["source_contracts"]["stage7"]
    pairs = np.asarray(preflight["stage7_pairs"], dtype=np.int64)
    types = np.asarray(preflight["stage7_scenario_types"], dtype=str)
    logs = np.asarray(preflight["stage7_logs"], dtype=str)
    pair_delta = pd.read_csv(ROOT / cfg["pair_delta_csv"])
    if len(pair_delta) != len(pairs):
        raise ValueError("Stage7 pair_delta length differs from frozen pair index")
    masks: dict[str, np.ndarray] = {"overall": np.ones(len(pairs), dtype=bool)}
    masks.update({name: np.isin(types, values) for name, values in STAGE7_TASKS.items()})
    dimension_by_scope = {
        "overall": "OVR.ALL",
        "following_interaction": "LON.CAR_FOLLOWING",
        "lane_change": "LAT.LANE_CHANGE",
        "stop_go_control": "LON.ACCEL_DECEL",
        "high_motion_dynamics": "LAT.DYNAMICS",
        "dense_or_vulnerable_interaction": "INT.MERGE_YIELD_CUTIN",
    }
    raw_rows: list[dict[str, Any]] = []
    nulls: dict[str, np.ndarray] = {}
    for rep in REPS:
        for scope in cfg["task_scopes"]:
            mask = masks[scope]
            stats, samples = paired_bdd(representations[rep], pairs[mask], repetitions=int(cfg["permutations"]), seed=int(cfg["seed"]))
            key = f"{rep}__{scope}"
            nulls[key] = samples.astype(np.float32)
            semantic = stage7_semantic_records(pair_delta.loc[mask].reset_index(drop=True), logs[mask], scope)
            parent = f"stage7_posthoc:{rep}:{scope}"
            raw_rows.append(row_base(
                result_id=parent, parent=parent, dimension=dimension_by_scope[scope], reference=cfg["reference"], target=cfg["target"],
                task=f"stage7_{scope}", mode="paired", n_pairs=int(mask.sum()), n_logs=int(np.unique(logs[mask]).size), representation=rep,
                source=cfg["context_dir"], classification=cfg["classification"],
                mapping_strength=("EXACT_DIMENSION" if scope == "overall" else ("TASK_SLICE_PROXY" if scope in {"following_interaction", "lane_change", "stop_go_control"} else "MIXED_PROXY")),
                semantic=semantic, direction=direction_for(dimension_by_scope[scope], semantic, scope=scope), stats=stats,
                corrected="N/A_PENDING_POST_HOC_HOLM", detection=None,
                note="POST_HOC_STANDARDIZED_DESCRIPTIVE_EVALUATION: fixed Stage7 roster and checkpoint re-export; it is not a Stage6V confirmatory endpoint.",
            ))
    for rep in REPS:
        secondary = [row for row in raw_rows if row["representation_id"] == rep and row["task_id"] != "stage7_overall"]
        adjusted = holm_adjust([float(row["raw_p_value"]) for row in secondary])
        for row, corrected in zip(secondary, adjusted):
            row["corrected_p_value"] = float(corrected)
            row["detection_or_pass"] = bool(corrected < 0.05)
        overall = next(row for row in raw_rows if row["representation_id"] == rep and row["task_id"] == "stage7_overall")
        overall["corrected_p_value"] = "N/A_DESCRIPTIVE_OVERALL"
        overall["detection_or_pass"] = bool(float(overall["raw_p_value"]) < 0.05)
    np.savez_compressed(output_dir / "stage7_posthoc_null_samples.npz", **nulls)
    audit = {
        "classification": cfg["classification"],
        "pair_count": int(len(pairs)),
        "task_pair_counts": {scope: int(mask.sum()) for scope, mask in masks.items()},
        "common_swap_stream_within_task_across_representations": True,
        "permutations": int(cfg["permutations"]),
        "seed": int(cfg["seed"]),
        "raw_mmd2_cross_representation_ranking": False,
        "semantic_delta_is_target_minus_reference": True,
        "note": cfg["post_hoc_boundary"],
    }
    return raw_rows, audit


def primary_matrix(protocol: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cells = protocol["primary_dimension_cells"]
    matrix: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    for dimension in protocol["behavior_dimensions"]:
        config = cells[dimension]
        name, level = DIMENSION_META[dimension]
        item: dict[str, Any] = {"dimension_id": dimension, "behavior_dimension": name, "behavior_level": level, "source": config["source"], "evidence_status": config["evidence_status"]}
        candidates: list[Mapping[str, Any]] = []
        for rep in REPS:
            match: list[Mapping[str, Any]] = []
            if config["source"] == "stage6jk":
                match = [row for row in rows if row["dimension_id"] == dimension and row["representation_id"] == rep and row["task_id"] == f"stage6jk_{config['scope']}_{config['dose']}"]
                if dimension == "LON.COMFORT":
                    match = [row for row in rows if row["dimension_id"] == dimension and row["representation_id"] == rep and row["task_id"] == "stage6jk_overall_dose100_shared_parent"]
            elif config["source"] == "stage6s_v3":
                match = [row for row in rows if row["dimension_id"] == dimension and row["representation_id"] == rep and row["task_id"] == "stage6s_v3_following_interaction_confirmation"]
            elif config["source"] == "stage7":
                match = [row for row in rows if row["dimension_id"] == dimension and row["representation_id"] == rep and row["task_id"] == f"stage7_{config['scope']}"]
            if not match:
                item[rep] = "N/A"
                continue
            if len(match) != 1:
                raise ValueError(f"Ambiguous primary matrix cell {dimension}/{rep}: {len(match)} rows")
            row = match[0]
            candidates.append(row)
            item[rep] = f"{float(row['bdd_to_null_q95_ratio']):.2f}× / Z={float(row['z_bdd']):.2f}"
        passing = [row for row in candidates if bool(row["detection_or_pass"])]
        if passing:
            best = max(passing, key=lambda row: float(row["z_bdd"]))
            item["best_capability"] = f"{best['representation_id']} (max within-null Z={float(best['z_bdd']):.2f}; no raw-MMD² ranking)"
        else:
            item["best_capability"] = "N/A"
        matrix.append(item)
        if config["source"] == "none":
            gaps.append({"dimension_id": dimension, "behavior_dimension": name, "evidence_status": config["evidence_status"], "missing": "没有冻结的同维度场景、BDD及绑定semantic delta；N/A不等同于没有差异。"})
    return matrix, gaps


def gate_scorecard() -> list[dict[str, Any]]:
    final = read_json(STAGE6V_FINAL)
    paired = pd.read_csv(STAGE6JK_DECISIONS).set_index("representation")
    unpaired = pd.read_csv(STAGE6P_DECISIONS)
    waymo = pd.read_csv(WAYMO_DECISIONS)
    increment = read_json(STAGE6S_INCREMENT)
    output: list[dict[str, Any]] = []
    mapping = {"old64": ("old64", None), "A": ("A", "A_3407"), "B": ("B", "B_3407"), "C": ("C", "C_3407"), "ego13": ("ego13", "ego13")}
    for rep, (paired_key, unpaired_key) in mapping.items():
        stage6jk = bool(paired.loc[paired_key, "frozen_longitudinal_gate_pass"]) if paired_key in paired.index else "N/A"
        decision_row = unpaired[unpaired.representation == (unpaired_key or "old64")]
        stage6p = bool(decision_row.iloc[0].frozen_n400_gate_pass) if len(decision_row) == 1 else "N/A"
        if unpaired_key and unpaired_key in set(waymo.representation):
            waymo_gate: bool | str = bool(waymo.set_index("representation").loc[unpaired_key, "all_waymo_gates_pass"])
        else:
            waymo_gate = "N/A_NOT_A_STAGE6T_LEARNED_CANDIDATE"
        interaction = bool(increment["incremental_interaction_information_pass"]) if rep == "C" else "N/A_C_ONLY_DIAGNOSTIC"
        joint = bool(final["candidate_scorecards"][rep]["stage6p_n400_gate_pass"] and final["candidate_scorecards"][rep]["stage6jk_longitudinal_gate_pass"] and final["candidate_scorecards"][rep]["waymo_primary_all_gates_pass"]) if rep in final["candidate_scorecards"] else "N/A_NOT_ABC_CANDIDATE"
        output.append({
            "representation_id": rep,
            "representation_baseline": "old64" if rep == "old64" else "compared_to_old64_by_capability_not_raw_mmd2",
            "stage6jk_paired_gate_pass": stage6jk,
            "stage6p_unpaired_gate_pass": stage6p,
            "waymo_gate_pass": waymo_gate,
            "interaction_increment_gate_pass": interaction,
            "stage6v_joint_candidate_gate_pass": joint,
        })
    return output


def markdown_table(frame: pd.DataFrame) -> str:
    return frame.fillna("N/A").to_markdown(index=False)


def build_report(protocol: Mapping[str, Any], matrix: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], scorecard: Sequence[Mapping[str, Any]], gaps: Sequence[Mapping[str, Any]], stage7_audit: Mapping[str, Any]) -> str:
    primary = pd.DataFrame(matrix)
    display = primary[["behavior_dimension", "old64", "A", "B", "C", "ego13", "best_capability"]]
    following = pd.DataFrame([row for row in rows if row["task_id"] == "stage6jk_following_interaction_dose100" and row["representation_id"] in REPS])
    following = following[["representation_id", "raw_mmd2", "null_q95", "bdd_to_null_q95_ratio", "z_bdd", "raw_p_value", "corrected_p_value", "detection_or_pass", "n_pairs", "n_logs", "semantic_delta_target_minus_reference", "semantic_direction"]]
    interaction = pd.DataFrame([row for row in rows if row["task_id"] == "stage6s_v3_following_interaction_confirmation" and row["dimension_id"] == "INT.LONG_FOLLOWING"])
    interaction = interaction[["representation_id", "raw_mmd2", "null_q95", "bdd_to_null_q95_ratio", "z_bdd", "raw_p_value", "detection_or_pass", "semantic_delta_target_minus_reference", "semantic_direction", "evidence_status"]]
    lane = pd.DataFrame([row for row in rows if row["task_id"] == "stage7_lane_change"])
    lane = lane[["representation_id", "raw_mmd2", "null_q95", "bdd_to_null_q95_ratio", "z_bdd", "raw_p_value", "corrected_p_value", "detection_or_pass", "semantic_delta_target_minus_reference", "semantic_direction"]]
    style = pd.DataFrame([row for row in rows if row["task_id"] in {"stage7_overall", "stage7_following_interaction", "stage7_lane_change", "stage7_high_motion_dynamics", "stage7_dense_or_vulnerable_interaction"} and row["representation_id"] == "B"])
    style = style[["behavior_dimension", "contrast_label", "representation_id", "bdd_to_null_q95_ratio", "z_bdd", "corrected_p_value", "semantic_delta_target_minus_reference", "semantic_direction", "evidence_status"]]
    lines = [
        "# Standardized Fixed-Dimension BDD Evaluation Report", "",
        f"> 协议：`{protocol['schema_version']}`",
        "> 状态：`STANDARDIZED_FIXED_DIMENSION_BDD_MATRIX_COMPLETE`",
        "> 边界：不训练、不改 checkpoint、不改 planner、不重选场景；Stage7 的新表示导出均为 `POST_HOC_STANDARDIZED_DESCRIPTIVE_EVALUATION`，不会改写 Stage6V 联合结论。", "",
        "## 1. 三类 Reference（必须分开读）", "",
        "- **Behavior Reference**：每一行均明确 Reference planner/version 与 Target planner/version；semantic delta 统一为 **Target − Reference**。",
        "- **Null Reference**：paired 行使用该 representation 自己的 pair-label-swap/randomization null；unpaired 行使用该 representation 自己独立的 A/A calibration。BDD 行均保留 null q95。",
        "- **Representation Baseline**：old64 是历史 baseline。A/B/C/ego13 对 old64 的比较仅限检测能力与各自null标准化结果；**禁止用 raw MMD² 跨表示排序**。", "",
        "## 2. 固定行为维度 × representation 主矩阵", "",
        "单元格为 `BDD/null-q95 ratio / Z_BDD`。Best capability 按各自null内的 Z 与检出状态描述，绝不按 raw MMD² 排名。`N/A` 是证据缺口，不是没有差异。", "",
        markdown_table(display), "",
        "## 3. 同一跟车工况：Stage6J/K dose100 following_interaction", "",
        "Behavior Reference：`pdm_closed_longitudinal_conservative_v2 → pdm_closed_longitudinal_assertive_v2`；Null Reference：各 representation 的冻结 paired label-swap null；60个相同场景、52个相同log。跟车方向只由 speed/accel 语义解释，因此为 `TARGET_MORE_ACTIVE_FOLLOWING`，不写成 `CLOSER`。", "",
        markdown_table(following), "",
        "## 4. 相同 interaction confirmation：Stage6S-v3（80对、11 log）", "",
        "Behavior Reference：`pdm_closed_interaction_long_headway_v2 → pdm_closed_interaction_short_headway_v2`。front-gap、finite THW、closing acceleration、following-pressure acceleration 使用完全相同的冻结轨迹机制；THW仅为有限物理值，排除 sentinel/cap。每个 representation 的三条语义子行共享同一个 parent BDD，不能当作三次独立检验。", "",
        markdown_table(interaction), "",
        "C-neighbor-zero 的既有诊断：`C full − C neighbor-zero ΔZ = -7.852`，log-cluster 95% CI `[-33.393, 29.219]`，增量 interaction gate = `False`。这不改变主矩阵中的 C 列。", "",
        "## 5. 同一变道场景 slice：Stage7（事后描述性）", "",
        "Behavior Reference：`pdm_closed_conservative_v1 → pdm_closed_assertive_v1`；60个预处理scenario_type为 changing_lane 的固定场景。它是 lane-change **场景切片**，不自动证明ego完成了变道；semantic direction 因此保持限制性表述。", "",
        markdown_table(lane), "",
        "## 6. 业务 Style Report Card（Primary contrast：Conservative → Assertive）", "",
        "下表使用 B 作为当前最简单的 learned release-level candidate，用于让业务读者看到固定contrast中各切片的差异。它不是‘B优于所有representation’的证明，详见主矩阵。", "",
        markdown_table(style), "",
        "## 7. Representation gate 分拆（不再使用模糊 frozen_gate_result）", "",
        markdown_table(pd.DataFrame(scorecard)), "",
        "## 8. 确认性与事后描述性边界", "",
        "- **原预冻结确认性证据**：Stage6J/K dose-response paired 及 Stage6S-v3 interaction confirmation；它们的任务、样本、null与统计均沿用冻结输出。",
        "- **事后标准化描述性证据**：Stage7 old64/A/B/C/ego13 共用既有310对assertive/conservative rollout、固定pre-treatment task membership、primary seed3407及固定100,000次pair swap。其目的只是补齐同工况横向矩阵，不能取代Stage6V endpoint，也不能触发训练返工。",
        "- **unpaired release**：Stage6P属于representation scorecard，不被伪装成某个方向的行为画像；A/A FPR和detection仍单独解释。", "",
        "## 9. Evidence gaps / N/A", "",
        markdown_table(pd.DataFrame(gaps)) if gaps else "无固定维度证据缺口。", "",
        "## 10. 直接回答", "",
        "1. **同一跟车工况**：第3节列出old64/A/B/C/ego13逐一的 raw MMD²、null q95、ratio、Z、p与Holm；这是同一60对条件，允许比较各自相对null的检测强度，不允许比较raw MMD²大小。",
        "2. **同一纵向工况**：主矩阵的`纵向加速/减速`来自Stage6J/K dose100 overall；完整25/50/75/100与四个scope的逐representation行在`standardized_bdd_long.csv`中保留。",
        "3. **同一变道工况**：第5节给出Stage7固定60对场景slice的所有representation BDD。该证据是post-hoc descriptive，且不足以声称已验证ego executed lane-change差异。",
        "4. **interaction工况**：第4节给出Stage6S-v3相同80对的逐representation BDD；其轨迹机制已先行通过。C不具有相对于C-neighbor-zero的已证实增量interaction信息。",
        "5. **Reference定义**：每条长表行都分别携带behavior_reference、target、null_reference和representation baseline语境。",
        "6. **每维最可靠表示**：主矩阵给出按within-representation Z/检出描述的best capability；不构成universal representation排名。",
        "7. **结论边界**：Stage6J/K、Stage6S-v3是继承的确认性结果；Stage7全表示矩阵是事后描述性。",
        "8. **完整矩阵**：13个固定维度均已出现；无法支持的维度保持N/A并列出证据缺口。", "",
        "`STANDARDIZED_FIXED_DIMENSION_BDD_MATRIX_COMPLETE`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = read_json(protocol_path)
    preflight = load_preflight(protocol)
    if args.preflight_only:
        print(json.dumps({
            "status": "PREFLIGHT_PASS_STANDARDIZED_FIXED_DIMENSION_BDD_PROTOCOL",
            "protocol_sha256": sha256(protocol_path),
            "stage7_pair_count": int(len(preflight["stage7_pairs"])),
            "stage7_task_counts": preflight["stage7_task_counts"],
            "training_run": False,
            "simulation_run": False,
            "embedding_export_run": False,
        }, ensure_ascii=False, indent=2))
        return
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output_dir: {output_dir}")
    output_dir.mkdir(parents=True)
    reps, rep_checksums = build_stage7_representations(preflight, output_dir)
    stage7, stage7_audit = stage7_rows(protocol, preflight, reps, output_dir)
    rows = stage6jk_rows(protocol) + stage6s_rows(protocol) + stage7
    matrix, gaps = primary_matrix(protocol, rows)
    scorecard = gate_scorecard()
    write_csv(output_dir / "standardized_bdd_long.csv", rows)
    write_csv(output_dir / "fixed_dimension_primary_matrix.csv", matrix)
    write_csv(output_dir / "representation_gate_scorecard.csv", scorecard)
    write_csv(output_dir / "evidence_gap_matrix.csv", gaps)
    write_json(output_dir / "stage7_posthoc_standardized_audit.json", stage7_audit)
    report = build_report(protocol, matrix, rows, scorecard, gaps, stage7_audit)
    (output_dir / "standardized_fixed_dimension_bdd_evaluation_report_zh.md").write_text(report, encoding="utf-8")
    output_files = [
        "standardized_bdd_long.csv", "fixed_dimension_primary_matrix.csv", "representation_gate_scorecard.csv",
        "evidence_gap_matrix.csv", "stage7_posthoc_standardized_audit.json", "stage7_posthoc_null_samples.npz",
        "standardized_fixed_dimension_bdd_evaluation_report_zh.md",
    ]
    manifest = {
        "schema_version": "standardized_fixed_dimension_bdd_matrix_v1",
        "status": protocol["final_status_when_complete"],
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "unified_schema_sha256": sha256(SCHEMA),
        "source_contracts": protocol["source_contracts"],
        "stage7_representation_checkpoint_or_scaler_sha256": rep_checksums,
        "stage7_classification": protocol["source_contracts"]["stage7"]["classification"],
        "training_run": False,
        "simulation_run": False,
        "planner_modified": False,
        "checkpoint_modified": False,
        "scenario_selection_modified": False,
        "cross_representation_raw_mmd2_comparison_performed": False,
        "stage6v_joint_conclusion_modified": False,
        "row_count": len(rows),
        "matrix_dimension_count": len(matrix),
        "output_files": {name: sha256(output_dir / name) for name in output_files},
        "representation_files": {name: sha256(output_dir / "stage7_posthoc_representations" / f"{name}.npy") for name in REPS},
    }
    write_json(output_dir / "standardized_fixed_dimension_bdd_manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "output_dir": str(output_dir),
        "manifest_sha256": sha256(output_dir / "standardized_fixed_dimension_bdd_manifest.json"),
        "row_count": len(rows),
        "matrix_dimension_count": len(matrix),
        "stage7_post_hoc": True,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
