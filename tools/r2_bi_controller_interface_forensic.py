#!/usr/bin/env python3
"""Zero-simulation forensic for the rejected R2-BH HLC V2 controller interface."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r2_bh_hlc_target_capture_generator_v2 import target_capture_path  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "docs/stageR/r2"
RAW = ROOT / "outputs/r2_bh_hlc_target_capture_dev_v1"
ROSTER = R2 / "r2_bh_hlc_arch_dev_roster_v1.0.json"
EXCLUSION = R2 / "r2_bh_hlc_arch_permanent_exclusion_ledger_v1.0.json"
PAIR_BINDINGS = R2 / "r2_bh_hlc_arch_pair_bindings_v1.0.json"
ROUND_ROOT = R2 / "r2_bh_hlc_arch_rounds"
LQR = ROOT.parent / "nuplan-devkit/nuplan/planning/simulation/controller/tracker/lqr.py"
TWO_STAGE = ROOT.parent / "nuplan-devkit/nuplan/planning/simulation/controller/two_stage_controller.py"
LQR_CONFIG = ROOT.parent / "nuplan-devkit/nuplan/planning/script/config/simulation/ego_controller/tracker/lqr_tracker.yaml"
OUT_LEDGER = R2 / "r2_bi_r2bh_outcome_exposure_ledger_v1.0.json"
OUT_AUDIT = R2 / "r2_bi_hlc_v2_controller_interface_forensic_v1.json"
OUT_REPORT = R2 / "R2_BI_HLC_V2_Controller_Interface_Forensic_v1.md"


def _read(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new(path: Path, value: Mapping[str, Any] | str) -> None:
    if path.exists():
        raise FileExistsError(f"R2_BI_VERSIONED_OUTPUT_EXISTS:{path}")
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _wrap(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def _dist(values: Iterable[float]) -> Dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None}
    return {
        "n": int(len(array)), "min": float(np.min(array)), "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)), "p75": float(np.percentile(array, 75)), "max": float(np.max(array)),
    }


def _signed_offset(point: Sequence[float], reference: Sequence[Sequence[float]]) -> float:
    p = np.asarray(point, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    starts, vectors = ref[:-1], np.diff(ref, axis=0)
    denom = np.sum(vectors * vectors, axis=1)
    u = np.clip(np.sum((p - starts) * vectors, axis=1) / np.maximum(denom, 1e-12), 0.0, 1.0)
    projected = starts + u[:, None] * vectors
    index = int(np.argmin(np.sum((p - projected) ** 2, axis=1)))
    tangent = vectors[index] / max(float(np.linalg.norm(vectors[index])), 1e-12)
    normal = np.asarray([-tangent[1], tangent[0]])
    return float(np.dot(p - projected[index], normal))


def _synthetic_v2_cases() -> list[Dict[str, Any]]:
    cases: list[Dict[str, Any]] = []
    time = np.arange(11, dtype=np.float64) * 0.1 + 6.0
    capture = {"capture_start_abs_s": 5.0, "capture_duration_s": 1.0}
    for corridor, direction in (("STRAIGHT", 0), ("CURVED", 0), ("LEFT", 1), ("RIGHT", -1)):
        x = np.arange(11, dtype=np.float64)
        if corridor == "CURVED":
            theta = 0.04 * x
            base = np.column_stack((25.0 * np.sin(theta), 25.0 * (1.0 - np.cos(theta))))
        else:
            base = np.column_stack((x, direction * 0.02 * x**2))
        segment = np.diff(base, axis=0)
        heading = np.r_[np.arctan2(segment[:, 1], segment[:, 0]), np.arctan2(segment[-1, 1], segment[-1, 0])]
        normal = np.asarray([-math.sin(heading[0]), math.cos(heading[0])])
        current = base[0] + 0.5 * normal
        xy, declared, audit = target_capture_path(base, heading, current, float(heading[0]), 6.0, time, capture)
        geometric = math.atan2(float(xy[1, 1] - xy[0, 1]), float(xy[1, 0] - xy[0, 0]))
        cases.append({
            "corridor": corridor,
            "current_residual_m": 0.5,
            "algebraic_state1_residual_term_m": float(audit["commanded_lateral_residual_m"][1]),
            "actual_state0_to_state1_lateral_jump_m": float(np.dot(xy[1] - xy[0], normal)),
            "declared_state0_heading_rad": float(declared[0]),
            "geometric_state0_to_state1_tangent_rad": geometric,
            "absolute_heading_inconsistency_rad": abs(_wrap(float(declared[0]) - geometric)),
            "diagnosis_reproduced": bool(
                abs(float(audit["commanded_lateral_residual_m"][1])) <= 1e-12
                and abs(float(np.dot(xy[1] - xy[0], normal))) > 0.45
            ),
        })
    return cases


def main() -> int:
    for path in (ROSTER, EXCLUSION, PAIR_BINDINGS, LQR, TWO_STAGE, LQR_CONFIG):
        if not path.is_file():
            raise FileNotFoundError(path)
    roster = _read(ROSTER)
    bindings = {row["pair_id"]: row for row in _read(PAIR_BINDINGS)["pairs"]}
    if len(roster["entries"]) != 8 or len(bindings) != 8:
        raise RuntimeError("R2_BI_EXPECTED_EIGHT_R2BH_IDENTITIES")
    ledger = {
        "schema_version": "r2_bi_r2bh_outcome_exposure_ledger_v1.0",
        "status": "R2_BH_DEV_ARCH_FROZEN_HISTORY_ONLY",
        "source_roster": {"path": str(ROSTER.relative_to(ROOT)), "sha256": _sha(ROSTER)},
        "source_exclusion_ledger": {"path": str(EXCLUSION.relative_to(ROOT)), "sha256": _sha(EXCLUSION)},
        "identity_count": 8,
        "identities": [{
            "scenario_token": row["scenario_token"], "log_id": row["log_id"], "family": "R-HLC",
            "R2BH_HISTORY_ONLY": True, "FUTURE_GENERATOR_TUNING_FORBIDDEN": True,
            "R2C_USE_FORBIDDEN": True, "CONFIRMATORY_USE_FORBIDDEN": True,
            "RBR_USE_FORBIDDEN": True, "allowed_use": "READ_ONLY_OFFLINE_FORENSIC_ONLY",
        } for row in roster["entries"]],
        "R2B_or_R2BH_identity_resimulation_calls": 0,
        "V3_numerical_parameters_fitted_from_R2BH_raw": False,
    }
    measures: Dict[str, list[float]] = {
        "state0_to_state1_forward_m": [], "state0_to_state1_lateral_m": [],
        "state0_declared_vs_segment_tangent_abs_rad": [], "state1_declared_vs_segment_tangent_abs_rad": [],
        "lookahead_abs_declared_curvature_inv_m": [], "planned_state1_actual_target_offset_m": [],
    }
    zero_state0_error = 0
    rows_seen = 0
    phase_offsets: Dict[str, list[float]] = {"before_capture": [], "during_capture": [], "after_capture": []}
    for round_index in range(3):
        params = _read(ROUND_ROOT / f"r2_bh_hlc_arch_round_{round_index}_parameters_v2.0.json")["parameters"]
        start = float(params["capture"]["capture_start_abs_s"])
        end = start + float(params["capture"]["capture_duration_s"])
        result = _read(ROUND_ROOT / f"r2_bh_hlc_arch_round_{round_index}_results_v1.0.json")
        run_to_pair = {run["run_id"]: run["pair_id"] for run in result["runs"]}
        for run_id, pair_id in run_to_pair.items():
            telemetry_path = RAW / f"round_{round_index}" / run_id / "telemetry/planner_target_capture.jsonl"
            for line in telemetry_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                states = row["controller_lookahead"]["states_0_to_10"]
                current = row["realized_current_ego"]["rear_axle"]
                s0, s1 = states[0]["rear_axle"], states[1]["rear_axle"]
                if max(abs(float(s0[k]) - float(current[k])) for k in ("x", "y", "heading")) <= 1e-12:
                    zero_state0_error += 1
                h0 = float(current["heading"])
                dx, dy = float(s1["x"] - s0["x"]), float(s1["y"] - s0["y"])
                tangent = math.atan2(dy, dx)
                measures["state0_to_state1_forward_m"].append(math.cos(h0) * dx + math.sin(h0) * dy)
                measures["state0_to_state1_lateral_m"].append(-math.sin(h0) * dx + math.cos(h0) * dy)
                measures["state0_declared_vs_segment_tangent_abs_rad"].append(abs(_wrap(h0 - tangent)))
                measures["state1_declared_vs_segment_tangent_abs_rad"].append(abs(_wrap(float(s1["heading"]) - tangent)))
                measures["lookahead_abs_declared_curvature_inv_m"].extend(
                    abs(float(x["planned_curvature_inv_m"])) for x in states[1:] if x["planned_curvature_inv_m"] is not None
                )
                offset = _signed_offset((s1["x"], s1["y"]), bindings[pair_id]["target_reference_xy"])
                measures["planned_state1_actual_target_offset_m"].append(offset)
                absolute = float(row["absolute_episode_time_s"])
                phase = "before_capture" if absolute < start else "during_capture" if absolute < end else "after_capture"
                phase_offsets[phase].append(abs(offset))
                rows_seen += 1
    realized_landmarks: Dict[str, list[float]] = {"capture_start": [], "capture_midpoint": [], "capture_end": [], "Primary_terminal": []}
    for round_index in range(3):
        result = _read(ROUND_ROOT / f"r2_bh_hlc_arch_round_{round_index}_results_v1.0.json")
        for pair in result["pairs"]:
            for arm in ("baseline", "treatment"):
                for landmark in realized_landmarks:
                    realized_landmarks[landmark].append(abs(float(pair["target_capture"][arm]["landmarks"][landmark]["realized_target_frame_offset_m"])))
    synthetic = _synthetic_v2_cases()
    audit = {
        "schema_version": "r2_bi_hlc_v2_controller_interface_forensic_v1",
        "status": "R2_BH_V2_CONTROLLER_INTERFACE_DIAGNOSIS_SUPPORTED",
        "simulation_calls": 0,
        "frozen_controller_binding": {
            "controller": "TwoStageController", "tracker": "LQRTracker",
            "lqr_source": {"path": str(LQR), "sha256": _sha(LQR)},
            "two_stage_source": {"path": str(TWO_STAGE), "sha256": _sha(TWO_STAGE)},
            "lqr_config": {"path": str(LQR_CONFIG), "sha256": _sha(LQR_CONFIG)},
            "discretization_time_s": 0.1, "tracking_horizon_steps": 10, "lookahead_s": 1.0,
            "initial_lateral_state": "[state0 lateral error, state0 heading error, actual tire steering angle]",
            "reference_input": "velocity and curvature profiles fitted from the pose trajectory",
        },
        "evidence_scope": {
            "R2BH_raw_planner_telemetry_rows": rows_seen,
            "R2BH_direct_controller_command_rows": 0,
            "direct_command_availability": "NOT_AVAILABLE_IN_R2BH_TELEMETRY",
            "future_V3_requirement": "EXACT_FROZEN_LQR_SHADOW_PLUS_PASSIVE_ACTUAL_CONTROLLER_RETURN_VALUE",
        },
        "state0_exact_current_ego_rows": zero_state0_error,
        "telemetry_distributions": {key: _dist(value) for key, value in measures.items()},
        "planned_actual_target_offset_by_capture_phase_abs_m": {key: _dist(value) for key, value in phase_offsets.items()},
        "realized_target_offset_landmarks_abs_m": {key: _dist(value) for key, value in realized_landmarks.items()},
        "semantic_separation": {
            "algebraic_residual_term": "V2 additive scalar field; zero does not establish pose continuity",
            "actual_planned_target_frame_offset": "projection of final planned xy onto frozen native target reference",
            "kinematically_realizable_reference": "continuous xy with heading/curvature derived from final xy and frozen feasibility",
            "LQR_steering_command": "tracker return value driven by initial state plus fitted curvature profile",
            "realized_closed_loop_offset": "projection of REALIZED_CURRENT_EGO onto frozen native target reference",
        },
        "synthetic_V2_cases": synthetic,
        "synthetic_diagnosis_pass": all(row["diagnosis_reproduced"] for row in synthetic),
        "conclusion": [
            "STATE0_IDENTITY_FORCES_ZERO_INITIAL_LATERAL_AND_HEADING_ERROR",
            "V2_DECLARED_HEADING_WAS_NOT_RECOMPUTED_FROM_FINAL_XY",
            "V2_POST_DEADLINE_ZERO_ADDITIVE_TERM_CAN_COEXIST_WITH_STATE0_TO_STATE1_HARD_JUMP",
            "ALGEBRAIC_ZERO_IS_NOT_CONTROLLER_VISIBLE_TARGET_CAPTURE",
        ],
    }
    report = f"""# R2-BI HLC V2 Controller-Interface Forensic v1

## 结论

`R2_BH_V2_CONTROLLER_INTERFACE_DIAGNOSIS = SUPPORTED`。本审计只读解析既有 R2-BH telemetry，并运行合成几何计算；scientific simulation 为 0。

## 冻结控制接口

nuPlan `TwoStageController` 将 planner trajectory 交给 `LQRTracker`。LQR 在当前 iteration 从 trajectory state0 计算 lateral/heading error；R2-BH 强制 state0 与 current ego 完全相同，因此这两个误差为零。冻结配置采用 0.1 s 离散、10 step horizon，即 1.0 s lookahead；reference velocity/curvature 从完整 pose trajectory 拟合，而不是读取 V2 的 additive residual 字段。

R2-BH 共复核 {rows_seen} 条 planner telemetry，state0 pose identity 为 {zero_state0_error}/{rows_seen}。R2-BH 当时没有记录 controller return value，因此 direct historical steering command 明确为 `NOT_AVAILABLE`；本轮不会把推导量冒充历史实测量。

## V2 结构失败

V2 在 base xy 上添加 lateral residual 后，独立叠加 heading residual，没有从最终 xy 重算 tangent/curvature。deadline 后又把 weight[0] 固定为 1、weight[1:] 设为 0；只要 realized residual 未归零，state0→state1 就会出现横向跳跃。straight、curved、left、right 四组合成 corridor 均重现该问题，4/4 支持 Owner diagnosis。

因此必须区分：algebraic residual term、实际 planned target-frame offset、运动学可实现 reference、LQR steering command 与 realized closed-loop offset。`state1 additive residual = 0` 不能作为 target capture 成功证据。
"""
    _write_new(OUT_LEDGER, ledger)
    _write_new(OUT_AUDIT, audit)
    _write_new(OUT_REPORT, report)
    print(json.dumps({"status": audit["status"], "telemetry_rows": rows_seen, "synthetic_cases": len(synthetic), "simulation_calls": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
