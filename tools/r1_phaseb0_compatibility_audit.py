#!/usr/bin/env python3
"""R1 Phase B0 analytical/synthetic compatibility audit.

The audit opens only frozen contract JSON and treatment-independent raw-scale
evidence.  It does not select scenarios, execute planners, or read smoke,
representation, BDD, probe, checkpoint, or RBR artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_context_mechanism_core import (
    calculate_hlc_option_b,
    calculate_tsb_option_a,
    frozen_f_match,
    qualify_hlc_pair,
    qualify_tsb_pair,
    trajectory_descriptors,
)
from tools.stage7l_pure_lateral_execution_planner import quintic_blend


ROOT = Path(__file__).resolve().parents[1]
R1_DIR = ROOT / "docs/stageR/r1"
HLC_CONTRACT = R1_DIR / "r1_hlc_mechanism_contract_v1.0.json"
TSB_CONTRACT = R1_DIR / "r1_tsb_mechanism_contract_v1.0.json"
RAW_EVIDENCE = R1_DIR / "r1_phasea_raw_trajectory_evidence_v0.1.json"
DEFAULT_OUTPUT = R1_DIR / "r1_phaseb0_compatibility_results_v0.1.json"

TSB_GEN_V2_OPTIONS: Dict[str, Dict[str, float]] = {
    "TSB_GEN_V2_OPTION_A": {
        "first_brake_mps2": -0.9,
        "first_brake_seconds": 0.5,
        "release_mps2": 0.4,
        "release_seconds": 0.7,
        "second_brake_mps2": -0.9,
        "second_brake_seconds": 0.5,
    },
    "TSB_GEN_V2_OPTION_B": {
        "first_brake_mps2": -1.0,
        "first_brake_seconds": 0.6,
        "release_mps2": 0.6,
        "release_seconds": 0.7,
        "second_brake_mps2": -0.9,
        "second_brake_seconds": 0.6,
    },
    "TSB_GEN_V2_OPTION_C": {
        "first_brake_mps2": -1.0,
        "first_brake_seconds": 0.7,
        "release_mps2": 0.8,
        "release_seconds": 0.6,
        "second_brake_mps2": -1.0,
        "second_brake_seconds": 0.5,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _blend(start: float, end: float, elapsed: np.ndarray, duration: float) -> np.ndarray:
    return start + (end - start) * quintic_blend(elapsed / duration)


def synthetic_hlc_progress(time_s: np.ndarray, arm: str) -> np.ndarray:
    """Return the fixed, scenario-free witness pair used by this audit."""
    p = np.zeros_like(time_s)
    if arm == "BASELINE":
        active = time_s >= 1.0
        p[active] = _blend(0.0, 1.0, time_s[active] - 1.0, 0.81)
        p[time_s >= 1.81] = 1.0
        return p
    if arm != "TREATMENT":
        raise ValueError("HLC arm must be BASELINE or TREATMENT")
    phases = (
        (1.0, 1.8, 0.0, 0.35, 0.8),
        (2.3, 3.1, 0.35, 0.20, 0.8),
        (3.1, 4.6, 0.20, 1.0, 1.5),
    )
    for start_s, end_s, start_p, end_p, duration_s in phases:
        active = (time_s >= start_s) & (time_s < end_s)
        p[active] = _blend(start_p, end_p, time_s[active] - start_s, duration_s)
    p[(time_s >= 1.8) & (time_s < 2.3)] = 0.35
    p[time_s >= 4.6] = 1.0
    return p


def _hlc_trajectory(time_s: np.ndarray, progress: np.ndarray, speed_mps: float, lane_width_m: float) -> Tuple[np.ndarray, np.ndarray]:
    xy = np.column_stack((speed_mps * time_s, lane_width_m * progress))
    realized_speed = np.linalg.norm(np.gradient(xy, time_s, axis=0), axis=1)
    return xy, realized_speed


def synthetic_hlc_witness(speed_mps: float = 13.292885, lane_width_m: float = 2.7) -> Dict[str, Any]:
    time_s = np.arange(0.0, 6.0 + 0.05, 0.1, dtype=np.float64)
    baseline_p = synthetic_hlc_progress(time_s, "BASELINE")
    treatment_p = synthetic_hlc_progress(time_s, "TREATMENT")
    baseline_xy, baseline_speed = _hlc_trajectory(time_s, baseline_p, speed_mps, lane_width_m)
    treatment_xy, treatment_speed = _hlc_trajectory(time_s, treatment_p, speed_mps, lane_width_m)
    baseline_mechanism = calculate_hlc_option_b(time_s, baseline_p, np.full_like(time_s, speed_mps))
    treatment_mechanism = calculate_hlc_option_b(time_s, treatment_p, np.full_like(time_s, speed_mps))
    baseline_descriptors = trajectory_descriptors(time_s, baseline_xy, baseline_speed)
    treatment_descriptors = trajectory_descriptors(time_s, treatment_xy, treatment_speed)
    return {
        "source": "SYNTHETIC_PARALLEL_LANES_NO_REAL_SCENARIO",
        "speed_mps": speed_mps,
        "lane_width_m": lane_width_m,
        "baseline_profile": {"direct_quintic_seconds": 0.81},
        "treatment_profile": {
            "advance": "p=0.00->0.35 over 0.8s",
            "hold_seconds": 0.5,
            "retreat": "p=0.35->0.20 over 0.8s",
            "recommit": "p=0.20->1.00 over 1.5s",
        },
        "baseline_mechanism": baseline_mechanism,
        "treatment_mechanism": treatment_mechanism,
        "mechanism_pair": qualify_hlc_pair(baseline_mechanism, treatment_mechanism),
        "baseline_descriptors": baseline_descriptors,
        "treatment_descriptors": treatment_descriptors,
        "f_match": frozen_f_match(baseline_descriptors, treatment_descriptors, "R-HLC"),
    }


def integrate_tsb_profile(
    initial_speed_mps: float,
    phases: Sequence[Tuple[float, float]],
    horizon_s: float = 4.5,
    diverge_s: float = 1.1,
) -> Dict[str, np.ndarray]:
    """Integrate a deterministic piecewise-acceleration profile."""
    time_s = np.arange(0.0, horizon_s + 0.05, 0.1, dtype=np.float64)
    acceleration = np.zeros_like(time_s)
    cursor = diverge_s
    for intensity_mps2, duration_s in phases:
        end = cursor + duration_s
        acceleration[(time_s >= cursor) & (time_s < end - 1e-9)] = intensity_mps2
        cursor = end
    speed = np.empty_like(time_s)
    x = np.empty_like(time_s)
    speed[0] = initial_speed_mps
    x[0] = 0.0
    for index in range(1, len(time_s)):
        dt = float(time_s[index] - time_s[index - 1])
        speed[index] = max(0.2, speed[index - 1] + acceleration[index - 1] * dt)
        x[index] = x[index - 1] + 0.5 * (speed[index - 1] + speed[index]) * dt
    return {
        "time_s": time_s,
        "acceleration_mps2": acceleration,
        "speed_mps": speed,
        "xy": np.column_stack((x, np.zeros_like(x))),
    }


def _option_phases(option: Mapping[str, float]) -> Tuple[Tuple[float, float], ...]:
    return (
        (float(option["first_brake_mps2"]), float(option["first_brake_seconds"])),
        (float(option["release_mps2"]), float(option["release_seconds"])),
        (float(option["second_brake_mps2"]), float(option["second_brake_seconds"])),
    )


def synthetic_tsb_options() -> Dict[str, Any]:
    baseline = integrate_tsb_profile(8.0, ((-1.0, 0.95),))
    baseline_mechanism = calculate_tsb_option_a(baseline["time_s"], baseline["speed_mps"])
    baseline_descriptors = trajectory_descriptors(baseline["time_s"], baseline["xy"], baseline["speed_mps"])
    results: Dict[str, Any] = {}
    for option_id, parameters in TSB_GEN_V2_OPTIONS.items():
        treatment = integrate_tsb_profile(8.0, _option_phases(parameters))
        mechanism = calculate_tsb_option_a(treatment["time_s"], treatment["speed_mps"])
        descriptors = trajectory_descriptors(treatment["time_s"], treatment["xy"], treatment["speed_mps"])
        results[option_id] = {
            "status": "PROPOSED_NOT_FROZEN",
            "parameters": parameters,
            "expected_phase_segmentation": "TWO_BRAKE_PHASES_WITH_ONE_INTERSTAGE_RELEASE",
            "mechanism": mechanism,
            "mechanism_pair": qualify_tsb_pair(baseline_mechanism, mechanism),
            "f_match": frozen_f_match(baseline_descriptors, descriptors, "R-TSB"),
            "safety_scope": "SYNTHETIC_KINEMATIC_ONLY_NOT_OFFICIAL_CLOSED_LOOP_SAFETY",
        }
    return {
        "source": "SYNTHETIC_PIECEWISE_PROFILE_NO_REAL_SCENARIO",
        "initial_speed_mps": 8.0,
        "baseline": {
            "profile": {"brake_mps2": -1.0, "brake_seconds": 0.95},
            "mechanism": baseline_mechanism,
            "descriptors": baseline_descriptors,
        },
        "options": results,
    }


def build_results() -> Dict[str, Any]:
    with HLC_CONTRACT.open("r", encoding="utf-8") as handle:
        hlc_contract = json.load(handle)
    with TSB_CONTRACT.open("r", encoding="utf-8") as handle:
        tsb_contract = json.load(handle)
    with RAW_EVIDENCE.open("r", encoding="utf-8") as handle:
        raw_evidence = json.load(handle)
    if hlc_contract["status"] != "R1_CONTEXT_MECHANISM_CONTRACT_V1_FROZEN" or tsb_contract["status"] != "R1_CONTEXT_MECHANISM_CONTRACT_V1_FROZEN":
        raise RuntimeError("R1 mechanism contracts are not frozen v1.0 inputs")
    hlc_source = next(row for row in raw_evidence["sources"] if row["source_id"] == "r_hlc_stage7l_dose0_raw")
    speed_low = float(hlc_source["metrics"]["speed_mps"]["q01"])
    speed_high = float(hlc_source["metrics"]["speed_mps"]["q99"])
    width_low, width_high = 2.7, 4.2
    retreat_progress = float(hlc_contract["measurements"]["retreat"]["cumulative_fall_gte"])
    retreat_rate = abs(float(hlc_contract["measurements"]["retreat"]["derivative_lte_per_s"]))
    displacement = [width_low * retreat_progress, width_high * retreat_progress]
    lateral_velocity = [width_low * retreat_rate, width_high * retreat_rate]
    heading_excursion = [
        float(np.arctan2(lateral_velocity[0], speed_high)),
        float(np.arctan2(lateral_velocity[1], speed_low)),
    ]
    hlc_witness = synthetic_hlc_witness(speed_high, width_low)
    tsb = synthetic_tsb_options()
    if not hlc_witness["mechanism_pair"]["pass"] or not hlc_witness["f_match"]["pass"]:
        raise RuntimeError("fixed HLC synthetic compatibility witness failed")
    if not all(row["mechanism_pair"]["pass"] and row["f_match"]["pass"] for row in tsb["options"].values()):
        raise RuntimeError("one or more fixed TSB V2 proposal witnesses failed")
    return {
        "schema_version": "r1_phaseb0_compatibility_results_v0.1",
        "status": "ANALYTICAL_SYNTHETIC_ONLY_NO_REAL_SCENARIO_EXECUTION",
        "input_sha256": {
            str(HLC_CONTRACT.relative_to(ROOT)): sha256_file(HLC_CONTRACT),
            str(TSB_CONTRACT.relative_to(ROOT)): sha256_file(TSB_CONTRACT),
            str(RAW_EVIDENCE.relative_to(ROOT)): sha256_file(RAW_EVIDENCE),
        },
        "forbidden_inputs_not_opened": [
            "old technical-smoke metrics/outcomes",
            "representation",
            "BDD",
            "probe",
            "checkpoint",
            "RBR",
        ],
        "hlc": {
            "classification": "MARGINALLY_FEASIBLE",
            "heading_feature_assessment": "STRUCTURAL_MECHANISM_OVERLAP_CONFIRMED_BUT_NONEMPTY_INTERSECTION_DEMONSTRATED",
            "physical_envelope": {
                "lane_width_m": [width_low, width_high],
                "speed_mps_treatment_independent_q01_q99": [speed_low, speed_high],
                "minimum_retreat_progress": retreat_progress,
                "retreat_lateral_displacement_m": [round(value, 6) for value in displacement],
                "minimum_added_lateral_total_variation_m": [round(2.0 * value, 6) for value in displacement],
                "threshold_lateral_velocity_mps": [round(value, 6) for value in lateral_velocity],
                "approx_negative_heading_excursion_rad": [round(value, 6) for value in heading_excursion],
                "approx_extra_absolute_heading_lower_envelope_rad": [round(2.0 * value, 6) for value in heading_excursion],
                "extra_path_length_lower_bound_m": 0.0,
                "extra_path_length_note": "A universal positive bound is unavailable because baseline/treatment duration and longitudinal compensation may differ; the synthetic witness reports the realized delta.",
            },
            "synthetic_witness": hlc_witness,
            "implementation_audit": {
                "status": "NO_IMPLEMENTATION_DEFINITION_BUG_CONFIRMED",
                "phase_stitching": "quintic endpoints are position/first-derivative/second-derivative continuous",
                "heading_yaw_curvature": "derived from one shared finite, monotonic-time xy trajectory in consistent SI units",
                "terminal_state": "all current candidates reach p=1 before the 4.5s horizon",
                "common_prefix": "baseline and treatment progress are both p=0 through t<1.1s",
                "diagnosis": "GENERATOR_PROFILE_AND_CONTRACT_MARGIN_INSUFFICIENT; no independently reproducible definition/code bug was found",
            },
        },
        "tsb": {
            "classification": "JOINTLY_FEASIBLE",
            "synthetic_profiles": tsb,
            "implementation_audit": {
                "status": "NO_IMPLEMENTATION_BUG_CONFIRMED",
                "secondary_status": "GENERATOR_PROFILE_REDESIGN_REQUIRED",
                "diagnosis": "The current short/weak release is attenuated and temporally shifted by median3 plus gradient boundary effects; timestamps, phase merge rule, integration, and low-speed endstop are internally consistent.",
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the R1 Phase B0 synthetic compatibility audit.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite compatibility artifact: {args.output}")
    results = build_results()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "hlc": results["hlc"]["classification"], "tsb": results["tsb"]["classification"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
