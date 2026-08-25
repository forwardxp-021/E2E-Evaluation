#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import re

IDM_PARAMETERS = {
    "idm_longitudinal_conservative": {"target_velocity": 8.0, "min_gap_to_lead_agent": 2.0, "headway_time": 2.0, "accel_max": 0.8, "decel_max": 2.5},
    "idm_longitudinal_comfort": {"target_velocity": 10.0, "min_gap_to_lead_agent": 1.5, "headway_time": 1.5, "accel_max": 1.0, "decel_max": 3.0},
    "idm_longitudinal_aggressive": {"target_velocity": 12.0, "min_gap_to_lead_agent": 0.5, "headway_time": 1.0, "accel_max": 1.5, "decel_max": 4.0},
}
PARAM_UNITS = {"target_velocity": "m/s", "min_gap_to_lead_agent": "m", "headway_time": "s", "accel_max": "m/s^2", "decel_max": "m/s^2"}


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(name))


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def idm_parameter_markdown(planner_a: str = "idm_longitudinal_aggressive", planner_b: str = "idm_longitudinal_conservative") -> str:
    lines = [
        "## IDM parameter definitions",
        "",
        "These are nominal planner parameter differences. The analysis tests whether nominal parameter differences produce realized rollout differences.",
        "",
    ]
    for planner in ["idm_longitudinal_conservative", "idm_longitudinal_comfort", "idm_longitudinal_aggressive"]:
        vals = IDM_PARAMETERS[planner]
        short = planner.replace("idm_longitudinal_", "")
        lines += [f"### {short}", ""]
        for key, val in vals.items():
            lines.append(f"* {key} = {val:.1f} {PARAM_UNITS[key]}")
        lines.append("")
    if planner_a in IDM_PARAMETERS and planner_b in IDM_PARAMETERS:
        lines += [f"### {planner_a} - {planner_b} parameter differences", ""]
        a, b = IDM_PARAMETERS[planner_a], IDM_PARAMETERS[planner_b]
        for key in ["target_velocity", "min_gap_to_lead_agent", "headway_time", "accel_max", "decel_max"]:
            diff = a[key] - b[key]
            pct = diff / b[key] * 100.0 if abs(b[key]) > 1e-12 else float("nan")
            lines.append(f"* {key}: {diff:+.1f} {PARAM_UNITS[key]}, {pct:+.1f}%")
        lines.append("")
    return "\n".join(lines)


def find_planner_index(stage7f_dir: Path, planner: str) -> Path:
    idx_dir = stage7f_dir / "planner_indices"
    direct = idx_dir / f"{safe_name(planner)}.npy"
    if direct.exists():
        return direct
    matches = sorted(idx_dir.glob("*.npy")) if idx_dir.exists() else []
    for p in matches:
        if p.stem == planner:
            return p
    raise FileNotFoundError(f"Missing planner index for {planner}: expected {direct}; available={[p.name for p in matches]}")
