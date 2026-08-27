#!/usr/bin/env python3
"""Create read-only raw-trajectory evidence for the R1 Phase A draft.

The audit reads only predeclared ``ego_seq.npy`` and optional
``ego_seq_mask.npy`` files.  It never opens representations, BDD/probe results,
checkpoints, RBR files, planner outputs, or scenario-level outcome labels.  It
does not select individual scenarios: every valid frame in each declared source
contributes to the reported unstratified quantiles.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/stageR/r1/r1_phasea_raw_trajectory_evidence_v0.1.json"
DT_SECONDS = 0.1

SOURCES: tuple[dict[str, str | None], ...] = (
    {
        "source_id": "waymo_dynamic_v2_unstratified_development_raw",
        "purpose": "Raw measurement-scale reference; no residual-family or outcome filtering.",
        "ego_path": "outputs/stage6r_dynamic_full51_semantic_strict_part_00_09/shards/shard_000000/ego_seq.npy",
        "mask_path": None,
    },
    {
        "source_id": "r_tsb_stage6j_raw",
        "purpose": "Existing R0 DEVELOPMENT raw trajectory reference for longitudinal morphology.",
        "ego_path": "outputs/stage6j_pure_longitudinal_context_v1/ego_seq.npy",
        "mask_path": "outputs/stage6j_pure_longitudinal_context_v1/ego_seq_mask.npy",
    },
    {
        "source_id": "r_tsb_stage6k_dose25_raw",
        "purpose": "Existing R0 DEVELOPMENT raw trajectory reference for longitudinal morphology.",
        "ego_path": "outputs/stage6k_longitudinal_dose_context_v2_runtime_repaired/dose25/ego_seq.npy",
        "mask_path": "outputs/stage6k_longitudinal_dose_context_v2_runtime_repaired/dose25/ego_seq_mask.npy",
    },
    {
        "source_id": "r_hlc_stage7l_dose0_raw",
        "purpose": "Existing R0 DEVELOPMENT raw trajectory reference for lateral measurement scale only.",
        "ego_path": "outputs/stage7l_e_prospective_bdd_v1/contexts/dose0/ego_seq.npy",
        "mask_path": "outputs/stage7l_e_prospective_bdd_v1/contexts/dose0/ego_seq_mask.npy",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def quantile_summary(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("no finite raw trajectory values available")
    levels = (0.01, 0.05, 0.50, 0.95, 0.99)
    numbers = np.quantile(finite, levels)
    return {
        "n": int(finite.size),
        **{f"q{int(level * 100):02d}": round(float(value), 6) for level, value in zip(levels, numbers)},
    }


def source_summary(spec: dict[str, str | None]) -> dict[str, Any]:
    ego_path = ROOT / str(spec["ego_path"])
    mask_path = ROOT / str(spec["mask_path"]) if spec["mask_path"] else None
    if not ego_path.is_file():
        raise FileNotFoundError(f"missing declared raw trajectory source: {ego_path}")
    ego = np.load(ego_path, mmap_mode="r")
    if ego.ndim != 3 or ego.shape[-1] != 8:
        raise ValueError(f"expected raw ego trajectory [N,T,8], got {list(ego.shape)} at {ego_path}")
    if mask_path is None:
        valid = np.ones(ego.shape[:2], dtype=bool)
    else:
        if not mask_path.is_file():
            raise FileNotFoundError(f"missing declared validity mask: {mask_path}")
        valid = np.asarray(np.load(mask_path, mmap_mode="r"), dtype=bool)
        if valid.shape != ego.shape[:2]:
            raise ValueError(f"mask shape {list(valid.shape)} does not match ego prefix {list(ego.shape[:2])}")
    consecutive = valid[:, 1:] & valid[:, :-1]
    metrics = {
        "speed_mps": quantile_summary(ego[..., 5][valid]),
        "longitudinal_accel_mps2": quantile_summary(ego[..., 6][valid]),
        "lateral_velocity_mps": quantile_summary(ego[..., 3][valid]),
        "yaw_rate_radps": quantile_summary(ego[..., 7][valid]),
        "longitudinal_jerk_mps3": quantile_summary(((ego[:, 1:, 6] - ego[:, :-1, 6]) / DT_SECONDS)[consecutive]),
        "lateral_accel_mps2": quantile_summary(((ego[:, 1:, 3] - ego[:, :-1, 3]) / DT_SECONDS)[consecutive]),
    }
    return {
        "source_id": spec["source_id"],
        "purpose": spec["purpose"],
        "ego_path": str(spec["ego_path"]),
        "mask_path": spec["mask_path"],
        "shape": [int(value) for value in ego.shape],
        "valid_frame_count": int(valid.sum()),
        "consecutive_valid_pair_count": int(consecutive.sum()),
        "metrics": metrics,
    }


def write_new(path: Path, payload: dict[str, Any]) -> None:
    if path.resolve() != DEFAULT_OUTPUT.resolve():
        raise ValueError(f"R1 Phase A output is fixed to {DEFAULT_OUTPUT}")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    summaries = [source_summary(spec) for spec in SOURCES]
    payload = {
        "schema_version": "r1_phasea_raw_trajectory_evidence_v0.1",
        "status": "READ_ONLY_TREATMENT_INDEPENDENT_DEVELOPMENT_RAW_EVIDENCE",
        "sampling_interval_seconds": DT_SECONDS,
        "selection_rule": "all valid frames in every predeclared source; no scenario-, family-, or outcome-driven filtering",
        "forbidden_inputs_not_opened": ["embedding", "BDD", "probe", "checkpoint", "RBR", "detection", "planner rollout output"],
        "interpretation_limit": "Quantiles support only draft threshold/noise-sensitivity proposals; they do not freeze a mechanism rule or establish a scientific outcome.",
        "sources": summaries,
    }
    write_new(args.output, payload)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
