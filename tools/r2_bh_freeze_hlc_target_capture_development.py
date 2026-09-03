#!/usr/bin/env python3
"""Freeze R2-BH target-capture diagnosis, fresh DEV-ARCH roster, and contracts."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_b2_8_r3_prospective_selector import official_env  # noqa: E402
from tools.r1_b2_9_d_freeze_pair_bindings import _one as freeze_pair_binding  # noqa: E402
from tools.r1_future_compliant_smoke_selector_v1_3 import canonical_sha  # noqa: E402
from tools.r2_a_freeze_controller_id_design import _select_unique_family_suffix  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "docs/stageR/r2"
R2B_ROSTER = R2 / "r2_b_generator_calibration_roster_v1.0.json"
R2B_EXCLUSION = R2 / "r2_b_generator_calibration_permanent_exclusion_ledger_v1.0.json"
R2B_TSB_PARAMS = R2 / "r2_b_calibration_rounds/r2_b_tsb_round_0_parameters_v1.0.json"
R2B_TSB_RESULT = R2 / "r2_b_calibration_rounds/r2_b_tsb_round_0_results_v1.0.json"
R2B_GENERATOR = ROOT / "tools/r2_b_controller_aware_generator_v1.py"
PROTECTED = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"
PROTECTED_SHA = "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8"

OUT = {
    "exposure": R2 / "r2_bh_r2b_hlc_outcome_exposure_ledger_v1.0.json",
    "tsb": R2 / "r2_bh_tsb_family_development_candidate_v1.0.json",
    "forensic": R2 / "R2_BH_HLC_Target_Capture_Architecture_Forensic_v1.md",
    "invariant": R2 / "r2_bh_hlc_v1_reanchor_invariant_audit_v1.json",
    "roster": R2 / "r2_bh_hlc_arch_dev_roster_v1.0.json",
    "exclusion": R2 / "r2_bh_hlc_arch_permanent_exclusion_ledger_v1.0.json",
    "pairs": R2 / "r2_bh_hlc_arch_pair_bindings_v1.0.json",
    "contract": R2 / "r2_bh_hlc_architecture_contract_v2.0.json",
    "space": R2 / "r2_bh_hlc_arch_parameter_space_v2.0.json",
    "ledger": R2 / "r2_bh_hlc_arch_run_ledger_v1.0.json",
    "authorization": R2 / "r2_bh_scientific_owner_engineering_authorization_v1.0.json",
}


def read(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: Mapping[str, Any] | str) -> None:
    if path.exists():
        raise FileExistsError(f"R2_BH_VERSIONED_OUTPUT_EXISTS:{path}")
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def reanchor_invariant_audit() -> Dict[str, Any]:
    x = np.linspace(0.0, 20.0, 21)
    source = np.column_stack((x, np.zeros_like(x)))
    target = np.column_stack((x, np.full_like(x, 3.5)))
    cases = []
    for offset in (0.0, 0.25, 0.50, -0.25, -0.50):
        origin = np.asarray([0.0, 3.5 + offset])
        progress = np.ones_like(x)
        xy_before = source * (1.0 - progress[:, None]) + target * progress[:, None]
        translation = origin - xy_before[0]
        xy_v1 = xy_before + translation
        terminal_offset = float(xy_v1[-1, 1] - target[-1, 1])
        cases.append({
            "current_target_frame_lateral_offset_m": offset,
            "constant_reanchor_translation_xy_m": translation.tolist(),
            "planned_terminal_target_offset_m": terminal_offset,
            "offset_preserved_exactly": bool(abs(terminal_offset - offset) < 1e-12),
        })
    passed = all(row["offset_preserved_exactly"] for row in cases)
    return {
        "schema_version": "r2_bh_hlc_v1_reanchor_invariant_audit_v1",
        "status": "V1_REANCHOR_DIAGNOSIS_SUPPORTED" if passed else "V1_REANCHOR_DIAGNOSIS_NOT_SUPPORTED",
        "equation": "xy_v1(s)=source(s)*(1-p(s))+target(s)*p(s)+current_xy-xy_before(0)",
        "progress_fixed_for_test": 1.0,
        "native_parallel_reference_separation_m": 3.5,
        "cases": cases,
        "invariant": "terminal_target_offset_equals_current_target_frame_offset",
        "diagnosis_pass": passed,
        "simulation_calls": 0,
    }


def architecture_contract() -> Dict[str, Any]:
    return {
        "schema_version": "r2_bh_hlc_architecture_contract_v2.0",
        "status": "FROZEN_BEFORE_R2_BH_ENGINEERING_SIMULATION",
        "architecture": "BEHAVIOR_MORPHOLOGY_PLUS_FIXED_ABSOLUTE_TARGET_CAPTURE",
        "behavior_morphology": {
            "scientific_measurement": "FROZEN_REALIZED_NATIVE_SOURCE_TARGET_PROGRESS_P_OF_T",
            "stages": ["advance", "hold", "retreat", "recommit"],
            "thresholds_modified": False,
        },
        "target_capture": {
            "scientific_measurement": False,
            "frame": "OFFICIAL_NATIVE_TARGET_REFERENCE_FRAME",
            "clock": "ABSOLUTE_EPISODE_TIME",
            "lateral_rule": "e_y_future=e_y_current*w_abs(t_future)/w_abs(t_current); state0 exact; w_abs(capture_end)=0",
            "heading_rule": "e_heading_future=e_heading_current*w_abs(t_future)/w_abs(t_current); state0 exact; w_abs(capture_end)=0",
            "shape": "QUINTIC_C2",
            "after_capture_end": "STATE0_EXACT_CURRENT_EGO_AND_STATE1_PLUS_ZERO_TARGET_FRAME_RESIDUAL_COMMAND",
        },
        "geometry": {
            "source": "ROUTE_CONTINUOUS_V2_3_OFFICIAL_NATIVE_CORRIDOR",
            "no_extrapolation": True,
            "manual_centerline": False,
            "same_route_progression_invariant": True,
        },
        "state0_exact_current_ego": True,
        "pre_divergence": {"t_lt_s": 1.1, "baseline_treatment_full_trajectory_identical": True},
        "global_parameters_only": True,
        "scenario_token_or_log_lookup_forbidden": True,
        "maximum_rounds": 3,
        "R2B_round5": False,
    }


def parameter_space() -> Dict[str, Any]:
    return {
        "schema_version": "r2_bh_hlc_arch_parameter_space_v2.0",
        "status": "FROZEN_BEFORE_FIRST_R2_BH_SIMULATION",
        "maximum_rounds": 3,
        "round0_is_new_target_capture_architecture": True,
        "initialization": {
            "morphology": "PRE_OUTCOME_R2_A_SURROGATE_INITIALIZATION_REUSED_AS_ARCHITECTURE_START_ONLY",
            "capture": "STRUCTURAL_V1_REANCHOR_FORENSIC_AND_FIXED_PRIMARY80_TIME_BUDGET",
            "R2B_HLC_pair_outcomes_used_for_numerical_calibration": False,
        },
        "bounds": {
            "morphology": {
                "baseline_transition_duration_s": [2.2, 3.0],
                "advance_duration_s": [0.9, 1.3],
                "advance_progress": [0.40, 0.50],
                "hold_duration_s": [0.3, 0.7],
                "retreat_depth": [0.28, 0.46],
                "retreat_duration_s": [1.1, 1.7],
                "recommit_duration_s": [1.2, 1.7],
                "lag_precompensation_s": [0.2, 0.4]
            },
            "capture": {
                "capture_start_abs_s": [5.0, 5.5],
                "capture_duration_s": [1.0, 1.6]
            }
        },
        "round0": {
            "morphology": {
                "baseline_transition_duration_s": 2.6,
                "advance_duration_s": 1.1,
                "advance_progress": 0.44,
                "hold_duration_s": 0.5,
                "retreat_depth": 0.30,
                "retreat_duration_s": 1.3,
                "recommit_duration_s": 1.4,
                "lag_precompensation_s": 0.30
            },
            "capture": {
                "capture_start_abs_s": 5.4,
                "capture_duration_s": 1.4,
                "lateral_offset_decay_shape": "QUINTIC_C2",
                "heading_error_decay_shape": "QUINTIC_C2"
            }
        },
        "deterministic_aggregate_update": {
            "mechanism_not_all": {"retreat_depth_add": 0.06, "retreat_duration_add_s": 0.15},
            "endpoint_not_all": {"capture_duration_subtract_s": 0.20},
            "endpoint_offset_not_all": {"capture_start_abs_s_subtract": 0.20},
            "engineering_not_all": {"recommit_duration_add_s": 0.10},
            "clip_all_to_frozen_bounds": True,
            "identity_specific_update": False,
        },
    }


def main() -> int:
    if any(path.exists() for path in OUT.values()):
        raise FileExistsError("R2_BH_VERSIONED_OUTPUT_ALREADY_EXISTS")
    if sha(PROTECTED) != PROTECTED_SHA:
        raise PermissionError("PROTECTED_CSV_SHA_MISMATCH")
    audit = reanchor_invariant_audit()
    if not audit["diagnosis_pass"]:
        write_new(OUT["invariant"], audit)
        raise RuntimeError("R2_BH_V1_REANCHOR_DIAGNOSIS_NOT_SUPPORTED_STOP")
    r2b_roster, prior_exclusion = read(R2B_ROSTER), read(R2B_EXCLUSION)
    hlc_history = [row for row in r2b_roster["entries"] if row["family"] == "R-HLC"]
    exposure = {
        "schema_version": "r2_bh_r2b_hlc_outcome_exposure_ledger_v1.0",
        "status": "R2_B_HLC_DEV_CAL_FROZEN_ARCHITECTURE_DIAGNOSTIC_ONLY",
        "source_roster": {"path": str(R2B_ROSTER.relative_to(ROOT)), "sha256": sha(R2B_ROSTER)},
        "identity_count": 8,
        "identities": [{
            "scenario_token": row["scenario_token"], "log_id": row["log_id"],
            "R2B_HLC_CALIBRATION_HISTORY_ONLY": True, "R2BH_USE_FORBIDDEN": True,
            "R2C_USE_FORBIDDEN": True, "R2D_CONFIRMATORY_USE_FORBIDDEN": True,
            "RBR_SCIENTIFIC_USE_FORBIDDEN": True,
            "allowed_use": "READ_ONLY_ARCHITECTURE_DIAGNOSTIC",
        } for row in hlc_history],
        "R2B_HLC_resimulation_calls": 0,
    }
    tsb_result = read(R2B_TSB_RESULT)
    if not tsb_result["summary"]["development_success"]:
        raise RuntimeError("R2_BH_TSB_ROUND0_NOT_DEVELOPMENT_PASS")
    tsb = {
        "schema_version": "r2_bh_tsb_family_development_candidate_v1.0",
        "status": "TSB_FAMILY_DEVELOPMENT_CANDIDATE_FROZEN",
        "validation_status": "PENDING_FRESH_R2C_VALIDATION",
        "parameter_file": {"path": str(R2B_TSB_PARAMS.relative_to(ROOT)), "sha256": sha(R2B_TSB_PARAMS)},
        "round_result": {"path": str(R2B_TSB_RESULT.relative_to(ROOT)), "sha256": sha(R2B_TSB_RESULT)},
        "generator_architecture": {"path": str(R2B_GENERATOR.relative_to(ROOT)), "sha256": sha(R2B_GENERATOR)},
        "parameters": tsb_result["parameters"], "DEV_CAL_result_summary": tsb_result["summary"],
        "new_TSB_simulation_calls": 0, "TSB_parameters_adjusted": False,
    }
    official_env()
    warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)
    cutoff = max(row["selector_rank_sha256"] for row in hlc_history)
    rows, row_audits, source_audit = _select_unique_family_suffix(
        "R-HLC", cutoff, prior_exclusion, {}, set(), set()
    )
    for row in rows:
        row.update({
            "PERMANENT_ENGINEERING_ONLY": True, "R2C_USE_FORBIDDEN": True,
            "R2D_CONFIRMATORY_USE_FORBIDDEN": True, "RBR_SCIENTIFIC_USE_FORBIDDEN": True,
            "selection_role": "OUTCOME_BLIND_R2_BH_HLC_ARCHITECTURE_DEVELOPMENT",
        })
    if len(rows) != 8 or len({row["scenario_token"] for row in rows}) != 8:
        raise RuntimeError("R2_BH_HLC_ROSTER_CARDINALITY_FAIL")
    old_tokens = {row["scenario_token"] for row in prior_exclusion["entries"]}
    old_logs = {row["log_id"] for row in prior_exclusion["entries"]}
    if any(row["scenario_token"] in old_tokens or row["log_id"] in old_logs for row in rows):
        raise RuntimeError("R2_BH_HLC_ROSTER_FIREWALL_OVERLAP")
    roster = {
        "schema_version": "r2_bh_hlc_arch_dev_roster_v1.0",
        "status": "FROZEN_BEFORE_ANY_R2_BH_GENERATOR_OUTCOME",
        "selection_semantics": "CONTINUE_FROZEN_V1_3_HASH_RANK_AFTER_R2_B_HLC_PREFIX",
        "source_universe": r2b_roster["source_universe"],
        "pre_selection_firewall": {"path": str(R2B_EXCLUSION.relative_to(ROOT)), "sha256": sha(R2B_EXCLUSION), "count": 101},
        "entries": rows, "count": 8, "candidate_audits": row_audits, "source_audit": source_audit,
        "allowed_selection_inputs": ["context", "map", "route_reference", "Primary80", "technical_runtime_applicability"],
        "outcome_F_match_safety_BDD_representation_used": False,
    }
    new_entries = list(prior_exclusion["entries"]) + [{
        "scenario_token": row["scenario_token"], "log_id": row["log_id"], "family": "R-HLC",
        "sources": [str(OUT["roster"].relative_to(ROOT))],
        "reasons": ["R2_BH_HLC_TARGET_CAPTURE_ARCHITECTURE_DEV_IDENTITY"],
        "PERMANENT_ENGINEERING_ONLY": True, "R2C_USE_FORBIDDEN": True,
        "R2D_CONFIRMATORY_USE_FORBIDDEN": True, "RBR_SCIENTIFIC_USE_FORBIDDEN": True,
    } for row in rows]
    exclusion = {
        "schema_version": "r2_bh_hlc_arch_permanent_exclusion_ledger_v1.0",
        "status": "FROZEN_ADDITIVE_R2_BH_FIREWALL", "source_ledger": {"path": str(R2B_EXCLUSION.relative_to(ROOT)), "sha256": sha(R2B_EXCLUSION)},
        "entries": new_entries, "counts": {"pre_R2_BH": 101, "R2_BH_HLC_DEV_ARCH": 8, "effective_unique_identities": 109},
        "entry_removal_or_reduction_allowed": False,
    }
    pair_cache: Dict[str, Any] = {}
    pair_rows = []
    for index, row in enumerate(rows, 1):
        binding = freeze_pair_binding(row, pair_cache)
        binding["pair_id"] = f"R2BH-ARCH-HLC-{index:02d}"
        binding["future_realized_trace_used"] = False
        binding["future_safety_result_used"] = False
        pair_rows.append(binding)
    pairs = {"schema_version": "r2_bh_hlc_arch_pair_bindings_v1.0", "status": "FROZEN_8_OF_8_PRE_OUTCOME", "pairs": pair_rows}
    contract, space = architecture_contract(), parameter_space()
    ledger = {
        "schema_version": "r2_bh_hlc_arch_run_ledger_v1.0", "status": "FROZEN_PRE_EXECUTION",
        "roster_canonical_sha256": canonical_sha(roster), "pair_binding_canonical_sha256": canonical_sha(pairs),
        "architecture_contract_canonical_sha256": canonical_sha(contract), "parameter_space_canonical_sha256": canonical_sha(space),
        "rounds": [], "maximum_rounds": 3, "technical_rerun_policy": "FRESH_RUN_ID_AND_ROOT_TECHNICAL_FAILURE_ONLY",
        "identity_replacement": False, "R2C_identities_selected": False, "scientific_simulation": False,
        "TSB_simulation_calls": 0, "RBR_started": False,
    }
    authorization = {
        "schema_version": "r2_bh_scientific_owner_engineering_authorization_v1.0",
        "R2_BH_ENGINEERING_ONLY_HLC_SIMULATION_AUTHORIZED": True,
        "scope": "FRESH_FROZEN_R2_BH_HLC_DEV_ARCH_IDENTITIES_ONLY_MAX_3_ROUNDS",
        "TSB_simulation_authorized": False, "old_identity_rerun_authorized": False,
        "R2C_or_confirmatory_authorized": False, "RBR_authorized": False,
    }
    forensic = """# R2-BH HLC Target-Capture Architecture Forensic v1

## 结论

`V1_REANCHOR_DIAGNOSIS = PASS`。该结论来自代码与确定性合成几何审计，未运行 simulation。

## 数学结果

V1 先构造 `xy_before = source·(1-p)+target·p`，再对整条轨迹施加常量平移 `current_xy-xy_before[0]`。当 `p=1` 时，`xy_before=target`，所以终点相对 target center 的偏移等于当前相对 target 的偏移。常量 re-anchor 保证 state0 identity，却同时把当前 target-frame residual 搬到了全部 future states，不能形成有限时间 target-center attractor。

合成测试覆盖当前偏移 `0, +0.25, +0.50, -0.25, -0.50 m`，五种情况下 planned terminal offset 分别保持为相同数值，5/5 精确支持该不变量。

## R2-BH V2 原则

V2 将 behavior morphology 与 target capture 分离。state0 仍严格等于 current ego；state1+ 的 target-frame lateral/heading residual 使用固定 absolute-episode-time quintic 权重衰减，并在固定 capture end 归零。该内部 capture signal 不替代 frozen realized `p(t)` measurement，不使用自由空间路径或 geometry extrapolation。
"""
    for key, value in (("exposure", exposure), ("tsb", tsb), ("invariant", audit), ("roster", roster), ("exclusion", exclusion), ("pairs", pairs), ("contract", contract), ("space", space), ("ledger", ledger), ("authorization", authorization), ("forensic", forensic)):
        write_new(OUT[key], value)
    print(json.dumps({"diagnosis": audit["status"], "HLC_DEV_ARCH": 8, "final_exclusion_count": 109, "TSB_frozen": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
