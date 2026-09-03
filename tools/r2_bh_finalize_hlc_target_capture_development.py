#!/usr/bin/env python3
"""Finalize R2-BH HLC architecture development and enforce its stop rule."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "docs/stageR/r2"
LEDGER = R2 / "r2_bh_hlc_arch_run_ledger_v1.0.json"
SUMMARY = R2 / "r2_bh_hlc_arch_round_summary_v1.json"
CAPTURE_AUDIT = R2 / "r2_bh_hlc_target_capture_audit_v1.json"
FIREWALL = R2 / "r2_bh_generator_data_firewall_audit_v1.json"
DEV_MANIFEST = R2 / "r2_bh_hlc_arch_development_binding_manifest_v1.0.json"
REPORT = R2 / "R2_BH_HLC_Target_Capture_Development_Report_v1.md"
REQUEST = R2 / "R2_BH_R2C_Readiness_Request_v0.1.md"
TSB = R2 / "r2_bh_tsb_family_development_candidate_v1.0.json"
PROTECTED = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"
PROTECTED_SHA = "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8"


def load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Mapping[str, Any] | str) -> None:
    if path.exists():
        raise FileExistsError(f"R2_BH_VERSIONED_OUTPUT_EXISTS:{path}")
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def dist(values: Iterable[float]) -> Dict[str, Any]:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    q = np.quantile(array, [0, .25, .5, .75, 1])
    return {"n": len(array), **dict(zip(("min", "p25", "median", "p75", "max"), [round(float(value), 6) for value in q]))}


def raw_provenance(run_roots: Iterable[str]) -> Dict[str, Any]:
    records = []
    for run_root in sorted(set(run_roots)):
        root = ROOT / run_root
        files = []
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            files.append({"path": str(path.relative_to(root)), "sha256": sha(path), "bytes": path.stat().st_size})
        tree_sha = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        records.append({"run_root": run_root, "file_count": len(files), "tree_sha256": tree_sha})
    aggregate = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"run_count": len(records), "aggregate_tree_sha256": aggregate, "runs": records}


def component(path: str) -> Dict[str, Any]:
    target = ROOT / path
    return {"path": path, "sha256": sha(target), "bytes": target.stat().st_size}


def main() -> int:
    if sha(PROTECTED) != PROTECTED_SHA:
        raise PermissionError("PROTECTED_CSV_SHA_MISMATCH")
    ledger = load(LEDGER)
    if ledger["status"] != "R2_BH_ENGINEERING_EXECUTION_COMPLETE" or len(ledger["rounds"]) != 3:
        raise RuntimeError("R2_BH_THREE_ROUND_EXECUTION_NOT_COMPLETE")
    rounds, run_roots = [], []
    for row in ledger["rounds"]:
        result = load(ROOT / row["result_file"])
        rounds.append({
            "round_index": result["round_index"], "parameters": result["parameters"],
            "parameter_file": row["parameter_file"], "parameter_sha256": row["parameter_sha256"],
            "result_file": row["result_file"], "result_sha256": row["result_sha256"],
            "summary": result["summary"],
        })
        run_roots.extend(run["run_root"] for run in result["runs"])
    final = rounds[-1]
    converged = bool(final["summary"]["development_success"])
    if converged:
        raise RuntimeError("R2_BH_CONVERGED_PATH_REQUIRES_SELECTED_CANDIDATE_FREEZE")
    provenance = raw_provenance(run_roots)
    summary = {
        "schema_version": "r2_bh_hlc_arch_round_summary_v1",
        "status": "R2_BH_DEVELOPMENT_NOT_CONVERGED",
        "rounds_executed": 3, "round4_executed": False, "rounds": rounds,
        "final_counts": final["summary"]["counts"],
        "final_mechanism_margin_distributions": final["summary"]["mechanism_margin_distributions"],
        "final_terminal_offset_distributions_m": final["summary"]["absolute_terminal_target_offset_distributions_m"],
        "selected_HLC_V2_candidate_frozen": False, "complete_G_R2_candidate_frozen": False,
        "selected_parameter_sha256": None, "actual_HLC_engineering_runs": len(run_roots),
        "TSB_simulation_calls": 0, "scientific_simulation_calls": 0,
        "raw_output_provenance": provenance, "raw_outputs_committed": False,
    }
    write(SUMMARY, summary)
    capture_rounds = []
    for row in ledger["rounds"]:
        result = load(ROOT / row["result_file"])
        landmarks: Dict[str, Any] = {}
        for arm in ("baseline", "treatment"):
            arm_landmarks = {}
            for name in ("capture_start", "capture_midpoint", "capture_end", "Primary_terminal"):
                arm_landmarks[name] = {
                    "planner_state1_commanded_offset_m": dist(
                        pair["target_capture"][arm]["landmarks"][name]["planner_state1_commanded_target_frame_offset_m"]
                        for pair in result["pairs"]
                    ),
                    "realized_offset_m": dist(
                        pair["target_capture"][arm]["landmarks"][name]["realized_target_frame_offset_m"]
                        for pair in result["pairs"]
                    ),
                }
            landmarks[arm] = arm_landmarks
        capture_rounds.append({
            "round_index": result["round_index"], "landmarks": landmarks,
            "capture_end_zero_state1_command_16_of_16": result["summary"]["capture_end_zero_command_16_of_16"],
        })
    capture_audit = {
        "schema_version": "r2_bh_hlc_target_capture_audit_v1",
        "status": "PLANNER_TARGET_CENTER_COMMAND_ATTRACTION_PROVEN_REALIZED_TRANSFER_NOT_CONVERGED",
        "architecture_rule": "FIXED_ABSOLUTE_TIME_QUINTIC_C2_RESIDUAL_DECAY",
        "state0_exact_current_ego": True, "no_geometry_extrapolation": True,
        "scientific_progress_measurement_changed": False,
        "rounds": capture_rounds,
        "final_round_capture_end_zero_state1_command_16_of_16": True,
        "final_round_realized_treatment_terminal_offset_m": final["summary"]["absolute_terminal_target_offset_distributions_m"]["treatment"],
        "interpretation": "COMMAND_ATTRACTOR_PRESENT_BUT_CLOSED_LOOP_REALIZED_TARGET_CAPTURE_FAILED",
    }
    write(CAPTURE_AUDIT, capture_audit)
    roster = load(R2 / "r2_bh_hlc_arch_dev_roster_v1.0.json")
    firewall = {
        "schema_version": "r2_bh_generator_data_firewall_audit_v1", "status": "PASS",
        "fresh_DEV_ARCH_identity_count": len(roster["entries"]),
        "overlap_historical_R1_R2A_R2B": 0,
        "R2B_HLC_old_identities_rerun": 0, "TSB_new_simulation_calls": 0,
        "scenario_or_log_specific_adaptation": False, "scientific_thresholds_modified": False,
        "numerical_calibration_source": "FRESH_R2_BH_HLC_DEV_ARCH_ONLY",
        "R2C_identities_selected": False, "confirmatory_smoke_started": False, "RBR_started": False,
        "protected_CSV_sha256": sha(PROTECTED),
    }
    write(FIREWALL, firewall)
    paths = [
        "tools/r2_bh_hlc_target_capture_generator_v2.py",
        "tools/r2_bh_hlc_target_capture_planner_v2.py",
        "tools/r2_bh_execute_hlc_target_capture_development.py",
        "tools/r2_bh_freeze_hlc_target_capture_development.py",
        "configs/r1_official_technical_smoke_hydra/planner/r2_bh_hlc_target_capture_dev_v2.yaml",
        "tools/r2_b_controller_aware_generator_v1.py",
        "tools/r1_primary80_scientific_time_controller_v1.py",
        "tools/r1_b2_9_e_official_run_lifecycle.py",
        "tools/r1_b2_8_r3_2_post_run_evaluator_dispatcher.py",
        "tools/r1_b2_8_r3_1_official_safety_adapter.py",
        "tools/r1_official_metric_canonicalizer.py",
        "tools/r1_official_technical_smoke_evaluator_v2_1.py",
        "tools/r1_closed_loop_benchmark_v2_1.py",
        "tools/r1_closed_loop_benchmark_v2_3.py",
        "tools/r1_hlc_measurement_conformance_v1.py",
        "docs/stageR/r2/r2_bh_r2b_hlc_outcome_exposure_ledger_v1.0.json",
        "docs/stageR/r2/r2_bh_tsb_family_development_candidate_v1.0.json",
        "docs/stageR/r2/r2_bh_hlc_v1_reanchor_invariant_audit_v1.json",
        "docs/stageR/r2/r2_bh_hlc_arch_dev_roster_v1.0.json",
        "docs/stageR/r2/r2_bh_hlc_arch_permanent_exclusion_ledger_v1.0.json",
        "docs/stageR/r2/r2_bh_hlc_arch_pair_bindings_v1.0.json",
        "docs/stageR/r2/r2_bh_hlc_architecture_contract_v2.0.json",
        "docs/stageR/r2/r2_bh_hlc_arch_parameter_space_v2.0.json",
        "docs/stageR/r2/r2_bh_hlc_arch_run_ledger_v1.0.json",
        "docs/stageR/r2/r2_bh_hlc_arch_round_summary_v1.json",
        "docs/stageR/r2/r2_bh_hlc_target_capture_audit_v1.json",
        "docs/stageR/r2/r2_bh_generator_data_firewall_audit_v1.json",
    ]
    paths.extend(row["parameter_file"] for row in rounds)
    paths.extend(row["result_file"] for row in rounds)
    manifest = {
        "schema_version": "r2_bh_hlc_arch_development_binding_manifest_v1.0",
        "status": "R2_BH_DEVELOPMENT_NOT_CONVERGED",
        "component_SHA_closure": "PASS_WITH_NO_SELECTED_HLC_OR_COMPLETE_G_R2_CANDIDATE",
        "selected_HLC_V2_parameter_sha256": None,
        "complete_G_R2_candidate_manifest_created": False,
        "TSB_family_candidate": {"path": str(TSB.relative_to(ROOT)), "sha256": sha(TSB)},
        "components": [component(path) for path in dict.fromkeys(paths)],
        "raw_DEV_output_provenance": provenance, "raw_outputs_committed": False,
        "nuplan_runtime": {"version": "1.2.2", "python": "/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9", "nuplan_devkit_git_commit": "e9241677997dd86bfc0bcd44817ab04fe631405b"},
        "actual_HLC_engineering_runner_run_calls": len(run_roots), "TSB_simulation_calls": 0,
        "scientific_simulation_calls": 0, "R2C_identities_selected": False, "RBR_started": False,
        "protected_CSV_sha256": sha(PROTECTED),
    }
    write(DEV_MANIFEST, manifest)
    report = f"""# R2-BH HLC Target-Capture Development Report v1

## 结论

`R2_BH_DEVELOPMENT_NOT_CONVERGED`。V1 constant re-anchor diagnosis 经 5/5 合成 offset case 支持。V2 已建立固定 absolute-time target-center command attractor，但 3 轮 fresh DEV-ARCH closed-loop 结果没有实现 frozen mechanism 或 endpoint，因此没有冻结 HLC V2 candidate，也没有组合完整 G_R2 candidate。

## 架构证据

V2 将 behavior morphology 与 target capture 分离。state0 精确保持 current ego；state1+ 的 native target-frame lateral 与 heading residual 以 C2 quintic 权重衰减。capture start/end 固定在 episode absolute time，不随 replanning 重启。每轮 16/16 arm 在 capture end 的 state1 residual command 均为 0；scientific realized p(t) 定义未改变。

## 三轮结果

- Round 0：mechanism 0/8、endpoint 0/8、F_match 8/8、engineering 8/8、safety 4/8。
- Round 1：mechanism 0/8、endpoint 0/8、F_match 7/8、engineering 8/8、safety 4/8。
- Round 2：mechanism 0/8、endpoint 0/8、F_match 8/8、engineering 8/8、safety 4/8。

最终 treatment retreat count margin 全部为 -1，commit latency 与 monotonic margin 均因 measurement not OK 而不可评估。最终 endpoint gate：offset 0/8、heading 8/8、lateral velocity 8/8、route progress 7/8。treatment terminal offset |m| 的 min/p25/median/p75/max 为 `4.094948/4.433190/4.658622/5.511358/7.883494`。

## 防火墙与停止规则

8 个 DEV-ARCH identities 与 historical/R1/R2-A/R2-B 重叠为 0，全部永久 engineering-only。R2-B HLC 未重跑；TSB 新仿真为 0；科学阈值与 scenario-specific rule 均未改变。严格停止于 3 轮，不执行 Round 4，不选择 R2-C identity，不启动 confirmatory smoke 或 RBR。
"""
    write(REPORT, report)
    request = f"""# R2-BH → R2-C Readiness Request v0.1

## 当前处置

**NOT_READY / REQUEST_WITHHELD**。

TSB family development candidate 已机械冻结；HLC V2 在 3 轮 architecture-development 上限后仍为 mechanism 0/8、endpoint 0/8，未满足 hard requirements。因此没有创建 `r2_bh_selected_hlc_generator_parameters_v2.0.json` 或 `r2_bh_complete_g_r2_development_candidate_manifest_v1.0.json`，本轮不请求 R2-C fresh identity selection 授权。

Scientific Owner 后续如需继续，必须另行预注册新的 architecture phase；当前结果不授权 Round 4、阈值放宽、identity replacement、R2-C、confirmatory smoke 或 RBR。

R2-BH development binding manifest SHA256：`PENDING_EXTERNAL_SHA`
"""
    write(REQUEST, request)
    print(json.dumps({
        "status": manifest["status"], "rounds": 3, "HLC_engineering_runs": len(run_roots),
        "TSB_simulation_calls": 0, "development_manifest_sha256": sha(DEV_MANIFEST),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
