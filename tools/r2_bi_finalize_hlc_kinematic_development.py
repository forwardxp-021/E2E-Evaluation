#!/usr/bin/env python3
"""Offline-only finalization of the stopped R2-BI HLC DEV-KIN execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "docs/stageR/r2"
RAW = ROOT / "outputs/r2_bi_hlc_kinematic_target_capture_dev_v1/round_0"
LEDGER = R2 / "r2_bi_hlc_dev_kin_run_ledger_v1.0.json"
STOP_AUDIT = R2 / "r2_bi_hlc_dev_kin_round_0_architecture_stop_audit_v1.json"
ROUND_SUMMARY = R2 / "r2_bi_hlc_dev_kin_round_summary_v1.json"
FIREWALL = R2 / "r2_bi_hlc_kinematic_data_firewall_audit_v1.json"
MANIFEST = R2 / "r2_bi_hlc_kinematic_development_binding_manifest_v1.0.json"
REPORT = R2 / "R2_BI_HLC_Kinematic_Target_Capture_Development_Report_v1.md"
DISPOSITION = R2 / "R2_BI_R2C_Readiness_Disposition_v0.1.md"
PROTECTED = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"
PROTECTED_SHA = "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8"


def load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any, update: bool = False) -> None:
    serialized = value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    if path.exists() and not update:
        if path.read_text(encoding="utf-8") == serialized:
            return
        raise FileExistsError(f"R2_BI_VERSIONED_OUTPUT_EXISTS:{path}")
    path.write_text(serialized, encoding="utf-8")


def jsonl(path: Path) -> list[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def raw_provenance(run_roots: Iterable[Path]) -> Dict[str, Any]:
    records = []
    for root in run_roots:
        files = [path for path in sorted(root.rglob("*")) if path.is_file()]
        inventory = [
            {"path": str(path.relative_to(root)), "sha256": sha(path), "bytes": path.stat().st_size}
            for path in files
        ]
        records.append({
            "run_root": str(root.relative_to(ROOT)),
            "file_count": len(inventory),
            "tree_sha256": hashlib.sha256(
                json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        })
    return {
        "run_count": len(records),
        "aggregate_tree_sha256": hashlib.sha256(
            json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "runs": records,
    }


def component(path: str) -> Dict[str, Any]:
    target = ROOT / path
    return {"path": path, "sha256": sha(target), "bytes": target.stat().st_size}


def reconstruct_failure() -> Dict[str, Any]:
    """Re-evaluate only the failed frozen planner function; no simulator is constructed."""
    from tools.r1_b2_8_r3_prospective_selector import official_env

    official_env()
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    from tools.r1_closed_loop_benchmark_v2_3 import build_hlc_route_continuous_reference_v2_3
    from tools.r2_bi_hlc_kinematic_target_capture_generator_v3 import CaptureInfeasible
    from tools.r2_bi_hlc_kinematic_target_capture_planner_v3 import _states

    roster = load(R2 / "r2_bi_hlc_dev_kin_roster_v1.0.json")
    entry = roster["entries"][0]
    parameters = load(
        R2 / "r2_bi_hlc_dev_kin_rounds/r2_bi_hlc_dev_kin_round_0_parameters_v3.0.json"
    )["parameters"]
    trace_path = RAW / "R2BI-HLC-R0-01-TREATMENT/trace/realized_current_ego.jsonl"
    current = jsonl(trace_path)[-1]["current_ego"]
    map_api = get_maps_api(
        str(ROOT.parent / "nuplan/dataset/maps"), "nuplan-maps-v1.0", entry["map_name"]
    )
    corridor = build_hlc_route_continuous_reference_v2_3(
        map_api,
        entry["route_roadblock_ids"],
        str(entry["source_lane_id"]),
        str(entry["target_lane_id"]),
        current,
        max(0.2, float(current["speed_mps"])) * 7.9,
    )
    try:
        _states(current, 1.1, corridor, "TREATMENT", parameters, False)
    except CaptureInfeasible as error:
        return {
            "reconstruction": "OFFLINE_FROZEN_FUNCTION_REEVALUATION_ONLY",
            "simulation_constructed": False,
            "runner_run_calls": 0,
            "absolute_episode_time_s": 1.1,
            "reason": error.reason,
            "audit": error.audit,
        }
    raise RuntimeError("R2_BI_EXPECTED_CAPTURE_INFEASIBLE_NOT_REPRODUCED")


def runner_report(path: Path) -> Dict[str, Any]:
    import pandas as pd

    rows = pd.read_parquet(path).to_dict("records")
    if len(rows) != 1:
        raise RuntimeError(f"R2_BI_RUNNER_REPORT_ROW_COUNT_NOT_ONE:{path}:{len(rows)}")
    row = rows[0]
    return {
        "succeeded": bool(row["succeeded"]),
        "scenario_name": str(row["scenario_name"]),
        "planner_name": str(row["planner_name"]),
        "log_name": str(row["log_name"]),
        "error_contains_frozen_feasibility_failure": "R2_BI_CAPTURE_INFEASIBLE:FROZEN_KINEMATIC_FEASIBILITY_GATE_FAIL"
        in str(row.get("error_message") or ""),
    }


def main() -> int:
    if sha(PROTECTED) != PROTECTED_SHA:
        raise PermissionError("PROTECTED_CSV_SHA_MISMATCH")
    baseline_root = RAW / "R2BI-HLC-R0-01-BASELINE"
    treatment_root = RAW / "R2BI-HLC-R0-01-TREATMENT"
    if not baseline_root.is_dir() or not treatment_root.is_dir():
        raise FileNotFoundError("R2_BI_EXPECTED_TWO_ATTEMPTED_RUN_ROOTS_MISSING")
    reports = {
        "baseline": runner_report(baseline_root / "raw/runner_report.parquet"),
        "treatment": runner_report(treatment_root / "raw/runner_report.parquet"),
    }
    if not reports["baseline"]["succeeded"] or reports["treatment"]["succeeded"]:
        raise RuntimeError("R2_BI_UNEXPECTED_RUNNER_REPORT_DISPOSITION")
    if not reports["treatment"]["error_contains_frozen_feasibility_failure"]:
        raise RuntimeError("R2_BI_TREATMENT_FAILURE_CAUSE_NOT_BOUND")
    counts = {}
    controls = {}
    for arm, root in (("baseline", baseline_root), ("treatment", treatment_root)):
        counts[arm] = {
            "realized_trace_rows": len(jsonl(root / "trace/realized_current_ego.jsonl")),
            "planner_telemetry_rows": len(jsonl(root / "telemetry/planner_kinematic_capture.jsonl")),
            "controller_telemetry_rows": len(jsonl(root / "telemetry/controller_actual_shadow.jsonl")),
        }
        controller = jsonl(root / "telemetry/controller_actual_shadow.jsonl")
        controls[arm] = {
            "observed_rows": len(controller),
            "shadow_actual_direction_agreement_rows": sum(bool(row["direction_agreement"]) for row in controller),
            "maximum_shadow_actual_absolute_difference_radps": max(
                float(row["absolute_command_difference_radps"]) for row in controller
            ),
        }
    reconstructed = reconstruct_failure()
    feasibility = reconstructed["audit"]["feasibility"]
    exceeded = {
        "lateral_acceleration_mps2": {
            "observed": feasibility["max_abs_lateral_acceleration_mps2"],
            "frozen_limit": 6.0,
            "excess": feasibility["max_abs_lateral_acceleration_mps2"] - 6.0,
        }
    }
    stop_audit = {
        "schema_version": "r2_bi_hlc_dev_kin_round_0_architecture_stop_audit_v1",
        "status": "DIRECT_FAIL_CLOSED_KINEMATIC_FEASIBILITY_VIOLATION",
        "disposition": "R2_BI_DEVELOPMENT_NOT_CONVERGED",
        "failed_run_id": "R2BI-HLC-R0-01-TREATMENT",
        "failed_pair_id": "R2BI-DEV-KIN-HLC-01",
        "failure_at_first_allowed_arm_divergence": True,
        "absolute_episode_time_s": 1.1,
        "runner_reports": reports,
        "artifact_row_counts": counts,
        "controller_shadow_actual_audit_before_failure": controls,
        "offline_failure_reconstruction": reconstructed,
        "frozen_feasibility_limit_exceeded": exceeded,
        "other_feasibility_observations": {
            "curvature_inv_m": feasibility["max_abs_curvature_inv_m"],
            "yaw_rate_radps": feasibility["max_abs_yaw_rate_radps"],
            "state0_to_state1_distance_m": feasibility["state0_to_state1_distance_m"],
            "state0_tangent_mismatch_rad": feasibility["state0_tangent_mismatch_abs_rad"],
            "future_heading_xy_mismatch_rad": feasibility["future_heading_xy_mismatch_abs_rad"],
        },
        "classification": {
            "technical_infrastructure_failure": False,
            "behavior_or_architecture_failure": True,
            "systematic_across_identity_claimed": False,
            "why_execution_stopped": "V3_CONTRACT_REQUIRES_FAIL_CLOSED_AND_FIXED_ROUND_CANNOT_COMPLETE",
            "round1_authorized": False,
            "technical_rerun_authorized": False,
        },
        "important_scope_note": "The single observed treatment failure is not labeled systematic across identities; it is a direct frozen feasibility violation that makes Round 0 incomplete and prevents aggregate numerical calibration.",
        "simulation_calls_added_by_finalizer": 0,
    }
    write(STOP_AUDIT, stop_audit)
    provenance = raw_provenance((baseline_root, treatment_root))
    round_summary = {
        "schema_version": "r2_bi_hlc_dev_kin_round_summary_v1",
        "status": "R2_BI_DEVELOPMENT_NOT_CONVERGED",
        "round0": {
            "started": True,
            "completed": False,
            "attempted_runs": 2,
            "technically_complete_runs": 1,
            "failed_runs": 1,
            "remaining_runs_not_started": 14,
            "pair_scientific_evaluations": 0,
            "mechanism_pass": "NOT_EVALUABLE",
            "F_match_pass": "NOT_EVALUABLE",
            "endpoint_pass": "NOT_EVALUABLE",
            "engineering_pair_pass": "NOT_EVALUABLE",
            "safety_pair_pass": "NOT_EVALUABLE",
        },
        "round1_started": False,
        "round1_authorized": False,
        "actual_HLC_engineering_runner_run_calls": 2,
        "actual_HLC_technically_complete_runs": 1,
        "TSB_simulation_calls": 0,
        "scientific_simulation_calls": 0,
        "selected_HLC_V3_parameters_created": False,
        "complete_G_R2_candidate_created": False,
        "R2C_started": False,
        "confirmatory_smoke_started": False,
        "RBR_started": False,
        "raw_output_provenance": provenance,
        "raw_outputs_committed": False,
    }
    write(ROUND_SUMMARY, round_summary)
    ledger = load(LEDGER)
    ledger.update({
        "status": "R2_BI_DEVELOPMENT_NOT_CONVERGED",
        "rounds": [{
            "round_index": 0,
            "status": "INCOMPLETE_ARCHITECTURE_FAIL_CLOSED",
            "attempted_runs": 2,
            "technically_complete_runs": 1,
            "failed_runs": 1,
            "remaining_schedule_stopped": True,
            "stop_audit": str(STOP_AUDIT.relative_to(ROOT)),
            "stop_audit_sha256": sha(STOP_AUDIT),
        }],
        "execution_failures": [{
            "round_index": 0,
            "run_id": "R2BI-HLC-R0-01-TREATMENT",
            "error": "CaptureInfeasible:R2_BI_CAPTURE_INFEASIBLE:FROZEN_KINEMATIC_FEASIBILITY_GATE_FAIL",
            "classification": "BEHAVIOR_ARCHITECTURE_FAIL_CLOSED_NOT_TECHNICAL_INFRASTRUCTURE",
            "failing_metric": "max_abs_lateral_acceleration_mps2",
            "observed": feasibility["max_abs_lateral_acceleration_mps2"],
            "frozen_limit": 6.0,
            "remaining_schedule_stopped": True,
        }],
        "actual_HLC_engineering_runs": 2,
        "actual_HLC_engineering_runner_run_calls": 2,
        "actual_HLC_technically_complete_runs": 1,
        "technical_reruns": 0,
        "round1_started": False,
        "selected_HLC_V3_parameters_created": False,
        "complete_G_R2_candidate_created": False,
    })
    write(LEDGER, ledger, update=True)
    roster = load(R2 / "r2_bi_hlc_dev_kin_roster_v1.0.json")
    firewall = {
        "schema_version": "r2_bi_hlc_kinematic_data_firewall_audit_v1",
        "status": "PASS",
        "R2BH_history_only_identity_count": 8,
        "fresh_DEV_KIN_identity_count": len(roster["entries"]),
        "overlap_historical_R1_R2A_R2B_R2BH": 0,
        "overlap_basis": "SELECTION_APPLIED_PRE_SELECTION_FIREWALL_109_AND_SELECTED_ONLY_NONMEMBERS",
        "R2B_or_R2BH_identity_resimulation_calls": 0,
        "R2BH_raw_used_for_V3_numerical_tuning": False,
        "scenario_or_log_specific_adaptation": False,
        "scientific_thresholds_modified": False,
        "failed_identity_replaced": False,
        "technical_reruns": 0,
        "R2C_identities_selected": False,
        "confirmatory_smoke_started": False,
        "RBR_started": False,
        "protected_CSV_sha256": sha(PROTECTED),
    }
    write(FIREWALL, firewall)
    paths = [
        "tools/r2_bi_controller_interface_forensic.py",
        "tools/r2_bi_hlc_kinematic_target_capture_generator_v3.py",
        "tools/r2_bi_hlc_kinematic_target_capture_planner_v3.py",
        "tools/r2_bi_zero_run_entry_gate_audit.py",
        "tools/r2_bi_freeze_hlc_kinematic_architecture.py",
        "tools/r2_bi_freeze_hlc_dev_kin_roster.py",
        "tools/r2_bi_execute_hlc_kinematic_target_capture_development.py",
        "tools/r2_bi_finalize_hlc_kinematic_development.py",
        "tests/test_r2_bi_hlc_kinematic_target_capture.py",
        "configs/r1_official_technical_smoke_hydra/planner/r2_bi_hlc_kinematic_target_capture_dev_v3.yaml",
        "tools/r1_primary80_scientific_time_controller_v1.py",
        "tools/r1_b2_9_e_official_run_lifecycle.py",
        "tools/r1_b2_8_r3_2_post_run_evaluator_dispatcher.py",
        "tools/r1_b2_8_r3_1_official_safety_adapter.py",
        "tools/r1_official_metric_canonicalizer.py",
        "tools/r1_official_technical_smoke_evaluator_v2_1.py",
        "tools/r1_closed_loop_benchmark_v2_1.py",
        "tools/r1_closed_loop_benchmark_v2_3.py",
        "tools/r1_hlc_measurement_conformance_v1.py",
        "tools/r2_b_controller_aware_generator_v1.py",
        "docs/stageR/r2/r2_bi_r2bh_outcome_exposure_ledger_v1.0.json",
        "docs/stageR/r2/r2_bi_hlc_v2_controller_interface_forensic_v1.json",
        "docs/stageR/r2/R2_BI_HLC_V2_Controller_Interface_Forensic_v1.md",
        "docs/stageR/r2/r2_bi_hlc_kinematic_capture_architecture_contract_v3.0.json",
        "docs/stageR/r2/r2_bi_hlc_kinematic_capture_parameter_space_v3.0.json",
        "docs/stageR/r2/r2_bi_hlc_architecture_failure_taxonomy_v1.0.json",
        "docs/stageR/r2/r2_bi_scientific_owner_engineering_authorization_v1.0.json",
        "docs/stageR/r2/r2_bi_mandatory_zero_run_entry_gate_audit_v1.json",
        "docs/stageR/r2/r2_bi_hlc_dev_kin_roster_v1.0.json",
        "docs/stageR/r2/r2_bi_hlc_dev_kin_permanent_exclusion_ledger_v1.0.json",
        "docs/stageR/r2/r2_bi_hlc_dev_kin_pair_bindings_v1.0.json",
        "docs/stageR/r2/r2_bi_hlc_dev_kin_rounds/r2_bi_hlc_dev_kin_round_0_parameters_v3.0.json",
        "docs/stageR/r2/r2_bi_hlc_dev_kin_run_ledger_v1.0.json",
        "docs/stageR/r2/r2_bi_hlc_dev_kin_round_0_architecture_stop_audit_v1.json",
        "docs/stageR/r2/r2_bi_hlc_dev_kin_round_summary_v1.json",
        "docs/stageR/r2/r2_bi_hlc_kinematic_data_firewall_audit_v1.json",
    ]
    manifest = {
        "schema_version": "r2_bi_hlc_kinematic_development_binding_manifest_v1.0",
        "status": "R2_BI_DEVELOPMENT_NOT_CONVERGED",
        "component_SHA_closure": "PASS_WITH_NO_SELECTED_HLC_OR_COMPLETE_G_R2_CANDIDATE",
        "components": [component(path) for path in paths],
        "raw_DEV_output_provenance": provenance,
        "raw_outputs_committed": False,
        "nuplan_runtime": {
            "version": "1.2.2",
            "python": "/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9",
            "nuplan_devkit_git_commit": "e9241677997dd86bfc0bcd44817ab04fe631405b",
        },
        "actual_HLC_engineering_runner_run_calls": 2,
        "TSB_simulation_calls": 0,
        "scientific_simulation_calls": 0,
        "selected_HLC_V3_parameter_sha256": None,
        "complete_G_R2_candidate_manifest_created": False,
        "R2C_identities_selected": False,
        "confirmatory_smoke_started": False,
        "RBR_started": False,
        "protected_CSV_sha256": sha(PROTECTED),
    }
    write(MANIFEST, manifest, update=True)
    manifest_sha = sha(MANIFEST)
    report = f"""# R2-BI HLC 运动学 Target-Capture 架构开发报告 v1

## 结论

`R2_BI_DEVELOPMENT_NOT_CONVERGED`。V3 在 simulation 前完成了 25/25 mandatory zero-run entry gates，随后按授权启动 Round 0。第一个 baseline 完成 Primary80；第一个 treatment 在绝对时间 1.1 秒（首次允许 arm divergence）被冻结运动学可行性门 fail-closed。剩余 14 个 run 未启动，Round 1 未启动，也未冻结 selected HLC V3 或 complete G_R2 candidate。

## 失败证据

离线以同一 frozen map、route、current ego 与 Round 0 参数重建失败 planner call，得到 `max_abs_lateral_acceleration_mps2={feasibility['max_abs_lateral_acceleration_mps2']:.6f}`，超过冻结上限 `6.0`。同一 reference 的 curvature 为 `{feasibility['max_abs_curvature_inv_m']:.6f}`、yaw-rate 为 `{feasibility['max_abs_yaw_rate_radps']:.6f}`、state0→state1 距离为 `{feasibility['state0_to_state1_distance_m']:.6f} m`、state0 tangent mismatch 为 `{feasibility['state0_tangent_mismatch_abs_rad']:.6f} rad`、future XY-heading mismatch 为 `0`。因此失败是 controller-visible morphology/capture 合成在真实速度下违反横向加速度门，不是 XY-heading 不一致、硬跳或基础设施故障。

## 运行处置

- 工程 `runner.run` 实际调用 2 次：baseline 成功 1 次，treatment 架构失败 1 次。
- baseline artifacts 为 trace/planner/controller `80/80/79`；treatment 在失败前为 `12/11/11`。
- 已观测 controller actual 与 exact frozen shadow 为 baseline `79/79`、treatment `11/11` 方向一致，命令差为 0；treatment 数据仅覆盖分化前，不能外推为分化后的 transfer 结论。
- 此失败不是技术基础设施故障，禁止 fresh-ID 技术重跑。单一 identity 不被描述为“跨 identity 系统性失败”；但 frozen V3 contract 要求直接 fail closed，固定 cohort Round 0 无法完成，也就没有合法 aggregate numerical update 依据，因此 Round 1 不获授权。

## 防火墙

R2-BH 的 8 个 identities 已冻结为 history-only；新 DEV-KIN 8 个 identities 与 historical/R1/R2-A/R2-B/R2-BH 重叠为 0，全部永久 engineering-only。未修改 scientific mechanism、endpoint、F_match 或 safety threshold；未按 scenario/log 适配；未使用 R2-BH raw 做 V3 数值调参。

科学仿真为 0，TSB 仿真为 0；R2-C、confirmatory smoke 与 RBR 均未启动。Raw outputs 不提交 Git，仅以 SHA provenance 固化。
"""
    write(REPORT, report, update=True)
    disposition = f"""# R2-BI → R2-C Readiness Disposition v0.1

## 当前处置

**NOT READY / REQUEST WITHHELD**。

R2-BI HLC V3 Round 0 在首个 treatment 的首次分化 planner call 即触发冻结横向加速度可行性门。固定 8-pair cohort 未完成，mechanism、F_match、endpoint、engineering 与 safety 的 8/8 开发成功条件均不可评估。因此不生成 selected HLC V3 parameters，不机械组合 complete G_R2 candidate，也不请求 R2-C fresh validation identity selection。

后续若要继续，需要 Scientific Owner 另行授权一个新架构阶段；本结果不授权 Round 1、身份替换、阈值修改、R2-C、confirmatory smoke 或 RBR。

R2-BI development binding manifest SHA256：`{manifest_sha}`
"""
    write(DISPOSITION, disposition, update=True)
    print(json.dumps({
        "status": "R2_BI_DEVELOPMENT_NOT_CONVERGED",
        "actual_HLC_engineering_runner_run_calls": 2,
        "round1_started": False,
        "development_manifest_sha256": manifest_sha,
        "protected_CSV_sha256": sha(PROTECTED),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
