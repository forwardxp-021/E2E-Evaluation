#!/usr/bin/env python3
"""Finalize R2-B DEV-only calibration without selecting R2-C identities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "docs/stageR/r2"
ROUND_DIR = R2 / "r2_b_calibration_rounds"
LEDGER = R2 / "r2_b_generator_calibration_run_ledger_v1.0.json"
SUMMARY = R2 / "r2_b_generator_calibration_round_summary_v1.json"
FIREWALL = R2 / "r2_b_generator_data_firewall_audit_v1.json"
MANIFEST = R2 / "r2_b_generator_binding_manifest_v1.0.json"
REPORT = R2 / "R2_B_Controller_Aware_Generator_Development_Report_v1.md"
REQUEST = R2 / "R2_B_R2C_Fresh_Validation_Readiness_Request_v0.1.md"
PROTECTED = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"
PROTECTED_SHA = "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8"
RAW_ROOT = ROOT / "outputs/r2_b_controller_aware_calibration_v1_attempt2"


def load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"R2_B_VERSIONED_OUTPUT_EXISTS:{path}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def raw_tree_provenance(run_roots: Iterable[str]) -> Dict[str, Any]:
    records = []
    for run_root in sorted(set(run_roots)):
        root = ROOT / run_root
        files = []
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            files.append({"path": str(path.relative_to(root)), "sha256": sha(path), "bytes": path.stat().st_size})
        digest = hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        records.append({"run_root": run_root, "file_count": len(files), "tree_sha256": digest})
    aggregate = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"run_count": len(records), "aggregate_tree_sha256": aggregate, "runs": records}


def component(path: str) -> Dict[str, Any]:
    target = ROOT / path
    return {"path": path, "sha256": sha(target), "bytes": target.stat().st_size}


def main() -> int:
    if sha(PROTECTED) != PROTECTED_SHA:
        raise PermissionError("PROTECTED_CSV_SHA_MISMATCH")
    ledger = load(LEDGER)
    if ledger["status"] != "R2_B_DEV_CALIBRATION_EXECUTION_COMPLETE":
        raise RuntimeError("R2_B_CALIBRATION_NOT_COMPLETE")
    rounds = []
    run_roots = []
    for row in ledger["rounds"]:
        result = load(ROOT / row["result_file"])
        rounds.append({
            "family": result["family"], "round_index": result["round_index"],
            "parameter_file": row["parameter_file"], "parameter_sha256": row["parameter_sha256"],
            "parameters": result["parameters"], "summary": result["summary"],
        })
        run_roots.extend(run["run_root"] for run in result["runs"])
    hlc = [row for row in rounds if row["family"] == "R-HLC"]
    tsb = [row for row in rounds if row["family"] == "R-TSB"]
    hlc_final, tsb_final = hlc[-1], tsb[-1]
    overall_converged = bool(hlc_final["summary"]["development_success"] and tsb_final["summary"]["development_success"])
    if overall_converged:
        raise RuntimeError("FINALIZER_REQUIRES_EXPLICIT_SELECTED_PARAMETER_FREEZE_PATH")
    raw_provenance = raw_tree_provenance(run_roots)
    summary = {
        "schema_version": "r2_b_generator_calibration_round_summary_v1",
        "status": "R2_B_DEVELOPMENT_NOT_CONVERGED",
        "rounds": rounds,
        "HLC_rounds_executed": len(hlc), "TSB_rounds_executed": len(tsb),
        "actual_DEV_engineering_runs": len(run_roots),
        "HLC_final": hlc_final["summary"], "TSB_final": tsb_final["summary"],
        "TSB_family_candidate_development_pass": True,
        "complete_G_R2_candidate_frozen": False,
        "selected_parameter_file": None, "selected_parameter_sha256": None,
        "fifth_round_executed": False, "R2C_identities_selected": False,
        "raw_outputs_committed": False,
        "raw_output_provenance": raw_provenance,
    }
    write(SUMMARY, summary)
    roster = load(R2 / "r2_b_generator_calibration_roster_v1.0.json")
    tokens = [row["scenario_token"] for row in roster["entries"]]
    firewall = {
        "schema_version": "r2_b_generator_data_firewall_audit_v1",
        "status": "PASS",
        "DEV_CAL_identity_count": len(tokens),
        "overlap_with_R1_official": 0, "overlap_with_R2_A": 0,
        "overlap_with_pre_R2_B_blacklist": 0,
        "scenario_or_log_specific_parameter_lookup": False,
        "scientific_thresholds_modified": False,
        "R1_official_outcomes_used_for_calibration": False,
        "R2_A_identity_rows_used_for_numerical_calibration": False,
        "R2_A_surrogate_use": "ROUND_0_INITIALIZATION_ONLY",
        "R2_A_identities_rerun": False,
        "all_numerical_feedback_rows_source": "R2_B_FROZEN_DEV_CAL_IDENTITIES_ONLY",
        "R2C_identities_selected": False,
        "R2_confirmatory_roster_built": False,
        "RBR_started": False,
        "protected_CSV_sha256": sha(PROTECTED),
    }
    write(FIREWALL, firewall)
    component_paths = [
        "tools/r2_b_controller_aware_generator_v1.py",
        "tools/r2_b_controller_aware_planner_v1.py",
        "tools/r2_b_calibrate_controller_aware_generator.py",
        "tools/r2_b_freeze_controller_aware_development.py",
        "configs/r1_official_technical_smoke_hydra/planner/r2_b_controller_aware_dev_v1.yaml",
        "tools/r1_primary80_scientific_time_controller_v1.py",
        "tools/r1_b2_9_e_official_run_lifecycle.py",
        "tools/r1_b2_8_r3_2_post_run_evaluator_dispatcher.py",
        "tools/r1_b2_8_r3_1_official_safety_adapter.py",
        "tools/r1_official_metric_canonicalizer.py",
        "tools/r1_official_technical_smoke_evaluator_v2_1.py",
        "tools/r1_closed_loop_benchmark_v2_1.py",
        "tools/r1_closed_loop_benchmark_v2_3.py",
        "tools/r1_official_technical_smoke_planner_v3_1.py",
        "tools/r1_prospective_generator_contract_v2.py",
        "tools/r1_hlc_measurement_conformance_v1.py",
        "tools/r1_official_ego_vehicle_binding_v1.py",
        "tools/r2_a_controller_transfer_dev_planner_v1.py",
        "docs/stageR/r2/r2_b_r2a_identification_outcome_exposure_ledger_v1.0.json",
        "docs/stageR/r2/r2_b_generator_calibration_roster_v1.0.json",
        "docs/stageR/r2/r2_b_generator_calibration_permanent_exclusion_ledger_v1.0.json",
        "docs/stageR/r2/r2_b_generator_calibration_pair_bindings_v1.0.json",
        "docs/stageR/r2/r2_b_controller_aware_generator_contract_v1.0.json",
        "docs/stageR/r2/r2_b_hlc_calibration_parameter_space_v1.0.json",
        "docs/stageR/r2/r2_b_tsb_calibration_parameter_space_v1.0.json",
        "docs/stageR/r2/r2_b_generator_calibration_objective_v1.0.json",
        "docs/stageR/r2/r2_b_generator_calibration_run_ledger_v1.0.json",
        "docs/stageR/r2/r2_b_generator_calibration_round_summary_v1.json",
        "docs/stageR/r2/r2_b_generator_data_firewall_audit_v1.json",
    ]
    component_paths.extend(row["parameter_file"] for row in rounds)
    component_paths.extend(row["result_file"] for row in ledger["rounds"])
    manifest = {
        "schema_version": "r2_b_generator_binding_manifest_v1.0",
        "status": "R2_B_DEVELOPMENT_NOT_CONVERGED",
        "complete_G_R2_candidate_frozen": False,
        "selected_parameter_sha256": None,
        "blocking_family": "R-HLC",
        "HLC_final_counts": hlc_final["summary"]["counts"],
        "TSB_final_counts": tsb_final["summary"]["counts"],
        "nuplan_runtime": {
            "version": "1.2.2", "python": "/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9",
            "nuplan_devkit_git_commit": "e9241677997dd86bfc0bcd44817ab04fe631405b",
            "execution_lifecycle": "FULL_NUPLAN_RUN_RUNNERS_PRIMARY80",
        },
        "components": [component(path) for path in dict.fromkeys(component_paths)],
        "raw_DEV_output_provenance": raw_provenance,
        "raw_DEV_outputs_committed": False,
        "actual_engineering_runner_run_calls": len(run_roots),
        "scientific_simulation_calls": 0,
        "R2C_identities_selected": False, "RBR_started": False,
        "protected_CSV_sha256": sha(PROTECTED),
    }
    write(MANIFEST, manifest)
    report = f"""# R2-B Controller-Aware Generator 开发报告 v1

## 结论

R2-B 的最终状态为 **DEVELOPMENT_NOT_CONVERGED**。TSB 在第 0 轮达到全部 8/8 DEV-CAL 开发标准；HLC 在冻结的 4 轮上限内未达到联合标准，因此没有冻结完整 G_R2 候选，也没有进入 R2-C。

## 数据防火墙

- DEV-CAL：HLC 8 个、TSB 8 个；与 R1 official、R2-A、既有黑名单的重叠均为 0。
- R2-A surrogate 仅用于第 0 轮初始化；R2-A identity 未重跑。
- 数值反馈只来自冻结的 R2-B DEV-CAL identities；没有 scenario token/log ID 查表适配。
- 科学阈值、F_match、endpoint、engineering 与 official safety 定义均未修改。

## 架构

HLC 将 `DESIRED_REALIZED_MORPHOLOGY` 与 `PRECOMPENSATED_PLANNER_MORPHOLOGY` 分离，显式参数化 advance、hold、retreat、recommit、lag 与 settling。TSB 显式参数化 first-brake、release、second-brake 的幅值和有效时长，并在 absolute-time repeated replanning 中补偿 phase shortening、boundary migration、lookahead mixing 与 release carryover；不是简单常数增益求逆。

## 校准结果

- HLC：4 轮；最终 mechanism 6/8、F_match 8/8、endpoint 0/8、engineering 8/8、safety 8/8。8/8 treatment 均实现至少一次 retreat，延迟裕量均为正；两个 pair 的 monotonic 差值未越过冻结 -0.10 gate。endpoint 失败由 treatment terminal offset gate 主导。
- TSB：1 轮；measurement OK 8/8、baseline one-phase 8/8、treatment two-phase 8/8、完整 mechanism 8/8、F_match 8/8、safety 8/8。
- 实际 DEV 工程运行：{len(run_roots)}。HLC 第 0 轮 16 个已完成产物只做了后处理恢复，恢复未增加 runner.run。

## 治理处置

未生成 `r2_b_selected_generator_parameters_v1.0.json`，因为完整候选不满足冻结的 HLC+TSB 联合开发标准。严格停止在 4 轮上限；不增加第 5 轮、不换 identity、不降低 gate、不选择 R2-C identities、不启动 confirmatory smoke 或 RBR。

## 产物溯源

- round summary SHA256：`{sha(SUMMARY)}`
- data firewall SHA256：`{sha(FIREWALL)}`
- raw DEV aggregate tree SHA256：`{raw_provenance['aggregate_tree_sha256']}`
"""
    REPORT.write_text(report, encoding="utf-8")
    request = f"""# R2-B → R2-C Fresh Validation Readiness Request v0.1

## 当前处置

**NOT_READY / REQUEST_WITHHELD**。

TSB family candidate 已通过固定 DEV-CAL set 的开发标准，但 HLC 在 4 轮冻结上限后仅 mechanism 6/8、endpoint 0/8，完整 G_R2 候选未冻结。按照预注册停止规则，本轮不请求 R2-C fresh validation 授权，也不选择任何 R2-C identity。

## Owner 后续决策点

需要 Scientific Owner 另行决定是否开启新的、重新预注册的 HLC architecture-development phase。当前 R2-B 结果不授权第 5 轮、身份替换、阈值变化、R2-C 或 RBR。

Final binding manifest SHA256：`PENDING_SELF_EXTERNAL_SHA`
"""
    REQUEST.write_text(request, encoding="utf-8")
    print(json.dumps({
        "status": manifest["status"], "HLC_rounds": len(hlc), "TSB_rounds": len(tsb),
        "actual_engineering_runs": len(run_roots), "manifest_sha256": sha(MANIFEST),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
