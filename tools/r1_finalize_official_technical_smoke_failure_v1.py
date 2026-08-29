#!/usr/bin/env python3
"""Publish fail-closed R1 Phase-B2 artifacts after a pre-simulation technical stop.

This reporter never launches nuPlan, changes the raw claim ledger, substitutes
an identity, or resumes the smoke.  It is limited to the frozen case where an
official run was claimed but an unexpected executor exception prevented the
official simulator command from starting.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ROOT = Path(__file__).resolve().parents[1]
R1_DIR = ROOT / "docs/stageR/r1"
RAW_BUDGET = ROOT / "outputs/r1_official_compliant_technical_smoke_v1/official_run_budget_v1.0.json"
LEDGER = R1_DIR / "r1_official_technical_smoke_run_ledger_v1.0.csv"
PAIR = R1_DIR / "r1_official_technical_smoke_pair_metrics_v1.0.csv"
FAMILY = R1_DIR / "r1_official_technical_smoke_family_summary_v1.0.csv"
CONTEXT = R1_DIR / "r1_official_technical_smoke_context_identity_v1.0.csv"
SAFETY = R1_DIR / "r1_official_technical_smoke_safety_v1.0.csv"
MANIFEST = R1_DIR / "r1_official_technical_smoke_execution_manifest_v1.0.json"
REPORT = R1_DIR / "R1_Official_Compliant_Technical_Smoke_Report_v1.0.md"
READINESS = R1_DIR / "R1_Development_Roster_Freeze_Readiness_v0.7.md"
FAILURE = "UNEXPECTED_EXECUTOR_EXCEPTION_BEFORE_OFFICIAL_SIMULATOR_START:stage7c_environment() missing 1 required positional argument: 'args'"


def write_new_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite final artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite final artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_new_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite final artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-budget", type=Path, default=RAW_BUDGET)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw = json.loads(args.raw_budget.read_text(encoding="utf-8"))
    records = list(raw.get("records", []))
    if raw.get("authorized_cap") != 48 or raw.get("claimed_count") != 1 or len(records) != 1:
        raise ValueError("this fail-closed reporter accepts only the one-claim pre-simulation stop")
    record = dict(records[0])
    if record.get("claim_status") != "CLAIMED_BEFORE_SIMULATION" or record.get("execution_status") != "CLAIMED_NOT_STARTED":
        raise ValueError("raw budget is not the expected claimed-not-started technical stop")
    ledger_row = {**record, "execution_status": "STOPPED_BEFORE_OFFICIAL_SIMULATOR_START", "official_command_return_code": "NOT_STARTED", "technical_failure_status": "TECHNICAL_FAILURE", "technical_failure_reasons": FAILURE, "official_simulator_started": False}
    write_new_csv(LEDGER, [ledger_row], ["run_id", "scenario_token", "log_id", "family", "smoke_arm", "claim_status", "actual_run_number", "execution_status", "official_simulator_started", "official_command_return_code", "technical_failure_status", "technical_failure_reasons", "trace_sha256", "planner_binding_sha256", "canonical_metric_payload_sha256", "command_log_sha256"])
    write_new_csv(PAIR, [], ["scenario_token", "log_id", "family", "baseline_run_id", "treatment_run_id", "technical_execution_pass", "context_identity_pass", "mechanism_pair_status", "mechanism_pair_pass", "primary_f_match_status", "primary_f_match_pass", "secondary_heading_change_abs_total_delta", "endpoint_status", "endpoint_pass", "engineering_status", "engineering_pass", "pair_readiness"])
    write_new_csv(CONTEXT, [], ["scenario_token", "log_id", "family", "baseline_run_id", "treatment_run_id", "raw_history_canonical_hash_equal", "canonical_context_json_hash_equal", "pair_context_identity_pass", "baseline_raw_history_canonical_hash", "treatment_raw_history_canonical_hash", "baseline_canonical_context_json_hash", "treatment_canonical_context_json_hash"])
    write_new_csv(SAFETY, [], ["scenario_token", "log_id", "family", "baseline_run_id", "treatment_run_id", "baseline_at_fault_collision_count", "treatment_at_fault_collision_count", "baseline_drivable_area_compliance", "treatment_drivable_area_compliance", "baseline_safety_pass", "treatment_safety_pass", "pair_safety_pass"])
    family_rows = [
        {"family": "R-HLC", "required_pairs": 12, "completed_pairs": 0, "technical_execution_pass_pairs": 0, "context_identity_pass_pairs": 0, "mechanism_pair_pass_pairs": 0, "primary_f_match_pass_pairs": 0, "endpoint_pass_pairs": 0, "engineering_pass_pairs": 0, "safety_pass_pairs": 0, "readiness": "NOT_READY_FOR_FORMAL_DEVELOPMENT_ROSTER", "reason": "TECHNICAL_FAILURE_BEFORE_FIRST_OFFICIAL_SIMULATOR_START"},
        {"family": "R-TSB", "required_pairs": 12, "completed_pairs": 0, "technical_execution_pass_pairs": 0, "context_identity_pass_pairs": 0, "mechanism_pair_pass_pairs": 0, "primary_f_match_pass_pairs": 0, "endpoint_pass_pairs": "NOT_APPLICABLE", "engineering_pass_pairs": "DIAGNOSTIC_ONLY", "safety_pass_pairs": 0, "readiness": "NOT_READY_FOR_FORMAL_DEVELOPMENT_ROSTER", "reason": "BATCH_STOPPED_AFTER_R_HLC_PRE_SIMULATION_TECHNICAL_FAILURE"},
    ]
    write_new_csv(FAMILY, family_rows, ["family", "required_pairs", "completed_pairs", "technical_execution_pass_pairs", "context_identity_pass_pairs", "mechanism_pair_pass_pairs", "primary_f_match_pass_pairs", "endpoint_pass_pairs", "engineering_pass_pairs", "safety_pass_pairs", "readiness", "reason"])
    manifest = {"schema_version": "r1_official_technical_smoke_execution_manifest_v1.0", "status": "STOPPED_TECHNICAL_FAILURE_BEFORE_OFFICIAL_SIMULATOR_START", "authorized_run_cap": 48, "budget_claim_count": 1, "official_simulator_command_start_count": 0, "actual_official_closed_loop_run_count": 0, "technical_failure_count": 1, "failed_claimed_run": {"run_id": record["run_id"], "scenario_token": record["scenario_token"], "family": record["family"], "smoke_arm": record["smoke_arm"], "reason": FAILURE}, "raw_budget_path": str(args.raw_budget.resolve()), "raw_budget_modified": False, "unexecuted_frozen_schedule_count": 47, "pair_evaluation": "NOT_EVALUABLE_NO_OFFICIAL_SIMULATOR_OUTPUT", "safety": "NOT_EVALUABLE_NO_OFFICIAL_PARQUET_OUTPUT", "protocol_deviation": "NO_SCIENTIFIC_PROTOCOL_DEVIATION__TECHNICAL_EXECUTION_FAILURE_RECORDED", "continued_execution_forbidden": True, "scenario_replacement_forbidden": True, "rerun_forbidden": True, "formal_development_rollout_authorized": False, "rbr_training_authorized": False}
    write_new_json(MANIFEST, manifest)
    write_new_text(REPORT, """# R1 官方合规技术 Smoke 报告 v1.0\n\n## 结论\n\n本轮在第 1 个 pre-run claim 后、`run_simulation.py` 启动前发生 executor 环境装配异常，故依据冻结 technical-failure 规则立即停止。官方闭环 simulator 命令启动数为 `0`，实际官方 closed-loop run 数为 `0`；预算 claim 数为 `1`。未替换场景、未重跑、未继续其余 47 条日程。\n\n## 技术失败\n\n- claimed run：`R-HLC__7176d7e077925838__HLC_BASELINE_DECISIVE_MONOTONIC_LANE_CHANGE`。\n- 失败位置：executor 调用 `stage7c_environment()` 时，尚未构造/启动官方 simulator 命令。\n- 原因：`stage7c_environment() missing 1 required positional argument: 'args'`。\n- trace、planner binding、official Parquet、context identity、mechanism、F_match、endpoint、engineering 与 safety 均未产生，因此均为 `NOT_EVALUABLE`，不是 gate fail 或 scientific outcome。\n\n## Family 状态\n\n| family | 完成 pairs | 状态 | 原因 |\n|---|---:|---|---|\n| R-HLC | 0/12 | `NOT_READY_FOR_FORMAL_DEVELOPMENT_ROSTER` | 首条 simulation 前 technical failure |\n| R-TSB | 0/12 | `NOT_READY_FOR_FORMAL_DEVELOPMENT_ROSTER` | batch 已按 fail-closed 规则停止 |\n\n`R1_RESIDUAL_BENCHMARK_ENABLEMENT = GENERATOR_OR_ELIGIBILITY_REFINEMENT_REQUIRED`。这不是 formal D4 test，未产生 RBR superiority claim。\n\n## 治理结论\n\n未发生 scientific protocol deviation；记录的是技术执行失败。冻结 scope、selector salt、roster、generator 参数与 gate 均未修改。依据本轮授权，禁止在本 batch 内修复后重跑、替换 identity 或使用剩余额度。RBR-A/B/C 仍为 `NOT_AUTHORIZED`，不得开始 formal development rollout。\n""")
    write_new_text(READINESS, """# R1 Development Roster Freeze 就绪性 v0.7\n\n## 总状态：`NOT_READY_FOR_FORMAL_DEVELOPMENT_ROSTER_FREEZE_REVIEW`\n\nR1 Phase B2 的 official technical smoke 在首个 pre-run claim 后、官方 simulator 启动前发生技术异常并按冻结规则停止。虽然 V3 bound runtime replay 仍保持 `VERIFIED_ON_BOUND_RUNTIME`，本轮未生成任何 official closed-loop trace、Parquet safety metric 或合法 baseline/treatment pair，因而不能将 runtime determinism 结论外推为 generator smoke readiness。\n\n| 组件 | 状态 |\n|---|---|\n| fresh roster / selector / scope | `FROZEN_UNCHANGED` |\n| zero-budget preflight | `PASS_0_OF_48` |\n| official simulator command | `0_STARTED` |\n| budget claim ledger | `1_CLAIMED_THEN_STOPPED` |\n| R-HLC 12-pair readiness | `NOT_EVALUABLE` |\n| R-TSB 12-pair readiness | `NOT_EVALUABLE` |\n| R1 residual benchmark enablement | `GENERATOR_OR_ELIGIBILITY_REFINEMENT_REQUIRED` |\n| formal development rollout | `NOT_AUTHORIZED` |\n| RBR-A/B/C training | `NOT_AUTHORIZED` |\n\n本文件不授权重跑或修复后续跑；任何未来恢复必须由 scientific owner 以新的版本化授权决定。\n""")
    print(json.dumps({"status": manifest["status"], "actual_official_closed_loop_run_count": 0, "budget_claim_count": 1}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
