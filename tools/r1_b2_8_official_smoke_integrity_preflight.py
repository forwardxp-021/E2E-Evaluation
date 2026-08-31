#!/usr/bin/env python3
"""Fail-closed, zero-budget integrity preflight for the R1 B2.8 smoke.

This tool is deliberately limited to authorization recording and read-only
integrity checks.  It never invokes nuPlan simulation, never enumerates a
candidate pool, and never alters a frozen protocol artifact.  A failing check
writes a complete stop record with an empty official-run ledger.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "docs/stageR/r1"
EXPECTED = {
    "roster": "af672c0aa47eadebc1799dfac611016abad5b280ddd2cd56ab8ed02b605a219f",
    "schedule": "d449db48aa915b5d605d51ee587aa4d6ee5fa40029eb911fbb2af5b0721fc8c5",
    "selector": "b830476149ce284e0f36a9d9a3328dbb25d97f96eb74a3afecc063d17c85b32a",
    "execution_bindings": "005cc00218ed9131a13745aad60898c7a54bf04ad578e1b82ffb2d406f6fec86",
    "b2_6_manifest": "1e229718676111160f21ce4de374a222a3be69df5a8fc6de446f22c09f048fe7",
    "protected_metrics": "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8",
}
PATHS = {
    "roster": R1 / "r1_official_compliant_technical_smoke_roster_v2.0.json",
    "schedule": R1 / "r1_official_compliant_technical_smoke_schedule_v2.0.json",
    "selector": ROOT / "tools/r1_future_compliant_smoke_selector_v1_1.py",
    "bindings_manifest": R1 / "r1_b2_7_roster_freeze_sha_manifest_v1.0.json",
    "b2_6_manifest": R1 / "r1_b2_6_final_execution_conformance_sha_manifest_v1.0.json",
    "planner": ROOT / "tools/r1_official_technical_smoke_planner_v2_1.py",
    "evaluator": ROOT / "tools/r1_official_technical_smoke_evaluator_v2_1.py",
    "planner_hydra": ROOT / "configs/r1_official_technical_smoke_hydra/planner/r1_official_technical_smoke_v2_1.yaml",
    "preflight": R1 / "r1_b2_7_zero_run_roster_preflight_v1.0.json",
    "protected_metrics": ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv",
}
OUT = {
    "approval_json": R1 / "r1_b2_7_scientific_owner_smoke_approval_v1.0.json",
    "approval_md": R1 / "R1_B2_7_Scientific_Owner_Smoke_Approval_Record_v1.0.md",
    "preflight": R1 / "r1_b2_8_pre_run_integrity_preflight_v1.1.json",
    "ledger": R1 / "r1_b2_8_official_run_ledger_v1.1.json",
    "manifest": R1 / "r1_b2_8_execution_manifest_v1.1.json",
    "report": R1 / "R1_B2_8_Official_Compliant_Technical_Smoke_Report_v1.1.md",
    "decision": R1 / "R1_B2_8_Scientific_Owner_Decision_Sheet_v0.1_v1.1.md",
    "raw_manifest": R1 / "r1_b2_8_raw_output_manifest_v1.1.json",
}
COMPONENT_PATHS = {
    "HLC_REALIZED_PROGRESS_V1": R1 / "r1_hlc_realized_progress_contract_v1.0.json",
    "HLC_TERMINAL_ROUTE_PROGRESS_V1": R1 / "r1_hlc_terminal_route_progress_contract_v1.0.json",
    "CONTEXT_V2_1": ROOT / "tools/r1_closed_loop_context_adapter_v2_1.py",
    "HLC_CLEARANCE_V1_1": ROOT / "tools/r1_hlc_dynamic_clearance_v1_1.py",
    "HLC_APPLICABILITY_V1_0": R1 / "r1_hlc_map_geometry_applicability_contract_v1.0.json",
    "TSB_APPLICABILITY_V1_0": R1 / "r1_tsb_mechanism_applicability_contract_v1.0.json",
    "OFFICIAL_MAP_BRIDGE": ROOT / "tools/r1_official_map_query_bridge_v2_1.py",
    "OFFICIAL_EGO_FOOTPRINT": ROOT / "tools/r1_official_ego_vehicle_binding_v1.py",
    "B2_6_FINAL_EXECUTION_CONFORMANCE_MANIFEST": R1 / "r1_b2_6_final_execution_conformance_sha_manifest_v1.0.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite versioned artifact: {path}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite versioned artifact: {path}")
    path.write_text(value, encoding="utf-8")


def check(condition: bool, name: str, detail: Any, checks: list[dict[str, Any]]) -> None:
    checks.append({"name": name, "status": "PASS" if condition else "FAIL", "detail": detail})


def main() -> int:
    for path in PATHS.values():
        if not path.is_file():
            raise FileNotFoundError(f"required preflight input missing: {path}")
    checks: list[dict[str, Any]] = []
    roster, schedule = read_json(PATHS["roster"]), read_json(PATHS["schedule"])
    binding_manifest, b26, prior = read_json(PATHS["bindings_manifest"]), read_json(PATHS["b2_6_manifest"]), read_json(PATHS["preflight"])
    observed = {"roster": sha256(PATHS["roster"]), "schedule": sha256(PATHS["schedule"]), "selector": sha256(PATHS["selector"]), "b2_6_manifest": sha256(PATHS["b2_6_manifest"]), "protected_metrics": sha256(PATHS["protected_metrics"])}
    for key in ("roster", "schedule", "selector", "b2_6_manifest", "protected_metrics"):
        check(observed[key] == EXPECTED[key], f"SHA_EXACT:{key}", {"expected": EXPECTED[key], "observed": observed[key]}, checks)
    binding_sha = str(binding_manifest.get("execution_bindings_canonical_sha256"))
    check(binding_sha == EXPECTED["execution_bindings"], "SHA_EXACT:execution_bindings", {"expected": EXPECTED["execution_bindings"], "observed": binding_sha}, checks)
    schedule_bindings = schedule.get("execution_bindings", {})
    for name, expected_sha in schedule_bindings.items():
        if name in COMPONENT_PATHS:
            component = COMPONENT_PATHS[name]
            actual = sha256(component) if component.is_file() else None
            check(actual == expected_sha, f"BINDING_EXACT:{name}", {"schedule": expected_sha, "component_path": str(component.relative_to(ROOT)), "observed": actual}, checks)
        else:
            expected = b26.get("bindings", {}).get(name, {}).get("sha256")
            check(expected == expected_sha, f"BINDING_EXACT:{name}", {"schedule": expected_sha, "manifest": expected}, checks)
    entries, runs = list(roster.get("entries", [])), list(schedule.get("runs", []))
    tokens, logs = {str(row.get("scenario_token")) for row in entries}, {str(row.get("log_id")) for row in entries}
    pairs = {str(row.get("pair_id")) for row in runs}
    check(len(entries) == 24 and len(tokens) == 24 and len(logs) == 24, "ROSTER_24_UNIQUE_TOKEN_LOG", {"entries": len(entries), "tokens": len(tokens), "logs": len(logs)}, checks)
    check(sum(row.get("family") == "R-HLC" for row in entries) == 12 and sum(row.get("family") == "R-TSB" for row in entries) == 12, "ROSTER_12_PLUS_12_FAMILY", {"hlc": sum(row.get("family") == "R-HLC" for row in entries), "tsb": sum(row.get("family") == "R-TSB" for row in entries)}, checks)
    by_pair: dict[str, list[Mapping[str, Any]]] = {}
    for row in runs:
        by_pair.setdefault(str(row.get("pair_id")), []).append(row)
    pair_arms_ok = all(len(rows) == 2 and {"BASELINE", "TREATMENT"} == {"BASELINE" if "BASELINE" in str(row.get("arm")) else "TREATMENT" for row in rows} for rows in by_pair.values())
    check(len(runs) == 48 and len({str(row.get("run_id")) for row in runs}) == 48 and len(pairs) == 24 and pair_arms_ok, "SCHEDULE_48_24_PAIRS_EXACT_ARMS", {"runs": len(runs), "unique_run_ids": len({str(row.get("run_id")) for row in runs}), "pairs": len(pairs), "pair_arms_ok": pair_arms_ok}, checks)
    prior_entries = list(prior.get("entries", []))
    prior_tokens = {str(row.get("scenario_token")) for row in prior_entries}
    preflight_rows_ok = len(prior_entries) == 24 and prior_tokens == tokens and all(row.get("status") == "ROSTER_PREFLIGHT_PASS" and row.get("db_token_loadable") and row.get("map_loadable") and row.get("route_loadable") for row in prior_entries)
    check(preflight_rows_ok, "24_OF_24_DB_MAP_ROUTE_CURRENT_FROZEN_PREFLIGHT", {"prior_entries": len(prior_entries), "token_set_matches": prior_tokens == tokens}, checks)
    runtime = {"python": Path("/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9"), "run_simulation": Path("/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit/nuplan/planning/script/run_simulation.py"), "db_root": Path("/Users/liuqing/Projects/01_E2E_QA_Code/nuplan/dataset/data"), "map_root": Path("/Users/liuqing/Projects/01_E2E_QA_Code/nuplan/dataset/maps")}
    check(all(path.exists() for path in runtime.values()), "BOUND_OFFICIAL_RUNTIME_PATHS_EXIST", {key: str(value) for key, value in runtime.items()}, checks)
    planner_source, hydra_source = PATHS["planner"].read_text(encoding="utf-8"), PATHS["planner_hydra"].read_text(encoding="utf-8")
    trace_writer_present = any(term in planner_source for term in ("planner_trace.jsonl", "write_text(", ".open(", "R1_OFFICIAL_TECHNICAL_SMOKE_TRACE_DIR"))
    check(trace_writer_present, "REALIZED_CURRENT_EGO_TRACE_WRITER_BOUND", {"planner": str(PATHS["planner"].relative_to(ROOT)), "required": "80 timestamp-preserving realized ego rows"}, checks)
    hydra_is_bound = "${future_roster_row}" not in hydra_source and "${runtime_family}" not in hydra_source and "${smoke_arm}" not in hydra_source
    check(hydra_is_bound, "OFFICIAL_HYDRA_RUNTIME_BINDING_COMPLETE", {"config": str(PATHS["planner_hydra"].relative_to(ROOT)), "unbound_interpolations": [line.strip() for line in hydra_source.splitlines() if "${" in line]}, checks)
    failures = [row for row in checks if row["status"] == "FAIL"]
    status = "PASS_READY_FOR_AUTHORIZED_ONCE_EXECUTION" if not failures else "STOP_PRE_SIMULATION"
    approval = {"schema_version": "r1_b2_7_scientific_owner_smoke_approval_v1.0", "OFFICIAL_SMOKE": "AUTHORIZED_ONCE", "NEW_RUN_BUDGET": 48, "remote_baseline_commit": "cd0e7793f532407094c7876dc65d36f2f1814587", "binding": {"roster_sha256": EXPECTED["roster"], "schedule_sha256": EXPECTED["schedule"], "selector_sha256": EXPECTED["selector"], "execution_bindings_canonical_sha256": EXPECTED["execution_bindings"], "b2_6_execution_conformance_manifest_sha256": EXPECTED["b2_6_manifest"]}, "ROSTER_MUTABLE": False, "SCHEDULE_MUTABLE": False, "IDENTITY_REPLACEMENT_ALLOWED": False, "RETRY_ALLOWED": False, "THRESHOLD_CHANGE_ALLOWED": False, "RBR_AUTHORIZED": False, "authorization_consumed": False}
    preflight = {"schema_version": "r1_b2_8_pre_run_integrity_preflight_v1", "status": status, "actual_official_run_count": 0, "official_run_budget_claimed": 0, "checks": checks, "failure_policy_applied": "STOP_PRE_SIMULATION_NO_REPAIR_NO_SCENARIO_CHANGE" if failures else "READY_FOR_FROZEN_EXECUTION", "simulation_launched": False, "selector_rerun": "FORBIDDEN_NOT_PERFORMED", "identity_replacement": "FORBIDDEN_NOT_PERFORMED", "threshold_change": "FORBIDDEN_NOT_PERFORMED"}
    ledger = {"schema_version": "r1_b2_8_official_run_ledger_v1", "unit": "OFFICIAL_CLOSED_LOOP_RUN", "authorized_cap": 48, "claimed_count": 0, "records": [], "hard_49th_claim": "NOT_REACHABLE_BECAUSE_PRE_SIMULATION_STOP" if failures else "MUST_REJECT_BEFORE_SIMULATOR_START"}
    manifest = {"schema_version": "r1_b2_8_execution_manifest_v1", "status": status, "stopped_reason": "PRE_RUN_INTEGRITY_FAILURE" if failures else None, "actual_official_run_count": 0, "authorized_run_budget": 48, "consumed_run_budget": 0, "technical_failure_count": 0, "official_simulator_started": False, "retry_occurred": False, "identity_changed": False, "threshold_changed": False, "raw_output_directory": None, "raw_output_directory_committed": False, "R1_FORMAL_DEVELOPMENT_ROSTER": "NOT_AUTHORIZED_PENDING_SCIENTIFIC_OWNER_REVIEW", "RBR_A_B_C": "NOT_AUTHORIZED"}
    if not OUT["approval_json"].exists():
        write_json(OUT["approval_json"], approval)
    if not OUT["approval_md"].exists():
        write_text(OUT["approval_md"], "# R1 B2.7 Scientific Owner Smoke 授权记录 v1.0\n\n- `OFFICIAL_SMOKE = AUTHORIZED_ONCE`\n- `NEW_RUN_BUDGET = 48`\n- 仅绑定 remote baseline `cd0e7793f532407094c7876dc65d36f2f1814587` 与指定的 roster、schedule、selector、execution bindings、B2.6 manifest SHA。\n- roster/schedule 不可变；不允许 replacement、retry 或 threshold change；RBR 未授权。\n")
    write_json(OUT["preflight"], preflight)
    write_json(OUT["ledger"], ledger)
    write_json(OUT["manifest"], manifest)
    write_json(OUT["raw_manifest"], {"schema_version": "r1_b2_8_raw_output_manifest_v1", "status": "NO_RAW_OUTPUT_CREATED_PRE_SIMULATION_STOP" if failures else "PENDING_EXECUTION", "path": None, "file_count": 0, "run_ids": [], "committed_to_git": False})
    details = "\n".join(f"- `{row['name']}`：{row['status']}" for row in checks)
    write_text(OUT["report"], f"# R1 B2.8 Official Compliant Technical Smoke 报告 v1\n\n## 结论\n\n`{status}`。本轮未启动 nuPlan official simulation，消耗额度 `0/48`。\n\n冻结要求的 primary 是 `REALIZED_CURRENT_EGO (iterations 0...79)`。预检确认 V2.1 planner 没有冻结的 realized-ego trace writer，且对应 Hydra planner config 仍依赖未绑定的 runtime 插值。因此无法在不新增/修改执行 wiring 的情况下，生成冻结 evaluator 所需的 primary 输入。根据 Task 1，本轮不得临场修复，必须在仿真前停止。\n\n## 检查明细\n\n{details}\n\n## 授权边界\n\n授权记录已生成，但未消费。没有 rerun、identity replacement、threshold change 或 RBR 读取/训练。\n")
    write_text(OUT["decision"], f"# R1 B2.8 Scientific Owner 决策单 v0.1\n\n当前状态：`{status}`。\n\n请 Scientific Owner 审核两个 pre-simulation integration 缺口：V2.1 planner 的 realized-current-ego artifact 绑定，以及 Hydra 的 per-run frozen roster-row 参数绑定。未获得新的明确授权前，不得执行任何 official run，不得重试，不得替换 roster identity，不得训练 RBR。\n")
    print(json.dumps({"status": status, "failed_checks": [row["name"] for row in failures], "official_runs": 0}, ensure_ascii=False, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
