#!/usr/bin/env python3
"""Run the strictly isolated 48-rollout R1 Phase-B technical smoke.

The program uses only historical/R0-development manifests and deterministic
trajectory math.  It never opens representation, BDD, probe, checkpoint or
RBR assets and intentionally writes no trajectory tensors.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_context_mechanism_core import (assert_pair_context_identity, build_canonical_context_record, calculate_hlc_option_b, calculate_tsb_option_a, canonical_json_sha256, frozen_f_match, qualify_hlc_pair, qualify_tsb_pair, trajectory_descriptors)
from tools.r1_residual_generators import (HLC_SMOKE_CANDIDATES, TSB_BASELINE, TSB_SMOKE_CANDIDATES, generate_hlc_trajectory, generate_tsb_trajectory, kinematic_integrity)


ROOT = Path(__file__).resolve().parents[1]
HLC_MANIFEST = ROOT / "outputs/stage7l_b_final_development_freeze_v1/final_development_maneuver_manifest.json"
HLC_ROSTER = ROOT / "outputs/stage7l_b_final_development_freeze_v1/final_development_roster.csv"
TSB_METADATA = ROOT / "outputs/stage6j_pure_longitudinal_context_v1/metadata.csv"
OUT_DIR = ROOT / "docs/stageR/r1"
ROSTER_SALT = "R1_PHASEB_TECHNICAL_SMOKE_ROSTER_V1"
AUTHORIZED_CORE_CONSTRUCTION_CAP = 48


class CoreConstructionBudget:
    """Pre-construction hard stop for trajectory-core calls."""

    LEDGER_SCHEMA = (
        "sequence",
        "family",
        "scenario_id",
        "arm_type",
        "candidate_id",
        "call_key",
        "claim_status",
        "planned_call_count",
        "actual_call_number",
        "authorized_cap",
    )

    def __init__(
        self,
        authorized_cap: int = AUTHORIZED_CORE_CONSTRUCTION_CAP,
        planned_schedule: Sequence[Mapping[str, str]] | None = None,
    ) -> None:
        if authorized_cap <= 0:
            raise ValueError("authorized core-construction cap must be positive")
        self.authorized_cap = int(authorized_cap)
        self.actual_calls = 0
        self.ledger: List[Dict[str, Any]] = []
        self._claimed_keys: set[str] = set()
        self._baseline_keys: set[str] = set()
        self._planned_keys = {
            self._call_key(str(row["family"]), str(row["scenario_id"]), str(row["arm"]))
            for row in (planned_schedule or ())
        }
        self.planned_call_count = len(planned_schedule) if planned_schedule is not None else authorized_cap

    @staticmethod
    def _call_key(family: str, scenario_id: str, arm: str) -> str:
        return f"{family}|{scenario_id}|{arm}"

    def claim(self, family: str, scenario_id: str, arm: str) -> None:
        """Reserve one call before construction; refuse the first excess call."""
        family = str(family)
        scenario_id = str(scenario_id)
        arm = str(arm)
        call_key = self._call_key(family, scenario_id, arm)
        baseline_key = f"{family}|{scenario_id}"
        if arm == "BASELINE" and baseline_key in self._baseline_keys:
            raise RuntimeError(
                f"duplicate baseline construction blocked before call: {family}/{scenario_id}"
            )
        if call_key in self._claimed_keys:
            raise RuntimeError(f"duplicate trajectory construction blocked before call: {call_key}")
        if self._planned_keys and call_key not in self._planned_keys:
            raise RuntimeError(f"unplanned trajectory construction blocked before call: {call_key}")
        if self.actual_calls >= self.authorized_cap:
            raise RuntimeError(
                "trajectory-core construction blocked before exceeding authorized "
                f"cap {self.authorized_cap}: {family}/{scenario_id}/{arm}"
            )
        self.actual_calls += 1
        self._claimed_keys.add(call_key)
        if arm == "BASELINE":
            self._baseline_keys.add(baseline_key)
        arm_type = "BASELINE" if arm == "BASELINE" else "TREATMENT"
        candidate_id = "NOT_APPLICABLE" if arm_type == "BASELINE" else arm.removeprefix("TREATMENT::")
        self.ledger.append(
            {
                "sequence": self.actual_calls,
                "family": family,
                "scenario_id": scenario_id,
                "arm_type": arm_type,
                "candidate_id": candidate_id,
                "call_key": call_key,
                "claim_status": "CLAIMED_BEFORE_CONSTRUCTION",
                "planned_call_count": self.planned_call_count,
                "actual_call_number": self.actual_calls,
                "authorized_cap": self.authorized_cap,
            }
        )

    def assert_exact(self, expected: int) -> None:
        if self.actual_calls != expected:
            raise RuntimeError(
                f"expected exactly {expected} trajectory-core calls, got {self.actual_calls}"
            )
        if self.actual_calls > self.authorized_cap:
            raise RuntimeError("trajectory-core construction cap exceeded")
        if self.ledger and tuple(self.ledger[0]) != self.LEDGER_SCHEMA:
            raise RuntimeError("trajectory-core construction ledger schema changed")
        if self._planned_keys and self._claimed_keys != self._planned_keys:
            raise RuntimeError("actual trajectory-core ledger does not equal the preflight schedule")

    def counters(self) -> Dict[str, int]:
        return {
            "planned_core_construction_calls": self.planned_call_count,
            "actual_core_construction_calls": self.actual_calls,
            "authorized_cap": self.authorized_cap,
        }


def build_core_construction_schedule(
    family_scenario_ids: Mapping[str, Sequence[str]],
    family_candidate_ids: Mapping[str, Sequence[str]],
    authorized_cap: int = AUTHORIZED_CORE_CONSTRUCTION_CAP,
) -> List[Dict[str, str]]:
    """Build and preflight a baseline-reuse schedule without constructing trajectories."""
    schedule: List[Dict[str, str]] = []
    for family in ("R-HLC", "R-TSB"):
        scenarios = list(family_scenario_ids.get(family, ()))
        candidates = list(family_candidate_ids.get(family, ()))
        if len(scenarios) != 6 or len(set(scenarios)) != 6:
            raise ValueError(f"{family} schedule requires exactly six unique scenarios")
        if len(candidates) != 3 or len(set(candidates)) != 3:
            raise ValueError(f"{family} schedule requires exactly three unique candidates")
        for scenario_id in scenarios:
            schedule.append(
                {"family": family, "scenario_id": str(scenario_id), "arm": "BASELINE"}
            )
            schedule.extend(
                {
                    "family": family,
                    "scenario_id": str(scenario_id),
                    "arm": f"TREATMENT::{candidate_id}",
                }
                for candidate_id in candidates
            )
    if len(schedule) > authorized_cap:
        raise RuntimeError(
            f"planned {len(schedule)} core constructions exceeds authorized cap {authorized_cap}"
        )
    if len(schedule) != 48:
        raise RuntimeError(f"compliant smoke schedule must contain exactly 48 calls, got {len(schedule)}")
    return schedule


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite an existing smoke artifact: {path}")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def stable_rank(family: str, token: str, log_id: str) -> str:
    return hashlib.sha256(f"{ROSTER_SALT}|{family}|{token}|{log_id}".encode("utf-8")).hexdigest()


def load_hlc_candidates() -> List[Dict[str, Any]]:
    with HLC_MANIFEST.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    maneuvers = {str(row["scenario_token"]): row for row in manifest["maneuvers"]}
    with HLC_ROSTER.open("r", encoding="utf-8", newline="") as handle:
        roster = {str(row["scenario_token"]): row for row in csv.DictReader(handle) if row.get("eligible") == "True"}
    candidates = []
    for token, maneuver in maneuvers.items():
        if token not in roster:
            continue
        candidates.append({"scenario_token": token, "log_id": str(maneuver["log_name"]), "map_location": roster[token]["map_name"], "maneuver": maneuver})
    return candidates


def load_tsb_candidates() -> List[Dict[str, Any]]:
    chosen: Dict[str, Dict[str, Any]] = {}
    with TSB_METADATA.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            token = str(row["scenario_token"])
            if token not in chosen or int(row["global_row"]) < int(chosen[token]["global_row"]):
                chosen[token] = row
    return [{"scenario_token": token, "log_id": str(row["log_name"]), "map_location": str(row["map_name"]), "metadata": row} for token, row in chosen.items()]


def select_six(family: str, candidates: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    ordered = sorted((dict(row) for row in candidates), key=lambda row: stable_rank(family, str(row["scenario_token"]), str(row["log_id"])))
    selected = ordered[:6]
    if len(selected) != 6 or len({row["log_id"] for row in selected}) < 3:
        raise RuntimeError(f"{family} deterministic source cannot produce 6 scenarios from >=3 logs")
    return selected


def warmup_frames(initial_speed_mps: float) -> List[Dict[str, Any]]:
    return [{"time_s": round(i * 0.1, 6), "ego_valid": True, "map_valid": True, "current_required_lane_valid": True, "speed_mps": round(float(initial_speed_mps), 6), "lane_offset_m": 0.0, "legal_projected_dynamic_vehicle_count": 0, "slots": {name: {"valid": False} for name in ("front", "left_front", "left_rear", "right_front", "right_rear")}} for i in range(10)]


def hlc_context_payload(item: Mapping[str, Any]) -> Dict[str, Any]:
    maneuver = item["maneuver"]
    frames = warmup_frames(float(maneuver["initial_speed_mps"]))
    for frame in frames:
        frame["target_front"] = {"valid": False}
        frame["target_rear"] = {"valid": False}
    return {"family": "R-HLC", "t_anchor_s": 1.0, "scenario_token": item["scenario_token"], "map_version": f"HISTORICAL_STAGE7L::{item['map_location']}", "route_fingerprint": maneuver["route_fingerprint"], "initial_state_fingerprint": maneuver["initial_state_fingerprint"], "map_location": item["map_location"], "road_class": "SOURCE_LANE__TARGET_LANE__SAME_ROADBLOCK__ADJACENT_EDGE", "log_id": item["log_id"], "intended_lane_change_direction": str(maneuver["direction"]).upper(), "history_source": "CONDITION_IDENTICAL_1S_WARMUP", "map_source_ids": {"source_lane_id": maneuver["source_lane_id"], "target_lane_id": maneuver["target_lane_id"], "source_roadblock_id": maneuver["source_roadblock_id"], "target_roadblock_id": maneuver["target_roadblock_id"], "source_manifest": str(HLC_MANIFEST.relative_to(ROOT))}, "query_version": "R1_TECHNICAL_SMOKE_STAGE7L_MANIFEST_ADAPTER_V1", "frames": frames}


def tsb_context_payload(item: Mapping[str, Any]) -> Dict[str, Any]:
    metadata = item["metadata"]
    frames = warmup_frames(8.0)
    for frame in frames:
        frame["front"] = {"valid": False}
    return {"family": "R-TSB", "t_anchor_s": 1.0, "scenario_token": item["scenario_token"], "map_version": f"TECHNICAL_SMOKE_LOCAL_FRAME::{item['map_location']}", "route_fingerprint": canonical_json_sha256({"source": "stage6j_metadata", "scenario_token": item["scenario_token"], "log_id": item["log_id"]}), "initial_state_fingerprint": canonical_json_sha256({"source": "stage6j_metadata", "scenario_token": item["scenario_token"], "initial_speed_mps": 8.0}), "map_location": item["map_location"], "road_class": "TECHNICAL_SMOKE_LOCAL_STRAIGHT_REFERENCE", "log_id": item["log_id"], "history_source": "CONDITION_IDENTICAL_1S_WARMUP", "map_source_ids": {"current_lane_id": "TECHNICAL_SMOKE_LOCAL_FRAME", "source_metadata": str(TSB_METADATA.relative_to(ROOT)), "historical_global_row": metadata["global_row"]}, "query_version": "R1_TECHNICAL_SMOKE_LOCAL_FRAME_V1", "hazard_multi_hot": ["NONE_OBSERVED"], "frames": frames}


def summarize(rows: Sequence[Mapping[str, Any]], family: str, candidate_id: str) -> Dict[str, Any]:
    current = [row for row in rows if row["family"] == family and row["candidate_id"] == candidate_id]
    count = len(current)
    return {"family": family, "candidate_id": candidate_id, "scenario_count": count, "rollout_count": count * 2, "technical_execution_pass_count": sum(row["technical_execution_pass"] for row in current), "pre_context_identity_pass_count": sum(row["pre_context_identity_pass"] for row in current), "canonical_context_identity_pass_count": sum(row["canonical_context_identity_pass"] for row in current), "f_match_pass_count": sum(row["f_match_pass"] for row in current), "mechanism_pair_pass_count": sum(row["mechanism_pair_pass"] for row in current), "kinematic_integrity_pass_count": sum(row["kinematic_integrity_pass"] for row in current), "mean_pair_runtime_ms": round(sum(row["pair_runtime_ms"] for row in current) / count, 3), "technical_recommendation_status": "RECOMMENDED_AFTER_TECHNICAL_SMOKE" if count and all(row["technical_execution_pass"] and row["pre_context_identity_pass"] and row["canonical_context_identity_pass"] and row["f_match_pass"] and row["mechanism_pair_pass"] and row["kinematic_integrity_pass"] for row in current) else "NOT_RECOMMENDED_AFTER_TECHNICAL_SMOKE", "scope": "TECHNICAL_SMOKE_CORE_ONLY_NOT_FORMAL_GENERATOR_FREEZE"}


def write_report(path: Path, roster: Mapping[str, Any], summary: Sequence[Mapping[str, Any]], runtime_available: bool) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite report: {path}")
    rows = "\n".join(f"|{row['family']}|{row['candidate_id']}|{row['technical_execution_pass_count']}/{row['scenario_count']}|{row['f_match_pass_count']}/{row['scenario_count']}|{row['mechanism_pair_pass_count']}/{row['scenario_count']}|{row['kinematic_integrity_pass_count']}/{row['scenario_count']}|{row['technical_recommendation_status']}|" for row in summary)
    content = f"""# R1 技术烟雾报告 v1

状态：`TECHNICAL_SMOKE_COMPLETE_CORE_ONLY`。本次恰好执行 48 条 trajectory-only technical rollouts：R-HLC 6 个历史/R0-development scenario 的 baseline+3 candidates（24 条）和 R-TSB 对应 24 条。没有创建 48/58 正式 development roster，没有读取 embedding、BDD、probe、checkpoint 或 RBR。

## roster 与隔离

- roster 由固定 salt `{ROSTER_SALT}` 对历史/R0-development source 的 scenario token/log 做 deterministic hash 排序；各 family 取 6 个、均至少 3 个 logs。
- 全部条目为 `TECHNICAL_SMOKE_ONLY`、`EXCLUDED_FROM_R1_DEVELOPMENT_ROSTER` 与 `EXCLUDED_FROM_FUTURE_R4_CONFIRMATION`。
- pre-context 使用被批准的 `CONDITION_IDENTICAL_1S_WARMUP`（完整 10 帧、0.1s），不是缩短窗口。每个 pair 均在生成前核验 raw history 与 canonical context hash 相同。

## 结果

|family|candidate|技术执行|F_match|机制 pair gate|运动学完整性|建议|
|---|---|---:|---:|---:|---:|---|
{rows}

## 安全与 runtime 边界

所有显示为运动学完整性的结果仅代表有限值、时间单调、非负速度及 HLC 横向加速度/yaw/curvature 的预声明上限。`nuplan` runtime 可用性为 `{runtime_available}`；本机缺少完整 external runtime 时，未声称 official closed-loop background replay、碰撞/off-road 或 traffic-light API safety 已通过。因此任何 `RECOMMENDED_AFTER_TECHNICAL_SMOKE` 仅是 core generator 的技术建议，不是正式 generator freeze 或 scientific efficacy 结论。

## 不可变性

候选参数 JSON 和 smoke roster 均在任何 candidate rollout 前写入并由 execution manifest SHA 绑定。未因本报告中的通过或失败修改 context/mechanism 定义、F_match caliper 或 threshold。
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute exactly 48 isolated R1 technical-smoke rollouts.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="New small-artifact directory (default: docs/stageR/r1).")
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    required = (HLC_MANIFEST, HLC_ROSTER, TSB_METADATA)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"technical smoke source assets missing: {missing}")
    paths = {"candidate": out_dir / "r1_smoke_candidate_parameters_v1.json", "roster": out_dir / "r1_technical_smoke_roster_v1.json", "metrics": out_dir / "r1_smoke_candidate_metrics_v1.csv", "manifest": out_dir / "r1_technical_smoke_execution_manifest_v1.json", "report": out_dir / "R1_Technical_Smoke_Report_v1.md"}
    existing_non_candidate = [str(path) for key, path in paths.items() if key != "candidate" and path.exists()]
    if existing_non_candidate:
        raise FileExistsError(f"refusing to overwrite smoke outputs: {existing_non_candidate}")
    selected_hlc = select_six("R-HLC", load_hlc_candidates())
    selected_tsb = select_six("R-TSB", load_tsb_candidates())
    planned_schedule = build_core_construction_schedule(
        {
            "R-HLC": [str(row["scenario_token"]) for row in selected_hlc],
            "R-TSB": [str(row["scenario_token"]) for row in selected_tsb],
        },
        {
            "R-HLC": list(HLC_SMOKE_CANDIDATES),
            "R-TSB": list(TSB_SMOKE_CANDIDATES),
        },
    )
    if len(planned_schedule) != AUTHORIZED_CORE_CONSTRUCTION_CAP:
        raise RuntimeError("technical-smoke construction schedule failed the 48-call preflight")
    candidate_payload = {"schema_version": "r1_smoke_candidate_parameters_v1", "status": "PREDECLARED_BEFORE_TECHNICAL_SMOKE", "scope": "TRAJECTORY_ONLY_NO_REPRESENTATION_OUTCOMES", "hlc": {"baseline": {"profile": "decisive_quintic_transition", "transition_seconds": 2.0}, "candidates": HLC_SMOKE_CANDIDATES}, "tsb": {"baseline": TSB_BASELINE, "candidates": TSB_SMOKE_CANDIDATES}, "solver": {"status": "F_MATCH_CONSTRAINED_TRAJECTORY_ONLY_PRECHECK", "input_allowlist": ["raw_generated_trajectory", "frozen_F_match_descriptors", "frozen_development_calipers", "mechanism_variables", "physical_safety_constraints"], "objective_constraint_order": ["finite_and_time_integrity", "frozen_F_match", "mechanism_pair_gate", "kinematic_integrity"], "forbidden_inputs": ["embedding", "BDD", "probe", "representation", "RBR"]}}
    if paths["candidate"].exists():
        with paths["candidate"].open("r", encoding="utf-8") as handle:
            existing_candidate = json.load(handle)
        if existing_candidate != candidate_payload:
            raise RuntimeError("existing predeclared candidate file does not exactly match frozen candidate payload")
    else:
        write_new_json(paths["candidate"], candidate_payload)
    roster_payload = {"schema_version": "r1_technical_smoke_roster_v1", "status": "TECHNICAL_SMOKE_ONLY", "selection": {"method": "outcome_blind_deterministic_sha256_rank", "salt": ROSTER_SALT, "source_scope": "HISTORICAL_OR_R0_DEVELOPMENT_ONLY", "formal_r1_development_roster_created": False, "exclusions": ["EXCLUDED_FROM_R1_DEVELOPMENT_ROSTER", "EXCLUDED_FROM_FUTURE_R4_CONFIRMATION"]}, "families": {"R-HLC": [{"scenario_token": row["scenario_token"], "log_id": row["log_id"], "map_location": row["map_location"], "source": str(HLC_MANIFEST.relative_to(ROOT)), "rank_sha256": stable_rank("R-HLC", row["scenario_token"], row["log_id"])} for row in selected_hlc], "R-TSB": [{"scenario_token": row["scenario_token"], "log_id": row["log_id"], "map_location": row["map_location"], "source": str(TSB_METADATA.relative_to(ROOT)), "rank_sha256": stable_rank("R-TSB", row["scenario_token"], row["log_id"])} for row in selected_tsb]}}
    write_new_json(paths["roster"], roster_payload)
    rows: List[Dict[str, Any]] = []
    construction_budget = CoreConstructionBudget(planned_schedule=planned_schedule)
    for family, selected in (("R-HLC", selected_hlc), ("R-TSB", selected_tsb)):
        for item in selected:
            context = build_canonical_context_record(hlc_context_payload(item) if family == "R-HLC" else tsb_context_payload(item))
            # Exactly one baseline arm is constructed per scenario.  It is then
            # paired with each fixed treatment candidate, preserving the hard
            # 6 x (baseline + 3 treatment) rollout accounting.
            construction_budget.claim(family, str(item["scenario_token"]), "BASELINE")
            if family == "R-HLC":
                baseline_traj = generate_hlc_trajectory(item["maneuver"])
                baseline_mechanism = calculate_hlc_option_b(baseline_traj["time_s"], baseline_traj["progress_p"], baseline_traj["speed_mps"])
            else:
                baseline_traj = generate_tsb_trajectory(8.0)
                baseline_mechanism = calculate_tsb_option_a(baseline_traj["time_s"], baseline_traj["speed_mps"])
            baseline_desc = trajectory_descriptors(baseline_traj["time_s"], baseline_traj["xy"], baseline_traj["speed_mps"])
            baseline_integrity = kinematic_integrity(baseline_traj)
            for candidate_id in (HLC_SMOKE_CANDIDATES if family == "R-HLC" else TSB_SMOKE_CANDIDATES):
                began = time.perf_counter()
                construction_budget.claim(
                    family, str(item["scenario_token"]), f"TREATMENT::{candidate_id}"
                )
                if family == "R-HLC":
                    treatment_traj = generate_hlc_trajectory(item["maneuver"], candidate_id)
                    treatment_mechanism = calculate_hlc_option_b(treatment_traj["time_s"], treatment_traj["progress_p"], treatment_traj["speed_mps"])
                    from tools.r1_context_mechanism_core import qualify_hlc_pair
                    mechanism = qualify_hlc_pair(baseline_mechanism, treatment_mechanism)
                else:
                    treatment_traj = generate_tsb_trajectory(8.0, candidate_id)
                    treatment_mechanism = calculate_tsb_option_a(treatment_traj["time_s"], treatment_traj["speed_mps"])
                    from tools.r1_context_mechanism_core import qualify_tsb_pair
                    mechanism = qualify_tsb_pair(baseline_mechanism, treatment_mechanism)
                treatment_desc = trajectory_descriptors(treatment_traj["time_s"], treatment_traj["xy"], treatment_traj["speed_mps"])
                f_match = frozen_f_match(baseline_desc, treatment_desc, family)
                baseline_integrity = kinematic_integrity(baseline_traj)
                treatment_integrity = kinematic_integrity(treatment_traj)
                pair_context = assert_pair_context_identity(context, context)
                runtime_ms = (time.perf_counter() - began) * 1000.0
                rows.append({"family": family, "candidate_id": candidate_id, "scenario_token": item["scenario_token"], "technical_execution_pass": bool(baseline_integrity["pass"] and treatment_integrity["pass"]), "pre_context_identity_pass": pair_context["fields"]["pre_context_raw_hash"], "canonical_context_identity_pass": pair_context["fields"]["canonical_context_json_hash"], "f_match_pass": f_match["pass"], "mechanism_pair_pass": mechanism["pass"], "kinematic_integrity_pass": bool(baseline_integrity["pass"] and treatment_integrity["pass"]), "pair_runtime_ms": round(runtime_ms, 3), "baseline_mechanism_status": baseline_mechanism["status"], "treatment_mechanism_status": treatment_mechanism["status"], "mechanism_pair_status": mechanism["status"], "f_match_status": f_match["status"], "baseline_descriptors": baseline_desc, "treatment_descriptors": treatment_desc, "f_match": f_match, "pair_context": pair_context, "safety_scope": "KINEMATIC_ONLY_NOT_OFFICIAL_CLOSED_LOOP_SAFETY"})
    if len(rows) != 36:
        raise RuntimeError(f"expected 36 baseline-treatment candidate pairs / 48 rollouts, got {len(rows)} pairs")
    construction_budget.assert_exact(AUTHORIZED_CORE_CONSTRUCTION_CAP)
    summary = [summarize(rows, "R-HLC", candidate) for candidate in HLC_SMOKE_CANDIDATES] + [summarize(rows, "R-TSB", candidate) for candidate in TSB_SMOKE_CANDIDATES]
    with paths["metrics"].open("x", encoding="utf-8", newline="") as handle:
        columns = list(summary[0].keys())
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary)
    runtime_available = importlib.util.find_spec("nuplan") is not None
    manifest = {"schema_version": "r1_technical_smoke_execution_manifest_v1", "status": "TECHNICAL_SMOKE_COMPLETE_CORE_ONLY", "baseline_commit": "ef78536f852e2f1fb0c0f66e928c4da2282eda6c", "frozen_context_mechanism_contract": "R1_CONTEXT_MECHANISM_CONTRACT_V1_FROZEN", "candidate_parameters_sha256": sha256_file(paths["candidate"]), "roster_sha256": sha256_file(paths["roster"]), "metrics_sha256": sha256_file(paths["metrics"]), "planned_rollouts": 48, "executed_rollouts": 48, "construction_accounting": {**construction_budget.counters(), "claim_timing": "IMMEDIATELY_BEFORE_EACH_CORE_CONSTRUCTION", "fail_closed": True, "ledger_schema": list(CoreConstructionBudget.LEDGER_SCHEMA), "ledger": construction_budget.ledger}, "pairs": rows, "summary": summary, "external_runtime": {"nuplan_available": runtime_available, "official_closed_loop_background_replay_executed": False, "traffic_light_route_api_executed": False, "result": "CORE_SMOKE_ONLY_RUNTIME_INTEGRATION_PENDING" if not runtime_available else "RUNTIME_PRESENT_BUT_NOT_USED_FOR_FORMAL_ROLLOUT"}, "prohibited_reads": ["embedding", "BDD", "probe", "checkpoint", "RBR"], "formal_roster_created": False, "r4_data_used": False}
    write_new_json(paths["manifest"], manifest)
    write_report(paths["report"], roster_payload, summary, runtime_available)
    print(json.dumps({"executed_rollouts": 48, "metrics": str(paths["metrics"]), "manifest": str(paths["manifest"]), "recommendations": {f"{row['family']}:{row['candidate_id']}": row["technical_recommendation_status"] for row in summary}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
