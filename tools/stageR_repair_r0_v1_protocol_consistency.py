#!/usr/bin/env python3
"""Generate the StageR/R0 v1 protocol-consistency repair artifacts.

This governance-only generator reads existing protocol/target/statistical manifests
and development descriptor summaries.  It does not read representations, BDD
results, planner rollout results, or future outcomes.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path.cwd()
BASE = ROOT / "docs/stageR/r0"
PROTOCOL = BASE / "protocol"
GOVERNANCE = BASE / "governance"
MANIFEST = BASE / "manifests"

FAMILIES = ("R-HLC", "R-TSB", "R-IP")
F_MATCH = {
    "R-HLC": [
        "ego13.mean_speed", "ego13.end_minus_start_speed",
        "ego13.heading_change_abs_total", "ego13.path_length",
    ],
    "R-TSB": [
        "ego13.mean_speed", "ego13.end_minus_start_speed",
        "ego13.mean_abs_accel", "ego13.path_length",
    ],
    "R-IP": [
        "ego13.mean_speed", "ego13.end_minus_start_speed", "ego13.path_length",
    ],
}
M_BEHAVIOR = {
    "R-HLC": [
        "raw33.lane_change_count_proxy", "raw33.lane_change_duration_mean_proxy",
        "raw33.lane_change_oscillation_score_proxy",
        "mechanism.hesitation_retreat_count", "mechanism.commit_latency_s",
        "mechanism.monotonic_transition_fraction",
    ],
    "R-TSB": [
        "raw33.rms_accel", "raw33.rms_jerk", "raw33.max_abs_jerk",
        "mechanism.brake_phase_count", "mechanism.interstage_release_fraction",
        "mechanism.second_brake_peak_ratio",
    ],
    "R-IP": [
        "raw33.left_gap_acceptance_proxy", "raw33.right_gap_acceptance_proxy",
        "raw33.yielding_score_proxy", "raw33.assertiveness_score_proxy",
        "mechanism.gap_acceptance_latency_s", "mechanism.minimum_accepted_rear_gap_m",
        "mechanism.yield_response_onset_s",
    ],
}
CONTEXT_MATCH = {
    "R-HLC": [
        "context.map_location", "context.road_class", "context.log_id",
        "context.intended_lane_change_direction", "context.initial_speed_mps",
        "context.initial_lane_offset_m", "context.traffic_density",
        "context.neighbor_availability_pattern", "context.target_lane_initial_front_gap_m",
        "context.target_lane_initial_rear_gap_m",
    ],
    "R-TSB": [
        "context.map_location", "context.road_class", "context.log_id",
        "context.initial_speed_mps", "context.initial_front_gap_m",
        "context.initial_lead_relative_speed_mps", "context.initial_thw_s",
        "context.traffic_density", "context.neighbor_availability_pattern",
        "context.planned_stop_or_hazard_class",
    ],
    "R-IP": [
        "context.map_location", "context.road_class", "context.log_id",
        "context.intended_lane_change_direction", "context.initial_speed_mps",
        "context.traffic_density", "context.neighbor_availability_pattern",
        "context.target_lane_initial_front_gap_m", "context.target_lane_initial_rear_gap_m",
        "context.target_lane_initial_rear_closing_speed_mps", "context.gap_opportunity_class",
    ],
}

CONTEXT_DEFINITIONS = {
    "context.map_location": ("category", "Map/location identity known before treatment"),
    "context.road_class": ("category", "Road/lane topology class known before treatment"),
    "context.log_id": ("identifier", "Log cluster identity; matching/blocking only"),
    "context.intended_lane_change_direction": ("category", "Pre-treatment intended maneuver direction"),
    "context.initial_speed_mps": ("m/s", "Ego speed at frozen pre-treatment anchor"),
    "context.initial_lane_offset_m": ("m", "Ego lane offset at pre-treatment anchor"),
    "context.traffic_density": ("count or frozen bin", "Pre-treatment traffic-density measure"),
    "context.neighbor_availability_pattern": ("bit pattern", "Pre-treatment semantic-slot availability"),
    "context.target_lane_initial_front_gap_m": ("m", "Target-lane front gap at pre-treatment anchor"),
    "context.target_lane_initial_rear_gap_m": ("m", "Target-lane rear gap at pre-treatment anchor"),
    "context.initial_front_gap_m": ("m", "Current-lane front gap at pre-treatment anchor"),
    "context.initial_lead_relative_speed_mps": ("m/s", "Lead relative speed at pre-treatment anchor"),
    "context.initial_thw_s": ("s", "THW at pre-treatment anchor with frozen validity/sentinel rule"),
    "context.planned_stop_or_hazard_class": ("category", "Pre-treatment route/hazard intent class"),
    "context.target_lane_initial_rear_closing_speed_mps": ("m/s", "Target-lane rear closing speed at pre-treatment anchor"),
    "context.gap_opportunity_class": ("category", "Pre-treatment gap-opportunity stratum"),
}

MECHANISM_DEFINITIONS = {
    "mechanism.hesitation_retreat_count": ("count", "Count of predeclared retreat/reversal episodes before commitment"),
    "mechanism.commit_latency_s": ("s", "Time from opportunity/intent anchor to committed transition"),
    "mechanism.monotonic_transition_fraction": ("fraction", "Fraction of transition progress consistent with committed direction"),
    "mechanism.brake_phase_count": ("count", "Count of separated braking phases under frozen acceleration thresholds"),
    "mechanism.interstage_release_fraction": ("fraction", "Release magnitude between first and second braking phases"),
    "mechanism.second_brake_peak_ratio": ("ratio", "Second-to-first peak braking magnitude ratio"),
    "mechanism.gap_acceptance_latency_s": ("s", "Time from frozen gap opportunity to commitment"),
    "mechanism.minimum_accepted_rear_gap_m": ("m", "Rear gap at frozen acceptance event"),
    "mechanism.yield_response_onset_s": ("s", "Latency from interaction cue to yielding response onset"),
}

D1_CORE = {
    "longitudinal": [
        ("ego13.mean_speed", "continuous", "R2"),
        ("ego13.end_minus_start_speed", "continuous", "R2"),
        ("ego13.rms_accel", "continuous", "R2"),
    ],
    "lateral": [
        ("ego13.rms_yaw_rate", "continuous", "R2"),
        ("ego13.heading_change_abs_total", "continuous", "R2"),
        ("raw33.lane_change_count_proxy", "categorical:any_count_gt_0", "balanced_accuracy"),
    ],
    "interaction": [
        ("raw33.mean_front_distance", "continuous", "R2"),
        ("raw33.mean_rel_speed", "continuous", "R2"),
        ("raw33.front_pressure_score", "continuous", "R2"),
    ],
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def evidence_by_target() -> dict[str, dict[str, str]]:
    path = MANIFEST / "r0_equivalence_margin_evidence_v0.1.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["target_id"]: row for row in csv.DictReader(handle)}


def make_role_rows(target_v1: dict[str, Any]) -> list[dict[str, Any]]:
    target_by_id = {row["target_id"]: row for row in target_v1["targets"]}
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        if set(F_MATCH[family]) & set(M_BEHAVIOR[family]):
            raise RuntimeError(f"Primary F/M overlap in {family}")
        for target_id, target in target_by_id.items():
            role = "F_match" if target_id in F_MATCH[family] else "M_behavior" if target_id in M_BEHAVIOR[family] else "Semantic_probe_only"
            rationale = {
                "F_match": "core fixed-summary descriptor controlled for this residual family",
                "M_behavior": "post-treatment morphology/mechanism qualification; never Primary F_match",
                "Semantic_probe_only": "retained for D1 semantic probing or secondary description; not a Primary D4 matching gate",
            }[role]
            rows.append({
                "residual_family": family,
                "feature_id": target_id,
                "role": role,
                "source_kind": "existing_target_v0.1",
                "unit": target["unit"],
                "pre_or_post_treatment": "post-treatment_or_whole-window" if role == "M_behavior" else "whole-window_descriptor_not_context",
                "primary_gate": "true" if role in {"F_match", "M_behavior"} else "false",
                "rationale": rationale,
                "implementation_status": "EXISTING_DEFINITION",
            })
        for feature_id in CONTEXT_MATCH[family]:
            unit, definition = CONTEXT_DEFINITIONS[feature_id]
            rows.append({
                "residual_family": family, "feature_id": feature_id, "role": "Context_match",
                "source_kind": "pre_treatment_contract", "unit": unit,
                "pre_or_post_treatment": "pre-treatment_only", "primary_gate": "true",
                "rationale": definition, "implementation_status": "BIND_EXACT_COLUMN_BEFORE_D4_EXECUTION",
            })
        for feature_id in [x for x in M_BEHAVIOR[family] if x.startswith("mechanism.")]:
            unit, definition = MECHANISM_DEFINITIONS[feature_id]
            rows.append({
                "residual_family": family, "feature_id": feature_id, "role": "M_behavior",
                "source_kind": "family_specific_mechanism_contract", "unit": unit,
                "pre_or_post_treatment": "post-treatment_mechanism", "primary_gate": "true",
                "rationale": definition, "implementation_status": "IMPLEMENT_AND_SHA_BIND_BEFORE_D4_EXECUTION",
            })
    return rows


def make_target_v2(target_v1: dict[str, Any], role_rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = json.loads(json.dumps(target_v1))
    result["schema_version"] = "r0_target_definition_v0.2"
    result["status"] = "READY_FOR_R0_V1_PROTOCOL_FREEZE"
    result["legacy_global_descriptor_sets_v0_1"] = result.pop("descriptor_sets")
    result["legacy_global_descriptor_sets_v0_1"]["status"] = "SUPERSEDED_BY_FAMILY_SPECIFIC_ROLES"
    role_index: dict[str, dict[str, str]] = {target["target_id"]: {} for target in result["targets"]}
    for row in role_rows:
        if row["feature_id"] in role_index:
            role_index[row["feature_id"]][row["residual_family"]] = row["role"]
    for target in result["targets"]:
        target["descriptor_role"] = "FAMILY_SPECIFIC"
        target["family_specific_d4_roles"] = role_index[target["target_id"]]
        target["used_for"]["matching"] = any(role == "F_match" for role in role_index[target["target_id"]].values())
        target["used_for"]["mechanism_validation"] = any(role == "M_behavior" for role in role_index[target["target_id"]].values())
    result["d4_family_specific_contracts"] = {
        family: {
            "F_match": F_MATCH[family],
            "Context_match": CONTEXT_MATCH[family],
            "M_behavior": M_BEHAVIOR[family],
            "Semantic_probe_only": sorted(row["target_id"] for row in result["targets"] if role_index[row["target_id"]][family] == "Semantic_probe_only"),
            "primary_f_match_count": len(F_MATCH[family]),
            "primary_f_m_overlap": [],
        }
        for family in FAMILIES
    }
    result["context_match_contract_definitions"] = {key: {"unit": unit, "definition": definition, "time_role": "PRE_TREATMENT_ONLY"} for key, (unit, definition) in CONTEXT_DEFINITIONS.items()}
    result["mechanism_contract_definitions"] = {key: {"unit": unit, "definition": definition, "time_role": "POST_TREATMENT_MECHANISM", "implementation_status": "REQUIRED_BEFORE_D4_EXECUTION"} for key, (unit, definition) in MECHANISM_DEFINITIONS.items()}
    result["role_rules"] = {
        "primary_f_match_and_primary_m_behavior_overlap_within_family": "PROHIBITED",
        "post_treatment_response_as_context_match": "PROHIBITED",
        "shared_f_match_across_families": "NOT_REQUIRED",
        "thw_and_front_gap_whole_window_features": "SEMANTIC_PROBE_ONLY; pre-treatment anchor variants may be Context_match",
    }
    return result


def d1_contract(evidence: dict[str, dict[str, str]]) -> dict[str, Any]:
    targets = []
    for family, members in D1_CORE.items():
        for target_id, kind, metric in members:
            if target_id == "raw33.lane_change_count_proxy":
                stats = {"development_rows": 135046, "positive_prevalence": 0.36552730180827275, "definition": "1[count_proxy>0]"}
            else:
                row = evidence[target_id]
                stats = {"development_rows": int(row["analysis_valid_rows"]), "iqr": float(row["iqr"]), "p05": float(row["p05"]), "median": float(row["median"]), "p95": float(row["p95"])}
            targets.append({"semantic_family": family, "target_id": target_id, "target_kind": kind, "primary_metric": metric, "development_target_statistics": stats, "selection_basis": "predeclared semantic coverage and nondegenerate development target support; no representation/Stage7L outcome used"})
    return {
        "schema_version": "r0_d1_core_semantic_targets_v0.1",
        "status": "FROZEN_FOR_R0_V1_PROTOCOL",
        "core_target_count": len(targets),
        "families": {family: [target_id for target_id, _, _ in members] for family, members in D1_CORE.items()},
        "targets": targets,
        "probe_contract": {"continuous": "linear ridge; held-out log/source-group split", "categorical": "linear logistic; held-out grouped evaluation", "ridge_grid": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000], "hyperparameter_selection": "development validation only", "bootstrap_cluster": "log; scenario/source group only when log unavailable and downgrade recorded"},
        "target_level_gates": {
            "continuous": {"point_estimate": "held-out R2 >= 0.10", "uncertainty": "log-cluster 95% CI lower bound > 0", "secondary_required_report": ["MAE", "NRMSE", "Spearman", "calibration_slope"]},
            "categorical": {"point_estimate": "balanced_accuracy >= 0.60", "uncertainty": "log-cluster 95% CI lower bound > 0.50", "secondary_required_report": ["AUROC", "macro_F1"]},
        },
        "family_gate": {"minimum_targets_passing": 2, "minimum_fraction_passing": "2/3", "all_three_families_required": False, "module_support_rule": "at least 2/3 semantic families pass in at least 2 learned representation families; A/B/C require >=2/3 seed direction consistency", "old64_single_seed": "SEED_REPLICATION_NOT_AVAILABLE; descriptive corroboration only"},
        "evaluability": {"minimum_independent_log_or_source_groups": 30, "categorical_minimum_groups_per_class": 50, "insufficient_support_result": "INCONCLUSIVE", "probe_failure_alone_means_information_absent": False},
        "evidence_level_without_audit_holdout": "DEVELOPMENT_DIAGNOSTIC_EVIDENCE",
        "stage7l_outcome_used_for_target_selection": False,
    }


def fallback_contract(evidence: dict[str, dict[str, str]]) -> dict[str, Any]:
    families = {}
    for family in FAMILIES:
        items = []
        for target_id in F_MATCH[family]:
            row = evidence[target_id]
            items.append({"target_id": target_id, "caliper": float(row["option_numerical_margin"]), "unit": row["option_unit"], "basis": "0.10 x frozen Waymo TRAIN robust IQR", "formal_equivalence_margin": False})
        families[family] = {"primary_f_match_count": len(items), "calipers": items, "balance_rule": "all Primary F_match absolute pair differences within caliper; report family-wise failures without dropping pairs by representation outcome"}
    return {
        "schema_version": "D4_DEVELOPMENT_BALANCE_FALLBACK_V1",
        "status": "FROZEN_DEVELOPMENT_ONLY",
        "scientific_acceptability": "ACCEPTABLE_FOR_R0_R1_DEVELOPMENT_FEASIBILITY_ONLY",
        "authorization_effect": "D4_CONDITIONALLY_NONBLOCKING_FOR_RBR_DEVELOPMENT; NOT sufficient for R4 confirmation or RBR training authorization by itself",
        "families": families,
        "allowed_uses": ["R0/R1 benchmark development", "descriptor-balanced hard-negative construction", "feasibility diagnosis"],
        "prohibited_claims": ["NOT_FORMAL_PHYSICAL_EQUIVALENCE", "NOT_R4_CONFIRMATORY_EQUIVALENCE", "not human perceptibility", "not evidence of model superiority"],
        "activation_conditions": ["family-specific roles frozen", "pre-treatment Context_match applied", "M_behavior separately confirms mechanism", "no representation/BDD/probe outcome used for selection", "whole-roster accounting retained"],
        "r4_upgrade_gate": "family-specific physical/material equivalence margins and TOST/IUT rule must be frozen before any R4 outcome is unblinded",
    }


def r4_freeze() -> dict[str, Any]:
    return {
        "schema_version": "r0_future_r4_reserved_source_or_generator_freeze_v0.1",
        "status": "FUTURE_R4_RESERVED_SOURCE_OR_GENERATOR_FROZEN",
        "frozen_date": "2026-08-26",
        "source_baseline_commit": "b4bcc9699c534ea6341c19b9a247f80c9e279cbe",
        "freeze_form": "FROZEN_PROSPECTIVE_ACQUISITION_RULE",
        "rule_id": "R4_PROSPECTIVE_CONTROLLED_SOURCE_RULE_V1",
        "eligible_source": "first newly acquired, research-licensed, nuPlan-compatible source batch satisfying schema/map/runnability and complete identity-ledger requirements after this freeze",
        "deterministic_source_tie_break": "lexicographic(dataset_release_id, source_manifest_sha256) among simultaneously eligible batches",
        "identity_exclusions": ["all Waymo train/val/historical-test", "all Stage6/Stage7/Stage7L identities", "all R0 development/audit identities", "any source/log/token exposed to representation outcome"],
        "selection": {"seed": 2026082601, "algorithm": "SHA-256", "sort_key": "sha256(seed|source_release|log_name|scenario_token)", "allocation": "log-disjoint whole roster"},
        "generation_contract": {"design": "paired baseline/treatment controlled planner", "families": list(FAMILIES), "pre_treatment_eligibility_only": True, "whole_roster_intention_to_evaluate": True, "realized_mechanism_exclusion": "PROHIBITED", "representation_outcome_selection": "PROHIBITED"},
        "bindings_required_before_rollout": ["exact acquired source manifest and SHA", "exact token/log roster", "planner/config/code SHA", "dose grid", "failure/missingness policy", "power and family allocation"],
        "final_confirmation_roster": "NOT_FROZEN_BY_DESIGN; form outcome-blind in R1 after mechanism/matching/equivalence/runnability rules stabilize",
        "immutability": "source/acquisition/generator rule cannot change after any RBR outcome is available",
        "rbr_training_authorization": "NOT_AUTHORIZED_BY_THIS_FREEZE_ALONE",
    }


def d4_markdown(fallback: dict[str, Any]) -> str:
    lines = [
        "# R0 D4 Family-Specific Matching Contract v0.1", "",
        "## Status", "", "`READY_FOR_R0_V1_PROTOCOL_FREEZE`。本合同取代 24 项 global F_match 设计；它不执行 representation、BDD、rollout 或训练。", "",
        "## Role rules", "",
        "- `F_match` 只控制该 residual family 必须消除的核心固定人工摘要；",
        "- `Context_match` 只允许 treatment/response 发生前已测量的 context；",
        "- `M_behavior` 只确认 rollout/episode 后的 morphology/mechanism；",
        "- 同一 feature 在同一 family 内不得同时是 Primary F_match 与 Primary M_behavior；",
        "- 三个 family 不要求共享 F_match；",
        "- whole-window THW/front-gap/closing 等会受 response 影响，全部移出 Primary F_match/Context_match；只有 frozen pre-treatment anchor 版本可作 Context_match。", "",
    ]
    for family in FAMILIES:
        lines += [f"## {family}", "", f"Primary F_match ({len(F_MATCH[family])})：`" + "`, `".join(F_MATCH[family]) + "`。", "", "Context_match：`" + "`, `".join(CONTEXT_MATCH[family]) + "`。", "", "M_behavior：`" + "`, `".join(M_BEHAVIOR[family]) + "`。", "", "其余 target 均为 `Semantic_probe_only`，不参与该 family 的 Primary matching/mechanism gate。", ""]
    lines += [
        "## D4 development fallback", "",
        "`D4_DEVELOPMENT_BALANCE_FALLBACK_V1` 冻结为 development-only bounded fallback。每个 Primary F_match 使用既有 Waymo TRAIN robust-IQR 证据中的 `0.10 × IQR` caliper，并要求该 family 全部核心 feature 通过。", "",
        "它可用于 R0/R1 benchmark development、hard-negative construction 与 feasibility diagnosis；其科学状态是 `NOT_FORMAL_PHYSICAL_EQUIVALENCE`、`NOT_R4_CONFIRMATORY_EQUIVALENCE`。因此 D4 可成为 `CONDITIONALLY_NONBLOCKING_FOR_RBR_DEVELOPMENT`，但 fallback 本身不授权 RBR training。", "",
        "R4 outcome 解盲前必须把每个 family 的 physical/material margin、TOST/IUT、cluster rule 与 final roster 一并冻结。若做不到，R4 equivalence 为 `NOT_EVALUABLE`，不得用 development caliper 替代。", "",
        "机器角色表：`docs/stageR/r0/manifests/r0_d4_family_specific_feature_roles_v0.1.csv`；fallback：`docs/stageR/r0/manifests/r0_d4_development_balance_fallback_v1.json`。", "",
    ]
    return "\n".join(lines)


def d1_markdown(contract: dict[str, Any]) -> str:
    return """# R0 D1 Semantic Retention Gate v0.1

## Frozen decision contract

`FROZEN_FOR_R0_V1_PROTOCOL`。CORE targets 在任何新 representation evaluation 前按语义覆盖与既有 development target support 选定；未使用 Stage7L outcome、embedding、BDD 或 probe outcome。

| Family | CORE targets | Family pass |
|---|---|---|
| longitudinal | `ego13.mean_speed`, `ego13.end_minus_start_speed`, `ego13.rms_accel` | 至少 2/3 |
| lateral | `ego13.rms_yaw_rate`, `ego13.heading_change_abs_total`, `raw33.lane_change_count_proxy -> any_count_gt_0` | 至少 2/3 |
| interaction | `raw33.mean_front_distance`, `raw33.mean_rel_speed`, `raw33.front_pressure_score` | 至少 2/3 |

连续 target 使用 log/source-grouped held-out linear ridge：Primary `R² >= 0.10` 且 log-cluster 95% CI lower bound `>0`；同时必须报告 MAE/NRMSE、Spearman 与 calibration slope。分类 target 使用 grouped linear logistic：balanced accuracy `>=0.60` 且 95% CI lower bound `>0.50`；同时报告 AUROC 与 macro-F1。

模块级 `D1_KNOWN_SEMANTIC_INFORMATION_PRESENT=SUPPORTED` 要求至少 2/3 semantic families 在至少两个 learned representation families 中通过；A/B/C 各自要求至少 2/3 seeds 方向一致。old64 单 seed 只能作 descriptive corroboration。

少于 30 个独立 log/source groups，或分类 target 任一类少于 50 个独立 groups，结果必须为 `INCONCLUSIVE`。单一 probe failure、样本不足或 CI 过宽不得解释为 information absent。没有 R0_AUDIT_HOLDOUT 时，所有结果仍限定为 `DEVELOPMENT_DIAGNOSTIC_EVIDENCE`。

机器合同：`docs/stageR/r0/manifests/r0_d1_core_semantic_targets_v0.1.json`。
"""


def r4_markdown() -> str:
    return """# R0 Future R4 Reserved Source or Generator Proposal v0.2

## Two-stage decision

```text
FUTURE_R4_RESERVED_SOURCE_OR_GENERATOR = FROZEN
freeze_form = FROZEN_PROSPECTIVE_ACQUISITION_RULE
FUTURE_R4_CONFIRMATION_ROSTER = NOT_FROZEN_BY_DESIGN
RBR_TRAINING_NOT_AUTHORIZED_BY_THIS_DOCUMENT_ALONE
```

R0 阶段冻结的是 future source/generator 的选择边界，不要求提前形成最终 token roster。最终 roster 可在 R1 的 mechanism、family-specific matching/equivalence 与 runnability 规则稳定后，从 reserved rule 产生的 source universe 中 outcome-blind 形成。

## Frozen prospective source rule

采用 `R4_PROSPECTIVE_CONTROLLED_SOURCE_RULE_V1`：选择本次冻结之后首个新获取、research-licensed、nuPlan-compatible 且具完整 source/log/token/SHA ledger 的 source batch。若同时存在多个合格 batch，按 `(dataset_release_id, source_manifest_sha256)` 字典序唯一确定。必须与 Waymo train/val/historical-test、Stage6/7/7L、R0 development/audit 以及任何已接触 representation outcome 的 identity 全部 log/token-disjoint。

source 内 token 按 `SHA256(2026082601|source_release|log_name|scenario_token)` 排序；最终分配必须 log-disjoint。只允许 pre-treatment eligibility、context、技术 runnability、family coverage 与预注册 power；禁止用 realized mechanism 或 representation/BDD/probe outcome 排序、删除或补样本。

## Controlled generation boundary

生成设计固定为 paired baseline/treatment、whole-roster/intention-to-evaluate，families 为 `R-HLC/R-TSB/R-IP`。在任何 rollout 前仍必须绑定 exact source/token roster、planner/config/code SHA、dose grid、failure/missingness policy 与 power allocation。弱 mechanism 不得改写为技术失败。

## Final roster boundary

`RESERVED_SOURCE_OR_GENERATOR_FREEZE` 与 `FINAL_CONFIRMATION_ROSTER_FREEZE` 是两个独立事件。R1 形成 final roster 时不得改变本规则；R4 outcome 解盲前还必须冻结 family-specific physical/material margins、TOST/IUT、model/readout/kernel/threshold 和 roster SHA。

机器 freeze：`docs/stageR/r0/manifests/r0_future_r4_reserved_source_or_generator_freeze_v0.1.json`。
"""


def protocol_v06() -> str:
    source = (PROTOCOL / "R0_Representation_Measurement_Audit_Protocol_v0.5_zh.md").read_text(encoding="utf-8")
    source = source.replace("# R0 Representation & Measurement Audit Protocol v0.5（StageR 分支集成稿）", "# R0 Representation & Measurement Audit Protocol v0.6（Protocol Consistency Repair）", 1)
    source = source.replace("> 文档状态：`PARAMETERIZATION_PREP_DRAFT`", "> 文档状态：`READY_FOR_R0_V1_PROTOCOL_FREEZE`", 1)
    source = source.replace("> Remote branch HEAD（2026-08-25核验）：`460832bde6266f1367a10bfe00e9b3bc176740ce`", "> v0.6 source baseline commit（2026-08-26）：`b4bcc9699c534ea6341c19b9a247f80c9e279cbe`", 1)
    source = source.replace("R0_PROTOCOL_V0_5_STAGER_BRANCH_INTEGRATED_DRAFT", "R0_PROTOCOL_V0_6_CONSISTENCY_REPAIRED\n> R0_V1_PROTOCOL_FREEZE_READY")
    source = source.replace("本协议 v0.5 在 v0.4 的基础上完成 StageR active branch 与本地 Work/Codex 数据流集成；v0.4 已补齐冻结前最后两项操作边界：", "本协议 v0.6 在 v0.5 的基础上完成 protocol consistency repair，并保留既有 StageR 数据流与操作边界：", 1)
    source = source.replace("## 0. v0.5 修订摘要", "## 0. v0.6 修订摘要", 1)
    source = source.replace("v0.5 继承 v0.4，并完成 StageR active branch 与本地 Work/Codex 数据流集成；v0.4 继承 v0.3 已完成的方法与操作修订，并补充冻结前最后两项边界。当前完整冻结前修订要点如下：", "v0.6 继承 v0.5，并完成 audit-holdout、R4 两阶段冻结、D1 gate、D4 family-specific matching/fallback 与四维 readiness 的一致性修复。以下保留既有方法沿革；本版新增规则以 §§24–26 为准。", 1)
    source = source.replace("3. 在查看新的 R0 metric 结果前冻结 `R0_DEVELOPMENT / R0_AUDIT_HOLDOUT`；", "3. 在查看新的 R0 metric 结果前冻结 `R0_DEVELOPMENT`，并记录 `R0_AUDIT_HOLDOUT` 是否存在；若不存在则绑定 development-only evidence level；", 1)
    source = source.replace("4. 同期锁定 `FUTURE_R4_RESERVED_POOL` 的数据源/token pool/生成规则；", "4. 同期锁定 `FUTURE_R4_RESERVED_SOURCE_OR_GENERATOR` 的 source universe、token/log pool、prospective acquisition rule 或 controlled generation rule；", 1)
    source = source.replace("3. `R0_AUDIT_HOLDOUT` 必须在 threshold/margin/readout 关键选择冻结前锁定。", "3. 若存在 `R0_AUDIT_HOLDOUT`，必须在 threshold/margin/readout 关键选择冻结前锁定；若不存在，记录 availability 并强制 development-only evidence level。", 1)
    source = source.replace("- 预先锁定的 `R0_AUDIT_HOLDOUT` 中存在可按同一 outcome-blind 规则评估的对应资产；", "- 若存在 `R0_AUDIT_HOLDOUT`，按同一 outcome-blind 规则评估；若不存在则 evidence level 限定为 development diagnostic，不阻塞执行；", 1)
    source = source.replace("- 已锁定或可在正式 RBR training 前锁定 `FUTURE_R4_RESERVED_POOL`；", "- 已锁定 `FUTURE_R4_RESERVED_SOURCE_OR_GENERATOR`；最终 confirmation roster 可在 R1 outcome-blind 形成；", 1)
    source = source.replace("- R0 data-tier split 与 `FUTURE_R4_RESERVED_POOL` 已冻结；", "- R0_DEVELOPMENT role 与 audit-holdout availability 已记录，且 `FUTURE_R4_RESERVED_SOURCE_OR_GENERATOR` 已冻结；", 1)
    source = source.replace("- F_match list；\n- per-feature equivalence margin；\n- margin rationale；", "- family-specific F_match / Context_match / M_behavior / Semantic_probe_only lists；\n- development-only robust-IQR balance calipers与适用边界；\n- R4 pre-unblind physical/material margin upgrade gate；", 1)
    source = source.replace("- D1 semantic target minimum interpretable thresholds；", "- D1 semantic target minimum interpretable thresholds（已由 v0.6 三族 gate 冻结）；", 1)
    source = source.replace("- D4 各 `F_match` equivalence margins；", "- D4 development calipers（已冻结）与 R4 physical/material margins（延后至 R4 outcome 解盲前冻结，不阻塞 R0 protocol freeze）；", 1)
    source = source.replace("3. 冻结 R0_DEVELOPMENT / R0_AUDIT_HOLDOUT；", "3. 冻结 R0_DEVELOPMENT，并记录 R0_AUDIT_HOLDOUT availability/evidence-level fallback；", 1)
    source = source.replace("4. 锁定 FUTURE_R4_RESERVED_POOL 的数据源/token pool/生成规则，并明确 R0_AUDIT_HOLDOUT 是否存在；", "4. 锁定 FUTURE_R4_RESERVED_SOURCE_OR_GENERATOR，并明确 final confirmation roster 在 R1 outcome-blind 形成；", 1)
    source = source.replace("10. 冻结 D4 equivalence margin rationale + TOST/CI + cluster inference；", "10. 冻结 D4 family-specific roles、development fallback及 R4 pre-unblind physical-margin/TOST/IUT upgrade gate；", 1)
    source = source.replace("v0.4 完成后，下一步仍不是训练模型，而是形成真正可执行的 **R0 Operational Freeze v1.0**。", "v0.6 consistency repair 完成后，下一步仍不是训练模型，而是形成带完整 SHA binding 的 **R0 Operational Freeze v1.0**。", 1)
    source = source.replace("按照 R0 阶段已锁定的 `FUTURE_R4_RESERVED_POOL`，在 R1 mechanism/matching/runnability 规则稳定后，prospectively 形成 `FUTURE_R4_CONFIRMATION_ROSTER`", "按照 R0 阶段已锁定的 `FUTURE_R4_RESERVED_SOURCE_OR_GENERATOR`，在 R1 mechanism/matching/runnability 规则稳定后，prospectively 形成 `FUTURE_R4_CONFIRMATION_ROSTER`", 1)
    source = source[:source.index("# 24.")]
    source += """# 24. v0.6 Protocol Freeze Readiness

```text
PROTOCOL_DEFINITION_BLOCKERS = 0
READY_FOR_R0_V1_PROTOCOL_FREEZE
R0_EXECUTION_READY_WITH_DEVELOPMENT_DIAGNOSTIC_EVIDENCE
RBR_TRAINING_NOT_AUTHORIZED
R4_FINAL_CONFIRMATION_NOT_READY
```

本状态只说明可执行定义与 fallback 已闭合。它不代表 R0 已执行、科学假设已支持、candidate training 已授权或 R4 final roster 已冻结。

# 25. v0.6 Change Record

v0.6 将 audit-holdout capacity 与 protocol definition 分离，恢复 R4 source/generator 与 final roster 的两阶段冻结，建立 family-specific D4 matching/fallback 和三族 D1 gate，并用四个独立 readiness domain 取代单一 `NOT_READY`。
"""
    addendum = """

---

# 26. v0.6 Normative Consistency Repair（优先于冲突的旧条款）

本节是 v0.6 的规范性修订。若 §§4.3、12.13、15.3、18.6、19、23、24 的旧文字与本节冲突，以本节为准；旧文字仅保留版本追溯。

## 26.1 四种 readiness 不再合并

每个 readiness item 必须同时记录 `blocks_protocol_freeze / blocks_r0_execution / blocks_rbr_training / limitation_type / limits_evidence_level`。`limitation_type` 至少使用：`PROTOCOL_DEFINITION_BLOCKER`、`EXECUTION_CAPACITY_LIMITATION`、`TRAINING_AUTHORIZATION_BLOCKER`、`EVIDENCE_LEVEL_LIMITATION`、`NONBLOCKING_LIMITATION`。

## 26.2 R0_AUDIT_HOLDOUT

`R0_AUDIT_HOLDOUT=NOT_AVAILABLE` 时：`blocks_protocol_freeze=false`、`blocks_r0_execution=false`、`limits_evidence_level=true`。R0 仍可按 frozen SAP 执行，但只能报告 `DEVELOPMENT_DIAGNOSTIC_EVIDENCE`。当前 19 runnable clean logs 对 150-log confirmatory reference 的 131-log缺口是 `EXECUTION_CAPACITY_LIMITATION`，不要求为 protocol v1.0 freeze 补齐。

## 26.3 FUTURE_R4 两阶段冻结

R0 只要求 `FUTURE_R4_RESERVED_SOURCE_OR_GENERATOR` 在正式 RBR training 前冻结；合法形式包括 source universe、token/log pool、prospective acquisition rule 或 controlled generation rule。当前冻结 `R4_PROSPECTIVE_CONTROLLED_SOURCE_RULE_V1`。`FUTURE_R4_CONFIRMATION_ROSTER` 可在 R1 outcome-blind 形成，不是 R0 protocol-freeze blocker。

## 26.4 D1 gate

D1 使用 `r0_d1_core_semantic_targets_v0.1.json` 的 9 个 CORE targets 与三族 gate。连续 target：held-out grouped `R²>=0.10` 且 log-cluster 95% CI lower>0；分类 target：balanced accuracy>=0.60 且 lower>0.50。每族至少 2/3 CORE targets；样本/cluster不足为 `INCONCLUSIVE`，不得推断 information absent。

## 26.5 D4 family-specific contract与fallback

global 24-feature F_match 被废止。Primary F_match 数量为 `R-HLC=4 / R-TSB=4 / R-IP=3`；Context_match 仅限 pre-treatment，M_behavior 与 F_match 在同 family 内零交集。`D4_DEVELOPMENT_BALANCE_FALLBACK_V1` 使用 frozen `0.10×development IQR` caliper，仅允许 development balance/feasibility：`NOT_FORMAL_PHYSICAL_EQUIVALENCE`、`NOT_R4_CONFIRMATORY_EQUIVALENCE`。它使 D4 在满足 activation conditions 时 `CONDITIONALLY_NONBLOCKING_FOR_RBR_DEVELOPMENT`，但不单独授权训练。family-specific physical/material margins 必须在任何 R4 outcome 解盲前冻结。

## 26.6 v1 protocol freeze conclusion

```text
PROTOCOL_DEFINITION_BLOCKERS = 0
PROTOCOL_FREEZE_READINESS = READY_FOR_R0_V1_PROTOCOL_FREEZE
R0_EXECUTION_READINESS = READY_WITH_DEVELOPMENT_DIAGNOSTIC_EVIDENCE
RBR_TRAINING_READINESS = NOT_AUTHORIZED
R4_CONFIRMATION_READINESS = SOURCE_OR_GENERATOR_FROZEN_FINAL_ROSTER_NOT_FROZEN
```

协议可冻结、R0 audit holdout 不可用、RBR training 未授权三者不矛盾。协议冻结只说明分析定义、fallback 与状态语义已闭合；不代表分析已经执行、科学结果已支持、训练已授权或 R4 roster 已就绪。

## 26.7 v0.6 change record

1. 将 audit holdout 缺失从 protocol blocker 降为 evidence-level/capacity limitation；
2. 引入三类 blocking booleans 与 limitation taxonomy；
3. 冻结 prospective R4 source/acquisition rule，延后 final roster；
4. 用 family-specific D4 roles 取代 24-feature global F_match；
5. 冻结 development-only D4 robust-IQR fallback 与 R4 upgrade gate；
6. 冻结 9-target、三族 D1 semantic-retention gate；
7. 分开报告 protocol/R0 execution/RBR training/R4 confirmation readiness。
"""
    return source.rstrip() + addendum.rstrip() + "\n"


def sap_v03(d1: dict[str, Any], fallback: dict[str, Any], r4: dict[str, Any], target_v2: dict[str, Any]) -> tuple[dict[str, Any], str]:
    sap = read_json(MANIFEST / "r0_statistical_analysis_plan_v0.2.json")
    sap["schema_version"] = "r0_statistical_analysis_plan_v0.3"
    sap["status"] = "READY_FOR_R0_V1_PROTOCOL_FREEZE"
    sap["data_roles"]["r0_audit_holdout"] = "NOT_AVAILABLE_NONBLOCKING; evidence limited to DEVELOPMENT_DIAGNOSTIC_EVIDENCE"
    sap["data_roles"]["future_r4_reserved_pool"] = "FUTURE_R4_RESERVED_SOURCE_OR_GENERATOR_FROZEN_PROSPECTIVE_ACQUISITION_RULE_V1"
    sap["readiness_semantics"] = {"r0_audit_holdout": {"blocks_protocol_freeze": False, "blocks_r0_execution": False, "blocks_rbr_training": False, "limits_evidence_level": True, "limitation_type": "EVIDENCE_LEVEL_LIMITATION"}, "confirmatory_capacity_150_logs": {"blocks_protocol_freeze": False, "blocks_r0_execution": False, "blocks_rbr_training": False, "limits_evidence_level": True, "limitation_type": "EXECUTION_CAPACITY_LIMITATION", "available_runnable_logs": 19, "reference_logs": 150, "gap": 131}}
    sap["d1"] = d1
    sap["d4"]["legacy_global_24_f_match_status"] = "SUPERSEDED"
    sap["d4"]["legacy_global_f_match_target_ids_v0_1"] = sap["d4"].pop("f_match_target_ids")
    sap["d4"]["legacy_global_m_behavior_target_ids_v0_1"] = sap["d4"].pop("m_behavior_target_ids")
    sap["d4"]["family_specific_contracts"] = target_v2["d4_family_specific_contracts"]
    sap["d4"]["development_balance_fallback"] = fallback
    sap["d4"]["equivalence_margin_status"] = "DEVELOPMENT_CALIPERS_FROZEN; R4_PHYSICAL_MARGINS_DEFERRED_TO_PRE_UNBLIND_R4_UPGRADE_GATE"
    sap["future_r4_reserved_source_or_generator"] = r4
    sap["readiness"] = {"protocol_freeze": "READY_FOR_R0_V1_PROTOCOL_FREEZE", "r0_execution": "READY_WITH_DEVELOPMENT_DIAGNOSTIC_EVIDENCE", "rbr_training": "NOT_AUTHORIZED", "r4_confirmation": "SOURCE_OR_GENERATOR_FROZEN_FINAL_ROSTER_NOT_FROZEN"}
    for hypothesis in sap["hypotheses"]:
        hypothesis["allowed_evidence_level"] = "DEVELOPMENT_DIAGNOSTIC_EVIDENCE; R0_AUDIT_HOLDOUT not required for execution"
        hypothesis["equivalence_method"] = "D4 family-specific: development fallback uses frozen robust-IQR balance calipers only; formal R4 uses pre-unblind physical/material margins with TOST/IUT"
    md = """# R0 Statistical Analysis Plan v0.3

## Status

`READY_FOR_R0_V1_PROTOCOL_FREEZE`。这不等于分析已执行或 RBR training 已授权。

## Readiness semantics

- 无 R0_AUDIT_HOLDOUT：不阻塞 protocol freeze，不阻塞 R0 execution；全部科学结果限定为 `DEVELOPMENT_DIAGNOSTIC_EVIDENCE`。
- 19 runnable clean logs vs 150-log reference：`EXECUTION_CAPACITY_LIMITATION`；131-log gap 只保留为未来 confirmatory planning reference。
- `FUTURE_R4_RESERVED_SOURCE_OR_GENERATOR` 已通过 `R4_PROSPECTIVE_CONTROLLED_SOURCE_RULE_V1` 冻结；final confirmation roster 在 R1 outcome-blind 形成。

## D1

CORE targets 共 9 项，longitudinal/lateral/interaction 各 3。连续 gate 为 held-out grouped `R²>=0.10` 且 cluster 95% CI lower>0；分类 gate 为 balanced accuracy>=0.60 且 lower>0.50；每族至少 2/3 targets。独立 groups 不足时为 `INCONCLUSIVE`。

## D4

Primary F_match 改为 family-specific：R-HLC=4、R-TSB=4、R-IP=3。Context_match 只使用 pre-treatment anchor；M_behavior 与同 family F_match 零交集。`D4_DEVELOPMENT_BALANCE_FALLBACK_V1` 采用 0.10×development IQR，只用于 development balance/feasibility，不是 formal equivalence。R4 physical/material margins 必须在 R4 outcome 解盲前冻结。

## Authorization

`R0_EXECUTION_READY_WITH_DEVELOPMENT_DIAGNOSTIC_EVIDENCE`；`RBR_TRAINING_NOT_AUTHORIZED`；`R4_FINAL_CONFIRMATION_ROSTER_NOT_FROZEN`。
"""
    return sap, md


def readiness_rows() -> list[dict[str, str]]:
    def row(item: str, status: str, p: bool, e: bool, t: bool, kind: str, limit: bool, evidence: str, action: str) -> dict[str, str]:
        return {"item": item, "status": status, "blocks_protocol_freeze": str(p).lower(), "blocks_r0_execution": str(e).lower(), "blocks_rbr_training": str(t).lower(), "limitation_type": kind, "limits_evidence_level": str(limit).lower(), "evidence": evidence, "next_action": action}
    return [
        row("owner_numerical_parameters", "18_OF_18_APPROVED", False, False, False, "NONBLOCKING_LIMITATION", False, "r0_scientific_owner_approval_v0.1.json", "none"),
        row("r0_audit_holdout", "NOT_AVAILABLE", False, False, False, "EVIDENCE_LEVEL_LIMITATION", True, "protocol v0.6 §26.2", "execute R0 as development diagnostic"),
        row("confirmatory_capacity", "19_RUNNABLE_LOGS_VS_150_REFERENCE_GAP_131", False, False, False, "EXECUTION_CAPACITY_LIMITATION", True, "r0_audit_sample_size_proposal_v0.1.csv", "retain for future confirmatory acquisition planning"),
        row("d1_semantic_gate_definition", "FROZEN_9_CORE_TARGETS_3_FAMILIES", False, False, False, "NONBLOCKING_LIMITATION", False, "r0_d1_core_semantic_targets_v0.1.json", "execute frozen D1"),
        row("d1_semantic_gate_execution", "NOT_EXECUTED", False, False, True, "TRAINING_AUTHORIZATION_BLOCKER", False, "protocol-only phase", "execute frozen D1 before candidate authorization"),
        row("d4_family_specific_contract", "FROZEN_FMATCH_4_4_3", False, False, False, "NONBLOCKING_LIMITATION", False, "r0_d4_family_specific_feature_roles_v0.1.csv", "implement/SHA-bind family mechanisms before D4 execution"),
        row("d4_development_fallback", "FROZEN_CONDITIONALLY_NONBLOCKING_FOR_RBR_DEVELOPMENT", False, False, False, "NONBLOCKING_LIMITATION", False, "r0_d4_development_balance_fallback_v1.json", "evaluate activation conditions"),
        row("d4_activation_evidence", "NOT_EXECUTED", False, False, True, "TRAINING_AUTHORIZATION_BLOCKER", False, "protocol-only phase", "demonstrate D4 activation conditions before training authorization"),
        row("d4_r4_physical_margins", "DEFERRED_TO_PRE_UNBLIND_R4_UPGRADE_GATE", False, False, False, "NONBLOCKING_LIMITATION", False, "protocol v0.6 §26.5", "freeze before any R4 outcome"),
        row("future_r4_reserved_source_or_generator", "FROZEN_PROSPECTIVE_ACQUISITION_RULE_V1", False, False, False, "NONBLOCKING_LIMITATION", False, "r0_future_r4_reserved_source_or_generator_freeze_v0.1.json", "bind acquired source/SHA before rollout"),
        row("future_r4_confirmation_roster", "NOT_FROZEN_BY_DESIGN", False, False, False, "NONBLOCKING_LIMITATION", False, "R0_Future_R4_Reserved_Pool_Proposal_v0.2.md", "form outcome-blind in R1"),
        row("sap", "READY_FOR_R0_V1_PROTOCOL_FREEZE", False, False, False, "NONBLOCKING_LIMITATION", False, "r0_statistical_analysis_plan_v0.3.json", "freeze SHA binding"),
        row("protocol_definition", "READY_FOR_R0_V1_PROTOCOL_FREEZE", False, False, False, "NONBLOCKING_LIMITATION", False, "R0_Representation_Measurement_Audit_Protocol_v0.6_zh.md", "create final frozen manifest/SHA binding"),
        row("rbr_training", "NOT_AUTHORIZED", False, False, True, "TRAINING_AUTHORIZATION_BLOCKER", False, "candidate authorization manifest remains absent/not authorized", "complete R0 decisions and candidate-specific authorization"),
        row("r4_confirmation", "NOT_READY_FINAL_ROSTER_AND_PHYSICAL_MARGINS_PENDING", False, False, False, "NONBLOCKING_LIMITATION", False, "two-stage R4 contract", "freeze roster/margins before unblinding"),
    ]


def readiness_markdown() -> str:
    return """# R0 v1 Freeze Readiness Report v0.4

## Four independent decisions

| Readiness domain | Decision |
|---|---|
| A. Protocol freeze | `READY_FOR_R0_V1_PROTOCOL_FREEZE` |
| B. R0 execution | `READY_WITH_DEVELOPMENT_DIAGNOSTIC_EVIDENCE` |
| C. RBR training authorization | `NOT_AUTHORIZED` |
| D. R4 confirmation | `SOURCE_OR_GENERATOR_FROZEN; FINAL_ROSTER_NOT_FROZEN; NOT_READY_FOR_CONFIRMATION` |

## Protocol blockers

`PROTOCOL_DEFINITION_BLOCKERS=0`。D1 的 9-target/三族 gate、D4 family-specific 角色、development-only fallback、R4 prospective source rule 和四维 readiness 语义均已定义。下一步可形成 v1.0 frozen manifest/SHA binding。

## Capacity and evidence

R0_AUDIT_HOLDOUT 仍为 `NOT_AVAILABLE`，但依据主协议 §4.2 不阻塞协议冻结或 R0 执行；所有结果必须标记 `DEVELOPMENT_DIAGNOSTIC_EVIDENCE`。19 runnable clean logs 与 150-log D0 confirmatory reference 的 131-log差距已降级为 `EXECUTION_CAPACITY_LIMITATION`，不要求为 v1 protocol freeze 获取新数据。

## D1 and D4

D1 gate 已可冻结：longitudinal/lateral/interaction 各 3 个 CORE targets，每族至少 2/3；连续和分类 gate 均同时要求 effect magnitude 与 grouped 95% CI，样本不足为 `INCONCLUSIVE`。

D4 Primary F_match 数量为 R-HLC=4、R-TSB=4、R-IP=3。`D4_DEVELOPMENT_BALANCE_FALLBACK_V1` 科学上只接受为 development balance/feasibility，因此 D4 可 `CONDITIONALLY_NONBLOCKING_FOR_RBR_DEVELOPMENT`；它不构成 formal physical/R4 equivalence，也不单独授权 RBR training。

## Remaining authorization work

RBR training 仍需完成 frozen R0 execution/decision records、D1/D4 activation evidence、candidate-specific authorization manifest 与 SHA bindings。R4 仍需在 outcome 解盲前冻结 exact source/roster、planner/config、family-specific physical/material margins、TOST/IUT 与完整 analysis stack。
"""


def main() -> None:
    target_v1 = read_json(MANIFEST / "r0_target_definition_v0.1.json")
    evidence = evidence_by_target()
    role_rows = make_role_rows(target_v1)
    target_v2 = make_target_v2(target_v1, role_rows)
    d1 = d1_contract(evidence)
    fallback = fallback_contract(evidence)
    r4 = r4_freeze()

    write_csv(MANIFEST / "r0_d4_family_specific_feature_roles_v0.1.csv", role_rows, list(role_rows[0]))
    write_json(MANIFEST / "r0_target_definition_v0.2.json", target_v2)
    write_json(MANIFEST / "r0_d1_core_semantic_targets_v0.1.json", d1)
    write_json(MANIFEST / "r0_d4_development_balance_fallback_v1.json", fallback)
    write_json(MANIFEST / "r0_future_r4_reserved_source_or_generator_freeze_v0.1.json", r4)
    (GOVERNANCE / "R0_D4_Family_Specific_Matching_Contract_v0.1.md").write_text(d4_markdown(fallback), encoding="utf-8")
    (GOVERNANCE / "R0_D1_Semantic_Retention_Gate_v0.1.md").write_text(d1_markdown(d1), encoding="utf-8")
    (GOVERNANCE / "R0_Future_R4_Reserved_Pool_Proposal_v0.2.md").write_text(r4_markdown(), encoding="utf-8")
    (PROTOCOL / "R0_Representation_Measurement_Audit_Protocol_v0.6_zh.md").write_text(protocol_v06(), encoding="utf-8")

    sap_json, sap_md = sap_v03(d1, fallback, r4, target_v2)
    write_json(MANIFEST / "r0_statistical_analysis_plan_v0.3.json", sap_json)
    (PROTOCOL / "R0_Statistical_Analysis_Plan_v0.3.md").write_text(sap_md, encoding="utf-8")
    readiness = readiness_rows()
    write_csv(MANIFEST / "r0_v1_numerical_freeze_readiness_v0.4.csv", readiness, list(readiness[0]))
    (GOVERNANCE / "R0_V1_Freeze_Readiness_Report_v0.4.md").write_text(readiness_markdown(), encoding="utf-8")

    outputs = [
        GOVERNANCE / "R0_D4_Family_Specific_Matching_Contract_v0.1.md",
        GOVERNANCE / "R0_D1_Semantic_Retention_Gate_v0.1.md",
        GOVERNANCE / "R0_Future_R4_Reserved_Pool_Proposal_v0.2.md",
        GOVERNANCE / "R0_V1_Freeze_Readiness_Report_v0.4.md",
        PROTOCOL / "R0_Representation_Measurement_Audit_Protocol_v0.6_zh.md",
        PROTOCOL / "R0_Statistical_Analysis_Plan_v0.3.md",
        MANIFEST / "r0_target_definition_v0.2.json",
        MANIFEST / "r0_d4_family_specific_feature_roles_v0.1.csv",
        MANIFEST / "r0_d1_core_semantic_targets_v0.1.json",
        MANIFEST / "r0_d4_development_balance_fallback_v1.json",
        MANIFEST / "r0_future_r4_reserved_source_or_generator_freeze_v0.1.json",
        MANIFEST / "r0_statistical_analysis_plan_v0.3.json",
        MANIFEST / "r0_v1_numerical_freeze_readiness_v0.4.csv",
        ROOT / "tools/stageR_repair_r0_v1_protocol_consistency.py",
    ]
    sha_rows = [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size} for path in outputs]
    write_csv(MANIFEST / "r0_v1_protocol_consistency_repair_sha256_v0.1.csv", sha_rows, list(sha_rows[0]))
    print(json.dumps({"status": "READY_FOR_R0_V1_PROTOCOL_FREEZE", "d1_core_targets": d1["core_target_count"], "d4_f_match_counts": {f: len(F_MATCH[f]) for f in FAMILIES}, "role_rows": len(role_rows), "r4_source_or_generator": r4["status"], "r0_audit_holdout": "NOT_AVAILABLE_NONBLOCKING", "rbr_training": "NOT_AUTHORIZED"}, indent=2))


if __name__ == "__main__":
    main()
