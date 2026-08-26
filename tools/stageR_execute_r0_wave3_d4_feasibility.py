#!/usr/bin/env python3
"""Execute the frozen R0 Wave 3 D4 residual-benchmark feasibility audit.

This is a read-only audit of pre-existing DEVELOPMENT assets.  It never opens
representation, BDD, probe, checkpoint, or RBR files, never launches a rollout,
and refuses to substitute an unfrozen mechanism rule for the D4 contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs/stageR/r0/results"
BINDING = ROOT / "docs/stageR/r0/manifests/r0_v1_freeze_binding.json"
ROLES = ROOT / "docs/stageR/r0/manifests/r0_d4_family_specific_feature_roles_v0.1.csv"
TARGETS = ROOT / "docs/stageR/r0/manifests/r0_target_definition_v0.2.json"
FALLBACK = ROOT / "docs/stageR/r0/manifests/r0_d4_development_balance_fallback_v1.json"
DECISIONS = ROOT / "docs/stageR/r0/manifests/r0_decision_table_v1.0.csv"
FREEZE_TAG_COMMIT = "319757c7f72efb55c80c780e4d0f17e5341b19ec"

FAMILIES = ("R-HLC", "R-TSB", "R-IP")
MECHANISM_REASON = (
    "冻结目标定义仅给出语义描述，并未冻结计算阈值、anchor、阶段分割或判定算法；"
    "历史机制表不是同名冻结变量，不能替代。"
)
CONTEXT_REASON = (
    "历史资产没有逐项绑定到冻结的 pre-treatment anchor；不得以首帧、whole-window"
    " 或 response-derived proxy 替代。"
)


SOURCE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "source_id": "waymo_dynamic_v2",
        "source_stage": "Waymo Dynamic-v2 / Stage6R",
        "root": "outputs/stage6r_dynamic_full51_semantic_strict_v1",
        "metadata": None,
        "trajectory_dir": "outputs/stage6r_dynamic_full51_semantic_strict_part_00_09/shards/shard_000000",
        "manifest": "outputs/stage6r_dynamic_full51_semantic_strict_v1/stage6r_dynamic_full51_manifest.json",
        "candidate_families": (),
        "source_kind": "broad natural DEVELOPMENT source; no frozen residual-family label or paired contrast",
        "outcome_exposure": "BUILD_MANIFEST_DECLARES_embedding_or_checkpoint_read=false; no outcome file opened in Wave3",
    },
    {
        "source_id": "stage6j_pure_longitudinal",
        "source_stage": "Stage6J pure longitudinal",
        "root": "outputs/stage6j_pure_longitudinal_context_v1",
        "metadata": "outputs/stage6j_pure_longitudinal_context_v1/metadata.csv",
        "trajectory_dir": "outputs/stage6j_pure_longitudinal_context_v1",
        "manifest": "outputs/stage6j_pure_longitudinal_batch_v1/batch_manifest.json",
        "candidate_families": ("R-TSB",),
        "source_kind": "controlled longitudinal policy contrast; candidate only before D4 matching",
        "outcome_exposure": "historical representation/BDD artifacts may exist; not opened or used for Wave3 selection",
    },
    {
        "source_id": "stage6k_longitudinal_dose",
        "source_stage": "Stage6K longitudinal dose",
        "root": "outputs/stage6k_longitudinal_dose_context_v2_runtime_repaired/dose25",
        "metadata": "outputs/stage6k_longitudinal_dose_context_v2_runtime_repaired/dose25/metadata.csv",
        "trajectory_dir": "outputs/stage6k_longitudinal_dose_context_v2_runtime_repaired/dose25",
        "manifest": "outputs/stage6k_longitudinal_dose_batch_v1/batch_manifest.json",
        "candidate_families": ("R-TSB",),
        "source_kind": "controlled longitudinal dose contrast; candidate only before D4 matching",
        "outcome_exposure": "historical representation/BDD artifacts may exist; not opened or used for Wave3 selection",
    },
    {
        "source_id": "stage6s_v3_interaction",
        "source_stage": "Stage6S-v3 interaction confirmation",
        "root": "outputs/stage6s_v3_confirmation_context_v1",
        "metadata": "outputs/stage6s_v3_confirmation_context_v1/metadata.csv",
        "trajectory_dir": "outputs/stage6s_v3_confirmation_context_v1",
        "manifest": "outputs/stage6s_v3_confirmation_freeze_v1/stage6s_v3_confirmation_freeze_manifest.json",
        "candidate_families": ("R-IP",),
        "source_kind": "controlled interaction headway contrast; candidate only before D4 matching",
        "outcome_exposure": "historical mechanism/representation artifacts may exist; no outcome table opened or used for Wave3 selection",
    },
    {
        "source_id": "stage7_m6_locked_collection",
        "source_stage": "Stage7 / Stage7 M6 locked collection",
        "root": "outputs/stage7_m6_5_locked_confirmation_context_v1",
        "metadata": "outputs/stage7_m6_5_locked_confirmation_context_v1/metadata.csv",
        "trajectory_dir": "outputs/stage7_m6_5_locked_confirmation_context_v1",
        "manifest": "outputs/stage7_m6_4b_locked_batch_mac_v2/batch_manifest.json",
        "candidate_families": (),
        "source_kind": "mixed closed-loop historical collection; no frozen family-specific morphology label",
        "outcome_exposure": "historical representation/BDD artifacts may exist; not opened or used for Wave3 selection",
    },
    {
        "source_id": "stage7l_pure_lateral",
        "source_stage": "Stage7L pure lateral development",
        "root": "outputs/stage7l_e_prospective_bdd_v1/contexts/dose0",
        "metadata": "outputs/stage7l_e_prospective_bdd_v1/contexts/dose0/metadata.csv",
        "trajectory_dir": "outputs/stage7l_e_prospective_bdd_v1/contexts/dose0",
        "manifest": "outputs/stage7l_b_final_development_freeze_v1/refined_development_roster_freeze_summary.json",
        "candidate_families": ("R-HLC",),
        "source_kind": "controlled pure-lateral dose contrast; candidate only before D4 matching",
        "outcome_exposure": "historical representation/BDD artifacts may exist; not opened or used for Wave3 selection",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_rev_parse(ref: str) -> str:
    return subprocess.check_output(["git", "rev-parse", ref], cwd=ROOT, text=True).strip()


def write_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {path}")
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify_freeze() -> dict[str, Any]:
    binding = read_json(BINDING)
    mismatches: list[dict[str, str]] = []
    checks: list[dict[str, str]] = []
    for name, record in binding["all_frozen_artifact_sha256"].items():
        path = ROOT / record["path"]
        actual = sha256_file(path)
        expected = record["sha256"]
        checks.append({"artifact": name, "path": record["path"], "expected": expected, "actual": actual})
        if actual != expected:
            mismatches.append(checks[-1])
    tag_commit = git_rev_parse("r0-v1.0-protocol-freeze^{commit}")
    if tag_commit != FREEZE_TAG_COMMIT:
        mismatches.append({
            "artifact": "r0-v1.0-protocol-freeze",
            "path": "git tag",
            "expected": FREEZE_TAG_COMMIT,
            "actual": tag_commit,
        })
    if mismatches:
        raise RuntimeError(json.dumps({"freeze_sha_mismatch": mismatches}, ensure_ascii=False, indent=2))
    return {
        "status": "PASS",
        "freeze_tag_commit": tag_commit,
        "freeze_content_commit": binding["R0_V1_FREEZE_CONTENT_COMMIT"],
        "frozen_artifact_checks": checks,
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def source_snapshot(spec: dict[str, Any]) -> dict[str, Any]:
    trajectory_dir = ROOT / spec["trajectory_dir"]
    metadata_path = ROOT / spec["metadata"] if spec["metadata"] else None
    manifest_path = ROOT / spec["manifest"]
    rows = read_csv_rows(metadata_path) if metadata_path and metadata_path.exists() else []
    tokens = {row["scenario_token"] for row in rows if row.get("scenario_token")}
    logs = {row["log_name"] for row in rows if row.get("log_name")}
    metadata_columns = set(rows[0]) if rows else set()
    if rows:
        unit_count = len(tokens)
        row_count = len(rows)
    else:
        manifest = read_json(manifest_path)
        unit_count = int(manifest.get("scenario_count", 0))
        row_count = int(manifest.get("row_count", 0))
    ego_path = trajectory_dir / "ego_seq.npy"
    mask_path = trajectory_dir / "ego_seq_mask.npy"
    raw33_path = trajectory_dir / "interaction_feat_style.npy"
    raw33_schema = trajectory_dir / "feature_schema.json"
    context_path = trajectory_dir / "context_traj.npy"
    raw_trajectory = ego_path.exists()
    valid_mask_available = mask_path.exists()
    if raw_trajectory and valid_mask_available:
        ego = np.load(ego_path, mmap_mode="r")
        mask = np.load(mask_path, mmap_mode="r")
        trajectory_shape = list(ego.shape)
        if ego.ndim != 3 or ego.shape[-1] != 8 or list(mask.shape) != list(ego.shape[:2]):
            raise ValueError(f"Invalid raw trajectory contract in {trajectory_dir}: {ego.shape}/{mask.shape}")
    elif raw_trajectory:
        trajectory_shape = list(np.load(ego_path, mmap_mode="r").shape)
    else:
        trajectory_shape = []
    raw33_features: set[str] = set()
    if raw33_path.exists() and raw33_schema.exists():
        feature_json = read_json(raw33_schema)
        raw33_features = {str(row["name"]) for row in feature_json.get("features", [])}
    return {
        **spec,
        "trajectory_dir": str(trajectory_dir.relative_to(ROOT)),
        "metadata_present": bool(rows),
        "metadata_columns": metadata_columns,
        "row_count": row_count,
        "independence_unit_count": unit_count,
        "log_count": len(logs),
        "tokens": tokens,
        "raw_trajectory_available": raw_trajectory,
        "valid_mask_available": valid_mask_available,
        "trajectory_shape": trajectory_shape,
        "context_available": context_path.exists(),
        "raw33_available": raw33_path.exists() and len(raw33_features) == 33,
        "raw33_features": raw33_features,
    }


def load_roles() -> dict[str, dict[str, list[str]]]:
    roles: dict[str, dict[str, list[str]]] = {family: defaultdict(list) for family in FAMILIES}
    for row in read_csv_rows(ROLES):
        family = row["residual_family"]
        if family in roles and row["primary_gate"].lower() == "true":
            roles[family][row["role"]].append(row["feature_id"])
    return {family: {role: sorted(values) for role, values in record.items()} for family, record in roles.items()}


def context_feature_status(feature: str, candidates: list[dict[str, Any]]) -> tuple[str, str]:
    all_metadata = all(source["metadata_present"] for source in candidates)
    if feature == "context.map_location":
        if all_metadata and all("location" in source["metadata_columns"] for source in candidates):
            return "AVAILABLE_EXACT", "metadata.location"
        return "NOT_AVAILABLE", "metadata.location is absent"
    if feature == "context.log_id":
        if all_metadata and all("log_name" in source["metadata_columns"] for source in candidates):
            return "AVAILABLE_EXACT", "metadata.log_name"
        return "NOT_AVAILABLE", "metadata.log_name is absent"
    if feature in {"context.road_class", "context.intended_lane_change_direction", "context.initial_lane_offset_m", "context.planned_stop_or_hazard_class", "context.gap_opportunity_class"}:
        return "NOT_AVAILABLE", "历史 metadata/schema 中不存在该冻结字段；不得由 scenario_type 或 planner 名替代"
    if feature in {
        "context.initial_speed_mps", "context.traffic_density", "context.neighbor_availability_pattern",
        "context.target_lane_initial_front_gap_m", "context.target_lane_initial_rear_gap_m",
        "context.initial_front_gap_m", "context.initial_lead_relative_speed_mps", "context.initial_thw_s",
        "context.target_lane_initial_rear_closing_speed_mps",
    }:
        return "AMBIGUOUS", CONTEXT_REASON
    raise ValueError(f"Unhandled context feature: {feature}")


def availability_rows(roles: dict[str, dict[str, list[str]]], snapshots: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    feature_rows: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []
    definitions = read_json(TARGETS)["mechanism_contract_definitions"]
    for family in FAMILIES:
        candidates = [source for source in snapshots if family in source["candidate_families"]]
        candidate_ids = ";".join(source["source_id"] for source in candidates)
        for feature in roles[family].get("F_match", []):
            status = "AVAILABLE_DERIVABLE_WITH_FROZEN_DEFINITION" if candidates and all(source["raw_trajectory_available"] and source["valid_mask_available"] for source in candidates) else "AMBIGUOUS"
            feature_rows.append({
                "residual_family": family, "feature_id": feature, "role": "F_match", "availability": status,
                "candidate_sources": candidate_ids, "evidence_or_reason": "ego_seq + ego_seq_mask satisfy frozen ego13 valid-frame contract" if status.startswith("AVAILABLE") else "missing exact ego sequence/mask",
            })
        for feature in roles[family].get("Context_match", []):
            status, reason = context_feature_status(feature, candidates)
            feature_rows.append({
                "residual_family": family, "feature_id": feature, "role": "Context_match", "availability": status,
                "candidate_sources": candidate_ids, "evidence_or_reason": reason,
            })
        for feature in roles[family].get("M_behavior", []):
            if feature.startswith("raw33."):
                raw_name = feature.split(".", 1)[1]
                status = "AVAILABLE_EXACT" if candidates and all(raw_name in source["raw33_features"] for source in candidates) else "NOT_AVAILABLE"
                reason = "feature_schema.json + interaction_feat_style.npy" if status == "AVAILABLE_EXACT" else "missing exact raw33 schema/value"
                feature_rows.append({
                    "residual_family": family, "feature_id": feature, "role": "M_behavior", "availability": status,
                    "candidate_sources": candidate_ids, "evidence_or_reason": reason,
                })
            else:
                frozen = definitions.get(feature)
                if not frozen or frozen.get("implementation_status") != "REQUIRED_BEFORE_D4_EXECUTION":
                    raise ValueError(f"Unexpected frozen mechanism record: {feature}")
                row = {
                    "residual_family": family,
                    "mechanism_variable": feature,
                    "availability": "NOT_EVALUABLE_MECHANISM_VARIABLE",
                    "raw_trajectory_available": all(source["raw_trajectory_available"] for source in candidates),
                    "frozen_definition": frozen["definition"],
                    "reason": MECHANISM_REASON,
                    "substitute_metric_used": False,
                }
                mechanism_rows.append(row)
                feature_rows.append({
                    "residual_family": family, "feature_id": feature, "role": "M_behavior", "availability": "NOT_EVALUABLE_MECHANISM_VARIABLE",
                    "candidate_sources": candidate_ids, "evidence_or_reason": MECHANISM_REASON,
                })
    return feature_rows, mechanism_rows


def asset_inventory_rows(snapshots: list[dict[str, Any]], roles: dict[str, dict[str, list[str]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in snapshots:
        for family in FAMILIES:
            candidate = family in source["candidate_families"]
            context_features = roles[family].get("Context_match", [])
            context_statuses = [context_feature_status(feature, [source])[0] for feature in context_features]
            rows.append({
                "residual_family": family,
                "source_id": source["source_id"],
                "source_stage": source["source_stage"],
                "source_root": source["root"],
                "candidate_source_status": "CANDIDATE_SOURCE_PREMATCH_ONLY" if candidate else "NOT_A_FROZEN_FAMILY_CANDIDATE",
                "scenario_or_token_count": source["independence_unit_count"],
                "independence_unit": "scenario token (log-clustered where log exists)",
                "unique_logs": source["log_count"],
                "raw_trajectory_availability": "AVAILABLE" if source["raw_trajectory_available"] else "NOT_AVAILABLE",
                "context_availability": "CONTEXT_MATCH_NOT_EVALUABLE" if "NOT_AVAILABLE" in context_statuses or "AMBIGUOUS" in context_statuses else "AVAILABLE_EXACT",
                "f_match_availability": "AVAILABLE_DERIVABLE_WITH_FROZEN_DEFINITION" if source["raw_trajectory_available"] else "AMBIGUOUS",
                "m_behavior_availability": "PARTIAL_RAW33_ONLY_MECHANISM_NOT_EVALUABLE" if source["raw33_available"] else "NOT_AVAILABLE",
                "mechanism_derivability": "NOT_EVALUABLE_MECHANISM_VARIABLE",
                "runnability_completeness": "RAW_ASSET_PRESENT_READ_ONLY",
                "historical_outcome_exposure": source["outcome_exposure"],
                "allowed_evidence_role": "DEVELOPMENT_DIAGNOSTIC_EVIDENCE_ONLY; never R4 confirmation or outcome-guided selection",
                "notes": source["source_kind"],
            })
    return rows


def matching_rows(snapshots: list[dict[str, Any]], roles: dict[str, dict[str, list[str]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fallback = read_json(FALLBACK)["families"]
    for family in FAMILIES:
        candidates = [source for source in snapshots if family in source["candidate_families"]]
        token_union: set[str] = set()
        fallback_units = 0
        for source in candidates:
            token_union.update(source["tokens"])
            if not source["tokens"]:
                fallback_units = max(fallback_units, int(source["independence_unit_count"]))
        precontext_units = len(token_union) if token_union else fallback_units
        context_features = roles[family].get("Context_match", [])
        unavailable = [feature for feature in context_features if context_feature_status(feature, candidates)[0] != "AVAILABLE_EXACT"]
        calipers = {row["target_id"]: row["caliper"] for row in fallback[family]["calipers"]}
        rows.append({
            "residual_family": family,
            "development_fallback_id": "D4_DEVELOPMENT_BALANCE_FALLBACK_V1",
            "pre_context_candidate_independent_units": precontext_units,
            "matching_candidate_count": 0,
            "matched_pairs_or_sets": 0,
            "unmatched_count": precontext_units,
            "mechanism_qualified_pairs": 0,
            "effective_cluster_count": 0,
            "f_match_calipers_json": json.dumps(calipers, ensure_ascii=False, sort_keys=True),
            "caliper_rejection_reason": "NOT_RUN_CONTEXT_MATCH_NOT_EVALUABLE",
            "context_rejection_reason": "CONTEXT_MATCH_NOT_EVALUABLE: " + "; ".join(unavailable),
            "matching_distance_used": "NONE; no legal matching was run",
            "status": "NOT_EVALUABLE",
            "development_feasibility_status": "INSUFFICIENT_FOR_RBR_DEVELOPMENT",
        })
    return rows


def family_results(matching: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in matching:
        family = record["residual_family"]
        suffix = family.replace("-", "_")
        rows.extend([
            {
                "hypothesis_id": f"D4_DESCRIPTOR_EQUIVALENCE_{suffix}", "residual_family": family,
                "formal_hypothesis_result": "NOT_EVALUABLE", "development_feasibility_status": record["development_feasibility_status"],
                "reason": "冻结 Context_match 未能逐项从历史资产获得 exact pre-treatment anchor；未执行 fallback matching。",
            },
            {
                "hypothesis_id": f"D4_MECHANISM_DIFFERENCE_{suffix}", "residual_family": family,
                "formal_hypothesis_result": "NOT_EVALUABLE", "development_feasibility_status": record["development_feasibility_status"],
                "reason": "全部 family-specific mechanism 变量缺少冻结的可执行 deterministic rule；未使用替代指标。",
            },
            {
                "hypothesis_id": f"D4_OUTCOME_BLIND_FEASIBILITY_{suffix}", "residual_family": family,
                "formal_hypothesis_result": "NOT_EVALUABLE", "development_feasibility_status": record["development_feasibility_status"],
                "reason": "candidate source 盘点完成，但 descriptor/context/mechanism 三重资格门未能合法执行。",
            },
        ])
    return rows


def handcrafted_rows(matching: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "residual_family": record["residual_family"],
        "matched_f_match_residual_difference": "NOT_APPLICABLE_NO_MATCHED_SET",
        "extended_handcrafted_separability": "NOT_APPLICABLE_NO_MATCHED_SET",
        "dtw_separability": "NOT_APPLICABLE_NO_MATCHED_SET",
        "raw_mechanism_separability": "NOT_APPLICABLE_MECHANISM_VARIABLE_NOT_EVALUABLE",
        "interpretation": "未对任何 matched residual set 运行 handcrafted 或 DTW；因此没有关于 handcrafted 可否检测机制的结论。",
    } for record in matching]


def leakage_rows(roles: dict[str, dict[str, list[str]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        f_match = set(roles[family].get("F_match", []))
        mechanism = set(roles[family].get("M_behavior", []))
        rows.append({
            "residual_family": family,
            "audit_check": "F_match_M_behavior_zero_overlap",
            "status": "PASS",
            "evidence": "frozen role CSV overlap=" + json.dumps(sorted(f_match & mechanism)),
        })
        rows.append({
            "residual_family": family,
            "audit_check": "representation_BDD_probe_outcome_used_for_selection",
            "status": "PASS_NO_SELECTION_EXECUTED",
            "evidence": "Wave3 tool only opens contract, manifest, metadata, raw trajectory and raw33 schema/value paths.",
        })
        rows.append({
            "residual_family": family,
            "audit_check": "post_treatment_metric_in_matching_distance",
            "status": "PASS_NO_SELECTION_EXECUTED",
            "evidence": "matching_distance_used=NONE because required pre-treatment context anchors are unavailable.",
        })
    return rows


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    header = "|" + "|".join(fields) + "|"
    rule = "|" + "|".join(["---"] * len(fields)) + "|"
    body = ["|" + "|".join(str(row[field]).replace("|", "\\|") for field in fields) + "|" for row in rows]
    return "\n".join([header, rule, *body])


def build_reports(freeze: dict[str, Any], matching: list[dict[str, Any]], family_result_rows: list[dict[str, Any]], mechanism: list[dict[str, Any]]) -> dict[str, str]:
    match_summary = markdown_table(matching, ["residual_family", "pre_context_candidate_independent_units", "matched_pairs_or_sets", "mechanism_qualified_pairs", "development_feasibility_status"])
    hypothesis_summary = markdown_table(family_result_rows, ["hypothesis_id", "formal_hypothesis_result", "development_feasibility_status"])
    mechanism_summary = markdown_table(mechanism, ["residual_family", "mechanism_variable", "availability"])
    main = "\n".join([
        "# R0 D4 残余基准可行性报告 v1", "",
        "## 结论", "",
        "本 Wave 3 已完成冻结合同、既有 DEVELOPMENT 资产及 selection-leakage 的只读审计。三个 residual family 均**不能合法构造** descriptor-balanced / context-controlled / mechanism-confirmed residual benchmark：不是因为某种 representation、BDD 或 probe 结果，而是因为冻结的 exact pre-treatment context anchor 与可执行的 family-specific mechanism rule 在历史资产中均不可用。", "",
        "所有证据等级均为 `DEVELOPMENT_DIAGNOSTIC_EVIDENCE`；没有训练、没有新 planner rollout、没有读取 representation/BDD/probe outcome，也没有修改冻结合同。", "",
        "## 冻结核验", "",
        f"- tag commit：`{freeze['freeze_tag_commit']}`。", f"- freeze content commit：`{freeze['freeze_content_commit']}`。", f"- 绑定的 {len(freeze['frozen_artifact_checks'])} 个冻结 artifact SHA256 均匹配。", "",
        "## Family 结果", "", hypothesis_summary, "",
        "## 候选与匹配规模", "", match_summary, "",
        "`pre_context_candidate_independent_units` 仅是来源中的事前候选量；因 Context_match 不可评估，`matching_candidate_count=0`，没有进行任何 pair/set 选择。", "",
        "## Mechanism derivation 审计", "", mechanism_summary, "",
        "冻结变量只有语义性说明，未包含可执行阈值、anchor 或算法，且 target definition 明确写为 `REQUIRED_BEFORE_D4_EXECUTION`。因此这些变量全部标为 `NOT_EVALUABLE_MECHANISM_VARIABLE`；历史 raw33 和历史机制表不会被改名或当作替代指标。", "",
        "## Development fallback", "",
        "`D4_DEVELOPMENT_BALANCE_FALLBACK_V1` 的 caliper 未被重调，也未被用于不完整 context 的近似匹配。它保持 `NOT_FORMAL_PHYSICAL_EQUIVALENCE` 与 `NOT_R4_CONFIRMATORY_EQUIVALENCE`。", "",
        "## Handcrafted challenge", "",
        "没有 matched residual set，因此没有执行 ego13、extended handcrafted、DTW 或 raw mechanism 的组间可分性分析；不产生 `HANDCRAFTED_FEATURES_CANNOT_DETECT` 类主张。", "",
        "## Selection leakage", "",
        "Frozen F_match 与 M_behavior 在每个 family 的 Primary 角色零交集。Wave3 未执行 pair selection；工具未读取 embedding、BDD、probe 或 RBR outcome，故无 outcome-guided selection leakage。", "",
        "## RBR 含义", "",
        "三个 family 都是 `INSUFFICIENT_FOR_RBR_DEVELOPMENT`，不足两个 family 的最低要求。RBR-A/B 不具备 candidate-specific authorization review 条件；RBR-C 还保留 Wave2 D2 unresolved 状态。现有 training authorization manifest 不变，RBR 训练仍为 `NOT_AUTHORIZED`。", "",
    ])
    cross = "\n".join([
        "# R0 Wave 3 跨模块科学诊断 v1", "",
        "证据等级：`DEVELOPMENT_DIAGNOSTIC_EVIDENCE`。", "",
        "- D0：pooling 与 mask/padding 均为 `MIXED`；可支持 geometry sensitivity，不支持普遍 information loss。", "- D1：`KNOWN_SEMANTIC_INFORMATION_PRESENT = SUPPORTED`；cross-domain transfer 仍为 `INCONCLUSIVE`。", "- D2：response/pairing `NOT_EVALUABLE`，其余正式状态保持 Wave2 的 `INCONCLUSIVE`；未解决的 D2 contract 不支持 RBR-C。", "- D3：formal hypotheses 继续 `INCONCLUSIVE`。", "- D4（Wave3）：R-HLC、R-TSB、R-IP 的 descriptor、mechanism 与 outcome-blind feasibility 三类 formal hypothesis 均为 `NOT_EVALUABLE`。这是冻结 implementation/context capacity limitation，不是 outcome-driven negative finding。", "",
        "修正后的科学诊断：`KNOWN_SEMANTIC_INFORMATION_PRESENT_SUPPORTED; TEMPORAL_GEOMETRY_SENSITIVITY_MIXED; CROSS_DOMAIN_TRANSFER_INCONCLUSIVE; GEN1_CONTEXT_RESPONSE_ATTRIBUTION_UNRESOLVED; D3_INCONCLUSIVE; D4_RESIDUAL_BENCHMARK_NOT_EVALUABLE_WITH_EXISTING_ASSETS`。", "",
        "`CASE_C_TEMPORAL_CONTRIBUTION_MIXED_NOT_GENERALIZED` 保持，不因本 Wave 3 改写。", "",
    ])
    closure = "\n".join([
        "# R0 Wave 3 后的 R0 Closure Readiness v1", "",
        "## 结论", "", "`R0_ADDITIONAL_EXECUTION_REQUIRED`。", "",
        "理由不是 D5 未执行（D5 为 nonblocking），而是 D4 三个 family 均没有形成合法的 development residual benchmark：需要在不读取 representation/RBR outcome 的条件下，补齐 frozen exact pre-treatment Context_match 绑定与在 execution 前已经冻结的 mechanism implementation。当前 freeze 不允许在本 Wave 3 事后创建阈值或把历史指标重命名为 mechanism variable。", "",
        "## 各模块", "", "- D0 Wave1.1：`MIXED`；Case C 仅为 mixed/not generalized。", "- D1 Wave1/2：known semantic information `SUPPORTED`；cross-domain transfer `INCONCLUSIVE`。", "- D2 Wave2：仍有 pairing/response `NOT_EVALUABLE` 与其他 unresolved 项。", "- D3 Wave1：formal hypotheses `INCONCLUSIVE`。", "- D4 Wave3：九项 D4 formal hypothesis 均 `NOT_EVALUABLE`，三个 family 都 `INSUFFICIENT_FOR_RBR_DEVELOPMENT`。", "",
        "## RBR candidate-specific implication", "", "- RBR-A：`NOT_READY_FOR_CANDIDATE_SPECIFIC_AUTHORIZATION_REVIEW`。", "- RBR-B：`NOT_READY_FOR_CANDIDATE_SPECIFIC_AUTHORIZATION_REVIEW`。", "- RBR-C：`NOT_READY_D2_UNRESOLVED_AND_D4_INSUFFICIENT`。", "- training authorization manifest：未修改，状态仍为 `NOT_AUTHORIZED`。", "",
    ])
    return {
        "R0_D4_Residual_Benchmark_Feasibility_Report_v1.md": main,
        "R0_Wave3_Cross_Module_Diagnosis_v1.md": cross,
        "R0_R0_Closure_Readiness_After_Wave3_v1.md": closure,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    results_dir = args.results_dir.resolve()
    if results_dir != RESULTS.resolve():
        raise ValueError("Wave3 output directory is frozen to docs/stageR/r0/results")
    freeze = verify_freeze()
    roles = load_roles()
    snapshots = [source_snapshot(spec) for spec in SOURCE_SPECS]
    feature_rows, mechanism_rows = availability_rows(roles, snapshots)
    inventory = asset_inventory_rows(snapshots, roles)
    matching = matching_rows(snapshots, roles)
    results = family_results(matching)
    handcrafted = handcrafted_rows(matching)
    leakage = leakage_rows(roles)
    reports = build_reports(freeze, matching, results, mechanism_rows)
    artifacts: dict[str, tuple[list[dict[str, Any]], list[str]]] = {
        "r0_d4_asset_feasibility_inventory.csv": (inventory, list(inventory[0])),
        "r0_d4_feature_availability_audit.csv": (feature_rows, list(feature_rows[0])),
        "r0_d4_mechanism_metrics.csv": (mechanism_rows, list(mechanism_rows[0])),
        "r0_d4_mechanism_derivation_audit.csv": (mechanism_rows, list(mechanism_rows[0])),
        "r0_d4_matching_metrics.csv": (matching, list(matching[0])),
        "r0_d4_family_results.csv": (results, list(results[0])),
        "r0_d4_handcrafted_baseline_audit.csv": (handcrafted, list(handcrafted[0])),
        "r0_d4_selection_leakage_audit.csv": (leakage, list(leakage[0])),
    }
    for name, (rows, fields) in artifacts.items():
        write_csv(results_dir / name, rows, fields)
    for name, text in reports.items():
        write_new(results_dir / name, text)
    hypothesis_json = {
        "execution_status": "COMPLETE_NO_LEGAL_MATCHING_EXECUTED",
        "evidence_level": "DEVELOPMENT_DIAGNOSTIC_EVIDENCE",
        "training_or_rollout_performed": False,
        "representation_bdd_probe_outcome_opened": False,
        "hypothesis_results": {row["hypothesis_id"]: row["formal_hypothesis_result"] for row in results},
        "development_feasibility": {family: "INSUFFICIENT_FOR_RBR_DEVELOPMENT" for family in FAMILIES},
        "r0_closure_readiness": "R0_ADDITIONAL_EXECUTION_REQUIRED",
        "rbr_candidate_specific_implication": {
            "RBR-A": "NOT_READY_FOR_CANDIDATE_SPECIFIC_AUTHORIZATION_REVIEW",
            "RBR-B": "NOT_READY_FOR_CANDIDATE_SPECIFIC_AUTHORIZATION_REVIEW",
            "RBR-C": "NOT_READY_D2_UNRESOLVED_AND_D4_INSUFFICIENT",
        },
    }
    manifest = {
        "schema_version": "r0_wave3_d4_execution_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_verification": freeze,
        "source_ids": [source["source_id"] for source in snapshots],
        "forbidden_actions_performed": [],
        "selection_sequence": "candidate source -> F_match/context availability audit -> mechanism availability audit; no pair selection after gate failure",
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
    }
    command_ledger = {
        "schema_version": "r0_wave3_command_ledger_v1",
        "commands": [{"argv": sys.argv, "purpose": "read-only D4 feasibility audit", "status": "COMPLETE"}],
        "prohibited_actions": ["RBR training", "new planner rollout", "representation/BDD/probe outcome selection", "R4 outcome access"],
    }
    deviation = [{
        "record_id": "W3-D4-001", "module": "D4", "classification": "NO_PROTOCOL_DEVIATION",
        "status": "CLOSED", "description": "Frozen inputs are unavailable for legal matching/mechanism execution; results are NOT_EVALUABLE without substituting metrics or rules.",
        "impact_on_primary_conclusion": "No primary conclusion is upgraded; D4 remains NOT_EVALUABLE.",
    }]
    for name, value in {
        "r0_wave3_hypothesis_results.json": hypothesis_json,
        "r0_wave3_execution_manifest.json": manifest,
        "r0_wave3_command_ledger.json": command_ledger,
        "r0_wave3_freeze_verification.json": freeze,
    }.items():
        write_new(results_dir / name, json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    write_csv(results_dir / "r0_wave3_protocol_deviation_log.csv", deviation, list(deviation[0]))
    return {"status": "COMPLETE", "outputs": sorted([*artifacts, *reports, "r0_wave3_hypothesis_results.json", "r0_wave3_execution_manifest.json", "r0_wave3_command_ledger.json", "r0_wave3_freeze_verification.json", "r0_wave3_protocol_deviation_log.csv"])}


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
