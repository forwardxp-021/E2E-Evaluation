#!/usr/bin/env python3
"""Generate the metadata-only S1R protocol-correction impact audit.

The audit reads frozen protocol files and identity keys from already-bound
historical manifests. It does not inspect scientific values, execute a runner,
construct a simulator, train a representation, or access a future evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/stageR/s1r_protocol_correction"
S1 = ROOT / "docs/stageR/s1"
HISTORY = ROOT / "docs/stageR/r0/manifests/r0_nuplan_historical_use_ledger_v0.1.csv"
CENSUS = ROOT / "docs/stageR/s2_preflight/S2_Preflight_Eligibility_Census_v1.json"
EXPOSURE = ROOT / "docs/stageR/s2_preflight/S2_Preflight_Exposure_Exclusion_Ledger_v1.json"
TASK_INPUT = Path("/Users/liuqing/.codex/attachments/772d7369-86de-4b5d-bcea-08c3bdfb8f72/pasted-text.txt")
V23 = ROOT / "RBR-64_博士研究总体方案_v2.3.md"
SESSION_RE = re.compile(r"^(\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}_veh-\d+)(?:_\d+_\d+)?$")
HEX16_RE = re.compile(r"^[0-9a-fA-F]{16}$")
TOKEN_KEYS = {"scenario_token", "actual_nuplan_token", "lidar_pc_token", "sample_token"}
LOG_KEYS = {"log_name", "logfile", "db_file", "db_name", "database"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_strings(values: Iterable[str]) -> str:
    h = hashlib.sha256()
    for value in sorted(set(values)):
        h.update(value.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def normal_token(value: Any) -> str | None:
    text = str(value).strip().lower()
    return text if HEX16_RE.fullmatch(text) else None


def normal_log(value: Any) -> str | None:
    text = Path(str(value).strip()).name.removesuffix(".db")
    return text if re.match(r"^20\d\d\.\d\d\.\d\d\.", text) else None


def extract_json_identity(value: Any, tokens: set[str], logs: set[str], key: str = "") -> None:
    if isinstance(value, dict):
        for child_key, child in value.items():
            extract_json_identity(child, tokens, logs, str(child_key).lower())
    elif isinstance(value, list):
        for child in value:
            extract_json_identity(child, tokens, logs, key)
    elif key in TOKEN_KEYS:
        token = normal_token(value)
        if token:
            tokens.add(token)
    elif key in LOG_KEYS:
        log_name = normal_log(value)
        if log_name:
            logs.add(log_name)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def session(log_name: str) -> str | None:
    match = SESSION_RE.fullmatch(Path(log_name).name.removesuffix(".db"))
    return match.group(1) if match else None


def identities(path: Path) -> tuple[set[str], set[str]]:
    tokens: set[str] = set()
    logs: set[str] = set()
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as stream:
            for row in csv.DictReader(stream):
                for key, value in row.items():
                    normalized = str(key).lower().strip()
                    if normalized in TOKEN_KEYS:
                        token = normal_token(value)
                        if token:
                            tokens.add(token)
                    elif normalized in LOG_KEYS:
                        log_name = normal_log(value)
                        if log_name:
                            logs.add(log_name)
    else:
        extract_json_identity(read_json(path), tokens, logs)
    return tokens, logs


def recursive_log_sessions(path: Path, allowed: set[str]) -> set[str]:
    result: set[str] = set()

    def walk(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                walk(child, str(child_key).lower())
        elif isinstance(value, list):
            for child in value:
                walk(child, key)
        elif isinstance(value, str) and any(marker in key for marker in ("log", "database", "db_file")):
            parsed = session(value)
            if parsed in allowed:
                result.add(parsed)

    walk(read_json(path))
    return result


def audit_sessions() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    census = read_json(CENSUS)
    exposure = read_json(EXPOSURE)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in census["logs"]:
        groups[row["session_id"]].append(row)
    all_sessions = set(groups)
    old_exposed = set(exposure["historical_sessions"]) & all_sessions
    old_conflict = set(exposure["conflict_sessions"]) & all_sessions
    evidence: dict[str, set[str]] = defaultdict(set)
    observed_classes: dict[str, set[str]] = defaultdict(set)

    history_rows = list(csv.DictReader(HISTORY.open(encoding="utf-8", newline="")))
    for row in history_rows:
        path = ROOT / row["source_manifest"]
        if sha256_file(path) != row["manifest_sha256"]:
            raise ValueError(f"historical source changed: {path}")
        tokens, logs = identities(path)
        if digest_strings(tokens) != row["nuplan_token_set_sha256"] or digest_strings(logs) != row["nuplan_log_set_sha256"]:
            raise ValueError(f"historical identity set changed: {path}")
        classification = (
            "EXPOSED"
            if row["outcome_already_unblinded"] == "true"
            else "RESERVED"
            if row["use_type"] == "FROZEN_OR_LOCKED_ROSTER"
            else "SCREENED"
        )
        source_evidence = "|".join(
            [row["historical_stage"], row["use_type"], classification, row["source_manifest"]]
        )
        for log_name in logs:
            grouped = session(log_name)
            if grouped in all_sessions:
                evidence[grouped].add(source_evidence)
                observed_classes[grouped].add("E1_UNRELATED_HISTORICAL_USE" if classification == "EXPOSED" else "E0_METADATA_ONLY")

    engineering_patterns = [
        "docs/stageR/r1/*run_ledger*.json",
        "docs/stageR/r1/*outcome_exposure*.json",
        "docs/stageR/r2/*run_ledger*.json",
        "docs/stageR/r2/*outcome_exposure*.json",
    ]
    engineering_paths: list[Path] = []
    for pattern in engineering_patterns:
        engineering_paths.extend(ROOT.glob(pattern))
    engineering_sessions: set[str] = set()
    for path in sorted(set(engineering_paths)):
        local = recursive_log_sessions(path, all_sessions)
        for grouped in local:
            engineering_sessions.add(grouped)
            evidence[grouped].add(f"R1_R2_BENCHMARK_ENGINEERING|{path.relative_to(ROOT)}")
            observed_classes[grouped].add("E2_BENCHMARK_ENGINEERING")

    rows: list[dict[str, Any]] = []
    class_counts = Counter()
    for grouped in sorted(all_sessions):
        if grouped in engineering_sessions:
            primary_class = "E2_BENCHMARK_ENGINEERING"
            confidence = "HIGH_IDENTITY_AND_ROLE_EVIDENCE"
        elif grouped in old_exposed:
            primary_class = "E1_UNRELATED_HISTORICAL_USE"
            confidence = "HIGH_IDENTITY_MEDIUM_CLAIM_RELEVANCE"
        elif grouped in old_conflict:
            primary_class = "E0_METADATA_ONLY"
            confidence = "HIGH_IDENTITY_RESERVATION_WITHOUT_OUTCOME"
        else:
            primary_class = "E5_UNTOUCHED_CONFIRMATORY"
            confidence = "MEDIUM_NO_MATCH_IN_BOUND_HISTORY_OR_RESERVATIONS"
        class_counts[primary_class] += 1
        local_rows = groups[grouped]
        classes = set(observed_classes[grouped])
        classes.add(primary_class)
        rows.append(
            {
                "session_id": grouped,
                "log_count": len(local_rows),
                "scenario_token_count": sum(row["scenario_count"] for row in local_rows),
                "vehicle_ids": ";".join(sorted({str(row["vehicle_id"]) for row in local_rows})),
                "map_locations": ";".join(sorted({str(row["map_location"]) for row in local_rows})),
                "old_strict_outcome_exposed": str(grouped in old_exposed).lower(),
                "old_strict_exposure_or_reservation_conflict": str(grouped in old_conflict).lower(),
                "observed_exposure_class": primary_class,
                "all_observed_exposure_classes": ";".join(sorted(classes)),
                "exposure_evidence_json": json.dumps(sorted(evidence[grouped]), ensure_ascii=False, separators=(",", ":")),
                "classification_confidence": confidence,
                "E3_primary_method_development_evidence": "NOT_FOUND_IN_BOUND_IDENTITY_LEDGER",
                "E4_primary_outcome_adaptation_evidence": "NOT_FOUND_PRIMARY_NEVER_AUTHORIZED",
                "controlled_V_exposure_policy_status": "PROVISIONALLY_ALLOWED_AFTER_OWNER_AMENDMENT",
                "current_V_certification": "NOT_CERTIFIED_APPLICABILITY_AND_PROTOCOL_AMENDMENT_PENDING",
                "benchmark_B_status": "RECORDED_BENCHMARK_ENGINEERING" if primary_class == "E2_BENCHMARK_ENGINEERING" else "NOT_BOUND_TO_B",
                "claim_A_unseen_generator_status": "NOT_UNSEEN" if primary_class == "E2_BENCHMARK_ENGINEERING" else "NOT_ESTABLISHED",
                "confirmatory_C_candidate": str(primary_class == "E5_UNTOUCHED_CONFIRMATORY").lower(),
                "statistical_cluster": grouped,
                "notes": "SESSION remains the dependence cluster; exposure role does not create independent logs.",
            }
        )
    summary = {
        "total_sessions": len(all_sessions),
        "old_strict_outcome_exposed_sessions": len(old_exposed),
        "old_strict_conflict_sessions": len(old_conflict),
        "classification_counts": dict(sorted(class_counts.items())),
        "E3_count": 0,
        "E4_count": 0,
        "UNKNOWN_count": 0,
        "controlled_V_exposure_policy_upper_bound": len(all_sessions),
        "controlled_V_current_certified": 0,
        "claim_A_generator_unseen_upper_bound": len(all_sessions - engineering_sessions),
        "confirmatory_C_candidate_sessions": class_counts["E5_UNTOUCHED_CONFIRMATORY"],
        "benchmark_engineering_sessions": len(engineering_sessions),
        "history_source_files_verified": len(history_rows),
        "benchmark_engineering_identity_sources": [str(path.relative_to(ROOT)) for path in sorted(set(engineering_paths))],
    }
    return rows, summary


def impact_rows() -> list[dict[str, str]]:
    rows = [
        ("S1_Scope_and_Claim_Freeze_Draft_v0.1.md", "Level 1/Level 2 questions", "BDD utility is Primary; Level 2 is conditional", "Role correction does not alter the dissertation question", "KEEP_AS_IS", "The estimand and claim hierarchy do not depend on an absolute untouched-session rule", "Preserve verbatim", "false", "false", "false"),
        ("S1_Scope_and_Claim_Freeze_Draft_v0.1.md", "HLC state", "HLC closed by scope; impossibility not established", "No HLC reopening", "KEEP_AS_IS", "Validation-role correction supplies no new HLC evidence", "Preserve verbatim", "false", "false", "false"),
        ("S1_Scope_and_Claim_Freeze_Draft_v0.1.md", "TSB state", "Frozen development candidate pending fresh qualification", "TSB remains unqualified but freshness is no longer the qualification's sole role", "OWNER_AMENDMENT_REQUIRED", "Candidate state stays frozen; qualification purpose must separate mechanism validity from exposure status", "Amend wording prospectively; keep candidate and negative limitations", "true", "false", "true"),
        ("S1_Scope_and_Claim_Freeze_Draft_v0.1.md", "RBR training architecture boundary", "Encoder selection is U-only", "U-only remains the core anti-bias firewall", "KEEP_AS_IS", "TSB or nuPlan outcomes must not select encoder architecture/objective/checkpoint/seed", "Preserve verbatim", "false", "false", "false"),
        ("S1_TSB_Applicability_Contract_Draft_v0.1.md", "Frozen TSB parameters and thresholds", "Generator, initial-speed screen, mechanism, F_match and safety are frozen", "Exposure correction does not change physical/measurement contracts", "KEEP_AS_IS", "Data role does not justify retuning generator or gates", "Preserve hashes and thresholds", "false", "false", "false"),
        ("S1_TSB_Applicability_Contract_Draft_v0.1.md", "Full denominator and failure retention", "LOW_SPEED_ENDSTOP and mechanism/safety failures remain in denominator", "No survivor selection remains mandatory", "KEEP_AS_IS", "Outcome-independent inclusion protects the controlled estimand", "Preserve verbatim", "false", "false", "false"),
        ("S1_TSB_Fresh_Qualification_Protocol_Draft_v0.1.md", "Q scientific purpose", "Q is completely fresh TSB qualification", "Q becomes benchmark engineering/mechanism qualification", "DEPRECATE_AND_REPLACE", "Absolute historical untouchedness is not necessary for Claim B controlled utility", "Replace Q role with prospective B qualification while retaining all gates", "true", "false", "true"),
        ("S1_TSB_Fresh_Qualification_Protocol_Draft_v0.1.md", "Whole-roster all-pass and stop rules", "All fixed pairs pass; first scientific failure stops", "Benchmark qualification still needs a predefined denominator", "KEEP_AS_IS", "Role correction does not license cherry-picking or top-up", "Preserve unless Owner separately changes the benchmark claim", "false", "false", "false"),
        ("S1_TSB_Fresh_Qualification_Protocol_Draft_v0.1.md", "Q20/Q12 role", "20 fresh pairs; Q12 pre-execution fallback", "Size may remain a B engineering choice but no longer determines V/C availability", "OWNER_AMENDMENT_REQUIRED", "Old capacity conclusion was driven by strict exposure exclusion", "Rejustify size and label under B; do not execute before amendment", "true", "false", "true"),
        ("S1_TSB_Fresh_Qualification_Protocol_Draft_v0.1.md", "Q to D transition", "Passing Q becomes D", "B evidence remains B; method development and V/C roles require explicit prospective bindings", "DEPRECATE_AND_REPLACE", "Automatic role conversion obscures claim-relevant adaptation", "Replace with U/B/V/C transition rules", "true", "false", "true"),
        ("S1_Handcrafted_Challenger_Contract_Draft_v0.1.md", "H definition", "Development-informed 30D handcrafted challenger", "H may use benchmark mechanism knowledge", "KEEP_AS_IS", "Development-informed H is scientifically disclosed and remains the sole comparator", "Preserve H definition and implementation", "false", "false", "false"),
        ("S1_Handcrafted_Challenger_Contract_Draft_v0.1.md", "H freeze timing", "No post-D/E feature change", "Freeze all H choices at PRIMARY_FREEZE_POINT", "WORDING_UPDATE_ONLY", "Logic is unchanged; terminology should name the new prospective boundary", "Add cross-reference in amendment only", "true", "false", "true"),
        ("S1_BDD_Statistical_Analysis_Plan_Draft_v0.1.md", "Primary metric/operating point", "Delta BDD at fixed alpha, m, pi and one comparison", "Primary choices freeze before V", "KEEP_AS_IS", "The anti-adaptation rule directly implements claim-relevant contamination control", "Preserve metric and operating point", "false", "false", "false"),
        ("S1_BDD_Statistical_Analysis_Plan_Draft_v0.1.md", "Paired/unpaired boundary", "Unpaired preferred; paired fallback has reduced claim", "Exposure correction alone does not alter design estimands", "KEEP_AS_IS", "Dependence and estimand differences remain", "Preserve; any switch still requires Owner amendment", "false", "false", "false"),
        ("S1_BDD_Statistical_Analysis_Plan_Draft_v0.1.md", "E disjointness", "E disjoint from all U/Q/D logs", "C should be untouched; V need only be free of claim-relevant adaptation", "OWNER_AMENDMENT_REQUIRED", "Complete historical disjointness is stronger than Claim B requires", "Split V controlled evaluation from C confirmatory robustness", "true", "false", "true"),
        ("S1_BDD_Statistical_Analysis_Plan_Draft_v0.1.md", "Cluster-aware inference", "Log/source dependence retained in bootstrap", "SESSION remains statistical cluster", "KEEP_AS_IS", "Exposure eligibility and dependence are distinct", "Strengthen SESSION wording without changing method", "false", "false", "false"),
        ("S1_Secondary_Diagnostics_SAP_Draft_v0.1.md", "Secondary non-veto diagnostics", "Diagnostics cannot replace Primary or select encoder", "No role correction changes diagnostic hierarchy", "KEEP_AS_IS", "Protects the Primary from favorable secondary substitution", "Preserve verbatim", "false", "false", "false"),
        ("S1_Data_Firewall_Draft_v0.1.json", "U role and encoder selection", "U solely selects encoder", "U meaning is preserved", "KEEP_AS_IS", "This is the strongest claim-relevant firewall", "Preserve structured fields", "false", "false", "false"),
        ("S1_Data_Firewall_Draft_v0.1.json", "Q/D/E roles", "Fresh Q, converted D, untouched E", "Use U/B/V/C with contamination tied to decision use", "DEPRECATE_AND_REPLACE", "Old roles conflate benchmark engineering exposure with Primary-method contamination", "Owner-approved replacement schema required", "true", "false", "true"),
        ("S1_Canonical_Schema_Draft_v0.1.json", "Scientific trace schema", "Canonical identity, trace and outcome fields", "Data role correction does not alter trace semantics", "KEEP_AS_IS", "Exposure taxonomy is governance metadata", "Preserve scientific fields", "false", "false", "false"),
        ("S1_Canonical_Schema_Draft_v0.1.json", "Role vocabulary", "Q/D/E-bound role values", "Schema must encode B/V/C and exposure taxonomy", "OWNER_AMENDMENT_REQUIRED", "Future manifests need unambiguous prospective roles", "Add versioned role extension; do not rewrite v0.1", "true", "false", "true"),
        ("S1_Protocol_Design_Report_v0.1.md", "Historical S1 conclusion", "S1 frozen under strict prospective firewall", "Historical record remains correct", "KEEP_AS_IS", "A later conceptual correction does not invalidate the recorded audit", "Preserve; cite from S1R addendum", "false", "false", "false"),
        ("S1_Protocol_Design_Manifest_v0.1.json", "Artifact hashes and approvals", "Binds frozen S1 package", "Must remain immutable", "KEEP_AS_IS", "Amendment requires a new manifest rather than hash rewrites", "Preserve byte-for-byte", "false", "false", "false"),
        ("S1_Protocol_Design_Manifest_v0.1.json", "Q contract/data roles", "Q20 strict freshness and Q-to-D", "Prospective amendment must supersede role semantics", "OWNER_AMENDMENT_REQUIRED", "Manifest is normative and cannot be changed indirectly", "Create separate Owner-approved amendment manifest", "true", "false", "true"),
        ("S2_Preflight_Eligibility_Report_v1.md", "243-session conflict capacity", "Old strict firewall leaves upper bound five", "Historical result remains correct under old policy", "KEEP_AS_IS", "Do not rewrite historical counts", "Relabel only in new S1R interpretation", "false", "false", "false"),
        ("S2_Preflight_Readiness_Report_v1.md", "S2 blocker", "Blocked by Q20 capacity plus engineering bindings", "Capacity component becomes historical after amendment; engineering blocks remain", "OWNER_AMENDMENT_REQUIRED", "New exposure policy changes future eligibility but not reset/execution readiness", "Before amendment keep blocker active; after amendment rerun preflight", "true", "false", "true"),
        ("S2_Capacity_Recovery_Report_v1.md", "No fresh external reservoir", "No new raw fresh source found", "Discovery fact remains true", "KEEP_AS_IS", "Role correction does not create new source files", "Preserve as historical discovery", "false", "false", "false"),
        ("S2_Capacity_Recovery_Report_v1.md", "Capacity interpretation", "No strict-fresh capacity recovery", "Does not equal no controlled-V capacity", "WORDING_UPDATE_ONLY", "Same counts answer a different scientific question", "Use S1R addendum; do not edit report", "true", "false", "true"),
        ("handover0917.md", "Current protocol entry", "S1 frozen; S2-PRE next under U/Q/D/E", "Now stale relative to completed S2-PRE and proposed correction", "WORDING_UPDATE_ONLY", "Handover should point to S1R but is not the amendment itself", "Update only after Owner disposition", "true", "false", "true"),
    ]
    fields = [
        "document", "section", "current_rule", "v2.3_concept", "impact_class", "scientific_reason",
        "recommended_action", "owner_approval_required", "historical_result_affected", "future_execution_affected",
    ]
    return [dict(zip(fields, row)) for row in rows]


def taxonomy_text(summary: dict[str, Any]) -> str:
    return f"""# S1R Exposure Taxonomy v1 — PROPOSED ONLY

状态：`PROPOSED_ONLY`。`OWNER_APPROVAL_REQUIRED = TRUE`。本文件不修改冻结 S1。

## 核心原则

隔离规则应对应当前 claim 的偏差路径：某 session 的 outcome 只有在被用于选择、修改或优化当前 Primary comparison 的 RBR、H、BDD statistic、calibration、operating point、cohort inclusion 或 sample-size decision 时，才构成 confirmatory contamination。历史使用仍需完整披露，但“用过”不自动等于“未来 controlled evaluation 不可用”。

| 类别 | 定义 | Controlled V | Confirmatory C |
|---|---|---|---|
| `E0_METADATA_ONLY` | 仅身份、schema、route 或其他 pre-outcome metadata | 可在 amendment 与 eligibility 完成后进入 | 可候选，但既有 reservation 必须先解决 |
| `E1_UNRELATED_HISTORICAL_USE` | 普通 Stage6/7/7L、old64/ego13 等与当前 Primary 方法选择无直接绑定的历史 outcome | 可透明标记后进入 | 不属于 E5 untouched |
| `E2_BENCHMARK_ENGINEERING` | HLC/TSB/controller/mechanism/F_match/safety/applicability 工程 | Claim B 可进入；Claim A unseen-generalization 不可当 unseen | 不属于 E5 untouched |
| `E3_PRIMARY_METHOD_DEVELOPMENT` | 直接用于当前 H、BDD readout/kernel/calibration/operating point/eligibility method 开发 | 不得用于同一冻结方法的无条件 confirmatory claim；只可作为开发/敏感性证据 | 排除 |
| `E4_PRIMARY_OUTCOME_DRIVEN_ADAPTATION` | 看过 Primary-like RBR-vs-H outcome 后改变方法、cohort、阈值、样本量或分析 | 排除对应 confirmatory claim | 严格排除 |
| `E5_UNTOUCHED_CONFIRMATORY` | 未参与 E1–E4 outcome use，且无冲突 reservation 的相对 untouched session | 可进入 | 首选 C 候选 |
| `UNKNOWN` | 身份或 decision-use provenance 不足 | 不默认可用 | 不可进入 |

E5 是 confirmatory role 标签，不表示“风险高于 E4”。逐会话 CSV 为了给出互斥统计，将无已知历史 exposure/reservation 的会话归入 E5；同时保留完整 evidence list。多重 exposure 使用 E4 > E3 > E2 > E1 > E0 的最高风险，E5 仅在没有 E1–E4 时成立。

## 当前证据分层

- E0：{summary['classification_counts'].get('E0_METADATA_ONLY', 0)}
- E1：{summary['classification_counts'].get('E1_UNRELATED_HISTORICAL_USE', 0)}
- E2：{summary['classification_counts'].get('E2_BENCHMARK_ENGINEERING', 0)}
- E3：0（当前绑定账本未发现直接证据）
- E4：0（当前 Primary 从未授权；未发现 outcome-driven adaptation）
- E5：{summary['classification_counts'].get('E5_UNTOUCHED_CONFIRMATORY', 0)}
- UNKNOWN：0 个身份分类；但 E1 的“与当前方法完全无 decision-use 关系”只有中等置信度，因为旧账本没有专门记录该因果用途。

这些计数是 exposure-policy reclassification，不是 applicability、route、reset 或生产执行资格认证。
"""


def role_text(summary: dict[str, Any]) -> str:
    return f"""# S1R Proposed Data Role Model v1 — U/B/V/C

状态：`PROPOSED_ONLY`。不得据此执行 simulation、training 或 evaluation。

## U — Representation Training

只用于 RBR encoder training/validation。Architecture、objective、checkpoint、seed handling 与 representation interface 只能由 U 的冻结目标决定。TSB/HLC/nuPlan V/C outcome、H 表现或 Primary BDD outcome 均不得进入 encoder selection。

## B — Benchmark Development

用于 TSB/HLC/controller、mechanism、F_match、safety、applicability 与 technical completeness。B exposure 必须透明标记。它排除 Claim A“TSB 对 unseen sessions 泛化”的 unseen 身份，但不自动排除 Claim B“在冻结 controlled intervention 下 RBR-BDD vs H-BDD utility”。

## V — Controlled Evaluation Pool

进入 V 前必须到达 `PRIMARY_FREEZE_POINT`，并通过既有 applicability、technical completeness、route/reference 与全 denominator 规则。按 exposure policy 的理论上界是 {summary['controlled_V_exposure_policy_upper_bound']} sessions；当前认证数仍为 {summary['controlled_V_current_certified']}，因为 amendment、production/reset/precontext 与 eligibility 尚未闭合。

## C — Confirmatory Robustness

优先使用 E5，相对 untouched，用于验证 V 结果能否复现。当前仅有 {summary['confirmatory_C_candidate_sessions']} 个 identity-level 候选，远不足以自行定义 C sample size。C 不是论文唯一合法证据来源，也未被构造或访问。

## PRIMARY_FREEZE_POINT

在任何 V outcome access 前必须一次性冻结并绑定 SHA：

1. RBR encoder architecture、objective、checkpoint、seed rule 与 representation interface；
2. H feature list、implementation、normalization、preprocessing 与 readout；
3. BDD statistic、scaler、kernel/bandwidth rule、null/calibration procedure；
4. Primary metric、operating point、FPR gate、paired/unpaired design；
5. scenario eligibility、failure retention、denominator、sample-size 与 stopping rule；
6. analysis plan、cluster-aware inference、missingness/failure policy 与 multiplicity family；
7. U/B/V/C identity bindings和 exposure taxonomy classification。

Freeze 后禁止依据 V/C outcome 添加或删除 H 特征、切换 encoder/kernel、重校 operating point、筛选 favorable scenarios、扩样或替换 Primary。任何必要修复必须先判定为 infrastructure-only，且不得观察或利用 Primary scientific value。

## Q/D/E 处置

- 保留：U-only encoder firewall、calibration 与 evaluation 分离、Primary 后不适配、E/C 不扩样、全 denominator、cluster-aware inference。
- 替换：absolute-fresh Q、Q PASS 自动变 D、所有历史使用均永久排除 E。
- 新语义：Q 的机制/测量工作归 B；D 的方法开发职能分配到 B 或显式 pre-freeze method development；大规模 controlled comparison 归 V；相对 untouched robustness 归 C。
"""


def report_text(summary: dict[str, Any], impacts: list[dict[str, str]], base_head: str) -> str:
    impact_counts = Counter(row["impact_class"] for row in impacts)
    return f"""# S1R Protocol Correction Impact Audit v1

状态：**S1R_IMPACT_AUDIT_SUPPORTS_PROTOCOL_AMENDMENT**

文档性质：`PROPOSED_ONLY`；`OWNER_APPROVAL_REQUIRED = TRUE`

审计基线：`{base_head}`

## Overall scientific verdict

科学上支持一个 prospective Owner amendment。理由不是旧规则造成容量不足，而是 data role 与当前 claim 的偏差路径不匹配：nuPlan 在本研究中承担 frozen closed-loop intervention 的 controlled validation platform；普通历史 rollout 或 benchmark engineering 本身不会让 RBR-vs-H comparison 偏向某一方法。真正需要隔离的是 outcome 对当前 encoder、H、BDD、calibration、operating point、cohort 与 sample-size 决策的反馈。

这一修正不降低 applicability、mechanism、F_match、safety、technical completeness、全 denominator、no-survivor、SESSION cluster 或统计校准要求。它也不把 TSB 升级为 scientific-qualified benchmark。

审计已读取并绑定 `RBR-64_博士研究总体方案_v2.3.md`（状态为 controlled-validation freeze draft），并以任务正文作为审计范围补充。v2.3 本身不等于 Owner-approved S1 amendment；冻结 S1 只有在 Scientific Owner 批准独立、版本化的 amendment 后才改变。

## Frozen contract impact

影响矩阵共 {len(impacts)} 条：KEEP_AS_IS={impact_counts['KEEP_AS_IS']}，WORDING_UPDATE_ONLY={impact_counts['WORDING_UPDATE_ONLY']}，OWNER_AMENDMENT_REQUIRED={impact_counts['OWNER_AMENDMENT_REQUIRED']}，DEPRECATE_AND_REPLACE={impact_counts['DEPRECATE_AND_REPLACE']}，UNKNOWN={impact_counts['UNKNOWN']}。

必须保持：BDD Primary、HLC closure、TSB candidate 与负面限制、U-only encoder firewall、H strong challenger、no survivor selection、SESSION clustering、paired/unpaired claim boundary、Primary metric/operating point及统计失败规则。

必须 amendment：Q 的职责、Q20/Q12 的角色解释、Q→D 自动转换、E 对所有历史 use 的绝对排除、canonical role vocabulary、S2 capacity blocker 的未来语义。旧文件全部保持原样，通过新的 versioned amendment supersede。

## 248 SESSION reclassification

| 互斥主分类 | sessions | 未来用途解释 |
|---|---:|---|
| E0 metadata-only/reservation-only | {summary['classification_counts'].get('E0_METADATA_ONLY', 0)} | outcome 未暴露；reservation 仍需 Owner 处置 |
| E1 unrelated historical use | {summary['classification_counts'].get('E1_UNRELATED_HISTORICAL_USE', 0)} | 可透明标记后进入 Claim B 的 V 上界 |
| E2 benchmark engineering | {summary['classification_counts'].get('E2_BENCHMARK_ENGINEERING', 0)} | Claim B 可用；Claim A 不能称 unseen |
| E3 Primary method development | 0 observed | 若后续发现，必须从对应 confirmatory claim 排除 |
| E4 outcome-driven adaptation | 0 observed | 严格排除；当前 Primary 从未授权 |
| E5 untouched confirmatory | {summary['classification_counts'].get('E5_UNTOUCHED_CONFIRMATORY', 0)} | C 的 identity-level 候选 |
| UNKNOWN | 0 identity class | decision-use provenance 仍有文档局限 |

旧 strict audit 的 242 exposed / 243 conflict / 5 remaining 完整保留。新解释下，exposure-policy controlled-V 上界为 248，但**当前 certified V=0**；不能从规则过严跳到“248 全部可执行”。C 候选仍只有 5，且未做完整 applicability certification。

## Claim A 与 Claim B

Claim A（TSB generator generalizes to unseen sessions）需要排除至少 {summary['benchmark_engineering_sessions']} 个有明确 benchmark-engineering exposure 的 sessions，并进一步处理 reservation 与 method-development provenance。

Claim B（frozen controlled intervention 下 RBR-BDD vs H-BDD utility）允许 E0/E1/E2 进入 V，前提是 PRIMARY_FREEZE_POINT 在 outcome access 前完成、E3/E4 不进入对应 confirmatory claim、全 denominator 保留且 SESSION-aware inference 生效。这两种 claim 不等价。

## 19 个明确答案

1. **v2.3 是否改变论文核心科学问题？** 否；它改变数据角色与有效性边界。
2. **BDD Primary 是否改变？** 否。
3. **HLC scientific state 是否改变？** 否；仍 closed by scope，impossibility 未建立。
4. **TSB scientific state 是否改变？** 否；仍 frozen development candidate，未 scientific-qualified。
5. **RBR encoder U-only firewall 是否改变？** 否，必须严格保留。
6. **H strong challenger 是否改变？** 否；仍 development-informed，并须在 Primary 前冻结。
7. **No survivor selection 是否改变？** 否。
8. **SESSION-level clustering 是否改变？** 否；eligibility 与 dependence 是不同问题。
9. **historical exposure → unusable 是否过严？** 对 Claim B 是；它没有区分 outcome 的 decision use。
10. **哪些 exposure 构成 claim-relevant contamination？** E3 直接方法开发与 E4 outcome-driven adaptation；尤其是 encoder/H/BDD/kernel/calibration/operating point/cohort/sample-size 的结果反馈。
11. **248 中多少可进入未来 V？** exposure-policy 上界 248；当前 certified 数为 0，须先 amendment 和重新 preflight。
12. **多少只能用于 benchmark development？** 对 Claim B 没有证据要求任何 session 永久 B-only；对 Claim A，66 个 E2 不能充当 unseen。
13. **多少应排除 confirmatory C？** 当前 243 个非 E5 session 不属于 untouched C 候选；E3/E4 若后续发现也必须排除。
14. **多少可作 untouched C 候选？** 5 个 identity-level 候选；未构造 C，样本量未定义。
15. **Q→D→E 哪些保留/替换？** 保留训练隔离、calibration/evaluation 分离和 freeze 后不适配；用 U/B/V/C 替换 absolute-fresh Q、自动 Q→D 和全历史-use禁入 E。
16. **是否建议 U/B/V/C？** 是，Owner amendment 后采用。
17. **S2 capacity blocker 如何处理？** Amendment 前仍是有效 blocker；amendment 后其 capacity 部分降级为旧 strict-firewall historical result，工程 binding/reset/precontext blocker仍在，必须重跑 metadata-only preflight。
18. **是否需要 Owner-approved S1 amendment？** 是。
19. **amendment 前 simulation 是否仍为 0？** 是，且 training/evaluation/E access 同样保持未授权。

## 对 S2/S3/S4 的影响

- S2：amendment 前不启动；之后先重做 role-aware metadata preflight，分别给 B/V/C 计数，并继续修复独立的 execution/reset/precontext blockers。
- S3：RBR training 仍只由 U 授权；nuPlan V/C outcome 不得用于 encoder selection。
- S4：Primary 只能在 freeze point 与 V/C identity binding 后执行；V 提供 controlled utility，C 提供相对 untouched robustness，不能事后扩样或换方法。

## Zero-run proof

`SIMULATION=0; RUNNER_RUN=0; TSB_ROLLOUT=0; HLC_ROLLOUT=0; RBR_TRAINING=0; NEW_SCIENTIFIC_OUTCOME_EXPOSURE=0; EVALUATION_EXECUTION=0`。历史文件只提取 identity keys，不分析 scientific values。Protected CSV 未修改。

## Reproduction

在当前绑定输入保持不变时，对空目录运行 `python tools/s1r_protocol_correction_audit.py --output-dir <EMPTY_OUTPUT_DIR>`，应生成与本包逐字节一致的六份产物。代码检查使用 `python -m py_compile tools/s1r_protocol_correction_audit.py` 与 `python tools/check_no_tmp_dependencies.py`。
"""


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != "20260825_stageR_new":
        raise RuntimeError(f"unexpected branch: {branch}")
    base_head = git("rev-parse", "HEAD")

    from tools.s2_preflight_metadata_census import verify_s1

    verify_s1()
    session_rows, summary = audit_sessions()
    impacts = impact_rows()

    report = output / "S1R_Protocol_Correction_Impact_Audit_v1.md"
    matrix = output / "S1R_Frozen_Contract_Impact_Matrix_v1.csv"
    taxonomy = output / "S1R_Exposure_Taxonomy_v1.md"
    sessions = output / "S1R_Session_Exposure_Reclassification_v1.csv"
    roles = output / "S1R_Proposed_Data_Role_Model_v1.md"
    manifest = output / "S1R_Amendment_Manifest_v1.json"

    write_text(report, report_text(summary, impacts, base_head))
    write_csv(matrix, impacts, list(impacts[0]))
    write_text(taxonomy, taxonomy_text(summary))
    write_csv(sessions, session_rows, list(session_rows[0]))
    write_text(roles, role_text(summary))

    protected = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"
    frozen_s1 = sorted(S1.glob("S1_*"))
    artifacts = [report, matrix, taxonomy, sessions, roles]
    payload = {
        "schema_version": "S1R_Amendment_Manifest_v1",
        "status": "S1R_IMPACT_AUDIT_SUPPORTS_PROTOCOL_AMENDMENT",
        "document_status": "PROPOSED_ONLY",
        "OWNER_APPROVAL_REQUIRED": True,
        "S1_PROTOCOL_AMENDED": False,
        "S2_AUTHORIZED": False,
        "RBR_TRAINING_AUTHORIZED": False,
        "PRIMARY_EVALUATION_AUTHORIZED": False,
        "date": "2026-09-17",
        "branch": branch,
        "base_HEAD": base_head,
        "governing_input": {
            "requested_v2_3_path": str(V23.relative_to(ROOT)),
            "requested_v2_3_found": V23.exists(),
            "requested_v2_3_sha256": sha256_file(V23),
            "requested_v2_3_document_status": "RBR64_STAGE_R_ROADMAP_V2_3_CONTROLLED_VALIDATION_FREEZE_DRAFT",
            "conceptual_correction_task_path": str(TASK_INPUT),
            "conceptual_correction_task_sha256": sha256_file(TASK_INPUT),
            "v2_2_sha256": sha256_file(ROOT / "RBR-64_博士研究总体方案_v2.2_S1正式入口版.md"),
            "handover0917_sha256": sha256_file(ROOT / "handover0917.md"),
            "AGENTS_sha256": sha256_file(ROOT / "AGENTS.md"),
        },
        "session_reclassification": summary,
        "impact_class_counts": dict(Counter(row["impact_class"] for row in impacts)),
        "frozen_S1_inputs": {str(path.relative_to(ROOT)): sha256_file(path) for path in frozen_s1},
        "historical_interpretation": {
            "OLD_STRICT_FIREWALL_CAPACITY": "242 outcome-exposed sessions; 243 exposure/reservation conflicts; upper bound 5",
            "CLAIM_RELEVANT_EVALUATION_ELIGIBILITY": "exposure-policy upper bound 248; currently certified 0",
            "historical_files_rewritten": False,
        },
        "generator": {"path": "tools/s1r_protocol_correction_audit.py", "sha256": sha256_file(Path(__file__).resolve())},
        "artifacts": {f"docs/stageR/s1r_protocol_correction/{path.name}": sha256_file(path) for path in artifacts},
        "counters": {
            "SIMULATION": 0,
            "RUNNER_RUN": 0,
            "TSB_ROLLOUT": 0,
            "HLC_ROLLOUT": 0,
            "RBR_TRAINING": 0,
            "NEW_SCIENTIFIC_OUTCOME_EXPOSURE": 0,
            "EVALUATION_EXECUTION": 0,
        },
        "protected_csv": {
            "path": str(protected.relative_to(ROOT)),
            "expected_sha256": "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8",
            "sha256": sha256_file(protected),
            "unchanged": sha256_file(protected) == "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8",
        },
    }
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": payload["status"], "sessions": summary["classification_counts"], "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    run(parser.parse_args().output_dir)
