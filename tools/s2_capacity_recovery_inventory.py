#!/usr/bin/env python3
"""Build the Stage S2 capacity-recovery metadata-only discovery package.

This tool intentionally inspects path names, symlink targets, Git object names,
frozen identity ledgers, and already-produced S2-PRE summaries only. It never
opens a nuPlan database, trajectory payload, simulation result, or scientific
outcome artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/stageR/s2_capacity_recovery"
CACHE_ROOT = Path("/Users/liuqing/Projects/01_E2E_QA_Code/nuplan/dataset/data/cache")
DATASET_ROOT = Path("/Users/liuqing/Projects/01_E2E_QA_Code/nuplan/dataset/nuplan-v1.1")
DOWNLOAD_ROOT = Path("/Users/liuqing/Projects/01_E2E_QA_Code/nuplan_downloads")
S2_PREFLIGHT = ROOT / "docs/stageR/s2_preflight"
SESSION_RE = re.compile(r"^(\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}_veh-\d+)")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_bytes(payload.encode("utf-8"))


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def path_inventory(path: Path, glob: str = "*") -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for entry in sorted(path.glob(glob), key=lambda item: item.name):
        stat = entry.lstat()
        rows.append(
            {
                "name": entry.name,
                "kind": "symlink" if entry.is_symlink() else "file" if entry.is_file() else "directory",
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "target": os.readlink(entry) if entry.is_symlink() else None,
            }
        )
    return rows


def session_count(entries: list[dict[str, Any]]) -> int:
    sessions = set()
    for entry in entries:
        match = SESSION_RE.match(entry["name"])
        if match:
            sessions.add(match.group(1))
    return len(sessions)


def aliases_resolve_within(entries: list[dict[str, Any]], roots: tuple[Path, ...], alias_root: Path) -> bool:
    allowed = tuple(root.resolve() for root in roots)
    for entry in entries:
        if entry["kind"] != "symlink":
            return False
        resolved = (alias_root / entry["name"]).resolve()
        if not any(resolved.parent == root for root in allowed):
            return False
    return True


def source_row(
    source_id: str,
    source_kind: str,
    location: str,
    materialization: str,
    entries: list[dict[str, Any]],
    identity_status: str,
    provenance_status: str,
    independence_status: str,
    exposure_status: str,
    reservation_status: str,
    compatibility: str,
    decision: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "source_kind": source_kind,
        "location": location,
        "materialization": materialization,
        "file_or_alias_count": len(entries),
        "session_count_from_filename": session_count(entries),
        "identity_status": identity_status,
        "provenance_status": provenance_status,
        "independence_status": independence_status,
        "historical_exposure_status": exposure_status,
        "reservation_status": reservation_status,
        "scientific_compatibility": compatibility,
        "freshness_decision": decision,
        "level_a_raw_fresh_sessions": 0,
        "level_b_metadata_ready_sessions": 0,
        "level_c_prospective_upper_bound_sessions": 0,
        "evidence_sha256": canonical_sha256(entries),
        "notes": notes,
    }


def build_source_inventory() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mini = path_inventory(CACHE_ROOT / "mini", "*.db")
    train = path_inventory(CACHE_ROOT / "train_pittsburgh", "*.db")
    frozen = mini + train
    expanded = path_inventory(CACHE_ROOT / "locked_pool_expanded_v1", "*.db")
    locked = path_inventory(CACHE_ROOT / "locked_pool_v1", "*.db")
    non_pgh = path_inventory(CACHE_ROOT / "mini_non_pittsburgh_v1", "*.db")
    splits = path_inventory(DATASET_ROOT / "splits")
    downloads = path_inventory(DOWNLOAD_ROOT)
    lfs_lines = [line for line in git("lfs", "ls-files", "-a").splitlines() if line]
    ref_names = [line for line in git("for-each-ref", "--format=%(refname)", "refs/heads", "refs/remotes", "refs/tags").splitlines() if line]
    raw_names = sorted(
        {
            line
            for line in git("log", "--all", "--pretty=format:", "--name-only", "--", "*.db", "*.db.gz", "*.zip", "*.tar", "*.tar.gz", "*.tgz", "*.tfrecord*").splitlines()
            if line
        }
    )
    git_entries = [{"name": name, "kind": "reachable_git_lfs_path", "size": None, "mtime_ns": None, "target": None} for name in raw_names]
    volumes = path_inventory(Path("/Volumes"))
    output_dirs = [
        {"name": path.name, "kind": "outcome_directory", "size": None, "mtime_ns": None, "target": None}
        for path in sorted((ROOT / "outputs").iterdir(), key=lambda item: item.name)
        if path.is_dir()
    ]
    archive_suffixes = (".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".rar", ".db.gz")
    archive_entries = []
    workspace_root = ROOT.parent
    for walk_root, dirnames, filenames in os.walk(workspace_root):
        dirnames[:] = [name for name in dirnames if name not in {".git", "outputs", "__pycache__", ".pytest_cache"}]
        for filename in filenames:
            if filename.lower().endswith(archive_suffixes):
                path = Path(walk_root) / filename
                stat = path.stat()
                archive_entries.append(
                    {
                        "name": str(path.relative_to(workspace_root)),
                        "kind": "archive_file",
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "target": None,
                    }
                )
    archive_entries.sort(key=lambda row: row["name"])

    rows = [
        source_row(
            "SRC-FROZEN-NUPLAN-MINI-TRAIN-PITTSBURGH",
            "LOCAL_RAW_NUPLAN_DB_UNIVERSE",
            str(CACHE_ROOT),
            "MATERIALIZED",
            frozen,
            "BOUND_BY_S2_PREFLIGHT",
            "BOUND_BY_R1_SOURCE_UNIVERSE",
            "SESSION_GROUPING_BOUND",
            "242_OF_248_SESSIONS_DECLARED_EXPOSED;243_CONFLICTED",
            "CONFLICT_UNION_BOUND",
            "COMPATIBLE",
            "NOT_NEW_ALREADY_FROZEN_UNIVERSE",
            "The only materialized nuPlan DB payloads found; 1,624 paths collapse to 1,621 unique logs and 248 sessions in S2-PRE.",
        ),
        source_row(
            "SRC-ALIAS-LOCKED-POOL-EXPANDED",
            "SYMLINK_ALIAS_POOL",
            str(CACHE_ROOT / "locked_pool_expanded_v1"),
            "MATERIALIZED_AS_SYMLINKS",
            expanded,
            "EXACT_TARGET_PATHS_OBSERVED",
            "LOCAL_ALIAS_PROVENANCE_CLEAR",
            "SAME_AS_FROZEN_SOURCE_GROUPS",
            "INHERITS_TARGET_EXPOSURE",
            "INHERITS_TARGET_RESERVATIONS",
            "COMPATIBLE_BUT_DUPLICATE",
            "NOT_FRESH_ALIAS_OF_FROZEN_UNIVERSE",
            "Every alias target resolves into mini or train_pittsburgh; no additional raw identity exists.",
        ),
        source_row(
            "SRC-ALIAS-LOCKED-POOL-V1",
            "SYMLINK_ALIAS_POOL",
            str(CACHE_ROOT / "locked_pool_v1"),
            "MATERIALIZED_AS_SYMLINKS",
            locked,
            "EXACT_TARGET_PATHS_OBSERVED",
            "LOCAL_ALIAS_PROVENANCE_CLEAR",
            "SAME_AS_FROZEN_SOURCE_GROUPS",
            "INHERITS_TARGET_EXPOSURE",
            "INHERITS_TARGET_RESERVATIONS",
            "COMPATIBLE_BUT_DUPLICATE",
            "NOT_FRESH_ALIAS_OF_FROZEN_UNIVERSE",
            "All 64 aliases resolve into the existing frozen raw DB roots.",
        ),
        source_row(
            "SRC-ALIAS-MINI-NON-PITTSBURGH",
            "SYMLINK_ALIAS_POOL",
            str(CACHE_ROOT / "mini_non_pittsburgh_v1"),
            "MATERIALIZED_AS_SYMLINKS",
            non_pgh,
            "EXACT_TARGET_PATHS_OBSERVED",
            "LOCAL_ALIAS_PROVENANCE_CLEAR",
            "SAME_AS_FROZEN_SOURCE_GROUPS",
            "INHERITS_TARGET_EXPOSURE",
            "INHERITS_TARGET_RESERVATIONS",
            "COMPATIBLE_BUT_DUPLICATE",
            "NOT_FRESH_ALIAS_OF_FROZEN_UNIVERSE",
            "All 61 aliases resolve into mini; the location label does not create a new independence group.",
        ),
        source_row(
            "SRC-UNMATERIALIZED-NUPLAN-SPLITS",
            "LOCAL_SPLIT_DEFINITION_DIRECTORY",
            str(DATASET_ROOT / "splits"),
            "EMPTY_DIRECTORY",
            splits,
            "NO_IDENTITIES_PRESENT",
            "PATH_PROVENANCE_CLEAR_PAYLOAD_ABSENT",
            "NOT_ASSESSABLE",
            "UNKNOWN",
            "UNKNOWN",
            "POTENTIALLY_COMPATIBLE_IF_MATERIALIZED",
            "NO_SOURCE_PAYLOAD",
            "The directory exists but contains no visible split files or database payloads.",
        ),
        source_row(
            "SRC-UNMATERIALIZED-DOWNLOAD-STAGING",
            "LOCAL_DOWNLOAD_STAGING",
            str(DOWNLOAD_ROOT),
            "NO_DATA_PAYLOAD",
            downloads,
            "NO_IDENTITIES_PRESENT",
            "ONLY_EMPTY_DOWNLOAD_LOG_OBSERVED",
            "NOT_ASSESSABLE",
            "UNKNOWN",
            "UNKNOWN",
            "POTENTIALLY_COMPATIBLE_IF_MATERIALIZED",
            "NO_SOURCE_PAYLOAD",
            "Only an empty curl log and filesystem metadata were observed; no download was initiated.",
        ),
        source_row(
            "SRC-GIT-REACHABLE-RAW-OBJECTS",
            "GIT_REFS_AND_LFS",
            "refs/heads + refs/remotes + refs/tags",
            "ONE_WAYMO_LFS_POINTER",
            git_entries,
            "REACHABLE_PATH_NAMES_ENUMERATED",
            "GIT_OBJECT_PROVENANCE_CLEAR",
            "NOT_NUPLAN_SESSION_IDENTITIES",
            "HISTORICALLY_USED_WAYMO_TRAINING_SOURCE",
            "NOT_A_Q_RESERVOIR",
            "INCOMPATIBLE_WITH_FROZEN_NUPLAN_Q_REPLAY_CONTRACT",
            "NOT_ELIGIBLE_SOURCE_TYPE",
            f"Enumerated {len(ref_names)} named refs without checkout; reachable raw paths contain the existing Waymo TFRecord pointer and no nuPlan DB/archive.",
        ),
        source_row(
            "SRC-LOCAL-RAW-ARCHIVES",
            "WORKSPACE_ARCHIVE_DISCOVERY",
            str(workspace_root),
            "NO_MATCHING_ARCHIVES",
            archive_entries,
            "NO_IDENTITIES_PRESENT",
            "PATH_SEARCH_COMPLETE_FOR_LISTED_SUFFIXES",
            "NOT_ASSESSABLE",
            "UNKNOWN",
            "UNKNOWN",
            "POTENTIALLY_COMPATIBLE_IF_PRESENT",
            "NO_SOURCE_PAYLOAD",
            "Searched outside result trees for zip/tar/tar.gz/tgz/7z/rar/db.gz payloads; none were found.",
        ),
        source_row(
            "SRC-LOCAL-OUTCOME-TREES",
            "UNTRACKED_AND_TRACKED_OUTPUT_DIRECTORIES",
            str(ROOT / "outputs"),
            "MATERIALIZED_RESULTS",
            output_dirs,
            "NOT_INSPECTED_BEYOND_DIRECTORY_NAMES",
            "OUTPUT_PROVENANCE",
            "NOT_RAW_INDEPENDENT_SOURCE",
            "OUTCOME_FIREWALL_ACTIVE",
            "NOT_A_SOURCE_RESERVOIR",
            "INELIGIBLE_DERIVED_RESULTS",
            "BLOCKED_BY_OUTCOME_FIREWALL",
            "Contents were not opened for discovery because these are historical scientific/engineering result trees, not fresh raw sessions.",
        ),
        source_row(
            "SRC-MOUNTED-EXTERNAL",
            "VISIBLE_VOLUME_ROOTS",
            "/Volumes",
            "SYSTEM_VOLUME_ONLY",
            volumes,
            "NO_EXTERNAL_DATASET_IDENTITY_VISIBLE",
            "VISIBLE_MOUNT_METADATA_ONLY",
            "NOT_ASSESSABLE",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "NO_EXTERNAL_SOURCE_VISIBLE",
            "Only the local Macintosh HD system volume is visible; no external dataset mount was found.",
        ),
    ]
    evidence = {
        "named_git_ref_count": len(ref_names),
        "reachable_raw_path_names": raw_names,
        "git_lfs_entries": lfs_lines,
        "visible_volumes": [row["name"] for row in volumes],
        "local_raw_archive_count": len(archive_entries),
        "frozen_db_path_count": len(frozen),
        "frozen_session_name_count": session_count(frozen),
        "alias_counts": {
            "locked_pool_expanded_v1": len(expanded),
            "locked_pool_v1": len(locked),
            "mini_non_pittsburgh_v1": len(non_pgh),
        },
        "alias_targets_all_within_frozen_roots": {
            "locked_pool_expanded_v1": aliases_resolve_within(expanded, (CACHE_ROOT / "mini", CACHE_ROOT / "train_pittsburgh"), CACHE_ROOT / "locked_pool_expanded_v1"),
            "locked_pool_v1": aliases_resolve_within(locked, (CACHE_ROOT / "mini", CACHE_ROOT / "train_pittsburgh"), CACHE_ROOT / "locked_pool_v1"),
            "mini_non_pittsburgh_v1": aliases_resolve_within(non_pgh, (CACHE_ROOT / "mini", CACHE_ROOT / "train_pittsburgh"), CACHE_ROOT / "mini_non_pittsburgh_v1"),
        },
    }
    if not all(evidence["alias_targets_all_within_frozen_roots"].values()):
        raise RuntimeError("alias pool contains a target outside frozen mini/train_pittsburgh roots")
    return rows, evidence


def build_historical_registry(exposure: dict[str, Any]) -> list[dict[str, Any]]:
    historical = set(exposure["historical_sessions"])
    conflicts = set(exposure["conflict_sessions"])
    reserved = {
        match.group(1)
        for log_name in exposure["reserved_logs"]
        if (match := SESSION_RE.match(log_name))
    }
    permanent = {
        match.group(1)
        for log_name in exposure["permanent_logs"]
        if (match := SESSION_RE.match(log_name))
    }
    evidence_sources = (
        "docs/stageR/s2_preflight/S2_Preflight_Exposure_Exclusion_Ledger_v1.json;"
        "docs/stageR/r0/manifests/r0_nuplan_historical_use_ledger_v0.1.csv"
    )
    rows = []
    for session in sorted(conflicts):
        exposed = session in historical
        reservation_conflict = session in reserved or session in permanent
        rows.append(
            {
                "session_id": session,
                "declared_outcome_exposed": str(exposed).lower(),
                "reservation_or_permanent_conflict": str(reservation_conflict).lower(),
                "exposure_stage": "MULTI_STAGE_FROZEN_AGGREGATE" if exposed else "R1_R2_OR_A5_RESERVATION_AGGREGATE",
                "exposure_type": (
                    "DECLARED_HISTORICAL_OUTCOME_EXPOSURE;RESERVATION_OR_PERMANENT_CONFLICT"
                    if exposed and reservation_conflict
                    else "DECLARED_HISTORICAL_OUTCOME_EXPOSURE"
                    if exposed
                    else "RESERVATION_OR_PERMANENT_CONFLICT_ONLY"
                ),
                "development_or_evaluation_use": "DECLARED_IN_FROZEN_AGGREGATE" if exposed else "NOT_ESTABLISHED",
                "training_use": "UNKNOWN_NOT_SEPARATELY_MATERIALIZED",
                "confidence": "HIGH_FOR_AGGREGATE_CLASSIFICATION;STAGE_SPECIFICITY_NOT_MATERIALIZED",
                "evidence_sources": evidence_sources,
                "notes": "Derived from frozen identity sets only; no historical result payload was reopened.",
            }
        )
    return rows


def build_report(base_head: str, source_rows: list[dict[str, Any]], evidence: dict[str, Any], census: dict[str, Any], exposure: dict[str, Any]) -> str:
    conflict_sessions = set(exposure["conflict_sessions"])
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in census["logs"]:
        groups[row["session_id"]].append(row)
    current_remaining = sorted(set(groups) - conflict_sessions)
    remaining_logs = sum(len(groups[session]) for session in current_remaining)
    remaining_tokens = sum(row["scenario_count"] for session in current_remaining for row in groups[session])
    return f"""# Stage S2 Capacity Recovery — Fresh Independent Source Discovery

日期：2026-09-17（Asia/Shanghai）

分支：`20260825_stageR_new`

审计基线 HEAD：`{base_head}`

最终容量状态：**CAPACITY_RECOVERY_NONE**

S2 状态：**S2_PREFLIGHT_BLOCKED_Q20_CAPACITY**

## 结论

本轮完成了 metadata-only、zero-simulation 的独立来源发现。没有发现冻结 `mini/train_pittsburgh` 全集之外、已物化且能够证明来源与独立性的 nuPlan 原始会话。三个看似额外的本地池均为符号链接子集；Git 可达历史只有现有 Waymo LFS 指针，没有 nuPlan DB/归档；下载暂存区没有数据；可见挂载点没有外部数据集。历史 `outputs/` 仅按目录名登记并受 outcome firewall 保护，未打开其中的科学结果。

因此新增容量三层均为 **0**：Level A raw fresh sessions = 0；Level B metadata-ready sessions = 0；Level C prospectively eligible upper bound sessions = 0。既有冻结全集在暴露/保留冲突后仍只有 **{len(current_remaining)} sessions / {remaining_logs} logs / {remaining_tokens:,} tokens** 的资格上界，且其正式初始速度、official native route/reference、forward support 与 mapping 仍未完整认证。合并新增来源后，Q20 上界仍为 5，Q12 上界仍为 5，二者均不可用。

`CAPACITY_RECOVERY_NONE` 表示本轮没有找到可提出 amendment 的新池。它不把 UNKNOWN 写成失败，也不声称现有 5 个会话科学合格。没有生成 Q/E roster、替补顺序或 `PROPOSED_SOURCE_UNIVERSE_AMENDMENT`。

## 搜索覆盖与证据

| 范围 | 结果 |
|---|---|
| 本地 nuPlan raw cache | 1,624 DB 路径；沿用 S2-PRE 去重结果 1,621 logs / 248 sessions；全部属于当前冻结全集 |
| 本地 alias pools | `locked_pool_expanded_v1`={evidence['alias_counts']['locked_pool_expanded_v1']}、`locked_pool_v1`={evidence['alias_counts']['locked_pool_v1']}、`mini_non_pittsburgh_v1`={evidence['alias_counts']['mini_non_pittsburgh_v1']}；全部指向冻结 raw cache |
| 未物化 split/download 路径 | split 目录无 payload；download staging 仅有空日志，未发起下载 |
| Git refs / LFS | {evidence['named_git_ref_count']} 个 named refs；可达 raw 路径只有现有 Waymo TFRecord，非冻结 nuPlan Q replay 来源 |
| 本地归档 | 排除 result trees 后检索 zip/tar/tar.gz/tgz/7z/rar/db.gz；发现 {evidence['local_raw_archive_count']} 个候选 payload |
| 外接/挂载路径 | 可见卷仅 `{'`, `'.join(evidence['visible_volumes']) or 'NONE'}`；无外部数据集挂载 |
| 历史输出树 | 仅枚举目录名；内容 `BLOCKED_BY_OUTCOME_FIREWALL`，且派生结果不构成 fresh raw source |

完整逐来源证据见 `S2_Fresh_Source_Inventory_v1.csv`。目录证据 SHA 绑定路径名、类型、大小、mtime 与符号链接目标；它不是 DB 内容哈希，也没有读取轨迹。Git 检索覆盖本地已存在的 heads、remotes 与 tags，不声称覆盖未获取的远端对象或当前不可见的物理存储。

## 三层容量与独立性

| 容量层 | 新来源 | 合并既有冻结剩余池 | 判定 |
|---|---:|---:|---|
| A：raw fresh sessions | 0 | 既有 5 不属于新来源 | 无容量恢复 |
| B：metadata-ready | 0 | 0 已完整认证；5 仍有 route/reference/forward/mapping UNKNOWN | 不可进入 roster |
| C：prospectively eligible upper bound | 0 | ≤5 | Q20/Q12 均不足 |

独立单位保持 **SESSION**。解析器继续使用 `YYYY.MM.DD.hh.mm.ss_veh-XX` 采集前缀；没有引入新命名格式，也没有把分段 LOG 重定义成独立单位。车辆 ID 不等同于驾驶员 ID，跨会话更高层依赖仍可能进一步降低容量。

`S2_Fresh_Session_Registry_v1.csv` 只有表头，因为没有任何来源通过 Level A；空表是零发现的显式结果，不是漏写。历史冲突表列出 243 个冻结 conflict sessions。其 stage 字段保守记为 frozen aggregate，因为现有冻结账本不提供逐 source-to-session stage join；为避免重开历史科学结果，本轮不猜测更细 stage。聚合暴露/保留分类本身为高置信度。

容量情景中的新来源 `N=0`：预留 Q20 后 `max(N-20, 0)=0`，容量缺口为 20；预留 Q12 后 `max(N-12, 0)=0`，容量缺口为 12。由于两个 Q 情景都无法组成，0 只表示没有剩余 potential future E reserve，不是已定义的 E sample size。

## 15 个明确答案

| # | 问题 | 答案 |
|---:|---|---|
| 1 | 当前 frozen universe 外是否发现新的 source reservoir？ | **否。** 本地候选均为冻结源别名、空目录、非 nuPlan 原始数据或结果树。 |
| 2 | 一共发现多少新的 raw candidate sessions？ | **0。** |
| 3 | 其中多少 session 可以证明未历史 exposure？ | **0。** 没有新的 session identity 可进入 freshness 证明。 |
| 4 | 多少 session freshness 为 UNKNOWN？ | **0 个已识别 session。** 若干未物化/不可见 source 的 provenance 为 UNKNOWN，但它们没有可计数身份，不能换算成 session。 |
| 5 | 多少 session metadata 足够支持 prospective filtering？ | **0。** Level B metadata-ready pool 为空。 |
| 6 | 按冻结规则 eligible upper bound 是多少？ | 新来源 Level C 为 **0**；与既有冻结剩余池合并仍为 **≤5 sessions**。`prospectively eligible != scientifically qualified TSB pair`。 |
| 7 | 是否存在 Q12 capacity？ | **否。** 合并上界 5 < 12；`Q12_CAPACITY_NOT_AVAILABLE`。 |
| 8 | 是否存在 Q20 capacity？ | **否。** 合并上界 5 < 20；`Q20_CAPACITY_NOT_AVAILABLE`。 |
| 9 | 如果预留 20 个 session 给 Q，未来还剩多少 fresh SESSION？ | 新来源 `N=0`，所以可剩 **0**，同时存在 20 个 session 的 Q 容量缺口；Q20 实际不可预留。 |
| 10 | 是否存在显著 potential future E reserve？ | **否。** Q20 本身不可组成；E 数量尚未定义、E 未构造且未访问。 |
| 11 | 是否需要 source universe amendment？ | **本轮不需要。** 没有找到可提案的新池。 |
| 12 | amendment 是否仅为 proposal？ | 本轮没有生成 amendment；若未来发现新池，只能 `PROPOSED_ONLY` 且必须 Owner 批准。 |
| 13 | 有哪些 UNKNOWN / AMBIGUOUS / BLOCKED provenance？ | 未物化 split/download、未获取远端与当前不可见存储为 UNKNOWN；outputs 为 `BLOCKED_BY_OUTCOME_FIREWALL`；没有把这些来源计成 session。 |
| 14 | 是否读取了任何 scientific outcome？ | **否。** 只读取冻结身份账本与路径元数据；历史结果 payload 未打开。 |
| 15 | simulation / runner / training / E access 是否全部保持 0？ | **是。** 六项硬计数全部为 0。 |

## 硬边界与计数

```text
SIMULATION_RUNS=0
RUNNER_RUN_CALLS=0
NEW_SCIENTIFIC_OUTCOME_EXPOSURE=0
RBR_TRAINING=0
E_CONSTRUCTION=0
E_ACCESS=0
```

本轮没有执行 S2、Q20/Q12、reset/precontext 修复或 RBR；也没有修改 S1/S1.1 冻结文件。`handover0917.md` 现已由用户提供并作为本轮权威入口保存；它不会追溯改写既有 S2-PRE manifest 中当时“未找到该文件”的历史记录。

Protected CSV 保持 SHA256=`e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8`。最终唯一状态为 `CAPACITY_RECOVERY_NONE`，并单独保持 `S2_STATUS = S2_PREFLIGHT_BLOCKED_Q20_CAPACITY`。
"""


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")

    base_head = git("rev-parse", "HEAD")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != "20260825_stageR_new":
        raise RuntimeError(f"unexpected branch: {branch}")

    census_path = S2_PREFLIGHT / "S2_Preflight_Eligibility_Census_v1.json"
    exposure_path = S2_PREFLIGHT / "S2_Preflight_Exposure_Exclusion_Ledger_v1.json"
    census = read_json(census_path)
    exposure = read_json(exposure_path)
    source_rows, evidence = build_source_inventory()
    historical_rows = build_historical_registry(exposure)

    source_path = output / "S2_Fresh_Source_Inventory_v1.csv"
    session_path = output / "S2_Fresh_Session_Registry_v1.csv"
    historical_path = output / "S2_Historical_Exposure_Registry_v1.csv"
    report_path = output / "S2_Capacity_Recovery_Report_v1.md"
    manifest_path = output / "S2_Capacity_Recovery_Manifest_v1.json"

    write_csv(source_path, list(source_rows[0]), source_rows)
    write_csv(
        session_path,
        [
            "session_id", "source_id", "log_count", "raw_file_count", "identity_evidence",
            "outcome_exposure", "development_use", "evaluation_use", "reservation_conflict",
            "independence_group", "provenance_status", "level_a_raw_fresh", "level_b_metadata_ready",
            "speed_status", "route_status", "reference_status", "forward_support_status", "mapping_status",
            "level_c_prospective_upper_bound", "reason",
        ],
        [],
    )
    write_csv(historical_path, list(historical_rows[0]), historical_rows)
    report_path.write_text(build_report(base_head, source_rows, evidence, census, exposure), encoding="utf-8")

    protected = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"
    handover = ROOT / "handover0917.md"
    artifacts = [source_path, session_path, historical_path, report_path]
    manifest = {
        "schema_version": "S2_Capacity_Recovery_Manifest_v1",
        "date": "2026-09-17",
        "branch": branch,
        "base_HEAD": base_head,
        "authoritative_handover": {"path": "handover0917.md", "sha256": sha256_file(handover)},
        "generator": {"path": "tools/s2_capacity_recovery_inventory.py", "sha256": sha256_file(Path(__file__).resolve())},
        "final_capacity_state": "CAPACITY_RECOVERY_NONE",
        "s2_status": "S2_PREFLIGHT_BLOCKED_Q20_CAPACITY",
        "independence_unit": "SESSION",
        "capacity": {
            "fresh_level_a_raw_sessions": 0,
            "fresh_level_b_metadata_ready_sessions": 0,
            "fresh_level_c_prospective_upper_bound_sessions": 0,
            "existing_frozen_remaining_session_upper_bound": census["denominator_summary"]["remaining_session_upper_bound"],
            "combined_q_session_upper_bound": census["denominator_summary"]["remaining_session_upper_bound"],
            "Q20": "NOT_AVAILABLE",
            "post_Q20_future_E_reservoir": "NOT_AVAILABLE_AND_E_NOT_DEFINED",
            "Q12": "NOT_AVAILABLE",
        },
        "source_discovery": {
            **evidence,
            "source_inventory_rows": len(source_rows),
            "fresh_session_registry_rows": 0,
            "historical_exposure_registry_rows": len(historical_rows),
            "proposed_source_universe_amendment": "NOT_CREATED_NO_NEW_FRESH_POOL",
            "remote_network_downloads": 0,
            "outcome_payloads_opened": 0,
        },
        "frozen_inputs": {
            str(census_path.relative_to(ROOT)): sha256_file(census_path),
            str(exposure_path.relative_to(ROOT)): sha256_file(exposure_path),
            "docs/stageR/r0/manifests/r0_nuplan_historical_use_ledger_v0.1.csv": sha256_file(ROOT / "docs/stageR/r0/manifests/r0_nuplan_historical_use_ledger_v0.1.csv"),
            "docs/stageR/r1/r1_fresh_smoke_source_universe_v0.1.json": sha256_file(ROOT / "docs/stageR/r1/r1_fresh_smoke_source_universe_v0.1.json"),
        },
        "artifacts": {f"docs/stageR/s2_capacity_recovery/{path.name}": sha256_file(path) for path in artifacts},
        "counters": {
            "SIMULATION_RUNS": 0,
            "RUNNER_RUN_CALLS": 0,
            "NEW_SCIENTIFIC_OUTCOME_EXPOSURE": 0,
            "RBR_TRAINING": 0,
            "E_CONSTRUCTION": 0,
            "E_ACCESS": 0,
        },
        "protected_csv": {"path": str(protected.relative_to(ROOT)), "sha256": sha256_file(protected), "expected_sha256": "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8", "unchanged": sha256_file(protected) == "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8"},
        "prohibitions_observed": {
            "Q_or_E_roster_created": False,
            "reset_or_precontext_repair_performed": False,
            "S1_frozen_files_modified": False,
            "source_universe_amended": False,
        },
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")

    print(json.dumps({"output": str(output), "artifacts": len(artifacts) + 1, "state": manifest["final_capacity_state"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args().output_dir)
