#!/usr/bin/env python3
"""Create the Chinese B2.1 scientific report from small versioned result tables.

This finalizer neither launches nuPlan nor reads raw official trace, Parquet,
database, representation, BDD, probes, checkpoints, or RBR artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_official_metric_canonicalizer import sha256_file


ROOT = Path(__file__).resolve().parents[1]
R1_DIR = ROOT / "docs/stageR/r1"
MANIFEST = R1_DIR / "r1_official_technical_smoke_execution_manifest_v1.1.json"
FAMILY = R1_DIR / "r1_official_technical_smoke_family_summary_v1.1.csv"
PAIR = R1_DIR / "r1_official_technical_smoke_pair_metrics_v1.1.csv"
STATUS_CORRECTION = R1_DIR / "r1_official_technical_smoke_b2_status_correction_v1.0.json"
REPORT = R1_DIR / "R1_Official_Compliant_Technical_Smoke_Report_v1.1.md"
READINESS = R1_DIR / "R1_Development_Roster_Freeze_Readiness_v0.8.md"


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite versioned final report: {path}")
    path.write_text(text, encoding="utf-8")


def _decision(manifest: Dict[str, Any], family_rows: List[Dict[str, str]]) -> str:
    if manifest["status"] != "COMPLETE":
        return "NOT_EVALUABLE_DUE_TO_B2_1_TECHNICAL_FAILURE"
    if all(row["readiness"] == "READY_FOR_FORMAL_DEVELOPMENT_ROSTER_REVIEW" for row in family_rows):
        return "READY_FOR_DEVELOPMENT_ROSTER_FREEZE_REVIEW"
    return "BENCHMARK_FAMILY_NOT_READY"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--family-summary", type=Path, default=FAMILY)
    parser.add_argument("--pair-metrics", type=Path, default=PAIR)
    parser.add_argument("--historical-status-correction", type=Path, default=STATUS_CORRECTION)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--readiness", type=Path, default=READINESS)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    correction = json.loads(args.historical_status_correction.read_text(encoding="utf-8"))
    family_rows, pairs = _read_csv(args.family_summary), _read_csv(args.pair_metrics)
    if len(family_rows) != 2 or {row["family"] for row in family_rows} != {"R-HLC", "R-TSB"}:
        raise ValueError("B2.1 family summary must contain exactly R-HLC and R-TSB")
    decision = _decision(manifest, family_rows)
    pair_counts = {family: sum(row["family"] == family for row in pairs) for family in ("R-HLC", "R-TSB")}
    rows = "\n".join(f"| {row['family']} | {row['completed_pairs']}/12 | {row['readiness']} | {row['reason']} |" for row in family_rows)
    readiness_rows = "".join(
        f"| {row['family']} | {row['completed_pairs']}/12 | {row['readiness']} |\n" for row in family_rows
    )
    report = f"""# R1 官方合规技术 Smoke 报告 v1.1\n\n## 结论\n\nB2.1 仅修正了 V3 已验证运行时的环境装配：以完整显式 roots 调用 `stage7c_environment(args)`。它没有修改 roster、selector、planner、生成器、context、mechanism、F-match、endpoint 或 safety 的冻结定义。\n\nB2.1 执行状态为 `{manifest['status']}`；新 official closed-loop run 为 `{manifest['actual_official_run_count']}/48`，技术失败数为 `{manifest['technical_failure_count']}`，形成 pair `{manifest['pair_result_count']}/24`。跨 family 科学决定为：`R1_RESIDUAL_BENCHMARK_ENABLEMENT = {decision}`。\n\n## 历史 B2 与 B2.1 的边界\n\n原 B2 仅有 1 次 pre-simulation technical claim，官方 simulator 启动数与实际 closed-loop run 均为 0；其 mechanism、F-match、endpoint 与 safety 均为 `NOT_EVALUABLE`。修正后的历史状态仍为 `{correction['R1_RESIDUAL_BENCHMARK_ENABLEMENT']}`，该历史 claim 标记为 `HISTORICAL_B2_PRE_SIMULATION_TECHNICAL_CLAIM / SIMULATOR_NOT_STARTED / NOT_PART_OF_B2_1_EVIDENCE`，未覆盖或删除。\n\n## Family 结果\n\n| family | 完成 pair | 状态 | 原因 |\n|---|---:|---|---|\n{rows}\n\nR-HLC pair 数：`{pair_counts['R-HLC']}/12`；R-TSB pair 数：`{pair_counts['R-TSB']}/12`。所有 gate 均按冻结规则逐 pair 记录；没有以多数、比例或事后阈值替代 12/12 要求。\n\n## 治理结论\n\n`SCIENTIFIC_PROTOCOL_DEVIATION = NO`。本次为 `EXECUTION_ENVIRONMENT_BINDING_CORRECTION`，不构成科学 protocol amendment。无论本轮结果如何，formal development rollout 与 `RBR_A/B/C` 训练仍为 `NOT_AUTHORIZED`；若两 family 均 ready，下一步也仅是 formal R1 development roster freeze review。\n\n## 可审计性\n\n- manifest SHA256：`{sha256_file(args.manifest)}`\n- family summary SHA256：`{sha256_file(args.family_summary)}`\n- pair metrics SHA256：`{sha256_file(args.pair_metrics)}`\n"""
    readiness = f"""# R1 Development Roster Freeze 就绪性 v0.8\n\n## 当前状态：`{decision}`\n\n本文件先纠正历史 B2 的语义，再记录 B2.1 的唯一恢复批次。历史 B2 的结论为 `{correction['R1_RESIDUAL_BENCHMARK_ENABLEMENT']}`，恢复动作是 `{correction['RECOVERY_ACTION']}`；这不是 generator 或 eligibility 失败。\n\n| family | B2.1 completed pairs | readiness |\n|---|---:|---|\n{readiness_rows}\nB2.1 official run：`{manifest['actual_official_run_count']}/48`；pair：`{manifest['pair_result_count']}/24`；技术失败：`{manifest['technical_failure_count']}`。\n\n只有 R-HLC 与 R-TSB 同时为 `READY_FOR_FORMAL_DEVELOPMENT_ROSTER_REVIEW` 才会将 residual benchmark enablement 标为 `READY_FOR_DEVELOPMENT_ROSTER_FREEZE_REVIEW`。本文件不授权 development rollout、RBR-A/B/C training 或任何新的 planner rollout。\n"""
    _write_new(args.report, report)
    _write_new(args.readiness, readiness)
    print(json.dumps({"status": decision, "report": str(args.report), "readiness": str(args.readiness)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
