#!/usr/bin/env python3
"""Freeze Stage 6M aggregation rules and immutable Stage 6H inputs before new results."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


STATUS = "FROZEN_BEFORE_STAGE6M_AGGREGATED_RELIABILITY_RESULTS"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def file_record(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {path}: {observed} != {expected_sha256}")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": observed}


def run(args: argparse.Namespace) -> dict[str, Any]:
    design_path = args.design_json.resolve()
    design = read_json(design_path)
    if design.get("schema_version") != "stage6m_context_balanced_unpaired_bdd_design_v1":
        raise ValueError("unexpected Stage 6M design schema")
    if design.get("issue") != 254:
        raise ValueError("Stage 6M issue must remain #254")
    tasks = design.get("tasks", [])
    if sum(int(row["pool_count"]) for row in tasks) != 800:
        raise ValueError("frozen task counts must sum to 800")
    if abs(sum(float(row["weight"]) for row in tasks) - 1.0) > 1e-12:
        raise ValueError("frozen task weights must sum to one")
    methods = [row.get("id") for row in design.get("methods", [])]
    if methods != [
        "raw_marginal",
        "task_conditioned",
        "context_balanced",
        "task_context_balanced",
    ]:
        raise ValueError("Stage 6M method definitions or order changed")
    if design.get("validity", {}).get("universal_raw_bdd_threshold_forbidden") is not True:
        raise ValueError("universal threshold prohibition is missing")

    source_sha = design["source_sha256"]
    sources = {
        "stage6h_config": file_record(args.stage6h_config.resolve(), source_sha["stage6h_config"]),
        "embedding_pool_summary": file_record(
            args.embedding_pool_summary.resolve(), source_sha["embedding_pool_summary"]
        ),
        "embedding_pool_metadata": file_record(
            args.embedding_pool_metadata.resolve(), source_sha["embedding_pool_metadata"]
        ),
        "trial_bdd": file_record(args.trial_bdd.resolve(), source_sha["trial_bdd"]),
        "log_assignments": file_record(
            args.log_assignments.resolve(), source_sha["log_assignments"]
        ),
        "fixed_scope_bandwidths": file_record(
            args.fixed_scope_bandwidths.resolve(), source_sha["fixed_scope_bandwidths"]
        ),
    }
    pool = read_json(args.embedding_pool_summary.resolve())
    if pool.get("pair_count") != 800 or pool.get("cluster_count") != 489:
        raise ValueError("unexpected frozen Stage 6H pool size")
    trials = pd.read_csv(args.trial_bdd)
    expected_scopes = {"overall", *(row["name"] for row in tasks)}
    if len(trials) != 14400 or set(trials["scope"].astype(str)) != expected_scopes:
        raise ValueError("unexpected Stage 6H trial table shape or scopes")
    expected_sizes = set(int(value) for value in design["sample_sizes_per_release"])
    if set(trials["target_scenarios_per_release"].astype(int)) != expected_sizes:
        raise ValueError("Stage 6H trial sample sizes differ from Stage 6M freeze")

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output_dir exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    shutil.copy2(design_path, output_dir / design_path.name)
    manifest = {
        "schema_version": "stage6m_context_balanced_unpaired_bdd_freeze_v1",
        "status": STATUS,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "issue": 254,
        "aggregated_stage6m_results_read_before_freeze": False,
        "design": design,
        "design_input": {
            "path": str(design_path),
            "bytes": design_path.stat().st_size,
            "sha256": sha256_file(design_path),
        },
        "sources": sources,
        "input_audit": {
            "pair_count": 800,
            "cluster_count": 489,
            "trial_rows": 14400,
            "scopes": sorted(expected_scopes),
            "sample_sizes": sorted(expected_sizes),
        },
    }
    manifest_path = output_dir / "stage6m_freeze_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "stage6m_freeze_report_zh.md").write_text(
        "# Stage 6M 预分析冻结\n\n"
        f"- 状态：`{STATUS}`\n"
        "- Issue：`#254`\n"
        "- 冻结输入：Stage6H 800 pairs / 489 logs / 14400 scope-trial rows。\n"
        "- 四种方法、task权重、A/A阈值规则、支持度门禁均已在聚合结果生成前冻结。\n"
        "- 不修改Stage6H/6I、Stage6J/6K或既有embedding。\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": STATUS, "manifest": str(manifest_path)}, ensure_ascii=False))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design_json", type=Path, required=True)
    parser.add_argument("--stage6h_config", type=Path, required=True)
    parser.add_argument("--embedding_pool_summary", type=Path, required=True)
    parser.add_argument("--embedding_pool_metadata", type=Path, required=True)
    parser.add_argument("--trial_bdd", type=Path, required=True)
    parser.add_argument("--log_assignments", type=Path, required=True)
    parser.add_argument("--fixed_scope_bandwidths", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
