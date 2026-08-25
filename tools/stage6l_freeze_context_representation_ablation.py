#!/usr/bin/env python3
"""Freeze Stage 6L representation-ablation inputs before new results are read."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage7_m6_4b_run_locked_rollouts import sha256_file


STATUS = "FROZEN_BEFORE_STAGE6L_REPRESENTATION_ABLATION"
EXPECTED_CHECKPOINT_SHA256 = "909022f5df03a3f01c2149da6c9b44c613e955a4d816e8ec4d5862f39f8bf0cc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design_json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage6j_context_dir", type=Path, required=True)
    parser.add_argument("--stage6j_embedding_dir", type=Path, required=True)
    parser.add_argument("--stage6k_contexts_dir", type=Path, required=True)
    parser.add_argument("--stage6k_embeddings_dir", type=Path, required=True)
    parser.add_argument("--stage6j_bdd_config", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path.resolve())}


def context_record(path: Path) -> dict[str, Any]:
    required = [
        "context_traj.npy", "ego_seq.npy", "ego_seq_mask.npy", "neighbor_seq.npy", "interaction_feat_style.npy",
        "metadata.csv", "feature_schema.json", "stage5d_context_schema.json",
        "nuplan_lane_assignment_by_row.csv", "warnings.json",
    ]
    record = {name: file_record(path / name) for name in required}
    neighbor = np.load(path / "neighbor_seq.npy", mmap_mode="r")
    if neighbor.shape != (366, 5, 150, 15):
        raise ValueError(f"Unexpected neighbor_seq shape in {path}: {neighbor.shape}")
    valid_rate = float(np.mean(neighbor[:, :, :, 0] > 0.5))
    if valid_rate <= 0.0:
        raise ValueError(f"Formal Stage 6L freeze rejects zero semantic-neighbor coverage: {path}")
    warnings = read_json(path / "warnings.json")
    validation = warnings.get("validation", {})
    if validation.get("pass") is not True:
        raise ValueError(f"Context validation is not PASS: {path}")
    record["coverage_audit"] = {
        "semantic_neighbor_valid_frame_rate": valid_rate,
        "nonzero": True,
        "context_validation_pass": True,
        "required_nonzero_neighbor_coverage_flag": validation.get("require_nonzero_neighbor_coverage"),
        "required_nonzero_neighbor_coverage_pass": validation.get("required_nonzero_neighbor_coverage_pass"),
    }
    return record


def embedding_record(path: Path) -> dict[str, Any]:
    return {name: file_record(path / name) for name in ["embedding.npy", "metadata.csv", "embedding_manifest.json"]}


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output_dir exists: {output_dir}; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    design = read_json(args.design_json.resolve())
    if design.get("schema_version") != "stage6l_context_representation_ablation_design_v1":
        raise ValueError("Unexpected Stage 6L design schema")
    checkpoint = file_record(args.checkpoint.resolve())
    if checkpoint["sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Frozen checkpoint SHA-256 differs")
    if design.get("frozen_checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Design checkpoint SHA-256 differs")
    representation_ids = [row.get("id") for row in design.get("representations", [])]
    if representation_ids != [
        "learned64_full_context", "learned64_neighbor_zero_input", "ego_kinematic_13d",
        "handcrafted_interaction_trajectory_46d",
    ]:
        raise ValueError("Frozen Stage 6L representation order/definitions changed")
    stats = design.get("statistics", {})
    if stats.get("paired_permutations") != 100000 or stats.get("log_cluster_permutations") != 100000:
        raise ValueError("Stage 6L randomization resolution changed")
    if stats.get("cross_representation_raw_mmd_comparison_forbidden") is not True:
        raise ValueError("Cross-representation raw MMD prohibition is missing")

    contexts: dict[str, Any] = {"dose100": context_record(args.stage6j_context_dir.resolve())}
    embeddings: dict[str, Any] = {"dose100": embedding_record(args.stage6j_embedding_dir.resolve())}
    for label in ["dose25", "dose50", "dose75"]:
        contexts[label] = context_record(args.stage6k_contexts_dir.resolve() / label)
        embeddings[label] = embedding_record(args.stage6k_embeddings_dir.resolve() / label)

    manifest = {
        "schema_version": "stage6l_context_representation_ablation_freeze_v1",
        "status": STATUS,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "issue": 253,
        "new_stage6l_representation_or_bdd_read": False,
        "design": design,
        "design_input": file_record(args.design_json.resolve()),
        "checkpoint": checkpoint,
        "stage6j_bdd_config": file_record(args.stage6j_bdd_config.resolve()),
        "context_inputs": contexts,
        "embedding_inputs": embeddings,
        "old_stage6j_k_outputs_are_read_only": True,
        "lane_pipeline_changed": False,
        "retraining_started": False,
        "tool_sha256": sha256_file(Path(__file__).resolve()),
    }
    manifest_path = output_dir / "stage6l_context_representation_ablation_freeze_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    report = [
        "# Stage 6L context-quality 表示消融冻结报告", "", "## 状态", "", f"`{STATUS}`", "",
        "- Issue: `#253`", "- 新消融表示或BDD结果已读取: `false`", "- lane pipeline已修改: `false`",
        "- 重训练已启动: `false`", "- Stage 6J/K历史输出: `只读`", "",
        "## 冻结表示", "",
    ]
    for row in design["representations"]:
        report.append(f"- `{row['id']}`：{row['role']}")
    report += [
        "", "## 关键解释边界", "",
        "- neighbor-zero是同checkpoint输入消融，不是独立训练的ego-only模型。",
        "- 每种表示独立确定bandwidth与paired null；禁止跨表示比较raw MMD大小。",
        "- handcrafted scaler只在dose100保守planner参考行上拟合。",
        "- lane-quality只作post-treatment描述性关联，不用于删样本、重加权或因果调整。", "",
    ]
    (output_dir / "stage6l_context_representation_ablation_freeze_report_zh.md").write_text("\n".join(report), encoding="utf-8")
    return manifest


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
