#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage7_m6_2_locked_task_bdd import paired_randomization_test  # noqa: E402
from tools.stage7_m6_4b_run_locked_rollouts import sha256_file  # noqa: E402
from tools.stage7_m6_scenario_conditioned_bdd import (  # noqa: E402
    holm_adjust,
    permutation_bdd,
    validate_and_build_pairs,
)


SCHEMA_VERSION = "stage6j_pure_longitudinal_paired_bdd_evidence_v1"


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_csv(
    path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_task_masks(
    metadata: pd.DataFrame,
    pair_indices: np.ndarray,
    task_definitions: Mapping[str, Sequence[str]],
) -> tuple[Dict[str, np.ndarray], List[str]]:
    required = {"global_row", "scenario_type"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"metadata missing task columns: {missing}")
    by_row = metadata.set_index("global_row", drop=False)
    pair_types: List[str] = []
    for position, (row_a, row_b) in enumerate(pair_indices):
        type_a = str(by_row.loc[int(row_a), "scenario_type"])
        type_b = str(by_row.loc[int(row_b), "scenario_type"])
        if type_a != type_b:
            raise ValueError(
                f"pair {position} has unequal pre-treatment scenario_type: {type_a}, {type_b}"
            )
        pair_types.append(type_a)
    types = np.asarray(pair_types, dtype=object)
    masks: Dict[str, np.ndarray] = {}
    covered = np.zeros(len(pair_indices), dtype=bool)
    for task, scenario_types in task_definitions.items():
        mask = np.isin(types, np.asarray(list(scenario_types), dtype=object))
        masks[task] = mask
        covered |= mask
    unmapped = sorted(set(types[~covered].tolist()))
    if unmapped or not covered.all():
        raise ValueError(f"frozen task definitions do not cover all pairs: {unmapped}")
    return masks, pair_types


def validate_inputs(
    config: Mapping[str, Any],
    embedding_dir: Path,
    checkpoint: Path,
    kinematic_gate_summary: Path,
) -> tuple[np.ndarray, pd.DataFrame, Dict[str, Any]]:
    if config.get("frozen_before_stage6j_bdd_result_read") is not True:
        raise ValueError("Stage6J BDD method was not frozen before result read")
    gate = read_json(kinematic_gate_summary)
    if (
        gate.get("status") != config.get("required_kinematic_gate_status")
        or gate.get("kinematic_gate_passed") is not True
        or gate.get("embedding_or_bdd_read") is not False
    ):
        raise ValueError("pre-frozen kinematic gate did not authorize embedding/BDD analysis")
    if sha256_file(checkpoint) != config.get("expected_checkpoint_sha256"):
        raise ValueError("Waymo checkpoint SHA-256 differs from the frozen BDD config")
    manifest = read_json(embedding_dir / "embedding_manifest.json")
    if (
        manifest.get("total_rows") != config.get("expected_row_count")
        or manifest.get("embedding_dim") != config.get("expected_embedding_dim")
        or manifest.get("context_layout_used") != "stage5d_context_dataset_direct"
        or manifest.get("stage5d_schema_matched") is not True
        or manifest.get("context_padded_to_checkpoint_dim") is not False
        or manifest.get("nonfinite_embedding_values") != 0
    ):
        raise ValueError(f"embedding manifest violates the frozen contract: {manifest}")
    embedding = np.asarray(
        np.load(embedding_dir / "embedding.npy", mmap_mode="r"), dtype=np.float64
    )
    expected_shape = (
        int(config["expected_row_count"]),
        int(config["expected_embedding_dim"]),
    )
    if embedding.shape != expected_shape or not np.isfinite(embedding).all():
        raise ValueError(f"expected finite embedding {expected_shape}, got {embedding.shape}")
    metadata = pd.read_csv(embedding_dir / "metadata.csv")
    if len(metadata) != len(embedding):
        raise ValueError("embedding and metadata row counts differ")
    return embedding, metadata, manifest


def run_analysis(
    config: Mapping[str, Any],
    embedding: np.ndarray,
    metadata: pd.DataFrame,
    paired_rows: Sequence[Mapping[str, str]],
) -> tuple[List[Dict[str, Any]], Dict[str, np.ndarray], np.ndarray]:
    planner_a = str(config["planner_a"])
    planner_b = str(config["planner_b"])
    pair_indices, scenarios = validate_and_build_pairs(
        metadata,
        paired_rows,
        len(embedding),
        planner_a=planner_a,
        planner_b=planner_b,
    )
    if pair_indices.shape != (int(config["expected_pair_count"]), 2):
        raise ValueError(f"unexpected complete pair shape: {pair_indices.shape}")
    primary = config["primary_analysis"]
    repetitions = int(primary["permutations"])
    seed = int(config["seed"])
    index_a, index_b = pair_indices[:, 0], pair_indices[:, 1]
    overall, overall_samples = permutation_bdd(
        embedding[index_a],
        embedding[index_b],
        repetitions=repetitions,
        seed=seed,
        paired_swap=True,
        progress_label="Stage6J overall paired BDD",
    )
    rows: List[Dict[str, Any]] = [
        {
            "scope": "overall",
            "role": "primary",
            "n_pairs": len(pair_indices),
            "mmd2": overall["mmd2"],
            "bandwidth": overall["bandwidth"],
            "exceedance_count": overall["exceedance_count"],
            "raw_p": overall["p_value"],
            "holm_p": overall["p_value"],
            "reject_0_05": bool(overall["p_value"] < float(primary["alpha"])),
            "randomization_mode": "monte_carlo",
            "permutations": repetitions,
        }
    ]
    null_samples: Dict[str, np.ndarray] = {"overall": overall_samples}
    task_config = config["task_conditioned_secondary"]
    task_masks, _ = build_task_masks(metadata, pair_indices, task_config["tasks"])
    task_rows: List[Dict[str, Any]] = []
    minimum_pairs = int(task_config["minimum_pairs_per_task"])
    for position, (task, mask) in enumerate(task_masks.items()):
        task_pairs = pair_indices[mask]
        if len(task_pairs) < minimum_pairs:
            raise ValueError(
                f"task {task} has {len(task_pairs)} pairs below frozen minimum {minimum_pairs}"
            )
        result, samples = paired_randomization_test(
            embedding[task_pairs[:, 0]],
            embedding[task_pairs[:, 1]],
            monte_carlo_repetitions=repetitions,
            seed=seed + position + 1,
            progress_label=f"Stage6J {task} paired BDD",
        )
        task_rows.append(
            {
                "scope": task,
                "role": "task_conditioned_secondary",
                "n_pairs": len(task_pairs),
                "mmd2": result["mmd2"],
                "bandwidth": result["bandwidth"],
                "exceedance_count": result["exceedance_count"],
                "raw_p": result["p_value"],
                "holm_p": math.nan,
                "reject_0_05": False,
                "randomization_mode": result["randomization_mode"],
                "permutations": repetitions,
            }
        )
        null_samples[task] = samples
    adjusted = holm_adjust([float(row["raw_p"]) for row in task_rows])
    for row, adjusted_p in zip(task_rows, adjusted):
        row["holm_p"] = float(adjusted_p)
        row["reject_0_05"] = bool(adjusted_p < float(task_config["family_alpha"]))
    rows.extend(task_rows)
    if len(set(scenarios)) != int(config["expected_pair_count"]):
        raise ValueError("scenario tokens are not unique across complete pairs")
    return rows, null_samples, pair_indices


def write_chinese_report(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> None:
    overall = rows[0]
    conclusion = "检出纯纵向风格分布差异" if overall["reject_0_05"] else "未检出纯纵向风格分布差异"
    lines = [
        "# Stage 6J 纯纵向 paired BDD 中文报告",
        "",
        f"## 主要结论：{conclusion}",
        "",
        f"总体183个同场景pair的BDD（MMD²）为`{overall['mmd2']:.8f}`，固定bandwidth为`{overall['bandwidth']:.8f}`，100000次pair内label swap的plus-one p为`{overall['raw_p']:.8g}`。",
        "",
        "| 分析范围 | 角色 | pair数 | BDD / MMD² | raw p | Holm p | 0.05结论 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scope']} | {row['role']} | {row['n_pairs']} | {row['mmd2']:.8f} | "
            f"{row['raw_p']:.8g} | {row['holm_p']:.8g} | {'reject' if row['reject_0_05'] else 'not reject'} |"
        )
    lines += [
        "",
        "## 解释边界",
        "",
        "- 总体检验是primary；三个task是pre-treatment secondary，并以Holm校正后的p值解释。",
        "- 该结果只回答Waymo训练的embedding能否区分受控、同场景、纯纵向PDM风格差异。",
        "- 这不等于异log/异场景的软件发布检出率，也不等于真实整车厂数据验证。",
        "- BDD不表示安全性、性能优劣或因果机制。",
        "- 183个场景来自先前技术成功的confirmation ledger，因此不能称为全新独立场景确认。",
        "",
        f"分析配置：`{config['schema_version']}`。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen Stage6J overall and task-conditioned paired BDD analysis."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage6j_paired_bdd_analysis.json"),
    )
    parser.add_argument("--embedding_dir", type=Path, required=True)
    parser.add_argument("--paired_delta_csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--kinematic_gate_summary", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"output_dir already exists: {args.output_dir}; use --overwrite"
            )
        shutil.rmtree(args.output_dir)
    config = read_json(args.config)
    embedding, metadata, manifest = validate_inputs(
        config, args.embedding_dir, args.checkpoint, args.kinematic_gate_summary
    )
    paired_rows = read_csv(args.paired_delta_csv)
    rows, null_samples, pair_indices = run_analysis(
        config, embedding, metadata, paired_rows
    )
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "stage6j_paired_bdd_results.csv", rows, list(rows[0]))
    np.savez_compressed(args.output_dir / "stage6j_paired_bdd_null_samples.npz", **null_samples)
    write_chinese_report(
        args.output_dir / "stage6j_paired_bdd_report_zh.md", rows, config
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PAIRED_BDD_ANALYSIS_COMPLETE",
        "dataset_role": config["dataset_role"],
        "pair_count": len(pair_indices),
        "row_count": len(embedding),
        "planner_a": config["planner_a"],
        "planner_b": config["planner_b"],
        "primary_result": rows[0],
        "task_results": rows[1:],
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "config_sha256": sha256_file(args.config),
        "embedding_manifest_sha256": sha256_file(
            args.embedding_dir / "embedding_manifest.json"
        ),
        "embedding_sha256": sha256_file(args.embedding_dir / "embedding.npy"),
        "paired_delta_sha256": sha256_file(args.paired_delta_csv),
        "kinematic_gate_summary_sha256": sha256_file(args.kinematic_gate_summary),
        "analysis_tool_sha256": sha256_file(Path(__file__).resolve()),
        "embedding_manifest": manifest,
        "interpretation_limits": config["interpretation_limits"],
    }
    write_json(args.output_dir / "stage6j_paired_bdd_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
