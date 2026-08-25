#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from tools import stage6e_calibrate_unpaired_release as stage6e
from tools import stage7_m6_4b_run_locked_rollouts as batch
from tools.stage7_m6_2_locked_task_bdd import PRETREATMENT_TASKS


SCHEMA_VERSION = "stage6h_expanded_embedding_pool_v1"
READY_STATUS = "EXPANDED_800_PAIR_EMBEDDING_POOL_READY"
EXPECTED_COMBINED_TASK_COUNTS = {
    "following_interaction": 182,
    "lane_change": 71,
    "stop_go_control": 182,
    "high_motion_dynamics": 182,
    "dense_or_vulnerable_interaction": 183,
}


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def essential_context_schema(path: Path) -> Dict[str, Any]:
    raw = stage6e.read_json(path)
    keys = [
        "schema_name", "context_dim", "dim_formula", "ego_channels",
        "neighbor_channels_per_slot", "neighbor_slots", "channels",
        "slot_assignment_method",
    ]
    return {key: raw.get(key) for key in keys}


def audit_source(
    label: str,
    embedding_dir: Path,
    expected_pairs: int,
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, Any]]:
    embedding_path = embedding_dir / "embedding.npy"
    metadata_path = embedding_dir / "metadata.csv"
    manifest_path = embedding_dir / "embedding_manifest.json"
    schema_path = embedding_dir / "stage7e_context_schema.json"
    for path in (embedding_path, metadata_path, manifest_path, schema_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    embedding = np.load(embedding_path, mmap_mode="r")
    metadata = pd.read_csv(metadata_path)
    manifest = stage6e.read_json(manifest_path)
    if embedding.shape != (expected_pairs * 2, 64):
        raise ValueError(f"{label} embedding shape mismatch: {embedding.shape}")
    if len(metadata) != len(embedding):
        raise ValueError(f"{label} metadata/embedding row mismatch")
    if not np.isfinite(embedding).all():
        raise ValueError(f"{label} embedding contains non-finite values")
    required = {"global_row", "scenario_token", "planner_name", "log_name", "map_name", "scenario_type"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"{label} metadata missing columns: {missing}")
    if metadata["global_row"].astype(int).tolist() != list(range(len(metadata))):
        raise ValueError(f"{label} global_row is not contiguous and aligned")
    planners = set(batch.EXPECTED_PLANNERS)
    pair_failures: List[str] = []
    for token, group in metadata.groupby("scenario_token", sort=False):
        if len(group) != 2 or set(group["planner_name"].astype(str)) != planners:
            pair_failures.append(str(token))
            continue
        for column in ("log_name", "map_name", "scenario_type"):
            if group[column].astype(str).nunique() != 1:
                pair_failures.append(str(token))
                break
    if pair_failures:
        raise ValueError(f"{label} pair audit failed: {pair_failures[:10]}")
    if metadata["scenario_token"].nunique() != expected_pairs:
        raise ValueError(f"{label} unique pair count mismatch")
    checkpoint = Path(str(manifest.get("checkpoint", "")))
    if not checkpoint.is_absolute():
        checkpoint = Path.cwd() / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    audit = {
        "label": label,
        "pairs": expected_pairs,
        "rows": len(metadata),
        "embedding_shape": list(embedding.shape),
        "embedding_sha256": batch.sha256_file(embedding_path),
        "metadata_sha256": batch.sha256_file(metadata_path),
        "manifest_sha256": batch.sha256_file(manifest_path),
        "context_schema_sha256": batch.sha256_file(schema_path),
        "essential_context_schema": essential_context_schema(schema_path),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": batch.sha256_file(checkpoint),
        "all_embeddings_finite": True,
        "all_pairs_complete": True,
    }
    return embedding, metadata, audit


def task_for_scenario_type(scenario_type: str) -> str:
    matches = [task for task, values in PRETREATMENT_TASKS.items() if scenario_type in values]
    if len(matches) != 1:
        raise ValueError(f"scenario_type does not map to exactly one frozen task: {scenario_type}")
    return matches[0]


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"output_dir already exists: {args.output_dir}")
    old_embedding, old_meta, old_audit = audit_source("existing_310", args.existing_embedding_dir, 310)
    new_embedding, new_meta, new_audit = audit_source("stage6g_490", args.new_embedding_dir, 490)
    if old_audit["checkpoint_sha256"] != new_audit["checkpoint_sha256"]:
        raise ValueError("old/new embeddings were not produced by the same checkpoint")
    if old_audit["essential_context_schema"] != new_audit["essential_context_schema"]:
        raise ValueError("old/new Stage5D essential context schemas differ")
    if list(old_meta.columns) != list(new_meta.columns):
        raise ValueError("old/new metadata columns or order differ")
    old_tokens = set(old_meta["scenario_token"].astype(str))
    new_tokens = set(new_meta["scenario_token"].astype(str))
    overlap = sorted(old_tokens & new_tokens)
    if overlap:
        raise ValueError(f"old/new scenario token overlap: {overlap}")

    parts: List[pd.DataFrame] = []
    for source_pool, frame in (("existing_310", old_meta), ("stage6g_490", new_meta)):
        current = frame.copy()
        current.insert(0, "source_global_row", current["global_row"].astype(int))
        current.insert(1, "source_scenario_index", current["scenario_index"].astype(int))
        current.insert(2, "source_pool", source_pool)
        parts.append(current)
    metadata = pd.concat(parts, ignore_index=True)
    metadata["global_row"] = np.arange(len(metadata), dtype=np.int64)
    pair_order = {token: index for index, token in enumerate(metadata["scenario_token"].drop_duplicates())}
    metadata["scenario_index"] = metadata["scenario_token"].map(pair_order).astype(np.int64)
    metadata["tensor_scenario_position"] = metadata["scenario_index"]
    embedding = np.concatenate(
        [np.asarray(old_embedding, dtype=np.float32), np.asarray(new_embedding, dtype=np.float32)], axis=0
    )
    if embedding.shape != (1600, 64) or len(metadata) != 1600:
        raise ValueError("combined embedding pool is not exactly 1600 rows × 64D")

    pair_rows: List[Dict[str, Any]] = []
    task_counts: Counter = Counter()
    for token, group in metadata.groupby("scenario_token", sort=False):
        task = task_for_scenario_type(str(group.iloc[0]["scenario_type"]))
        task_counts[task] += 1
        pair_rows.append(
            {
                "scenario_index": int(group.iloc[0]["scenario_index"]),
                "scenario_token": str(token),
                "source_pool": str(group.iloc[0]["source_pool"]),
                "task": task,
                "log_name": str(group.iloc[0]["log_name"]),
                "map_name": str(group.iloc[0]["map_name"]),
                "scenario_type": str(group.iloc[0]["scenario_type"]),
                "row_count": len(group),
                "planner_count": group["planner_name"].nunique(),
                "pair_invariants_pass": True,
                "embedding_rows_finite": bool(np.isfinite(embedding[group.index.to_numpy()]).all()),
            }
        )
    actual_tasks = {task: task_counts[task] for task in PRETREATMENT_TASKS}
    if actual_tasks != EXPECTED_COMBINED_TASK_COUNTS:
        raise ValueError(f"combined task counts mismatch: {actual_tasks}")
    if len(pair_rows) != 800 or not all(row["embedding_rows_finite"] for row in pair_rows):
        raise ValueError("combined 800-pair audit failed")

    args.output_dir.mkdir(parents=True)
    np.save(args.output_dir / "embedding.npy", embedding)
    metadata.to_csv(args.output_dir / "metadata.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(args.output_dir / "pair_audit.csv", index=False)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": READY_STATUS,
        "issue": "https://github.com/forwardxp-021/E2E-Evaluation/issues/246",
        "pair_count": 800,
        "row_count": 1600,
        "embedding_shape": [1600, 64],
        "cluster_count": int(metadata["log_name"].nunique()),
        "task_counts": actual_tasks,
        "old_new_token_overlap_count": 0,
        "all_pairs_complete": True,
        "all_embeddings_finite": True,
        "checkpoint_sha256": old_audit["checkpoint_sha256"],
        "essential_context_schema_sha256": canonical_hash(old_audit["essential_context_schema"]),
        "source_audits": [old_audit, new_audit],
        "outputs": {
            "embedding_sha256": batch.sha256_file(args.output_dir / "embedding.npy"),
            "metadata_sha256": batch.sha256_file(args.output_dir / "metadata.csv"),
            "pair_audit_sha256": batch.sha256_file(args.output_dir / "pair_audit.csv"),
        },
        "provenance": {
            "tool_sha256": batch.sha256_file(Path(__file__).resolve()),
            "git_commit": batch.resolve_git_commit(Path(__file__).resolve().parents[1]),
            "python": sys.version,
            "platform": platform.platform(),
        },
    }
    stage6e.write_json(args.output_dir / "stage6h_embedding_pool_summary.json", summary)
    report = [
        "# Stage 6H 800-pair embedding pool audit",
        "",
        f"- status: `{READY_STATUS}`",
        "- existing/new pairs: `310 + 490 = 800`",
        "- rows / embedding: `1600 / [1600,64]`",
        f"- independent log clusters: `{summary['cluster_count']}`",
        "- old/new token overlap: `0`",
        "- complete pairs / finite embeddings: `800/800`",
        f"- checkpoint SHA-256: `{summary['checkpoint_sha256']}`",
        "",
        "This pool is a public-data release-emulation input, not 800 independent road-test clusters.",
    ]
    (args.output_dir / "stage6h_embedding_pool_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and merge the existing 310 and Stage6G 490 embedding pairs.")
    parser.add_argument("--existing_embedding_dir", type=Path, required=True)
    parser.add_argument("--new_embedding_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
