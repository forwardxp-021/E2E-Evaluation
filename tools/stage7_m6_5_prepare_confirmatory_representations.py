#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage7_m5_representation_mechanism_analysis import (  # noqa: E402
    TRAJECTORY_FEATURES,
    build_trajectory_summary,
    robust_standardize,
)
from tools.stage7_m6_4b_run_locked_rollouts import sha256_file  # noqa: E402


SCHEMA_VERSION = "stage7_m6_5_confirmatory_representations_v1"


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare frozen M6.5 interaction and trajectory representation controls.")
    parser.add_argument("--embedding_path", type=Path, required=True)
    parser.add_argument("--interaction_feature_path", type=Path, required=True)
    parser.add_argument("--paired_delta_csv", type=Path, required=True)
    parser.add_argument("--metadata_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output_dir already exists: {args.output_dir}")
    embedding = np.asarray(np.load(args.embedding_path, mmap_mode="r"), dtype=np.float64)
    interaction = np.asarray(np.load(args.interaction_feature_path, mmap_mode="r"), dtype=np.float64)
    metadata = read_csv(args.metadata_csv)
    paired_rows = read_csv(args.paired_delta_csv)
    if embedding.ndim != 2 or embedding.shape[1] != 64 or not np.isfinite(embedding).all():
        raise ValueError(f"expected finite unchanged 64D embedding, got {embedding.shape}")
    if interaction.ndim != 2 or len(interaction) != len(embedding):
        raise ValueError("interaction feature rows must match embedding rows")
    if len(metadata) != len(embedding):
        raise ValueError("metadata rows must match embedding rows")
    trajectory, labels, groups, pair_indices = build_trajectory_summary(paired_rows, len(embedding))
    if len(pair_indices) != 310:
        raise ValueError(f"M6.5 requires exactly 310 complete pairs, got {len(pair_indices)}")
    if not np.array_equal(pair_indices, np.arange(len(embedding)).reshape(-1, 2)):
        raise ValueError("paired rows must preserve contiguous scenario-major planner order")
    if set(labels.tolist()) != {0, 1} or len(np.unique(groups)) != 310:
        raise ValueError("invalid planner labels or pair groups")
    args.output_dir.mkdir(parents=True)
    learned_path = args.output_dir / "learned_embedding.npy"
    interaction_path = args.output_dir / "interaction_features.npy"
    trajectory_path = args.output_dir / "trajectory_summary.npy"
    np.save(learned_path, embedding.astype(np.float32))
    np.save(interaction_path, robust_standardize(interaction).astype(np.float32))
    np.save(trajectory_path, robust_standardize(trajectory).astype(np.float32))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "CONFIRMATORY_REPRESENTATIONS_READY",
        "row_count": len(embedding),
        "pair_count": len(pair_indices),
        "representations": {
            "learned_embedding": {
                "path": str(learned_path.resolve()),
                "shape": list(embedding.shape),
                "transform": "unchanged original 64D embedding; float32 serialization only",
            },
            "interaction_features": {
                "path": str(interaction_path.resolve()),
                "shape": list(interaction.shape),
                "transform": "M5 pooled median/IQR robust_standardize; frozen M6.2 applies its locked pooled transform again",
            },
            "trajectory_summary": {
                "path": str(trajectory_path.resolve()),
                "shape": [len(trajectory), len(TRAJECTORY_FEATURES)],
                "features": list(TRAJECTORY_FEATURES),
                "transform": "M5 trajectory summary then pooled median/IQR robust_standardize; frozen M6.2 applies its locked pooled transform again",
            },
        },
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in {
                "embedding": args.embedding_path,
                "interaction_features": args.interaction_feature_path,
                "paired_delta": args.paired_delta_csv,
                "metadata": args.metadata_csv,
            }.items()
        },
        "output_hashes": {
            path.name: sha256_file(path)
            for path in (learned_path, interaction_path, trajectory_path)
        },
        "preparation_tool_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json(args.output_dir / "m6_5_representation_manifest.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
