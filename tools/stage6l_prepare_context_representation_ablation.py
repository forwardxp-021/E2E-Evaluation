#!/usr/bin/env python3
"""Build frozen Stage 6L A-D representations without changing source contexts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage7_m6_4b_run_locked_rollouts import sha256_file
from tools.stage7e_embed_stage6_dataset import embed_context, load_checkpoint


FREEZE_STATUS = "FROZEN_BEFORE_STAGE6L_REPRESENTATION_ABLATION"
REPRESENTATIONS = [
    "learned64_full_context",
    "learned64_neighbor_zero_input",
    "ego_kinematic_13d",
    "handcrafted_interaction_trajectory_46d",
]
DOSE_LABELS = ["dose25", "dose50", "dose75", "dose100"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze_manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage6j_context_dir", type=Path, required=True)
    parser.add_argument("--stage6j_embedding_dir", type=Path, required=True)
    parser.add_argument("--stage6k_contexts_dir", type=Path, required=True)
    parser.add_argument("--stage6k_embeddings_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def wrap_angle(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values) + np.pi) % (2.0 * np.pi) - np.pi


def ego_kinematic_features(ego: np.ndarray, mask: np.ndarray, dt: float = 0.1) -> np.ndarray:
    ego = np.asarray(ego, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if ego.ndim != 3 or ego.shape[-1] != 8 or mask.shape != ego.shape[:2]:
        raise ValueError(f"Invalid ego/mask shapes: {ego.shape}/{mask.shape}")
    rows = np.empty((len(ego), 13), dtype=np.float64)
    for row_index, (sequence, valid) in enumerate(zip(ego, mask)):
        values = sequence[valid]
        if len(values) < 2:
            raise ValueError(f"Row {row_index} has fewer than two valid frames")
        speed = values[:, 5]
        accel = np.diff(speed, prepend=speed[0]) / dt
        jerk = np.diff(accel, prepend=accel[0]) / dt
        heading_delta = wrap_angle(np.diff(values[:, 4]))
        yaw_rate = np.concatenate([[0.0], heading_delta / dt])
        displacement = np.diff(values[:, :2], axis=0)
        rows[row_index] = [
            np.mean(speed),
            np.std(speed),
            np.quantile(speed, 0.95),
            speed[-1] - speed[0],
            np.sqrt(np.mean(accel**2)),
            np.mean(np.abs(accel)),
            np.quantile(np.abs(accel), 0.95),
            np.sqrt(np.mean(jerk**2)),
            np.quantile(np.abs(jerk), 0.95),
            np.sqrt(np.mean(yaw_rate**2)),
            np.mean(np.abs(yaw_rate)),
            np.sum(np.abs(heading_delta)),
            np.sum(np.linalg.norm(displacement, axis=1)),
        ]
    return rows


def feature_names(schema_path: Path) -> list[str]:
    schema = read_json(schema_path)
    names = [str(row["name"]) for row in schema.get("features", [])]
    if len(names) != 33:
        raise ValueError(f"Expected 33 interaction features in {schema_path}, got {len(names)}")
    return names


def fit_reference_scaler(values: np.ndarray, scale_floor: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    finite_values = np.where(np.isfinite(values), values, np.nan)
    median = np.nanmedian(finite_values, axis=0)
    if not np.isfinite(median).all():
        raise ValueError("Reference scaler has all-nonfinite feature columns")
    q25 = np.nanquantile(finite_values, 0.25, axis=0)
    q75 = np.nanquantile(finite_values, 0.75, axis=0)
    scale = np.maximum(q75 - q25, float(scale_floor))
    return median, scale


def apply_scaler(values: np.ndarray, median: np.ndarray, scale: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    filled = np.where(np.isfinite(values), values, median[None, :])
    scaled = (filled - median[None, :]) / scale[None, :]
    if not np.isfinite(scaled).all():
        raise ValueError("Scaled handcrafted representation contains non-finite values")
    return scaled.astype(np.float32)


def validate_metadata(metadata: pd.DataFrame) -> None:
    required = {"global_row", "scenario_token", "planner_name", "log_name", "scenario_type"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"metadata missing: {missing}")
    if len(metadata) != 366 or set(metadata["global_row"].astype(int)) != set(range(366)):
        raise ValueError("Stage 6L expects exhaustive global_row 0..365")


def dose_paths(args: argparse.Namespace, label: str) -> tuple[Path, Path]:
    if label == "dose100":
        return args.stage6j_context_dir.resolve(), args.stage6j_embedding_dir.resolve()
    return args.stage6k_contexts_dir.resolve() / label, args.stage6k_embeddings_dir.resolve() / label


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output_dir exists: {output_dir}; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    freeze = read_json(args.freeze_manifest.resolve())
    if freeze.get("status") != FREEZE_STATUS or freeze.get("new_stage6l_representation_or_bdd_read") is not False:
        raise ValueError("Stage 6L freeze manifest is invalid")
    design = freeze["design"]
    if [row["id"] for row in design["representations"]] != REPRESENTATIONS:
        raise ValueError("Frozen representation definitions changed")
    if sha256_file(args.checkpoint.resolve()) != freeze["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint differs from freeze")

    raw_by_dose: dict[str, dict[str, Any]] = {}
    canonical_tokens: list[str] | None = None
    schema_names: list[str] | None = None
    for label in DOSE_LABELS:
        context_dir, embedding_dir = dose_paths(args, label)
        metadata = pd.read_csv(context_dir / "metadata.csv").sort_values("global_row").reset_index(drop=True)
        validate_metadata(metadata)
        current_tokens = metadata["scenario_token"].astype(str).tolist()
        if canonical_tokens is None:
            canonical_tokens = current_tokens
        elif current_tokens != canonical_tokens:
            raise ValueError(f"Row-level scenario token order differs for {label}")
        names = feature_names(context_dir / "feature_schema.json")
        if schema_names is None:
            schema_names = names
        elif names != schema_names:
            raise ValueError(f"Interaction feature schema differs for {label}")
        context = np.load(context_dir / "context_traj.npy", mmap_mode="r")
        ego = np.load(context_dir / "ego_seq.npy", mmap_mode="r")
        mask = np.load(context_dir / "ego_seq_mask.npy", mmap_mode="r")
        interaction = np.asarray(np.load(context_dir / "interaction_feat_style.npy", mmap_mode="r"), dtype=np.float64)
        learned_full = np.asarray(np.load(embedding_dir / "embedding.npy", mmap_mode="r"), dtype=np.float32)
        if context.shape != (366, 150, 83) or learned_full.shape != (366, 64) or interaction.shape != (366, 33):
            raise ValueError(f"Unexpected Stage 6L input shape for {label}")
        ego_features = ego_kinematic_features(ego, mask)
        raw_by_dose[label] = {
            "metadata": metadata,
            "context": context,
            "learned_full": learned_full,
            "ego_features": ego_features,
            "interaction": interaction,
            "context_dir": context_dir,
            "embedding_dir": embedding_dir,
        }

    reference = raw_by_dose["dose100"]
    reference_mask = reference["metadata"]["planner_name"].astype(str).to_numpy() == design["planner_b"]
    if int(reference_mask.sum()) != 183:
        raise ValueError("Expected 183 dose100 conservative reference rows")
    ego_median, ego_scale = fit_reference_scaler(reference["ego_features"][reference_mask], 1e-6)
    combined_reference = np.concatenate([reference["ego_features"], reference["interaction"]], axis=1)
    combined_median, combined_scale = fit_reference_scaler(combined_reference[reference_mask], 1e-6)
    scaler_dir = output_dir / "scalers"
    scaler_dir.mkdir()
    np.savez_compressed(scaler_dir / "handcrafted_reference_scalers.npz", ego_median=ego_median, ego_scale=ego_scale, combined_median=combined_median, combined_scale=combined_scale)

    ckpt = load_checkpoint(args.checkpoint.resolve())
    rep_root = output_dir / "representations"
    metadata_root = output_dir / "metadata"
    metadata_root.mkdir()
    records: dict[str, Any] = {}
    for rep in REPRESENTATIONS:
        (rep_root / rep).mkdir(parents=True)
        records[rep] = {}
    for label in DOSE_LABELS:
        current = raw_by_dose[label]
        shutil.copy2(current["context_dir"] / "metadata.csv", metadata_root / f"{label}.csv")
        full = current["learned_full"]
        masked_context = np.asarray(current["context"], dtype=np.float32).copy()
        masked_context[:, :, 8:83] = 0.0
        neighbor_zero, embed_meta = embed_context(masked_context, ckpt, args.batch_size, args.device)
        ego_scaled = apply_scaler(current["ego_features"], ego_median, ego_scale)
        combined = np.concatenate([current["ego_features"], current["interaction"]], axis=1)
        combined_scaled = apply_scaler(combined, combined_median, combined_scale)
        arrays = {
            "learned64_full_context": full,
            "learned64_neighbor_zero_input": neighbor_zero,
            "ego_kinematic_13d": ego_scaled,
            "handcrafted_interaction_trajectory_46d": combined_scaled,
        }
        for rep, values in arrays.items():
            if values.shape != (366, int(next(row["dimension"] for row in design["representations"] if row["id"] == rep))):
                raise ValueError(f"Representation shape mismatch: {rep}/{label}/{values.shape}")
            path = rep_root / rep / f"{label}.npy"
            np.save(path, np.asarray(values, dtype=np.float32))
            records[rep][label] = {"path": str(path), "shape": list(values.shape), "sha256": sha256_file(path)}
        records["learned64_neighbor_zero_input"][label]["encoder_metadata"] = embed_meta

    scaler_json = {
        "reference": design["handcrafted_scaling"]["reference"],
        "reference_rows": int(reference_mask.sum()),
        "ego_feature_names": next(row["features"] for row in design["representations"] if row["id"] == "ego_kinematic_13d"),
        "interaction_feature_names": schema_names,
        "ego_scale_floor_hits": int(np.sum(ego_scale <= 1e-6)),
        "combined_scale_floor_hits": int(np.sum(combined_scale <= 1e-6)),
        "scaler_npz_sha256": sha256_file(scaler_dir / "handcrafted_reference_scalers.npz"),
    }
    write_json(scaler_dir / "handcrafted_reference_scalers.json", scaler_json)
    manifest = {
        "schema_version": "stage6l_context_representation_preparation_v1",
        "status": "STAGE6L_A_D_REPRESENTATIONS_READY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_manifest_sha256": sha256_file(args.freeze_manifest.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
        "source_outputs_modified": False,
        "lane_pipeline_changed": False,
        "retraining_performed": False,
        "neighbor_zero_interpretation": "same_checkpoint_input_ablation_not_independently_trained_ego_only_model",
        "handcrafted_scaler": scaler_json,
        "representations": records,
        "metadata": {label: {"path": str(metadata_root / f"{label}.csv"), "sha256": sha256_file(metadata_root / f"{label}.csv")} for label in DOSE_LABELS},
        "tool_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json(output_dir / "stage6l_representation_manifest.json", manifest)
    report = [
        "# Stage 6L A–D 表示准备报告", "", "## 状态", "", "`STAGE6L_A_D_REPRESENTATIONS_READY`", "",
        "- 原 Stage 6J/K 输出被修改: `false`", "- lane pipeline被修改: `false`", "- 重训练: `false`", "",
        "## 表示", "",
        "- A：现有64D完整上下文embedding，逐字节复用。",
        "- B：同checkpoint保留ego 0:8通道、将neighbor 8:83通道置零后重新推理；它不是独立ego-only训练模型。",
        "- C：13D显式ego运动学摘要。",
        "- D：13D ego摘要 + 33D interaction/trajectory handcrafted摘要。",
        "- C/D只用dose100保守planner的183行拟合median/IQR scaler。", "",
    ]
    (output_dir / "stage6l_representation_preparation_report_zh.md").write_text("\n".join(report), encoding="utf-8")
    return manifest


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
