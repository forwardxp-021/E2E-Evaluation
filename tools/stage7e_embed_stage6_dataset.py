#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from tools.train_context_behavior_embedding import ContextFlattenGRUEncoder


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def reset_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"output_dir exists: {path}. Use --overwrite.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_stage7d_paths(dataset_dir: Path):
    shard_dir = dataset_dir / "shards" / "shard_000"
    required = {
        "ego": shard_dir / "ego_seq.npy",
        "neighbor": shard_dir / "neighbor_seq.npy",
        "features": shard_dir / "interaction_feat_style.npy",
        "metadata": shard_dir / "metadata.csv",
        "feature_schema": dataset_dir / "feature_schema.json",
        "shard_manifest": dataset_dir / "shard_manifest.json",
        "planner_indices": dataset_dir / "planner_policy_indices",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Stage 7E missing Stage 7D input(s): " + ", ".join(missing))
    return required


def build_context(ego: np.ndarray, neighbor: np.ndarray, max_neighbors: int) -> np.ndarray:
    if ego.ndim != 3 or ego.shape[-1] != 8:
        raise ValueError(f"ego_seq.npy must have shape [rows,T,8], got {list(ego.shape)}")
    if neighbor.ndim != 4 or neighbor.shape[0] != ego.shape[0] or neighbor.shape[2] != ego.shape[1] or neighbor.shape[-1] != 9:
        raise ValueError(f"neighbor_seq.npy must have shape [rows,K,T,9] aligned to ego, got {list(neighbor.shape)}")
    k = min(int(max_neighbors), int(neighbor.shape[1]))
    neigh = np.asarray(neighbor[:, :k], dtype=np.float32).transpose(0, 2, 1, 3).reshape(ego.shape[0], ego.shape[1], k * 9)
    return np.concatenate([np.asarray(ego, dtype=np.float32), neigh], axis=-1)


def load_checkpoint(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"checkpoint/model_path does not exist: {path}")
    return torch.load(path, map_location="cpu")


def embed_context(context: np.ndarray, checkpoint: Path, batch_size: int, device: str) -> tuple[np.ndarray, dict]:
    ckpt = load_checkpoint(checkpoint)
    emb_dim = int(ckpt.get("embedding_dim", 64))
    ckpt_context_dim = ckpt.get("context_dim")
    if ckpt_context_dim is not None and int(ckpt_context_dim) != int(context.shape[-1]):
        raise ValueError(
            f"Checkpoint context_dim={ckpt_context_dim} does not match Stage 7E context_dim={context.shape[-1]}. "
            "Adjust --max_neighbors to match the existing Stage 5/6 encoder; do not train a Stage 7-only embedding unless explicitly approved."
        )
    nonfinite_context = int((~np.isfinite(context)).sum())
    if nonfinite_context:
        context = np.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    model = ContextFlattenGRUEncoder(int(context.shape[-1]), embedding_dim=emb_dim)
    try:
        model.load_state_dict(ckpt["model"], strict=False)
    except RuntimeError as exc:
        raise RuntimeError(
            "Existing encoder checkpoint is incompatible with the Stage 7E context tensor. "
            "Check --max_neighbors and use the same context layout as the trained Stage 5/6 encoder."
        ) from exc
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    model.to(dev).eval()
    chunks = []
    nonfinite = 0
    with torch.no_grad():
        for start in tqdm(range(0, len(context), batch_size), desc="Stage 7E embedding", unit="batch"):
            batch = torch.from_numpy(np.asarray(context[start:start + batch_size], dtype=np.float32)).to(dev)
            z = model(batch).cpu().numpy().astype(np.float32)
            nonfinite += int((~np.isfinite(z)).sum())
            chunks.append(z)
    emb = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, emb_dim), dtype=np.float32)
    return emb, {
        "embedding_dim": emb_dim,
        "device": str(dev),
        "nonfinite_context_values_replaced_with_zero": nonfinite_context,
        "nonfinite_embedding_values": nonfinite,
    }


def run(args) -> None:
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    reset_dir(output_dir, args.overwrite)
    paths = load_stage7d_paths(dataset_dir)

    ego = np.load(paths["ego"], mmap_mode="r")
    neighbor = np.load(paths["neighbor"], mmap_mode="r")
    features = np.load(paths["features"], mmap_mode="r")
    metadata = pd.read_csv(paths["metadata"])
    context = build_context(ego, neighbor, args.max_neighbors)
    embedding, emb_meta = embed_context(context, Path(args.checkpoint or args.model_path), args.batch_size, args.device)

    if not (embedding.shape[0] == features.shape[0] == len(metadata) == ego.shape[0]):
        raise ValueError(f"Row count mismatch: embedding={embedding.shape[0]} features={features.shape[0]} metadata={len(metadata)} ego={ego.shape[0]}")
    if int(context.shape[0]) != int(ego.shape[0]):
        raise ValueError("Stage 7E row semantics violation: context rows changed relative to Stage 7D ego rows.")

    np.save(output_dir / "embedding.npy", embedding)
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    shutil.copytree(paths["planner_indices"], output_dir / "planner_policy_indices")
    emb_shard = output_dir / "embeddings" / "shard_000000"
    emb_shard.mkdir(parents=True, exist_ok=True)
    np.save(emb_shard / "embeddings.npy", embedding)

    manifest = {
        "stage": "7E",
        "purpose": "embed_stage7d_stage6_compatible_idm_dataset_with_existing_stage5_stage6_encoder",
        "dataset_dir": str(dataset_dir),
        "checkpoint": str(Path(args.checkpoint or args.model_path)),
        "total_rows": int(embedding.shape[0]),
        "embedding_dim": int(embedding.shape[1]) if embedding.ndim == 2 else 0,
        "embedding_path": "embedding.npy",
        "embedding_shard_paths": ["embeddings/shard_000000/embeddings.npy"],
        "source_shard_manifest": str(paths["shard_manifest"]),
        "row_semantics": "one row = one scenario × one planner-controlled nuPlan ego rollout",
        "row_order": "unchanged from Stage 7D shard row order",
        "multi_agent_ego_expansion": False,
        "max_neighbors_used_for_encoder_context": int(args.max_neighbors),
        **emb_meta,
    }
    write_json(output_dir / "embedding_manifest.json", manifest)

    validation = {
        "pass": bool(embedding.shape[0] == len(metadata) and emb_meta["nonfinite_embedding_values"] == 0),
        "embedding_rows": int(embedding.shape[0]),
        "metadata_rows": int(len(metadata)),
        "stage7d_rows": int(ego.shape[0]),
        "planner_policy_indices_copied": sorted(p.name for p in (output_dir / "planner_policy_indices").glob("*.npy")),
        "row_order_unchanged_from_stage7d": True,
        "no_multi_agent_ego_expansion": True,
    }
    warnings = []
    if emb_meta["nonfinite_context_values_replaced_with_zero"]:
        warnings.append(
            "WARNING: Stage 7E replaced non-finite encoder context values with 0.0 before running the existing encoder; "
            "row order and row count were unchanged."
        )
    if not validation["pass"]:
        warnings.append("Stage 7E validation failed; inspect row counts and nonfinite embeddings.")
    write_json(output_dir / "warnings.json", {"warnings": warnings, "validation": validation})
    report = [
        "# Stage 7E IDM Embedding Smoke Report", "",
        f"- validation.pass: **{str(validation['pass']).upper()}**",
        f"- rows: `{embedding.shape[0]}`",
        f"- embedding shape: `{list(embedding.shape)}`",
        "- row semantics: `one row = one scenario × one planner-controlled nuPlan ego rollout`",
        "- multi-agent ego expansion: `false`",
        "- Stage 6-compatible planner_policy_indices copied unchanged.",
    ]
    (output_dir / "embedding_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if not validation["pass"]:
        raise RuntimeError("Stage 7E validation failed; see warnings.json")


def parse_args():
    p = argparse.ArgumentParser(description="Stage 7E smoke: embed Stage 7D Stage 6-compatible IDM data with existing context behavior encoder.")
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--model_path")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_neighbors", type=int, default=5, help="Use first K Stage 7D neighbors to match the existing Stage 5 context encoder input dimension.")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if not args.checkpoint and not args.model_path:
        raise ValueError("Provide --checkpoint or --model_path for the existing Stage 5/6 encoder.")
    return args


if __name__ == "__main__":
    run(parse_args())
