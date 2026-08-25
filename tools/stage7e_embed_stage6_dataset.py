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


def load_context_dataset_paths(context_dataset_dir: Path):
    required = {
        "context": context_dataset_dir / "context_traj.npy",
        "features": context_dataset_dir / "interaction_feat_style.npy",
        "metadata": context_dataset_dir / "metadata.csv",
        "feature_schema": context_dataset_dir / "feature_schema.json",
        "shard_manifest": context_dataset_dir / "shard_manifest.json",
        "planner_indices": context_dataset_dir / "planner_policy_indices",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Stage 7E missing context dataset input(s): " + ", ".join(missing))
    return required


STAGE5D83_DEPRECATION_ERROR = (
    "Final Stage7E Stage5D context must be built by build_nuplan_5neighbor_context_dataset.py and loaded via --context_dataset_dir. "
    "Stage7D top-K neighbor_seq cannot be relabeled as Stage5D semantic slots."
)


def make_stage5d83_schema() -> dict:
    from tools.stage5d_context_core import make_stage5d_context_schema

    schema = make_stage5d_context_schema(schema_name="stage5d83_context_dataset_direct")
    schema.update({
        "built_by_stage5d_training_script": "tools/build_waymo_5neighbor_context_dataset.py",
        "final_stage7e_nuplan_builder": "tools/build_nuplan_5neighbor_context_dataset.py",
        "stage7e_final_input_mode": "--context_dataset_dir",
        "deprecated_stage7d_topk_reconstruction": True,
        "deprecation_message": STAGE5D83_DEPRECATION_ERROR,
        "stage6_input_contract": "Stage 6 BDD/report-card consumes exported embedding vectors; interaction_feat_style.npy is used for reports/evaluation, not as encoder input.",
    })
    return schema

def load_checkpoint(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"checkpoint/model_path does not exist: {path}")
    return torch.load(path, map_location="cpu")



def embed_context(context: np.ndarray, ckpt: dict, batch_size: int, device: str) -> tuple[np.ndarray, dict]:
    emb_dim = int(ckpt.get("embedding_dim", 64))
    ckpt_context_dim = ckpt.get("context_dim")
    if ckpt_context_dim is None:
        raise ValueError("Checkpoint is missing required context_dim; Stage 7E must build checkpoint-compatible context exactly.")
    if int(ckpt_context_dim) != int(context.shape[-1]):
        raise ValueError(
            f"Checkpoint context_dim={ckpt_context_dim} does not match final Stage 7E context_dim={context.shape[-1]}. "
            "Use --context_dataset_dir produced by build_nuplan_5neighbor_context_dataset.py."
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
            "Use a context dataset with the same Stage 5D context layout as the trained Stage 5/6 encoder."
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
        "checkpoint_context_dim": int(ckpt_context_dim),
        "final_context_dim": int(context.shape[-1]),
        "device": str(dev),
        "nonfinite_context_values_replaced_with_zero": nonfinite_context,
        "nonfinite_embedding_values": nonfinite,
    }


def run(args) -> None:
    context_dataset_dir = Path(args.context_dataset_dir)
    output_dir = Path(args.output_dir)
    reset_dir(output_dir, args.overwrite)
    checkpoint_path = Path(args.checkpoint or args.model_path)
    ckpt = load_checkpoint(checkpoint_path)
    if "context_dim" not in ckpt:
        raise ValueError("Existing Stage 5/6 encoder checkpoint must contain checkpoint['context_dim'].")

    paths = load_context_dataset_paths(context_dataset_dir)
    context = np.load(paths["context"], mmap_mode="r")
    features = np.load(paths["features"], mmap_mode="r")
    metadata = pd.read_csv(paths["metadata"])
    if int(ckpt["context_dim"]) != int(context.shape[-1]):
        raise ValueError(f"checkpoint['context_dim']={ckpt['context_dim']} does not match context_traj.npy last dimension={context.shape[-1]}.")
    context_meta = {
        "context_layout_requested": "context_dataset_dir",
        "context_layout_used": "stage5d_context_dataset_direct",
        "base_context_layout": "context_traj.npy",
        "base_context_dim": int(context.shape[-1]),
        "checkpoint_context_dim": int(ckpt["context_dim"]),
        "final_context_dim": int(context.shape[-1]),
        "context_padded_to_checkpoint_dim": False,
        "padding_dim": 0,
        "stage5d_schema_matched": int(context.shape[-1]) == 83,
        "does_not_rebuild_context_from_stage7d_neighbor_seq": True,
    }
    source_manifest = paths["shard_manifest"]
    ego_rows = context.shape[0]
    embedding, emb_meta = embed_context(np.asarray(context), ckpt, args.batch_size, args.device)

    if not (embedding.shape[0] == features.shape[0] == len(metadata) == ego_rows):
        raise ValueError(f"Row count mismatch: embedding={embedding.shape[0]} features={features.shape[0]} metadata={len(metadata)} ego/context={ego_rows}")
    if int(context.shape[0]) != int(ego_rows):
        raise ValueError("Stage 7E row semantics violation: context rows changed relative to source rows.")

    np.save(output_dir / "embedding.npy", embedding)
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    shutil.copytree(paths["planner_indices"], output_dir / "planner_policy_indices")
    emb_shard = output_dir / "embeddings" / "shard_000000"
    emb_shard.mkdir(parents=True, exist_ok=True)
    np.save(emb_shard / "embeddings.npy", embedding)

    manifest = {
        "stage": "7E",
        "purpose": "embed_stage5d_common_core_context_dataset_with_existing_stage5_stage6_encoder",
        "context_dataset_dir": str(context_dataset_dir),
        "checkpoint": str(checkpoint_path),
        "total_rows": int(embedding.shape[0]),
        "embedding_dim": int(embedding.shape[1]) if embedding.ndim == 2 else 0,
        "embedding_path": "embedding.npy",
        "embedding_shard_paths": ["embeddings/shard_000000/embeddings.npy"],
        "source_shard_manifest": str(source_manifest),
        "row_semantics": "one row = one scenario × one planner-controlled nuPlan ego rollout",
        "row_order": "unchanged from context dataset row order",
        "multi_agent_ego_expansion": False,
        **context_meta,
        **emb_meta,
    }
    write_json(output_dir / "embedding_manifest.json", manifest)
    np.save(output_dir / "context_traj.npy", np.asarray(context, dtype=np.float32))
    schema_src = context_dataset_dir / "stage5d_context_schema.json"
    if schema_src.exists():
        shutil.copy2(schema_src, output_dir / "stage7e_context_schema.json")
    else:
        write_json(output_dir / "stage7e_context_schema.json", make_stage5d83_schema())

    validation = {
        "pass": bool(
            embedding.shape[0] == len(metadata)
            and emb_meta["nonfinite_embedding_values"] == 0
            and emb_meta["checkpoint_context_dim"] == emb_meta["final_context_dim"]
        ),
        "embedding_rows": int(embedding.shape[0]),
        "metadata_rows": int(len(metadata)),
        "context_dataset_rows": int(ego_rows),
        "planner_policy_indices_copied": sorted(p.name for p in (output_dir / "planner_policy_indices").glob("*.npy")),
        "row_order_unchanged_from_context_dataset": True,
        "no_multi_agent_ego_expansion": True,
        "checkpoint_context_dim": int(emb_meta["checkpoint_context_dim"]),
        "final_context_dim": int(emb_meta["final_context_dim"]),
        "checkpoint_context_dim_matches_final_context_dim": bool(emb_meta["checkpoint_context_dim"] == emb_meta["final_context_dim"]),
        "context_layout_used": context_meta["context_layout_used"],
        "context_padded_to_checkpoint_dim": False,
        "stage5d_schema_matched": bool(context_meta.get("stage5d_schema_matched", False)),
        "does_not_rebuild_context_from_stage7d_neighbor_seq": True,
    }
    warnings = []
    if emb_meta["nonfinite_context_values_replaced_with_zero"]:
        warnings.append(
            "WARNING: Stage 7E replaced non-finite encoder context values with 0.0 before running the existing encoder; "
            "row order and row count were unchanged."
        )
    if not validation["pass"]:
        warnings.append("Stage 7E validation failed; inspect row counts and nonfinite embeddings.")
    write_json(output_dir / "warnings.json", {**context_meta, "warnings": warnings, "validation": validation})
    report = [
        "# Stage 7E IDM Embedding Report", "",
        "> Final Stage 7E path: direct Stage 5D common-core context dataset embedding via --context_dataset_dir.",
        "",
        f"- validation.pass: **{str(validation['pass']).upper()}**",
        f"- rows: `{embedding.shape[0]}`",
        f"- embedding shape: `{list(embedding.shape)}`",
        f"- context layout: `{context_meta['context_layout_used']}`",
        f"- checkpoint context_dim: `{emb_meta['checkpoint_context_dim']}`",
        f"- final context_dim: `{emb_meta['final_context_dim']}`",
        "- context rebuilt from Stage 7D neighbor_seq: `false`",
        "- row semantics: `one row = one scenario × one planner-controlled nuPlan ego rollout`",
        "- multi-agent ego expansion: `false`",
        "- Stage 6-compatible planner_policy_indices copied unchanged.",
    ]
    (output_dir / "embedding_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if not validation["pass"]:
        raise RuntimeError("Stage 7E validation failed; see warnings.json")

def parse_args():
    p = argparse.ArgumentParser(description="Stage 7E: embed a Stage 5D-compatible nuPlan context dataset with the existing context behavior encoder.")
    p.add_argument("--context_dataset_dir", required=True, help="Stage 5D-compatible nuPlan context dataset directory containing context_traj.npy. Stage 7E final path never rebuilds context from Stage 7D neighbor_seq.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--model_path")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if not args.checkpoint and not args.model_path:
        raise ValueError("Provide --checkpoint or --model_path for the existing Stage 5/6 encoder.")
    return args


if __name__ == "__main__":
    run(parse_args())
