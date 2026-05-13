#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset

from train_human_behavior_embedding import Enc, AuxRegHead, load_feature_names, select_aux_target_indices
from trajectory_preprocessing import load_traj_as_dense_array, normalize_local, sanitize_trajectory_array


def _safe_spearman(pred: np.ndarray, target: np.ndarray) -> float:
    corr = spearmanr(pred, target).correlation
    return float(corr) if np.isfinite(corr) else float("nan")


def run(a):
    data_dir = Path(a.data_dir)
    traj = np.load(data_dir / "traj.npy", allow_pickle=True)
    feat = np.load(data_dir / "feat_style.npy").astype(np.float32)
    split = np.load(data_dir / "split.npy", allow_pickle=True).astype(str)

    traj_raw = load_traj_as_dense_array(traj)
    traj_clean, sdiag = sanitize_trajectory_array(
        traj,
        mode=a.traj_nan_mode,
        max_nan_ratio=a.max_traj_nan_ratio,
    )
    retained = sdiag["retained_indices"]
    feat = feat[retained]
    split = split[retained]

    feat_finite_mask = np.isfinite(feat).all(axis=1)
    if not feat_finite_mask.all():
        traj_clean = traj_clean[feat_finite_mask]
        feat = feat[feat_finite_mask]
        split = split[feat_finite_mask]

    feature_names = load_feature_names(a.data_dir)
    aux_targets = [x.strip() for x in a.aux_targets.split(",") if x.strip()]
    aux_idx = select_aux_target_indices(feature_names, aux_targets)
    y = feat[:, aux_idx]
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.clip(y, -a.aux_target_clip, a.aux_target_clip)

    mask = split == a.eval_split
    x_eval = traj_clean[mask]
    y_eval = y[mask]

    checkpoint_path = Path(a.checkpoint)
    obj = torch.load(checkpoint_path, map_location="cpu")
    emb_dim = int(obj.get("embedding_dim", 64))
    aux_hidden = int(obj.get("model_architecture", {}).get("aux_hidden_dim", 128))

    enc = Enc(emb_dim)
    enc.load_state_dict(obj["model"], strict=False)
    aux = AuxRegHead(emb_dim, aux_hidden, len(aux_idx))
    aux.load_state_dict(obj.get("aux_head", {}), strict=False)

    device = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    enc.to(device).eval()
    aux.to(device).eval()

    ds = TensorDataset(torch.from_numpy(x_eval), torch.from_numpy(y_eval))
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False)

    pred_parts, target_parts = [], []
    with torch.no_grad():
        for xb, yb in dl:
            z = enc(normalize_local(xb.to(device)))
            pred_parts.append(aux(z).cpu().numpy())
            target_parts.append(yb.numpy())

    preds = np.concatenate(pred_parts, axis=0) if pred_parts else np.zeros((0, len(aux_targets)), dtype=np.float32)
    targets = np.concatenate(target_parts, axis=0) if target_parts else np.zeros((0, len(aux_targets)), dtype=np.float32)

    metrics = {}
    for i, name in enumerate(aux_targets):
        mae = float(np.mean(np.abs(preds[:, i] - targets[:, i]))) if len(preds) else float("nan")
        rmse = float(np.sqrt(np.mean((preds[:, i] - targets[:, i]) ** 2))) if len(preds) else float("nan")
        sp = _safe_spearman(preds[:, i], targets[:, i]) if len(preds) else float("nan")
        metrics[name] = {"mae": mae, "rmse": rmse, "spearman": sp}

    summary = {
        "data_dir": str(data_dir),
        "checkpoint": str(checkpoint_path),
        "eval_split": a.eval_split,
        "n_input_samples": int(len(traj_raw)),
        "n_retained_after_traj_sanitize": int(len(retained)),
        "n_eval_samples": int(len(x_eval)),
        "traj_sanitize": {
            "mode": a.traj_nan_mode,
            "max_nan_ratio": float(a.max_traj_nan_ratio),
            "repaired_count": int(sdiag["repaired_count"]),
            "dropped_count": int(sdiag["dropped_count"]),
        },
        "aux_targets": aux_targets,
        "metrics": metrics,
    }

    out_path = Path(a.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("aux_eval_done")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="outputs/waymo_human_v1_full51")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eval_split", default="test")
    p.add_argument("--aux_targets", default="rms_accel,rms_jerk,max_abs_accel,max_abs_jerk,mean_thw,min_thw")
    p.add_argument("--aux_target_clip", type=float, default=10.0)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out_path", required=True)
    p.add_argument("--traj_nan_mode", choices=["interpolate", "zero", "drop"], default="interpolate")
    p.add_argument("--max_traj_nan_ratio", type=float, default=0.2)
    run(p.parse_args())
