#!/usr/bin/env python3
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset

from train_human_behavior_embedding import Enc, AuxRegHead, load_feature_names, select_aux_target_indices
from trajectory_preprocessing import (
    assert_finite_array,
    compute_traj_nan_stats,
    load_traj_as_dense_array,
    normalize_local,
    sanitize_trajectory_array,
)


def _safe_torch_load(checkpoint_path: Path):
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def _safe_spearman(pred: np.ndarray, target: np.ndarray, min_valid_pairs: int, name: str, warnings: list[str]):
    valid = np.isfinite(pred) & np.isfinite(target)
    valid_pairs = int(valid.sum())
    if valid_pairs < min_valid_pairs:
        warnings.append(
            f"{name}: spearman unavailable (valid_pairs={valid_pairs} < min_valid_pairs={min_valid_pairs})"
        )
        return None, valid_pairs
    pred_v = pred[valid]
    target_v = target[valid]
    if float(np.std(pred_v)) < 1e-12:
        warnings.append(f"{name}: spearman unavailable (prediction variance near zero)")
        return None, valid_pairs
    if float(np.std(target_v)) < 1e-12:
        warnings.append(f"{name}: spearman unavailable (target variance near zero)")
        return None, valid_pairs
    corr = spearmanr(pred_v, target_v).correlation
    if corr is None or not np.isfinite(corr):
        warnings.append(f"{name}: spearman unavailable (scipy returned non-finite)")
        return None, valid_pairs
    return float(corr), valid_pairs


def _create_smoke_inputs(tmp_root: Path):
    data_dir = tmp_root / "d"
    data_dir.mkdir(parents=True, exist_ok=True)
    n, t = 24, 16
    traj = np.random.randn(n, t, 4).astype(np.float32)
    traj[0, 2:7, 0] = np.nan
    traj[1, 3:6, 1] = np.nan
    traj[2, 0:2, 2] = np.inf
    np.save(data_dir / "traj.npy", traj)

    target_names = ["rms_accel", "rms_jerk", "max_abs_accel", "max_abs_jerk", "mean_thw", "min_thw"]
    feat = np.random.randn(n, len(target_names)).astype(np.float32)
    np.save(data_dir / "feat_style.npy", feat)
    (data_dir / "feat_style_columns.txt").write_text("\n".join(target_names) + "\n", encoding="utf-8")

    split = np.array(["train"] * (n // 2) + ["test"] * (n - n // 2), dtype="<U8")
    np.save(data_dir / "split.npy", split)

    emb_dim, aux_hidden = 16, 32
    enc = Enc(emb_dim)
    aux = AuxRegHead(emb_dim, aux_hidden, len(target_names))
    ckpt = tmp_root / "smoke_model.pt"
    torch.save(
        {
            "model": enc.state_dict(),
            "aux_head": aux.state_dict(),
            "embedding_dim": emb_dim,
            "model_architecture": {"aux_hidden_dim": aux_hidden},
            "aux_regression": True,
        },
        ckpt,
    )
    return data_dir, ckpt


def run(a):
    warnings = []
    if a.smoke_test:
        tmp_root = Path(tempfile.mkdtemp())
        data_dir, checkpoint_path = _create_smoke_inputs(tmp_root)
    else:
        data_dir = Path(a.data_dir)
        checkpoint_path = Path(a.checkpoint)

    traj = np.load(data_dir / "traj.npy", allow_pickle=True)
    feat = np.load(data_dir / "feat_style.npy").astype(np.float32)
    split = np.load(data_dir / "split.npy", allow_pickle=True).astype(str)

    traj_raw = load_traj_as_dense_array(traj)
    raw_stats = compute_traj_nan_stats(traj_raw)
    traj_clean, sdiag = sanitize_trajectory_array(traj, mode=a.traj_nan_mode, max_nan_ratio=a.max_traj_nan_ratio)
    clean_stats = compute_traj_nan_stats(traj_clean)

    retained = sdiag["retained_indices"]
    dropped = sdiag["dropped_indices"]
    if len(dropped) > 0 and not a.allow_drop:
        raise RuntimeError(
            f"sanitize dropped {len(dropped)} rows; row-aligned aux evaluation requires zero dropped rows. "
            "Use different sanitization or explicitly set --allow_drop."
        )

    feat = feat[retained]
    split = split[retained]
    assert len(traj_clean) == len(feat) == len(split), "Sanitized arrays are not aligned"

    if a.fail_on_nonfinite:
        assert_finite_array(traj_clean, "traj_clean")
        assert_finite_array(feat, "feat_style")

    feature_names = load_feature_names(str(data_dir))
    aux_targets = [x.strip() for x in a.aux_targets.split(",") if x.strip()]
    aux_idx = select_aux_target_indices(feature_names, aux_targets)

    y = feat[:, aux_idx]
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.clip(y, -a.aux_target_clip, a.aux_target_clip)

    mask = split == a.eval_split
    x_eval = traj_clean[mask]
    y_eval = y[mask]
    assert x_eval.shape[0] == y_eval.shape[0], "x/y eval sample count mismatch"
    assert_finite_array(x_eval, "x_eval")
    assert_finite_array(y_eval, "y_eval")

    obj = _safe_torch_load(checkpoint_path)
    emb_dim = int(obj.get("embedding_dim", 64))
    aux_hidden = int(obj.get("model_architecture", {}).get("aux_hidden_dim", 128))

    enc = Enc(emb_dim)
    enc.load_state_dict(obj["model"], strict=False)

    has_aux_head = ("aux_head" in obj and bool(obj.get("aux_head"))) or bool(obj.get("aux_regression", False))
    if not has_aux_head and not a.allow_missing_aux_head:
        raise RuntimeError(
            "This checkpoint does not contain an auxiliary regression head. "
            "Please run Stage 4F training with --aux_regression."
        )
    aux_head_loaded = "aux_head" in obj and bool(obj.get("aux_head"))
    aux = AuxRegHead(emb_dim, aux_hidden, len(aux_idx))
    if aux_head_loaded:
        aux.load_state_dict(obj.get("aux_head", {}), strict=False)
    else:
        warnings.append("aux_head missing in checkpoint; using randomly initialized aux head due to --allow_missing_aux_head")

    device = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    enc.to(device).eval()
    aux.to(device).eval()

    ds = TensorDataset(torch.from_numpy(x_eval).float(), torch.from_numpy(y_eval).float())
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False)

    pred_parts, target_parts = [], []
    with torch.no_grad():
        for bi, (xb, yb) in enumerate(dl):
            x_local = normalize_local(xb.to(device))
            z = enc(x_local)
            assert_finite_array(z, f"embedding_batch_{bi}")
            pred = aux(z)
            if not bool(torch.isfinite(pred).all()):
                bad = torch.nonzero(~torch.isfinite(pred), as_tuple=False)[:10].tolist()
                raise RuntimeError(f"non-finite aux predictions in batch {bi}; bad indices={bad}")
            pred_parts.append(pred.detach().cpu().numpy())
            target_parts.append(yb.detach().cpu().numpy())

    preds = np.concatenate(pred_parts, axis=0) if pred_parts else np.zeros((0, len(aux_targets)), dtype=np.float32)
    targets = np.concatenate(target_parts, axis=0) if target_parts else np.zeros((0, len(aux_targets)), dtype=np.float32)

    metrics = {}
    for i, name in enumerate(aux_targets):
        mae = float(np.mean(np.abs(preds[:, i] - targets[:, i]))) if len(preds) else None
        rmse = float(np.sqrt(np.mean((preds[:, i] - targets[:, i]) ** 2))) if len(preds) else None
        sp, valid_pairs = _safe_spearman(preds[:, i], targets[:, i], a.min_valid_pairs, name, warnings)
        metrics[name] = {"mae": mae, "rmse": rmse, "spearman": sp, "valid_pairs": valid_pairs}

    row_aligned = len(dropped) == 0 and len(traj_raw) == len(traj_clean)
    summary = {
        "data_dir": str(data_dir),
        "checkpoint": str(checkpoint_path),
        "eval_split": a.eval_split,
        "n_total_rows": int(len(traj_raw)),
        "n_eval_samples": int(len(x_eval)),
        "aux_targets": aux_targets,
        "aux_target_indices": [int(i) for i in aux_idx],
        "device": str(device),
        "batch_size": int(a.batch_size),
        "traj_nan_count_raw": int(raw_stats["nan_count"]),
        "traj_inf_count_raw": int(raw_stats["inf_count"]),
        "traj_nan_count_after_sanitize": int(clean_stats["nan_count"]),
        "traj_inf_count_after_sanitize": int(clean_stats["inf_count"]),
        "traj_repaired_sample_count": int(sdiag["repaired_count"]),
        "dropped_sample_count": int(sdiag["dropped_count"]),
        "row_aligned": bool(row_aligned),
        "retained_indices": [int(i) for i in retained.tolist()] if len(dropped) > 0 else None,
        "dropped_indices": [int(i) for i in dropped.tolist()] if len(dropped) > 0 else None,
        "aux_head_loaded": bool(aux_head_loaded),
        "metrics": metrics,
        "warnings": warnings,
    }

    out_path = Path(a.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"n_eval_samples={summary['n_eval_samples']}")
    print(f"traj_nan_count_raw={summary['traj_nan_count_raw']}")
    print(f"traj_nan_count_after_sanitize={summary['traj_nan_count_after_sanitize']}")
    for k, v in metrics.items():
        print(f"spearman[{k}]={v['spearman']}")
    print("aux_eval_done")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="outputs/waymo_human_v1_full51")
    p.add_argument("--checkpoint")
    p.add_argument("--eval_split", default="test")
    p.add_argument("--aux_targets", default="rms_accel,rms_jerk,max_abs_accel,max_abs_jerk,mean_thw,min_thw")
    p.add_argument("--aux_target_clip", type=float, default=10.0)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out_path", required=True)
    p.add_argument("--traj_nan_mode", choices=["interpolate", "zero", "drop"], default="interpolate")
    p.add_argument("--max_traj_nan_ratio", type=float, default=0.2)
    p.add_argument("--fail_on_nonfinite", action="store_true")
    p.add_argument("--allow_drop", action="store_true")
    p.add_argument("--min_valid_pairs", type=int, default=100)
    p.add_argument("--allow_missing_aux_head", action="store_true")
    p.add_argument("--smoke_test", action="store_true")
    args = p.parse_args()
    if not args.smoke_test and not args.checkpoint:
        raise ValueError("--checkpoint is required unless --smoke_test is set")
    run(args)
