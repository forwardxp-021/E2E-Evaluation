#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import torch


def load_traj_as_dense_array(traj: np.ndarray) -> np.ndarray:
    if isinstance(traj, np.ndarray) and traj.dtype != object:
        arr = traj.astype(np.float32)
        if arr.ndim != 3 or arr.shape[-1] != 4:
            raise ValueError(f"traj must be (N,T,4), got {arr.shape}")
        return arr
    if not isinstance(traj, np.ndarray) or traj.dtype != object:
        raise ValueError("traj must be ndarray (N,T,4) or object array of (T,4)")
    n = len(traj)
    if n == 0:
        return np.zeros((0, 0, 4), dtype=np.float32)
    t = traj[0].shape[0]
    out = np.zeros((n, t, 4), dtype=np.float32)
    for i, sample in enumerate(traj):
        sample = np.asarray(sample, dtype=np.float32)
        if sample.shape != (t, 4):
            raise ValueError("all object traj samples must have same shape (T,4)")
        out[i] = sample
    return out


def compute_traj_nan_stats(arr: np.ndarray) -> Dict[str, float]:
    total = float(arr.size) if arr.size else 1.0
    finite = np.isfinite(arr)
    return {
        "shape": list(arr.shape),
        "nan_count": int(np.isnan(arr).sum()),
        "inf_count": int(np.isinf(arr).sum()),
        "finite_ratio": float(finite.sum() / total),
    }


def assert_finite_array(arr, name: str) -> None:
    if isinstance(arr, torch.Tensor):
        finite = torch.isfinite(arr)
        if not bool(finite.all()):
            idx = torch.nonzero(~finite, as_tuple=False)[:10].tolist()
            raise RuntimeError(f"{name} has non-finite values, first bad indices={idx}")
    else:
        finite = np.isfinite(arr)
        if not bool(finite.all()):
            idx = np.argwhere(~finite)[:10].tolist()
            raise RuntimeError(f"{name} has non-finite values, first bad indices={idx}")


def sanitize_trajectory_array(traj, *, mode="interpolate", max_nan_ratio=0.2) -> Tuple[np.ndarray, Dict]:
    arr = load_traj_as_dense_array(traj)
    n, t, d = arr.shape
    dropped = np.zeros(n, dtype=bool)
    repaired = np.zeros(n, dtype=bool)
    for i in range(n):
        sample = arr[i]
        nan_ratio = (~np.isfinite(sample)).sum() / sample.size
        if mode == "drop" and nan_ratio > 0:
            dropped[i] = True
            continue
        if mode == "drop" and nan_ratio > max_nan_ratio:
            dropped[i] = True
            continue
        if mode == "zero":
            if nan_ratio > 0:
                repaired[i] = True
            arr[i] = np.nan_to_num(sample, nan=0.0, posinf=0.0, neginf=0.0)
            continue

        changed = False
        for k in range(d):
            vec = sample[:, k]
            good = np.isfinite(vec)
            if not np.any(good):
                dropped[i] = True
                changed = True
                break
            if np.all(good):
                continue
            changed = True
            idx = np.where(good)[0]
            vals = vec[good]
            vec[:] = np.interp(np.arange(t), idx, vals)
            sample[:, k] = vec
        if dropped[i]:
            continue
        if nan_ratio > max_nan_ratio:
            dropped[i] = True
            continue
        if not np.isfinite(sample).all():
            changed = True
            arr[i] = np.nan_to_num(sample, nan=0.0, posinf=0.0, neginf=0.0)
        if changed:
            repaired[i] = True
    kept = np.where(~dropped)[0]
    arr = arr[kept]
    return arr.astype(np.float32), {
        "repaired_count": int(repaired.sum()),
        "dropped_count": int(dropped.sum()),
        "retained_indices": kept.astype(np.int64),
        "dropped_indices": np.where(dropped)[0].astype(np.int64),
    }


def normalize_local(tr: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    assert_finite_array(tr, "normalize_local.input")
    xy = tr[..., :2]
    v = tr[..., 2:4]
    x0 = xy[:, 0:1, :]
    xy = xy - x0

    speed = torch.linalg.norm(v, dim=-1)
    valid_v = speed > eps
    first_valid_idx = valid_v.float().argmax(dim=1)
    has_valid_v = valid_v.any(dim=1)
    batch_idx = torch.arange(v.shape[0], device=tr.device)
    v0 = v[batch_idx, first_valid_idx]

    disp_step = min(5, xy.shape[1] - 1)
    disp = xy[:, disp_step, :] - xy[:, 0, :]
    use_disp = (~has_valid_v) | (torch.linalg.norm(v0, dim=-1) <= eps)
    heading_vec = torch.where(use_disp[:, None], disp, v0)

    bad_h = torch.linalg.norm(heading_vec, dim=-1) <= eps
    if bad_h.any():
        heading_vec = heading_vec.clone()
        heading_vec[bad_h] = torch.tensor([1.0, 0.0], device=tr.device, dtype=tr.dtype)

    h = torch.atan2(heading_vec[:, 1], heading_vec[:, 0])
    c = torch.cos(-h)[:, None]
    s = torch.sin(-h)[:, None]

    xr = xy[:, :, 0] * c - xy[:, :, 1] * s
    yr = xy[:, :, 0] * s + xy[:, :, 1] * c
    vr = v[:, :, 0] * c - v[:, :, 1] * s
    vy = v[:, :, 0] * s + v[:, :, 1] * c
    out = torch.stack([xr, yr, vr, vy], dim=-1)
    assert_finite_array(out, "normalize_local.output")
    return out
