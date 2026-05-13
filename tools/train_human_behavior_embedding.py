#!/usr/bin/env python3
import argparse
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from trajectory_preprocessing import (
    load_traj_as_dense_array,
    normalize_local,
    sanitize_trajectory_array,
)


class HumanDS(Dataset):
    def __init__(self, traj, feat, aux_targets=None):
        self.traj = traj.astype(np.float32)
        self.feat = feat.astype(np.float32)
        self.aux_targets = None if aux_targets is None else aux_targets.astype(np.float32)

    def __len__(self):
        return len(self.traj)

    def __getitem__(self, i):
        if self.aux_targets is None:
            return self.traj[i], self.feat[i], np.zeros((0,), dtype=np.float32)
        return self.traj[i], self.feat[i], self.aux_targets[i]


class Enc(nn.Module):
    def __init__(self, emb=64, hid=128):
        super().__init__()
        self.gru = nn.GRU(4, hid, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, emb))

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


class AuxRegHead(nn.Module):
    def __init__(self, emb_dim, hidden_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z):
        return self.mlp(z)


def load_feature_names(data_dir):
    p = Path(data_dir) / "feature_names_style.json"
    if p.exists():
        names = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(names, dict):
            names = names.get("feature_names", [])
        return list(names)
    return ['mean_speed', 'rms_accel', 'rms_jerk', 'rms_yaw_rate_proxy', 'rms_curvature_proxy', 'mean_thw', 'min_thw', 'max_abs_accel', 'max_abs_jerk']


def select_aux_target_indices(feature_names, aux_targets):
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    missing = [n for n in aux_targets if n not in name_to_idx]
    if missing:
        raise ValueError(f"Aux targets missing from feature names: {missing}")
    return [name_to_idx[n] for n in aux_targets]


def comfort_metric_alignment_loss(
    z,
    comfort_targets,
    metric_loss_type="mse",
    metric_distance="euclidean",
    metric_pair_sample=0,
    metric_use_z_normalized=True,
    metric_detach_target=True,
    eps=1e-8,
):
    bsz = int(z.shape[0])
    if bsz < 2:
        print("[warn] metric alignment skipped: batch size < 2")
        return torch.zeros((), device=z.device, dtype=z.dtype)
    if metric_loss_type == "rank":
        raise NotImplementedError("metric_loss_type=rank is not implemented yet; use mse or huber.")
    if metric_use_z_normalized:
        z = F.normalize(z, dim=1, eps=eps)
    if metric_distance == "euclidean":
        d_z = torch.cdist(z, z, p=2)
    elif metric_distance == "cosine":
        d_z = 1.0 - (z @ z.T)
    else:
        raise ValueError(f"Unknown metric_distance={metric_distance}")
    c = comfort_targets.detach() if metric_detach_target else comfort_targets
    d_c = torch.cdist(c, c, p=2)
    mask = ~torch.eye(bsz, dtype=torch.bool, device=z.device)
    d_z_off = d_z[mask]
    d_c_off = d_c[mask]
    if metric_pair_sample > 0 and metric_pair_sample < d_z_off.numel():
        idx = torch.randperm(d_z_off.numel(), device=z.device)[:metric_pair_sample]
        d_z_off = d_z_off[idx]
        d_c_off = d_c_off[idx]
    if d_z_off.numel() < 2:
        print("[warn] metric alignment skipped: too few off-diagonal pairs")
        return torch.zeros((), device=z.device, dtype=z.dtype)
    d_z_norm = (d_z_off - d_z_off.mean()) / (d_z_off.std(unbiased=False) + eps)
    d_c_norm = (d_c_off - d_c_off.mean()) / (d_c_off.std(unbiased=False) + eps)
    if metric_loss_type == "mse":
        loss = F.mse_loss(d_z_norm, d_c_norm)
    elif metric_loss_type == "huber":
        loss = F.smooth_l1_loss(d_z_norm, d_c_norm)
    else:
        raise ValueError(f"Unknown metric_loss_type={metric_loss_type}")
    if not torch.isfinite(loss):
        print("[warn] non-finite metric alignment loss; fallback to zero")
        return torch.zeros((), device=z.device, dtype=z.dtype)
    return loss


def soft_loss(z, f, temperature, feature_temperature, feature_weights=None):
    z = F.normalize(z, dim=1, eps=1e-8)
    logits = z @ z.T / temperature
    f = F.normalize(f, dim=1, eps=1e-8)
    if feature_weights is not None:
        fw = feature_weights.to(f.device).view(1, -1)
        f = f * torch.sqrt(fw)
    dist2 = torch.cdist(f, f, p=2) ** 2
    target_logits = -dist2 / feature_temperature
    b = z.shape[0]
    eye = torch.eye(b, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(eye, -1e9)
    target_logits = target_logits.masked_fill(eye, -1e9)
    log_q = F.log_softmax(logits, dim=1)
    p = F.softmax(target_logits, dim=1)
    row_valid = (~eye).any(dim=1)
    loss = -(p[row_valid] * log_q[row_valid]).sum(dim=1).mean()
    return loss, logits, target_logits


def run(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    warnings = []
    if args.smoke_test:
        td = Path(tempfile.mkdtemp()) / 'd'
        td.mkdir(parents=True)
        n, t = 96, 80
        traj = np.random.randn(n, t, 4).astype(np.float32)
        feat = np.random.randn(n, 16).astype(np.float32)
        traj[1, 2:5, 2] = np.nan
        traj[5, :2, 1] = np.nan
        traj[7, :, 3] = np.nan
        feat[3, 0] = np.inf
        feat[4, 1] = np.nan
        feat[8, 2] = 1e8
        split = np.array(['train'] * 64 + ['val'] * 16 + ['test'] * 16)
    else:
        d = Path(args.data_dir)
        traj = np.load(d / 'traj.npy', allow_pickle=True)
        feat = np.load(d / 'feat_style.npy')
        split = np.load(d / 'split.npy', allow_pickle=True).astype(str)

    traj_raw = load_traj_as_dense_array(traj)
    traj_nan_raw = int(np.isnan(traj_raw).sum())
    traj_inf_raw = int(np.isinf(traj_raw).sum())
    feat_nan_raw = int(np.isnan(feat).sum())
    feat_inf_raw = int(np.isinf(feat).sum())

    traj_clean, sdiag = sanitize_trajectory_array(traj, mode=args.traj_nan_mode, max_nan_ratio=args.max_traj_nan_ratio)
    retained = sdiag["retained_indices"]
    dropped = sdiag["dropped_indices"]
    feat = feat[retained]
    split = split[retained]

    feat_finite_mask = np.isfinite(feat).all(axis=1)
    dropped_feat_idx = np.where(~feat_finite_mask)[0]
    if dropped_feat_idx.size > 0:
        drop_global = retained[dropped_feat_idx]
        keep_feat = feat_finite_mask
        traj_clean = traj_clean[keep_feat]
        split = split[keep_feat]
        feat = feat[keep_feat]
        retained = retained[keep_feat]
        dropped = np.unique(np.concatenate([dropped, drop_global]))

    np.save(out / 'retained_indices.npy', retained)
    np.save(out / 'dropped_indices.npy', dropped)

    if args.fail_on_nonfinite and (not np.isfinite(traj_clean).all() or not np.isfinite(feat).all()):
        raise RuntimeError("Non-finite values remain after sanitization")

    train_mask = split == 'train'
    mu = feat[train_mask].mean(0, keepdims=True)
    sd = feat[train_mask].std(0, keepdims=True)
    sd = np.maximum(sd, args.feature_std_eps)
    feat_norm = (feat - mu) / sd
    feat_norm = np.nan_to_num(feat_norm, nan=0.0, posinf=0.0, neginf=0.0)
    min_before, max_before = float(feat_norm.min()), float(feat_norm.max())
    clipped_count = int((np.abs(feat_norm) > args.feature_clip).sum())
    feat_norm = np.clip(feat_norm, -args.feature_clip, args.feature_clip)
    min_after, max_after = float(feat_norm.min()), float(feat_norm.max())

    print(f"feat raw finite ratio: {np.isfinite(feat).mean():.6f}")
    print(f"feat norm before clip: min={min_before:.4f}, max={max_before:.4f}")
    print(f"feat norm after clip: min={min_after:.4f}, max={max_after:.4f}, clipped={clipped_count}")
    print(f"feat has_nan={np.isnan(feat_norm).any()}, has_inf={np.isinf(feat_norm).any()}")

    feature_names = load_feature_names(args.data_dir)
    weights = {k:1.0 for k in feature_names}
    if args.feature_weight_mode == 'jerk_comfort':
        weights.update({'rms_accel':2.0,'rms_jerk':4.0,'max_abs_accel':2.0,'max_abs_jerk':4.0,'mean_thw':2.0,'min_thw':2.0})
    elif args.feature_weight_mode == 'lateral':
        weights.update({'rms_yaw_rate_proxy':3.0,'rms_curvature_proxy':3.0,'heading_change_total':2.0})
    elif args.feature_weight_mode == 'custom':
        weights.update(json.loads(Path(args.feature_weights_json).read_text()))
    fw = np.ones(feat_norm.shape[1], dtype=np.float32)
    for i, n in enumerate(feature_names[:feat_norm.shape[1]]):
        fw[i] = float(weights.get(n, 1.0))
    (out / 'feature_weights.json').write_text(json.dumps({'mode': args.feature_weight_mode, 'weights': weights}, indent=2))
    fw_t = torch.tensor(fw, dtype=torch.float32)
    dev = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    aux_targets = [x.strip() for x in args.aux_targets.split(",") if x.strip()]
    metric_targets = [x.strip() for x in args.metric_targets.split(",") if x.strip()]
    aux_idx = []
    aux_data = None
    if args.aux_regression:
        aux_idx = select_aux_target_indices(feature_names, aux_targets)
        source_feat = feat_norm if args.aux_target_normalize else feat
        aux_data = source_feat[:, aux_idx]
        aux_data = np.nan_to_num(aux_data, nan=0.0, posinf=0.0, neginf=0.0)
        aux_data = np.clip(aux_data, -args.aux_target_clip, args.aux_target_clip)
        if aux_data.ndim != 2 or aux_data.shape[0] != len(feat):
            raise RuntimeError(f"bad aux target shape: {aux_data.shape}")
    metric_idx = []
    metric_data = None
    if args.comfort_metric_alignment:
        metric_idx = select_aux_target_indices(feature_names, metric_targets)
        source_feat = feat_norm if args.metric_target_normalize else feat
        metric_data = source_feat[:, metric_idx]
        metric_data = np.nan_to_num(metric_data, nan=0.0, posinf=0.0, neginf=0.0)
        metric_data = np.clip(metric_data, -args.metric_target_clip, args.metric_target_clip)
        if metric_data.ndim != 2 or metric_data.shape[0] != len(feat):
            raise RuntimeError(f"bad metric target shape: {metric_data.shape}")
        if not np.isfinite(metric_data).all():
            raise RuntimeError("metric targets contain non-finite values after sanitization")
    combo_data = None
    if args.aux_regression and args.comfort_metric_alignment:
        combo_data = np.concatenate([aux_data, metric_data], axis=1)
    elif args.aux_regression:
        combo_data = aux_data
    elif args.comfort_metric_alignment:
        combo_data = metric_data
    tr_ds = HumanDS(traj_clean[split == 'train'], feat_norm[split == 'train'], None if combo_data is None else combo_data[split == 'train'])
    va_ds = HumanDS(traj_clean[split == 'val'], feat_norm[split == 'val'], None if combo_data is None else combo_data[split == 'val'])
    tr = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    va = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = Enc(args.embedding_dim).to(dev)
    aux_head = AuxRegHead(args.embedding_dim, args.aux_hidden_dim, len(aux_idx)).to(dev) if args.aux_regression else None
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if aux_head is not None:
        opt = torch.optim.AdamW(list(model.parameters()) + list(aux_head.parameters()), lr=args.lr)
    if args.aux_loss_type == "huber":
        aux_criterion = nn.HuberLoss()
    elif args.aux_loss_type == "mse":
        aux_criterion = nn.MSELoss()
    else:
        aux_criterion = nn.SmoothL1Loss()
    best, bad, logs, debug = 1e9, 0, [], []

    def run_batch(tb, fb, ab, batch_i, train_mode, ep):
        tb, fb = tb.to(dev), fb.to(dev)
        ab = ab.to(dev)
        tb = normalize_local(tb)
        assert torch.isfinite(tb).all(), "non-finite trajectory batch before forward"
        z = model(tb)
        loss_soft, logits, tlogits = soft_loss(z, fb, args.temperature, args.feature_temperature, feature_weights=fw_t)
        aux_loss = torch.tensor(0.0, device=dev)
        metric_loss = torch.tensor(0.0, device=dev)
        aux_target_batch = ab[:, :len(aux_idx)] if args.aux_regression else None
        metric_target_batch = ab[:, -len(metric_idx):] if args.comfort_metric_alignment else None
        if args.aux_regression:
            pred_aux = aux_head(z)
            aux_loss = aux_criterion(pred_aux, aux_target_batch)
        if args.comfort_metric_alignment:
            metric_loss = comfort_metric_alignment_loss(
                z=z,
                comfort_targets=metric_target_batch,
                metric_loss_type=args.metric_loss_type,
                metric_distance=args.metric_distance,
                metric_pair_sample=args.metric_pair_sample,
                metric_use_z_normalized=args.metric_use_z_normalized,
                metric_detach_target=args.metric_detach_target,
            )
        aux_weight = args.aux_loss_weight if ep >= args.aux_warmup_epochs else 0.0
        loss_total = loss_soft + aux_weight * aux_loss + args.metric_loss_weight * metric_loss
        finite = bool(torch.isfinite(loss_total).item()) and bool(torch.isfinite(logits).all().item()) and bool(torch.isfinite(tlogits).all().item())
        if (len(debug) < args.debug_n_batches) and train_mode:
            debug.append({
                "batch": int(batch_i), "traj_min": float(tb.min().item()), "traj_max": float(tb.max().item()),
                "traj_has_nan": bool(torch.isnan(tb).any().item()), "traj_has_inf": bool(torch.isinf(tb).any().item()),
                "feat_min": float(fb.min().item()), "feat_max": float(fb.max().item()),
                "feat_has_nan": bool(torch.isnan(fb).any().item()), "feat_has_inf": bool(torch.isinf(fb).any().item()),
                "embedding_min": float(z.min().item()), "embedding_max": float(z.max().item()),
                "logits_min": float(logits.min().item()), "logits_max": float(logits.max().item()),
                "target_logits_min": float(tlogits.min().item()), "target_logits_max": float(tlogits.max().item()),
                "loss_finite": finite,
            })
        if not finite:
            msg = f"non-finite batch={batch_i}, logits[{logits.min().item():.4f},{logits.max().item():.4f}], tlogits[{tlogits.min().item():.4f},{tlogits.max().item():.4f}]"
            if args.skip_bad_batches:
                warnings.append(msg)
                print("[warn]", msg)
                return None
            raise RuntimeError(msg)
        return loss_soft, aux_loss, metric_loss, loss_total

    epoch_bar = tqdm(range(args.epochs), desc="Training epochs")
    for ep in epoch_bar:
        model.train(); tls, tla, tlm, tlt = [], [], [], []
        if aux_head is not None:
            aux_head.train()
        for bi, (tb, fb, ab) in enumerate(tqdm(tr, desc=f"Epoch {ep + 1}/{args.epochs} - Train", leave=False)):
            losses = run_batch(tb, fb, ab, bi, train_mode=True, ep=ep)
            if losses is None:
                continue
            ls, la, lm, lt = losses
            opt.zero_grad(); lt.backward(); opt.step()
            tls.append(float(ls.item())); tla.append(float(la.item())); tlm.append(float(lm.item())); tlt.append(float(lt.item()))
        model.eval(); vls, vla, vlm, vlt = [], [], [], []
        if aux_head is not None:
            aux_head.eval()
        with torch.no_grad():
            for bi, (tb, fb, ab) in enumerate(tqdm(va, desc=f"Epoch {ep + 1}/{args.epochs} - Val", leave=False)):
                losses = run_batch(tb, fb, ab, bi, train_mode=False, ep=ep)
                if losses is not None:
                    ls, la, lm, lt = losses
                    vls.append(float(ls.item())); vla.append(float(la.item())); vlm.append(float(lm.item())); vlt.append(float(lt.item()))
        t_soft, v_soft = (float(np.mean(tls)) if tls else float('nan')), (float(np.mean(vls)) if vls else float('nan'))
        t_aux, v_aux = (float(np.mean(tla)) if tla else 0.0), (float(np.mean(vla)) if vla else 0.0)
        t_metric, v_metric = (float(np.mean(tlm)) if tlm else 0.0), (float(np.mean(vlm)) if vlm else 0.0)
        t_total, v_total = (float(np.mean(tlt)) if tlt else float('nan')), (float(np.mean(vlt)) if vlt else float('nan'))
        logs.append({'epoch': ep + 1, 'train_soft_loss': t_soft, 'train_aux_loss': t_aux, 'train_metric_loss': t_metric, 'train_total_loss': t_total, 'val_soft_loss': v_soft, 'val_aux_loss': v_aux, 'val_metric_loss': v_metric, 'val_total_loss': v_total})
        metric_value = v_total if args.early_stop_metric == "total" else (v_soft if args.early_stop_metric == "soft" else v_aux)
        epoch_bar.set_postfix(train_total_loss=f"{t_total:.4f}", val_total_loss=f"{v_total:.4f}", best_val=f"{best if np.isfinite(best) else float('nan'):.4f}")
        if np.isfinite(metric_value) and metric_value < best:
            best = metric_value; bad = 0
            torch.save({'model': model.state_dict(), 'aux_head': None if aux_head is None else aux_head.state_dict(), 'embedding_dim': args.embedding_dim, 'aux_regression': args.aux_regression, 'aux_targets': aux_targets, 'comfort_metric_alignment': args.comfort_metric_alignment, 'metric_targets': metric_targets, 'feature_weight_mode': args.feature_weight_mode, 'model_architecture': {'embedding_dim': args.embedding_dim, 'aux_hidden_dim': args.aux_hidden_dim}, 'train_summary': {'best_val_total_loss': best, 'early_stop_metric': args.early_stop_metric}}, out / 'model.pt')
        else:
            bad += 1
        if bad >= args.patience:
            break

    pd.DataFrame(logs).to_csv(out / 'train_log.csv', index=False)
    (out / 'train_debug.json').write_text(json.dumps(debug, indent=2))
    summary = {
        'data_dir': args.data_dir,
        'n_total': int(len(traj_raw)), 'n_retained': int(len(retained)), 'n_dropped': int(len(dropped)),
        'n_train': int((split == 'train').sum()), 'n_val': int((split == 'val').sum()),
        'traj_shape': list(traj_clean.shape), 'feat_shape': list(feat_norm.shape),
        'traj_nan_count_raw': traj_nan_raw, 'traj_inf_count_raw': traj_inf_raw,
        'traj_nan_count_after_sanitize': int(np.isnan(traj_clean).sum()), 'traj_inf_count_after_sanitize': int(np.isinf(traj_clean).sum()),
        'feat_nan_count_raw': feat_nan_raw, 'feat_inf_count_raw': feat_inf_raw,
        'feat_norm_min_before_clip': min_before, 'feat_norm_max_before_clip': max_before,
        'feat_norm_min_after_clip': min_after, 'feat_norm_max_after_clip': max_after,
        'feature_clip': args.feature_clip, 'temperature': args.temperature, 'feature_temperature': args.feature_temperature,
        'batch_size': args.batch_size, 'epochs': args.epochs, 'best_val_loss': best,
        'final_train_loss': logs[-1]['train_total_loss'] if logs else None, 'final_val_loss': logs[-1]['val_total_loss'] if logs else None,
        'stopped_early': bad >= args.patience,
        'warnings': warnings,
        'dropped_due_to_nonfinite_traj': int(sdiag['dropped_count']),
        'dropped_due_to_nonfinite_feat': int(len(dropped_feat_idx)),
        'traj_repaired_count': int(sdiag['repaired_count']),
        'feature_clipped_values': clipped_count,
        'feature_weight_mode': args.feature_weight_mode,
        'feature_weights_json': args.feature_weights_json,
        'feature_weights': weights,
        'aux_regression': args.aux_regression,
        'aux_targets': aux_targets,
        'aux_target_indices': aux_idx,
        'aux_loss_weight': args.aux_loss_weight,
        'aux_loss_type': args.aux_loss_type,
        'aux_hidden_dim': args.aux_hidden_dim,
        'aux_target_clip': args.aux_target_clip,
        'comfort_metric_alignment': args.comfort_metric_alignment,
        'metric_targets': metric_targets,
        'metric_target_indices': metric_idx,
        'metric_loss_weight': args.metric_loss_weight,
        'metric_loss_type': args.metric_loss_type,
        'metric_distance': args.metric_distance,
        'metric_pair_sample': args.metric_pair_sample,
        'metric_target_clip': args.metric_target_clip,
        'early_stop_metric': args.early_stop_metric,
        'best_val_total_loss': best,
        'final_train_soft_loss': logs[-1]['train_soft_loss'] if logs else None,
        'final_train_aux_loss': logs[-1]['train_aux_loss'] if logs else None,
        'final_train_total_loss': logs[-1]['train_total_loss'] if logs else None,
        'final_val_soft_loss': logs[-1]['val_soft_loss'] if logs else None,
        'final_val_aux_loss': logs[-1]['val_aux_loss'] if logs else None,
        'final_train_metric_loss': logs[-1]['train_metric_loss'] if logs else None,
        'final_val_metric_loss': logs[-1]['val_metric_loss'] if logs else None,
        'final_val_total_loss': logs[-1]['val_total_loss'] if logs else None,
    }
    (out / 'train_summary.json').write_text(json.dumps(summary, indent=2))
    (out / 'val_metrics.json').write_text(json.dumps({'best_val_loss': best}, indent=2))
    plt.figure(); plt.plot([x['epoch'] for x in logs], [x['train_total_loss'] for x in logs], label='train_total'); plt.plot([x['epoch'] for x in logs], [x['val_total_loss'] for x in logs], label='val_total'); plt.legend(); plt.tight_layout(); plt.savefig(out / 'training_curve.png'); plt.close()
    final_train_loss = logs[-1]['train_total_loss'] if logs else float('nan')
    final_val_loss = logs[-1]['val_total_loss'] if logs else float('nan')
    print('=' * 50)
    print('训练完成!')
    print(f'最终 Train Loss: {final_train_loss:.6f}')
    print(f'最终 Val Loss: {final_val_loss:.6f}')
    print(f'最佳 Val Loss: {best:.6f}')
    print('=' * 50)
    print('train_done')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='outputs/waymo_human_v1_full51')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--embedding_dim', type=int, default=64)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--feature_temperature', type=float, default=1.0)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--smoke_test', action='store_true')
    p.add_argument('--traj_nan_mode', choices=['interpolate', 'zero', 'drop'], default='interpolate')
    p.add_argument('--max_traj_nan_ratio', type=float, default=0.2)
    p.add_argument('--fail_on_nonfinite', action='store_true')
    p.add_argument('--feature_clip', type=float, default=10.0)
    p.add_argument('--feature_std_eps', type=float, default=1e-6)
    p.add_argument('--skip_bad_batches', action='store_true')
    p.add_argument('--debug_n_batches', type=int, default=1)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--feature_weight_mode', choices=['uniform','jerk_comfort','lateral','custom'], default='uniform')
    p.add_argument('--feature_weights_json', default=None)
    p.add_argument('--aux_regression', action='store_true')
    p.add_argument('--aux_targets', default='rms_accel,rms_jerk,max_abs_accel,max_abs_jerk,mean_thw,min_thw')
    p.add_argument('--aux_loss_weight', type=float, default=0.2)
    p.add_argument('--aux_loss_type', choices=['huber', 'mse', 'smooth_l1'], default='huber')
    p.add_argument('--aux_hidden_dim', type=int, default=128)
    p.add_argument('--aux_target_clip', type=float, default=10.0)
    p.add_argument('--aux_target_normalize', action='store_true', default=True)
    p.add_argument('--aux_warmup_epochs', type=int, default=0)
    p.add_argument('--save_aux_predictions', action='store_true')
    p.add_argument('--early_stop_metric', choices=['total', 'soft', 'aux'], default='total')
    p.add_argument('--comfort_metric_alignment', action='store_true')
    p.add_argument('--metric_targets', default='rms_accel,rms_jerk,max_abs_accel,max_abs_jerk,mean_thw,min_thw')
    p.add_argument('--metric_loss_weight', type=float, default=0.1)
    p.add_argument('--metric_loss_type', choices=['mse', 'huber', 'rank'], default='mse')
    p.add_argument('--metric_distance', choices=['euclidean', 'cosine'], default='euclidean')
    p.add_argument('--metric_target_normalize', action='store_true', default=True)
    p.add_argument('--metric_target_clip', type=float, default=10.0)
    p.add_argument('--metric_pair_sample', type=int, default=0)
    p.add_argument('--metric_detach_target', action='store_true', default=True)
    p.add_argument('--metric_use_z_normalized', action='store_true', default=True)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    run(a)
    print('[warning] 训练不使用 pseudo labels；feature weighting 仅用于弱监督特征对齐，不是 pseudo-label 监督训练。')
